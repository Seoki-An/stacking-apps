"""Closed-loop desktop execution, virtual NUC capture, and desktop perception."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation

from stacking_common.artifacts.execution import build_execution_command
from stacking_common.execution import ExecutionFailure
from stacking_excavator.acquisition.storage import atomic_json, load_product
from stacking_excavator.planning.replay import recorded_planning_profile
from stacking_excavator.planning.trajectory import pose_matrix
from .runtime import SimulationRuntime
from .saved_plan import load_placement
from .virtual_nuc import VirtualNuc


class EmulationRuntime(SimulationRuntime):
    """All control mutations run on the owner thread; cloud work is isolated.

    In-hand checks may continue a held path within 2 cm / 10 degrees. Final
    scene verification allows 5 cm for registration against the measured stone
    catalog (which differs from the rendered PLY surface). These are emulation
    acceptance bounds, not changes to motion or commissioned perception limits.
    """
    def __init__(self, repository, directory, *, clock=time.monotonic):
        self.repository = Path(repository).resolve()
        self.worker = self.worker_log = None
        self.gate = None
        self.record = self.context = None
        self.expected_attachment = None
        self.delivery_paused = False
        device = VirtualNuc(repository, Path(directory) / 'virtual-nuc', clock=clock)
        super().__init__(repository, directory, device=device, clock=clock)
        # Old jobs are evidence only. A new owner never restarts a capture,
        # consumes a late proposal, or silently resumes an interrupted path.
        saved = self.state.get('emulation_source')
        try:
            if saved:
                self.load_placement(saved['path'], step=saved['step'])
        except BaseException:
            self.close()
            raise
        self.state['emulation_stage'] = 'needs_replan' if self.state['needs_replan'] else 'idle'
        self._persist()

    def load_placement(self, path, *, step=None):
        if self._active():
            raise ValueError('stop the emulation before loading another placement')
        record, context = load_placement(self.repository, path, step=step)
        profile = recorded_planning_profile(context)
        self.device.configure(record, profile)
        self.record, self.context = record, context
        self.prepared_command = build_execution_command(record, context)
        self.state['emulation_source'] = {'path': str(Path(path).resolve()), 'step': record.step}
        self._persist()
        return self.prepared_command

    def start(self, command, *, expected_revision):
        if self.record is None or command is not self.prepared_command:
            raise ValueError('load a committed placement for this virtual world first')
        scene = self.review.snapshot()
        priors = {str(o.object_id): pose_matrix(o.pose) for o in
                  (*self.record.problem.pick_scene_objects, *self.record.problem.place_scene_objects)}
        priors[str(self.record.problem.action.object_id)] = pose_matrix(self.record.problem.pick_pose)
        for object_id, observation in scene['poses'].items():
            if object_id not in priors or not self._compatible(priors[object_id], observation['pose'], .02):
                raise ValueError('accepted scene differs from the saved plan; replan using current observations')
        self.gate = None
        self.expected_attachment = None
        super().start(command, expected_revision=expected_revision)
        self.state['emulation_stage'] = 'executing'
        self.state['emulation_error'] = None
        self._persist()

    def _event(self, status, reason=None, failure=None, *, command=None, **diagnostics):
        if diagnostics.get('transition') == 'sample_completed':
            self._completed_sample = (self.state['segment_index'], self.state['sample_index'])
            grasp = diagnostics.get('grasp')
            if grasp:
                self.device.grasp(grasp, self.record.problem.action.object_id)
                if grasp == 'acquire':
                    si, sample = self._completed_sample
                    segment = self.record.result.segments[si]
                    target = (segment.target_poses[sample] if segment.target_poses is not None
                              else self.record.problem.pick_pose)
                    event = next(e for e in self.command.grasp_events
                                 if (e.segment_index, e.sample_index) == (si, sample))
                    planned_gripper = self.device.gripper_pose(segment.positions[sample], event.opening)
                    self.expected_attachment = np.linalg.inv(planned_gripper) @ pose_matrix(target)
            self.device.save_world()
        super()._event(status, reason, failure, command=command, **diagnostics)

    def _advance_path(self):
        self._completed_sample = None
        super()._advance_path()
        if self._completed_sample is not None and self.state['status'] == 'running':
            si, sample = self._completed_sample
            segment = self.command.segments[si]
            if sample == len(segment.positions)-1 and segment.name in ('pick_retreat', 'handoff_pick'):
                self._begin_gate('inhand')

    def _finish(self, status, failure=None, reason=None):
        if status == 'succeeded' and self.gate is None:
            self._begin_gate('scene')
            return
        self._stop_worker()
        self.state['emulation_stage'] = 'complete' if status == 'succeeded' else 'needs_replan'
        self.state['emulation_error'] = reason
        self.device.save_world()
        super()._finish(status, failure, reason)

    def _begin_gate(self, kind):
        self.device.stop()
        object_id = self.record.problem.action.object_id
        if kind == 'inhand':
            if self.expected_attachment is None:
                raise ValueError('missing predicted in-hand pose')
            priors = {object_id: self.expected_attachment.tolist()}
        else:
            priors = {o.object_id: pose_matrix(o.pose).tolist() for o in self.record.problem.place_scene_objects}
            priors[object_id] = pose_matrix(self.record.problem.place_pose).tolist()
        target = self.context.profiles['inventory_profile']['target']
        corners = np.array([[x, y, z] for x in (0, target['width'])
                            for y in (0, target['length']) for z in (0, target['height'])]) + target['origin']
        transform = pose_matrix(recorded_planning_profile(self.context).world_T_stack.parent_T_child)
        corners = corners @ transform[:3, :3].T + transform[:3, 3]
        job_id = uuid4().hex
        directory = self.directory / 'emulation' / job_id
        directory.mkdir(parents=True)
        self.gate = {'job_id': job_id, 'kind': kind, 'directory': str(directory),
                     'revision': self.state['scene_revision'], 'priors': priors,
                     'command_id': self.command.command_id, 'stage': 'capture'}
        request = {'repository': str(self.repository), 'job_id': job_id, 'kind': kind,
                   'revision': self.gate['revision'], 'priors': priors,
                   'target_corners': corners.tolist(), 'ground_height': 0.,
                   'spool': str(self.device.directory / 'captures'),
                   'snapshot': self.device.capture_snapshot()}
        atomic_json(directory / 'capture-request.json', request)
        atomic_json(directory / 'priors.json', {'scene_revision': self.gate['revision'],
                    'ground_height': 0., 'poses': priors})
        self.state['emulation_stage'] = f'{kind}: capture'
        self._event('held', transition='capture_requested', job_id=job_id, purpose=kind)
        self._launch(['-m', 'stacking_excavator.execution.emulation_worker', 'capture',
                      str(directory / 'capture-request.json'), str(directory / 'capture-result.json')])

    def _launch(self, arguments):
        directory = Path(self.gate['directory'])
        self.worker_log = (directory / 'worker.log').open('ab')
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = '1'
        env['PYTHONPATH'] = os.pathsep.join([str(self.repository / 'apps/excavator/src'),
            str(self.repository / 'packages/stacking-common/src'), env.get('PYTHONPATH', '')])
        try:
            self.worker = subprocess.Popen([sys.executable, *arguments], env=env,
                stdout=self.worker_log, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            self.worker_log.close()
            self.worker_log = None
            raise
        self.worker_deadline = time.monotonic() + 180.

    def _stop_worker(self):
        if self.worker is not None:
            if self.worker.poll() is None:
                self.worker.kill()
            self.worker.wait(timeout=5)
            self.worker = None
        if self.worker_log is not None:
            self.worker_log.close()
            self.worker_log = None

    def tick(self):
        try:
            super().tick()
            if self.gate and self.state['status'] == 'held':
                self._poll_gate()
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            if self._active():
                self._finish('failed', ExecutionFailure.PERCEPTION_FAILED, str(error))
            else:
                raise
        return self.snapshot()

    def _poll_gate(self):
        directory = Path(self.gate['directory'])
        if self.worker is not None:
            code = self.worker.poll()
            if code is None:
                if time.monotonic() > self.worker_deadline:
                    raise ValueError('capture/identification worker timed out')
                return
            self._stop_worker()
            if code:
                raise ValueError(f"{self.gate['stage']} failed; inspect {directory / 'worker.log'}")
            if self.gate['stage'] == 'capture':
                result = json.loads((directory / 'capture-result.json').read_text())
                self.gate.update(stage='delivery_pending', product=result['product'],
                                 manifest_sha256=result['manifest_sha256'])
                self.state['emulation_stage'] = f"{self.gate['kind']}: delivery pending"
                self._event('held', transition='capture_completed', job_id=self.gate['job_id'], **result)
            elif self.gate['stage'] == 'deliver':
                result = json.loads((directory / 'delivery-result.json').read_text())
                if result['delivered'] is not True:
                    self.gate['stage'] = 'delivery_pending'
                    self.state['emulation_stage'] = f"{self.gate['kind']}: delivery paused"
                    return
                manifest, _ = load_product(directory / 'received', expected_revision=self.gate['revision'])
                if (manifest['sha256'] != self.gate['manifest_sha256'] or
                        manifest['capture_id'] != self.gate['job_id'] or
                        manifest['product_id'] != self.gate['job_id'] + '-crop' or
                        manifest['kind'] != self.gate['kind']):
                    raise ValueError('delivered crop does not belong to the active capture')
                self.gate['stage'] = 'identify'
                self.state['emulation_stage'] = f"{self.gate['kind']}: identifying"
                self._launch(['-m', 'stacking_excavator.perception.cli', 'identify',
                    str(directory / 'received'), str(directory / 'priors.json'),
                    str(directory / 'identification'), '--repository', str(self.repository)])
            elif self.gate['stage'] == 'identify':
                proposal = json.loads((directory / 'identification/proposal.json').read_text())
                self._validate_proposal(proposal)
                token = self.review.stage(proposal)
                self.gate.update(stage='review', token=token, proposal=proposal)
                refs_path = self.directory / 'proposal_paths.json'
                refs = json.loads(refs_path.read_text()) if refs_path.exists() else {}
                refs[token] = str(directory / 'identification/proposal.json')
                atomic_json(refs_path, refs)
                self.state['emulation_stage'] = f"{self.gate['kind']}: awaiting review"
                self._event('held', transition='perception_ready', token=token, valid=proposal['valid'])
        if self.gate['stage'] == 'delivery_pending' and (
                not self.delivery_paused or not self.gate.get('partial_attempted')):
            request = {'product': self.gate['product'], 'destination': str(directory / 'received')}
            if self.delivery_paused:
                request['byte_budget'] = 1024
                self.gate['partial_attempted'] = True
            atomic_json(directory / 'delivery-request.json', request)
            self.gate['stage'] = 'deliver'
            self.state['emulation_stage'] = f"{self.gate['kind']}: delivering"
            self._launch(['-m', 'stacking_excavator.execution.emulation_worker', 'deliver',
                          str(directory / 'delivery-request.json'), str(directory / 'delivery-result.json')])

    def _validate_proposal(self, proposal):
        gate = self.gate
        expected = {'capture_id': gate['job_id'], 'product_id': gate['job_id'] + '-crop',
                    'source_manifest_sha256': gate['manifest_sha256'],
                    'kind': gate['kind'], 'scene_revision': gate['revision']}
        if (any(proposal.get(key) != value for key, value in expected.items()) or
                sorted(o['object_id'] for o in proposal['objects']) != sorted(gate['priors'])):
            raise ValueError('perception result does not belong to the active capture')

    @staticmethod
    def _compatible(prior, observed, position_limit):
        relative = np.linalg.inv(prior) @ np.asarray(observed)
        return (np.linalg.norm(relative[:3, 3]) <= position_limit and
                Rotation.from_matrix(relative[:3, :3]).magnitude() <= np.deg2rad(10.))

    def decide(self, token, *, accept, expected_revision):
        # A late review click cannot revive expired feedback or an operator lease.
        self.tick()
        gate = self.gate
        if not gate or gate['stage'] != 'review' or token != gate['token']:
            return super().decide(token, accept=accept, expected_revision=expected_revision)
        if (self.state['status'] != 'held' or self.state['needs_replan'] or
                gate['command_id'] != self.command.command_id or expected_revision != gate['revision']):
            raise ValueError('capture decision no longer belongs to an active verification gate')
        proposal = gate['proposal']
        self._validate_proposal(proposal)
        compatible = proposal['valid'] and accept
        checks = []
        for obj in proposal['objects']:
            prior = np.array(gate['priors'][obj['object_id']])
            if obj['pose'] is None:
                compatible = False
                continue
            observed = np.array(obj['pose'])
            relative = np.linalg.inv(prior) @ observed
            translation = float(np.linalg.norm(relative[:3, 3]))
            rotation = float(np.rad2deg(Rotation.from_matrix(relative[:3, :3]).magnitude()))
            position_limit = .02 if gate['kind'] == 'inhand' else .05
            checks.append({'object_id': obj['object_id'], 'translation_m': translation,
                           'rotation_degrees': rotation, 'position_limit_m': position_limit})
            compatible = compatible and translation <= position_limit and rotation <= 10.
        # Invalid evidence is never accepted. Valid but incompatible observations
        # may update the scene, but cannot authorize the remaining saved motion.
        result = self.review.decide(token, accept=accept, expected_revision=expected_revision,
            released_object_id=self.record.problem.action.object_id if gate['kind'] == 'scene' else None)
        self.state['scene_revision'] = result['revision']
        self._event('held', transition='perception_decided', token=token,
                    accepted=accept, compatible=bool(compatible), purpose=gate['kind'], verification=checks)
        if not compatible:
            self._finish('failed', ExecutionFailure.PERCEPTION_FAILED, 'observation rejected or changed geometry; needs_replan')
        elif gate['kind'] == 'scene':
            # Keep gate non-null while finishing to avoid requesting another scan.
            self._finish('succeeded')
            self.gate = None
        else:
            self.gate = None
            self.state['emulation_stage'] = 'executing'
            super().resume()
        return result

    def resume(self):
        if self.gate is not None:
            raise ValueError('review the current capture before continuing execution')
        super().resume()

    def reconcile(self, *, empty_gripper=False):
        if empty_gripper and self.device.world and self.device.world['attached_id'] is not None:
            if self._active():
                raise ValueError('stop emulation before empty-gripper reconciliation')
            self.device.grasp('release', self.device.world['attached_id'])
        super().reconcile(empty_gripper=empty_gripper)
        self.gate = None

    def snapshot(self):
        snapshot = super().snapshot()
        snapshot['virtual_world'] = deepcopy(self.device.world)
        snapshot['capture'] = None if self.gate is None else {k: self.gate[k]
            for k in ('job_id', 'kind', 'stage', 'revision')}
        return snapshot

    def close(self):
        self._stop_worker()
        self.device.save_world()
        super().close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path.cwd())
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--step', type=int, default=1)
    args = parser.parse_args(argv)
    from .cli import SimulationClock
    clock = SimulationClock()
    runtime = EmulationRuntime(args.repository, args.session, clock=clock)
    try:
        command = runtime.load_placement(args.run, step=args.step)
        runtime.start(command, expected_revision=runtime.review.snapshot()['revision'])
        last = None
        deadline = time.monotonic() + 600.
        while runtime.state['status'] in ('running', 'held') and time.monotonic() < deadline:
            clock.advance()
            runtime.heartbeat()
            runtime.tick()
            stage = runtime.state['emulation_stage']
            if stage != last:
                print(stage, flush=True)
                last = stage
            if runtime.gate and runtime.gate['stage'] == 'review' and runtime.state['status'] == 'held':
                proposal = runtime.gate['proposal']
                runtime.decide(proposal['sha256'], accept=bool(proposal['valid']),
                               expected_revision=proposal['scene_revision'])
            if runtime.worker:
                time.sleep(.02)
        if runtime._active():
            runtime.cancel()
        atomic_json(args.session / 'report.json', runtime.snapshot())
        print('result:', runtime.state['status'], flush=True)
        return 0 if runtime.state['status'] == 'succeeded' else 1
    finally:
        runtime.close()


if __name__ == '__main__':
    raise SystemExit(main())
