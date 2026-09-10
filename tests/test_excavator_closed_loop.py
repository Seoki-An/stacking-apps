"""Commands, measured world motion, capture gates, and desktop identification.

The numerical fixture is the successful stone-18 direct path from test6.stack,
with its frozen profiles and asset hashes. No user .stack file is needed.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from stacking_common.artifacts import StackRunStore, decode_placement_plan, decode_run_context
from stacking_common.artifacts.mcts_debug import decode_mcts_step
from stacking_common.visualization.mcts import MCTSDebugArtifact
from stacking_excavator.acquisition.capture import GRIPPER_FRAME, digest
from stacking_excavator.acquisition.storage import load_product
from stacking_excavator.execution.cli import SimulationClock
from stacking_excavator.execution.emulation import EmulationRuntime

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/planning/excavator_closed_loop.json'


@pytest.fixture
def saved_run(tmp_path):
    values = json.loads(FIXTURE.read_text())
    path = tmp_path / 'placement.stack'
    with StackRunStore.create_mcts(path, MCTSDebugArtifact('world', ()),
                                   metadata={'platform': 'excavator'}) as writer:
        writer.append_run_context(decode_run_context(values['context']))
        writer.append_placement_plan(decode_placement_plan(values['plan']))
        writer.append_mcts_step(decode_mcts_step(values['step'], frame_id='world'))
    return path


@pytest.fixture
def owner(tmp_path, saved_run):
    clock = SimulationClock()
    runtime = EmulationRuntime(ROOT, tmp_path / 'session', clock=clock)
    command = runtime.load_placement(saved_run, step=1)
    yield runtime, clock, command
    runtime.close()


def advance(runtime, clock, count=1):
    for _ in range(count):
        clock.advance()
        runtime.heartbeat()
        runtime.tick()


def reach_gate(runtime, clock):
    for _ in range(20000):
        advance(runtime, clock)
        if runtime.gate or not runtime._active():
            break
    assert runtime.state['status'] == 'held', runtime.snapshot()
    assert runtime.gate is not None
    assert runtime.device.stopped


def stage_observation(runtime, *, offset=0., valid=True):
    gate = runtime.gate
    gate['manifest_sha256'] = 'a' * 64
    objects = []
    for object_id, prior in gate['priors'].items():
        pose = np.array(prior)
        pose[0, 3] += offset
        objects.append({'object_id': object_id, 'pose': pose.tolist(),
                        'valid': valid, 'reasons': [] if valid else ['missing_stone']})
    proposal = {'schema_version': 1, 'scene_revision': gate['revision'],
                'kind': gate['kind'], 'frame': GRIPPER_FRAME if gate['kind'] == 'inhand' else 'world',
                'capture_id': gate['job_id'], 'product_id': gate['job_id'] + '-crop',
                'source_manifest_sha256': gate['manifest_sha256'], 'objects': objects,
                'valid': valid, 'opening': [{'valid': True}]}
    proposal['sha256'] = digest(proposal)
    token = runtime.review.stage(proposal)
    gate.update(stage='review', proposal=proposal, token=token)
    return token


def test_measured_world_and_both_review_gates(owner, monkeypatch):
    runtime, clock, command = owner
    monkeypatch.setattr(runtime, '_launch', lambda args: None)
    runtime.start(command, expected_revision=0)
    reach_gate(runtime, clock)
    assert runtime.gate['kind'] == 'inhand'
    world = runtime.device.world
    assert world['attached_id'] == 18
    np.testing.assert_allclose(world['poses']['18'],
        runtime.device.gripper_pose() @ world['gripper_T_object'], atol=1e-10)
    q = runtime.device.positions.copy()
    count = len(runtime.device.acknowledgements)
    advance(runtime, clock, 20)
    np.testing.assert_array_equal(q, runtime.device.positions)
    assert len(runtime.device.acknowledgements) == count
    with pytest.raises(ValueError, match='review'):
        runtime.resume()
    token = stage_observation(runtime, offset=.005)
    runtime.decide(token, accept=True, expected_revision=0)
    assert runtime.review.snapshot()['revision'] == 1
    assert runtime.state['status'] == 'running'
    reach_gate(runtime, clock)
    assert runtime.gate['kind'] == 'scene'
    assert world['attached_id'] is None
    # Release preserves the actual carried pose; it does not teleport to the goal.
    from stacking_excavator.planning.trajectory import pose_matrix
    assert not np.array_equal(world['poses']['18'], pose_matrix(runtime.record.problem.place_pose))
    token = stage_observation(runtime, offset=.025)
    with pytest.raises(ValueError, match='release verification'):
        runtime.review.decide(token, accept=True, expected_revision=1, released_object_id=999)
    assert runtime.review.snapshot()['revision'] == 1
    assert runtime.review.snapshot()['attachment'] is not None
    runtime.decide(token, accept=True, expected_revision=1)
    assert runtime.state['status'] == 'succeeded'
    assert runtime.review.snapshot()['revision'] == 2
    assert runtime.review.snapshot()['attachment'] is None
    assert runtime.device.stopped
    with pytest.raises(ValueError, match='consumed'):
        runtime.decide(token, accept=True, expected_revision=2)
    # Scene corrections invalidate saved motions even when the caller supplies
    # the latest revision and loads a fresh command ID.
    replacement = runtime.load_placement(runtime.state['emulation_source']['path'], step=1)
    with pytest.raises(ValueError, match='accepted scene differs'):
        runtime.start(replacement, expected_revision=2)


@pytest.mark.parametrize('offset,accept,valid', [(.1, True, True), (0., False, True), (0., False, False)])
def test_changed_rejected_or_missing_stone_prevents_transfer(owner, monkeypatch, offset, accept, valid):
    runtime, clock, command = owner
    monkeypatch.setattr(runtime, '_launch', lambda args: None)
    runtime.start(command, expected_revision=0)
    reach_gate(runtime, clock)
    count = len(runtime.device.acknowledgements)
    token = stage_observation(runtime, offset=offset, valid=valid)
    runtime.decide(token, accept=accept, expected_revision=0)
    advance(runtime, clock, 10)
    assert runtime.state['needs_replan'] and runtime.state['status'] == 'failed'
    assert runtime.device.stopped
    assert len(runtime.device.acknowledgements) == count
    assert runtime.review.snapshot()['revision'] == int(accept)


@pytest.mark.parametrize('field,value', [('capture_id', 'old-job'), ('product_id', 'wrong-product'),
    ('source_manifest_sha256', 'b'*64), ('scene_revision', 99), ('kind', 'scene'), ('objects', [])])
def test_wrong_capture_cannot_authorize_motion(owner, monkeypatch, field, value):
    runtime, clock, command = owner
    monkeypatch.setattr(runtime, '_launch', lambda args: None)
    runtime.start(command, expected_revision=0)
    reach_gate(runtime, clock)
    token = stage_observation(runtime)
    runtime.gate['proposal'][field] = value
    with pytest.raises(ValueError, match='active capture'):
        runtime.decide(token, accept=True, expected_revision=0)
    assert runtime.state['status'] == 'held' and runtime.device.stopped
    assert runtime.review.snapshot()['revision'] == 0


@pytest.mark.parametrize('fault', ['lease', 'feedback'])
def test_late_final_scene_acceptance_cannot_bypass_device_health(owner, monkeypatch, fault):
    runtime, clock, command = owner
    monkeypatch.setattr(runtime, '_launch', lambda args: None)
    runtime.start(command, expected_revision=0)
    reach_gate(runtime, clock)
    runtime.decide(stage_observation(runtime), accept=True, expected_revision=0)
    reach_gate(runtime, clock)
    token = stage_observation(runtime)
    if fault == 'feedback':
        runtime.device.feedback_enabled = False
        clock.advance(.6)
    else:
        clock.advance(2.)
    with pytest.raises(ValueError, match='active verification gate'):
        runtime.decide(token, accept=True, expected_revision=1)
    assert runtime.state['status'] == 'failed' and runtime.state['needs_replan']
    assert runtime.review.snapshot()['revision'] == 1
    assert runtime.device.stopped


def test_cancel_kills_owned_capture_and_restart_preserves_world(owner, monkeypatch):
    runtime, clock, command = owner
    monkeypatch.setattr(runtime, '_launch', lambda args: None)
    runtime.start(command, expected_revision=0)
    reach_gate(runtime, clock)
    runtime.worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    worker = runtime.worker
    runtime.cancel()
    assert worker.poll() is not None
    saved = deepcopy(runtime.device.world)
    directory = runtime.directory
    runtime.close()
    with pytest.raises(ValueError, match='closed'):
        runtime.tick()
    reopened = EmulationRuntime(ROOT, directory, clock=clock)
    try:
        assert reopened.state['needs_replan']
        assert reopened.gate is None and reopened.worker is None
        assert reopened.device.world == saved
        assert reopened.device.stopped
        assert reopened.device.acknowledgements == []
        with pytest.raises(ValueError, match='healthy held'):
            reopened.resume()
        detached = deepcopy(saved['poses']['18'])
        reopened.reconcile(empty_gripper=True)
        assert reopened.device.world['attached_id'] is None
        assert reopened.device.world['poses']['18'] == detached
    finally:
        reopened.close()


def test_world_snapshot_is_not_mutable_and_missed_pickup_stays_in_field(owner):
    runtime, clock, command = owner
    snapshot = runtime.snapshot()
    snapshot['virtual_world']['poses'].clear()
    assert runtime.device.world['poses']
    original = deepcopy(runtime.device.world['poses']['18'])
    runtime.device.missing_stone = True
    runtime.device.grasp('acquire', 18)
    advance(runtime, clock, 5)
    assert runtime.device.world['attached_id'] is None
    assert runtime.device.world['poses']['18'] == original


def test_owner_crash_at_review_never_consumes_the_pending_proposal(owner, monkeypatch):
    runtime, clock, command = owner
    monkeypatch.setattr(runtime, '_launch', lambda args: None)
    runtime.start(command, expected_revision=0)
    reach_gate(runtime, clock)
    token = stage_observation(runtime)
    saved = deepcopy(runtime.device.world)
    # Model process death after the durable held event, without a shutdown event.
    runtime._lock.close()
    runtime.closed = True
    reopened = EmulationRuntime(ROOT, runtime.directory, clock=clock)
    try:
        assert reopened.state['status'] == 'failed'
        assert reopened.state['needs_replan']
        assert reopened.history()[-1]['failure'] == 'interrupted'
        assert reopened.gate is None and reopened.worker is None
        assert reopened.review.snapshot()['revision'] == 0
        assert reopened.review.proposals()[0]['decision'] is None
        assert reopened.review.proposals()[0]['proposal']['sha256'] == token
        assert reopened.device.world == saved and reopened.device.stopped
    finally:
        reopened.close()


def test_worker_start_failure_stops_motion(owner, monkeypatch):
    runtime, clock, command = owner
    def fail(args):
        raise OSError('capture process unavailable')
    monkeypatch.setattr(runtime, '_launch', fail)
    runtime.start(command, expected_revision=0)
    for _ in range(20000):
        advance(runtime, clock)
        if not runtime._active():
            break
    assert runtime.state['status'] == 'failed'
    assert runtime.state['needs_replan'] and runtime.device.stopped
    assert 'capture process unavailable' in runtime.state['emulation_error']


def test_live_execution_console_gate_and_world_layers(saved_run, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from stacking_excavator.gui.operations.session import SessionConsole
    app = QApplication.instance() or QApplication([])
    console = SessionConsole(ROOT)
    try:
        console.open_session(tmp_path / 'ui-session')
        console.load_emulation(saved_run, step=1)
        console.timer.stop()
        runtime = console.runtime
        clock = SimulationClock()
        clock.seconds = runtime.clock()
        runtime.clock = runtime.device.clock = clock
        monkeypatch.setattr(runtime, '_launch', lambda args: None)
        console.start()
        reach_gate(runtime, clock)
        stage_observation(runtime)
        console._buttons()
        assert console.accept_capture_button.isEnabled()
        assert console.reject_capture_button.isEnabled()
        console.decide_capture(True)
        frame = console.execution_frame()
        assert frame.robot_parts
        assert any(o.instance_id == 'world/18' for o in frame.objects)
        assert any(o.instance_id == 'accepted/attachment' for o in frame.overlays)
        assert runtime.state['status'] == 'running'
        console.control('hold')
        assert runtime.state['status'] == 'held'
        console.control('cancel')
        console.open_session(tmp_path / 'ui-session')
        assert console.projector is console.runtime.device.projector
        assert console.runtime.state['needs_replan']
    finally:
        console.close()
        console.deleteLater()
        app.processEvents()


def test_real_commands_raycast_transfer_icp_and_review(owner):
    """Actual subprocesses and Open3D, with a deliberately interrupted crop copy."""
    runtime, clock, command = owner
    runtime.delivery_paused = True
    runtime.start(command, expected_revision=0)
    paused = False
    reviewed = []
    deadline = time.monotonic() + 240
    while runtime._active() and time.monotonic() < deadline:
        advance(runtime, clock)
        gate = runtime.gate
        if gate and gate['stage'] == 'delivery_pending' and gate.get('partial_attempted'):
            received = Path(gate['directory']) / 'received'
            assert not (received / 'receipt.json').exists()
            assert list(received.glob('*.partial'))
            with pytest.raises((ValueError, FileNotFoundError)):
                load_product(received)
            paused = True
            runtime.delivery_paused = False
        if gate and gate['stage'] == 'review':
            proposal = gate['proposal']
            assert proposal['valid'], proposal
            reviewed.append(gate['kind'])
            request = json.loads((Path(gate['directory']) / 'priors.json').read_text())
            assert 'snapshot' not in request and 'world' not in request
            product, frames = load_product(Path(gate['directory']) / 'received', expected_revision=gate['revision'])
            assert len(frames) == 3 and product['capture_id'] == gate['job_id']
            assert all(len(frame.joint_positions) == 6 for frame in frames)
            # Result is computed by ICP, not copied from a prior/world matrix.
            assert any(not np.array_equal(o['pose'], gate['priors'][o['object_id']]) for o in proposal['objects'])
            runtime.decide(gate['token'], accept=True, expected_revision=gate['revision'])
        if runtime.worker:
            time.sleep(.02)
    assert paused
    assert reviewed == ['inhand', 'scene'], runtime.snapshot()
    assert runtime.state['status'] == 'succeeded', runtime.snapshot()
    assert runtime.review.snapshot()['attachment'] is None
    assert runtime.review.snapshot()['revision'] == 2
    assert runtime.device.world['attached_id'] is None


@pytest.mark.parametrize('fault', ['missed_pickup', 'grasp_offset'])
def test_real_fault_scan_blocks_transfer(owner, fault):
    runtime, clock, command = owner
    runtime.device.missing_stone = fault == 'missed_pickup'
    runtime.device.slip = np.array([.1, 0., 0.]) if fault == 'grasp_offset' else np.zeros(3)
    runtime.start(command, expected_revision=0)
    deadline = time.monotonic() + 180
    reviewed = False
    while runtime._active() and time.monotonic() < deadline:
        advance(runtime, clock)
        gate = runtime.gate
        if gate and gate['stage'] == 'review':
            assert gate['kind'] == 'inhand'
            reviewed = True
            runtime.decide(gate['token'], accept=bool(gate['proposal']['valid']),
                           expected_revision=gate['revision'])
        if runtime.worker:
            time.sleep(.02)
    assert reviewed, runtime.snapshot()
    assert runtime.state['status'] == 'failed' and runtime.state['needs_replan']
    assert runtime.device.stopped
    assert not any(e['segment_index'] >= 2 and e['diagnostics'].get('transition') == 'sample_completed'
                   for e in runtime.history())
