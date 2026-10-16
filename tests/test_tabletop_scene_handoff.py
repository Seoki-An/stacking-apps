"""Accepted SceneID never falls through into unrefined saved robot motions."""

from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('PySide6')
from PySide6.QtWidgets import QApplication

from stacking_common.artifacts import StackRunStore
from stacking_tabletop.acquisition.hand_eye_capture import atomic_json, digest
from stacking_tabletop.acquisition.stack_scene_capture import _finish_stack_scene
from stacking_tabletop.acquisition.storage import write_xyz_pcd
from stacking_tabletop.execution import scene_refinement
from stacking_tabletop.execution.sequence import OperatorGate, execute_saved_run
from stacking_tabletop.gui.operations.execution_view import ExecutionView
from stacking_tabletop.gui.operations.execution_worker import ExecutionWorker
from test_tabletop_saved_execution import ROOT, executor_for, saved


def report(directory, step=1):
    directory.mkdir()
    pcd = directory / 'scene.pcd'
    write_xyz_pcd(pcd, np.array([[0., 0., .1]]))
    sceneid = directory / 'sceneid_result.json'
    atomic_json(sceneid, {'status': 'identified', 'accepted_snapshot': True})
    status = dict(status='complete', step=step, pcd=str(pcd),
                  pcd_sha256=digest(pcd), viewer=str(directory / 'overview.html'),
                  scene_identification=dict(path=str(sceneid), sha256=digest(sceneid),
                                            status='identified', accepted_snapshot=True))
    atomic_json(directory / 'capture.json', status)
    return status


class Processing:
    def __init__(self, output):
        self.output = output

    def finish(self, executor):
        return _finish_stack_scene(executor, self.output)

    def abort(self):
        pass


def test_scene_acceptance_commits_release_but_never_queues_next_pickup(saved, tmp_path):
    output = tmp_path / 'capture'
    status = report(output)
    transport, executor = executor_for(saved)
    executor.stack_scene_handler = lambda _: Processing(output)
    results = execute_saved_run(saved, executor)
    assert [result.status.value for result in results] == ['succeeded']
    checkpoint = StackRunStore.load_execution_checkpoint(saved.path)
    assert checkpoint.completed_steps == (1,)
    assert checkpoint.pending_steps == (2,)
    assert not checkpoint.requires_reconciliation
    assert [command.step for command in StackRunStore.load_execution_commands(saved.path)] == [1]
    pending = scene_refinement.pending_scene_refinement(saved.path)
    assert pending['step'] == 1
    assert pending['capture_directory'] == str(output)
    events = StackRunStore.load_execution_events(saved.path)
    accepted = next(event for event in events if 'scene_refinement_required' in event.diagnostics)
    assert accepted.diagnostics['scene_refinement_required']['pcd_sha256'] == status['pcd_sha256']
    assert accepted.diagnostics['scene_refinement_required']['sceneid_sha256'] == status['scene_identification']['sha256']
    assert events[-1].status.value == 'succeeded'
    calls = len(transport.calls)
    with pytest.raises(ValueError, match='requires remaining-plan refinement'):
        execute_saved_run(saved, executor)
    assert len(transport.calls) == calls


def test_last_placement_only_reviews_scene_without_refinement(saved, tmp_path):
    output = tmp_path / 'last-capture'
    report(output, 2)
    gates, events = [], []
    executor = SimpleNamespace(
        saved=saved, _command=SimpleNamespace(step=2), pending_scene_refinement=None,
        _gate_wait=lambda name, message: gates.append((name, message)),
        _emit=lambda status, **values: events.append(values),
    )
    _finish_stack_scene(executor, output)
    assert gates[0][0] == 'stack_scene_result'
    assert 'refine' not in gates[0][1]
    assert executor.pending_scene_refinement is None
    assert not any('scene_refinement_required' in event for event in events)
    assert any('scene_poses_accepted' in event for event in events)


def test_interrupted_acceptance_can_retry_only_capture_without_pickup(saved, tmp_path, monkeypatch):
    from stacking_tabletop.acquisition import stack_scene_capture
    from stacking_tabletop.execution.recovery import retry_released_stack_capture

    first, second = tmp_path / 'first-capture', tmp_path / 'second-capture'
    report(first)
    report(second)
    _, executor = executor_for(saved)
    executor.stack_scene_handler = lambda _: Processing(first)
    telemetry = executor._send_telemetry

    def interrupt_after_acceptance(force=False):
        if executor.pending_scene_refinement is not None and force:
            raise RuntimeError('interrupted after acceptance before terminal success')
        return telemetry(force=force)

    executor._send_telemetry = interrupt_after_acceptance
    results = execute_saved_run(saved, executor)
    assert results[0].status.value == 'failed'
    assert scene_refinement.pending_scene_refinement(saved.path)['capture_directory'] == str(first)
    retry_released_stack_capture(saved.path, 'operator verified empty gripper and placed stone',
                                 robot_stopped=True, placed_scene_confirmed=True, gripper_empty=True)
    transport, retry = executor_for(saved)
    transport.close = lambda: None
    monkeypatch.setattr(stack_scene_capture, 'check_stack_capture_runtime', lambda _: None)
    worker = ExecutionWorker(saved, repository=ROOT,
                             device_factory=lambda *_: (transport, retry.arm, retry.gripper))
    worker.runtime = transport.runtime
    worker._stack_scene = lambda _: Processing(second)
    refined, completed, failures = [], [], []
    worker._refine_remaining = lambda pending, executor=None: refined.append(pending)
    worker.gate_changed.connect(lambda gate: worker.executor.approve(gate.token) if gate else None)
    worker.completed.connect(completed.append)
    worker.failed.connect(failures.append)
    worker.run()
    assert failures == []
    results = completed[0]
    assert [result.status.value for result in results] == ['succeeded']
    assert results[0].diagnostics['stack_scene_recovery'] is True
    assert StackRunStore.load_execution_checkpoint(saved.path).completed_steps == (1,)
    assert [command.step for command in StackRunStore.load_execution_commands(saved.path)] == [1, 1]
    commands = StackRunStore.load_execution_commands(saved.path)
    retry_events = [event for event in StackRunStore.load_execution_events(saved.path)
                    if event.command_id == commands[-1].command_id]
    assert not any(event.segment_index == 0 for event in retry_events)
    assert not any(call[0] == 'flag' for call in transport.calls)
    assert scene_refinement.pending_scene_refinement(saved.path)['capture_directory'] == str(second)
    assert refined[0]['capture_directory'] == str(second)


def test_worker_refinement_failure_blocks_old_suffix_and_retry_has_no_devices(saved, tmp_path, monkeypatch):
    from stacking_tabletop.acquisition import stack_scene_capture
    from stacking_tabletop.perception import service_client

    output = tmp_path / 'capture'
    report(output)
    monkeypatch.setattr(stack_scene_capture, 'check_stack_capture_runtime', lambda _: None)
    monkeypatch.setattr(ExecutionWorker, '_stack_scene', lambda self, _: Processing(output))
    transport, initial_executor = executor_for(saved)
    closed = []
    transport.close = lambda: closed.append(True)
    worker = ExecutionWorker(saved, repository=ROOT, device_factory=lambda *_: (
        transport, initial_executor.arm, initial_executor.gripper))
    worker.runtime = transport.runtime
    worker.capture_stack_scene = True
    worker.gate_changed.connect(lambda gate: worker.executor.approve(gate.token) if gate else None)
    requests, failures = [], []

    def unavailable(repository, operation, **arguments):
        arguments['check']()
        assert operation == 'refine_remaining'
        assert StackRunStore.load_execution_checkpoint(saved.path).completed_steps == (1,)
        requests.append(arguments)
        raise RuntimeError('refinement service unavailable')

    monkeypatch.setattr(service_client, 'request', unavailable)
    worker.failed.connect(failures.append)
    worker.run()
    assert failures == ['refinement service unavailable']
    assert closed == [True]
    assert len(requests) == 1
    assert [command.step for command in StackRunStore.load_execution_commands(saved.path)] == [1]
    assert scene_refinement.pending_scene_refinement(saved.path) is not None

    # A subsequent Start retries perception only. This fake service branch
    # isolates the handoff protocol from the separately tested branch writer.
    branch = tmp_path / 'refined.stack'
    shutil.copyfile(saved.path, branch)
    original_pending = scene_refinement.pending_scene_refinement
    monkeypatch.setattr(scene_refinement, 'pending_scene_refinement',
                        lambda path: None if Path(path) == branch else original_pending(path))

    def success(repository, operation, **arguments):
        arguments['check']()
        assert operation == 'refine_remaining'
        assert arguments['run'] == str(saved.path)
        assert arguments['capture'] == str(output)
        assert arguments['step'] == 1
        return {'run': str(branch)}

    monkeypatch.setattr(service_client, 'request', success)
    retry = ExecutionWorker(saved, repository=ROOT,
                            device_factory=lambda *_: pytest.fail('retry must not connect hardware'))
    ready, completed, retry_failures = [], [], []
    retry.refined_run_ready.connect(ready.append)
    retry.completed.connect(completed.append)
    retry.failed.connect(retry_failures.append)
    retry.run()
    assert retry_failures == []
    assert ready == [str(branch)]
    assert completed == [()]
    assert [command.step for command in StackRunStore.load_execution_commands(saved.path)] == [1]


def test_refined_branch_is_loaded_only_after_worker_finished(saved, tmp_path):
    app = QApplication.instance() or QApplication([])
    branch = tmp_path / 'refined.stack'
    shutil.copyfile(saved.path, branch)
    view = ExecutionView(ROOT, worker_factory=lambda *_: pytest.fail('must not auto-start'))
    view.load(saved.path)
    running = True
    view.worker = SimpleNamespace(isRunning=lambda: running)
    loaded = []
    view.refined_run_loaded.connect(loaded.append)
    try:
        view._refined_run_ready(str(branch))
        assert view.path == saved.path
        assert loaded == []
        running = False
        view._finished()
        assert view.path == branch
        assert loaded == [str(branch)]
        assert not view.running
        assert 'nothing has started automatically' in view.status.text()
    finally:
        view.worker = None
        view.close()
        app.processEvents()


def test_refined_scene_keeps_observation_over_historical_release(saved, monkeypatch):
    from stacking_tabletop.gui.operations.execution_scene import ExecutionScene

    corrected = np.array([.12, -.03, .18, 0., 0., 0., 1.])
    historical = corrected.copy()
    historical[0] -= .004
    newer = corrected.copy()
    newer[0] += .002
    info = StackRunStore.info(saved.path)
    monkeypatch.setattr(StackRunStore, 'info', lambda path: SimpleNamespace(
        metadata={**dict(info.metadata), 'scene_refinement': {'completed_step': 1}}))
    monkeypatch.setattr(StackRunStore, 'load_planning_checkpoint', lambda path, step: SimpleNamespace(
        context_sha256=saved.context.sha256,
        state=SimpleNamespace(placed_object_indices=[0], object_ids=[1], object_poses={1: corrected}),
        world_T_stack=SimpleNamespace(child_pose_to_parent=lambda pose: pose.copy()),
    ))
    scene = ExecutionScene(saved, ROOT)
    scene.event(SimpleNamespace(status=SimpleNamespace(value='succeeded'), diagnostics={
        'step': 1, 'release_pose': historical,
    }))
    np.testing.assert_array_equal(scene.released[1], corrected)
    scene.event(SimpleNamespace(status=SimpleNamespace(value='succeeded'), diagnostics={
        'step': 2, 'release_pose': newer,
    }))
    np.testing.assert_array_equal(scene.released[1], newer)


def test_scene_result_gate_makes_remaining_refinement_explicit(saved):
    app = QApplication.instance() or QApplication([])
    view = ExecutionView(ROOT)
    view.load(saved.path)
    view.worker = SimpleNamespace(isRunning=lambda: True)
    try:
        view.set_gate(OperatorGate('one', 'command', 1, 'stack_scene_result', 'Review'))
        assert view.continue_button.text() == 'Accept SceneID and refine remaining plan'
        view.set_gate(OperatorGate('two', 'command', 2, 'stack_scene_result', 'Review'))
        assert view.continue_button.text() == 'Continue'
    finally:
        view.worker = None
        view.close()
        app.processEvents()
