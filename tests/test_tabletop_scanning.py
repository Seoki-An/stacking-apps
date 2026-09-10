"""Scan geometry and held-gate execution, using simulated devices only."""

from contextlib import contextmanager
import json
from types import SimpleNamespace

import numpy as np
import pytest

from stacking_common.artifacts import StackRunStore
from stacking_tabletop.acquisition.femto import CaptureError
from stacking_tabletop.acquisition.scan_paths import (
    build_scan, joint_leg, Q_SCAN, PIVOT_POINT_TOOL_M, ScanMotion,
)
from stacking_tabletop.acquisition.scanning import scan_held_object
from stacking_tabletop.execution import UR5eKinematics
from stacking_tabletop.execution.motion import validate_joint_path
from stacking_tabletop.execution.sequence import execute_saved_run
from test_tabletop_saved_execution import ROOT, saved, executor_for


@pytest.mark.parametrize('kind,count', [('wrist', 36), ('pivot', 76)])
def test_native_scan_coverage_limits_and_exact_return(kind, count):
    kin = UR5eKinematics(ROOT / 'apps/tabletop/assets/ur5e/ur5e_ik.urdf')
    anchor = Q_SCAN + .01
    motions = build_scan(kind, kin, anchor)
    assert sum(m.capture for m in motions) == count
    np.testing.assert_array_equal(motions[-1].target, anchor)
    previous = anchor
    for motion in motions:
        validate_joint_path(joint_leg(previous, motion.target))
        assert np.all(motion.target >= kin.lower) and np.all(motion.target <= kin.upper)
        assert abs(motion.target[5]) < np.deg2rad(300)
        previous = motion.target
    if kind == 'wrist':
        assert [round(np.rad2deg(m.target[5])) for m in motions if m.capture] == list(range(-180, 180, 10))
    else:
        pivot = np.r_[PIVOT_POINT_TOOL_M, 1.]
        neutral = kin.pose(Q_SCAN) @ pivot
        for motion in motions:
            if motion.capture:
                assert np.linalg.norm((kin.pose(motion.target) @ pivot)[:3] - neutral[:3]) < .0001
        for side in (0, 1):
            views = [m for m in motions if m.label == f'pivot_side_{side}_view']
            assert len(views) == 11
            assert abs(views[0].pivot_x_angle_deg) == 50
            assert abs(views[-1].pivot_x_angle_deg) == 50
        turns = [m.wrist3_turn_deg for m in motions if m.wrist3_turn_deg is not None]
        assert max(turns) == 360
        assert max(m.target[5] for m in motions) == pytest.approx(np.pi)


def test_scan_preparation_cancellation_and_invalid_branch():
    kin = UR5eKinematics(ROOT / 'apps/tabletop/assets/ur5e/ur5e_ik.urdf')
    def cancel():
        raise InterruptedError('cancel precomputation')
    with pytest.raises(InterruptedError):
        build_scan('wrist', kin, Q_SCAN, check=cancel)
    bad = Q_SCAN.copy()
    bad[5] = np.deg2rad(360)
    with pytest.raises(ValueError):
        build_scan('pivot', kin, bad)
    with pytest.raises(InterruptedError):
        kin.solve(kin.pose(Q_SCAN), Q_SCAN, check=cancel)


class Session:
    """Small storage spy; real capture persistence has separate round-trip tests."""
    instances = []

    def __init__(self, output, calibration, *, in_hand):
        assert in_hand
        output.mkdir()
        self.output, self.views, self.complete = output, [], False
        self.instances.append(self)

    def __enter__(self):
        return self

    def add(self, burst):
        self.views.append(burst)

    def __exit__(self, kind, error, traceback):
        self.complete = kind is None


def short_scan(kind, kin, anchor, *, check):
    check()
    # Small motions exercise the real adapter against the Cartesian fake's FK.
    target = anchor.copy()
    target[0] += .003
    return (ScanMotion(target, True, 'view'), ScanMotion(anchor.copy(), False, 'return_execution_gate'))


def setup_scan(saved, tmp_path, *, mode='success'):
    transport, executor = executor_for(saved)
    kin = executor.arm.kinematics
    kin.lower, kin.upper = np.full(6, -6.28), np.full(6, 6.28)
    kin.urdf_path = saved.assets['robot/urdf']
    gates, previews, anchors = [], [], []
    camera = SimpleNamespace(calibration=None, calls=0)
    Session.instances = []
    paused_at = None

    def capture(*, in_hand, check):
        nonlocal paused_at
        assert in_hand
        camera.calls += 1
        check()
        if mode == 'preflight_failure' or (mode == 'capture_failure' and camera.calls == 2):
            raise CaptureError('simulated missing exact-time TF')
        if mode == 'slip' and camera.calls == 2:
            for _ in range(5):
                transport.feedback.receive_gripper(transport.position + 100)
            check()
        if mode == 'pause' and camera.calls == 2:
            executor.hold(executor.active_command_id)
            paused_at = transport.now
            check()
        joints = transport.joints.copy()
        if mode == 'wrong_joints' and camera.calls == 2:
            joints[0] += .1
        return [SimpleNamespace(robot={'joints': joints}, number=camera.calls)] * 7
    camera.capture = capture

    def gate(value):
        if value is None:
            return
        gates.append(value)
        if value.name == 'place' and len(anchors) == 0:
            anchors.append(transport.joints.copy())
            assert not executor.request_scan('stale', 'wrist', 'primary', tmp_path / 'scan')
            assert executor.request_scan(value.token, 'wrist', 'primary', tmp_path / 'scan')
            assert not executor.request_scan(value.token, 'pivot', 'secondary', tmp_path / 'scan2')
            assert not executor.approve(value.token)
        elif value.name == 'scan_review' and mode == 'cancel_review':
            executor.cancel(value.command_id)
        else:
            if value.name == 'place' and value.step == 1:
                np.testing.assert_allclose(transport.joints, anchors[0], atol=1e-9)
                original = next(g for g in gates if g.name == 'place')
                assert value.token != original.token
                assert not executor.approve(original.token)
            executor.approve(value.token)

    def scan(ex, request):
        scan_held_object(ex, request, camera, planner=short_scan,
                         session_factory=Session, prepared=previews.append)
    executor.scan_handler = scan
    executor.gate_changed = gate
    if mode == 'cancel_motion':
        execute_path = executor.arm.execute_path
        def move(path, **kwargs):
            progress = kwargs.get('progress')
            def sample(index, count, q):
                if progress is not None:
                    progress(index, count, q)
                if executor._phase.startswith('scan/') and index > 0:
                    executor.cancel(executor.active_command_id)
            kwargs['progress'] = sample
            return execute_path(path, **kwargs)
        executor.arm.execute_path = move
    if mode == 'storage_failure':
        def failed_add(self, burst):
            raise OSError('simulated disk full')
        class FailedSession(Session):
            add = failed_add
        executor.scan_handler = lambda ex, request: scan_held_object(ex, request, camera,
            planner=short_scan, session_factory=FailedSession, prepared=previews.append)
    def sleep():
        if paused_at is not None and transport.now - paused_at > 65:
            executor.resume(executor.active_command_id)
    transport.on_sleep = sleep
    return transport, executor, camera, gates, previews


@pytest.mark.parametrize('mode', ['success', 'pause'])
def test_scanning_returns_to_fresh_gate_and_keeps_durable_sequence(saved, tmp_path, mode):
    transport, executor, camera, gates, previews = setup_scan(saved, tmp_path, mode=mode)
    results = execute_saved_run(saved, executor)
    assert all(result.status.value == 'succeeded' for result in results)
    assert [g.name for g in gates] == ['review', 'close', 'place', 'scan_review', 'place', 'release',
                                       'review', 'close', 'place', 'release']
    assert Session.instances[0].complete
    assert len(Session.instances[0].views) == 1
    assert camera.calls == (3 if mode == 'pause' else 2)
    assert Session.instances[0].views[0][0].number == camera.calls
    assert previews[0]['capture_count'] == 1
    details = json.loads((tmp_path / 'scan/scan_plan.json').read_text())
    assert details['context_sha256'] == saved.context.sha256
    assert details['robot_urdf_sha256'] == saved.context.asset_hashes['robot/urdf']
    events = StackRunStore.load_execution_events(saved.path)
    assert any('scan_completed' in e.diagnostics for e in events)
    assert sum(e.diagnostics.get('operator_decision') == 'approve_place' for e in events) == 2
    if mode == 'pause':
        assert executor.runtime.paused_seconds > 65


@pytest.mark.parametrize('mode,failure', [
    ('preflight_failure', 'perception_failed'), ('capture_failure', 'perception_failed'),
    ('wrong_joints', 'perception_failed'), ('cancel_review', 'cancelled'), ('slip', 'gripper_fault'),
    ('cancel_motion', 'cancelled'), ('storage_failure', 'controller_fault'),
])
def test_scan_failure_never_releases_returns_or_advances(saved, tmp_path, mode, failure):
    transport, executor, camera, gates, previews = setup_scan(saved, tmp_path, mode=mode)
    results = execute_saved_run(saved, executor)
    assert len(results) == 1 and results[0].failure.value == failure
    assert [call[2] for call in transport.calls if call[0] == 'flag'] == [2]  # Initial open only.
    assert StackRunStore.load_execution_checkpoint(saved.path).requires_reconciliation
    assert len(StackRunStore.load_execution_commands(saved.path)) == 1
    assert not any(g.name == 'release' for g in gates)
    assert not any('scan_completed' in e.diagnostics for e in StackRunStore.load_execution_events(saved.path))
    if mode in ('capture_failure', 'wrong_joints', 'slip', 'storage_failure'):
        assert transport.joints[0] == pytest.approx(.028)  # Stays at view, no automatic return.
    if mode == 'cancel_motion':
        assert .025 < transport.joints[0] < .028
        assert camera.calls == 1  # Only preflight, no scan view.
    if Session.instances:
        assert not Session.instances[0].complete
    else:
        assert mode == 'preflight_failure'


def test_worker_closes_camera_on_scan_failure(saved, monkeypatch):
    pytest.importorskip('PySide6')
    from stacking_tabletop.gui.operations.execution_worker import ExecutionWorker
    closed = []
    @contextmanager
    def camera_factory(name, runtime):
        assert name == 'secondary'
        try:
            yield object()
        finally:
            closed.append(True)
    def fail(*args, **kwargs):
        raise CaptureError('failed camera')
    monkeypatch.setattr('stacking_tabletop.acquisition.scanning.scan_held_object', fail)
    worker = ExecutionWorker(saved, camera_factory=camera_factory)
    with pytest.raises(CaptureError):
        worker._scan(None, ('pivot', 'secondary', '/unused'))
    assert closed == [True]
