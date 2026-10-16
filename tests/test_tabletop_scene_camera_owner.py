"""Pick-scene capture shares the bringup camera without opening a second pipeline."""
from types import SimpleNamespace as NS

import pytest

from stacking_tabletop.perception import service_client as client
from stacking_tabletop.perception.service import PerceptionResources


@pytest.fixture
def remote(tmp_path, monkeypatch):
    from stacking_tabletop.acquisition import hand_eye_capture, realsense
    monkeypatch.setattr(client, 'service_directory', lambda _: tmp_path)
    (tmp_path / 'service.sock').touch()
    monkeypatch.setattr(hand_eye_capture, 'digest', lambda _: 'profile-hash')
    def forbidden(*args, **kwargs):
        pytest.fail('opened a second camera')
    monkeypatch.setattr(realsense, 'RealSenseCamera', forbidden)
    arm = NS(_check_motion=lambda: None, runtime=NS(paused_seconds=0.))
    calibration = NS(profile=NS(serial='pinned'), report=lambda: {'artifacts': 'reviewed'})
    return tmp_path, arm, calibration


def test_scene_capture_uses_persistent_owner_and_preserves_options(remote, monkeypatch):
    path, arm, calibration = remote
    calls = []
    def request(repository, operation, **kwargs):
        calls.append((operation, kwargs))
        return dict(wrist_capture_options_version=2, wrist_enabled=True)
    monkeypatch.setattr(client, 'request', request)
    with client.scene_wrist(arm, calibration, path, depth_size=(848, 480), fps=15, count=10) as capture:
        capture(path / 'view')
    assert [op for op, _ in calls] == ['ping', 'wrist_ready', 'wrist_capture']
    for _, args in calls[1:]:
        assert args['count'] == 10 and args['fps'] == 15
        assert args['capture_depth'] is False
        assert args['calibration_report'] == calibration.report()
        assert args['serial'] == 'pinned'
    assert calls[-1][1]['directory'] == str(path / 'view')


@pytest.mark.parametrize('failure', ['old_service', 'busy', 'disconnected', 'hold'])
def test_service_failure_never_falls_back_to_direct_camera(remote, monkeypatch, failure):
    path, arm, calibration = remote
    def request(repository, operation, **kwargs):
        if failure == 'old_service':
            return {'ready': True}
        if operation == 'ping':
            return dict(wrist_capture_options_version=2, wrist_enabled=True)
        if failure in ('busy', 'disconnected'):
            raise RuntimeError(failure)
        if operation == 'wrist_capture':
            arm.runtime.paused_seconds += 1
    monkeypatch.setattr(client, 'request', request)
    with pytest.raises(RuntimeError, match={'old_service': 'restart hardware', 'hold': 'hold interrupted'}.get(failure, failure)):
        with client.scene_wrist(arm, calibration, path, depth_size=(848, 480), fps=15, count=10) as capture:
            capture(path / 'view')


@pytest.mark.parametrize('mismatch', [None, 'calibration', 'fps', 'size'])
def test_server_checks_reviewed_geometry_and_streams_before_saving(tmp_path, monkeypatch, mismatch):
    from stacking_tabletop.acquisition import hand_eye_capture
    resources = PerceptionResources(tmp_path)
    resources.calibration_hash = 'profile-hash'
    resources.calibration = NS(profile=NS(serial='pinned'), report=lambda: {'artifacts': 'reviewed'})
    resources.robot = object()
    streams = dict(depth=dict(width=848, height=480, fps=15), color=dict(fps=15))
    monkeypatch.setattr(resources, 'wrist_ready', lambda _, **kwargs: NS(report=dict(streams=streams)))
    calls = []
    monkeypatch.setattr(hand_eye_capture, 'capture_view', lambda *args, **kwargs: calls.append(kwargs))
    arguments = dict(profile_sha256='profile-hash', serial='pinned', calibration_report=resources.calibration.report(),
                     depth_size=[848, 480], fps=15, count=10, directory=str(tmp_path / 'view'))
    if mismatch == 'calibration':
        arguments['calibration_report'] = {'artifacts': 'wrong'}
    elif mismatch == 'fps':
        arguments['fps'] = 10
    elif mismatch == 'size':
        arguments['depth_size'] = [640, 480]
    if mismatch:
        with pytest.raises(ValueError, match='differs'):
            resources.dispatch('wrist_capture', arguments, lambda: None)
        assert not calls
    else:
        resources.dispatch('wrist_capture', arguments, lambda: None)
        assert calls == [{'count': 10}]


@pytest.mark.parametrize('service_disabled', [False, True])
def test_local_camera_when_service_absent_or_wrist_disabled(tmp_path, monkeypatch, service_disabled):
    from stacking_tabletop.acquisition import realsense, hand_eye_capture
    from stacking_tabletop.ros import hand_eye
    monkeypatch.setattr(client, 'service_directory', lambda _: tmp_path)
    if service_disabled:
        (tmp_path / 'service.sock').touch()
    def request(*args, **kwargs):
        assert service_disabled
        return dict(wrist_capture_options_version=2, wrist_enabled=False)
    monkeypatch.setattr(client, 'request', request)
    events = []
    class Robot:
        def __init__(self, *_): pass
        def __enter__(self): return self
        def __exit__(self, *_): events.append('robot_closed')
    class Camera:
        report = {}
        def __init__(self, calibration, **kwargs):
            events.append(kwargs)
        def __enter__(self): return self
        def __exit__(self, *_): events.append('camera_closed')
    monkeypatch.setattr(hand_eye, 'HandEyeRobot', Robot)
    monkeypatch.setattr(realsense, 'RealSenseCamera', Camera)
    monkeypatch.setattr(hand_eye_capture, 'capture_view', lambda *args, **kwargs: events.append(kwargs))
    arm = NS(_check_motion=lambda: None)
    with client.scene_wrist(arm, object(), tmp_path, depth_size=(848, 480), fps=10, count=7) as capture:
        capture(tmp_path / 'view')
    assert events == [dict(depth_size=(848, 480), fps=10, capture_depth=False), dict(count=7), 'camera_closed', 'robot_closed']
