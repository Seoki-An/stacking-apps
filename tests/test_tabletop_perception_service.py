"""Persistent perception protocol and cancellation; no physical devices."""

import threading
import time
from types import SimpleNamespace

import pytest

from stacking_tabletop.perception import service_client as client
from stacking_tabletop.perception.service import PerceptionServer, PerceptionResources


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(client, 'service_directory', lambda repository: tmp_path)
    calls = []
    cancelled = threading.Event()
    def dispatch(operation, arguments, check):
        calls.append((operation, arguments))
        if operation == 'wait':
            try:
                while True:
                    check()
                    time.sleep(.01)
            except InterruptedError:
                cancelled.set()
                raise
        if operation == 'fail':
            raise ValueError('bad observation')
        return {'call': len(calls)}
    instance = PerceptionServer(str(tmp_path/'service.sock'), SimpleNamespace(dispatch=dispatch))
    thread = threading.Thread(target=instance.serve_forever, kwargs={'poll_interval': .01})
    thread.start()
    try:
        yield calls, cancelled
    finally:
        instance.stopping.set()
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=2.)


def test_service_reuses_process_and_survives_request_failure(server):
    calls, _ = server
    assert client.request('.', 'ping')['ready']
    assert client.request('.', 'identify', object_id=3) == {'call': 1}
    with pytest.raises(RuntimeError, match='bad observation'):
        client.request('.', 'fail')
    assert client.request('.', 'identify', object_id=4) == {'call': 3}
    assert calls[0][1] == {'object_id': 3}


def test_client_cancellation_interrupts_service_job(server):
    _, cancelled = server
    checks = 0
    def check():
        nonlocal checks
        checks += 1
        if checks >= 3:
            raise InterruptedError('operator abort')
    with pytest.raises(InterruptedError, match='operator abort'):
        client.request('.', 'wait', check=check)
    assert cancelled.wait(2.)
    assert client.request('.', 'ping')['ready']


def test_service_timeout_disconnects_and_releases_job(server):
    _, cancelled = server
    with pytest.raises(TimeoutError):
        client.request('.', 'wait', timeout=.05)
    assert cancelled.wait(2.)


def test_missing_service_reports_hardware_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(client, 'service_directory', lambda repository: tmp_path)
    with pytest.raises(RuntimeError, match='start_perception:=true'):
        client.request('.', 'ping')


def test_wrist_pipeline_reused_and_closed_on_failure(tmp_path, monkeypatch):
    import stacking_tabletop.acquisition.hand_eye as calibration_module
    import stacking_tabletop.acquisition.hand_eye_capture as capture
    import stacking_tabletop.acquisition.realsense as realsense
    import stacking_tabletop.ros.hand_eye as ros
    cameras = []
    class Camera:
        def __init__(self, *args, **kwargs):
            self.report = {'streams': {'color': {}}}
            if kwargs.get('capture_depth', True):
                self.report['streams']['depth'] = {}
            self.closed = False
            cameras.append(self)
        def close(self):
            self.closed = True
    class Robot:
        def __init__(self, *_): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
    monkeypatch.setattr(calibration_module.HandEyeCalibration, 'load', lambda _: object())
    monkeypatch.setattr(capture, 'digest', lambda _: 'hash')
    monkeypatch.setattr(capture, 'warmup', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realsense, 'RealSenseCamera', Camera)
    monkeypatch.setattr(ros, 'HandEyeRobot', Robot)
    resources = PerceptionResources(tmp_path)
    resources.wrist_ready(lambda: None)
    resources.wrist_ready(lambda: None)
    assert len(cameras) == 1
    resources.wrist_ready(lambda: None, capture_depth=False)
    resources.wrist_ready(lambda: None, capture_depth=False)
    assert len(cameras) == 2 and cameras[0].closed
    assert set(cameras[1].report['streams']) == {'color'}
    def fail(*_args, **_kwargs):
        raise RuntimeError('no frame')
    monkeypatch.setattr(capture, 'warmup', fail)
    with pytest.raises(RuntimeError, match='no frame'):
        resources.wrist_ready(lambda: None, capture_depth=False)
    assert cameras[1].closed and resources.camera is None
    resources.close()


def test_stack_processing_dispatch_preserves_repository_and_cancellation(tmp_path, monkeypatch):
    from stacking_tabletop.acquisition import stack_scene_processing, hand_eye_fast_stereo
    calls = []
    def reconstruct(*args, check, **kwargs):
        check()
        calls.append(('reconstruct', args))
    def process(run, repository, capture, step, *, reconstruct):
        assert repository == tmp_path
        assert (run, capture, step) == ('run.stack', 'capture', 2)
        reconstruct('session')
        return {'status': 'complete'}
    monkeypatch.setattr(stack_scene_processing, 'process_stack_scene', process)
    monkeypatch.setattr(hand_eye_fast_stereo, 'reconstruct_with_runtime', reconstruct)
    check = lambda: calls.append(('check',))
    resources = PerceptionResources(tmp_path)
    assert resources.dispatch('stack_scene', dict(run='run.stack', capture='capture', step=2),
                              check) == {'status': 'complete'}
    assert calls == [('check',), ('check',), ('reconstruct', ('session',))]


def test_execution_worker_routes_identification_to_server(tmp_path, monkeypatch):
    from contextlib import nullcontext
    import numpy as np
    from stacking_tabletop.acquisition import scanning
    from stacking_tabletop.gui.operations.execution_worker import ExecutionWorker
    from stacking_tabletop.perception.inhand_execution import InHandIdentification
    result = InHandIdentification(
        object_id=3, initial_tool_T_object=np.eye(4), identified_tool_T_object=np.eye(4),
        accepted=True, reasons=(), correction_translation_m=0., correction_rotation_degrees=0.,
        target_coverage=1., source_coverage=1., rmse_m=0., correspondences=100,
        observation_points=100, capture_directory=tmp_path,
        report_path=tmp_path/'inhand_identification.json',
    )
    monkeypatch.setattr(scanning, 'scan_held_object', lambda *args, **kwargs: tmp_path)
    calls = []
    def request(repository, operation, **kwargs):
        calls.append((repository, operation, kwargs))
        return result.report()
    monkeypatch.setattr(client, 'request', request)
    executor = SimpleNamespace(held_object_id=3, _health=lambda: None,
                               _emit=lambda *args, **kwargs: None,
                               initial_tool_T_object=lambda: np.eye(4))
    worker = ExecutionWorker(None, repository=tmp_path,
                             camera_factory=lambda *args: nullcontext(object()))
    actual = worker._scan(executor, ('held_scan', 'primary'))
    assert actual.accepted and actual.object_id == 3
    assert calls[0][:2] == (tmp_path, 'inhand')
    assert calls[0][2]['capture'] == str(tmp_path)
    # A stale/misrouted result must not change the held attachment estimate.
    monkeypatch.setattr(client, 'request', lambda *args, **kwargs: dict(result.report(), object_id=4))
    rejected = worker._scan(executor, ('held_scan', 'primary'))
    assert not rejected.accepted and 'another held object' in rejected.error


def test_service_routes_refinement_without_starting_camera(tmp_path, monkeypatch):
    from stacking_tabletop.execution import scene_refinement
    calls = []
    def refine(*, repository, run, capture, step, check):
        check()
        calls.append((repository, run, capture, step))
        return {'run': 'refined.stack'}
    monkeypatch.setattr(scene_refinement, 'refine_execution_run', refine)
    resources = PerceptionResources(tmp_path)
    reply = resources.dispatch('refine_remaining',
        dict(run='source.stack', capture='capture', step=2), lambda: None)
    assert reply == {'run': 'refined.stack'}
    assert calls == [(tmp_path, 'source.stack', 'capture', 2)]
    assert resources.camera is None
