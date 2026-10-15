"""Qt controls exercise the real executor/worker with simulated hardware only."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('PySide6')
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtGui import QCloseEvent

from stacking_common.artifacts import StackRunStore
from stacking_tabletop.execution import DynamixelAdapter, HardwareFeedback, UR5eAdapter
from stacking_tabletop.execution.motion import UR_JOINT_NAMES
from stacking_tabletop.gui.operations.execution_view import ExecutionView
from stacking_tabletop.gui.operations.execution_worker import ExecutionWorker
from test_tabletop_hardware_adapters import FakeTransport, Kinematics
from test_tabletop_saved_execution import ROOT, saved


def test_loading_validated_scene_is_passive_and_uses_recorded_assets(saved):
    app = QApplication.instance() or QApplication([])
    view = ExecutionView(ROOT, worker_factory=lambda *_: pytest.fail('loading must not connect'))
    view.load(saved.path)
    assert view.start_button.isEnabled()
    assert not view.continue_button.isEnabled()
    assert not view.manual_controls.isEnabled()
    frame = view.scene.frame()
    assert frame.robot_parts == ()  # No measured state yet.
    assert frame.environment_boxes[0].bounds == pytest.approx(tuple(v for pair in zip(
        saved.planning.obstacles[0].minimum, saved.planning.obstacles[0].maximum) for v in pair))
    assert frame.object_mesh_paths[1] == saved.assets['object/1']
    assert view.scene.projector.planning is saved.planning or view.scene.projector.planning.robot.home_positions[0] == 0
    view.close()
    app.processEvents()


def test_execution_tab_shows_read_only_live_robot_state(saved):
    from stacking_tabletop.execution.sequence import OperatorGate

    app = QApplication.instance() or QApplication([])
    feedback = HardwareFeedback()
    feedback.receive_joints(UR_JOINT_NAMES, saved.planning.robot.home_positions)
    feedback.receive_pose(np.eye(4))
    feedback.receive_gripper(700)

    class PreviewTransport:
        def __init__(self):
            self.feedback = feedback
            self.closed = False

        def close(self):
            self.closed = True

    transport = PreviewTransport()
    view = ExecutionView(ROOT, preview_transport_factory=lambda: transport)
    view.load(saved.path)
    view.set_live_preview_enabled(True)
    try:
        assert len(view.scene.frame().robot_parts) > 0
        assert "Measured joints" in view.telemetry_label.text()
        assert "700 counts" in view.telemetry_label.text()
        assert transport.closed is False
        view.worker = SimpleNamespace(isRunning=lambda: True)
        view.set_gate(OperatorGate(
            "align", "command", 1, "start_alignment", "Align to saved home"
        ))
        np.testing.assert_array_equal(
            view.scene.scan_preview, saved.planning.robot.home_positions
        )
        assert any(part.part_id.startswith("scan_preview/")
                   for part in view.scene.frame().robot_parts)
        view.set_gate(OperatorGate(
            "review", "command", 1, "review", "Review saved motion"
        ))
        frame = view.scene.frame()
        assert frame.motion_path_points.shape[1] == 3
        assert len(frame.motion_path_points) > 1
        assert view.motion_review_controls.isEnabled()
        assert view.motion_slider.maximum() > 0
        preview = [part for part in frame.robot_parts
                   if part.part_id.startswith("motion_preview/")]
        assert preview and all(part.color == "#57e389" for part in preview)
        view.motion_slider.setValue(1)
        np.testing.assert_array_equal(
            view.scene.motion_preview[0], view._motion_samples[1][3]
        )
        assert "saved sample 2/" in view.motion_label.text()
        view.motion_play_button.click()
        assert view._motion_preview_timer.isActive()
        view.set_gate(None)
        assert not view._motion_preview_timer.isActive()
        assert not view.motion_review_controls.isEnabled()
        assert view.scene.frame().motion_path_points is None
        assert not any(part.part_id.startswith("motion_preview/")
                       for part in view.scene.frame().robot_parts)
        view.worker = None
    finally:
        view.set_live_preview_enabled(False)
        view.close()
        app.processEvents()
    assert transport.closed is True


def test_worker_gates_and_abort_leave_durable_recovery_requirement(saved):
    app = QApplication.instance() or QApplication([])
    closed = []
    def factory(saved, runtime):
        transport = FakeTransport()
        def sleep(duration):
            transport.sleep(duration)
            time.sleep(.0001)
        runtime.clock = lambda: transport.now
        runtime.sleep = sleep
        transport.close = lambda: closed.append(True)
        kin = Kinematics()
        kin.grasp_pose = kin.pose
        gripper = DynamixelAdapter(transport, runtime=runtime)
        arm = UR5eAdapter(transport, kin, gripper=gripper)
        return transport, arm, gripper
    view = ExecutionView(ROOT, worker_factory=lambda saved, parent: ExecutionWorker(saved, parent, device_factory=factory))
    view.load(saved.path)
    view.stack_capture.setChecked(False)  # This test exercises the existing pickup/abort gates.
    view.start_button.click()
    gates = []
    seen = set()
    deadline = time.monotonic() + 10
    try:
        while view.running and time.monotonic() < deadline:
            app.processEvents()
            if view.gate is not None and view.gate.token not in seen:
                seen.add(view.gate.token)
                gates.append(view.gate.name)
                assert view.continue_button.isEnabled()
                assert view.manual_controls.isEnabled() == view.gate.manual
                if view.gate.name == 'place':
                    view.abort_button.click()
                    assert not view.scan_controls.isEnabled()
                else:
                    view.continue_button.click()
                    assert not view.continue_button.isEnabled()
            time.sleep(.001)
        app.processEvents()
        assert not view.running
        assert gates == ['review', 'close', 'place']
        assert closed == [True]
        assert view.reconcile_button.isEnabled()
        assert not view.start_button.isEnabled()
        assert StackRunStore.load_execution_checkpoint(saved.path).requires_reconciliation
        assert len(view.scene.frame().robot_parts) > 0
    finally:
        view.abort()
        if view.worker is not None:
            assert view.worker.wait(3000)
        view.close()
        app.processEvents()


def test_window_refuses_close_and_artifact_switch_until_worker_exits():
    from stacking_tabletop.gui.operations.window import StackingOperationsWindow
    app = QApplication.instance() or QApplication([])
    aborted = []
    class Window(StackingOperationsWindow):
        def __init__(self):
            QMainWindow.__init__(self)
            self._video_exporting = False
            self.execution_view = SimpleNamespace(
                running=True,
                abort=lambda: aborted.append(True),
                set_live_preview_enabled=lambda _enabled: None,
            )
    window = Window()
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted() and aborted == [True]
    window.open_artifact('/does/not/exist.stack')
    window._save_trajectory_video(30)
    window.execution_view.running = False
    window.close()
    app.processEvents()


def test_scan_controls_request_review_preview_and_clear(saved, tmp_path, monkeypatch):
    from stacking_tabletop.execution.sequence import OperatorGate
    from PySide6.QtWidgets import QFileDialog
    app = QApplication.instance() or QApplication([])
    view = ExecutionView(ROOT)
    view.load(saved.path)
    assert not view.scan_controls.isEnabled()
    requests = []
    executor = SimpleNamespace(request_scan=lambda *args: requests.append(args) or True)
    view.worker = SimpleNamespace(executor=executor, isRunning=lambda: True)
    gate = OperatorGate('place-token', 'command', 1, 'place', 'Inspect held object')
    view.set_gate(gate)
    view.receive_telemetry({'joints': [0.] * 6, 'gripper_raw': 500, 'holding': True, 'object_pose': None})
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args: str(tmp_path))
    view.scan_kind.setCurrentText('pivot')
    view.scan_camera.setCurrentText('secondary')
    view.scan_button.click()
    assert requests[0][:3] == ('place-token', 'pivot', 'secondary')
    assert requests[0][3].parent == tmp_path and not requests[0][3].exists()
    assert not view.continue_button.isEnabled() and not view.scan_controls.isEnabled()
    view.set_gate(None)
    view.prepare_scan_preview({'positions': [[0.] * 6, [.1] * 6], 'kind': 'pivot',
                              'capture_count': 76, 'output': str(requests[0][3])})
    view.set_gate(OperatorGate('scan-token', 'command', 1, 'scan_review', 'Review scan'))
    assert view.continue_button.isEnabled() and not view.scan_controls.isEnabled()
    assert view.scan_preview.isEnabled()
    view.scan_preview.setValue(1)
    parts = view.scene.frame().robot_parts
    preview = [p for p in parts if p.part_id.startswith('scan_preview/')]
    measured = [p for p in parts if not p.part_id.startswith('scan_preview/')]
    assert len(preview) == len(measured) > 0
    assert all(p.color == '#1ad9ff' for p in preview)
    import pyvista as pv
    assert pv.Color(preview[0].color).hex_rgb == '#1ad9ff'
    view.set_gate(None)
    assert view.scene.scan_preview is None and not view.scan_preview.isEnabled()
    assert len(view.scene.frame().robot_parts) == len(measured)
    view.worker = None
    view.close()
    app.processEvents()


def test_scan_directory_dialog_cannot_submit_to_a_departed_gate(saved, tmp_path, monkeypatch):
    from stacking_tabletop.execution.sequence import OperatorGate
    from PySide6.QtWidgets import QFileDialog
    app = QApplication.instance() or QApplication([])
    view = ExecutionView(ROOT)
    view.load(saved.path)
    view.worker = SimpleNamespace(isRunning=lambda: True,
        executor=SimpleNamespace(request_scan=lambda *args: pytest.fail('gate is gone')))
    view.set_gate(OperatorGate('token', 'command', 1, 'place', 'Inspect held object'))
    def choose(*args):
        view.set_gate(None)  # Simulate a queued abort/fault signal in the modal event loop.
        return str(tmp_path)
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', choose)
    view.scan_button.click()
    view.worker = None
    view.close()
    app.processEvents()


def test_stack_capture_option_preview_and_observation(saved, tmp_path):
    import numpy as np
    from stacking_tabletop.execution.sequence import OperatorGate
    from stacking_tabletop.acquisition.storage import write_xyz_pcd
    from stacking_tabletop.acquisition.hand_eye_capture import digest
    app = QApplication.instance() or QApplication([])
    view = ExecutionView(ROOT)
    view.load(saved.path)
    try:
        assert view.stack_capture.isChecked() and view.stack_capture.isEnabled()
        view.worker = SimpleNamespace(executor=None, isRunning=lambda: True)
        view.set_gate(OperatorGate('token','command',1,'stack_scan_review','Review wrist route'))
        view.prepare_scan_preview(dict(positions=[[0.]*6,[.1]*6],kind='stacked scene RGB',capture_count=8,output=str(tmp_path)))
        assert view.scan_preview.isEnabled() and not view.stack_capture.isEnabled()
        assert not view.scan_controls.isEnabled()
        pcd=tmp_path/'scene.pcd';write_xyz_pcd(pcd,np.array([[.1,.2,.3]]))
        viewer=tmp_path/'overview.html';viewer.write_text('<html>PCD</html>')
        view.receive_event(SimpleNamespace(sequence=1,status=SimpleNamespace(value='held'),reason=None,
            diagnostics=dict(stack_scene_completed=str(tmp_path),stack_scene_pcd=str(pcd),
                stack_scene_pcd_sha256=digest(pcd),stack_scene_viewer=str(viewer))))
        assert view.stack_viewer_button.isEnabled()
        np.testing.assert_allclose(view.scene.frame().observed_scene_points,[[-.1,-.2,.3]])
        view.set_gate(OperatorGate('token2','command',1,'stack_scene_result','Review PCD'))
        assert not view.scan_preview.isEnabled() and view.continue_button.isEnabled()
    finally:
        view.worker=None
        view.close();app.processEvents()
