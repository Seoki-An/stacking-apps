"""Qt process supervision and planning handoff for the field workflow."""
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip('PySide6')
from PySide6.QtCore import QProcess
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget
from stacking_tabletop.gui.operations.field_scan import FieldScanDialog, inspect_joint_pick_output

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def wait_for(condition):
    deadline = time.monotonic() + 10
    while not condition() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert condition()


def test_field_dialog_builds_cli_and_rejects_camera_conflict(app, tmp_path):
    dialog = FieldScanDialog(ROOT)
    try:
        dialog.output.setText(str(tmp_path / 'capture'))
        args = dialog.arguments()
        assert 'stacking_tabletop.acquisition.field_workflow' in args
        assert '--visualize' in args
        assert '--automatic-ordering' not in args
        dialog.camera.setCurrentIndex(2)
        assert '--both-cameras' in dialog.arguments()
        dialog.running_camera.setChecked(True)
        with pytest.raises(ValueError, match='own'):
            dialog.arguments()
        dialog.mode.setCurrentIndex(1)
        (tmp_path / 'field_scan.json').write_text('{}')
        dialog.source.setText(str(tmp_path))
        assert 'stacking_tabletop.perception.field_cli' in dialog.arguments()
        assert '--both-cameras' not in dialog.arguments()
        assert not dialog.running
        assert not dialog.plan_button.isEnabled()
    finally:
        dialog.close()


def test_field_dialog_builds_integrated_wrist_rgb_femto_workflows(app, tmp_path):
    dialog = FieldScanDialog(ROOT)
    capture = tmp_path / 'capture'
    capture.mkdir()
    try:
        dialog.mode.setCurrentIndex(2)
        dialog.output.setText(str(tmp_path / 'pick-live'))
        args = dialog.arguments()
        assert args[2:4] == ['stacking_tabletop.acquisition.pick_scene_workflow', 'run']
        assert '--plan' not in args and '--approve' not in args
        assert '--output' in args and '--camera' in args
        assert not dialog.automatic.isEnabled()
        assert 'before the first' in dialog.note.text()
        dialog.camera.setCurrentIndex(2)
        with pytest.raises(ValueError, match='one fixed FEMTO'):
            dialog.arguments()
        dialog.camera.setCurrentIndex(0)
        dialog.mode.setCurrentIndex(3)
        dialog.output.setText(str(tmp_path / 'pick-existing'))
        dialog.source.setText(str(capture))
        args = dialog.arguments()
        assert args[2:4] == ['stacking_tabletop.acquisition.pick_scene_workflow', 'process']
        assert str(capture) in args and '--output' in args
    finally:
        dialog.close()


def test_field_dialog_resumes_failed_joint_fit_without_capture_or_mvs(app, tmp_path):
    failed = tmp_path / 'failed-pick'
    failed.mkdir()
    (failed / 'pick_scene_workflow.json').write_text('{}')
    dialog = FieldScanDialog(ROOT)
    try:
        dialog.mode.setCurrentIndex(4)
        dialog.source.setText(str(failed))
        dialog.manual.setChecked(True)
        args = dialog.arguments()
        assert args[2:4] == ['stacking_tabletop.acquisition.pick_scene_workflow', 'resume']
        assert str(failed) in args and '--manual-init-gui' in args
        assert '--output' not in args and '--camera' not in args
        assert not dialog.output.isEnabled()
        assert 'No camera capture or MVS reconstruction is repeated.' in dialog.note.text()
    finally:
        dialog.close()


def test_completed_process_enables_explicit_planning_handoff(app, tmp_path, monkeypatch):
    dialog = FieldScanDialog(ROOT)
    destination = tmp_path / 'new-field'
    dialog.output.setText(str(destination))
    real_start = dialog.process.start
    code = (
        'from pathlib import Path; import json; '
        f'p=Path({str(destination)!r}); p.mkdir(); '
        'poses=p/"pose_data.pkl"; poses.write_bytes(b"fixture"); '
        '(p/"field_workflow.json").write_text(json.dumps({"status":"complete","pose_data":str(poses)})); '
        'print("Identification finished", flush=True)'
    )
    monkeypatch.setattr(dialog.process, 'start', lambda *args: real_start(sys.executable, ['-u', '-c', code]))
    received = []
    dialog.poses_available.connect(received.append)
    try:
        dialog.start()
        wait_for(lambda: dialog.plan_button.isEnabled())
        assert not received  # Never starts planning automatically.
        assert 'Identification finished' in dialog.log.toPlainText()
        dialog.plan_button.click()
        assert received == [destination / 'pose_data.pkl']
    finally:
        dialog.close()


def test_pick_workbench_uses_only_rgb_bound_joint_poses_and_opens_rgb_cloud(app, tmp_path, monkeypatch):
    root = tmp_path / 'joint-pick'
    dense = root / 'rgb_dense'
    report_dir = root / 'pose_identification'
    geometry = root / 'femto_geometry'
    fused = root / 'fused_identification'
    for directory in (dense / 'refined', dense / 'rgb_only', report_dir, geometry, fused / 'comparison'):
        directory.mkdir(parents=True, exist_ok=True)
    (dense / 'summary.json').write_text('{"status":"complete"}')
    (geometry / 'geometry.json').write_text('{"catalog_pose_fitted":false}')
    (dense / 'refined/rgb_dense_voxel_1mm.pcd').write_text('RGB cloud fixture')
    viewer = dense / 'rgb_only/overview.html'
    viewer.write_text('<html>RGB only</html>')
    pose_viewer = fused / 'comparison/overview.html'
    pose_viewer.write_text('<html>Joint pose comparison</html>')
    (fused / 'pose_data.pkl').write_bytes(b'joint-pose')
    (root / 'pose_data.pkl').write_bytes(b'joint-pose')
    report = {'status': 'complete', 'pose_source': 'rgb_femto_first_joint_icp',
              'kind': 'femto_rgb_joint_pose_identification',
              'source_sha256': {'dense_summary.json': hashlib.sha256(
                  (dense / 'summary.json').read_bytes()).hexdigest(),
                  'femto_geometry.json': hashlib.sha256(
                  (geometry / 'geometry.json').read_bytes()).hexdigest()},
              'poses': [{'stone_id': 9}]}
    (report_dir / 'stone_poses.json').write_text(json.dumps(report))
    (root / 'pick_scene_workflow.json').write_text(json.dumps({
        'schema_version': 2, 'status': 'complete', 'production_pose_accepted': True,
        'pose_sequence': 'femto_regions_rgb_mvs_joint_catalog_fit',
        'dense_dir': str(dense), 'fused_pose_data': str(fused / 'pose_data.pkl'),
        'pose_viewer': str(pose_viewer)}))
    published = {'status': 'complete', 'production_pose_accepted': True,
                 'pose_source': 'rgb_femto_first_joint_icp',
                 'preliminary_femto_identification': None,
                 'coarse_femto_geometry': 'femto_geometry/geometry.json',
                 'pose_viewer': 'fused_identification/comparison/overview.html',
                 'pose_data': str(root / 'pose_data.pkl'),
                 'joint_identification': str(report_dir / 'stone_poses.json'),
                 'pose_warnings': [{'stone_id': 9}]}
    (root / 'pick_scene_poses.json').write_text(json.dumps(published))

    dialog = FieldScanDialog(ROOT)
    opened = []
    monkeypatch.setattr('stacking_tabletop.gui.operations.field_scan.QDesktopServices.openUrl',
                        lambda url: opened.append(url.toLocalFile()) or True)
    try:
        dialog.mode.setCurrentIndex(3)
        dialog.output.setText(str(root))
        dialog._finished(0, QProcess.ExitStatus.NormalExit)
        assert dialog.pose_path == root / 'pose_data.pkl'
        assert dialog.rgb_viewer == viewer
        assert dialog.pose_viewer == pose_viewer
        assert dialog.rgb_button.isEnabled() and dialog.pose_button.isEnabled()
        assert dialog.plan_button.isEnabled()
        assert opened == [str(pose_viewer)]
        dialog.rgb_button.click()
        dialog.pose_button.click()
        assert opened == [str(pose_viewer), str(viewer), str(pose_viewer)]
        assert 'joint FEMTO/RGB' in dialog.log.toPlainText()
        assert 'Pose warnings' in dialog.log.toPlainText()
        (dense / 'summary.json').write_text('{"status":"different"}')
        with pytest.raises(ValueError, match='bound to the selected RGB'):
            inspect_joint_pick_output(root)
        (dense / 'summary.json').write_text('{"status":"complete"}')
        published.pop('pose_source')
        (root / 'pick_scene_poses.json').write_text(json.dumps(published))
        with pytest.raises(ValueError, match='not from joint FEMTO/RGB'):
            inspect_joint_pick_output(root)
    finally:
        dialog.close()


def test_existing_capture_selection_is_resolved_before_process_start(app, tmp_path, monkeypatch):
    dialog = FieldScanDialog(ROOT)
    start = Mock()
    monkeypatch.setattr(dialog.process, 'start', start)
    try:
        dialog.mode.setCurrentIndex(1)
        dialog.output.setText(str(tmp_path / 'retry'))
        dialog.source.setText(str(tmp_path))
        dialog.start()
        start.assert_not_called()
        assert 'select a capture containing field_scan.json' in dialog.log.toPlainText()
        merged = tmp_path / 'merged'
        merged.mkdir()
        (merged / 'field_scan.json').write_text('{}')
        (merged / 'pose_identification').mkdir()
        for selected in (tmp_path, merged / 'pose_identification'):
            dialog.source.setText(str(selected))
            args = dialog.arguments()
            assert args[3] == str(merged)
            assert dialog.source.text() == str(merged)
    finally:
        dialog.close()


def test_stop_waits_for_child_cleanup_and_never_enables_planning(app, tmp_path, monkeypatch):
    dialog = FieldScanDialog(ROOT)
    dialog.output.setText(str(tmp_path / 'new-field'))
    marker = tmp_path / 'cleanup'
    real_start = dialog.process.start
    code = (
        'import signal,time,sys\nfrom pathlib import Path\n'
        'def stop(*args):\n'
        f' Path({str(marker)!r}).write_text("cleaned")\n sys.exit(130)\n'
        'signal.signal(signal.SIGINT,stop)\nprint("READY",flush=True)\n'
        'while True: time.sleep(.02)\n'
    )
    monkeypatch.setattr(dialog.process, 'start', lambda *args: real_start(sys.executable, ['-u', '-c', code]))
    try:
        dialog.start()
        wait_for(lambda: 'READY' in dialog.log.toPlainText())
        dialog.reject()
        wait_for(lambda: not dialog.running)
        assert marker.read_text() == 'cleaned'
        assert not dialog.plan_button.isEnabled()
    finally:
        dialog.close()


def test_console_field_action_passes_only_identified_ids_to_new_plan(app, tmp_path, monkeypatch):
    import stacking_tabletop.gui.operations.window as windows
    class View(QWidget):
        def __init__(self, parent, *args): super().__init__(parent)
        def show_empty(self): pass
        def render_frame(self, *args, **kwargs): pass
    monkeypatch.setattr(windows, 'OperationsSceneView', View)
    window = windows.StackingOperationsWindow(ROOT)
    try:
        window._field_scan()
        dialog = window._field_dialog
        assert dialog is not None and not dialog.running
        window.execution_view.worker = SimpleNamespace(isRunning=lambda: True)
        start = Mock()
        monkeypatch.setattr(dialog.process, 'start', start)
        dialog.start()
        start.assert_not_called()
        assert 'finish active execution' in dialog.log.toPlainText()
        window.execution_view.worker = None
        pose_path = tmp_path / 'pose_data.pkl'
        pose_path.write_bytes(b'fixture')
        report = tmp_path / 'pose_identification'
        report.mkdir()
        (report / 'stone_poses.json').write_text(json.dumps({'poses':[{'stone_id': 7}, {'stone_id': 12}]}))
        dialog.poses_available.emit(pose_path)
        assert window._planning_dialog.pick_path.text() == str(pose_path)
        assert window._planning_dialog.object_ids.text() == '7 12'
        assert not window._planning_dialog.running
    finally:
        window.execution_view.worker = None
        window.close()
