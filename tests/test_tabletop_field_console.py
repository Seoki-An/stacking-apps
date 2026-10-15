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
from PySide6.QtCore import QProcess, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget
from stacking_tabletop.gui.operations.field_scan import FieldScanDialog, inspect_joint_pick_output, retained_pick_viewers

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
        assert args[args.index('--dense-backend') + 1] == 'fast-foundation-stereo'
        assert args[args.index('--viewpoint-pattern') + 1] == 'coverage9'
        dialog.viewpoints.setCurrentIndex(dialog.viewpoints.findData('coverage13'))
        assert '0.20 m' in dialog.note.text()
        dialog.viewpoints.setCurrentIndex(1)
        compact = dialog.arguments()
        assert compact[compact.index('--viewpoint-pattern') + 1] == 'coverage9'
        assert '0.08 m' in dialog.note.text()
        dialog.viewpoints.setCurrentIndex(dialog.viewpoints.findData('coverage17'))
        extended = dialog.arguments()
        assert extended[extended.index('--viewpoint-pattern') + 1] == 'coverage17'
        assert '0.35 m' in dialog.note.text()
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
        assert '--viewpoint-pattern' not in args and not dialog.viewpoints.isEnabled()
        assert args[args.index('--dense-backend') + 1] == 'fast-foundation-stereo'
        assert args[args.index('--pose-backend') + 1] == 'foundationpose'
        assert not dialog.manual.isEnabled()
        dialog.pose_backend.setCurrentIndex(dialog.pose_backend.findData('icp'))
        dialog.dense_backend.setCurrentIndex(dialog.dense_backend.findData('colmap'))
        args = dialog.arguments()
        assert args[args.index('--dense-backend') + 1] == 'colmap'
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
        dialog.pose_backend.setCurrentIndex(dialog.pose_backend.findData('icp'))
        dialog.manual.setChecked(True)
        args = dialog.arguments()
        assert args[2:4] == ['stacking_tabletop.acquisition.pick_scene_workflow', 'resume']
        assert str(failed) in args and '--manual-init-gui' in args
        assert '--output' not in args and '--camera' not in args
        assert not dialog.output.isEnabled()
        assert '--dense-backend' not in args and not dialog.dense_backend.isEnabled()
        assert 'No camera capture or MVS reconstruction is repeated.' in dialog.note.text()
    finally:
        dialog.close()


@pytest.mark.parametrize('exit_code', [0, 1])
def test_capture_region_preview_is_plan_only_and_preserves_capture_output(app, tmp_path, monkeypatch, exit_code):
    dialog = FieldScanDialog(ROOT)
    dialog.mode.setCurrentIndex(dialog.mode.findData('pick_capture'))
    dialog.output.setText(str(tmp_path/'capture'))
    # Keep the command's generated preview directory within this test.
    dialog.repository = tmp_path
    calls, opened = [], []
    monkeypatch.setattr(dialog.process, 'start', lambda executable, args: calls.append(args))
    monkeypatch.setattr('stacking_tabletop.gui.operations.field_scan.QDesktopServices.openUrl',
                        lambda url: opened.append(url.toLocalFile()) or True)
    try:
        dialog.preview_coverage()
        args = calls[0]
        assert args[2:4] == ['stacking_tabletop.acquisition.scene_capture_cli', 'plan']
        assert args[args.index('--viewpoint-pattern')+1] == 'coverage9'
        directory = Path(args[args.index('--output')+1])
        directory.mkdir(parents=True)
        (directory/'coverage.html').write_text('fixture')
        dialog._finished(exit_code, QProcess.ExitStatus.NormalExit)
        assert opened == [str(directory/'coverage.html')]
        assert dialog.output.text() == str(tmp_path/'capture')
        assert not (tmp_path/'capture').exists()
        assert dialog.start_button.isEnabled() and dialog.inputs.isEnabled()
        assert dialog._coverage_preview_directory is None
        assert dialog.pose_path is None
        assert not (directory/'plan.json').exists()
        if exit_code:
            assert 'desired camera coverage only' in dialog.log.toPlainText()
    finally:
        dialog.close()


@pytest.mark.parametrize('has_pose_viewer', [False, True])
def test_failed_pick_keeps_diagnostic_viewers_but_disables_planning(app, tmp_path, monkeypatch, has_pose_viewer):
    root = tmp_path/'failed'
    dense = root/'rgb_dense'
    (dense/'rgb_only').mkdir(parents=True)
    (dense/'summary.json').write_text(json.dumps({'status': 'complete'}))
    rgb = dense/'rgb_only/overview.html'
    rgb.write_text('reconstructed RGB')
    (root/'pick_scene_workflow.json').write_text(json.dumps(
        {'status': 'failed', 'stage': 'joint_pose_identification'}))
    pose = root/'fused_identification/comparison/overview.html'
    if has_pose_viewer:
        pose.parent.mkdir(parents=True)
        pose.write_text('rejected pose diagnostics')
    dialog = FieldScanDialog(ROOT)
    opened = []
    monkeypatch.setattr('stacking_tabletop.gui.operations.field_scan.QDesktopServices.openUrl',
                        lambda url: opened.append(url.toLocalFile()) or True)
    try:
        dialog.mode.setCurrentIndex(dialog.mode.findData('pick_capture'))
        dialog.output.setText(str(root))
        dialog._finished(1, QProcess.ExitStatus.NormalExit)
        assert dialog.rgb_button.isEnabled()
        assert dialog.pose_button.isEnabled() == has_pose_viewer
        assert dialog.pose_path is None and not dialog.plan_button.isEnabled()
        assert opened == [str(pose if has_pose_viewer else rgb)]
        # A linked external folder must not be offered as this run's output.
        (root/'pick_scene_workflow.json').write_text(json.dumps({'dense_dir': str(tmp_path/'outside')}))
        assert retained_pick_viewers(root)[0] is None
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


@pytest.mark.parametrize('backend', ['icp', 'foundationpose'])
def test_pick_workbench_uses_only_rgb_bound_joint_poses_and_opens_rgb_cloud(app, tmp_path, monkeypatch, backend):
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
    report = {'status': 'complete', 'pose_source': ('foundationpose_rgb_femto_validated' if backend == 'foundationpose' else 'rgb_femto_first_joint_icp'),
              'kind': 'femto_rgb_joint_pose_identification', 'icp_refinement_applied': False,
              'source_sha256': {'dense_summary.json': hashlib.sha256(
                  (dense / 'summary.json').read_bytes()).hexdigest(),
                  'femto_geometry.json': hashlib.sha256(
                  (geometry / 'geometry.json').read_bytes()).hexdigest()},
              'poses': [{'stone_id': 9}]}
    (report_dir / 'stone_poses.json').write_text(json.dumps(report))
    (root / 'pick_scene_workflow.json').write_text(json.dumps({
        'pose_backend': backend, 'schema_version': 2, 'status': 'complete', 'production_pose_accepted': True,
        'pose_sequence': 'femto_regions_rgb_mvs_joint_catalog_fit',
        'dense_dir': str(dense), 'fused_pose_data': str(fused / 'pose_data.pkl'),
        'pose_viewer': str(pose_viewer)}))
    published = {'status': 'complete', 'production_pose_accepted': True,
                 'pose_source': ('foundationpose_rgb_femto_validated' if backend == 'foundationpose' else 'rgb_femto_first_joint_icp'),
                 'preliminary_femto_identification': None,
                 'coarse_femto_geometry': 'femto_geometry/geometry.json',
                 'pose_viewer': 'fused_identification/comparison/overview.html',
                 'pose_data': str(root / 'pose_data.pkl'),
                 'joint_identification': str(report_dir / 'stone_poses.json'),
                 'pose_warnings': [{'stone_id': 9}]}
    (root / 'pick_scene_poses.json').write_text(json.dumps(published))

    dialog = FieldScanDialog(ROOT)
    review = Mock(return_value=False)
    monkeypatch.setattr(dialog, '_ask_model_id_review', review)
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
        review.assert_called_once_with()
        assert dialog.mode.findData('pick_reidentify') == -1
        started = Mock()
        monkeypatch.setattr(dialog.process, 'start', started)
        review.return_value = True
        dialog.plan_button.setEnabled(False)
        dialog._finished(0, QProcess.ExitStatus.NormalExit)
        assert review.call_count == 2
        assert dialog.mode.currentData() == 'pick_existing'
        args = started.call_args.args[1]
        assert args[2:4] == ['stacking_tabletop.acquisition.pick_scene_workflow', 'reidentify']
        assert args[4] == str(root)
        assert Path(args[args.index('--output') + 1]).parent == root.parent
        assert args[args.index('--output') + 1] != str(root)
        assert '--camera' not in args
        assert not dialog.plan_button.isEnabled()
        assert dialog._result_directory == Path(args[args.index("--output") + 1])
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


@pytest.mark.parametrize('choice', ['Accept poses', 'Change model IDs'])
def test_post_identification_prompt_offers_accept_or_return_to_ids(app, choice):
    dialog = FieldScanDialog(ROOT)
    observed = []

    def choose():
        prompt = QApplication.activeModalWidget()
        assert isinstance(prompt, QMessageBox)
        observed.extend(button.text() for button in prompt.buttons())
        next(button for button in prompt.buttons() if button.text() == choice).click()

    try:
        QTimer.singleShot(0, choose)
        assert dialog._ask_model_id_review() == (choice == 'Change model IDs')
        assert set(observed) == {'Accept poses', 'Change model IDs'}
    finally:
        dialog.close()


def test_reidentification_after_resume_keeps_new_result_directory(app, tmp_path, monkeypatch):
    previous = tmp_path / 'previous'
    previous.mkdir()
    (previous / 'pick_scene_workflow.json').write_text('{}')
    dialog = FieldScanDialog(ROOT)
    started = Mock()
    monkeypatch.setattr(dialog.process, 'start', started)
    try:
        dialog.mode.setCurrentIndex(dialog.mode.findData('pick_resume'))
        dialog.source.setText(str(previous))
        dialog._completed_workflow = previous
        dialog._change_model_ids()
        args = started.call_args.args[1]
        assert args[3] == 'reidentify' and args[4] == str(previous)
        assert dialog._result_directory != previous
        assert dialog._result_directory == Path(args[args.index('--output') + 1])
        assert dialog.mode.currentData() == 'pick_resume'
    finally:
        dialog.close()


def test_wrist_only_gui_uses_no_femto_and_supports_saved_capture(app, tmp_path):
    dialog = FieldScanDialog(ROOT)
    try:
        dialog.camera.setCurrentIndex(dialog.camera.findData('both'))
        dialog.mode.setCurrentIndex(dialog.mode.findData('pick_wrist_capture'))
        dialog.output.setText(str(tmp_path/'new'))
        args = dialog.arguments()
        assert args[args.index('--sensor-mode')+1] == 'wrist-only'
        assert '--camera' not in args
        assert not dialog.camera.isEnabled()
        assert dialog.viewpoints.isEnabled() and not dialog.coverage_button.isHidden()
        assert args[args.index('--dense-backend')+1] == 'fast-foundation-stereo'
        assert args[args.index('--pose-backend')+1] == 'foundationpose'
        dialog.mode.setCurrentIndex(dialog.mode.findData('pick_wrist_existing'))
        dialog.source.setText(str(tmp_path))
        args = dialog.arguments()
        assert 'process' in args and str(tmp_path) in args
        assert '--viewpoint-pattern' not in args
    finally:
        dialog.close()
