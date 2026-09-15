"""Qt process supervision and planning handoff for the field workflow."""
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
from stacking_tabletop.gui.operations.field_scan import FieldScanDialog

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
