"""Copying live planning logs must not let subsequent output replace them."""

import importlib
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QByteArray, QTimer, Qt
from PySide6.QtGui import QAction, QTextCursor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QFileSystemModel, QPushButton

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["tabletop", "excavator"])
def dialog(request):
    app = QApplication.instance() or QApplication([])
    module = importlib.import_module(f"stacking_{request.param}.gui.operations.planning_run")
    widget = module.PlanningRunDialog(ROOT)
    widget.show()
    yield widget
    widget.close()
    app.processEvents()


def test_copy_selection_survives_new_output_and_empty_reads(dialog, monkeypatch):
    initial = "Search started\nCandidate 1: reward=2\n"
    dialog.log.setPlainText(initial)
    dialog.log.setFocus()
    QTest.keyClick(dialog.log, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClick(dialog.log, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    assert QApplication.clipboard().text() == initial
    selected = dialog.log.textCursor().selectedText()
    for chunk in (b"Candidate 2: ", b"reward=3\n", b""):
        monkeypatch.setattr(dialog.process, "readAllStandardOutput", lambda: QByteArray(chunk))
        dialog._read_output()
        assert dialog.log.textCursor().selectedText() == selected
    assert dialog.log.toPlainText() == initial + "Candidate 2: reward=3\n"
    QTest.keyClick(dialog.log, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    assert QApplication.clipboard().text() == initial


def test_new_output_appends_even_when_cursor_and_scroll_are_at_start(dialog, monkeypatch):
    initial = "".join(f"Iteration {index}\n" for index in range(100))
    dialog.log.setPlainText(initial)
    cursor = dialog.log.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.Start)
    dialog.log.setTextCursor(cursor)
    scrollbar = dialog.log.verticalScrollBar()
    scrollbar.setValue(0)
    monkeypatch.setattr(dialog.process, "readAllStandardOutput", lambda: QByteArray(b"New iteration\n"))
    dialog._read_output()
    assert dialog.log.toPlainText() == initial + "New iteration\n"
    assert scrollbar.value() == 0
    scrollbar.setValue(scrollbar.maximum())
    dialog._read_output()
    assert scrollbar.value() == scrollbar.maximum()


def test_save_browse_exposes_file_management_without_deleting_anything(dialog, tmp_path):
    existing = tmp_path / "existing.stack"
    existing.write_text("keep this file")
    directory = tmp_path / "existing directory"
    directory.mkdir()
    dialog.output_path.setText(str(tmp_path / "new.stack"))
    observed = {}

    def inspect_and_cancel():
        picker = QApplication.activeModalWidget()
        if not isinstance(picker, QFileDialog):
            observed["wrong_dialog"] = type(picker).__name__
            if picker is not None:
                picker.close()
            return
        try:
            observed["widget_dialog"] = picker.testOption(QFileDialog.Option.DontUseNativeDialog)
            observed["readonly"] = picker.testOption(QFileDialog.Option.ReadOnly)
            observed["model_readonly"] = picker.findChild(QFileSystemModel).isReadOnly()
            observed["actions"] = [action.text().replace("&", "") for action in picker.findChildren(QAction)]
            observed["directory"] = picker.directory().absolutePath()
        finally:
            picker.reject()

    QTimer.singleShot(50, inspect_and_cancel)
    button = dialog.output_path.parentWidget().findChild(QPushButton)
    button.click()
    assert observed["widget_dialog"]
    assert not observed["readonly"] and not observed["model_readonly"]
    assert {"Rename", "Delete", "New Folder"}.issubset(observed["actions"])
    assert Path(observed["directory"]) == tmp_path
    assert existing.read_text() == "keep this file" and directory.is_dir()
    assert dialog.output_path.text() == str(tmp_path / "new.stack")
