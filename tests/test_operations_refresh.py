"""Refresh controls use real Qt widgets with the renderer stubbed out."""

import json
import os
import sqlite3
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyvistaqt")
from PySide6.QtWidgets import QApplication, QWidget
from stacking_common.artifacts import StackRunStore, decode_legacy_mcts_debug
from stacking_tabletop.gui.operations import window as windows

ROOT = Path(__file__).resolve().parents[1]


def test_tabletop_refresh_button_action_and_busy_guards(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    artifact = decode_legacy_mcts_debug(json.loads(
        (ROOT / "tests/fixtures/legacy_visualization/mcts_candidate_viewer.json").read_text()))

    class View(QWidget):
        def __init__(self, parent, *args):
            super().__init__(parent)
            self.resets = []

        def render_frame(self, frame, **kwargs):
            self.resets.append(kwargs["reset_camera"])

        def show_empty(self):
            pass

    monkeypatch.setattr(windows, "OperationsSceneView", View)
    window = windows.StackingOperationsWindow(ROOT)
    errors = Mock()
    monkeypatch.setattr(windows.QMessageBox, "critical", errors)
    path = tmp_path / "planning.stack"
    try:
        assert not window.refresh_action.isEnabled()
        assert not window.search_inspector.refresh_button.isEnabled()
        assert window.refresh_action.shortcut().toString() == "Ctrl+R"
        with StackRunStore.create_mcts(path, replace(artifact, steps=())) as writer:
            writer.append_mcts_step(artifact.steps[0])
            window.open_artifact(path)
            assert window.refresh_action.isEnabled()
            assert window.search_inspector.refresh_button.isEnabled()
            writer.append_mcts_step(artifact.steps[1])
            window._video_exporting = True
            window.refresh_action.trigger()
            assert len(window.controller.artifact.steps) == 1
            window._video_exporting = False
            window.execution_view.worker = Mock()
            window.execution_view.worker.isRunning.return_value = True
            window.search_inspector.refresh_button.click()
            assert len(window.controller.artifact.steps) == 1
            window.execution_view.worker = None
            window.search_inspector.refresh_button.click()
            assert window.search_inspector.step_selector.count() == 2
            assert window.scene_view.resets[-1] is False
            assert window.tabs.currentWidget() is window.search_inspector
            assert "Refreshed" in window.statusBar().currentMessage()
            loaded = window.controller.artifact
            monkeypatch.setattr(window.controller, "reload", Mock(side_effect=sqlite3.OperationalError("locked")))
            window.refresh_action.trigger()
            errors.assert_called_once()
            assert window.controller.artifact is loaded
            assert window._artifact_path == path
    finally:
        window._video_exporting = False
        window.execution_view.worker = None
        window.close()
        app.processEvents()
