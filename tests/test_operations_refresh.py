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
from PySide6.QtWidgets import QApplication, QMenu, QToolBar, QWidget
from stacking_common.artifacts import StackRunStore, decode_legacy_mcts_debug
from stacking_tabletop.gui.operations import window as windows

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("recorded_environment", [False, True])
def test_tabletop_refresh_button_action_and_busy_guards(tmp_path, monkeypatch, recorded_environment):
    app = QApplication.instance() or QApplication([])
    artifact = decode_legacy_mcts_debug(json.loads(
        (ROOT / "tests/fixtures/legacy_visualization/mcts_candidate_viewer.json").read_text()))

    class View(QWidget):
        def __init__(self, parent, *args):
            super().__init__(parent)
            self.resets = []

        def render_frame(self, frame, **kwargs):
            self.frame = frame
            self.resets.append(kwargs["reset_camera"])

        def show_empty(self):
            pass

    monkeypatch.setattr(windows, "OperationsSceneView", View)
    resume = Mock()
    monkeypatch.setattr(windows.StackingOperationsWindow, "_resume_planning", resume)
    window = windows.StackingOperationsWindow(ROOT)
    if recorded_environment:
        from types import SimpleNamespace
        from stacking_common.artifacts.planning_context import _values
        planning = window._default_projector.planning
        recorded = replace(planning, world_T_stack=replace(
            planning.world_T_stack, parent_T_child=[.2, .3, .4, 0., 0., 0., 1.]))
        contexts = Mock(return_value=(SimpleNamespace(profiles={"planning_profile": _values(recorded)}),))
        monkeypatch.setattr(StackRunStore, "load_run_contexts", contexts)
    errors = Mock()
    monkeypatch.setattr(windows.QMessageBox, "critical", errors)
    path = tmp_path / "planning.stack"
    try:
        planning = next(toolbar for toolbar in window.findChildren(QToolBar)
                        if toolbar.windowTitle() == "Planning")
        assert [action.text() for action in planning.actions()] == [
            "New planning run…", "Open stacking run…", "Resume planning…",
            "Refresh run", "Export summary…",
        ]
        resume_action = planning.actions()[2]
        file_menu = next(menu for menu in window.findChildren(QMenu) if menu.title() == "File")
        assert resume_action in file_menu.actions()
        planning.widgetForAction(resume_action).click()
        resume.assert_called_once()
        assert not window.refresh_action.isEnabled()
        assert not window.search_inspector.refresh_button.isEnabled()
        assert window.refresh_action.shortcut().toString() == "Ctrl+R"
        with StackRunStore.create_mcts(path, replace(artifact, steps=())) as writer:
            writer.append_mcts_step(artifact.steps[0])
            window.open_artifact(path)
            pickup = next(box for box in window.scene_view.frame.environment_boxes
                          if box.component_id == "pick_table")
            if recorded_environment:
                assert pickup.pose == pytest.approx((-.2, -.3, -.4, 0., 0., 0., 1.))
            else:
                assert pickup.pose == pytest.approx((.55, 0., -.115, 0., 0., 0., 1.))
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
