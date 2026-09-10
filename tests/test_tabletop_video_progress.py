"""Exercise export progress with Qt widgets and a stubbed viewport/encoder."""

import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyvistaqt")
imageio = pytest.importorskip("imageio.v2")

from PySide6.QtWidgets import QApplication, QMainWindow, QProgressDialog

from stacking_tabletop.gui.operations import window as window_module
from stacking_tabletop.gui.operations.controller import OperationsController
from stacking_tabletop.gui.operations.search_inspector import SearchInspector


@pytest.mark.parametrize("outcome", ["success", "cancel", "error"])
@pytest.mark.parametrize("timed", [False, True])
def test_video_progress_restores_viewer_and_preserves_destination(
    tmp_path, monkeypatch, outcome, timed
):
    app = QApplication.instance() or QApplication([])
    fixture = (
        Path(__file__).parent / "fixtures/legacy_visualization/mcts_candidate_viewer.json"
    )
    destination = tmp_path / "trajectory.mp4"
    destination.write_bytes(b"existing video")

    class Window(window_module.StackingOperationsWindow):
        def __init__(self):
            QMainWindow.__init__(self)
            self._video_exporting = False
            self.controller = OperationsController()
            self.controller.load(fixture)
            if timed:
                artifact = self.controller.artifact
                step = artifact.steps[0]
                candidate = self.controller.active_candidate
                tracks = tuple(replace(track, times=np.linspace(0, 0.2, len(track.poses)))
                               for track in candidate.timeline.tracks)
                updated = replace(candidate, timeline=replace(candidate.timeline, tracks=tracks))
                step = replace(step, candidates=tuple(
                    updated if value is candidate else value for value in step.candidates))
                self.controller.set_artifact(replace(artifact, steps=(step,)))
            self.controller.set_timeline_frame(1)
            self.search_inspector = SearchInspector(self.controller, parent=self)
            self.scene_view = Mock()
            self.scene_view.screenshot.return_value = np.zeros((12, 16, 3), np.uint8)

        def _refresh_scene(self, *_args):
            pass

    dialogs = []

    class Progress(QProgressDialog):
        def __init__(self, *args):
            super().__init__(*args)
            self.values = []
            self.labels = []
            dialogs.append(self)

        def setValue(self, value):
            self.values.append(value)
            super().setValue(value)
            if outcome == "cancel" and value == 1:
                self.cancel()

        def setLabelText(self, text):
            self.labels.append(text)
            super().setLabelText(text)

    monkeypatch.setattr(window_module, "QProgressDialog", Progress)
    monkeypatch.setattr(
        window_module.QFileDialog, "getSaveFileName",
        lambda *args: (str(destination), "MP4 video (*.mp4)"),
    )
    errors = Mock()
    monkeypatch.setattr(window_module.QMessageBox, "critical", errors)
    window = Window()
    original_mode = window.controller.pose_mode
    assert ("duration is preserved" in window.search_inspector.video_fps.toolTip()) == timed
    writer = Mock()

    def get_writer(path, **kwargs):
        assert kwargs["fps"] == 30
        assert dialogs[0].isVisible()
        assert window._video_exporting
        assert not window.search_inspector.video_button.isEnabled()
        Path(path).write_bytes(b"new video")
        return writer

    def append_data(image):
        # Processing progress events must not allow a nested export.
        window._save_trajectory_video(30)
        if outcome == "error":
            raise RuntimeError("encoding failed")

    writer.append_data.side_effect = append_data
    monkeypatch.setattr(imageio, "get_writer", get_writer)
    try:
        window._save_trajectory_video(30)
        assert len(dialogs) == 1
        assert not dialogs[0].isVisible()
        assert not window._video_exporting
        assert window.search_inspector.video_button.isEnabled()
        assert window.controller.pose_mode is original_mode
        assert window.controller.timeline_frame == 1
        window.scene_view.setUpdatesEnabled.assert_called_with(True)
        assert list(tmp_path.iterdir()) == [destination]
        writer.close.assert_called_once()
        if outcome == "success":
            count = 7 if timed else 3
            assert dialogs[0].values == list(range(count + 2))
            assert writer.append_data.call_count == count
            frames = [call.args[0] for call in window.scene_view.render_frame.call_args_list]
            if timed:
                expected = window.controller.active_candidate.timeline.poses_at_time(0.2)
                for overlay, pose in zip(frames[-1].overlays, expected):
                    np.testing.assert_allclose(overlay.pose, pose.pose)
            assert dialogs[0].labels[-1] == "Finishing MP4…"
            assert destination.read_bytes() == b"new video"
            errors.assert_not_called()
        else:
            assert destination.read_bytes() == b"existing video"
            assert dialogs[0].values[-1] < 4
            if outcome == "error":
                errors.assert_called_once()
            else:
                errors.assert_not_called()
                assert "cancelled" in window.statusBar().currentMessage()
    finally:
        window.close()
        app.processEvents()
