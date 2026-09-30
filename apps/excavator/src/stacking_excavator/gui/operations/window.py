"""Excavator desktop console for offline planning, search, and motion replay."""

from pathlib import Path
import os
import sqlite3
import tempfile

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QComboBox, QLabel, QFileDialog, QMainWindow, QMessageBox, QProgressDialog,
    QSplitter, QTabWidget,
)

from stacking_common.artifacts import StackRunStore, export_run_summary
from stacking_common.profiles.inventory import load_inventory_profile
from stacking_common.profiles.visualization import load_visualization_profile
from stacking_common.visualization import CandidatePoseMode

from .controller import OperationsController
from .plan_view import PlanView
from .planning_run import PlanningRunDialog
from .scene_view import OperationsSceneView
from .search_inspector import SearchInspector
from .session import SessionConsole
from stacking_excavator.planning.resume import load_resume_checkpoint


class ExcavatorOperationsWindow(QMainWindow):
    def __init__(self, repository, artifact_path=None, session_path=None):
        super().__init__()
        self.repository = Path(repository).resolve()
        self._artifact_path = None
        self._planning_dialog = None
        self._video_exporting = False
        self.controller = OperationsController(self.repository)
        self.setWindowTitle("Excavator Stacking Operations Console — Offline")
        self.setMinimumSize(900, 600)
        self.resize(1500, 900)
        config = self.repository / "apps/excavator/config"
        inventory = load_inventory_profile(config / "inventory.yml")
        visualization = load_visualization_profile(config / "visualization.yml")
        self.scene_view = OperationsSceneView(self, self.repository / inventory.model_directory, visualization)
        self.tabs = QTabWidget()
        self.plan_view = PlanView(self.controller)
        self.search_inspector = SearchInspector(self.controller, visualization.layer_visibility)
        self.tabs.addTab(self.plan_view, "Plan")
        self.tabs.addTab(self.search_inspector, "Search Inspector")
        self.session = SessionConsole(self.repository, self)
        self.tabs.addTab(self.session.perception, "Perception")
        self.tabs.addTab(self.session.execution, "Execution")
        self.tabs.addTab(self.session.history, "History")
        self.tabs.setCurrentWidget(self.search_inspector)
        self.search_inspector.open_requested.connect(self._choose_artifact)
        self.search_inspector.refresh_requested.connect(self._reload)
        self.search_inspector.video_requested.connect(self._save_trajectory_video)
        self.search_inspector.state_changed.connect(self._refresh_scene)
        self.plan_view.state_changed.connect(self._refresh_scene)
        self.tabs.currentChanged.connect(self._tab_changed)
        self.session.changed.connect(self._refresh_session_scene)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.scene_view)
        splitter.setSizes([500, 1000])
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        menu = self.menuBar().addMenu("File")
        toolbar = self.addToolBar("Planning")
        for label, slot, shortcut in (
            ("New planning run…", self._new_planning, "Ctrl+N"),
            ("Open stacking run…", self._choose_artifact, "Ctrl+O"),
            ("Resume planning…", self._resume_planning, None),
            ("Refresh run", self._reload, "Ctrl+R"),
            ("Export summary…", self._export_summary, None),
        ):
            action = QAction(label, self)
            action.triggered.connect(slot)
            if shortcut:
                action.setShortcut(shortcut)
            menu.addAction(action)
            toolbar.addAction(action)
            if slot == self._reload:
                self.refresh_action = action
                action.setEnabled(False)
        reset = QAction("Reset camera", self)
        reset.setShortcut("R")
        reset.triggered.connect(self._reset_camera)
        self.menuBar().addMenu("View").addAction(reset)
        rendering = self.addToolBar("Rendering")
        rendering.addWidget(QLabel("Objects: "))
        self.object_geometry_selector = QComboBox()
        self.object_geometry_selector.addItem("DSF surface", "dsf")
        self.object_geometry_selector.addItem("PLY mesh", "ply")
        self.object_geometry_selector.currentIndexChanged.connect(self._object_geometry_changed)
        rendering.addWidget(self.object_geometry_selector)
        self.statusBar().showMessage("Open a recorded run or start offline planning")
        if artifact_path:
            self.open_artifact(artifact_path)
        if session_path:
            self.session.open_session(session_path)
            self.tabs.setCurrentWidget(self.session.perception)

    def _refresh_session_scene(self):
        if self.tabs.currentWidget() in (self.session.perception, self.session.execution):
            key = ("perception", None if self.session.proposal is None else self.session.proposal["sha256"]) if (
                self.tabs.currentWidget() is self.session.perception) else ("execution",)
            if key != getattr(self, "_session_camera_key", None):
                self.controller.request_camera_reset()
                self._session_camera_key = key
            self._refresh_scene()

    def _object_geometry_changed(self, _index):
        self.scene_view.set_object_geometry_mode(self.object_geometry_selector.currentData())
        self._refresh_scene()

    def open_artifact(self, path, *, refresh=False):
        if self._video_exporting:
            return
        self.plan_view._stop_playback()
        self.search_inspector._timeline_render_timer.stop()
        try:
            if refresh:
                self.controller.reload(path)
            else:
                self.controller.load(path)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, sqlite3.Error) as error:
            QMessageBox.critical(self, "Cannot refresh excavator run" if refresh else "Cannot open excavator run", str(error))
            return
        self._artifact_path = Path(path).resolve()
        self.refresh_action.setEnabled(True)
        self.search_inspector.refresh_button.setEnabled(True)
        self.search_inspector.refresh(str(self._artifact_path))
        self.plan_view.refresh()
        if not refresh and self.controller.current_plan is not None:
            self.tabs.setCurrentWidget(self.plan_view)
        self._refresh_scene()
        if refresh:
            self.statusBar().showMessage(
                f"Refreshed {self._artifact_path.name}: {len(self.controller.artifact.steps)} steps, "
                f"{len(self.controller.planning_records)} planning attempts"
            )

    def _choose_artifact(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open excavator run",
            str(self.repository), "Stacking runs (*.stack)")
        if filename:
            self.open_artifact(filename)

    def _reload(self):
        if self._artifact_path:
            if self._video_exporting:
                self.statusBar().showMessage("Finish video export before refreshing the run.")
                return
            self.open_artifact(self._artifact_path, refresh=True)

    def _new_planning(self):
        self._show_planning()

    def _resume_planning(self):
        try:
            if self._artifact_path is None:
                raise ValueError("Open a run with a committed planning checkpoint first")
            checkpoint = load_resume_checkpoint(self._artifact_path)
            if checkpoint.state.terminated:
                raise ValueError("This sequence is already complete")
            if StackRunStore.info(self._artifact_path).status == "running":
                raise ValueError("Stop the source run before resuming it")
            if StackRunStore.load_execution_commands(self._artifact_path):
                raise ValueError("A run with execution records requires physical reconciliation")
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(self, "Cannot resume planning", str(error))
            return
        self._show_planning(self._artifact_path)

    def _show_planning(self, source=None):
        if self._planning_dialog is not None and self._planning_dialog.running:
            self._planning_dialog.showNormal()
            self._planning_dialog.raise_()
            return
        if self._planning_dialog is not None:
            self._planning_dialog.close()
            self._planning_dialog.deleteLater()
        self._planning_dialog = PlanningRunDialog(self.repository, source, self)
        self._planning_dialog.run_available.connect(self._open_planning_run)
        self._planning_dialog.show()

    def _open_planning_run(self, path):
        self.open_artifact(path, refresh=Path(path).resolve() == self._artifact_path)

    def _export_summary(self):
        if self._artifact_path is None:
            return
        filename, selected = QFileDialog.getSaveFileName(self, "Export run summary",
            str(self._artifact_path.with_suffix(".json")), "JSON (*.json);;CSV (*.csv)")
        if filename:
            path = Path(filename)
            if not path.suffix:
                path = path.with_suffix(".csv" if selected.startswith("CSV") else ".json")
            try:
                export_run_summary(self._artifact_path, path)
            except (OSError, ValueError) as error:
                QMessageBox.critical(self, "Cannot export summary", str(error))

    def _refresh_scene(self, *_args):
        if self._video_exporting:
            return
        current = self.tabs.currentWidget()
        if current in (self.session.perception, self.session.execution, self.session.history):
            try:
                if current is self.session.perception:
                    self.scene_view.render_perception(self.session.proposal, self.session.cloud_path,
                        layers={k: v.isChecked() for k, v in self.session.layers.items()},
                        reset_camera=self.controller.consume_camera_reset_request())
                else:
                    frame = self.session.execution_frame()
                    if frame is None:
                        self.scene_view.show_empty()
                    else:
                        self.scene_view.render_frame(frame,
                            reset_camera=self.controller.consume_camera_reset_request(), environment_visibility={})
                self.statusBar().showMessage("Desktop perception and execution simulation · no hardware connection")
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                self.statusBar().showMessage(f"Cannot render session: {error}")
            return
        self.plan_view.refresh()
        step = self.controller.current_step
        if step is None:
            self.scene_view.show_empty()
            self.statusBar().showMessage("No completed search step or planning attempt recorded")
            return
        showing_plan = self.tabs.currentWidget() is self.plan_view and self.controller.current_plan is not None
        try:
            frame = self.controller.planning_frame() if showing_plan else self.controller.frame()
            self.scene_view.render_frame(frame,
                reset_camera=self.controller.consume_camera_reset_request(),
                environment_visibility=self.search_inspector.environment_visibility())
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            self.plan_view._stop_playback()
            self.statusBar().showMessage(f"Cannot render replay: {error}")
            return
        if showing_plan:
            segment = self.controller.current_plan_segment
            details = "No motion trajectory · robot at profile home" if segment is None else (
                f"{segment.name} · sample {self.controller.plan_sample_index + 1}/{self.controller.plan_sample_count}"
                + (" · object track unavailable" if segment.target_poses is None else ""))
            self.statusBar().showMessage(f"Offline replay · step {step.step} · {details}")
        else:
            self.statusBar().showMessage(f"Search · step {step.step} · {self.controller.pose_mode.value}")

    def _reset_camera(self):
        self.controller.request_camera_reset()
        self._refresh_scene()

    def _tab_changed(self, *_args):
        self.plan_view._stop_playback()
        if self.tabs.currentWidget() is self.search_inspector:
            self.search_inspector.refresh()
        self.controller.request_camera_reset()
        self._refresh_scene()

    def closeEvent(self, event):
        if self.session.running:
            self.session.stop_identification()
            self.statusBar().showMessage("Stopping identification; close again when it exits")
            event.ignore()
            return
        if self._planning_dialog is not None and self._planning_dialog.running:
            self._planning_dialog.reject()
            self.statusBar().showMessage("Stopping offline planning; close again when it exits")
            event.ignore()
            return
        if self._video_exporting:
            event.ignore()
            return
        self.plan_view._stop_playback()
        self.session.close()
        self.scene_view.close()
        super().closeEvent(event)

    def _save_trajectory_video(self, fps: float) -> None:
        if self._video_exporting:
            return
        candidate = self.controller.active_candidate
        if candidate is None or candidate.timeline.frame_count == 0:
            QMessageBox.information(
                self,
                "No trajectory",
                "The active candidate has no trajectory frames to export.",
            )
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save trajectory video",
            f"step_{self.controller.current_step.step:02d}_"
            f"rank_{candidate.rank:02d}_object_{candidate.object_id}.mp4",
            "MP4 video (*.mp4)",
        )
        if not filename:
            return
        destination = Path(filename)
        if destination.suffix.lower() != ".mp4":
            destination = destination.with_suffix(".mp4")
        try:
            import imageio.v2 as imageio
        except ImportError:
            QMessageBox.critical(
                self,
                "Video export unavailable",
                "Install the excavator operations extra with imageio-ffmpeg.",
            )
            return

        original_pose_mode = self.controller.pose_mode
        original_frame = self.controller.timeline_frame
        temporary_path: Path | None = None
        writer = None
        saved = False
        cancelled = False
        export_error = None
        video_times = candidate.timeline.video_times(fps)
        frame_count = (candidate.timeline.frame_count if video_times is None
                       else len(video_times))
        environment_visibility = self.search_inspector.environment_visibility()
        progress = QProgressDialog(
            f"Preparing {destination.name}…", "Cancel", 0, frame_count + 1, self
        )
        progress.setWindowTitle("Save trajectory video")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        self._video_exporting = True
        self.search_inspector.video_button.setEnabled(False)
        self.scene_view.setUpdatesEnabled(False)
        self.statusBar().showMessage(f"Generating {destination.name}…")
        try:
            progress.setValue(0)
            progress.show()
            QApplication.processEvents()
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = tempfile.NamedTemporaryFile(
                prefix=f".{destination.stem}.",
                suffix=".mp4",
                dir=destination.parent,
                delete=False,
            )
            temporary_path = Path(temporary.name)
            temporary.close()
            writer = imageio.get_writer(
                temporary_path,
                fps=float(fps),
                codec="libx264",
                macro_block_size=2,
            )
            self.controller.set_pose_mode(CandidatePoseMode.TRAJECTORY)
            for frame_index in range(frame_count):
                if progress.wasCanceled():
                    cancelled = True
                    break
                if video_times is None:
                    self.controller.set_timeline_frame(frame_index)
                frame = self.controller.frame(
                    trajectory_time=None if video_times is None else float(video_times[frame_index])
                )
                self.scene_view.render_frame(
                    frame,
                    reset_camera=False,
                    environment_visibility=environment_visibility,
                )
                image = self.scene_view.screenshot(return_img=True)
                if image is None:
                    raise RuntimeError("the viewport did not return a video frame")
                height = int(image.shape[0]) - int(image.shape[0]) % 2
                width = int(image.shape[1]) - int(image.shape[1]) % 2
                writer.append_data(image[:height, :width, :3])
                progress.setLabelText(
                    f"Rendering frames: {frame_index + 1} / {frame_count}"
                )
                progress.setValue(frame_index + 1)
                QApplication.processEvents()
            cancelled = cancelled or progress.wasCanceled()
            if not cancelled:
                progress.setLabelText("Finishing MP4…")
                progress.setCancelButton(None)
                QApplication.processEvents()
            writer.close()
            writer = None
            if not cancelled:
                os.replace(temporary_path, destination)
                temporary_path = None
                saved = True
                progress.setValue(frame_count + 1)
        except Exception as error:
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
            export_error = str(error)
        finally:
            progress.close()
            progress.deleteLater()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            self.controller.set_pose_mode(original_pose_mode)
            if self.controller.timeline_frame_count > 0:
                self.controller.set_timeline_frame(
                    min(original_frame, self.controller.timeline_frame_count - 1)
                )
            self.scene_view.setUpdatesEnabled(True)
            self._video_exporting = False
            self.search_inspector.refresh()
            self._refresh_scene()
        if saved:
            self.statusBar().showMessage(f"Saved {destination}")
        elif export_error is not None:
            QMessageBox.critical(self, "Video export failed", export_error)
        elif cancelled:
            self.statusBar().showMessage("Trajectory video export cancelled")
