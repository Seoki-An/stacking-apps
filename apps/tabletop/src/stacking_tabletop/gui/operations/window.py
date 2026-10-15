"""Main window for tabletop planning, search inspection, and execution."""

from pathlib import Path
from dataclasses import replace
import os
import sqlite3
import tempfile

from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSplitter,
    QTabWidget,
)
from PySide6.QtCore import Qt

from stacking_common.visualization import CandidatePoseMode
from stacking_common.artifacts import StackRunStore, export_run_summary
from stacking_tabletop.config import (
    load_inventory_profile,
    load_planning_profile,
    load_visualization_profile,
)
from stacking_tabletop.planning import TabletopRobotTrajectoryProjector

from .controller import OperationsController
from .execution_view import ExecutionView
from .plan_view import PlanView
from .scene_view import OperationsSceneView
from .search_inspector import SearchInspector
from .environment import planning_scene_boxes, search_scene_boxes


class StackingOperationsWindow(QMainWindow):
    """Unified tabletop scene for plans, search diagnostics, and execution."""

    def __init__(
        self,
        repository: Path,
        artifact_path: Path | None = None,
    ) -> None:
        super().__init__()
        self._video_exporting = False
        self._artifact_path = None
        self._planning_dialog = None
        self._field_dialog = None
        self.repository = Path(repository)
        planning = load_planning_profile(
            self.repository / "apps/tabletop/config/planning.yml"
        )
        self.controller = OperationsController(
            trajectory_projector=TabletopRobotTrajectoryProjector(
                planning,
                self.repository,
            )
        )
        self.setWindowTitle("Tabletop Stacking Operations Console")
        self._default_projector = self.controller.trajectory_projector
        self._recorded_projectors = {}
        self._search_planning = planning
        self.setMinimumSize(900, 600)
        self.resize(1500, 900)
        self._build_ui()
        if artifact_path is not None:
            self.open_artifact(artifact_path)
        else:
            self.statusBar().showMessage("No stacking run loaded")

    @property
    def execution_running(self):
        view = getattr(self, "execution_view", None)
        return view is not None and view.running

    @property
    def field_running(self):
        dialog = getattr(self, '_field_dialog', None)
        return dialog is not None and dialog.running

    def _build_ui(self) -> None:
        inventory = load_inventory_profile(
            self.repository / "apps/tabletop/config/inventory.yml"
        )
        visualization = load_visualization_profile(
            self.repository / "apps/tabletop/config/visualization.yml"
        )
        self._visualization = visualization
        asset_directory = self.repository / inventory.model_directory
        self.scene_view = OperationsSceneView(
            self,
            asset_directory,
            visualization,
        )
        self.tabs = QTabWidget()
        self.plan_view = PlanView(self.controller)
        self.search_inspector = SearchInspector(
            self.controller,
            visualization.layer_visibility,
        )
        self.execution_view = ExecutionView(self.repository)
        self.execution_view.can_start = lambda: not self.field_running
        self.execution_view.state_changed.connect(self._refresh_scene)
        self.tabs.addTab(self.plan_view, "Plan")
        self.tabs.addTab(self.search_inspector, "Search Inspector")
        self.tabs.addTab(self.execution_view, "Execution")
        self.tabs.setCurrentWidget(self.search_inspector)
        self.search_inspector.open_requested.connect(self._choose_artifact)
        self.search_inspector.refresh_requested.connect(self._reload)
        self.search_inspector.video_requested.connect(self._save_trajectory_video)
        self.search_inspector.state_changed.connect(self._refresh_scene)
        self.plan_view.state_changed.connect(self._refresh_scene)
        self.tabs.currentChanged.connect(self._tab_changed)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.scene_view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 1120])
        self.setCentralWidget(splitter)

        file_menu = self.menuBar().addMenu("File")
        planning_toolbar = self.addToolBar("Planning")
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
            file_menu.addAction(action)
            planning_toolbar.addAction(action)
            if slot == self._reload:
                self.refresh_action = action
                action.setEnabled(False)
            elif slot == self._resume_planning:
                action.setToolTip("Continue from the loaded run's planning checkpoint into a new run")
        view_menu = self.menuBar().addMenu("View")
        reset_action = QAction("Reset camera", self)
        reset_action.setShortcut("R")
        reset_action.triggered.connect(self._reset_camera)
        view_menu.addAction(reset_action)
        self.addToolBarBreak()
        rendering = self.addToolBar("Rendering")
        rendering.addWidget(QLabel("Objects: "))
        self.object_geometry_selector = QComboBox()
        self.object_geometry_selector.addItem("DSF surface", "dsf")
        self.object_geometry_selector.addItem("PLY mesh", "ply")
        self.object_geometry_selector.currentIndexChanged.connect(self._object_geometry_changed)
        rendering.addWidget(self.object_geometry_selector)
        rendering.addWidget(QLabel("Robot / gripper: "))
        self.robot_geometry_selector = QComboBox()
        plan_selector = self.plan_view.robot_geometry_selector
        for index in range(plan_selector.count()):
            self.robot_geometry_selector.addItem(plan_selector.itemText(index), plan_selector.itemData(index))
        self.robot_geometry_selector.currentIndexChanged.connect(plan_selector.setCurrentIndex)
        plan_selector.currentIndexChanged.connect(self.robot_geometry_selector.setCurrentIndex)
        rendering.addWidget(self.robot_geometry_selector)
        acquisition = self.addToolBar("Acquisition")
        field_action = QAction("Field scan / identify…", self)
        field_action.triggered.connect(self._field_scan)
        file_menu.addSeparator()
        file_menu.addAction(field_action)
        acquisition.addAction(field_action)

    def _object_geometry_changed(self, _index):
        self.scene_view.set_object_geometry_mode(self.object_geometry_selector.currentData())
        self._refresh_scene()

    def open_artifact(self, path: str | Path, *, refresh: bool = False) -> None:
        if self._video_exporting or self.execution_running:
            return
        path = Path(path).resolve()
        self.plan_view._stop_playback()
        self.search_inspector._timeline_render_timer.stop()
        try:
            search_planning = self._default_projector.planning
            if path.suffix.lower() == ".stack":
                from stacking_tabletop.execution.saved_plan import recorded_planning_profile
                contexts = StackRunStore.load_run_contexts(path)
                if contexts:
                    search_planning = recorded_planning_profile(contexts[-1])
            if refresh:
                self.controller.reload(path)
            else:
                self.controller.load(path)
        except (OSError, TypeError, ValueError, KeyError, RuntimeError, sqlite3.Error) as error:
            QMessageBox.critical(
                self,
                "Cannot refresh stacking run" if refresh else "Cannot open stacking run",
                str(error),
            )
            return
        self.search_inspector.refresh(str(path))
        self._artifact_path = path
        self._search_planning = search_planning
        self.refresh_action.setEnabled(path.suffix.lower() == ".stack")
        self.search_inspector.refresh_button.setEnabled(self.refresh_action.isEnabled())
        self.plan_view.refresh()
        self.execution_view.load(path)
        self._refresh_scene()
        if refresh:
            self.statusBar().showMessage(
                f"Refreshed {path.name}: {len(self.controller.artifact.steps)} steps, "
                f"{len(self.controller.planning_records)} planning attempts"
            )

    def _reload(self):
        if self._artifact_path is not None and self._artifact_path.suffix.lower() == ".stack":
            if self._video_exporting or self.execution_running:
                self.statusBar().showMessage("Finish video export or execution before refreshing the run.")
                return
            self.open_artifact(self._artifact_path, refresh=True)

    def _export_summary(self):
        if self._artifact_path is None or self._artifact_path.suffix != ".stack":
            QMessageBox.information(self, "Export summary", "Open a .stack run first.")
            return
        filename, selected = QFileDialog.getSaveFileName(self, "Export run summary",
            str(self._artifact_path.with_suffix(".json")), "JSON (*.json);;CSV (*.csv)",
            options=QFileDialog.Option.DontUseNativeDialog)
        if not filename:
            return
        path = Path(filename)
        if not path.suffix:
            path = path.with_suffix(".csv" if selected.startswith("CSV") else ".json")
        try:
            export_run_summary(self._artifact_path, path)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot export summary", str(error))
            return
        self.statusBar().showMessage(f"Summary exported: {path}")

    def _field_scan(self):
        if self._field_dialog is None:
            from .field_scan import FieldScanDialog
            self._field_dialog = FieldScanDialog(self.repository, self)
            self._field_dialog.can_start = lambda: not self.execution_running and not self._video_exporting and not (
                self._planning_dialog is not None and self._planning_dialog.running)
            self._field_dialog.poses_available.connect(self._plan_field_poses)
            self._field_dialog.running_changed.connect(self.execution_view._controls)
        self._field_dialog.show()
        self._field_dialog.raise_()

    def _plan_field_poses(self, path):
        if self.execution_running or self.field_running or self._video_exporting or (
                self._planning_dialog is not None and self._planning_dialog.running):
            self.statusBar().showMessage('Finish the active workflow before creating a plan from field poses.')
            return
        import json
        try:
            report = json.loads((Path(path).parent / 'pose_identification/stone_poses.json').read_text())
            if not Path(path).is_file() or not report['poses']:
                raise ValueError('field pickup pose output is missing or empty')
        except (OSError, ValueError, KeyError) as error:
            self.statusBar().showMessage(f'Cannot load field poses: {error}')
            return
        self._new_planning()
        self._planning_dialog.pick_path.setText(str(path))
        self._planning_dialog.object_ids.setText(' '.join(str(item['stone_id']) for item in report['poses']))

    def _new_planning(self):
        if self.execution_running or self.field_running or self._video_exporting:
            return
        if self._planning_dialog is not None and self._planning_dialog.running:
            self._planning_dialog.showNormal()
            self._planning_dialog.raise_()
            return
        if self._planning_dialog is not None:
            self._planning_dialog.close()
            self._planning_dialog.deleteLater()
        from .planning_run import PlanningRunDialog
        self._planning_dialog = PlanningRunDialog(self.repository, parent=self)
        self._planning_dialog.can_start = lambda: not self.execution_running and not self.field_running and not self._video_exporting
        self._planning_dialog.run_available.connect(self._open_planning_run)
        self._planning_dialog.show()

    def _resume_planning(self):
        if self.execution_running or self.field_running or self._video_exporting:
            return
        if self._planning_dialog is not None and self._planning_dialog.running:
            self._planning_dialog.showNormal()
            self._planning_dialog.raise_()
            return
        try:
            if self._artifact_path is None:
                raise ValueError("Open a checkpoint-bearing .stack run first.")
            checkpoint = StackRunStore.load_planning_checkpoint(self._artifact_path)
            if checkpoint.state.terminated:
                raise ValueError("This planning sequence is already complete.")
            if StackRunStore.load_execution_commands(self._artifact_path):
                raise ValueError("Execution records require physical reconciliation, not offline planning resume.")
            config = StackRunStore.info(self._artifact_path).metadata["config"]["path"]
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.critical(self, "Cannot resume planning", str(error))
            return
        filename, _ = QFileDialog.getSaveFileName(self, "New planning branch",
            str(self._artifact_path.with_name(self._artifact_path.stem + "-resumed.stack")),
            "Stacking runs (*.stack)", options=QFileDialog.Option.DontUseNativeDialog)
        if not filename:
            return
        destination = Path(filename)
        if not destination.suffix:
            destination = destination.with_suffix(".stack")
        from .planning_run import PlanningRunDialog
        if self._planning_dialog is not None:
            self._planning_dialog.close()
            self._planning_dialog.deleteLater()
        self._planning_dialog = PlanningRunDialog(self.repository, self._artifact_path,
                                                  destination, config, self)
        self._planning_dialog.run_available.connect(self._open_planning_run)
        self._planning_dialog.show()

    def _choose_artifact(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open stacking run",
            str(self.repository),
            "Stacking runs (*.stack);;JSON fixtures (*.json);;All files (*)",
        )
        if filename:
            self.open_artifact(filename)

    def _open_planning_run(self, path):
        self.open_artifact(path, refresh=Path(path).resolve() == self._artifact_path)

    def _projector_for_current_plan(self):
        record = self.controller.current_plan
        digest = record.context_sha256 if record is not None else None
        if digest is not None and self._artifact_path is not None:
            key = (self._artifact_path, digest)
            if key not in self._recorded_projectors:
                from stacking_tabletop.execution.saved_plan import recorded_planning_profile
                context = StackRunStore.load_run_context(self._artifact_path, digest)
                planning = recorded_planning_profile(context)
                self._recorded_projectors[key] = TabletopRobotTrajectoryProjector(planning, self.repository)
            projector = self._recorded_projectors[key]
        else:
            projector = self._default_projector
        return projector

    def _search_frame(self, *, trajectory_time=None):
        frame = self.controller.frame(trajectory_time=trajectory_time)
        planning = self._search_planning
        record = self.controller.current_plan
        if record is not None and record.context_sha256 is not None:
            planning = self._projector_for_current_plan().planning
        return replace(frame, environment_boxes=search_scene_boxes(planning, self._visualization))

    def _planning_frame(self):
        if self.controller.trajectory_projector is None:
            return self.controller.planning_frame()
        projector = self._projector_for_current_plan()
        record = self.controller.current_plan
        self.controller.trajectory_projector = projector
        frame = self.controller.planning_frame()
        if record is not None and self.controller.current_plan_segment is None:
            frame = replace(frame, robot_parts=projector.project(
                projector.planning.robot.home_positions,
                projector.planning.robot.gripper_opening,
                geometry_mode=self.controller.robot_geometry_mode))
        return replace(frame,
                       environment_boxes=planning_scene_boxes(projector.planning))

    def _refresh_scene(self, *_args) -> None:
        if self._video_exporting:
            return
        self.plan_view.refresh()
        step = self.controller.current_step
        if step is None:
            self.scene_view.show_empty()
            self.statusBar().showMessage("No stacking run loaded")
            return
        execution = getattr(self, "execution_view", None)
        if execution is not None and self.tabs.currentWidget() is execution and execution.scene is not None:
            self.scene_view.render_frame(execution.scene.frame(geometry_mode=self.controller.robot_geometry_mode),
                reset_camera=self.controller.consume_camera_reset_request(),
                environment_visibility={"support": True, "obstacle": True})
            self.statusBar().showMessage("Execution · measured robot · estimated object pose · translucent blue saved placement")
            return
        showing_plan = (
            self.tabs.currentWidget() is self.plan_view
            and self.controller.current_plan is not None
        )
        self.scene_view.render_frame(
            (
                self._planning_frame()
                if showing_plan
                else self._search_frame()
            ),
            reset_camera=self.controller.consume_camera_reset_request(),
            environment_visibility=(
                self.search_inspector.environment_visibility()
            ),
        )
        if showing_plan:
            plan = self.controller.current_plan
            segment = self.controller.current_plan_segment
            trajectory = ""
            if segment is not None:
                trajectory = (
                    f" · {segment.name} · sample "
                    f"{self.controller.plan_sample_index + 1}/"
                    f"{self.controller.plan_sample_count}"
                )
            else:
                trajectory = " · no motion trajectory · robot at profile home"
            self.statusBar().showMessage(
                f"Step {step.step} · plan attempt {plan.attempt} · "
                f"{plan.result.status.value} · {plan.problem.frame_id} frame"
                f"{trajectory}"
            )
            return
        active = self.controller.active_candidate
        candidate = "none" if active is None else active.candidate_id
        self.statusBar().showMessage(
            f"Step {step.step} · active candidate {candidate} · "
            f"{self.controller.pose_mode.value}"
        )

    def _reset_camera(self) -> None:
        if self._video_exporting or self.controller.current_step is None:
            return
        self.controller.request_camera_reset()
        self._refresh_scene()

    def _save_trajectory_video(self, fps: float) -> None:
        if self._video_exporting or self.execution_running:
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
            options=QFileDialog.Option.DontUseNativeDialog,
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
                "Install the tabletop operations extra with imageio-ffmpeg.",
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
                frame = self._search_frame(
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

    def closeEvent(self, event) -> None:
        if self.field_running:
            self._field_dialog.reject()
            event.ignore()
            self.statusBar().showMessage('Stopping field capture; close again after its camera driver exits.')
            return
        if self.execution_running:
            self.execution_view.abort()
            event.ignore()
            self.statusBar().showMessage("Stopping execution; close again after the worker exits.")
            return
        if getattr(self, "_planning_dialog", None) is not None and self._planning_dialog.running:
            self._planning_dialog.reject()
            event.ignore()
            self.statusBar().showMessage("Stopping offline planning; close again after it exits.")
            return
        if self._video_exporting:
            event.ignore()
            return
        super().closeEvent(event)

    def _tab_changed(self, *_args) -> None:
        if self.tabs.currentWidget() is not self.plan_view:
            self.plan_view._stop_playback()
        if self.tabs.currentWidget() is self.search_inspector:
            self.search_inspector.refresh()
        self.controller.request_camera_reset()
        self._refresh_scene()
