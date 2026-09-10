"""Qt workbench with asynchronous stages and embedded operator interactions."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from stacking_common.workflows.object_build import ObjectBuildWorkflow
from stacking_common.reconstruction import (
    DsfPartitionOptions,
    load_dsf_components,
    load_point_cloud,
)
from stacking_tabletop.config import load_inventory_profile, load_perception_profile
from stacking_tabletop.perception.extraction import ManualObjectSelection
from stacking_tabletop.perception.object_build_cli import default_workflow_directory
from stacking_tabletop.perception.reconstruction import _dsf_surface_mesh

from .controller import WorkbenchController, artifact_render_format
from .operators import ViewportRequest, WorkbenchOperators
from .worker import StageWorker


class ObjectBuildWindow(QMainWindow):
    """Persistent prepare, merge, reconstruct operator workbench."""

    def __init__(self, repository: Path, workflow_root: Path | None) -> None:
        super().__init__()
        self.repository = repository
        self.controller = (
            None if workflow_root is None else WorkbenchController(workflow_root)
        )
        self.profile = load_perception_profile(
            repository / "apps/tabletop/config/perception.yml"
        )
        self.inventory = load_inventory_profile(
            repository / "apps/tabletop/config/inventory.yml"
        )
        self.operators = WorkbenchOperators(self.profile)
        self.operators.interaction_requested.connect(self._begin_interaction)
        self.worker: StageWorker | None = None
        self.capture_directory: Path | None = None
        self.request: ViewportRequest | None = None
        self.interaction_phase: str | None = None
        self.interaction_points: np.ndarray | None = None
        self.selection_bounds: np.ndarray | None = None
        self.picked: list[np.ndarray] = []
        self.alignment_transform = np.eye(4)
        self.alignment_initial_transform = np.eye(4)
        self.alignment_source: np.ndarray | None = None
        self.alignment_target: np.ndarray | None = None
        self.shutting_down = False
        self.setWindowTitle("Tabletop Object Reconstruction Workbench")
        self.setMinimumSize(900, 600)
        self.resize(1500, 900)
        self._build_ui()
        self.statusBar().setSizeGripEnabled(True)
        self.statusBar().showMessage(
            "Drag the corner to resize the window; drag panel dividers to resize views."
        )
        self.refresh()

    def _build_ui(self) -> None:
        self.stage_tree = QTreeWidget()
        self.stage_tree.setHeaderLabels(["Stage", "Status", "Attempt"])
        self.stage_tree.itemSelectionChanged.connect(self._show_selected_artifact)
        rendering = QGroupBox("Status rendering")
        rendering_layout = QHBoxLayout(rendering)
        self.show_mesh = QCheckBox("Mesh")
        self.show_pcd = QCheckBox("PCD")
        self.show_dsf = QCheckBox("DSF")
        self.rendering_controls = (
            self.show_mesh,
            self.show_pcd,
            self.show_dsf,
        )
        for control in self.rendering_controls:
            control.setChecked(True)
            control.toggled.connect(
                lambda _checked=False: self._show_selected_artifact(
                    reset_camera=False
                )
            )
            rendering_layout.addWidget(control)
        rendering_layout.addStretch()
        status_panel = QWidget()
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.addWidget(rendering)
        status_layout.addWidget(self.stage_tree, 1)
        self.viewer = QtInteractor(self)
        self.viewer.installEventFilter(self)
        self.viewer.set_background("#20242b")
        self.viewer.add_axes()
        self._register_alignment_keys()

        session = QGroupBox("Object session")
        session_layout = QGridLayout(session)
        self.object_id_selector = QComboBox()
        for object_id in self.inventory.object_ids:
            self.object_id_selector.addItem(str(object_id), object_id)
        if self.controller is not None:
            index = self.object_id_selector.findData(
                self.controller.workflow.session.object_id
            )
            if index >= 0:
                self.object_id_selector.setCurrentIndex(index)
        self.start_object_button = QPushButton("Start new object session")
        self.start_object_button.clicked.connect(self._start_object_session)
        session_layout.addWidget(QLabel("Object ID"), 0, 0)
        session_layout.addWidget(self.object_id_selector, 0, 1)
        session_layout.addWidget(self.start_object_button, 1, 0, 1, 2)
        self.session_label = QLabel(
            "No active workflow"
            if self.controller is None
            else str(self.controller.root)
        )
        self.session_label.setWordWrap(True)
        session_layout.addWidget(self.session_label, 2, 0, 1, 2)
        self.capture_label = QLabel("No raw capture directory selected")
        self.choose_button = QPushButton("Choose capture directory")
        self.choose_button.clicked.connect(self._choose_capture)
        self.prepare_button = QPushButton("Run prepare")
        self.operation_progress = QProgressBar()
        self.operation_progress.setTextVisible(True)
        self.operation_progress.setVisible(False)
        self.merge_button = QPushButton("Run merge")
        self.reconstruct_button = QPushButton("Run reconstruct")
        self.cancel_button = QPushButton("Cancel interaction")
        self.prepare_button.clicked.connect(lambda: self._run_stage("prepare"))
        self.merge_button.clicked.connect(lambda: self._run_stage("merge"))
        self.reconstruct_button.clicked.connect(
            lambda: self._run_stage("reconstruct")
        )
        self.cancel_button.clicked.connect(self._cancel)
        self.manual_cleanup = QCheckBox("Manual cleanup picks")
        self.inspect_icp = QCheckBox("Inspect/reprocess in-hand ICP views")
        self.inspect_icp.setToolTip(
            "During Prepare, review each automatic in-hand ICP candidate. "
            "Accept or reject views, and press U to go back and rebuild the "
            "accumulated reference."
        )
        self.manual_partition = QCheckBox("Manual DSF partition seeds")
        self.review_result = QCheckBox("Review reconstruction")
        self.review_result.setChecked(True)

        tuning = QGroupBox("Reconstruction tuning")
        tuning_layout = QGridLayout(tuning)
        reconstruction = self.profile.reconstruction
        self.coacd_components = QSpinBox()
        self.coacd_components.setRange(1, 64)
        self.coacd_components.setValue(
            reconstruction.convex_decomposition.maximum_components
        )
        self.coacd_threshold = QDoubleSpinBox()
        self.coacd_threshold.setDecimals(4)
        self.coacd_threshold.setRange(0.0001, 1.0)
        self.coacd_threshold.setSingleStep(0.005)
        self.coacd_threshold.setValue(
            reconstruction.convex_decomposition.threshold
        )
        self.dsf_vertices = QSpinBox()
        self.dsf_vertices.setRange(4, 1024)
        self.dsf_vertices.setValue(reconstruction.dsf_fitting.total_vertices)
        self.dsf_iterations = QSpinBox()
        self.dsf_iterations.setRange(1, 100000)
        self.dsf_iterations.setValue(
            reconstruction.dsf_fitting.maximum_iterations
        )
        self.dsf_ransac = QCheckBox("Use DSF RANSAC")
        self.dsf_ransac.setChecked(reconstruction.dsf_fitting.ransac.enabled)
        self.reconstruction_controls = (
            self.coacd_components,
            self.coacd_threshold,
            self.dsf_vertices,
            self.dsf_iterations,
            self.dsf_ransac,
        )
        self.coacd_components.valueChanged.connect(
            self._update_dsf_vertex_minimum
        )
        self._update_dsf_vertex_minimum(self.coacd_components.value())
        for row, (label, widget) in enumerate(
            (
                ("CoACD max parts", self.coacd_components),
                ("CoACD threshold", self.coacd_threshold),
                ("DSF vertices", self.dsf_vertices),
                ("DSF max evaluations", self.dsf_iterations),
            )
        ):
            tuning_layout.addWidget(QLabel(label), row, 0)
            tuning_layout.addWidget(widget, row, 1)
        tuning_layout.addWidget(self.dsf_ransac, 4, 0, 1, 2)

        interaction = QGroupBox("Viewport interaction")
        interaction_layout = QVBoxLayout(interaction)
        self.interaction_label = QLabel("No operator input requested")
        self.interaction_label.setWordWrap(True)
        interaction_layout.addWidget(self.interaction_label)
        step_layout = QHBoxLayout()
        self.translation_step = QDoubleSpinBox()
        self.translation_step.setRange(0.01, 100.0)
        self.translation_step.setValue(
            1000.0 * self.profile.cross_capture_merge.initial_translation_step
        )
        self.translation_step.setSuffix(" mm")
        self.rotation_step = QDoubleSpinBox()
        self.rotation_step.setRange(0.01, 45.0)
        self.rotation_step.setValue(
            self.profile.cross_capture_merge.initial_rotation_step_degrees
        )
        self.rotation_step.setSuffix(" deg")
        step_layout.addWidget(self.translation_step)
        step_layout.addWidget(self.rotation_step)
        interaction_layout.addLayout(step_layout)
        nudge_layout = QGridLayout()
        self.nudge_buttons = []
        labels = (
            ("−X", 0, -1, False), ("+X", 0, 1, False),
            ("−Y", 1, -1, False), ("+Y", 1, 1, False),
            ("−Z", 2, -1, False), ("+Z", 2, 1, False),
            ("−Roll", 0, -1, True), ("+Roll", 0, 1, True),
            ("−Pitch", 1, -1, True), ("+Pitch", 1, 1, True),
            ("−Yaw", 2, -1, True), ("+Yaw", 2, 1, True),
        )
        for index, (label, axis, sign, rotation) in enumerate(labels):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, a=axis, s=sign, r=rotation: self._nudge(a, s, r)
            )
            nudge_layout.addWidget(button, index // 2, index % 2)
            self.nudge_buttons.append(button)
        interaction_layout.addLayout(nudge_layout)
        action_layout = QHBoxLayout()
        self.undo_button = QPushButton("Undo pick (U)")
        self.accept_button = QPushButton("Accept")
        self.secondary_button = QPushButton("Reject / cancel")
        self.undo_button.clicked.connect(self._undo_pick)
        self.accept_button.clicked.connect(self._accept_interaction)
        self.secondary_button.clicked.connect(self._reject_interaction)
        action_layout.addWidget(self.undo_button)
        action_layout.addWidget(self.accept_button)
        action_layout.addWidget(self.secondary_button)
        interaction_layout.addLayout(action_layout)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        controls = QWidget()
        controls.setMinimumWidth(220)
        controls_layout = QVBoxLayout(controls)
        for widget in (
            session, self.capture_label,
            self.choose_button,
            self.prepare_button, self.operation_progress,
            self.merge_button, self.reconstruct_button,
            self.inspect_icp,
            self.manual_cleanup, self.manual_partition, self.review_result,
            tuning,
            self.cancel_button, interaction,
        ):
            controls_layout.addWidget(widget)
        controls_layout.addStretch()
        self.upper_splitter = QSplitter()
        self.upper_splitter.setChildrenCollapsible(False)
        self.upper_splitter.setHandleWidth(10)
        self.upper_splitter.setOpaqueResize(False)
        self.upper_splitter.addWidget(controls)
        self.upper_splitter.addWidget(self.viewer)
        self.upper_splitter.addWidget(status_panel)
        self.upper_splitter.setStretchFactor(0, 0)
        self.upper_splitter.setStretchFactor(1, 1)
        self.upper_splitter.setStretchFactor(2, 0)
        self.upper_splitter.setSizes([250, 1010, 240])
        self.vertical_splitter = QSplitter()
        self.vertical_splitter.setOrientation(Qt.Vertical)
        self.vertical_splitter.setChildrenCollapsible(False)
        self.vertical_splitter.setHandleWidth(10)
        self.vertical_splitter.setOpaqueResize(False)
        self.vertical_splitter.addWidget(self.upper_splitter)
        self.vertical_splitter.addWidget(self.log)
        self.vertical_splitter.setStretchFactor(0, 1)
        self.vertical_splitter.setStretchFactor(1, 0)
        self.vertical_splitter.setSizes([730, 150])
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self.vertical_splitter)
        self.setCentralWidget(central)
        self._set_interaction_enabled(False)

    def refresh(self) -> None:
        self.stage_tree.clear()
        busy = self.worker is not None
        self.object_id_selector.setEnabled(not busy and self.request is None)
        self.start_object_button.setEnabled(not busy and self.request is None)
        if self.controller is None:
            self.prepare_button.setEnabled(False)
            self.merge_button.setEnabled(False)
            self.reconstruct_button.setEnabled(False)
            self.choose_button.setEnabled(False)
            self.cancel_button.setEnabled(self.request is not None)
            self.interaction_label.setText(
                "Choose an object ID and start a new object session."
            )
            return
        stages = self.controller.refresh()
        for stage in stages:
            item = QTreeWidgetItem(
                [stage.name, stage.status, stage.attempt_id or "—"]
            )
            item.setData(0, 256, stage.name)
            self.stage_tree.addTopLevelItem(item)
            for artifact in stage.artifacts:
                child = QTreeWidgetItem(
                    [Path(artifact.path).name, artifact.frame, ""]
                )
                child.setData(0, 256, artifact.path)
                child.setToolTip(0, artifact.artifact_id)
                item.addChild(child)
            item.setExpanded(True)
        self.prepare_button.setEnabled(not busy)
        self.merge_button.setEnabled(not busy and self.controller.can_run("merge"))
        self.reconstruct_button.setEnabled(
            not busy and self.controller.can_run("reconstruct")
        )
        self.cancel_button.setEnabled(self.request is not None)
        self.choose_button.setEnabled(not busy)
        self.manual_cleanup.setEnabled(not busy)
        self.inspect_icp.setEnabled(not busy)
        self.manual_partition.setEnabled(not busy)
        self.review_result.setEnabled(not busy)
        for control in self.reconstruction_controls:
            control.setEnabled(not busy)
        if busy and self.request is None:
            self.interaction_label.setText(
                "Background stage processing; workflow controls are disabled."
            )
        elif not busy and self.request is None:
            self.interaction_label.setText("No operator input requested")

    def _start_object_session(self) -> None:
        if self.worker is not None or self.request is not None:
            return
        if self.controller is not None:
            reconstruct = self.controller.stage("reconstruct")
            if reconstruct.status != "accepted":
                answer = QMessageBox.question(
                    self,
                    "Switch unfinished object?",
                    "The current object has not completed reconstruction. Start "
                    "a new object session anyway? The current session remains "
                    "saved and can be resumed later.",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
        object_id = int(self.object_id_selector.currentData())
        root = default_workflow_directory(self.repository, object_id)
        ObjectBuildWorkflow.create(
            root,
            object_id=object_id,
            session_id=root.name,
            platform="tabletop",
            density_kg_m3=self.inventory.density_for(object_id),
        )
        self.controller = WorkbenchController(root)
        self.capture_directory = None
        self.capture_label.setText("No raw capture directory selected")
        self.session_label.setText(str(root))
        self.operation_progress.setVisible(False)
        self.viewer.clear()
        self.viewer.add_axes()
        self.log.appendPlainText(f"\nstarted object {object_id}: {root}")
        self.refresh()

    def _choose_capture(self) -> None:
        value = QFileDialog.getExistingDirectory(self, "Raw capture directory")
        if value:
            self.capture_directory = Path(value)
            self.capture_label.setText(value)

    def _update_dsf_vertex_minimum(self, components: int) -> None:
        self.dsf_vertices.setMinimum(4 * components)

    def _run_stage(self, stage: str) -> None:
        if self.worker is not None:
            return
        if self.controller is None:
            QMessageBox.warning(
                self,
                "Object session required",
                "Choose an object ID and start an object session first.",
            )
            return
        if stage == "prepare" and self.capture_directory is None:
            QMessageBox.warning(self, "Capture required", "Choose a capture directory.")
            return
        self.worker = StageWorker(
            self.repository,
            self.controller.root,
            stage,
            self.capture_directory,
            self.operators,
            manual_cleanup=self.manual_cleanup.isChecked(),
            manual_partition=self.manual_partition.isChecked(),
            review_result=self.review_result.isChecked(),
            inspect_icp=self.inspect_icp.isChecked(),
            convex_decomposition_profile=replace(
                self.profile.reconstruction.convex_decomposition,
                maximum_components=self.coacd_components.value(),
                threshold=self.coacd_threshold.value(),
            ),
            dsf_fitting_profile=replace(
                self.profile.reconstruction.dsf_fitting,
                total_vertices=self.dsf_vertices.value(),
                maximum_iterations=self.dsf_iterations.value(),
                ransac=replace(
                    self.profile.reconstruction.dsf_fitting.ransac,
                    enabled=self.dsf_ransac.isChecked(),
                ),
            ),
        )
        self.worker.succeeded.connect(self._stage_succeeded)
        self.worker.failed.connect(self._stage_failed)
        self.worker.progress.connect(self._stage_progress)
        self.worker.finished.connect(self._worker_finished)
        self.operation_progress.setVisible(False)
        self.log.appendPlainText(f"\nrunning {stage} in the workbench")
        self.worker.start()
        self.refresh()

    def _stage_succeeded(self, stage: str, attempt: str) -> None:
        if self.shutting_down:
            return
        self.log.appendPlainText(f"accepted {stage} attempt: {attempt}")
        if self.operation_progress.isVisible():
            self.operation_progress.setRange(0, 1)
            self.operation_progress.setValue(1)
            self.operation_progress.setFormat(f"{stage.capitalize()} complete")

    def _stage_failed(self, stage: str, message: str) -> None:
        if self.shutting_down:
            return
        self.log.appendPlainText(f"{stage} failed: {message}")
        if self.operation_progress.isVisible():
            self.operation_progress.setRange(0, 1)
            self.operation_progress.setValue(0)
            self.operation_progress.setFormat(f"{stage.capitalize()} stopped")
        QMessageBox.critical(self, f"{stage} failed", message)

    def _stage_progress(self, completed: int, total: int, message: str) -> None:
        self.operation_progress.setVisible(True)
        if total <= 0:
            self.operation_progress.setRange(0, 0)
            self.operation_progress.setFormat(message)
        else:
            self.operation_progress.setRange(0, total)
            self.operation_progress.setValue(completed)
            self.operation_progress.setFormat(f"{message} — %p%")
        if self.request is None:
            self.interaction_label.setText(message)

    def _worker_finished(self) -> None:
        worker = self.worker
        completed_stage = (
            worker.stage
            if worker is not None and worker.completed_attempt_id is not None
            else None
        )
        self.worker = None
        if self.shutting_down:
            return
        if self.request is not None:
            self.request.reject(RuntimeError("stage ended during operator input"))
        self._finish_request()
        self.refresh()
        if completed_stage is not None:
            self._show_stage_artifact(completed_stage)

    def _cancel(self) -> None:
        if self.request is not None:
            self.request.reject(RuntimeError("operator cancelled the interaction"))
            self._finish_request()

    def _begin_interaction(self, request: ViewportRequest) -> None:
        if self.request is not None:
            request.reject(RuntimeError("another viewport interaction is active"))
            return
        self.request = request
        self.picked = []
        self.operation_progress.setVisible(False)
        if request.kind == "prepare_continue":
            completed, total, next_capture = request.payload
            answer = QMessageBox.question(
                self,
                "Continue preparation?",
                f"Prepared {completed} of {total} captures. Proceed with "
                f"the next capture?\n\n{next_capture.name}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            request.resolve(answer == QMessageBox.StandardButton.Yes)
            self.request = None
            self.refresh()
            return
        if request.kind == "object_selection":
            self._start_object_selection(request.payload)
        elif request.kind == "icp_inspection":
            self._start_icp_inspection(request.payload)
        elif request.kind == "alignment":
            self._start_alignment(request.payload)
        elif request.kind == "cleanup":
            self._start_picking(
                request.payload.points,
                "Hover over an unwanted point and press P to mark its red "
                "removal neighborhood.",
                self.profile.reconstruction.cleanup.manual_remove_distance,
                "cleanup",
                keyboard_picking=True,
            )
        elif request.kind == "partition":
            self._start_picking(
                request.payload.points,
                "Hover inside each desired DSF component and press P to add "
                "a partition seed.",
                DsfPartitionOptions().marker_radius,
                "partition",
                keyboard_picking=True,
            )
        elif request.kind == "review":
            self._start_review(request.payload)
        else:
            request.reject(ValueError(f"unsupported viewport request: {request.kind}"))
            self._finish_request()
            return
        self._set_interaction_enabled(True)
        self.refresh()

    def _start_object_selection(self, cloud) -> None:
        import pyvista as pv

        self._reset_viewport_interaction()
        points = np.asarray(cloud.points)
        self.interaction_points = points
        self.selection_bounds = np.asarray(pv.PolyData(points).bounds)
        self.interaction_phase = "object_crop"
        self.interaction_label.setText(
            "Move and resize the green box around the object, then apply the crop."
        )
        self.accept_button.setText("Apply crop")
        self._render_points(points)
        self.viewer.add_box_widget(
            lambda value: setattr(
                self, "selection_bounds", np.asarray(pv.wrap(value).bounds)
            ),
            bounds=tuple(self.selection_bounds),
            rotation_enabled=False,
            color="green",
            use_planes=False,
        )

    def _start_alignment(self, payload) -> None:
        self._reset_viewport_interaction()
        source, target, source_index = payload
        self.alignment_source = np.asarray(source.points)
        self.alignment_target = np.asarray(target.points)
        self.alignment_transform = np.eye(4)
        self.alignment_transform[:3, 3] = (
            self.alignment_target.mean(axis=0) - self.alignment_source.mean(axis=0)
        )
        self.alignment_initial_transform = self.alignment_transform.copy()
        self.interaction_phase = "alignment"
        self.interaction_label.setText(
            f"Align capture {source_index + 1}: blue is fixed, orange is moving. "
            "Keys A/D, S/W, F/R translate; U/O, I/K, J/L rotate; "
            "[ and ] change step; X resets."
        )
        self.accept_button.setText("Accept alignment")
        self._render_alignment(reset_camera=True)

    def _start_icp_inspection(self, case) -> None:
        import pyvista as pv

        self._reset_viewport_interaction()
        self.interaction_phase = "icp_inspection"
        registration = case.registration
        quality = (
            "anchor view"
            if registration is None
            else (
                f"target/source coverage "
                f"{registration.target_coverage:.2f}/"
                f"{registration.source_coverage:.2f}, "
                f"RMSE {1000.0 * registration.rmse:.2f} mm"
            )
        )
        status = "accepted" if case.decision.accepted else (
            "rejected: " + ", ".join(case.decision.reasons)
        )
        self.interaction_label.setText(
            f"Inspect view {case.index + 1}/{case.total}: "
            f"{case.metadata.filename}\n{quality}; automatic {status}.\n"
            "Gray=accepted reference, red=recorded pose, green/orange=ICP "
            "candidate. Accept, reject, or press U to revisit the previous view."
        )
        self.accept_button.setText("Accept view")
        self.secondary_button.setText("Reject view")
        self.undo_button.setText("Back (U)")
        self.viewer.clear()
        self.viewer.add_axes()
        if case.reference is not None and len(case.reference):
            self.viewer.add_points(
                pv.PolyData(case.reference.points),
                color="#8e949c",
                point_size=3,
            )
        self.viewer.add_points(
            pv.PolyData(case.source.points),
            color="#f04444",
            point_size=3,
        )
        self.viewer.add_points(
            pv.PolyData(case.candidate.points),
            color="#32d46d" if case.decision.accepted else "#f0a126",
            point_size=4,
        )
        self.viewer.reset_camera()
        self.viewer.render()

    def _start_picking(
        self,
        points: np.ndarray,
        message: str,
        radius: float,
        phase: str,
        *,
        keyboard_picking: bool = False,
    ) -> None:
        self._reset_viewport_interaction()
        self.interaction_phase = phase
        self.interaction_points = np.asarray(points)
        self.pick_radius = radius
        self.interaction_label.setText(message)
        self.accept_button.setText("Accept picks")
        self._render_points(self.interaction_points)
        self.viewer.enable_point_picking(
            callback=self._pick_point,
            left_clicking=not keyboard_picking,
            show_message=False,
            show_point=True,
            point_size=12,
            color="red",
        )
        if keyboard_picking:
            observer = getattr(
                self.viewer,
                "_picking_right_clicking_observer",
                None,
            )
            if observer is not None:
                self.viewer.iren.remove_observer(observer)
                self.viewer._picking_right_clicking_observer = None

    def _start_review(self, result) -> None:
        import pyvista as pv

        self._reset_viewport_interaction()
        self.interaction_phase = "review"
        self.interaction_label.setText(
            "Inspect the reconstructed mesh and smooth DSF surfaces. Accept or retry."
        )
        self.accept_button.setText("Accept result")
        self.secondary_button.setText(
            "Retry seed selection"
            if result.split_method == "manual_seeds"
            else "Reject result"
        )
        self.viewer.clear()
        self.viewer.add_axes()
        self.viewer.add_points(pv.PolyData(result.centered.points), point_size=3)
        self.viewer.add_mesh(
            _mesh_polydata(pv, result.mesh.vertices, result.mesh.triangles),
            color="lightsteelblue", opacity=0.35, show_edges=True,
        )
        colors = ("orange", "limegreen", "magenta", "cyan", "yellow")
        for index, component in enumerate(result.components):
            self.viewer.add_mesh(
                _dsf_surface_mesh(pv, component),
                color=colors[index % len(colors)], opacity=0.5,
            )
        self.viewer.reset_camera()

    def _render_points(self, points: np.ndarray) -> None:
        import pyvista as pv

        self.viewer.clear()
        self.viewer.add_axes()
        cloud = pv.PolyData(points)
        cloud["z"] = points[:, 2]
        self.viewer.add_points(cloud, scalars="z", point_size=4)
        self.viewer.reset_camera()
        self.viewer.render()

    def _render_alignment(self, *, reset_camera: bool = False) -> None:
        import pyvista as pv

        camera_position = None if reset_camera else self.viewer.camera_position
        source = _transform_points(self.alignment_source, self.alignment_transform)
        self.viewer.clear()
        self.viewer.add_axes()
        self.viewer.add_points(
            pv.PolyData(self.alignment_target), color="#2674ff", point_size=4
        )
        self.viewer.add_points(pv.PolyData(source), color="#ff591a", point_size=4)
        if reset_camera:
            self.viewer.reset_camera()
        elif camera_position is not None:
            self.viewer.camera_position = camera_position
        self.viewer.render()

    def _nudge(self, axis: int, sign: int, rotation: bool) -> None:
        if self.interaction_phase != "alignment":
            return
        update = np.eye(4)
        if rotation:
            vector = np.zeros(3)
            vector[axis] = np.deg2rad(sign * self.rotation_step.value())
            update[:3, :3] = _axis_angle(vector)
            center = _transform_points(
                self.alignment_source, self.alignment_transform
            ).mean(axis=0)
            update[:3, 3] = center - update[:3, :3] @ center
        else:
            update[axis, 3] = sign * self.translation_step.value() / 1000.0
        self.alignment_transform = update @ self.alignment_transform
        self._render_alignment()

    def _register_alignment_keys(self) -> None:
        bindings = {
            "a": (0, -1, False),
            "d": (0, 1, False),
            "s": (1, -1, False),
            "w": (1, 1, False),
            "f": (2, -1, False),
            "u": (0, 1, True),
            "o": (0, -1, True),
            "i": (1, 1, True),
            "k": (1, -1, True),
            "j": (2, 1, True),
            "l": (2, -1, True),
        }
        for key, (axis, sign, rotation) in bindings.items():
            self.viewer.clear_events_for_key(key)
            self.viewer.add_key_event(
                key,
                lambda a=axis, s=sign, value=rotation: self._nudge(a, s, value),
            )
        self.viewer.clear_events_for_key("bracketleft")
        self.viewer.add_key_event(
            "bracketleft",
            lambda: self._scale_alignment_step(0.5),
        )
        self.viewer.clear_events_for_key("bracketright")
        self.viewer.add_key_event(
            "bracketright",
            lambda: self._scale_alignment_step(2.0),
        )
        self.viewer.clear_events_for_key("x")
        self.viewer.add_key_event("x", self._reset_alignment)
        self.viewer.add_key_event("u", self._undo_pick)

    def eventFilter(self, watched, event) -> bool:
        if (
            watched is self.viewer
            and event.type() == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_R
            and self.interaction_phase == "alignment"
        ):
            self._nudge(2, 1, False)
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def _scale_alignment_step(self, factor: float) -> None:
        if self.interaction_phase != "alignment":
            return
        self.translation_step.setValue(self.translation_step.value() * factor)
        self.rotation_step.setValue(self.rotation_step.value() * factor)

    def _reset_alignment(self) -> None:
        if self.interaction_phase != "alignment":
            return
        self.alignment_transform = self.alignment_initial_transform.copy()
        self._render_alignment()

    def _pick_point(self, point) -> None:
        if point is None:
            return
        import pyvista as pv

        center = np.asarray(point, dtype=np.float64)
        self.picked.append(center)
        self.viewer.add_mesh(
            pv.Sphere(radius=self.pick_radius, center=center),
            color="red", opacity=0.25,
            name=f"operator_pick_{len(self.picked)}", pickable=False,
        )

    def _undo_pick(self) -> None:
        if self.interaction_phase == "icp_inspection" and self.request is not None:
            self.request.resolve("back")
            self._finish_request()
            return
        if self.picked:
            self.viewer.remove_actor(f"operator_pick_{len(self.picked)}")
            self.picked.pop()

    def _accept_interaction(self) -> None:
        if self.request is None:
            return
        if self.interaction_phase == "object_crop":
            bounds = self.selection_bounds
            lower, upper = bounds[[0, 2, 4]], bounds[[1, 3, 5]]
            cropped = self.interaction_points[
                np.all(
                    (self.interaction_points >= lower)
                    & (self.interaction_points <= upper), axis=1,
                )
            ]
            if not len(cropped):
                QMessageBox.warning(self, "Empty crop", "The crop contains no points.")
                return
            self._start_picking(
                cropped,
                "Optionally hover over gripper points and press P to mark "
                "removal regions, then accept.",
                self.profile.in_hand_capture.manual_remove_distance,
                "object_remove",
                keyboard_picking=True,
            )
            self._set_interaction_enabled(True)
            return
        if self.interaction_phase == "object_remove":
            value = ManualObjectSelection(
                self.selection_bounds,
                np.asarray(self.picked).reshape(-1, 3),
                self.profile.in_hand_capture.manual_remove_distance,
                self.profile.in_hand_capture.manual_max_remove_ratio,
            )
        elif self.interaction_phase == "alignment":
            value = self.alignment_transform.copy()
            self._stage_progress(0, 0, "Manual seed accepted; starting ICP refinement")
        elif self.interaction_phase == "icp_inspection":
            value = "accept"
        elif self.interaction_phase in {"cleanup", "partition"}:
            if self.interaction_phase == "partition" and not self.picked:
                QMessageBox.warning(self, "Seeds required", "Pick at least one seed.")
                return
            value = np.asarray(self.picked, dtype=np.float64).reshape(-1, 3)
        elif self.interaction_phase == "review":
            value = True
        else:
            return
        self.request.resolve(value)
        self._finish_request()

    def _reject_interaction(self) -> None:
        if self.request is None:
            return
        if self.interaction_phase == "review":
            self.request.resolve(False)
        elif self.interaction_phase == "icp_inspection":
            self.request.resolve("reject")
        else:
            self.request.reject(RuntimeError("operator rejected the interaction"))
        self._finish_request()

    def _finish_request(self) -> None:
        self._reset_viewport_interaction()
        self.request = None
        self.interaction_phase = None
        self.picked = []
        self.interaction_label.setText(
            "Background stage processing; workflow controls are disabled."
            if self.worker is not None
            else "No operator input requested"
        )
        self.accept_button.setText("Accept")
        self.secondary_button.setText("Reject / cancel")
        self.undo_button.setText("Undo pick (U)")
        self._set_interaction_enabled(False)

    def _set_interaction_enabled(self, enabled: bool) -> None:
        self.accept_button.setEnabled(enabled)
        self.secondary_button.setEnabled(enabled)
        self.undo_button.setEnabled(enabled)
        alignment = enabled and self.interaction_phase == "alignment"
        self.translation_step.setEnabled(alignment)
        self.rotation_step.setEnabled(alignment)
        for button in self.nudge_buttons:
            button.setEnabled(alignment)

    def _reset_viewport_interaction(self) -> None:
        """Remove stale picking state and restore camera manipulation."""

        try:
            self.viewer.disable_picking()
        except (AttributeError, RuntimeError):
            pass
        try:
            self.viewer.clear_box_widgets()
        except (AttributeError, RuntimeError):
            pass
        self.viewer.enable_trackball_style()

    def _show_selected_artifact(self, *, reset_camera: bool = True) -> None:
        if self.request is not None or self.controller is None:
            return
        selected = self.stage_tree.selectedItems()
        if not selected:
            return
        item = selected[0]
        stage_item = item if item.parent() is None else item.parent()
        stage = stage_item.data(0, 256)
        if not isinstance(stage, str) or stage not in self.controller.stages:
            return
        self._show_stage_rendering(stage, reset_camera=reset_camera)

    def _show_stage_rendering(
        self,
        stage: str,
        *,
        reset_camera: bool,
    ) -> None:
        import pyvista as pv

        camera_position = None if reset_camera else self.viewer.camera_position
        self.viewer.clear()
        self.viewer.add_axes()
        colors = ("orange", "limegreen", "magenta", "cyan", "yellow")
        rendered = 0
        for artifact in self.controller.stage(stage).artifacts:
            path = self.controller.artifact_path(artifact)
            if not path.is_file():
                continue
            try:
                artifact_format = artifact_render_format(path)
                if artifact_format == "pcd" and self.show_pcd.isChecked():
                    points = load_point_cloud(path).points
                    cloud = pv.PolyData(points)
                    cloud["z"] = points[:, 2]
                    self.viewer.add_points(
                        cloud,
                        scalars="z",
                        point_size=4,
                        name=f"status_pcd_{rendered}",
                    )
                    rendered += 1
                elif artifact_format == "mesh" and self.show_mesh.isChecked():
                    self.viewer.add_mesh(
                        pv.read(path),
                        color="lightsteelblue",
                        opacity=0.35,
                        show_edges=True,
                        name=f"status_mesh_{rendered}",
                    )
                    rendered += 1
                elif artifact_format == "dsf" and self.show_dsf.isChecked():
                    for index, component in enumerate(load_dsf_components(path)):
                        self.viewer.add_mesh(
                            _dsf_surface_mesh(pv, component),
                            color=colors[index % len(colors)],
                            opacity=0.65,
                            smooth_shading=True,
                            name=f"status_dsf_{rendered}_{index}",
                        )
                    rendered += 1
            except Exception as error:
                self.log.appendPlainText(f"preview failed for {path}: {error}")
        if reset_camera and rendered:
            self.viewer.reset_camera()
        elif camera_position is not None:
            self.viewer.camera_position = camera_position
        self.viewer.render()

    def _show_stage_artifact(self, stage: str) -> None:
        if self.controller is None:
            return
        path = self.controller.preview_artifact(stage)
        if path is None:
            return
        relative = str(path.relative_to(self.controller.root))
        for index in range(self.stage_tree.topLevelItemCount()):
            item = self.stage_tree.topLevelItem(index)
            if item.data(0, 256) != stage:
                continue
            for child_index in range(item.childCount()):
                child = item.child(child_index)
                if child.data(0, 256) == relative:
                    self.stage_tree.blockSignals(True)
                    self.stage_tree.setCurrentItem(child)
                    self.stage_tree.blockSignals(False)
                    break
        self.log.appendPlainText(f"displaying {stage} result: {path}")
        self._show_stage_rendering(stage, reset_camera=True)

    def closeEvent(self, event) -> None:
        worker = self.worker
        if worker is not None and worker.isRunning():
            answer = QMessageBox.question(
                self,
                "Cancel running stage?",
                "A reconstruction stage is still running. Cancel the current "
                "attempt and close the workbench?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self.shutting_down = True
        if self.request is not None:
            self.request.reject(RuntimeError("workbench closed during operator input"))
            self.request = None
        if worker is not None and worker.isRunning():
            stage = worker.stage
            worker.requestInterruption()
            if not worker.wait(1500):
                worker.terminate()
            if not worker.wait(5000):
                self.shutting_down = False
                QMessageBox.critical(
                    self,
                    "Unable to stop stage",
                    "The background stage did not stop. The workbench remains "
                    "open to avoid destroying a live worker thread.",
                )
                event.ignore()
                return
            self.worker = None
            try:
                self.controller.finish_interrupted_process(
                    stage,
                    cancelled=True,
                    exit_code=-1,
                )
            except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
                print(f"warning: could not mark interrupted {stage} attempt: {error}")
        self._reset_viewport_interaction()
        self.viewer.close()
        super().closeEvent(event)


def _mesh_polydata(pv, vertices: np.ndarray, triangles: np.ndarray):
    faces = np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
    return pv.PolyData(vertices, faces)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _axis_angle(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle == 0.0:
        return np.eye(3)
    axis = vector / angle
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
