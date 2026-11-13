"""Search Inspector controls for MCTS artifacts and replay."""

from collections.abc import Mapping

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from stacking_common.visualization import CandidatePoseMode, CandidateRenderMode

from .controller import OperationsController


class SearchInspector(QWidget):
    """Controls for inspecting search content from one loaded run."""

    open_requested = Signal()
    refresh_requested = Signal()
    video_requested = Signal(float)
    state_changed = Signal()

    def __init__(
        self,
        controller: OperationsController,
        environment_defaults: Mapping[str, bool] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self._tree_dialog = None
        self._updating = False
        self._timeline_render_timer = QTimer(self)
        self._timeline_render_timer.setSingleShot(True)
        self._timeline_render_timer.setInterval(25)
        self._timeline_render_timer.timeout.connect(self.state_changed.emit)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        self.scroll_area.setWidget(content)
        outer.addWidget(self.scroll_area)
        layout = QVBoxLayout(content)

        source = QGroupBox("Stacking run")
        source_layout = QVBoxLayout(source)
        self.open_button = QPushButton("Open stacking run…")
        self.open_button.clicked.connect(lambda: self.open_requested.emit())
        self.refresh_button = QPushButton("Refresh run")
        self.refresh_button.setEnabled(False)
        self.refresh_button.setToolTip("Reload committed updates from the open .stack file (Ctrl+R)")
        self.refresh_button.clicked.connect(lambda: self.refresh_requested.emit())
        self.source_label = QLabel("No run loaded")
        self.source_label.setWordWrap(True)
        source_layout.addWidget(self.open_button)
        source_layout.addWidget(self.refresh_button)
        source_layout.addWidget(self.source_label)
        layout.addWidget(source)

        navigation = QGroupBox("Search Inspector")
        form = QFormLayout(navigation)
        self.step_selector = QComboBox()
        self.candidate_selector = QComboBox()
        self.render_selector = QComboBox()
        self.pose_selector = QComboBox()
        for label, mode in (
            ("All candidates", CandidateRenderMode.ALL),
            ("All candidates — opaque", CandidateRenderMode.ALL_OPAQUE),
            ("Active candidate", CandidateRenderMode.SELECTED),
            ("State only", CandidateRenderMode.STATE_ONLY),
        ):
            self.render_selector.addItem(label, mode.value)
        for label, mode in (
            ("Initial", CandidatePoseMode.INITIAL),
            ("Solved", CandidatePoseMode.SOLVED),
            ("Designed action", CandidatePoseMode.ACTION),
            ("Initial + solved", CandidatePoseMode.INITIAL_AND_SOLVED),
            ("Simulation start", CandidatePoseMode.SIMULATION_START),
            ("Final", CandidatePoseMode.FINAL),
            ("Simulation start + final", CandidatePoseMode.SIMULATION_START_AND_FINAL),
            ("Trajectory", CandidatePoseMode.TRAJECTORY),
            ("Best sequence", CandidatePoseMode.BEST_SEQUENCE),
        ):
            self.pose_selector.addItem(label, mode.value)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setMinimum(0)
        self.timeline_label = QLabel("No trajectory")
        self.video_fps = QSpinBox()
        self.video_fps.setRange(1, 60)
        self.video_fps.setValue(30)
        self.video_button = QPushButton("Save trajectory MP4…")
        self.video_button.clicked.connect(
            lambda: self.video_requested.emit(float(self.video_fps.value()))
        )
        form.addRow("Step", self.step_selector)
        self.tree_button = QPushButton("MCTS tree…")
        self.tree_button.setEnabled(False)
        self.tree_button.clicked.connect(self._show_tree)
        form.addRow("Search tree", self.tree_button)
        form.addRow("Candidate", self.candidate_selector)
        self.candidate_failure_reason = QLabel()
        self.candidate_failure_reason.setWordWrap(True)
        self.candidate_failure_reason.setTextFormat(Qt.TextFormat.PlainText)
        self.candidate_failure_reason.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("Failure reason", self.candidate_failure_reason)
        form.addRow("Rendering", self.render_selector)
        form.addRow("Pose", self.pose_selector)
        form.addRow("Replay", self.timeline)
        form.addRow("", self.timeline_label)
        form.addRow("Video FPS", self.video_fps)
        form.addRow("", self.video_button)
        layout.addWidget(navigation)

        outcome = QGroupBox("Search outcome")
        outcome_layout = QVBoxLayout(outcome)
        self.failure_details = QPlainTextEdit()
        self.failure_details.setReadOnly(True)
        self.failure_details.setMinimumHeight(220)
        outcome_layout.addWidget(self.failure_details)
        layout.addWidget(outcome)

        diagnostics = QGroupBox("Diagnostic layers")
        diagnostic_layout = QVBoxLayout(diagnostics)
        self.contacts = QCheckBox("Simulation contacts")
        self.forces = QCheckBox("Pose-solve forces")
        diagnostic_layout.addWidget(self.contacts)
        diagnostic_layout.addWidget(self.forces)
        layout.addWidget(diagnostics)

        environment = QGroupBox("Environment layers")
        environment_layout = QVBoxLayout(environment)
        defaults = dict(environment_defaults or {})
        self.environment_controls = {}
        for layer, label, checked in (
            ("target", "Target structure", True),
            ("ground", "Ground", defaults.get("ground", True)),
            ("support", "Support tables", defaults.get("support", False)),
            (
                "obstacle",
                "Planning obstacles",
                defaults.get("obstacle", False),
            ),
        ):
            control = QCheckBox(label)
            control.setChecked(checked)
            control.toggled.connect(self._environment_changed)
            self.environment_controls[layer] = control
            environment_layout.addWidget(control)
        layout.addWidget(environment)

        self.summary = QLabel("Load a stacking run to inspect search results.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        layout.addStretch()

        self.step_selector.currentIndexChanged.connect(self._step_changed)
        self.candidate_selector.currentIndexChanged.connect(
            self._candidate_changed
        )
        self.render_selector.currentIndexChanged.connect(self._render_changed)
        self.pose_selector.currentIndexChanged.connect(self._pose_changed)
        self.timeline.valueChanged.connect(self._timeline_changed)
        self.contacts.toggled.connect(self._layers_changed)
        self.forces.toggled.connect(self._layers_changed)
        self.refresh()

    def refresh(self, source: str | None = None) -> None:
        self._updating = True
        try:
            if source is not None:
                self.source_label.setText(source)
            artifact = self.controller.artifact
            step = self.controller.current_step
            self.step_selector.clear()
            if artifact is not None:
                for value in artifact.steps:
                    outcome = "succeeded" if value.succeeded else (
                        value.failure_reason or "unsuccessful"
                    )
                    self.step_selector.addItem(
                        f"Step {value.step} — {outcome}", value.step
                    )
                self.step_selector.setCurrentIndex(self.controller.step_index)
            self.candidate_selector.clear()
            if step is not None:
                for candidate in step.candidates:
                    marker = " • planned" if candidate.selected else ""
                    self.candidate_selector.addItem(
                        f"#{candidate.rank} object {candidate.object_id} — "
                        f"{candidate.outcome.value}{marker}",
                        candidate.candidate_id,
                    )
                    self.candidate_selector.setItemData(
                        self.candidate_selector.count() - 1,
                        self._candidate_failure_text(candidate), Qt.ItemDataRole.ToolTipRole,
                    )
                active_index = self.candidate_selector.findData(
                    self.controller.active_candidate_id
                )
                if active_index >= 0:
                    self.candidate_selector.setCurrentIndex(active_index)
            self._select_data(self.render_selector, self.controller.render_mode.value)
            self._select_data(self.pose_selector, self.controller.pose_mode.value)
            frame_count = self.controller.timeline_frame_count
            self.timeline.setMaximum(max(frame_count - 1, 0))
            self.timeline.setValue(
                min(self.controller.timeline_frame, max(frame_count - 1, 0))
            )
            self.timeline.setEnabled(frame_count > 0)
            self.video_fps.setEnabled(frame_count > 0)
            active = self.controller.active_candidate
            self.candidate_failure_reason.setText(self._candidate_failure_text(active))
            timed = (active is not None and bool(active.timeline.tracks)
                     and all(track.times is not None for track in active.timeline.tracks))
            self.video_fps.setToolTip(
                "Output frames per second; simulation duration is preserved."
                if timed else
                "This trajectory has no timestamps: one sample is exported per frame, "
                "so FPS changes playback speed."
            )
            self.video_button.setEnabled(frame_count > 0)
            self.timeline_label.setText(
                "No trajectory"
                if frame_count == 0
                else f"Frame {self.timeline.value() + 1} of {frame_count}"
            )
            self.contacts.setChecked(self.controller.show_contacts)
            self.forces.setChecked(self.controller.show_forces)
            enabled = step is not None
            for widget in (
                self.step_selector,
                self.candidate_selector,
                self.render_selector,
                self.pose_selector,
                self.contacts,
                self.forces,
            ):
                widget.setEnabled(enabled)
            self.summary.setText(self._summary())
            self.failure_details.setPlainText(self.controller.search_diagnostics)
            self.tree_button.setEnabled(step is not None)
            if self._tree_dialog is not None:
                self._tree_view.set_step(step)
        finally:
            self._updating = False

    def _show_tree(self):
        from stacking_common.rendering.mcts_tree import MCTSTreeView

        if self._tree_dialog is None:
            self._tree_dialog = QDialog(self)
            self._tree_dialog.setWindowTitle("MCTS search tree")
            self._tree_dialog.resize(1100, 700)
            self._tree_view = MCTSTreeView(self._tree_dialog)
            layout = QVBoxLayout(self._tree_dialog)
            layout.addWidget(self._tree_view)
        self._tree_view.set_step(self.controller.current_step)
        self._tree_dialog.show()
        self._tree_dialog.raise_()

    @staticmethod
    def _candidate_failure_text(candidate) -> str:
        if candidate is None:
            return "No candidate selected."
        if candidate.rejection_reason:
            return candidate.rejection_reason
        if candidate.outcome.value == "succeeded":
            return "None — candidate evaluation succeeded."
        return "Not evaluated yet."

    @staticmethod
    def _select_data(selector: QComboBox, value: str) -> None:
        index = selector.findData(value)
        if index >= 0:
            selector.setCurrentIndex(index)

    def _summary(self) -> str:
        step = self.controller.current_step
        if step is None:
            return "Load a stacking run to inspect search results."
        rejected = sum(step.rejected_expansions.values())
        return (
            f"{len(step.scene.objects)} placed objects · "
            f"{len(step.candidates)} admitted candidates · "
            f"{rejected} generation rejections"
        )

    def environment_visibility(self) -> dict[str, bool]:
        """Return the platform scene layers selected by the operator."""

        return {
            layer: control.isChecked()
            for layer, control in self.environment_controls.items()
        }

    def _step_changed(self, index: int) -> None:
        if self._updating or index < 0:
            return
        self._timeline_render_timer.stop()
        self.controller.select_step(index)
        self.refresh()
        self.state_changed.emit()

    def _candidate_changed(self, index: int) -> None:
        if self._updating or index < 0:
            return
        self._timeline_render_timer.stop()
        self.controller.select_candidate(self.candidate_selector.itemData(index))
        self.refresh()
        self.state_changed.emit()

    def _render_changed(self, index: int) -> None:
        if self._updating or index < 0:
            return
        self.controller.set_render_mode(self.render_selector.itemData(index))
        self.state_changed.emit()

    def _pose_changed(self, index: int) -> None:
        if self._updating or index < 0:
            return
        self._timeline_render_timer.stop()
        self.controller.set_pose_mode(self.pose_selector.itemData(index))
        self.refresh()
        self.state_changed.emit()

    def _timeline_changed(self, frame: int) -> None:
        if self._updating or self.controller.timeline_frame_count == 0:
            return
        self.controller.set_timeline_frame(frame)
        self.timeline_label.setText(
            f"Frame {frame + 1} of {self.controller.timeline_frame_count}"
        )
        self._timeline_render_timer.start()

    def _layers_changed(self) -> None:
        if self._updating:
            return
        self.controller.set_diagnostic_layers(
            contacts=self.contacts.isChecked(),
            forces=self.forces.isChecked(),
        )
        self.state_changed.emit()

    def _environment_changed(self) -> None:
        if not self._updating:
            self.state_changed.emit()
