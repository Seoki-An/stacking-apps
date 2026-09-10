"""Motion-plan inspection for the operations console."""

import time

import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .controller import OperationsController


class PlanView(QWidget):
    """Inspect world-frame planning inputs and normalized motion segments."""

    state_changed = Signal()

    def __init__(self, controller: OperationsController, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._updating = False
        self._displayed_plan_key = None
        layout = QVBoxLayout(self)

        title = QLabel("Plan")
        title.setStyleSheet("font-size: 16px; font-weight: 600")
        layout.addWidget(title)

        overview = QGroupBox("Planning attempt")
        form = QFormLayout(overview)
        self.attempt_selector = QComboBox()
        self.status = QLabel("No plan recorded")
        self.object_id = QLabel("—")
        self.frame_id = QLabel("—")
        self.pick_pose = QLabel("—")
        self.place_pose = QLabel("—")
        for label in (self.pick_pose, self.place_pose):
            label.setWordWrap(True)
        form.addRow("Attempt", self.attempt_selector)
        form.addRow("Status", self.status)
        form.addRow("Object", self.object_id)
        form.addRow("Frame", self.frame_id)
        form.addRow("Pick", self.pick_pose)
        form.addRow("Place", self.place_pose)
        layout.addWidget(overview)

        motion = QGroupBox("Motion segments")
        motion_layout = QVBoxLayout(motion)
        self.playback_scope_selector = QComboBox()
        self.playback_scope_selector.addItem("Whole step — all motion segments", "step")
        self.playback_scope_selector.addItem("Selected segment", "segment")
        motion_layout.addWidget(self.playback_scope_selector)
        self.segments = QListWidget()
        motion_layout.addWidget(self.segments)
        timeline = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.sample_slider = QSlider(Qt.Orientation.Horizontal)
        self.sample_label = QLabel("No trajectory")
        timeline.addWidget(self.play_button)
        timeline.addWidget(self.sample_slider, 1)
        timeline.addWidget(self.sample_label)
        motion_layout.addLayout(timeline)
        self.robot_geometry_selector = QComboBox()
        self.robot_geometry_selector.addItem("Visual meshes", "visual")
        self.robot_geometry_selector.addItem("Collision meshes", "collision")
        self.robot_geometry_selector.addItem("Visual + collision wireframe", "visual_collision")
        motion_layout.addWidget(self.robot_geometry_selector)
        layout.addWidget(motion)

        self.details = QLabel(
            "Load a .stack run containing placement-plan records."
        )
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        layout.addStretch()

        self.attempt_selector.currentIndexChanged.connect(
            self._attempt_changed
        )
        self.segments.currentRowChanged.connect(self._segment_changed)
        self.sample_slider.valueChanged.connect(self._sample_changed)
        self.play_button.clicked.connect(self._toggle_playback)
        self.playback_scope_selector.currentIndexChanged.connect(self._scope_changed)
        self.robot_geometry_selector.currentIndexChanged.connect(self._geometry_changed)
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(16)
        self.playback_timer.timeout.connect(self._advance_playback)
        self.refresh()

    def refresh(self) -> None:
        key = (
            id(self.controller.artifact), self.controller.step_index,
            self.controller.plan_attempt, id(self.controller.current_plan),
        )
        if key == self._displayed_plan_key:
            # A render triggered by playback must not rebuild widgets or stop
            # the same timer that advanced the sample.
            was_updating = self._updating
            self._updating = True
            try:
                self.segments.setCurrentRow(self.controller.plan_segment_index)
                self._configure_timeline()
            finally:
                self._updating = was_updating
            return
        self._displayed_plan_key = key
        self._stop_playback()
        self._updating = True
        try:
            plans = self.controller.current_plans
            self.attempt_selector.clear()
            for record in plans:
                self.attempt_selector.addItem(
                    f"Attempt {record.attempt} — {record.result.status.value}",
                    record.attempt,
                )
            if self.controller.plan_attempt is not None:
                index = self.attempt_selector.findData(
                    self.controller.plan_attempt
                )
                if index >= 0:
                    self.attempt_selector.setCurrentIndex(index)
            self.attempt_selector.setEnabled(bool(plans))
            self._show(self.controller.current_plan)
        finally:
            self._updating = False

    def _show(self, record) -> None:
        self.segments.clear()
        if record is None:
            self.status.setText("No plan recorded for this search step")
            self.object_id.setText("—")
            self.frame_id.setText("—")
            self.pick_pose.setText("—")
            self.place_pose.setText("—")
            self.details.setText(
                "Planning attempts are stored incrementally in the same "
                ".stack run as their MCTS step."
            )
            self._configure_timeline()
            return

        problem = record.problem
        result = record.result
        self.status.setText(result.status.value)
        self.object_id.setText(str(problem.action.object_id))
        self.frame_id.setText(problem.frame_id)
        self.pick_pose.setText(_format_pose(problem.pick_pose))
        self.place_pose.setText(_format_pose(problem.place_pose))
        for segment in result.segments:
            timing = "path only" if segment.times is None else "timed"
            opening = (
                ""
                if segment.openings is None
                else (
                    f" · opening {segment.openings[0]:.3f}"
                    f"→{segment.openings[-1]:.3f}"
                )
            )
            self.segments.addItem(
                f"{segment.name} · {segment.positions.shape[0]} samples · "
                f"{segment.positions.shape[1]} joints · {timing}{opening}"
            )
        if result.segments:
            self.segments.setCurrentRow(
                min(self.controller.plan_segment_index, len(result.segments) - 1)
            )
        self._configure_timeline()
        if result.succeeded:
            score = "—" if result.score is None else f"{result.score:.4g}"
            self.details.setText(
                f"{len(result.segments)} segments · score {score} · "
                f"candidate {record.candidate_id or 'not recorded'}"
            )
        else:
            retryable = bool(result.diagnostics.get("retryable", False))
            self.details.setText(
                f"{result.reason} · retryable={'yes' if retryable else 'no'}"
            )

    def _attempt_changed(self, index: int) -> None:
        if self._updating or index < 0:
            return
        self.controller.select_plan_attempt(
            self.attempt_selector.itemData(index)
        )
        self.refresh()
        self.state_changed.emit()

    def _segment_changed(self, index: int) -> None:
        if self._updating or index < 0:
            return
        self.controller.select_plan_segment(index)
        self.controller.set_plan_interpolation(0.0)
        self._last_tick = time.monotonic()
        self._configure_timeline()
        self.state_changed.emit()

    def _sample_changed(self, sample: int) -> None:
        if self._updating or self.controller.plan_sample_count == 0:
            return
        if self.playback_scope_selector.currentData() == "step":
            remaining = sample
            for index, segment in enumerate(self.controller.current_plan.result.segments):
                if remaining < len(segment.positions):
                    if index != self.controller.plan_segment_index:
                        self.controller.select_plan_segment(index)
                    sample = remaining
                    break
                remaining -= len(segment.positions)
        self.controller.set_plan_sample(sample)
        self._last_tick = time.monotonic()
        self._configure_timeline()
        self.state_changed.emit()

    def _timeline_count(self) -> int:
        record = self.controller.current_plan
        if record is None:
            return 0
        if self.playback_scope_selector.currentData() == "step":
            return sum(len(segment.positions) for segment in record.result.segments)
        return self.controller.plan_sample_count

    def _timeline_index(self) -> int:
        index = self.controller.plan_sample_index
        record = self.controller.current_plan
        if record is not None and self.playback_scope_selector.currentData() == "step":
            index += sum(
                len(segment.positions)
                for segment in record.result.segments[:self.controller.plan_segment_index]
            )
        return index

    def _scope_changed(self, _index: int) -> None:
        self._stop_playback()
        self._configure_timeline()

    def _configure_timeline(self) -> None:
        count = self._timeline_count()
        was_updating = self._updating
        self._updating = True
        try:
            self.segments.setCurrentRow(self.controller.plan_segment_index)
            self.sample_slider.setRange(0, max(count - 1, 0))
            self.sample_slider.setValue(
                min(self._timeline_index(), max(count - 1, 0))
            )
            self.sample_slider.setEnabled(count > 0)
            self.play_button.setEnabled(count > 1)
            self.sample_label.setText(
                "No trajectory"
                if count == 0
                else f"{self.sample_slider.value() + 1}/{count}"
            )
        finally:
            self._updating = was_updating

    def _toggle_playback(self) -> None:
        if self.playback_timer.isActive():
            self._stop_playback()
            return
        if self.play_button.isEnabled():
            if self._timeline_index() == self._timeline_count() - 1:
                self.sample_slider.setValue(0)
            self.play_button.setText("Pause")
            self._last_tick = time.monotonic()
            self.playback_timer.start()

    def _geometry_changed(self, index: int) -> None:
        self.controller.set_robot_geometry_mode(self.robot_geometry_selector.itemData(index))
        self.state_changed.emit()

    def _advance_playback(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_tick
        self._last_tick = now
        fraction = getattr(self.controller, "plan_sample_fraction", 0.0)
        while True:
            segment = self.controller.current_plan_segment
            sample = self.controller.plan_sample_index
            if self._timeline_index() >= self._timeline_count() - 1:
                self._stop_playback()
                break
            if sample + 1 >= len(segment.positions):
                # Segment boundaries may change attachment/opening semantics.
                # Do not invent a path connecting two separate segments.
                duration = 0.25
            elif segment.times is not None:
                duration = max(float(segment.times[sample + 1] - segment.times[sample]), 1e-6)
            else:
                # Untimed paths use a display-only nominal 0.6 rad/s speed.
                duration = max(float(np.max(np.abs(segment.positions[sample + 1] - segment.positions[sample]))) / 0.6, 0.1)
            remaining = (1.0 - fraction) * duration
            if elapsed < remaining:
                self.controller.set_plan_interpolation(fraction + elapsed / duration)
                self.state_changed.emit()
                break
            elapsed -= remaining
            self.sample_slider.setValue(self._timeline_index() + 1)
            fraction = 0.0
        self._last_tick = now

    def _stop_playback(self) -> None:
        timer = getattr(self, "playback_timer", None)
        if timer is not None:
            timer.stop()
        button = getattr(self, "play_button", None)
        if button is not None:
            button.setText("Play")


def _format_pose(pose) -> str:
    return (
        f"xyz [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}] · "
        f"xyzw [{pose[3]:.3f}, {pose[4]:.3f}, "
        f"{pose[5]:.3f}, {pose[6]:.3f}]"
    )
