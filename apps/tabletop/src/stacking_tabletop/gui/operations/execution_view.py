"""Supervised saved-plan execution with explicit gates and recovery evidence."""

from PySide6.QtCore import Qt, QTimer, Signal, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QFileDialog, QPlainTextEdit, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from stacking_common.artifacts import StackRunStore
from stacking_tabletop.execution.saved_plan import load_execution_run
from stacking_tabletop.execution.recovery import (
    accept_released_placement,
    pending_released_stack_capture,
    reconcile_saved_run,
    released_placement_recovery,
    retry_released_stack_capture,
)

from .execution_worker import ExecutionWorker


def connect_live_state():
    """Open a read-only view of the existing controller telemetry."""
    from stacking_tabletop.ros.hardware import TabletopRosTransport
    return TabletopRosTransport(node_name="tabletop_execution_preview")


class ExecutionView(QWidget):
    state_changed = Signal()
    refined_run_loaded = Signal(str)

    def __init__(self, repository=None, parent=None, *, worker_factory=ExecutionWorker,
                 preview_transport_factory=connect_live_state):
        super().__init__(parent)
        self.repository = repository
        self.worker_factory = worker_factory
        self.preview_transport_factory = preview_transport_factory
        self.can_start = lambda: True
        self.worker = self.saved = self.gate = self.scene = None
        self._preview_transport = None
        self._preview_requested = False
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(100)
        self._preview_timer.timeout.connect(self._poll_live_state)
        self._motion_samples = ()
        self._motion_preview_timer = QTimer(self)
        self._motion_preview_timer.setInterval(33)
        self._motion_preview_timer.timeout.connect(self._advance_motion_preview)
        self.path = None
        self._paused = False
        self._aborting = False
        self._scan_pending = False
        self._scan_positions = None
        self._stack_viewer = None
        self._released_recovery = None
        self._refined_path = None
        layout = QVBoxLayout(self)
        self.status = QLabel("Open a .stack run to validate saved execution.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.start_button = QPushButton("Start supervised execution")
        self.start_button.clicked.connect(self.start)
        self.stack_capture = QCheckBox('Collect stacked-scene PCD with wrist RGB after each placement')
        self.stack_capture.setChecked(True)
        layout.addWidget(self.stack_capture)
        self.stack_viewer_button = QPushButton('Open stacked-scene PCD')
        self.stack_viewer_button.setEnabled(False)
        self.stack_viewer_button.clicked.connect(self.open_stack_viewer)
        layout.addWidget(self.stack_viewer_button)
        layout.addWidget(self.start_button)
        self.gate_label = QLabel()
        self.gate_label.setWordWrap(True)
        layout.addWidget(self.gate_label)
        self.motion_review_controls = QWidget()
        row = QHBoxLayout(self.motion_review_controls)
        self.motion_play_button = QPushButton("Play saved motion")
        self.motion_play_button.clicked.connect(self._toggle_motion_preview)
        self.motion_slider = QSlider(Qt.Orientation.Horizontal)
        self.motion_slider.valueChanged.connect(self.preview_motion_sample)
        row.addWidget(self.motion_play_button)
        row.addWidget(self.motion_slider)
        layout.addWidget(self.motion_review_controls)
        self.motion_label = QLabel("Saved-motion review appears at the first execution gate.")
        self.motion_label.setWordWrap(True)
        layout.addWidget(self.motion_label)
        self.continue_button = QPushButton("Continue")
        self.continue_button.clicked.connect(self.approve)
        layout.addWidget(self.continue_button)
        row = QHBoxLayout()
        self.hold_button = QPushButton("Hold")
        self.hold_button.clicked.connect(self.toggle_hold)
        self.abort_button = QPushButton("Abort")
        self.abort_button.clicked.connect(self.abort)
        row.addWidget(self.hold_button)
        row.addWidget(self.abort_button)
        layout.addLayout(row)
        self.manual_controls = QWidget()
        row = QHBoxLayout(self.manual_controls)
        self.joint = QComboBox()
        self.joint.addItems([f"Joint {i}" for i in range(1, 7)])
        row.addWidget(self.joint)
        for label, direction in (("−0.5°", -1), ("+0.5°", 1), ("Reset", 0)):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, d=direction: self.adjust(d))
            row.addWidget(button)
        layout.addWidget(self.manual_controls)
        self.scan_controls = QWidget()
        row = QHBoxLayout(self.scan_controls)
        self.scan_kind = QComboBox()
        self.scan_kind.addItems(['wrist', 'pivot'])
        self.scan_camera = QComboBox()
        self.scan_camera.addItems(['primary', 'secondary'])
        self.scan_button = QPushButton('Prepare held-object scan…')
        self.scan_button.clicked.connect(self.request_scan)
        for widget in (self.scan_kind, self.scan_camera, self.scan_button):
            row.addWidget(widget)
        layout.addWidget(self.scan_controls)
        self.scan_label = QLabel('Scanning is available at the held-object placement review gate.')
        self.scan_label.setWordWrap(True)
        layout.addWidget(self.scan_label)
        self.scan_preview = QSlider(Qt.Orientation.Horizontal)
        self.scan_preview.valueChanged.connect(self.preview_scan_pose)
        self.scan_preview.setEnabled(False)
        layout.addWidget(self.scan_preview)
        self.telemetry_label = QLabel("Measured robot state appears after connection.\nObject pose is a kinematic estimate; inspect the physical grasp.")
        self.telemetry_label.setWordWrap(True)
        layout.addWidget(self.telemetry_label)
        self.history = QPlainTextEdit()
        self.history.setReadOnly(True)
        self.history.setMaximumBlockCount(300)
        layout.addWidget(self.history)
        self.reconcile_button = QPushButton("Record physical reconciliation…")
        self.reconcile_button.clicked.connect(self.reconcile)
        layout.addWidget(self.reconcile_button)
        self.retry_stack_capture_button = QPushButton("Retry stacked-scene capture…")
        self.retry_stack_capture_button.clicked.connect(self.retry_stack_capture)
        layout.addWidget(self.retry_stack_capture_button)
        self.accept_placement_button = QPushButton("Accept already released placement…")
        self.accept_placement_button.clicked.connect(self.accept_placed_step)
        layout.addWidget(self.accept_placement_button)
        self._controls()

    @property
    def running(self):
        return self.worker is not None and self.worker.isRunning()

    def load(self, path):
        if self.running:
            raise ValueError("finish execution before changing the run")
        self._set_motion_review(False)
        self._close_live_preview()
        self.path, self.saved, self.scene = path, None, None
        self.gate = None
        self._stack_viewer = None
        self._released_recovery = None
        self.stack_viewer_button.setEnabled(False)
        self._scan_pending = False
        self._scan_positions = None
        self.scan_preview.setEnabled(False)
        self.scan_label.setText('Scanning is available at the held-object placement review gate.')
        self._aborting = self._paused = False
        self.gate_label.clear()
        self.telemetry_label.setText("Measured robot state appears after connection.\nObject pose is a kinematic estimate; inspect the physical grasp.")
        self.history.clear()
        try:
            self.saved = load_execution_run(path, self.repository)
            from .execution_scene import ExecutionScene
            self.scene = ExecutionScene(self.saved, self.repository)
            checkpoint = StackRunStore.load_execution_checkpoint(path)
            for event in StackRunStore.load_execution_events(path):
                self.scene.event(event)
            self.status.setText(f"Validated {len(self.saved.plans)} saved steps; {len(checkpoint.completed_steps)} completed.")
            if checkpoint.requires_reconciliation:
                self.status.setText("An earlier attempt requires physical reconciliation before retry.")
                self._released_recovery = released_placement_recovery(path)
                if self._released_recovery is not None:
                    self.status.setText(
                        f"Step {self._released_recovery.step} has a durably recorded release. "
                        "Retry only the failed stacked-scene capture, or explicitly skip it."
                    )
            elif checkpoint.pending_steps:
                from stacking_tabletop.execution.scene_refinement import pending_scene_refinement
                pending = pending_scene_refinement(path)
                if pending is not None:
                    self.status.setText(
                        'An accepted stacked-scene observation requires refinement. '
                        'Retry refinement without robot motion; the old remaining '
                        'motions cannot execute.'
                    )
                recovery = pending_released_stack_capture(
                    path, checkpoint.pending_steps[0]
                )
                if recovery is not None:
                    self.status.setText(
                        f"Step {recovery.step} is ready to resume at stacked-scene "
                        "capture; pickup, placement, and release will not repeat."
                    )
        except Exception as error:
            self.saved = self.scene = None
            self.status.setText(f"Execution unavailable: {error}")
        self._controls()
        self.state_changed.emit()
        if self._preview_requested:
            self._open_live_preview()

    def set_live_preview_enabled(self, enabled):
        """Observe ROS state while this tab is visible without commanding hardware."""
        self._preview_requested = bool(enabled)
        if self._preview_requested:
            self._open_live_preview()
        else:
            self._close_live_preview()

    def _open_live_preview(self):
        if self.saved is None or self.running or self._preview_transport is not None:
            return
        try:
            self._preview_transport = self.preview_transport_factory()
        except Exception as error:
            self.telemetry_label.setText(f"Live robot preview unavailable: {error}")
            return
        self.telemetry_label.setText("Waiting for live UR5e joint and tool telemetry…")
        self._preview_timer.start()
        self._poll_live_state()

    def _close_live_preview(self):
        self._preview_timer.stop()
        transport, self._preview_transport = self._preview_transport, None
        if transport is not None:
            try:
                transport.close()
            except Exception as error:
                self.telemetry_label.setText(f"Live robot preview cleanup failed: {error}")

    def _poll_live_state(self):
        transport = self._preview_transport
        if transport is None or self.scene is None:
            return
        try:
            joints = transport.feedback.joint_positions()
            hand = transport.feedback.gripper_state()
        except Exception as error:
            self.telemetry_label.setText(f"Waiting for live robot telemetry: {error}")
            return
        self.receive_telemetry({
            "joints": joints,
            "gripper_raw": hand.position,
            "holding": hand.holding,
            "object_pose": None,
        })

    def _controls(self):
        ready = self.saved is not None
        active = self.running and not self._aborting
        self.stack_capture.setEnabled(not self.running)
        checkpoint = StackRunStore.load_execution_checkpoint(self.path) if ready else None
        from stacking_tabletop.execution.scene_refinement import pending_scene_refinement
        pending = pending_scene_refinement(self.path) if ready else None
        scan_retry = (pending is not None
                      and pending_released_stack_capture(self.path, pending['step']) is not None)
        refinement_pending = pending is not None and not scan_retry
        self.start_button.setText('Retry scene refinement (no robot motion)'
                                  if refinement_pending else 'Retry stacked-scene capture'
                                  if scan_retry else 'Start supervised execution')
        self.start_button.setEnabled(ready and not self.running and self.can_start() and bool(checkpoint.pending_steps)
                                     and not checkpoint.requires_reconciliation)
        self.reconcile_button.setEnabled(ready and not self.running and checkpoint.requires_reconciliation)
        self.accept_placement_button.setEnabled(
            ready and not self.running and checkpoint.requires_reconciliation
            and self._released_recovery is not None
        )
        self.retry_stack_capture_button.setEnabled(
            ready and not self.running and checkpoint.requires_reconciliation
            and self._released_recovery is not None
        )
        self.continue_button.setEnabled(active and self.gate is not None and not self._paused and not self._scan_pending)
        self.manual_controls.setEnabled(active and self.gate is not None and self.gate.manual and not self._paused)
        self.scan_controls.setEnabled(active and self.gate is not None and self.gate.name == 'place'
                                      and not self._paused and not self._scan_pending)
        self.motion_review_controls.setEnabled(
            active and self.gate is not None and self.gate.name == "review"
            and bool(self._motion_samples) and not self._paused
        )
        self.hold_button.setEnabled(active)
        self.abort_button.setEnabled(self.running)

    def start(self):
        if not self.start_button.isEnabled() or self.running or not self.can_start():
            return
        try:
            self.saved.verify_assets()
        except Exception as error:
            self.status.setText(str(error))
            self.start_button.setEnabled(False)
            return
        self._close_live_preview()
        self.worker = self.worker_factory(self.saved, self)
        self.worker.repository = self.repository
        self.worker.capture_stack_scene = self.stack_capture.isChecked()
        self.worker.scan_prepared.connect(self.prepare_scan_preview)
        self.worker.gate_changed.connect(self.set_gate)
        self.worker.telemetry.connect(self.receive_telemetry)
        self.worker.event_received.connect(self.receive_event)
        self.worker.plan_selected.connect(self.select_plan)
        self.worker.refined_run_ready.connect(self._refined_run_ready)
        self.worker.refinement_status.connect(self.status.setText)
        self.worker.failed.connect(self.status.setText)
        self.worker.completed.connect(self._completed)
        self.worker.finished.connect(self._finished)
        self.status.setText("Connecting to the existing UR5e and gripper controllers…")
        self._paused = False
        self._aborting = False
        self._refined_path = None
        self.worker.start()
        self._controls()

    def _completed(self, results):
        if results:
            last = results[-1]
            self.status.setText(last.reason or f"Execution {last.status.value}.")
        else:
            self.status.setText("Execution ended before another step was queued.")

    def _finished(self):
        self.set_gate(None)
        self._paused = False
        self.hold_button.setText("Hold")
        self.telemetry_label.setText("Session ended; last received state:\n" + self.telemetry_label.text())
        refined, self._refined_path = self._refined_path, None
        if refined is not None:
            # A queued ready signal only stores the path. Changing the scene
            # and saved command source is allowed only after the worker ends.
            self.load(refined)
            if self.saved is not None:
                self.refined_run_loaded.emit(str(refined))
                self.status.setText(
                    'Refined plan loaded with completed placements preserved. '
                    'Start supervised execution to review the next motion; '
                    'nothing has started automatically.'
                )
        self._controls()
        if self._preview_requested:
            self._open_live_preview()

    def _refined_run_ready(self, path):
        from pathlib import Path
        self._refined_path = Path(path)

    def set_gate(self, gate):
        self._set_motion_review(gate is not None and gate.name == "review")
        if gate is None or gate.name not in ('scan_review', 'stack_scan_review'):
            self._scan_positions = None
            self.scan_preview.setEnabled(False)
            if self.scene is not None:
                self.scene.scan_preview = None
                self.state_changed.emit()
        if gate is not None and gate.name == "start_alignment" and self.scene is not None:
            self.scene.scan_preview = self.saved.planning.robot.home_positions
            self.state_changed.emit()
        self.gate = gate
        self._scan_pending = False
        self.continue_button.setText(
            'Skip identification / continue with current estimate'
            if gate is not None and gate.name == 'place' else
            'Accept SceneID and refine remaining plan'
            if gate is not None and gate.name == 'stack_scene_result'
            and any(plan.step > gate.step for plan in self.saved.plans) else 'Continue'
        )
        self.gate_label.setText("" if gate is None else f"Step {gate.step}: {gate.message}")
        self._controls()

    def _set_motion_review(self, enabled):
        self._stop_motion_preview()
        scene = self.scene
        if not enabled or scene is None:
            self._motion_samples = ()
            self.motion_slider.setRange(0, 0)
            self.motion_label.setText("Saved-motion review appears at the first execution gate.")
            if scene is not None:
                scene.show_motion_review = False
                scene.motion_preview = None
            self.motion_review_controls.setEnabled(False)
            self.state_changed.emit()
            return
        default_opening = float(self.saved.planning.robot.gripper_opening)
        samples = []
        for segment in scene.plan.result.segments:
            for index, joints in enumerate(segment.positions):
                opening = (default_opening if segment.openings is None
                           else float(segment.openings[index]))
                samples.append((segment.name, index, len(segment.positions), joints, opening))
        self._motion_samples = tuple(samples)
        scene.show_motion_review = True
        self.motion_slider.setRange(0, max(0, len(samples) - 1))
        self.motion_slider.setValue(0)
        if samples:
            self.preview_motion_sample(0)
        else:
            scene.motion_preview = None
            self.motion_label.setText("The selected step contains no saved motion samples.")
            self.state_changed.emit()

    def preview_motion_sample(self, index):
        if self.scene is None or not self._motion_samples:
            return
        index = min(max(int(index), 0), len(self._motion_samples) - 1)
        name, local_index, segment_length, joints, opening = self._motion_samples[index]
        self.scene.motion_preview = (joints, opening)
        self.motion_label.setText(
            f"{name}: {local_index + 1}/{segment_length} · saved sample "
            f"{index + 1}/{len(self._motion_samples)}. Green line is the saved tool path."
        )
        self.state_changed.emit()

    def _toggle_motion_preview(self):
        if self._motion_preview_timer.isActive():
            self._stop_motion_preview()
            return
        if not self._motion_samples:
            return
        if self.motion_slider.value() == self.motion_slider.maximum():
            self.motion_slider.setValue(0)
        self.motion_play_button.setText("Pause saved motion")
        self._motion_preview_timer.start()

    def _advance_motion_preview(self):
        value = self.motion_slider.value()
        if value >= self.motion_slider.maximum():
            self._stop_motion_preview()
            return
        self.motion_slider.setValue(value + 1)

    def _stop_motion_preview(self):
        self._motion_preview_timer.stop()
        self.motion_play_button.setText("Play saved motion")

    def approve(self):
        executor = self.worker.executor if self.running else None
        if executor is not None and self.gate is not None and not self._paused:
            if executor.approve(self.gate.token):
                self.continue_button.setEnabled(False)
                self.manual_controls.setEnabled(False)
                self.scan_controls.setEnabled(False)

    def request_scan(self):
        if not self.scan_controls.isEnabled():
            return
        from pathlib import Path
        from datetime import datetime
        from uuid import uuid4
        gate, worker = self.gate, self.worker
        parent = QFileDialog.getExistingDirectory(self, 'Choose parent directory for a new scan')
        if not parent or self.gate is not gate or not self.running or worker.executor is None or self._paused or self._aborting:
            return
        output = Path(parent) / f'scan-step-{gate.step}-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}'
        if worker.executor.request_scan(gate.token, self.scan_kind.currentText(), self.scan_camera.currentText(), output):
            self._scan_pending = True
            self.scan_label.setText(f'Preparing scan and checking the selected camera…\n{output}')
            self._controls()

    def prepare_scan_preview(self, value):
        self._scan_positions = value['positions']
        self.scan_label.setText(f"{value['kind']} scan: {value['capture_count']} views → {value['output']}\n"
                               'Scrub the cyan robot preview to inspect entry, scan and return poses.')
        self.scan_preview.setRange(0, len(self._scan_positions) - 1)
        self.scan_preview.setValue(0)
        self.scan_preview.setEnabled(True)
        self.preview_scan_pose(0)

    def preview_scan_pose(self, index):
        if self.scene is not None and self._scan_positions is not None:
            self.scene.scan_preview = self._scan_positions[index]
            self.state_changed.emit()

    def adjust(self, direction):
        executor = self.worker.executor if self.running else None
        if executor is not None and self.gate is not None and not self._paused:
            executor.adjust(self.gate.token, None if direction == 0 else self.joint.currentIndex(), direction)

    def toggle_hold(self):
        executor = self.worker.executor if self.running else None
        if executor is None or executor.active_command_id is None:
            return
        if self._paused:
            executor.resume(executor.active_command_id)
            self.hold_button.setEnabled(False)
            self.scan_controls.setEnabled(False)
        else:
            executor.hold(executor.active_command_id)
            self._paused = True
            self.hold_button.setText("Resume")
            self.status.setText("Hold requested; gripper operation, if active, must finish first.")
            self._controls()

    def abort(self):
        if self.running:
            self._aborting = True
            self.worker.abort()
            self.status.setText("Abort requested; waiting for worker shutdown. The gripper remains closed if holding.")
            self.continue_button.setEnabled(False)
            self.manual_controls.setEnabled(False)
            self.hold_button.setEnabled(False)
            self.scan_controls.setEnabled(False)

    def select_plan(self, plan):
        self.scene.select_plan(plan)
        self.state_changed.emit()

    def receive_telemetry(self, value):
        import numpy as np
        joints = ", ".join(f"{q:.1f}" for q in np.rad2deg(value["joints"]))
        gripper = value["gripper_raw"]
        gripper_text = "unavailable" if gripper is None else f"{gripper:.0f} counts"
        source = value.get("object_pose_source")
        estimate = (
            "unavailable"
            if value.get("object_pose") is None
            else "identified attachment"
            if source == "identified"
            else "planned grasp attachment"
        )
        self.telemetry_label.setText(
            f"Measured joints (°): {joints}\nGripper: {gripper_text}; "
            f"hold={value['holding']}\nObject pose: {estimate}"
        )
        self.scene.telemetry = value
        self.state_changed.emit()

    def open_stack_viewer(self):
        if self._stack_viewer is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._stack_viewer)))

    def receive_event(self, event):
        self.history.appendPlainText(f"{event.sequence}: {event.status.value} · {event.diagnostics.get('phase', '')} · {event.reason or ''}")
        if 'stack_scene_stage' in event.diagnostics:
            self.scan_label.setText(f"Stacked scene: {event.diagnostics['stack_scene_stage']}\n"
                                   f"{event.diagnostics['stack_scene_directory']}")
        if 'stack_scene_completed' in event.diagnostics:
            from pathlib import Path
            self._stack_viewer = Path(event.diagnostics['stack_scene_viewer'])
            self.stack_viewer_button.setEnabled(self._stack_viewer.is_file())
            identified = event.diagnostics.get('stack_scene_identification')
            message = f"Stacked-scene PCD saved: {event.diagnostics['stack_scene_pcd']}"
            if identified is not None:
                message += f"\nSceneID: {identified['status']} → {identified['path']}"
            self.scan_label.setText(message)
        if 'scan_view' in event.diagnostics:
            self.scan_label.setText(f"Captured view {event.diagnostics['scan_view']} / {event.diagnostics['scan_view_count']}\n"
                                   f"{event.diagnostics['capture_directory']}")
        if 'scan_completed' in event.diagnostics:
            self.scan_label.setText(f"Scan saved: {event.diagnostics['scan_completed']}")
        if 'inhand_preparation_failed' in event.diagnostics:
            self.scan_label.setText(
                'Scan preparation failed: ' + event.diagnostics['inhand_preparation_failed']
                + '\nRetry preparation, or skip identification at the placement review.'
            )
        if 'inhand_identification_started' in event.diagnostics:
            self.scan_label.setText(
                "Identifying the held stone from the completed scan…\n"
                f"{event.diagnostics['inhand_identification_started']}"
            )
        if 'inhand_identification' in event.diagnostics:
            result = event.diagnostics['inhand_identification']
            if result['status'] == 'failed':
                message = f"In-hand identification failed: {result['error']}"
            else:
                correction = (
                    f"{1000.0 * result['correction_translation_m']:.1f} mm / "
                    f"{result['correction_rotation_degrees']:.2f}°"
                )
                message = (
                    f"In-hand identification {result['status']}: {correction}; "
                    f"RMSE {1000.0 * result['rmse_m']:.1f} mm"
                )
            self.scan_label.setText(f"{message}\n{result['report_path']}")
        if event.diagnostics.get("operator_decision") == "resume":
            self._paused = False
            self.hold_button.setText("Hold")
            self._controls()
        self.scene.event(event)
        self.state_changed.emit()

    def reconcile(self):
        if self.running or not self.reconcile_button.isEnabled():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Physical reconciliation")
        layout = QVBoxLayout(dialog)
        checks = [QCheckBox(text) for text in (
            "Robot is stopped; the previous controller session has ended",
            "Pickup and stack match the saved scene for a whole-step retry",
            "Gripper and held object have been checked and reset for pickup",
        )]
        for check in checks:
            layout.addWidget(check)
        notes = QPlainTextEdit()
        notes.setPlaceholderText("Record who checked the hardware and how the scene was restored.")
        layout.addWidget(notes)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        save = buttons.button(QDialogButtonBox.StandardButton.Save)
        save.setEnabled(False)
        def validate():
            save.setEnabled(all(check.isChecked() for check in checks) and bool(notes.toPlainText().strip()))
        for check in checks:
            check.toggled.connect(validate)
        notes.textChanged.connect(validate)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                reconcile_saved_run(self.path, notes.toPlainText(), robot_stopped=checks[0].isChecked(),
                    scene_restored=checks[1].isChecked(), gripper_checked=checks[2].isChecked())
                self.load(self.path)
            except Exception as error:
                self.status.setText(str(error))

    def accept_placed_step(self):
        recovery = self._released_recovery
        if self.running or recovery is None or not self.accept_placement_button.isEnabled():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Accept released placement")
        layout = QVBoxLayout(dialog)
        message = QLabel(
            f"Step {recovery.step} has durable release evidence. This records the "
            "placement as complete and starts the next execution at step "
            f"{recovery.step + 1}; it sends no robot command."
        )
        message.setWordWrap(True)
        layout.addWidget(message)
        checks = [QCheckBox(text) for text in (
            "Robot is stopped; the previous controller session has ended",
            "The released stone is physically at the saved placement",
            "The gripper is empty and the remaining pickup/stack scene is ready for the next step",
        )]
        for check in checks:
            layout.addWidget(check)
        notes = QPlainTextEdit()
        notes.setPlaceholderText("Record who verified the released stone and remaining scene.")
        layout.addWidget(notes)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        save = buttons.button(QDialogButtonBox.StandardButton.Save)
        save.setEnabled(False)

        def validate():
            save.setEnabled(all(check.isChecked() for check in checks)
                            and bool(notes.toPlainText().strip()))

        for check in checks:
            check.toggled.connect(validate)
        notes.textChanged.connect(validate)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                accept_released_placement(
                    self.path, notes.toPlainText(),
                    robot_stopped=checks[0].isChecked(),
                    placed_scene_confirmed=checks[1].isChecked(),
                    gripper_empty=checks[2].isChecked(),
                )
                self.load(self.path)
            except Exception as error:
                self.status.setText(str(error))

    def retry_stack_capture(self):
        recovery = self._released_recovery
        if (self.running or recovery is None
                or not self.retry_stack_capture_button.isEnabled()):
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Retry stacked-scene capture")
        layout = QVBoxLayout(dialog)
        message = QLabel(
            f"Step {recovery.step} has durable release evidence. This will reconnect "
            "to the current robot pose, show the stacked-scene route for review, capture "
            "the scene, and return home. It will not repeat pickup, placement, or release."
        )
        message.setWordWrap(True)
        layout.addWidget(message)
        checks = [QCheckBox(text) for text in (
            "Robot is stopped; the previous controller session has ended",
            "The released stone is physically at the saved placement",
            "The gripper is empty and the remaining scene has not been disturbed",
        )]
        for check in checks:
            layout.addWidget(check)
        notes = QPlainTextEdit()
        notes.setPlaceholderText("Record who verified the released stone, gripper, and scene.")
        layout.addWidget(notes)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        save = buttons.button(QDialogButtonBox.StandardButton.Save)
        save.setEnabled(False)

        def validate():
            save.setEnabled(
                all(check.isChecked() for check in checks)
                and bool(notes.toPlainText().strip())
            )

        for check in checks:
            check.toggled.connect(validate)
        notes.textChanged.connect(validate)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                retry_released_stack_capture(
                    self.path, notes.toPlainText(),
                    robot_stopped=checks[0].isChecked(),
                    placed_scene_confirmed=checks[1].isChecked(),
                    gripper_empty=checks[2].isChecked(),
                )
                self.load(self.path)
                self.start()
            except Exception as error:
                self.status.setText(str(error))

    def closeEvent(self, event):
        self._stop_motion_preview()
        self.set_live_preview_enabled(False)
        super().closeEvent(event)
