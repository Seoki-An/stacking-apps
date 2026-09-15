"""Supervised saved-plan execution with explicit gates and recovery evidence."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QFileDialog, QPlainTextEdit, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from stacking_common.artifacts import StackRunStore
from stacking_tabletop.execution.saved_plan import load_execution_run
from stacking_tabletop.execution.recovery import reconcile_saved_run

from .execution_worker import ExecutionWorker


class ExecutionView(QWidget):
    state_changed = Signal()

    def __init__(self, repository=None, parent=None, *, worker_factory=ExecutionWorker):
        super().__init__(parent)
        self.repository = repository
        self.worker_factory = worker_factory
        self.can_start = lambda: True
        self.worker = self.saved = self.gate = self.scene = None
        self.path = None
        self._paused = False
        self._aborting = False
        self._scan_pending = False
        self._scan_positions = None
        layout = QVBoxLayout(self)
        self.status = QLabel("Open a .stack run to validate saved execution.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.start_button = QPushButton("Start supervised execution")
        self.start_button.clicked.connect(self.start)
        layout.addWidget(self.start_button)
        self.gate_label = QLabel()
        self.gate_label.setWordWrap(True)
        layout.addWidget(self.gate_label)
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
        self._controls()

    @property
    def running(self):
        return self.worker is not None and self.worker.isRunning()

    def load(self, path):
        if self.running:
            raise ValueError("finish execution before changing the run")
        self.path, self.saved, self.scene = path, None, None
        self.gate = None
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
        except Exception as error:
            self.saved = self.scene = None
            self.status.setText(f"Execution unavailable: {error}")
        self._controls()
        self.state_changed.emit()

    def _controls(self):
        ready = self.saved is not None
        active = self.running and not self._aborting
        checkpoint = StackRunStore.load_execution_checkpoint(self.path) if ready else None
        self.start_button.setEnabled(ready and not self.running and self.can_start() and bool(checkpoint.pending_steps)
                                     and not checkpoint.requires_reconciliation)
        self.reconcile_button.setEnabled(ready and not self.running and checkpoint.requires_reconciliation)
        self.continue_button.setEnabled(active and self.gate is not None and not self._paused and not self._scan_pending)
        self.manual_controls.setEnabled(active and self.gate is not None and self.gate.manual and not self._paused)
        self.scan_controls.setEnabled(active and self.gate is not None and self.gate.name == 'place'
                                      and not self._paused and not self._scan_pending)
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
        self.worker = self.worker_factory(self.saved, self)
        self.worker.repository = self.repository
        self.worker.scan_prepared.connect(self.prepare_scan_preview)
        self.worker.gate_changed.connect(self.set_gate)
        self.worker.telemetry.connect(self.receive_telemetry)
        self.worker.event_received.connect(self.receive_event)
        self.worker.plan_selected.connect(self.select_plan)
        self.worker.failed.connect(self.status.setText)
        self.worker.completed.connect(self._completed)
        self.worker.finished.connect(self._finished)
        self.status.setText("Connecting to the existing UR5e and gripper controllers…")
        self._paused = False
        self._aborting = False
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
        self._controls()

    def set_gate(self, gate):
        if gate is None or gate.name != 'scan_review':
            self._scan_positions = None
            self.scan_preview.setEnabled(False)
            if self.scene is not None:
                self.scene.scan_preview = None
                self.state_changed.emit()
        self.gate = gate
        self._scan_pending = False
        self.gate_label.setText("" if gate is None else f"Step {gate.step}: {gate.message}")
        self._controls()

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
        self.telemetry_label.setText(f"Measured joints (°): {joints}\nGripper: {value['gripper_raw']:.0f} counts; hold={value['holding']}\nObject pose: kinematic estimate")
        self.scene.telemetry = value
        self.state_changed.emit()

    def receive_event(self, event):
        self.history.appendPlainText(f"{event.sequence}: {event.status.value} · {event.diagnostics.get('phase', '')} · {event.reason or ''}")
        if 'scan_view' in event.diagnostics:
            self.scan_label.setText(f"Captured view {event.diagnostics['scan_view']} / {event.diagnostics['scan_view_count']}\n"
                                   f"{event.diagnostics['capture_directory']}")
        if 'scan_completed' in event.diagnostics:
            self.scan_label.setText(f"Scan saved: {event.diagnostics['scan_completed']}")
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
