"""Configure and supervise offline planning without blocking the console."""

import os
from pathlib import Path
import signal
import sqlite3
import sys

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit,
    QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)
from stacking_common.artifacts import StackRunStore
from stacking_common.workflows.planning_output import PlanningRunOutput
from stacking_excavator import ExcavatorApplication
from stacking_excavator.planning.resume import load_resume_checkpoint


class PlanningRunDialog(QDialog):
    run_available = Signal(object)

    def __init__(self, repository, source=None, parent=None):
        super().__init__(parent, Qt.WindowType.Window | Qt.WindowType.WindowMinMaxButtonsHint
                         | Qt.WindowType.WindowCloseButtonHint)
        self.repository = Path(repository).resolve()
        self.source = None if source is None else Path(source)
        self.destination = None
        self._run_output = None
        self._announced_path = None
        self._run_timer = QTimer(self)
        self._run_timer.setInterval(500)
        self._run_timer.timeout.connect(self._check_run_available)
        self.setWindowTitle("Resume excavator planning" if source else "New excavator planning run")
        self.resize(850, 650)
        layout = QVBoxLayout(self)
        self.inputs = QWidget()
        form = QFormLayout(self.inputs)
        self.pick_path = QLineEdit()
        self.pick_path.setPlaceholderText("Optional for search-only unless pose_from_scan is enabled")
        self.output_path = QLineEdit()
        self.config_path = QLineEdit(str(self.repository / "apps/excavator/config/mcts_cpu.yml"))
        if self.source:
            self.output_path.setText(str(self.source.with_name(self.source.stem + "-resumed.stack")))
            metadata = StackRunStore.info(self.source).metadata
            self.config_path.setText(metadata.get("config", {}).get("path", self.config_path.text()))
        else:
            self._file_row(form, "Recorded pickup poses", self.pick_path, "Pickup poses (*.json *.pkl *.pickle)")
        self._file_row(form, "Save new run", self.output_path, "Stacking run (*.stack)", save=True)
        self.overwrite = QCheckBox("Overwrite existing run on successful completion")
        form.addRow(self.overwrite)
        self._file_row(form, "MCTS profile", self.config_path, "YAML (*.yml *.yaml)")
        self.steps = QSpinBox()
        self.steps.setRange(1, 1000)
        self.steps.setValue(1)
        self.wall_seconds = QSpinBox()
        self.wall_seconds.setRange(1, 86400)
        self.wall_seconds.setValue(600)
        self.wall_seconds.setSuffix(" s")
        self.search_only = QCheckBox("Placement search only")
        self.search_only.setEnabled(self.source is None)
        if self.source:
            self.search_only.setChecked(metadata.get("mode") == "search")
        form.addRow("Additional steps" if source else "Steps", self.steps)
        form.addRow("Run time limit", self.wall_seconds)
        self.pose_from_scan = QComboBox()
        self.pose_from_scan.addItem("Use configuration (no override)", None)
        self.pose_from_scan.addItem("true", True)
        self.pose_from_scan.addItem("false", False)
        if self.source:
            checkpoint = load_resume_checkpoint(self.source)
            context = StackRunStore.load_run_context(self.source, checkpoint.context_sha256)
            scan_enabled = context.profiles["pose_from_scan"]
            self.pose_from_scan.setCurrentIndex(self.pose_from_scan.findData(scan_enabled))
        self.pose_from_scan.setEnabled(source is None)
        self.pose_from_scan.setToolTip(
            "Use configuration leaves orientation.pose_from_scan unchanged; true/false "
            "overrides it for this new run without editing YAML. "
            "Controls MCTS orientation initialization and filtering; motion planning uses "
            "recorded pickup poses in either mode. Resume preserves the recorded value.")
        form.addRow("Pose from scan (recorded)" if source else "Pose from scan override", self.pose_from_scan)
        form.addRow(self.search_only)
        layout.addWidget(self.inputs)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        layout.addWidget(self.log, 1)
        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start planning")
        self.stop_button = QPushButton("Close")
        self.open_button = QPushButton("Open resulting run")
        self.open_button.setEnabled(False)
        for button in (self.start_button, self.stop_button, self.open_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.reject)
        self.open_button.clicked.connect(lambda: self.run_available.emit(self.destination))
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(self.repository))
        environment = QProcessEnvironment.systemEnvironment()
        sources = [self.repository / "packages/stacking-common/src", self.repository / "apps/excavator/src"]
        environment.insert("PYTHONPATH", os.pathsep.join([*map(str, sources), environment.value("PYTHONPATH")]))
        self.process.setProcessEnvironment(environment)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)

    def _file_row(self, form, label, field, file_filter, *, save=False):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field)
        button = QPushButton("Browse…")
        layout.addWidget(button)
        def choose():
            picker = QFileDialog.getSaveFileName if save else QFileDialog.getOpenFileName
            filename, _ = picker(self, label, field.text() or str(self.repository), file_filter)
            if filename:
                field.setText(filename)
        button.clicked.connect(choose)
        form.addRow(label, row)

    @property
    def running(self):
        return self.process.state() != QProcess.ProcessState.NotRunning

    def arguments(self):
        if not self.output_path.text().strip():
            raise ValueError("Choose a new output .stack path")
        destination = Path(self.output_path.text()).expanduser().resolve()
        if destination.suffix != ".stack":
            destination = destination.with_suffix(".stack")
        output = PlanningRunOutput(destination, overwrite=self.overwrite.isChecked(), source=self.source)
        config = Path(self.config_path.text()).expanduser().resolve()
        if not config.is_file():
            raise ValueError("Choose an existing MCTS YAML profile")
        arguments = ["-u", "-m", "stacking_excavator.planning.cli",
            "--repository", str(self.repository), "--save-run", str(output.path),
            "--config", str(config), "--steps", str(self.steps.value()),
            "--max-run-time", str(self.wall_seconds.value())]
        if self.source:
            arguments += ["--resume-run", str(self.source)]
        else:
            scan_override = self.pose_from_scan.currentData()
            if scan_override is not None:
                arguments.append("--pose-from-scan" if scan_override else "--no-pose-from-scan")
            scan_enabled = (ExcavatorApplication.from_repository(self.repository).pose_from_scan
                            if scan_override is None else scan_override)
            pickup_text = self.pick_path.text().strip()
            if pickup_text:
                pickup = Path(pickup_text).expanduser().resolve()
                if not pickup.is_file():
                    raise ValueError("Choose an existing pickup pose file")
                arguments += ["--pick-pose-data", str(pickup)]
            elif not self.search_only.isChecked() or scan_enabled:
                raise ValueError("Choose recorded pickup poses before starting")
            if self.search_only.isChecked():
                arguments.append("--search-only")
        self._run_output = output
        self.destination = output.path
        return arguments

    def start(self):
        if self.running:
            return
        try:
            arguments = self.arguments()
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "Cannot start planning", str(error))
            return
        self.log.clear()
        if self._run_output is not None and self._run_output.path != self._run_output.destination:
            self.log.appendPlainText(f"Existing run retained until success. Working output: {self.destination}")
        self.inputs.setEnabled(False)
        self.start_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.stop_button.setText("Stop planning")
        self._announced_path = None
        self._run_timer.start()
        self.process.start(sys.executable, arguments)

    def _check_run_available(self, *, final=False):
        path = self.destination
        if path is None or not path.is_file() or (not final and path == self._announced_path):
            return
        try:
            StackRunStore.info(path)
            StackRunStore.load_mcts(path)
            # The viewer requires context, committed separately from the header.
            if not StackRunStore.load_run_contexts(path):
                return
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
            return
        self._announced_path = path
        self._run_timer.stop()
        self.open_button.setEnabled(True)
        self.run_available.emit(path)

    def _read_output(self):
        self.log.insertPlainText(bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace"))
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _finished(self, code, status):
        self._run_timer.stop()
        self._read_output()
        self.log.appendPlainText(f"\nPlanning process exited: {code}")
        self.inputs.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setText("Close")
        self.stop_button.setEnabled(True)
        if self._run_output is not None:
            try:
                self.destination = self._run_output.finish(code == 0 and status != QProcess.ExitStatus.CrashExit)
            except (OSError, ValueError, sqlite3.Error) as error:
                self.log.appendPlainText(f"Could not replace existing run: {error}")
            self.log.appendPlainText(f"Result path: {self.destination}")
        self._check_run_available(final=True)

    def _error(self, error):
        self.log.appendPlainText(self.process.errorString())
        if error == QProcess.ProcessError.FailedToStart:
            self._finished(-1, None)

    def reject(self):
        if self.running:
            pid = self.process.processId()
            if pid:
                try:
                    # The CLI supervisor owns/reaps its native worker group.
                    os.kill(pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                self.stop_button.setEnabled(False)
                self.stop_button.setText("Stopping…")
            return
        super().reject()

    def closeEvent(self, event):
        if self.running:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)
