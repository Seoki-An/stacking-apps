"""Configure new tabletop plans and supervise offline planning processes."""

import os
from pathlib import Path
import signal
import sqlite3
import sys

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from stacking_common.artifacts import StackRunStore
from stacking_common.workflows.planning_output import PlanningRunOutput


class PlanningRunDialog(QDialog):
    run_available = Signal(object)

    def __init__(self, repository, source=None, destination=None, config=None, parent=None):
        super().__init__(parent, Qt.WindowType.Window | Qt.WindowType.WindowMinMaxButtonsHint
                         | Qt.WindowType.WindowCloseButtonHint)
        self.repository = Path(repository).resolve()
        self.source = None if source is None else Path(source).resolve()
        self.destination = None if destination is None else Path(destination).resolve()
        self.config = Path(config) if config is not None else self.repository / "apps/tabletop/config/mcts_cpu.yml"
        self.can_start = lambda: True
        self._run_output = None
        self._announced_path = None
        self._run_timer = QTimer(self)
        self._run_timer.setInterval(500)
        self._run_timer.timeout.connect(self._check_run_available)
        self.setWindowTitle("Resume tabletop planning" if self.source else "New tabletop planning run")
        self.resize(900, 650)
        layout = QVBoxLayout(self)
        self.inputs = QWidget()
        form = QFormLayout(self.inputs)
        self.config_path = QLineEdit(str(self.config))
        self.output_path = QLineEdit(str(self.destination) if self.destination else "")
        self.pick_path = QLineEdit()
        self.pick_path.setPlaceholderText("Use profile pickup poses; if unset, use its synthetic pickup scene")
        self.object_ids = QLineEdit()
        self.object_ids.setPlaceholderText("Use profile IDs, or enter e.g. 1 2 3")
        self._file_row(form, "MCTS profile", self.config_path, "YAML (*.yml *.yaml)")
        self._file_row(form, "Save new run", self.output_path, "Stacking runs (*.stack)", save=True)
        self.overwrite = QCheckBox("Overwrite existing run on successful completion")
        form.addRow(self.overwrite)
        self._file_row(form, "Trusted pickup poses (optional)", self.pick_path, "Pickup poses (*.pkl *.pickle)")
        form.addRow("Object IDs (optional)", self.object_ids)
        self.steps = QSpinBox()
        self.steps.setRange(1, 1000)
        self.steps.setValue(1)
        self.wall_seconds = QSpinBox()
        self.wall_seconds.setRange(1, 86400)
        self.wall_seconds.setValue(600)
        self.wall_seconds.setSuffix(" s")
        form.addRow("Steps", self.steps)
        form.addRow("Run time limit", self.wall_seconds)
        self.pose_from_scan = QComboBox()
        self.pose_from_scan.addItem("Use configuration (no override)", None)
        self.pose_from_scan.addItem("On", True)
        self.pose_from_scan.addItem("Off", False)
        self.pose_from_scan.setEnabled(self.source is None)
        self.pose_from_scan.setToolTip(
            "Override orientation.pose_from_scan for this run. On uses scan-aligned pickup orientations; "
            "Off uses model principal axes. Recorded pickup poses still drive grasp planning in either mode. "
            "Without a pickup file, the existing synthetic-scene settings apply. Resume preserves the recorded value.")
        form.addRow("Pose from scan override", self.pose_from_scan)
        self.search_only = QCheckBox("Placement search only (omit grasp and motion planning)")
        form.addRow(self.search_only)
        note = QLabel("Plans are computed offline. Simulation and search settings come from the selected profile. "
                      "Synthetic pickup scenes are for software checks; hardware plans need measured pickup poses.")
        note.setWordWrap(True)
        form.addRow(note)
        layout.addWidget(self.inputs)
        # Existing resume callers already select and validate these paths. Keep
        # their immediate launch and recorded-request behavior unchanged.
        review_output = self.source is None or (self.destination is not None and self.destination.exists())
        self.inputs.setVisible(review_output)
        if self.source is not None:
            for widget in (self.pick_path.parentWidget(), self.object_ids, self.steps,
                           self.wall_seconds, self.pose_from_scan, self.search_only, note):
                form.setRowVisible(widget, False)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        layout.addWidget(self.log, 1)
        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start planning")
        self.start_button.clicked.connect(self.start)
        self.start_button.setVisible(review_output)
        self.stop_button = QPushButton("Close")
        self.stop_button.clicked.connect(self.reject)
        self.open_button = QPushButton("Open resulting run")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(lambda: self.run_available.emit(self.destination))
        for button in (self.start_button, self.stop_button, self.open_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(self.repository))
        environment = QProcessEnvironment.systemEnvironment()
        sources = [self.repository / "packages/stacking-common/src", self.repository / "apps/tabletop/src"]
        environment.insert("PYTHONPATH", os.pathsep.join([*map(str, sources), environment.value("PYTHONPATH")]))
        self.process.setProcessEnvironment(environment)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        if self.source is not None and not review_output:
            self.start()

    def _file_row(self, form, label, field, file_filter, *, save=False):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field)
        button = QPushButton("Browse…")
        layout.addWidget(button)
        def choose():
            picker = QFileDialog.getSaveFileName if save else QFileDialog.getOpenFileName
            options = ({"options": QFileDialog.Option.DontUseNativeDialog} if save else {})
            filename, _ = picker(self, label, field.text() or str(self.repository), file_filter, **options)
            if filename:
                field.setText(filename)
        button.clicked.connect(choose)
        form.addRow(label, row)

    def _path(self, text):
        path = Path(text.strip()).expanduser()
        return path.resolve() if path.is_absolute() else (self.repository / path).resolve()

    def arguments(self):
        if not self.output_path.text().strip():
            raise ValueError("Choose a new output .stack path")
        destination = self._path(self.output_path.text())
        if destination.suffix != ".stack":
            destination = destination.with_suffix(".stack")
        output = PlanningRunOutput(destination, overwrite=self.overwrite.isChecked(), source=self.source)
        config = self._path(self.config_path.text())
        if not config.is_file():
            raise ValueError("Choose an existing MCTS YAML profile")
        arguments = ["-u", str(self.repository / "apps/tabletop/scripts/debug/mcts.py"),
                     "--repository", str(self.repository), "--config", str(config),
                     "--save-run", str(output.path)]
        if self.source is not None:
            arguments += ["--resume-run", str(self.source), "--plan-motion"]
        else:
            arguments += ["--steps", str(self.steps.value()), "--max-run-time", str(self.wall_seconds.value()),
                          "--no-plan-motion" if self.search_only.isChecked() else "--plan-motion"]
            scan_override = self.pose_from_scan.currentData()
            if scan_override is not None:
                arguments.append("--pose-from-scan" if scan_override else "--no-pose-from-scan")
            if self.pick_path.text().strip():
                pickup = self._path(self.pick_path.text())
                if not pickup.is_file():
                    raise ValueError("Choose an existing trusted pickup pose file")
                arguments += ["--pick-pose-data", str(pickup)]
            if self.object_ids.text().strip():
                try:
                    ids = [int(value) for value in self.object_ids.text().replace(",", " ").split()]
                except ValueError as error:
                    raise ValueError("Object IDs must be positive integers separated by spaces or commas") from error
                if not ids or any(value <= 0 for value in ids) or len(set(ids)) != len(ids):
                    raise ValueError("Object IDs must be positive and unique")
                arguments += ["--object-ids", *map(str, ids)]
        self._run_output = output
        self.destination = output.path
        return arguments

    def start(self):
        if self.running:
            return
        try:
            if not self.can_start():
                raise ValueError("Finish hardware execution or video export before starting planning")
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
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
            # Creation and the first committed header need not be simultaneous.
            return
        self._announced_path = path
        self._run_timer.stop()
        self.open_button.setEnabled(True)
        self.run_available.emit(path)

    @property
    def running(self):
        return self.process.state() != QProcess.ProcessState.NotRunning

    def _read_output(self):
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not text:
            return
        scrollbar = self.log.verticalScrollBar()
        position = scrollbar.value()
        selection = self.log.textCursor()
        follow = not selection.hasSelection() and position == scrollbar.maximum()
        # Process output must not replace the user's selection (including after
        # Ctrl+A / Ctrl+C). Keep its endpoint fixed when appending at that point.
        selection.setKeepPositionOnInsert(True)
        self.log.setTextCursor(selection)
        output = QTextCursor(self.log.document())
        output.movePosition(QTextCursor.MoveOperation.End)
        output.insertText(text)
        scrollbar.setValue(scrollbar.maximum() if follow else position)

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
                # Signal the supervisor, which terminates/reaps its native worker
                # group and marks the journal. Do not kill only the supervisor.
                try:
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
