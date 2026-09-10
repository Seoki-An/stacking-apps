"""Nonblocking offline planning process dialog."""

import os
from pathlib import Path
import signal
import sys

from PySide6.QtCore import QProcess, Signal
from PySide6.QtWidgets import QDialog, QPlainTextEdit, QPushButton, QVBoxLayout


class PlanningRunDialog(QDialog):
    run_available = Signal(object)

    def __init__(self, repository, source, destination, config, parent=None):
        super().__init__(parent)
        self.destination = Path(destination)
        self.setWindowTitle("Resume offline planning — no robot execution")
        self.resize(900, 500)
        layout = QVBoxLayout(self)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)
        self.stop_button = QPushButton("Stop planning")
        self.stop_button.clicked.connect(self.reject)
        layout.addWidget(self.stop_button)
        self.open_button = QPushButton("Open resulting run")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(lambda: self.run_available.emit(self.destination))
        layout.addWidget(self.open_button)
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(repository))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(lambda error: self.log.appendPlainText(self.process.errorString()))
        self.process.start(sys.executable, [
            str(Path(repository) / "apps/tabletop/scripts/debug/mcts.py"),
            "--repository", str(repository), "--config", str(config),
            "--resume-run", str(source), "--save-run", str(destination), "--plan-motion",
        ])

    @property
    def running(self):
        return self.process.state() != QProcess.ProcessState.NotRunning

    def _read_output(self):
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.log.insertPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _finished(self, code, status):
        self._read_output()
        self.log.appendPlainText(f"\nPlanning process exited: {code}")
        self.stop_button.setText("Close")
        self.stop_button.setEnabled(True)
        self.open_button.setEnabled(self.destination.is_file())

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
