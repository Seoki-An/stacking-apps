"""Operations-console entry to fixed-field acquisition and identification."""

from datetime import datetime
import json
import os
from pathlib import Path
import signal
import sys
from uuid import uuid4

from PySide6.QtCore import QProcess, QProcessEnvironment, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)


class FieldScanDialog(QDialog):
    poses_available = Signal(object)
    running_changed = Signal()

    def __init__(self, repository, parent=None):
        super().__init__(parent)
        self.repository = Path(repository).resolve()
        self.can_start = lambda: True
        self.pose_path = None
        self._stopping = False
        self.setWindowTitle('Field capture and stone identification')
        self.resize(850, 650)
        layout = QVBoxLayout(self)
        self.inputs = QWidget()
        form = QFormLayout(self.inputs)
        self.mode = QComboBox()
        self.mode.addItems(['Capture and identify', 'Identify existing capture'])
        self.camera = QComboBox()
        for label, value in [('Primary FEMTO', 'primary'), ('Secondary FEMTO', 'secondary'), ('Both FEMTO cameras', 'both')]:
            self.camera.addItem(label, value)
        self.output = QLineEdit(str(self.repository / '.data/field_pcd' /
                                   f'field_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}'))
        self.source = QLineEdit()
        self.source.setPlaceholderText('Capture, workflow root, or pose_identification folder')
        self.models = QLineEdit()
        self.models.setPlaceholderText('Use measured PCD models from the inventory profile')
        self.height = QDoubleSpinBox()
        self.height.setRange(-10, 10)
        self.height.setDecimals(3)
        self.height.setSuffix(' m')
        from stacking_tabletop.config import load_perception_profile
        self.height.setValue(load_perception_profile(self.repository / 'apps/tabletop/config/perception.yml').segmentation.designated_table_height)
        self.running_camera = QCheckBox('Use an already running camera driver (single camera only)')
        self.automatic = QCheckBox('Assign stone IDs automatically')
        self.manual = QCheckBox('Manually align assigned models before ICP')
        self.visualize = QCheckBox('Show filtered cloud and final pose review')
        self.visualize.setChecked(True)
        form.addRow('Workflow', self.mode)
        form.addRow('Camera', self.camera)
        for label, field in [('New output directory', self.output), ('Existing capture', self.source), ('Model PCD directory', self.models)]:
            row = QHBoxLayout()
            row.addWidget(field)
            button = QPushButton('Browse…')
            button.clicked.connect(lambda _checked=False, f=field: self._browse(f))
            row.addWidget(button)
            form.addRow(label, row)
        form.addRow('Pickup table height', self.height)
        for check in (self.running_camera, self.automatic, self.manual, self.visualize):
            form.addRow(check)
        layout.addWidget(self.inputs)
        note = QLabel('Fixed cameras only; this workflow sends no robot or gripper commands. '
                      'By default, assign an ID or Skip to each cluster in the identification window. '
                      'Stop external camera drivers before using owned capture.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        layout.addWidget(self.log, 1)
        row = QHBoxLayout()
        self.start_button = QPushButton('Start')
        self.stop_button = QPushButton('Close')
        self.plan_button = QPushButton('New plan using these poses')
        self.plan_button.setEnabled(False)
        for button in (self.start_button, self.stop_button, self.plan_button):
            row.addWidget(button)
        layout.addLayout(row)
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(self.repository))
        environment = QProcessEnvironment.systemEnvironment()
        sources = [self.repository / 'packages/stacking-common/src', self.repository / 'apps/tabletop/src']
        environment.insert('PYTHONPATH', os.pathsep.join([*map(str, sources), environment.value('PYTHONPATH')]))
        self.process.setProcessEnvironment(environment)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        self.process.stateChanged.connect(lambda _state: self.running_changed.emit())
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.reject)
        self.plan_button.clicked.connect(lambda: self.poses_available.emit(self.pose_path))

    @property
    def running(self):
        return self.process.state() != QProcess.ProcessState.NotRunning

    def _browse(self, field):
        directory = QFileDialog.getExistingDirectory(self, 'Choose directory', str(self.repository))
        if directory:
            if field is self.output:
                directory = str(Path(directory) / f'field_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}')
            field.setText(directory)

    def arguments(self):
        capture = self.mode.currentIndex() == 0
        args = ['-u', '-m', 'stacking_tabletop.acquisition.field_workflow' if capture
                else 'stacking_tabletop.perception.field_cli']
        if capture:
            if not self.output.text().strip():
                raise ValueError('choose a new output directory')
            if Path(self.output.text()).expanduser().exists():
                raise ValueError('output directory already exists; choose a new directory')
            args += ['--output-dir', self.output.text()]
            if self.camera.currentData() == 'both':
                if self.running_camera.isChecked():
                    raise ValueError('both-camera capture must own the two sequential drivers')
                args.append('--both-cameras')
            else:
                args += ['--camera', self.camera.currentData()]
            if self.running_camera.isChecked():
                args.append('--use-running-camera')
        else:
            if not self.source.text().strip() or not Path(self.source.text()).expanduser().is_dir():
                raise ValueError('choose an existing field capture directory')
            from stacking_tabletop.perception.field_source import resolve_field_capture_directory
            source = resolve_field_capture_directory(self.source.text())
            self.source.setText(str(source))
            args.append(str(source))
            if not self.output.text().strip() or Path(self.output.text()).expanduser().exists():
                raise ValueError('choose a new output directory for identification results')
            args += ['--identification-output', self.output.text()]
        args += ['--repository', str(self.repository), '--pick-table-height', str(self.height.value())]
        if self.models.text().strip():
            args += ['--stone-directory', self.models.text()]
        for check, flag in [(self.automatic, '--automatic-ordering'), (self.manual, '--manual-init-gui'), (self.visualize, '--visualize')]:
            if check.isChecked():
                args.append(flag)
        return args

    def start(self):
        if self.running:
            return
        try:
            if not self.can_start():
                raise ValueError('finish active execution or planning before starting a field workflow')
            args = self.arguments()
        except ValueError as error:
            self.log.appendPlainText(str(error))
            return
        self.pose_path = None
        self._stopping = False
        self.plan_button.setEnabled(False)
        self.inputs.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setText('Stop')
        self.log.appendPlainText('Starting field workflow…')
        self.process.start(sys.executable, args)

    def _read(self):
        self.log.appendPlainText(bytes(self.process.readAllStandardOutput()).decode(errors='replace').rstrip())

    def _finished(self, code, exit_status):
        self._read()
        self.inputs.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.stop_button.setText('Close')
        if code == 0 and exit_status == QProcess.ExitStatus.NormalExit and not self._stopping:
            try:
                if self.mode.currentIndex() == 0:
                    manifest = Path(self.output.text()).expanduser() / 'field_workflow.json'
                    result = json.loads(manifest.read_text())
                    if result['status'] != 'complete':
                        raise ValueError('field workflow did not complete')
                    self.pose_path = Path(result['pose_data'])
                else:
                    self.pose_path = Path(self.output.text()).expanduser() / 'pose_data.pkl'
                if not self.pose_path.is_file():
                    raise ValueError('pickup pose output is missing')
                self.plan_button.setEnabled(True)
                self.log.appendPlainText(f'Ready for planning: {self.pose_path}')
            except (OSError, ValueError, KeyError) as error:
                self.pose_path = None
                self.log.appendPlainText(str(error))
        else:
            self.log.appendPlainText('Field workflow stopped or failed; see the log and retained output.')
        self.running_changed.emit()

    def _error(self, error):
        self.log.appendPlainText(self.process.errorString())
        if error == QProcess.ProcessError.FailedToStart:
            self._finished(1, QProcess.ExitStatus.CrashExit)

    def reject(self):
        if self.running:
            self._stopping = True
            pid = self.process.processId()
            if pid:
                try:
                    os.kill(pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
            self.stop_button.setEnabled(False)
            self.stop_button.setText('Stopping…')
            return
        super().reject()

    def closeEvent(self, event):
        if self.running:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)
