"""Operations-console entry to FEMTO and wrist-RGB field identification."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
from uuid import uuid4

from PySide6.QtCore import QProcess, QProcessEnvironment, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)


def inspect_joint_pick_output(directory):
    """Verify that workbench pickup poses came from accepted RGB/FEMTO fusion."""
    root = Path(directory).expanduser().resolve()
    workflow = json.loads((root / 'pick_scene_workflow.json').read_text())
    published = json.loads((root / 'pick_scene_poses.json').read_text())
    report_path = root / 'pose_identification/stone_poses.json'
    report = json.loads(report_path.read_text())
    if (workflow.get('status') != 'complete' or not workflow.get('production_pose_accepted')
            or published.get('status') != 'complete' or not published.get('production_pose_accepted')):
        raise ValueError('pick-scene workflow did not accept the pickup poses')
    wrist_only = workflow.get('sensor_mode') == 'wrist-only'
    expected_source = ('rgb_femto_first_joint_icp' if workflow.get('schema_version') == 2
                       else 'balanced_femto_rgb_joint_icp')
    if workflow.get('pose_backend') == 'foundationpose':
        expected_source = 'foundationpose_wrist_rgb' if wrist_only else 'foundationpose_rgb_femto_validated'
        if report.get('icp_refinement_applied') is not False:
            raise ValueError('FoundationPose result unexpectedly applied ICP refinement')
    elif workflow.get('pose_backend', 'icp') != 'icp':
        raise ValueError('unknown pickup pose backend')
    if wrist_only and workflow.get('pose_backend') == 'icp':
        expected_source = 'wrist_rgb_icp'
    if wrist_only and (published.get('sensor_mode') != 'wrist-only' or report.get('sensor_mode') != 'wrist-only'):
        raise ValueError('wrist-only sensor provenance differs between outputs')
    if (published.get('pose_source') != expected_source
            or report.get('pose_source') != expected_source
            or report.get('kind') != ('wrist_rgb_pose_identification' if wrist_only else 'femto_rgb_joint_pose_identification')
            or report.get('status') != 'complete'):
        raise ValueError('pickup poses are not from joint FEMTO/RGB identification; reprocess the capture')
    pose = root / 'pose_data.pkl'
    if (Path(published['pose_data']).resolve() != pose
            or Path(published['joint_identification']).resolve() != report_path):
        raise ValueError('published pickup pose references do not match this workflow')
    dense = Path(workflow['dense_dir']).resolve()
    if not dense.is_relative_to(root):
        raise ValueError('RGB reconstruction is outside the selected workflow')
    dense_summary = dense / 'summary.json'
    rgb_pcd = dense / 'refined/rgb_dense_voxel_1mm.pcd'
    if (not rgb_pcd.is_file() or report.get('source_sha256', {}).get('dense_summary.json')
            != hashlib.sha256(dense_summary.read_bytes()).hexdigest()):
        raise ValueError('joint pickup poses are not bound to the selected RGB reconstruction')
    if workflow.get('schema_version') == 2:
        if (published.get('preliminary_femto_identification') is not None
                or workflow.get('pose_sequence') != ('rgb_mvs_regions_catalog_fit' if wrist_only else
                                                   'femto_regions_rgb_mvs_joint_catalog_fit')):
            raise ValueError('pickup poses did not follow RGB-first catalog identification')
        geometry_name = 'rgb_geometry' if wrist_only else 'femto_geometry'
        geometry = root / geometry_name / 'geometry.json'
        if (not geometry.is_file() or published.get('coarse_rgb_geometry' if wrist_only else 'coarse_femto_geometry') != geometry_name + '/geometry.json'
                or json.loads(geometry.read_text()).get('catalog_pose_fitted') is not False
                or report.get('source_sha256', {}).get(geometry_name + '.json')
                != hashlib.sha256(geometry.read_bytes()).hexdigest()):
            raise ValueError('joint pickup poses are not bound to coarse FEMTO geometry')
    fused_pose = Path(workflow['fused_pose_data']).resolve()
    if (not pose.is_file() or not fused_pose.is_file()
            or hashlib.sha256(pose.read_bytes()).digest() != hashlib.sha256(fused_pose.read_bytes()).digest()):
        raise ValueError('published pickup poses differ from the joint identification result')
    viewer = dense / 'rgb_only/overview.html'
    if not viewer.is_file():
        viewer = dense / 'comparison/overview.html'
    pose_viewer = fused_pose.parent / 'comparison/overview.html'
    if published.get('pose_viewer') is not None and (root / published['pose_viewer']).resolve() != pose_viewer:
        raise ValueError('published pose viewer does not match the joint identification result')
    if workflow.get('pose_viewer') is not None and Path(workflow['pose_viewer']).resolve() != pose_viewer:
        raise ValueError('workflow pose viewer does not match the joint identification result')
    return (pose, viewer if viewer.is_file() else None,
            pose_viewer if pose_viewer.is_file() else None, published.get('pose_warnings', []))


def retained_pick_viewers(directory):
    """Find diagnostic viewers without treating a failed run as accepted poses."""
    root = Path(directory).expanduser().resolve()
    workflow = json.loads((root/'pick_scene_workflow.json').read_text())
    stage = workflow.get('stage', '')
    dense = Path(workflow.get('resume_dense_dir') or workflow.get('dense_dir') or
                 root/('rgb_dense_full' if 'full' in stage else 'rgb_dense'))
    if not dense.is_absolute():
        dense = root/dense
    dense = dense.resolve()
    rgb = None
    if dense.is_relative_to(root) and (dense/'summary.json').is_file():
        summary = json.loads((dense/'summary.json').read_text())
        path = dense/'rgb_only/overview.html'
        if summary.get('status') == 'complete' and path.resolve().is_relative_to(root) and path.is_file():
            rgb = path
    attempt = workflow.get('resume_count', 0)
    identification = (f'fused_identification_resume_{attempt:02d}' if attempt else
                      'fused_identification_full' if 'full' in stage else 'fused_identification')
    pose = root/identification/'comparison/overview.html'
    if not pose.resolve().is_relative_to(root) or not pose.is_file():
        pose = None
    return rgb, pose


class FieldScanDialog(QDialog):
    poses_available = Signal(object)
    running_changed = Signal()

    def __init__(self, repository, parent=None):
        super().__init__(parent)
        self.repository = Path(repository).resolve()
        self.can_start = lambda: True
        self.pose_path = None
        self._completed_workflow = None
        self.rgb_viewer = None
        self.pose_viewer = None
        self._stopping = False
        self._result_directory = None
        self._coverage_preview_directory = None
        self.setWindowTitle('Scene capture and stone identification')
        self.resize(850, 650)
        layout = QVBoxLayout(self)
        self.inputs = QWidget()
        form = QFormLayout(self.inputs)
        self.mode = QComboBox()
        for label, value in [
            ('Capture FEMTO and identify', 'field_capture'),
            ('Identify existing FEMTO capture', 'field_existing'),
            ('Capture wrist RGB + FEMTO and identify', 'pick_capture'),
            ('Identify existing wrist RGB + FEMTO capture', 'pick_existing'),
            ('Resume failed wrist RGB identification', 'pick_resume'),
            ('Capture wrist RGB only and identify (no FEMTO)', 'pick_wrist_capture'),
            ('Identify saved wrist RGB only capture', 'pick_wrist_existing'),
        ]:
            self.mode.addItem(label, value)
        self.camera = QComboBox()
        for label, value in [('Primary FEMTO', 'primary'), ('Secondary FEMTO', 'secondary'), ('Both FEMTO cameras', 'both')]:
            self.camera.addItem(label, value)
        self.viewpoints = QComboBox()
        self.viewpoints.addItem('Wider: 13 views, including four side views', 'coverage13')
        self.viewpoints.addItem('Compact: 9 near and oblique views', 'coverage9')
        self.viewpoints.addItem('Extended: 17 views, including 45°/60° left and right', 'coverage17')
        self.viewpoints.setCurrentIndex(self.viewpoints.findData('coverage9'))
        self.viewpoints.currentIndexChanged.connect(self._mode_changed)
        self.dense_backend = QComboBox()
        self.dense_backend.addItem('Fast-FoundationStereo (default)', 'fast-foundation-stereo')
        self.dense_backend.addItem('COLMAP MVS (ICP pose estimation)', 'colmap')
        self.pose_backend = QComboBox()
        self.pose_backend.addItem('FoundationPose (default)', 'foundationpose')
        self.pose_backend.addItem('Multistart ICP', 'icp')
        self.pose_backend.currentIndexChanged.connect(self._mode_changed)
        self.output = QLineEdit(str(self.repository / '.data/field_pcd' /
                                   f'field_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}'))
        self.source = QLineEdit()
        self.source.setPlaceholderText('Existing FEMTO workflow or wrist+FEMTO scene capture')
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
        self.visualize = QCheckBox('Show review windows and write comparison viewers')
        self.visualize.setChecked(True)
        form.addRow('Workflow', self.mode)
        form.addRow('Camera', self.camera)
        form.addRow('Wrist viewpoints', self.viewpoints)
        self.coverage_button = QPushButton('Preview capture region (no motion)')
        self.coverage_button.clicked.connect(self.preview_coverage)
        form.addRow(self.coverage_button)
        form.addRow('RGB reconstruction', self.dense_backend)
        form.addRow('Pose estimation', self.pose_backend)
        for label, field in [('New output directory', self.output), ('Existing capture / workflow', self.source),
                             ('Model PCD directory', self.models)]:
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
        self.note = QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        layout.addWidget(self.log, 1)
        row = QHBoxLayout()
        self.start_button = QPushButton('Start')
        self.stop_button = QPushButton('Close')
        self.plan_button = QPushButton('New plan using these poses')
        self.plan_button.setEnabled(False)
        self.rgb_button = QPushButton('Open RGB PCD')
        self.rgb_button.setEnabled(False)
        self.rgb_button.clicked.connect(self._open_rgb_viewer)
        self.pose_button = QPushButton('Open identified poses')
        self.pose_button.setEnabled(False)
        self.pose_button.clicked.connect(self._open_pose_viewer)
        for button in (self.start_button, self.stop_button, self.rgb_button,
                       self.pose_button, self.plan_button):
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
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self._mode_changed()

    @property
    def running(self):
        return self.process.state() != QProcess.ProcessState.NotRunning

    def _browse(self, field):
        directory = QFileDialog.getExistingDirectory(self, 'Choose directory', str(self.repository))
        if directory:
            if field is self.output:
                directory = str(Path(directory) / f'field_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}')
            field.setText(directory)

    def _mode_changed(self):
        mode = self.mode.currentData()
        resume = mode == 'pick_resume'
        existing = mode.endswith('_existing') or resume
        wrist = mode.startswith('pick_')
        wrist_only = mode.startswith('pick_wrist_')
        live_wrist = mode in ('pick_capture', 'pick_wrist_capture')
        self.rgb_button.setVisible(wrist)
        self.pose_button.setVisible(wrist)
        self.source.setEnabled(existing)
        self.output.setEnabled(not resume)
        self.camera.setEnabled(not existing and not wrist_only)
        self.viewpoints.setEnabled(live_wrist)
        self.coverage_button.setVisible(live_wrist)
        self.dense_backend.setEnabled(wrist and not resume and not wrist_only)
        if wrist_only:
            self.dense_backend.setCurrentIndex(self.dense_backend.findData('fast-foundation-stereo'))
        self.pose_backend.setEnabled(wrist)
        learned = wrist and self.pose_backend.currentData() == 'foundationpose'
        self.manual.setEnabled(not learned)
        if learned:
            self.manual.setChecked(False)
        self.running_camera.setEnabled(mode == 'field_capture')
        if mode != 'field_capture':
            self.running_camera.setChecked(False)
        self.automatic.setEnabled(not wrist)
        if wrist and self.automatic.isChecked():
            self.automatic.setChecked(False)
        if wrist:
            if wrist_only:
                self.note.setText(
                    ('Reprocess saved wrist RGB without robot motion. ' if existing else
                     'Capture from q_capture_init_deg, then return to the original starting pose. Keep bringup running; '
                     'use an empty tool and check route clearance. ') +
                    'Fast-FoundationStereo reconstructs the configured pickup workspace. '
                    'RGB table/stone segmentation supplies FoundationPose masks. Review stone IDs after reconstruction. '
                    'No FEMTO camera or calibration is used; scene stability is not independently checked.')
            elif resume:
                self.note.setText('Resume a failed joint pose identification using its saved RGB MVS output. '
                                  'Choose the failed workflow directory, then review stone IDs and estimate their poses. '
                                  'No camera capture or MVS reconstruction is repeated.')
            elif existing:
                self.note.setText('Reprocesses the saved wrist RGB + FEMTO capture without robot motion. '
                                  'FEMTO supplies coarse stone regions; wrist RGB MVS runs before the first '
                                  'catalog pose estimate. Both RGB and FEMTO validate the result. Review stone IDs after reconstruction. '
                                  'Close each review window to continue the workflow.')
            else:
                route = {
                    'coverage9': '9 near/oblique views (camera lateral travel up to 0.08 m)',
                    'coverage13': '13 views with four 30° side orbits (camera lateral travel up to 0.20 m)',
                    'coverage17': '17 views with 45°/60° left/right orbits (camera lateral travel up to 0.35 m)',
                }[self.viewpoints.currentData()]
                self.note.setText(f'Wrist capture first moves to the configured capture pose, then takes {route}. '
                                  'Views are relative to q_capture_init_deg in scene_capture.yml; '
                                  'successful capture returns to the original starting pose. '
                                  'Keep bringup running, use an empty tool, check clearance for the full robot/tool '
                                  'path and stop external camera drivers. Routes are not collision checked. '
                                  'FEMTO provides coarse regions; wrist RGB MVS runs before the first catalog '
                                  'pose estimate. Both clouds validate the result. Review stone IDs after reconstruction.')
        else:
            self.note.setText('The fixed FEMTO workflow sends no robot or gripper commands. Assign an ID or Skip '
                              'to each cluster in the identification window. Stop external camera drivers before capture.')

    def arguments(self, *, reidentify_source=None):
        mode = 'pick_reidentify' if reidentify_source is not None else self.mode.currentData()
        wrist_only = mode.startswith('pick_wrist_')
        field = mode.startswith('field_')
        wrist = mode.startswith('pick_')
        resume = mode == 'pick_resume'
        reidentify = mode == 'pick_reidentify'
        capture = mode.endswith('_capture')
        if not resume and not self.output.text().strip():
            raise ValueError('choose a new output directory')
        if not resume and Path(self.output.text()).expanduser().exists():
            raise ValueError('output directory already exists; choose a new directory')
        if field and capture:
            args = ['-u', '-m', 'stacking_tabletop.acquisition.field_workflow']
            args += ['--output-dir', self.output.text()]
            if self.camera.currentData() == 'both':
                if self.running_camera.isChecked():
                    raise ValueError('both-camera capture must own the two sequential drivers')
                args.append('--both-cameras')
            else:
                args += ['--camera', self.camera.currentData()]
            if self.running_camera.isChecked():
                args.append('--use-running-camera')
        elif field:
            args = ['-u', '-m', 'stacking_tabletop.perception.field_cli']
            if not self.source.text().strip() or not Path(self.source.text()).expanduser().is_dir():
                raise ValueError('choose an existing field capture directory')
            from stacking_tabletop.perception.field_source import resolve_field_capture_directory
            source = resolve_field_capture_directory(self.source.text())
            self.source.setText(str(source))
            args.append(str(source))
            if not self.output.text().strip() or Path(self.output.text()).expanduser().exists():
                raise ValueError('choose a new output directory for identification results')
            args += ['--identification-output', self.output.text()]
        else:
            args = ['-u', '-m', 'stacking_tabletop.acquisition.pick_scene_workflow',
                    'reidentify' if reidentify else 'resume' if resume else 'run' if capture else 'process']
            if resume or reidentify:
                source = Path(reidentify_source or self.source.text()).expanduser()
                if not (source / 'pick_scene_workflow.json').is_file():
                    raise ValueError('choose a wrist+FEMTO workflow directory')
                args.append(str(source))
            elif capture:
                if not wrist_only and self.camera.currentData() == 'both':
                    raise ValueError('wrist scene capture supports one fixed FEMTO camera')
                if self.running_camera.isChecked():
                    raise ValueError('wrist scene capture must own FEMTO driver scheduling')
                args += ['--viewpoint-pattern', self.viewpoints.currentData()]
                if not wrist_only:
                    args += ['--camera', self.camera.currentData()]
            else:
                if not self.source.text().strip() or not Path(self.source.text()).expanduser().is_dir():
                    raise ValueError('choose an existing complete wrist+FEMTO scene capture')
                args.append(self.source.text())
            if wrist_only:
                args += ['--sensor-mode', 'wrist-only']
            args += ['--pose-backend', self.pose_backend.currentData()]
            if not resume and not reidentify:
                if self.pose_backend.currentData() == 'foundationpose' and self.dense_backend.currentData() != 'fast-foundation-stereo':
                    raise ValueError('FoundationPose needs Fast-FoundationStereo reconstruction; choose ICP for COLMAP')
                args += ['--dense-backend', self.dense_backend.currentData()]
            if not resume:
                args += ['--output', self.output.text()]
        args += ['--repository', str(self.repository), '--pick-table-height', str(self.height.value())]
        if self.models.text().strip():
            args += ['--stone-directory', self.models.text()]
        for check, flag in [(self.automatic, '--automatic-ordering'), (self.manual, '--manual-init-gui'), (self.visualize, '--visualize')]:
            if check.isChecked() and (not wrist or check is not self.automatic):
                args.append(flag)
        return args

    def preview_coverage(self):
        if self.running or self.mode.currentData() not in ('pick_capture', 'pick_wrist_capture'):
            return
        if not self.can_start():
            self.log.appendPlainText('finish active execution or planning before previewing capture')
            return
        from stacking_tabletop.acquisition.pick_scene_workflow import (
            STANDARD_DISTANCE_M, STANDARD_NEAR_OFFSET_M, STANDARD_OBLIQUE_OFFSET_M)
        # A preview has its own folder, so Start can still use the selected output.
        directory = self.repository / '.data/field_pcd' / f'coverage_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}'
        self._coverage_preview_directory = directory
        self._stopping = False
        args = ['-u', '-m', 'stacking_tabletop.acquisition.scene_capture_cli', 'plan',
                '--profile', str(self.repository/'apps/tabletop/config/hand_eye.yml'),
                '--distance', str(STANDARD_DISTANCE_M), '--offset', str(STANDARD_NEAR_OFFSET_M),
                '--oblique-offset', str(STANDARD_OBLIQUE_OFFSET_M),
                '--viewpoint-pattern', self.viewpoints.currentData(), '--output', str(directory)]
        self.inputs.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setText('Stop')
        self.log.appendPlainText('Previewing the configured capture pose and approach/return from the current '
                                 'robot pose; no motion or camera capture. Keep bringup running.')
        self.process.start(sys.executable, args)

    def start(self, *, reidentify_source=None):
        if self.running:
            return
        try:
            if not self.can_start():
                raise ValueError('finish active execution or planning before starting a field workflow')
            args = self.arguments(reidentify_source=reidentify_source)
        except ValueError as error:
            self.log.appendPlainText(str(error))
            return
        self.pose_path = None
        self._completed_workflow = None
        self.rgb_viewer = None
        self.pose_viewer = None
        self.pose_button.setText('Open identified poses')
        self._stopping = False
        self.plan_button.setEnabled(False)
        self.rgb_button.setEnabled(False)
        self.pose_button.setEnabled(False)
        self.inputs.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setText('Stop')
        self._result_directory = Path(
            self.source.text() if self.mode.currentData() == 'pick_resume' and reidentify_source is None
            else self.output.text()).expanduser().resolve()
        self.log.appendPlainText('Starting scene workflow…')
        self.process.start(sys.executable, args)

    def _read(self):
        self.log.appendPlainText(bytes(self.process.readAllStandardOutput()).decode(errors='replace').rstrip())

    def _finished(self, code, exit_status):
        self._read()
        self.inputs.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.stop_button.setText('Close')
        if self._coverage_preview_directory is not None:
            directory = self._coverage_preview_directory
            self._coverage_preview_directory = None
            path = directory/'coverage.html'
            if code == 0 and exit_status == QProcess.ExitStatus.NormalExit and not self._stopping and path.is_file():
                self.log.appendPlainText(f'Capture region preview: {path}')
                if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
                    self.log.appendPlainText('Open the preview file manually.')
            else:
                self.log.appendPlainText(f'Capture route stopped or failed. Diagnostics: {directory}')
                if not self._stopping and path.is_file():
                    self.log.appendPlainText('Showing desired camera coverage only. The robot route is invalid; '
                                             'this preview cannot be used to start capture.')
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            return
        if code == 0 and exit_status == QProcess.ExitStatus.NormalExit and not self._stopping:
            review_ids = False
            try:
                if self.mode.currentData() == 'field_capture':
                    manifest = Path(self.output.text()).expanduser() / 'field_workflow.json'
                    result = json.loads(manifest.read_text())
                    if result['status'] != 'complete':
                        raise ValueError('field workflow did not complete')
                    self.pose_path = Path(result['pose_data'])
                elif self.mode.currentData().startswith('pick_'):
                    result_dir = self._result_directory or (
                        self.source.text() if self.mode.currentData() == 'pick_resume' else self.output.text())
                    self.pose_path, self.rgb_viewer, self.pose_viewer, warnings = inspect_joint_pick_output(result_dir)
                    self._completed_workflow = Path(result_dir).expanduser().resolve()
                    schema = json.loads((self._completed_workflow / 'pick_scene_workflow.json').read_text()).get('schema_version')
                    review_ids = schema == 2
                    if self.rgb_viewer is not None:
                        self.rgb_button.setEnabled(True)
                        self.log.appendPlainText(f'RGB reconstruction: {self.rgb_viewer}')
                    if self.pose_viewer is not None:
                        self.pose_button.setEnabled(True)
                        self.log.appendPlainText(f'Identified poses: {self.pose_viewer}')
                        if self.visualize.isChecked():
                            self._open_pose_viewer()
                    origin = json.loads((Path(result_dir).expanduser() / 'pick_scene_poses.json').read_text()).get('data_origin', 'hardware')
                    sensor_mode = json.loads((self._completed_workflow / 'pick_scene_workflow.json').read_text()).get('sensor_mode')
                    source_label = 'wrist RGB only' if sensor_mode == 'wrist-only' else 'joint FEMTO/RGB'
                    self.log.appendPlainText(f'Published {source_label} pickup poses (data source: {origin})')
                    if warnings:
                        self.log.appendPlainText(f'Pose warnings: {warnings}')
                else:
                    self.pose_path = Path(self.output.text()).expanduser() / 'pose_data.pkl'
                if not self.pose_path.is_file():
                    raise ValueError('pickup pose output is missing')
                if review_ids and self._ask_model_id_review():
                    self._change_model_ids()
                else:
                    self.plan_button.setEnabled(True)
                    self.log.appendPlainText(f'Ready for planning: {self.pose_path}')
            except (OSError, ValueError, KeyError) as error:
                self.pose_path = None
                self.log.appendPlainText(str(error))
        else:
            self.log.appendPlainText('Field workflow stopped or failed; see the log and retained output.')
            self.pose_path = None
            self.plan_button.setEnabled(False)
            if self.mode.currentData().startswith('pick_'):
                directory = self._result_directory or (
                    self.source.text() if self.mode.currentData() == 'pick_resume' else self.output.text())
                try:
                    self.rgb_viewer, self.pose_viewer = retained_pick_viewers(directory)
                    self.rgb_button.setEnabled(self.rgb_viewer is not None)
                    self.pose_button.setEnabled(self.pose_viewer is not None)
                    self.pose_button.setText('Open pose diagnostics')
                    if self.rgb_viewer is not None:
                        self.log.appendPlainText(f'Retained RGB reconstruction: {self.rgb_viewer}')
                    if self.pose_viewer is not None:
                        self.log.appendPlainText(f'Unaccepted pose diagnostics: {self.pose_viewer}')
                    else:
                        self.log.appendPlainText('No pose-comparison viewer was produced before the failure.')
                    if self.visualize.isChecked():
                        if self.pose_viewer is not None:
                            self._open_pose_viewer()
                        elif self.rgb_viewer is not None:
                            self._open_rgb_viewer()
                except (OSError, ValueError, KeyError) as error:
                    self.log.appendPlainText(f'Could not inspect retained viewers: {error}')
        self.running_changed.emit()

    def _ask_model_id_review(self):
        prompt = QMessageBox(self)
        prompt.setWindowTitle('Review identified poses')
        prompt.setIcon(QMessageBox.Icon.Question)
        prompt.setText('Pose identification is complete. Accept these poses or change the model IDs?')
        prompt.setInformativeText('Changing IDs returns to model selection using the saved observation clouds. '
                                  'Capture and reconstruction will not run again. The previous result is preserved.')
        accept = prompt.addButton('Accept poses', QMessageBox.ButtonRole.AcceptRole)
        change = prompt.addButton('Change model IDs', QMessageBox.ButtonRole.ActionRole)
        prompt.setDefaultButton(accept)
        prompt.setEscapeButton(accept)
        prompt.exec()
        return prompt.clickedButton() is change

    def _change_model_ids(self):
        if self.running or self._completed_workflow is None:
            return
        source = self._completed_workflow
        self.output.setText(str(source.with_name(
            f'{source.name}_reidentify_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}')))
        self.start(reidentify_source=source)

    def _open_rgb_viewer(self):
        if self.rgb_viewer is not None and not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.rgb_viewer))):
            self.log.appendPlainText(f'Open RGB reconstruction manually: {self.rgb_viewer}')

    def _open_pose_viewer(self):
        if self.pose_viewer is not None and not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.pose_viewer))):
            self.log.appendPlainText(f'Open identified poses manually: {self.pose_viewer}')

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
