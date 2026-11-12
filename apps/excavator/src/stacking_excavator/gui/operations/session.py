"""Perception review, simulated execution, and durable session history widgets."""
import json
from pathlib import Path
import sys
import sqlite3
from uuid import uuid4
import numpy as np
from scipy.spatial.transform import Rotation
from PySide6.QtCore import QObject, QProcess, QTimer, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QPlainTextEdit, QTableWidget, QTableWidgetItem, QCheckBox, QComboBox, QHeaderView, QInputDialog)
from stacking_common.profiles.planning import load_planning_profile
from stacking_common.visualization.operations import OperationsFrame, RenderObject
from stacking_excavator.execution.runtime import SimulationRuntime
from stacking_excavator.execution.cli import demonstration_command
from stacking_excavator.execution.saved_plan import load_command
from stacking_excavator.execution.emulation import EmulationRuntime
from stacking_common.artifacts import StackRunStore
from stacking_excavator.planning.trajectory import ExcavatorRobotTrajectoryProjector
from stacking_excavator.acquisition.storage import atomic_json, file_hash


def matrix_pose(matrix):
    matrix = np.asarray(matrix)
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()]


class SessionConsole(QObject):
    changed = Signal()

    def __init__(self, repository, parent=None):
        super().__init__(parent)
        self.repository = Path(repository)
        self.runtime = None
        self.command = None
        self.proposal = None
        self.cloud_path = None
        self.projector = None
        self._history_key = None
        self.perception, self.execution, self.history = (QWidget() for _ in range(3))
        self.perception_layout = QVBoxLayout(self.perception)
        self.execution_layout = QVBoxLayout(self.execution)
        history_layout = QVBoxLayout(self.history)
        for layout in (self.perception_layout, self.execution_layout):
            self._button(layout, 'Open session folder…', self.choose_session)
        self.session_label = QLabel('Open a session directory for review and simulation.')
        self.execution_layout.addWidget(self.session_label)
        self.product = self._path(self.perception_layout, 'Verified crop product', directory=True)
        self.priors = self._path(self.perception_layout, 'Initial poses JSON')
        self.identify_button = self._button(self.perception_layout, 'Identify in background', self.identify)
        self.stop_button = self._button(self.perception_layout, 'Stop identification', self.stop_identification)
        self._button(self.perception_layout, 'Open proposal.json…', self.choose_proposal)
        self.proposals = QComboBox()
        self.proposals.currentIndexChanged.connect(self.select_proposal)
        self.perception_layout.addWidget(self.proposals)
        self.layers = {}
        for name in ('Observed cloud', 'Initial poses', 'Proposed poses', 'Crop boundary'):
            checkbox = QCheckBox(name)
            checkbox.setChecked(True)
            checkbox.toggled.connect(lambda _checked: self.changed.emit())
            self.perception_layout.addWidget(checkbox)
            self.layers[name] = checkbox
        self.accept_button = self._button(self.perception_layout, 'Accept proposal', lambda: self.decide(True))
        self.reject_button = self._button(self.perception_layout, 'Reject proposal', lambda: self.decide(False))
        self.perception_info = QPlainTextEdit()
        self.perception_info.setReadOnly(True)
        self.perception_layout.addWidget(self.perception_info)
        self._button(self.execution_layout, 'Load committed placement…', self.choose_plan)
        self._button(self.execution_layout, 'Load closed-loop emulation…', self.choose_emulation)
        demos = QHBoxLayout()
        self.execution_layout.addLayout(demos)
        self._button(demos, 'Load direct demo', lambda: self.load_demo(False))
        self._button(demos, 'Load regrasp demo', lambda: self.load_demo(True))
        self.plan_label = QLabel('No simulation command loaded.')
        self.plan_label.setWordWrap(True)
        self.execution_layout.addWidget(self.plan_label)
        self.start_button = self._button(self.execution_layout, 'Run simulation', self.start)
        review_controls = QHBoxLayout()
        self.execution_layout.addLayout(review_controls)
        self.accept_capture_button = self._button(review_controls, 'Accept capture and continue',
                                                   lambda: self.decide_capture(True))
        self.reject_capture_button = self._button(review_controls, 'Reject capture / replan',
                                                   lambda: self.decide_capture(False))
        motion_controls = QHBoxLayout()
        self.execution_layout.addLayout(motion_controls)
        for label, method in [('Hold', 'hold'), ('Resume', 'resume'), ('Cancel', 'cancel')]:
            self._button(motion_controls, label, lambda m=method: self.control(m))
        self.heartbeat = QCheckBox('Operator heartbeat')
        self.heartbeat.setChecked(True)
        self.execution_layout.addWidget(self.heartbeat)
        self.fault = QComboBox()
        self.fault.addItems(['No injected fault', 'Stalled device', 'Missing feedback', 'Unknown gripper',
                            'Pause crop delivery', 'Missed pickup', 'Grasp slip (10 cm)'])
        self.execution_layout.addWidget(self.fault)
        self.empty_gripper = QCheckBox('Confirm simulated gripper is empty')
        self.execution_layout.addWidget(self.empty_gripper)
        self._button(self.execution_layout, 'Reconcile simulation', self.reconcile)
        self.status = QLabel('Simulation idle')
        self.status.setWordWrap(True)
        self.execution_layout.addWidget(self.status)
        self.joints = QTableWidget(6, 4)
        self.joints.setHorizontalHeaderLabels(['Joint', 'Measured rad', 'Planned rad', 'Error rad'])
        self.joints.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.execution_layout.addWidget(self.joints)
        self.history_text = QPlainTextEdit()
        self.history_text.setReadOnly(True)
        history_layout.addWidget(self.history_text)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._identified)
        self.process.errorOccurred.connect(lambda _error: self._error(self.process.errorString()))
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        self._buttons()

    @property
    def running(self):
        return self.process.state() != QProcess.ProcessState.NotRunning

    def _button(self, layout, label, slot):
        button = QPushButton(label)
        button.clicked.connect(lambda _checked=False: self._guard(slot))
        layout.addWidget(button)
        return button

    def _path(self, layout, label, directory=False):
        layout.addWidget(QLabel(label))
        row = QHBoxLayout()
        edit = QLineEdit()
        button = QPushButton('Browse…')
        def browse():
            path = (QFileDialog.getExistingDirectory(None, label,
                        options=QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog)
                    if directory else
                    QFileDialog.getOpenFileName(None, label, '', 'JSON (*.json)',
                        options=QFileDialog.Option.DontUseNativeDialog)[0])
            if path:
                edit.setText(path)
        button.clicked.connect(browse)
        row.addWidget(edit)
        row.addWidget(button)
        layout.addLayout(row)
        return edit

    def _guard(self, action):
        try:
            action()
        except (ValueError, OSError, RuntimeError, KeyError, TypeError, sqlite3.Error) as error:
            self._error(str(error))
        self._buttons()

    def _error(self, message):
        self.status.setText(message)
        self.perception_info.appendPlainText(message)

    def choose_session(self):
        path = QFileDialog.getExistingDirectory(None, 'Open session folder',
            options=QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog)
        if path:
            self.open_session(path)

    def open_session(self, path):
        if self.running or (self.runtime and self.runtime.state['status'] in ('running', 'held')):
            raise ValueError('stop identification and execution before switching sessions')
        if self.runtime:
            self.runtime.close()
            self.runtime = None
        runtime_type = EmulationRuntime if (Path(path) / 'virtual-nuc/world.json').exists() else SimulationRuntime
        self.runtime = runtime_type(self.repository, path)
        self.projector = self.runtime.device.projector if isinstance(self.runtime, EmulationRuntime) else None
        self.start_button.setText('Run emulation' if isinstance(self.runtime, EmulationRuntime) else 'Run simulation')
        self.timer.start()
        self.command = self.proposal = self.cloud_path = None
        self._history_key = None
        self.session_label.setText(f'SIMULATION · {Path(path).resolve()}')
        self.refresh_proposals()
        self._buttons()
        self.changed.emit()

    def identify(self):
        if self.runtime is None or self.running:
            raise ValueError('open a session and wait for the current identification')
        product = Path(self.product.text()).resolve()
        priors = Path(self.priors.text()).resolve()
        request = json.loads(priors.read_text())
        if request['scene_revision'] != self.runtime.review.snapshot()['revision']:
            raise ValueError('initial poses use a stale scene revision')
        if not (product / 'manifest.json').is_file():
            raise ValueError('choose a verified crop product directory')
        self.output = self.runtime.directory / 'perception' / uuid4().hex
        self.process.setProgram(sys.executable)
        self.process.setArguments(['-m', 'stacking_excavator.perception.cli', 'identify', str(product),
            str(priors), str(self.output), '--repository', str(self.repository)])
        self.process.start()
        self.perception_info.setPlainText('Identifying recorded crop in a separate process…')

    def stop_identification(self):
        if self.running:
            pid = self.process.processId()
            self.process.terminate()
            # Kill only this owned worker if it cannot finish a native ICP call.
            QTimer.singleShot(1500, lambda: self._kill_worker(pid))

    def _kill_worker(self, pid):
        if self.running and self.process.processId() == pid:
            self.process.kill()

    def _read_output(self):
        self.perception_info.appendPlainText(bytes(self.process.readAllStandardOutput()).decode(errors='replace'))

    def _identified(self, code, status):
        if code == 0 and status == QProcess.ExitStatus.NormalExit:
            self._guard(lambda: self.load_proposal(self.output / 'proposal.json'))
        else:
            self._error('Identification stopped or failed; no proposal was accepted.')
        self._buttons()

    def choose_proposal(self):
        path = QFileDialog.getOpenFileName(None, 'Open perception proposal', '', 'JSON (*.json)',
            options=QFileDialog.Option.DontUseNativeDialog)[0]
        if path:
            self.load_proposal(path)

    def load_proposal(self, path):
        if self.runtime is None:
            raise ValueError('open a simulation session first')
        path = Path(path).resolve()
        proposal = json.loads(path.read_text())
        token = self.runtime.review.stage(proposal)
        # Session-local display references survive reopening. Registration inputs
        # remain verified independently by the perception worker.
        references = self.runtime.directory / 'proposal_paths.json'
        paths = json.loads(references.read_text()) if references.exists() else {}
        paths[token] = str(path)
        atomic_json(references, paths)
        self.refresh_proposals(token)
        self._history_key = None
        self.changed.emit()

    def refresh_proposals(self, selected=None):
        self.proposals.blockSignals(True)
        self.proposals.clear()
        for row in self.runtime.review.proposals():
            p = row['proposal']
            self.proposals.addItem(f"{p['kind']} · {row['decision'] or 'pending'} · {p['sha256'][:10]}", row)
            if p['sha256'] == selected:
                self.proposals.setCurrentIndex(self.proposals.count()-1)
        self.proposals.blockSignals(False)
        self.select_proposal()

    def select_proposal(self, *_args):
        row = self.proposals.currentData()
        self.proposal = None if row is None else row['proposal']
        self.cloud_path = None
        if self.proposal:
            p = self.proposal
            lines = [f"{p['kind']} · {p['frame']} · revision {p['scene_revision']}",
                     f"Decision: {row['decision'] or 'pending'} · {'passes gates' if p['valid'] else 'cannot accept'}"]
            for obj in p['objects']:
                lines.append(f"\nStone {obj['object_id']}: {', '.join(obj['reasons']) or 'passes gates'}")
                metrics = obj.get('metrics')
                if metrics:
                    lines.append(f"Coverage: target {metrics['target_coverage']:.1%}, source {metrics['source_coverage']:.1%}")
                    if metrics['rmse'] is not None:
                        lines.append(f"RMSE: {metrics['rmse'] * 1000:.1f} mm")
            for i, opening in enumerate(p.get('opening', [])):
                lines.append(f"Opening view {i+1}: estimated {opening.get('estimated')}, measured {opening.get('measured')}")
            for alignment in p.get('gripper_alignment', []):
                if alignment.get('transform') is not None:
                    lines.append(f"{alignment['frame_id']}: correction {alignment['translation_m'] * 1000:.1f} mm / "
                                 f"{alignment['rotation_degrees']:.1f}° · {', '.join(alignment['reasons']) or 'passes gates'}")
            lines.append('\nFull proposal and provenance are available in History.')
            self.perception_info.setPlainText('\n'.join(lines))
            refs = self.runtime.directory / 'proposal_paths.json'
            paths = json.loads(refs.read_text()) if refs.exists() else {}
            path = paths.get(self.proposal['sha256'])
            if path and Path(path).with_name('processed.ply').is_file():
                cloud = Path(path).with_name('processed.ply')
                if file_hash(cloud) == self.proposal.get('processed_cloud_sha256'):
                    self.cloud_path = cloud
                else:
                    self.perception_info.appendPlainText('Display cloud is missing its matching hash; showing poses only.')
        self._buttons()
        self.changed.emit()

    def decide(self, accept):
        if self.runtime is None or self.proposal is None:
            raise ValueError('select a staged proposal')
        self.runtime.decide(self.proposal['sha256'], accept=accept,
                            expected_revision=self.proposal['scene_revision'])
        self.refresh_proposals(self.proposal['sha256'])
        self._history_key = None
        self.changed.emit()

    def choose_plan(self):
        if self.runtime is None:
            raise ValueError('open a simulation session first')
        path = QFileDialog.getOpenFileName(None, 'Load committed excavator placement', '', 'Stacking run (*.stack)',
            options=QFileDialog.Option.DontUseNativeDialog)[0]
        if path:
            self.set_command(load_command(self.repository, path))

    def choose_emulation(self):
        if self.runtime is None:
            raise ValueError('open a session folder first')
        path = QFileDialog.getOpenFileName(None, 'Choose emulation placement', '', 'Stacking run (*.stack)',
            options=QFileDialog.Option.DontUseNativeDialog)[0]
        if not path:
            return
        plans = StackRunStore.load_committed_placement_plans(path)
        choices = [f'Step {p.step} — stone {p.problem.action.object_id}' for p in plans if p.result.succeeded]
        if not choices:
            raise ValueError('run has no committed motion plans')
        choice, ok = QInputDialog.getItem(None, 'Emulation placement', 'Placement', choices, 0, False)
        if ok:
            self.load_emulation(path, step=[p for p in plans if p.result.succeeded][choices.index(choice)].step)

    def load_emulation(self, path, *, step):
        if self.runtime is None or self.running or self.runtime.state['status'] in ('running', 'held'):
            raise ValueError('open an idle session before loading emulation')
        if not isinstance(self.runtime, EmulationRuntime):
            directory = self.runtime.directory
            self.runtime.close()
            self.runtime = EmulationRuntime(self.repository, directory)
        self.set_command(self.runtime.load_placement(path, step=step))
        self.projector = self.runtime.device.projector
        self.start_button.setText('Run emulation')
        self.session_label.setText(f'CLOSED-LOOP EMULATION · {self.runtime.directory.resolve()}')
        self.plan_label.setText(f'{Path(path).name} · step {step} · capture and review gates enabled\n'
                               'Solid: virtual world · gold: held stone · translucent blue: accepted estimate')
        self.changed.emit()

    def decide_capture(self, accept):
        gate = getattr(self.runtime, 'gate', None)
        if not gate or gate['stage'] != 'review':
            raise ValueError('wait for the current capture to finish identification')
        self.runtime.decide(gate['token'], accept=accept, expected_revision=gate['revision'])
        self.refresh_proposals()
        self.changed.emit()

    def load_demo(self, regrasp):
        if self.runtime is None:
            raise ValueError('open a simulation session first')
        self.set_command(demonstration_command(self.runtime.device, regrasp=regrasp))

    def set_command(self, command):
        if self.runtime.state['status'] in ('running', 'held'):
            raise ValueError('stop the active attempt before loading another command')
        if isinstance(self.runtime, EmulationRuntime) and command is not getattr(self.runtime, 'prepared_command', None):
            raise ValueError('use Load closed-loop emulation for this session; open a new session for joint demos')
        self.command = command
        self.command_revision = self.runtime.review.snapshot()['revision']
        self.plan_label.setText(f'Step {command.step} · {len(command.segments)} segments · revision {self.command_revision}')
        self._buttons()

    def start(self):
        if self.runtime is None or self.command is None:
            raise ValueError('load a simulation command first')
        self.runtime.start(self.command, expected_revision=self.command_revision)
        self.changed.emit()

    def control(self, method):
        if self.runtime is None:
            raise ValueError('open a simulation session first')
        getattr(self.runtime, method)()
        self.changed.emit()

    def reconcile(self):
        if self.runtime is None:
            raise ValueError('open a simulation session first')
        self.runtime.reconcile(empty_gripper=self.empty_gripper.isChecked())
        self.command = None  # Deliberately load a fresh command after reconciliation.
        self.empty_gripper.setChecked(False)
        self._history_key = None

    def tick(self):
        if self.runtime is None:
            return
        try:
            device = self.runtime.device
            device.stalled = self.fault.currentIndex() == 1
            device.feedback_enabled = self.fault.currentIndex() != 2
            device.gripper_known = self.fault.currentIndex() != 3
            if isinstance(self.runtime, EmulationRuntime):
                self.runtime.delivery_paused = self.fault.currentIndex() == 4
                device.missing_stone = self.fault.currentIndex() == 5
                device.slip = np.array([.1 if self.fault.currentIndex() == 6 else 0., 0., 0.])
            snapshot = self.runtime.tick()
            if self.heartbeat.isChecked():
                self.runtime.heartbeat()
            self.status.setText(f"SIMULATION · {snapshot['status']} · segment {snapshot['segment_index']+1} "
                f"sample {snapshot['sample_index']+1}\nOpening: {snapshot['measured']['opening']} · "
                f"needs_replan: {snapshot['needs_replan']}")
            if isinstance(self.runtime, EmulationRuntime):
                self.status.setText(self.status.text() + '\n' + snapshot['emulation_stage'])
                if snapshot.get('emulation_error'):
                    self.status.setText(self.status.text() + '\n' + snapshot['emulation_error'])
                gate = self.runtime.gate
                if gate and gate['stage'] == 'review' and (self.proposal is None or self.proposal['sha256'] != gate['token']):
                    self.refresh_proposals(gate['token'])
            for i, (name, measured) in enumerate(zip(device.joint_names, snapshot['measured']['positions'])):
                planned = None if snapshot['planned'] is None else snapshot['planned'][i]
                if self.command is not None and snapshot['status'] not in ('running', 'held'):
                    planned = self.command.segments[0].positions[0, i]
                values = [name, f'{measured:.4f}', '—' if planned is None else f'{planned:.4f}',
                          '—' if planned is None else f'{planned-measured:.4f}']
                for column, value in enumerate(values):
                    self.joints.setItem(i, column, QTableWidgetItem(value))
            key = (snapshot['command_id'], snapshot['event_sequence'], snapshot['accepted_scene']['revision'])
            if key != self._history_key:
                self.history_text.setPlainText('\n'.join(json.dumps(e) for e in self.runtime.history()) +
                    '\n\nPerception reviews:\n' + json.dumps(self.runtime.review.proposals(), indent=2))
                self._history_key = key
            self._buttons()
            self.changed.emit()
        except (ValueError, OSError, RuntimeError, KeyError, sqlite3.Error) as error:
            self.runtime.device.stop()
            self.timer.stop()
            self._error(str(error))

    def _buttons(self):
        available = self.runtime is not None
        self.identify_button.setEnabled(available and not self.running)
        self.stop_button.setEnabled(self.running)
        row = self.proposals.currentData()
        current = available and row is not None and row['decision'] is None and (
            row['proposal']['scene_revision'] == self.runtime.review.snapshot()['revision'])
        self.accept_button.setEnabled(bool(current and row['proposal']['valid']))
        self.reject_button.setEnabled(bool(current))
        self.start_button.setEnabled(available and self.command is not None and
            self.command.command_id != self.runtime.state['command_id'] and
            self.runtime.state['status'] not in ('running', 'held') and not self.runtime.state['needs_replan'])
        gate = getattr(self.runtime, 'gate', None)
        reviewing = bool(gate and gate['stage'] == 'review' and self.runtime.state['status'] == 'held')
        self.accept_capture_button.setEnabled(reviewing and gate['proposal']['valid'])
        self.reject_capture_button.setEnabled(reviewing)

    def execution_frame(self):
        if self.runtime is None:
            return None
        if self.projector is None:
            profile = load_planning_profile(self.repository / 'apps/excavator/config/planning.yml')
            self.projector = ExcavatorRobotTrajectoryProjector(profile, self.repository)
        snapshot = self.runtime.snapshot()
        measured = snapshot['measured']
        # Unknown opening is labelled in the panel; use the last simulator value
        # solely to display its mechanical state.
        parts = self.projector.project(np.array(measured['positions']), self.runtime.device.opening)
        objects = tuple(RenderObject(f'accepted/{i}', int(i), matrix_pose(value['pose']), 'placed')
                        for i, value in snapshot['accepted_scene']['poses'].items())
        overlays = ()
        if isinstance(self.runtime, EmulationRuntime) and snapshot['virtual_world']:
            world = snapshot['virtual_world']
            # Solid geometry is simulator truth. Translucent blue geometry is
            # the last accepted estimate; neither is supplied as an ICP answer.
            overlays = tuple(RenderObject(o.instance_id, o.object_id, o.pose, 'candidate_solved',
                                          opacity=.35, color=(.2, .7, 1.)) for o in objects)
            attachment = snapshot['accepted_scene']['attachment']
            if attachment and world['attached_id'] == attachment['object']['object_id']:
                estimate = self.runtime.device.gripper_pose() @ np.array(attachment['object']['pose'])
                overlays += (RenderObject('accepted/attachment', world['attached_id'], matrix_pose(estimate),
                                           'candidate_solved', opacity=.35, color=(.2, .7, 1.)),)
            objects = tuple(RenderObject(f'world/{i}', int(i), matrix_pose(pose), 'placed',
                                        color=(1., .72, .12) if int(i) == world['attached_id'] else None)
                            for i, pose in world['poses'].items())
        return OperationsFrame(0 if self.command is None else self.command.step, objects, overlays, (), (), None, None, None, 0, 0,
                               robot_parts=parts, environment_boxes=())

    def close(self):
        self.timer.stop()
        if self.runtime:
            self.runtime.close()
