"""Operator-gated direct pick/place execution; no ROS or Qt imports."""

from dataclasses import dataclass
import queue
import threading
import time
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation

from stacking_common.artifacts import StackRunStore, build_execution_command
from stacking_common.artifacts.execution import encode_execution_command
from stacking_common.execution import ExecutionEvent, ExecutionFailure, ExecutionResult, ExecutionStatus
from stacking_common.execution.values import canonical_json

from .motion import CartesianStreamLimits, pose_error
from .runtime import HardwareError
from .saved_plan import validate_command


@dataclass(frozen=True)
class OperatorGate:
    token: str
    command_id: str
    step: int
    name: str
    message: str
    manual: bool = False


class TabletopMotionExecutor:
    """One worker executes commands; UI threads only submit control requests.

    Hold is observed within arm sampling/waits. During an indivisible gripper
    operation it is deferred until that bounded operation finishes. Cancel is
    immediate at the next runtime check and never opens the gripper.
    """

    def __init__(self, saved, arm, gripper, *, gate_changed=None, telemetry=None, scan_handler=None):
        if arm.gripper is not gripper or arm.runtime is not gripper.runtime:
            raise ValueError("executor needs paired arm/gripper adapters")
        self.saved, self.arm, self.gripper = saved, arm, gripper
        self.runtime = arm.runtime
        self.gate_changed = gate_changed or (lambda value: None)
        self.telemetry = telemetry or (lambda value: None)
        self.scan_handler = scan_handler
        self._scan_request = None
        self._lock = threading.Lock()
        self._active = None
        self._gate = None
        self._approved = threading.Event()
        self._hold = threading.Event()
        self._resume = threading.Event()
        self._manual = queue.Queue()
        self._polling = False
        self._busy_gripper = False
        self._last_telemetry = float("-inf")
        self._last_progress = float("-inf")

    @property
    def active_command_id(self):
        with self._lock:
            return self._active

    def approve(self, token):
        with self._lock:
            if self._gate is None or self._gate.token != token or self._scan_request is not None:
                return False
            self._approved.set()
            return True

    def request_scan(self, token, kind, camera, directory):
        if kind not in ('wrist', 'pivot') or camera not in ('primary', 'secondary'):
            raise ValueError('unsupported scan or FEMTO selection')
        with self._lock:
            if (self.scan_handler is None or self._gate is None or self._gate.token != token
                    or self._gate.name != 'place' or self._approved.is_set() or self._scan_request is not None):
                return False
            self._scan_request = (kind, camera, directory)
            return True

    def adjust(self, token, joint=None, direction=0):
        with self._lock:
            if self._gate is None or self._gate.token != token or not self._gate.manual or self._approved.is_set():
                return False
            if joint is not None and (joint not in range(6) or direction not in (-1, 1)):
                raise ValueError("manual correction requires a joint and +/- direction")
            self._manual.put((token, joint, direction))
            return True

    def hold(self, command_id):
        with self._lock:
            if command_id == self._active:
                self._resume.clear()
                self._hold.set()

    def resume(self, command_id):
        with self._lock:
            if command_id == self._active:
                self._resume.set()

    def cancel(self, command_id):
        with self._lock:
            if command_id == self._active:
                self.runtime.cancel_event.set()

    def _emit(self, status, *, reason=None, failure=None, **diagnostics):
        event = ExecutionEvent(self._command.command_id, self._sequence + 1, time.time_ns(), status,
                               segment_index=self._segment, failure=failure, reason=reason,
                               diagnostics={"phase": self._phase, "step": self._command.step, **diagnostics})
        try:
            self._progress(event)
        except Exception:
            self._journal_failed = True
            raise
        self._sequence += 1
        return event

    def _health(self):
        self.arm._check_motion(self._attached)

    def _stationary(self, timeout=5.0):
        deadline = self.runtime.deadline(timeout) - self.runtime.paused_seconds
        previous = self.arm.feedback.arm_state()[0]
        since = self.runtime.active_time()
        while True:
            self._health()
            now = self.runtime.active_time()
            if now >= deadline:
                raise HardwareError(ExecutionFailure.TRACKING_ERROR, "robot did not become stationary for operator gate")
            joints = self.arm.feedback.arm_state()[0]
            if np.max(np.abs(joints - previous)) > np.deg2rad(.5):
                since, previous = now, joints
            elif now - since >= .25:
                return
            self.runtime.sleep(.02)

    def _poll(self):
        if self._polling:
            return
        self._polling = True
        try:
            if self._hold.is_set() and not self._busy_gripper:
                paused_at = self.runtime.clock()
                if not self.arm.hold_actual_pose():
                    raise HardwareError(ExecutionFailure.CONTROLLER_FAULT, "could not request actual-pose hold")
                self._stationary()
                self._emit("held", hold_confirmed=True)
                while not self._resume.is_set():
                    self._health()
                    self._send_telemetry()
                    self.runtime.sleep(.02)
                self.arm.validate_start(self._reference)
                self._hold.clear()
                self._resume.clear()
                if self._contact:
                    self.arm.set_contact_compliance()
                self.runtime.paused_seconds += self.runtime.clock() - paused_at
                self._emit("running", operator_decision="resume")
            self._send_telemetry()
        finally:
            self._polling = False

    def _send_telemetry(self, force=False):
        now = self.runtime.clock()
        if not force and now - self._last_telemetry < .1:
            return
        self._last_telemetry = now
        joints, _ = self.arm.feedback.arm_state()
        gripper = self.gripper.feedback.gripper_state()
        pose = self._release_pose
        if self._attached and self._attachment is not None:
            pose = _pose(self.arm.kinematics.grasp_pose(joints) @ self._attachment)
        self.telemetry({"step": self._command.step, "object_id": self._plan.problem.action.object_id,
                        "joints": joints.tolist(), "gripper_raw": gripper.position,
                        "holding": gripper.holding, "phase": self._phase,
                        "object_pose": None if pose is None else np.asarray(pose).tolist()})

    def _gate_wait(self, name, message, *, manual=False):
        self._phase = name
        self._stationary()
        gate = OperatorGate(uuid4().hex, self._command.command_id, self._command.step, name, message, manual)
        anchor = self.arm.feedback.arm_state()[0]
        offset = np.zeros(6)
        with self._lock:
            self._approved.clear()
            self._gate = gate
        self._emit("held", gate=name, gate_token=gate.token, hold_confirmed=True)
        self.gate_changed(gate)
        try:
            while not self._approved.is_set():
                self._health()
                with self._lock:
                    scan, self._scan_request = self._scan_request, None
                    if scan is not None:
                        self._gate = None
                if scan is not None:
                    self.gate_changed(None)
                    self.scan_handler(self, scan)
                    self._phase = name
                    self._stationary()
                    gate = OperatorGate(uuid4().hex, self._command.command_id, self._command.step, name,
                        'Scan finished. Inspect the held object and saved placement, then Continue.', manual)
                    with self._lock:
                        self._approved.clear()
                        self._gate = gate
                    self._emit('held', gate=name, gate_token=gate.token, hold_confirmed=True)
                    self.gate_changed(gate)
                    continue
                try:
                    token, joint, direction = self._manual.get_nowait()
                except queue.Empty:
                    self.runtime.wait(.02, self._health)
                    continue
                if token != gate.token:
                    continue
                requested = offset.copy()
                if joint is None:
                    requested[:] = 0
                else:
                    requested[joint] += direction * np.deg2rad(.5)
                if np.max(np.abs(requested)) > np.deg2rad(5) + 1e-9:
                    self._emit("held", manual_rejected="correction limit is ±5 degrees")
                    continue
                current = self.arm.feedback.arm_state()[0]
                target = anchor + requested
                self.arm.execute_path([current, target], attached=self._attached,
                                      limits=CartesianStreamLimits(100., .10, .02, .20),
                                      progress=lambda _i, _n, q: setattr(self, "_reference", q))
                offset = requested
                self._reference = target
                self._emit("held", manual_offset_degrees=np.rad2deg(offset).tolist())
            self._health()
            self._emit("running", operator_decision=f"approve_{name}",
                       manual_offset_degrees=np.rad2deg(offset).tolist())
            return offset
        finally:
            with self._lock:
                self._gate = None
                self._approved.clear()
            self.gate_changed(None)

    def _profile(self, contact):
        self._contact = contact
        if contact:
            self.arm.set_contact_compliance()
        else:
            self.arm.set_rigid_tracking()

    def _move(self, index, positions=None):
        self._segment = index
        segment = self._command.segments[index]
        self._phase = segment.name
        self._emit("running", segment_started=True)
        path = segment.positions if positions is None else positions

        def progress(reference_index, count, joints):
            self._reference = joints
            if self.runtime.clock() - self._last_progress >= 1.0 or reference_index == count - 1:
                self._last_progress = self.runtime.clock()
                self._emit("running", reference_index=reference_index, reference_count=count)

        if index == 0:
            split = len(path) - self.saved.grasp_steps
            self._profile(False)
            self.arm.execute_path(path[:split + 1], attached=self._attached, progress=progress)
            self._profile(True)
            self.arm.execute_path(path[split:], attached=self._attached, progress=progress)
        elif index == 3:
            # Transfer ended at pre-place; this whole segment is the final
            # contact approach, including when len(path) == grasp_steps.
            self._profile(True)
            self.arm.execute_path(path, attached=self._attached, progress=progress)
        else:
            self.arm.execute_path(path, attached=self._attached, progress=progress)

    def _grip(self, action):
        self._busy_gripper = True
        try:
            action()
        finally:
            self._busy_gripper = False

    def execute(self, command, *, progress):
        with self._lock:
            if self._active is not None:
                raise RuntimeError("an execution command is already active")
            self._active = command.command_id
        self._command, self._progress = command, progress
        self._sequence, self._segment, self._phase = 0, None, "preflight"
        self._journal_failed = False
        self._attached, self._contact = False, False
        self._attachment = self._release_pose = None
        self._reference = self.saved.planning.robot.home_positions.copy()
        old_poll = self.runtime.poll
        try:
            self._plan = next(plan for plan in self.saved.plans if plan.step == command.step)
            expected = build_execution_command(self._plan, self.saved.context,
                execution_attempt=command.execution_attempt, command_id=command.command_id)
            if canonical_json(encode_execution_command(expected)) != canonical_json(encode_execution_command(command)):
                raise ValueError("command differs from the validated saved plan")
            validate_command(command, self.saved.planning, self.saved.grasp_steps)
            self.saved.verify_assets()
            self._emit("running")
            self.arm.wait_until_ready()
            self.gripper.wait_until_ready()
            self.arm.validate_start(command.segments[0].positions[0])
            for event in command.grasp_events:
                actual = self.arm.kinematics.grasp_pose(command.segments[event.segment_index].positions[event.sample_index])
                distance, angle = pose_error(actual, _matrix(event.frame_from_grasp))
                if distance > .001 or angle > np.deg2rad(.5):
                    raise ValueError("saved grasp-event pose disagrees with the recorded robot model")
            hand = self.gripper.feedback.gripper_state()
            if hand.holding or hand.hold_goal is not None:
                raise ValueError("gripper already holds an object; reconcile before starting")
            self.runtime.poll = self._poll
            self._gate_wait("review", "Review the selected saved motion and confirm the pickup scene is ready. The gripper will open.")
            self._profile(False)
            self._grip(self.gripper.open)
            self._move(0)
            pick_offset = self._gate_wait("close", "Check the pickup pose, then close the gripper.", manual=True)
            self._grip(lambda: self.gripper.close(command.grasp_events[0].opening))
            self.gripper.require_hold()
            self._attached = True
            self._attachment = np.linalg.solve(
                self.arm.kinematics.grasp_pose(self.arm.feedback.arm_state()[0]),
                _matrix(self._plan.problem.pick_pose))
            self._profile(False)
            self._move(1, _taper(command.segments[1].positions, pick_offset))
            self._gate_wait("place", "Check the held object and saved placement before carrying it to the stack.")
            self._move(2)
            self._move(3)
            place_offset = self._gate_wait("release", "Check the placement and support, then release the object.", manual=True)
            joints = self.arm.feedback.arm_state()[0]
            self._release_pose = _pose(self.arm.kinematics.grasp_pose(joints) @ self._attachment)
            # OPEN disarms the guard; the preceding attachment remains an estimate
            # until the release command finishes. Do not require hold during OPEN.
            self._attached = False
            self._grip(self.gripper.open)
            self._profile(False)
            self._move(4, _taper(command.segments[4].positions, place_offset))
            self._send_telemetry(force=True)
            event = self._emit("succeeded", release_pose=self._release_pose.tolist())
        except (Exception, KeyboardInterrupt) as error:
            self.runtime.poll = None
            held = self.arm.hold_actual_pose()
            if self._journal_failed:
                # The callback may have persisted an event before raising. Its
                # durable state is uncertain; never guess another sequence ID.
                raise
            failure = (error.failure if isinstance(error, HardwareError) else
                       ExecutionFailure.CANCELLED if isinstance(error, KeyboardInterrupt) else
                       ExecutionFailure.INVALID_PLAN if isinstance(error, (ValueError, KeyError, StopIteration)) else
                       ExecutionFailure.CONTROLLER_FAULT)
            status = "cancelled" if failure is ExecutionFailure.CANCELLED else "failed"
            event = self._emit(status, failure=failure, reason=str(error) or "execution interrupted", hold_requested=held)
        finally:
            self.runtime.poll = old_poll
            with self._lock:
                self._active = self._gate = None
                self._scan_request = None
            self._hold.clear()
            self._resume.clear()
            self.gate_changed(None)
        return ExecutionResult(command.command_id, event.status, event.failure, event.reason, event.diagnostics)


def execute_saved_run(saved, executor, *, events=None, next_step=None):
    """Journal from this worker thread; queue durably before each physical step."""
    saved.verify_assets()
    checkpoint = StackRunStore.load_execution_checkpoint(saved.path)
    if checkpoint.requires_reconciliation:
        raise ValueError("previous physical execution requires reconciliation before another attempt")
    commands = StackRunStore.load_execution_commands(saved.path)
    results = []
    with StackRunStore.open_append(saved.path) as writer:
        for plan in saved.plans:
            if plan.step in checkpoint.completed_steps:
                continue
            if executor.runtime.cancel_event.is_set():
                break
            if next_step is not None:
                next_step(plan)
            attempt = max((item.execution_attempt for item in commands if item.step == plan.step), default=0) + 1
            command = build_execution_command(plan, saved.context, execution_attempt=attempt)
            writer.append_execution_command(command)
            def record(event):
                writer.append_execution_event(event)
                if events is not None:
                    events(event)
            result = executor.execute(command, progress=record)
            results.append(result)
            if result.status is not ExecutionStatus.SUCCEEDED:
                break
    return tuple(results)


def _matrix(pose):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_quat(np.asarray(pose)[3:]).as_matrix()
    value[:3, 3] = np.asarray(pose)[:3]
    return value


def _pose(matrix):
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()]


def _taper(path, offset):
    return np.asarray(path) + np.linspace(1., 0., len(path))[:, None] * offset
