"""Thread-safe feedback values; ROS messages never cross this boundary."""

from dataclasses import dataclass
import threading
import time

import numpy as np

from stacking_common.execution import ExecutionFailure

from .gripper import GripperHoldGuard, GRIPPER_CLOSED_POSITION, GRIPPER_OPEN_POSITION
from .motion import canonical_joint_positions, validate_pose
from .runtime import HardwareError


@dataclass(frozen=True)
class GripperState:
    position: float | None
    stamp: float
    sequence: int
    hold_goal: int | None
    hold_sequence: int
    holding: bool


class HardwareFeedback:
    """Local receipt times measure freshness, independent of the ROS clock.

    Encoder values are raw counts. The hold guard sees every valid callback,
    not just whichever sample the command worker happens to poll.
    """

    def __init__(self, *, clock=time.monotonic):
        self.clock = clock
        self._lock = threading.RLock()
        self._ready = False
        self._joints = None
        self._joint_stamp = float("-inf")
        self._pose = None
        self._pose_stamp = float("-inf")
        self._position = None
        self._gripper_stamp = float("-inf")
        self._sequence = 0
        self._hold_goal = None
        self._hold_sequence = 0
        self._expected_hold = None
        self._guard = GripperHoldGuard()
        self._fault = None

    def receive_ready(self, ready):
        with self._lock:
            self._ready = bool(ready)

    @property
    def ready(self):
        with self._lock:
            return self._ready

    def receive_joints(self, names, positions):
        joints = canonical_joint_positions(names, positions)
        if joints is None:
            return False
        with self._lock:
            self._joints = joints.copy()
            self._joint_stamp = self.clock()
        return True

    def receive_pose(self, pose):
        try:
            pose = validate_pose(pose)
        except (ValueError, TypeError):
            return False
        with self._lock:
            self._pose = pose.copy()
            self._pose_stamp = self.clock()
        return True

    def arm_state(self, maximum_age=1.0):
        with self._lock:
            now = self.clock()
            if self._joints is None or now - self._joint_stamp > maximum_age:
                raise HardwareError(ExecutionFailure.CONTROLLER_FAULT, "missing or stale /joint_states")
            if self._pose is None or now - self._pose_stamp > maximum_age:
                raise HardwareError(ExecutionFailure.CONTROLLER_FAULT, "missing or stale /ee_pose_ur5e")
            return self._joints.copy(), self._pose.copy()

    def actual_pose(self, maximum_age=1.0):
        with self._lock:
            if self._pose is None or self.clock() - self._pose_stamp > maximum_age:
                raise HardwareError(ExecutionFailure.CONTROLLER_FAULT, "missing or stale /ee_pose_ur5e")
            return self._pose.copy()

    def receive_gripper(self, position):
        if not np.isfinite(position) or not 50.0 <= position <= 1700.0:
            return False
        with self._lock:
            self._position = float(position)
            self._gripper_stamp = self.clock()
            self._sequence += 1
            if self._guard.observe(position):
                self.receive_fault("gripper opened during acknowledged hold")
        return True

    def receive_hold(self, goal):
        with self._lock:
            self._hold_goal = int(goal) if goal > 0 else None
            self._hold_sequence += 1
            if self._guard.armed and goal != self._expected_hold:
                self.receive_fault("acknowledged gripper hold was lost or changed")

    def receive_fault(self, reason):
        with self._lock:
            if str(reason).strip() and self._fault is None:
                self._fault = str(reason).strip()

    def check_gripper(self):
        with self._lock:
            if self._guard.armed and self.clock() - self._gripper_stamp > 3.0:
                self.receive_fault("gripper telemetry lost during acknowledged hold")
            if self._fault is not None:
                raise HardwareError(ExecutionFailure.GRIPPER_FAULT, self._fault)

    def gripper_state(self):
        with self._lock:
            return GripperState(self._position, self._gripper_stamp, self._sequence,
                                self._hold_goal, self._hold_sequence, self._guard.armed)

    def arm_hold(self, after_sequence):
        with self._lock:
            self.check_gripper()
            if (self._hold_sequence <= after_sequence or self._hold_goal is None
                    or not GRIPPER_CLOSED_POSITION <= self._hold_goal <= GRIPPER_OPEN_POSITION
                    or self._position is None or self.clock() - self._gripper_stamp > 3.5):
                return False
            self._expected_hold = self._hold_goal
            self._guard.arm(self._position)
            return True

    def disarm_hold(self):
        with self._lock:
            self._guard.disarm()
            self._expected_hold = None
