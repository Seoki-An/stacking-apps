"""Supervised SR-gripper client for the existing Dynamixel ROS driver."""

import numpy as np

from stacking_common.execution import ExecutionFailure

from .gripper import (
    GRIPPER_CLOSED_POSITION, GRIPPER_OPEN_POSITION, GRIPPER_GRASP_CLOSE_RATE,
    GRIPPER_GRASP_COMMAND_RATE_HZ, GRIPPER_GRASP_CLOSING_MARGIN,
    GripperContactDetector, gripper_position_for_opening_angle, gripper_position_ramp,
)
from .runtime import AdapterRuntime, HardwareError


class DynamixelAdapter:
    """Commands are blocking; telemetry and cancellation stay asynchronous.

    This is a client of ur5e_admittance's gripper_node, not an SDK/bus owner.
    Closing succeeds only after a fresh clear → service → hold acknowledgement.
    The caller owns operator gates and arm hold on a gripper-command failure.
    """

    def __init__(self, transport, *, runtime=None):
        self.transport = transport
        self.feedback = transport.feedback
        self.runtime = runtime if runtime is not None else AdapterRuntime()

    def check(self):
        self.runtime.check(self.transport)
        self.feedback.check_gripper()

    def wait_until_ready(self, timeout=20.0):
        deadline = self.runtime.deadline(timeout)

        def ready():
            state = self.feedback.gripper_state()
            return (self.transport.gripper_connected() and state.position is not None
                    and self.runtime.clock() - state.stamp <= 1.0)

        self.runtime.until(ready, deadline, self.check, "gripper driver/service/feedback not ready")

    def _state(self, maximum_age=3.5):
        self.check()
        state = self.feedback.gripper_state()
        if state.position is None or self.runtime.clock() - state.stamp > maximum_age:
            raise HardwareError(ExecutionFailure.GRIPPER_FAULT, "missing or stale /gripper/state")
        return state

    def open(self, timeout=7.0):
        """Explicit release; no open command is sent during construction/cleanup."""
        deadline = self.runtime.deadline(timeout)
        start = self._state(1.0)
        if not self.transport.gripper_connected():
            raise HardwareError(ExecutionFailure.GRIPPER_FAULT, "gripper driver disconnected")
        self.feedback.disarm_hold()
        self.transport.publish_gripper_flag(2)

        def reached():
            state = self._state()
            return state.sequence > start.sequence and abs(state.position - GRIPPER_OPEN_POSITION) <= 20

        self.runtime.until(reached, deadline, self.check, "gripper OPEN timed out")

    def close(self, opening_angle, timeout=15.0):
        """Close toward the saved opening with the legacy 0.40 rad margin."""
        goal = gripper_position_for_opening_angle(opening_angle, GRIPPER_GRASP_CLOSING_MARGIN)
        self.close_to_position(goal, timeout=timeout)

    def close_to_position(self, goal, timeout=15.0):
        deadline = self.runtime.deadline(timeout)
        if (isinstance(goal, bool) or not isinstance(goal, (int, np.integer))
                or not GRIPPER_CLOSED_POSITION <= goal <= GRIPPER_OPEN_POSITION):
            raise ValueError("raw gripper goal must be an integer in 91..1500")
        start = self._state(1.0)
        if start.holding:
            raise HardwareError(ExecutionFailure.GRIPPER_FAULT, "gripper already has an acknowledged hold")
        if goal > start.position:
            raise ValueError("CLOSE goal must not open the gripper")
        if not self.transport.gripper_connected():
            raise HardwareError(ExecutionFailure.GRIPPER_FAULT, "gripper driver disconnected")
        period = 1.0 / GRIPPER_GRASP_COMMAND_RATE_HZ
        ramp = gripper_position_ramp(start.position, goal, GRIPPER_GRASP_CLOSE_RATE * period)
        detector = GripperContactDetector(start.position, start.sequence)

        def stopped():
            state = self._state()
            if state.sequence <= start.sequence:
                return False
            return abs(goal - state.position) <= 20 or detector.observe(state.position, state.sequence)

        for position in ramp:
            self.check()
            if self.runtime.clock() + period >= deadline:
                raise HardwareError(ExecutionFailure.TIMEOUT, "gripper CLOSE timed out during ramp")
            self.transport.publish_gripper_position(position)
            self.runtime.wait(period, self.check)
            if stopped():
                break
        else:
            self.runtime.until(stopped, deadline, self.check, "gripper CLOSE timed out waiting for contact")
        self._install_hold(start.hold_sequence, deadline)

    def _install_hold(self, clear_after, deadline):
        def cleared():
            state = self._state()
            return state.hold_sequence > clear_after and state.hold_goal is None

        self.runtime.until(cleared, min(deadline, self.runtime.clock() + 2.0), self.check,
                           "gripper did not freshly clear the previous hold")
        ack_after = self.feedback.gripper_state().hold_sequence
        future = self.transport.request_gripper_hold()
        try:
            self.runtime.until(future.done, min(deadline, self.runtime.clock() + 2.0), self.check,
                               "gripper hold service timed out")
            try:
                response = future.result()
            except Exception as error:
                raise HardwareError(ExecutionFailure.GRIPPER_FAULT, f"gripper hold service failed: {error}") from error
            if response is None or not response.success:
                reason = "no response" if response is None else str(response.message)
                raise HardwareError(ExecutionFailure.GRIPPER_FAULT, f"gripper hold failed: {reason}")
            self.runtime.until(lambda: self.feedback.arm_hold(ack_after),
                               min(deadline, self.runtime.clock() + 2.0), self.check,
                               "gripper hold was not freshly acknowledged")
        finally:
            if not future.done():
                future.cancel()

    def require_hold(self):
        self._state(3.0)
        if not self.feedback.gripper_state().holding:
            raise HardwareError(ExecutionFailure.GRIPPER_FAULT, "no acknowledged gripper hold for lift/carry")
