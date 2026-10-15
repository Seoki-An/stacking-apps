"""UR5e Cartesian-reference adapter for the existing admittance controller."""

import numpy as np

from stacking_common.execution import ExecutionFailure

from .motion import (
    CartesianStreamLimits,
    UR_JOINT_NAMES,
    pose_error,
    sample_joint_path,
    validate_joint_path,
)
from .runtime import AdapterRuntime, HardwareError


# Existing Float64MultiArray contract: linear M/B/K, then angular M/B/K.
RIGID_ADMITTANCE_GAINS = (
    10., 10., 10., 650., 650., 650., 2000., 2000., 2000.,
    5., 5., 5., 150., 150., 150., 300., 300., 300.,
)
CONTACT_ADMITTANCE_GAINS = (
    8., 8., 8., 600., 600., 600., 1200., 1200., 1200.,
    5., 5., 5., 100., 100., 100., 200., 200., 200.,
)


class UR5eAdapter:
    """Execute one reviewed joint path using physical tool0 FK references.

    The sequence coordinator owns phase selection and operator gates. Attached
    paths explicitly require the Dynamixel adapter's acknowledged hold. A device
    failure/cancel requests rigid hold at measured pose, never gripper release.
    A hold publication is best effort and does not confirm that hardware stopped.
    """

    def __init__(self, transport, kinematics, *, gripper=None, runtime=None):
        self.transport = transport
        self.feedback = transport.feedback
        self.kinematics = kinematics
        self.gripper = gripper
        self.runtime = (runtime if runtime is not None else
                        gripper.runtime if gripper is not None else AdapterRuntime())
        if gripper is not None and (gripper.transport is not transport or gripper.runtime is not self.runtime):
            raise ValueError("arm and gripper must share one transport and command runtime")

    def _check(self):
        self.runtime.check(self.transport)
        self.feedback.check_gripper()

    def _check_motion(self, attached=False):
        self._check()
        if not self.feedback.ready or not self.transport.arm_connected():
            raise HardwareError(ExecutionFailure.CONTROLLER_FAULT, "UR5e controller no longer ready/connected")
        self.feedback.arm_state()
        if attached:
            if self.gripper is None:
                raise HardwareError(ExecutionFailure.GRIPPER_FAULT, "attached motion requires a gripper adapter")
            self.gripper.require_hold()

    def wait_until_ready(self, timeout=20.0):
        def ready():
            if not self.feedback.ready or not self.transport.arm_connected():
                return False
            try:
                self.feedback.arm_state()
            except HardwareError:
                return False
            return True

        self.runtime.until(ready, self.runtime.deadline(timeout), self._check,
                           "UR5e controller and fresh feedback not ready")

    def start_offset(self, expected):
        """Return the shortest safe joint offset from live state to ``expected``."""
        expected = validate_joint_path([expected])[0]
        self._check_motion()
        actual, measured_pose = self.feedback.arm_state()
        delta = expected - actual
        delta[:5] = (delta[:5] + np.pi) % (2 * np.pi) - np.pi
        # Wrist 3 must return to the recorded physical cable branch literally.
        delta[5] = expected[5] - actual[5]
        translation, rotation = pose_error(measured_pose, self.kinematics.pose(actual))
        if translation > 0.015 or rotation > np.deg2rad(5.0):
            raise HardwareError(ExecutionFailure.TRACKING_ERROR, "UR5e model/controller FK mismatch exceeds 15 mm / 5 degrees")
        return actual, delta

    def validate_start(self, expected):
        """Match saved joints and planning FK to the live controller feedback."""
        _actual, delta = self.start_offset(expected)
        error = np.abs(delta)
        if np.max(error) > np.deg2rad(2.0):
            index = int(np.argmax(error))
            raise HardwareError(
                ExecutionFailure.TRACKING_ERROR,
                "UR5e start differs from plan by more than 2 degrees: "
                f"{UR_JOINT_NAMES[index]} error={np.rad2deg(error[index]):.1f} degrees",
            )

    def set_rigid_tracking(self):
        self._set_profile(RIGID_ADMITTANCE_GAINS, False, 0.0)

    def set_contact_compliance(self):
        self._set_profile(CONTACT_ADMITTANCE_GAINS, True, 1.6)

    def _set_profile(self, gains, use_wrench, settle_time):
        try:
            for _ in range(3):
                self._check_motion()
                self.transport.publish_gains(gains)
                self.transport.publish_wrench(use_wrench)
                self.runtime.wait(0.05, self._check_motion)
            self.runtime.wait(settle_time, self._check_motion)
        except (Exception, KeyboardInterrupt):
            self.hold_actual_pose()
            raise

    def execute_path(self, path, *, limits=CartesianStreamLimits(), attached=False,
                     timeout=120.0, settle_timeout=15.0, progress=None):
        """Retiming preserves the joint path and all three reference speed limits.

        Progress reports commanded samples; measured state remains in feedback.
        The deadline includes sampling, streaming and endpoint settling. Slow
        publishing never causes a burst of commands to catch up with a deadline.
        """
        path = validate_joint_path(path)
        limits.validate()
        deadline = self.runtime.deadline(timeout) - self.runtime.paused_seconds
        self.runtime.deadline(settle_timeout)  # Validate before any command.
        try:
            self.validate_start(path[0])

            def check():
                self._check_motion(attached)
                if self.runtime.active_time() >= deadline:
                    raise HardwareError(ExecutionFailure.TIMEOUT, "UR5e path timed out")

            samples = sample_joint_path(path, self.kinematics, limits, check=check)
            check()
            # Sampling may take time: verify the robot did not move meanwhile.
            self.validate_start(path[0])
            for index, (joints, pose) in enumerate(samples):
                check()
                published_at = self.runtime.clock()
                self.transport.publish_pose(pose)
                if progress is not None:
                    progress(index, len(samples), joints.copy())
                self.runtime.wait(max(0., published_at + 1.0 / limits.rate_hz - self.runtime.clock()), check)
            self._settle(samples[-1][1], min(deadline, self.runtime.active_time() + settle_timeout), check)
        except (Exception, KeyboardInterrupt):
            self.hold_actual_pose()
            raise

    def _settle(self, target, deadline, check):
        stable_since = None
        while True:
            check()
            now = self.runtime.active_time()
            if now >= deadline:
                raise HardwareError(ExecutionFailure.TIMEOUT, "UR5e endpoint did not settle")
            self.transport.publish_pose(target)
            translation, rotation = pose_error(self.feedback.actual_pose(), target)
            if translation <= 0.010 and rotation <= np.deg2rad(5.0):
                if stable_since is None:
                    stable_since = now
                elif now - stable_since >= 0.25:
                    return
            else:
                stable_since = None
            self.runtime.wait(min(0.02, deadline - now), check)

    def hold_actual_pose(self):
        """Request rigid hold even after cancellation; return publication success."""
        try:
            if not self.transport.ok():
                return False
            pose = self.feedback.actual_pose()
            for _ in range(3):
                self.transport.publish_gains(RIGID_ADMITTANCE_GAINS)
                self.transport.publish_wrench(False)
                self.transport.publish_pose(pose)
                self.runtime.sleep(0.02)
            return True
        except Exception:
            return False
