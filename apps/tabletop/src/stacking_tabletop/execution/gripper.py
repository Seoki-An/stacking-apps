"""SR-gripper encoder mapping and contact/hold detection, ported from tabletop.

The raw mapping is deliberately unchanged: its 0.9 rad denominator differs
from the model's mechanical opening limit. Retuning requires physical grasps.
"""

import math

import numpy as np

GRIPPER_CLOSED_POSITION = 91
GRIPPER_OPEN_POSITION = 1500
GRIPPER_MAX_OPENING_ANGLE = 0.9
GRIPPER_MIN_OPENING_ANGLE = -0.1
GRIPPER_GRASP_CLOSING_MARGIN = 0.40
GRIPPER_GRASP_CLOSE_RATE = 500.0
GRIPPER_GRASP_COMMAND_RATE_HZ = 10.0
GRIPPER_CONTACT_MOVEMENT_COUNTS = 10.0
GRIPPER_CONTACT_STABLE_BAND_COUNTS = 5.0
GRIPPER_CONTACT_STABLE_SAMPLES = 25
GRIPPER_HOLD_OPENING_LIMIT_COUNTS = 50.0
GRIPPER_HOLD_OPENING_SAMPLES = 5


class GripperContactDetector:
    """Detect stable contact using distinct encoder samples.

    The stability reference is fixed for a candidate window, so slow cumulative
    motion cannot look like a stall. Stone motion simply begins a new window.
    """

    def __init__(
        self,
        initial_position: float,
        initial_sequence: int,
        movement_counts: float = GRIPPER_CONTACT_MOVEMENT_COUNTS,
        stable_band_counts: float = GRIPPER_CONTACT_STABLE_BAND_COUNTS,
        stable_samples: int = GRIPPER_CONTACT_STABLE_SAMPLES,
    ) -> None:
        if not np.isfinite(initial_position):
            raise ValueError("initial gripper position must be finite")
        if movement_counts <= 0.0 or stable_band_counts <= 0.0:
            raise ValueError("gripper contact thresholds must be positive")
        if stable_samples < 1:
            raise ValueError("stable gripper sample count must be positive")
        self.initial_position = float(initial_position)
        self.last_sequence = int(initial_sequence)
        self.movement_counts = float(movement_counts)
        self.stable_band_counts = float(stable_band_counts)
        self.stable_samples = int(stable_samples)
        self.stable_reference = None
        self.stable_sample_count = 0
        self.moved = False

    def observe(self, position: float, sequence: int) -> bool:
        """Return true after a complete stable window of fresh samples."""

        sequence = int(sequence)
        if sequence <= self.last_sequence:
            return False
        self.last_sequence = sequence
        position = float(position)
        if not np.isfinite(position):
            raise ValueError("gripper position must be finite")
        # Raw counts decrease while this gripper closes. Opening motion must
        # never satisfy the prerequisite for a grasp-contact decision.
        if self.initial_position - position > self.movement_counts:
            self.moved = True
        if self.stable_reference is None:
            self.stable_reference = position
            self.stable_sample_count = 0
        elif abs(position - self.stable_reference) < self.stable_band_counts:
            self.stable_sample_count += 1
        else:
            self.stable_reference = position
            self.stable_sample_count = 0
        return self.moved and self.stable_sample_count >= self.stable_samples


class GripperHoldGuard:
    """Detect sustained opening after an acknowledged grasp hold."""

    def __init__(
        self,
        opening_limit_counts: float = GRIPPER_HOLD_OPENING_LIMIT_COUNTS,
        opening_samples: int = GRIPPER_HOLD_OPENING_SAMPLES,
    ) -> None:
        if opening_limit_counts <= 0.0:
            raise ValueError("gripper opening limit must be positive")
        if opening_samples < 1:
            raise ValueError("gripper opening sample count must be positive")
        self.opening_limit_counts = float(opening_limit_counts)
        self.opening_samples = int(opening_samples)
        self.reference_position = None
        self.opening_sample_count = 0

    @property
    def armed(self) -> bool:
        return self.reference_position is not None

    def arm(self, position: float) -> None:
        if not np.isfinite(position):
            raise ValueError("gripper hold position must be finite")
        self.reference_position = float(position)
        self.opening_sample_count = 0

    def disarm(self) -> None:
        self.reference_position = None
        self.opening_sample_count = 0

    def observe(self, position: float) -> bool:
        if not self.armed:
            return False
        position = float(position)
        if not np.isfinite(position):
            raise ValueError("gripper position must be finite")
        self.reference_position = min(self.reference_position, position)
        if position - self.reference_position > self.opening_limit_counts:
            self.opening_sample_count += 1
        else:
            self.opening_sample_count = 0
        return self.opening_sample_count >= self.opening_samples


def gripper_position_for_opening_angle(
    opening_angle: float,
    closing_margin: float = 0.0,
) -> int:
    """Convert a URDF finger angle to a raw Dynamixel position.

    A positive closing margin commands additional travel toward fully closed
    so small calibration errors do not leave the fingers short of contact.
    """

    opening_angle = float(opening_angle)
    closing_margin = float(closing_margin)
    if not np.isfinite(opening_angle) or not (
        GRIPPER_MIN_OPENING_ANGLE
        <= opening_angle
        <= GRIPPER_MAX_OPENING_ANGLE
    ):
        raise ValueError(
            "gripper opening angle must be finite and within "
            f"[{GRIPPER_MIN_OPENING_ANGLE}, {GRIPPER_MAX_OPENING_ANGLE}] rad"
        )
    if not np.isfinite(closing_margin) or not (
        0.0 <= closing_margin <= GRIPPER_MAX_OPENING_ANGLE
    ):
        raise ValueError(
            "gripper closing margin must be finite and within "
            f"[0, {GRIPPER_MAX_OPENING_ANGLE}] rad"
        )
    commanded_angle = max(0.0, opening_angle - closing_margin)
    fraction = commanded_angle / GRIPPER_MAX_OPENING_ANGLE
    return round(
        GRIPPER_CLOSED_POSITION
        + fraction * (GRIPPER_OPEN_POSITION - GRIPPER_CLOSED_POSITION)
    )


def gripper_position_ramp(
    start_position: float,
    goal_position: int,
    max_step: float,
) -> tuple[int, ...]:
    """Return a minimum-jerk raw-position trajectory to the target.

    ``max_step`` bounds the peak step size. Repeated integer positions are
    retained so publishing at a fixed rate preserves the eased timing.
    """

    start_position = float(start_position)
    goal_position = int(goal_position)
    max_step = float(max_step)
    if not np.isfinite(start_position):
        raise ValueError("gripper start position must be finite")
    if not GRIPPER_CLOSED_POSITION <= goal_position <= GRIPPER_OPEN_POSITION:
        raise ValueError("raw gripper goal is outside the commissioned range")
    if not np.isfinite(max_step) or max_step <= 0.0:
        raise ValueError("gripper ramp step must be positive and finite")

    distance = abs(goal_position - start_position)
    # Quintic smoothstep has a peak normalized slope of 1.875. Scale the
    # duration so the eased trajectory does not exceed the requested step.
    step_count = max(1, math.ceil(1.875 * distance / max_step))
    progress = np.linspace(0.0, 1.0, step_count + 1)[1:]
    blend = 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5
    positions = np.rint(
        start_position + (goal_position - start_position) * blend
    ).astype(int)
    positions[-1] = goal_position
    return tuple(int(position) for position in positions)
