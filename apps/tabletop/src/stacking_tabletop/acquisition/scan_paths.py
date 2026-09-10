"""Tabletop scan geometry, retaining legacy pivot validation and view coverage."""

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

from stacking_tabletop.execution.motion import validate_joint_path

Q_SCAN = np.deg2rad([82., -73., 95., -182., -86., 0.])

@dataclass(frozen=True)
class ScanMotion:
    target: np.ndarray
    capture: bool
    label: str
    pivot_side: int | None = None
    pivot_x_angle_deg: float | None = None
    wrist3_turn_deg: float | None = None

PIVOT_POINT_TOOL_M = np.array([0.0, 0.0, 0.20], dtype=np.float64)
PIVOT_X_MIN_DEG = -50.0
PIVOT_X_MAX_DEG = 50.0
PIVOT_X_STEP_DEG = 10.0
PIVOT_WRIST3_TURN_DEG = 180.0
PIVOT_WRIST3_TRANSITION_STEP_DEG = 10.0
# A 10-degree Cartesian pivot near the negative endpoint requires about 20.33
# degrees at wrist_1 with the calibrated real robot. Keep this guard tight
# enough to reject an actual IK branch jump while allowing that valid motion.
PIVOT_MAX_JOINT_STEP_DEG = 30.0
PIVOT_MAX_WAYPOINT_ERROR_M = 0.0001
# References interpolate in joint space before Cartesian streaming, so the held
# point does not follow the exact Cartesian pivot arc between waypoints. The
# real calibration produces a 3.05 mm transient at the endpoint transition;
# captures still occur at the validated invariant-point waypoints.
PIVOT_MAX_INTERPOLATED_DRIFT_M = 0.004

def _inclusive_degrees(start: float, stop: float, step: float) -> np.ndarray:
    if not all(np.isfinite([start, stop, step])) or step <= 0.0:
        raise ValueError("angle range and step must be finite with a positive step")
    distance = stop - start
    if distance < 0.0:
        raise ValueError("angle range must be ascending")
    values = (
        start
        + np.arange(
            int(np.floor(distance / step)) + 1,
            dtype=np.float64,
        )
        * step
    )
    if stop - values[-1] > 1e-12:
        values = np.append(values, stop)
    return values


def _transition_degrees(
    start: float,
    stop: float,
    maximum_step: float,
    include_stop: bool,
) -> np.ndarray:
    """Return bounded intermediate angles between arbitrary endpoints."""
    if not all(np.isfinite([start, stop, maximum_step])) or maximum_step <= 0.0:
        raise ValueError("transition angles require a positive finite step")
    distance = float(stop - start)
    if abs(distance) < 1e-12:
        return np.empty(0, dtype=np.float64)
    segment_count = int(np.ceil(abs(distance) / maximum_step))
    values = np.linspace(start, stop, segment_count + 1, dtype=np.float64)[1:]
    return values if include_stop else values[:-1]


def _pivot_pose(
    neutral_pose: np.ndarray,
    pivot_base: np.ndarray,
    tool_offset: np.ndarray,
    x_angle_deg: float,
) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = (
        neutral_pose[:3, :3]
        @ Rotation.from_euler("x", np.deg2rad(x_angle_deg)).as_matrix()
    )
    pose[:3, 3] = pivot_base - pose[:3, :3] @ tool_offset
    return pose


def pivot_scan_motions(
    q_scan,
    kinematics,
) -> tuple[ScanMotion, ...]:
    """Generate two invariant-point sweeps within one captured wrist-3 turn."""
    q_scan = np.asarray(q_scan, dtype=np.float64)
    if q_scan.shape != (6,):
        raise ValueError("q_scan must contain six joints")
    tool_offset = np.asarray(PIVOT_POINT_TOOL_M, dtype=np.float64)
    if tool_offset.shape != (3,) or not np.all(np.isfinite(tool_offset)):
        raise ValueError("PIVOT_POINT_TOOL_M must contain three finite values")

    view_angles = _inclusive_degrees(
        PIVOT_X_MIN_DEG,
        PIVOT_X_MAX_DEG,
        PIVOT_X_STEP_DEG,
    )
    # The first world-frame sweep runs from the configured maximum to the
    # minimum.  After the 180-degree wrist-3 turn, append_pivot() negates the
    # local X angle; reusing this descending list therefore makes the second
    # world-frame sweep run from the minimum back to the maximum.
    ordered_view_angles = view_angles[::-1]
    approach_angles = _transition_degrees(
        0.0,
        ordered_view_angles[0],
        PIVOT_X_STEP_DEG,
        include_stop=False,
    )
    neutral_return_angles = _transition_degrees(
        ordered_view_angles[-1],
        0.0,
        PIVOT_X_STEP_DEG,
        include_stop=True,
    )
    wrist_turns = _inclusive_degrees(
        PIVOT_WRIST3_TRANSITION_STEP_DEG,
        PIVOT_WRIST3_TURN_DEG,
        PIVOT_WRIST3_TRANSITION_STEP_DEG,
    )
    neutral_pose = kinematics.pose(q_scan)
    pivot_base = neutral_pose[:3, :3] @ tool_offset + neutral_pose[:3, 3]
    motions = []
    current = q_scan.copy()

    def append_pivot(side, neutral, angle, capture, label):
        nonlocal current
        # A 180-degree wrist-3 turn reverses the tool-X axis in the world
        # frame. Negate the second sweep's local angle so both sweeps tilt in
        # the same world-frame tilt convention.
        signed_angle = angle if side == 0 else -angle
        target_pose = _pivot_pose(neutral, pivot_base, tool_offset, signed_angle)
        current = kinematics.solve(target_pose, current)
        motions.append(
            ScanMotion(
                target=current.copy(),
                capture=capture,
                label=label,
                pivot_side=side,
                pivot_x_angle_deg=float(signed_angle),
                wrist3_turn_deg=float(side * PIVOT_WRIST3_TURN_DEG),
            )
        )

    for angle in approach_angles:
        append_pivot(0, neutral_pose, angle, True, "pivot_side_0_approach")
    for angle in ordered_view_angles:
        append_pivot(0, neutral_pose, angle, True, "pivot_side_0_view")
    for angle in neutral_return_angles:
        append_pivot(0, neutral_pose, angle, True, "pivot_side_0_return")

    wrist_start = current.copy()
    for turn in wrist_turns:
        final_turn = abs(turn - PIVOT_WRIST3_TURN_DEG) < 1e-12
        current = wrist_start.copy()
        current[5] += np.deg2rad(turn)
        motions.append(
            ScanMotion(
                target=current.copy(),
                capture=True,
                label="pivot_wrist3_turn",
                pivot_side=1 if final_turn else None,
                pivot_x_angle_deg=0.0 if final_turn else None,
                wrist3_turn_deg=float(turn),
            )
        )
    half_turn_neutral = q_scan.copy()
    half_turn_neutral[5] += np.deg2rad(PIVOT_WRIST3_TURN_DEG)
    half_turn_pose = kinematics.pose(half_turn_neutral)

    # Begin the second sign-compensated sweep from wrist-flipped Q_SCAN.
    for angle in approach_angles:
        append_pivot(1, half_turn_pose, angle, True, "pivot_side_1_approach")
    for angle in ordered_view_angles:
        append_pivot(1, half_turn_pose, angle, True, "pivot_side_1_view")
    for angle in neutral_return_angles:
        append_pivot(1, half_turn_pose, angle, True, "pivot_side_1_return")

    # The second pivot return ends at wrist-flipped Q_SCAN (X=0). Continue the
    # captured wrist-3 rotation forward from 180 to 360 degrees. The scan
    # runner then returns to Q_SCAN without capturing. Turns here are relative
    # to the selected starting branch (the migrated runner starts at -180).
    wrist_continuations = _transition_degrees(
        PIVOT_WRIST3_TURN_DEG,
        2.0 * PIVOT_WRIST3_TURN_DEG,
        PIVOT_WRIST3_TRANSITION_STEP_DEG,
        include_stop=True,
    )
    for turn in wrist_continuations:
        final_turn = abs(turn - 2.0 * PIVOT_WRIST3_TURN_DEG) < 1e-12
        current = q_scan.copy()
        current[5] += np.deg2rad(turn)
        motions.append(
            ScanMotion(
                target=current.copy(),
                capture=True,
                label="pivot_wrist3_continue",
                pivot_side=0 if final_turn else None,
                pivot_x_angle_deg=0.0 if final_turn else None,
                wrist3_turn_deg=float(turn),
            )
        )

    targets = np.asarray([motion.target for motion in motions])
    if np.any(targets < kinematics.lower) or np.any(targets > kinematics.upper):
        raise RuntimeError("pivot scan generated a target outside UR5e joint limits")
    joint_steps_deg = np.abs(np.rad2deg(np.diff(np.vstack([q_scan, targets]), axis=0)))
    for index, (motion, steps) in enumerate(zip(motions, joint_steps_deg)):
        if np.max(steps) > PIVOT_MAX_JOINT_STEP_DEG + 1e-6:
            joint = int(np.argmax(steps))
            raise RuntimeError(
                "pivot scan IK produced an unexpected joint step: "
                f"motion {index}, joint {joint}, "
                f"step={steps[joint]:.2f} deg"
            )
    point_tool = np.append(tool_offset, 1.0)

    def pivot_error(target):
        point = (kinematics.pose(target) @ point_tool)[:3]
        return float(np.linalg.norm(point - pivot_base))

    pivot_errors = [pivot_error(target) for target in targets]
    if max(pivot_errors) > PIVOT_MAX_WAYPOINT_ERROR_M:
        raise RuntimeError(
            "pivot scan failed invariant-point validation: "
            f"maximum error={max(pivot_errors) * 1000.0:.3f} mm"
        )
    interpolated_errors = []
    previous = q_scan
    for target in targets:
        interpolated_errors.extend(
            pivot_error(previous + fraction * (target - previous))
            for fraction in (0.25, 0.5, 0.75)
        )
        previous = target
    if max(interpolated_errors) > PIVOT_MAX_INTERPOLATED_DRIFT_M:
        raise RuntimeError(
            "pivot scan joint interpolation moves the fixed point too far: "
            f"maximum sampled drift={max(interpolated_errors) * 1000.0:.3f} mm"
        )
    return tuple(motions)




def build_scan(kind, kinematics, anchor, *, check=lambda: None):
    """Precompute every reference, including entry and return, before moving."""
    if kind not in ('wrist', 'pivot'):
        raise ValueError('scan kind must be wrist or pivot')
    check()
    anchor = validate_joint_path([anchor])[0]
    motions = [ScanMotion(Q_SCAN.copy(), False, 'enter_scan_pose')]
    if kind == 'wrist':
        for angle in range(-180, 180, 10):
            target = Q_SCAN.copy()
            target[5] = np.deg2rad(angle)
            motions.append(ScanMotion(target, True, 'wrist_view', wrist3_turn_deg=float(angle)))
    else:
        # Same complete orientation coverage without commanding +360 degrees.
        start = Q_SCAN.copy()
        start[5] = -np.pi
        motions.append(ScanMotion(start, False, 'enter_pivot_branch'))
        motions.extend(pivot_scan_motions(start, kinematics))
    motions.extend((ScanMotion(Q_SCAN.copy(), False, 'return_scan_pose'),
                    ScanMotion(anchor.copy(), False, 'return_execution_gate')))
    previous = anchor
    for motion in motions:
        check()
        validate_joint_path([motion.target])
        if np.any(motion.target < kinematics.lower) or np.any(motion.target > kinematics.upper):
            raise ValueError('scan target exceeds robot joint limits')
        # Explicit interpolation covers long entry/return moves without changing
        # their physical joint branch or aliasing a whole wrist revolution.
        validate_joint_path(joint_leg(previous, motion.target))
        previous = motion.target
    return tuple(motions)


def joint_leg(start, target):
    count = max(2, int(np.ceil(np.max(np.abs(np.asarray(target) - start)) / np.deg2rad(10))) + 1)
    return np.linspace(start, target, count)
