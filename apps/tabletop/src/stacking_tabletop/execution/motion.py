"""Bounded UR5e reference sampling retained from the legacy tabletop bridge."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

UR_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
WRIST3_JOINT_INDEX = UR_JOINT_NAMES.index("wrist_3_joint")
WRIST3_SOFT_LIMIT = np.deg2rad(300.0)


@dataclass(frozen=True)
class CartesianStreamLimits:
    """Bounded reference limits for the admittance-controller bridge."""

    rate_hz: float = 100.0
    max_joint_speed: float = 0.40
    max_tcp_speed: float = 0.08
    max_angular_speed: float = 0.70

    def validate(self) -> None:
        values = (
            self.rate_hz,
            self.max_joint_speed,
            self.max_tcp_speed,
            self.max_angular_speed,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("all Cartesian stream limits must be positive and finite")


def canonical_joint_positions(
    names: Sequence[str], positions: Sequence[float]
) -> np.ndarray | None:
    """Return UR joints in canonical order, accepting optional prefixes."""

    if len(names) != len(positions):
        return None
    by_name = {}
    for name, position in zip(names, positions):
        for canonical in UR_JOINT_NAMES:
            if name == canonical or name.endswith("/" + canonical):
                if canonical in by_name:
                    return None
                by_name[canonical] = float(position)
                break
    if len(by_name) != len(UR_JOINT_NAMES):
        return None
    result = np.asarray([by_name[name] for name in UR_JOINT_NAMES], dtype=float)
    return result if np.all(np.isfinite(result)) else None


def pose_error(actual: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return translation metres and shortest rotation radians."""

    actual = np.asarray(actual, dtype=float).reshape(4, 4)
    target = np.asarray(target, dtype=float).reshape(4, 4)
    position = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    rotation = float(
        Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).magnitude()
    )
    return position, rotation


def sample_joint_path(
    q_path: Sequence[Sequence[float]],
    kinematics,
    limits: CartesianStreamLimits = CartesianStreamLimits(),
    *,
    check=None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Time-parameterize a joint path and return fixed-rate FK references.

    Every interval duration is chosen from the largest joint, TCP translation,
    or TCP rotation requirement. Linear interpolation remains on the planned
    joint-space path; FK converts each sample to the controller reference.
    """

    limits.validate()
    q_values = [np.asarray(q, dtype=float).reshape(6) for q in q_path]
    if not q_values:
        raise ValueError("joint path is empty")
    if not all(np.all(np.isfinite(q)) for q in q_values):
        raise ValueError("joint path contains non-finite values")

    validate_joint_path(q_values)
    def fk(q):
        if check is not None:
            check()
        return validate_pose(kinematics.pose(q))

    poses = [fk(q) for q in q_values]
    samples = [(q_values[0].copy(), poses[0].copy())]
    for q_start, q_end, pose_start, pose_end in zip(
        q_values[:-1], q_values[1:], poses[:-1], poses[1:]
    ):
        joint_duration = float(np.max(np.abs(q_end - q_start))) / limits.max_joint_speed
        tcp_duration = (
            float(np.linalg.norm(pose_end[:3, 3] - pose_start[:3, 3]))
            / limits.max_tcp_speed
        )
        angular_distance = float(
            Rotation.from_matrix(pose_start[:3, :3].T @ pose_end[:3, :3]).magnitude()
        )
        angular_duration = angular_distance / limits.max_angular_speed
        steps = max(
            1,
            int(
                np.ceil(
                    max(joint_duration, tcp_duration, angular_duration) * limits.rate_hz
                )
            ),
        )
        while True:
            interval = []
            for step in range(1, steps + 1):
                fraction = step / steps
                q = q_start + fraction * (q_end - q_start)
                pose = fk(q)
                interval.append((q, pose))

            worst_ratio = 1.0
            previous_q, previous_pose = samples[-1]
            for q, pose in interval:
                position_step, rotation_step = pose_error(previous_pose, pose)
                worst_ratio = max(
                    worst_ratio,
                    float(np.max(np.abs(q - previous_q)))
                    * limits.rate_hz
                    / limits.max_joint_speed,
                    position_step * limits.rate_hz / limits.max_tcp_speed,
                    rotation_step * limits.rate_hz / limits.max_angular_speed,
                )
                previous_q, previous_pose = q, pose
            if worst_ratio <= 1.0 + 1e-9:
                samples.extend(interval)
                break
            steps = int(np.ceil(steps * worst_ratio * 1.001))
    return samples


def validate_pose(value):
    """Reject malformed transforms rather than silently repairing commands."""
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("pose must be a finite 4x4 transform")
    rotation = pose[:3, :3]
    if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0)):
        raise ValueError("pose must contain a proper rigid transform")
    return pose.copy()


def validate_joint_path(values):
    path = np.asarray(values, dtype=float)
    if path.ndim != 2 or path.shape[1] != 6 or not len(path) or not np.all(np.isfinite(path)):
        raise ValueError("joint path must contain finite six-joint samples")
    if np.any(np.abs(path[:, WRIST3_JOINT_INDEX]) > WRIST3_SOFT_LIMIT + 1e-9):
        raise ValueError("wrist_3 exceeds the ±300 degree cable-routing limit")
    if np.any(np.abs(np.diff(path[:, WRIST3_JOINT_INDEX])) > np.pi + 1e-9):
        raise ValueError("wrist_3 path contains a raw branch jump")
    return path.copy()
