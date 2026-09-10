"""Explicit rigid transforms between application coordinate frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stacking_common.domain._validation import pose_array


@dataclass(frozen=True)
class FrameTransform:
    """Pose of a child frame in its parent frame.

    ``parent_T_child`` uses the common scalar-last pose layout
    ``[x, y, z, qx, qy, qz, qw]``.
    """

    parent_frame: str
    child_frame: str
    parent_T_child: np.ndarray

    def __post_init__(self) -> None:
        parent = str(self.parent_frame).strip()
        child = str(self.child_frame).strip()
        if not parent or not child:
            raise ValueError("frame names must not be empty")
        if parent == child:
            raise ValueError("parent and child frames must be different")
        pose = pose_array(self.parent_T_child, "frame transform")
        pose[3:] /= np.linalg.norm(pose[3:])
        object.__setattr__(self, "parent_frame", parent)
        object.__setattr__(self, "child_frame", child)
        object.__setattr__(self, "parent_T_child", pose)

    def child_pose_to_parent(self, child_T_object: np.ndarray) -> np.ndarray:
        """Convert an object pose from the child frame to the parent frame."""

        return compose_poses(self.parent_T_child, child_T_object)

    def parent_pose_to_child(self, parent_T_object: np.ndarray) -> np.ndarray:
        """Convert an object pose from the parent frame to the child frame."""

        return compose_poses(inverse_pose(self.parent_T_child), parent_T_object)

    def inverse(self) -> "FrameTransform":
        """Return the same relation with parent and child exchanged."""

        return FrameTransform(
            self.child_frame,
            self.parent_frame,
            inverse_pose(self.parent_T_child),
        )


def compose_poses(
    first_T_second: np.ndarray,
    second_T_object: np.ndarray,
) -> np.ndarray:
    """Compose two scalar-last rigid poses."""

    first = pose_array(first_T_second, "first pose")
    second = pose_array(second_T_object, "second pose")
    first_q = _normalized_quaternion(first[3:])
    second_q = _normalized_quaternion(second[3:])
    result = np.empty(7, dtype=float)
    result[:3] = first[:3] + _rotation_matrix(first_q) @ second[:3]
    result[3:] = _normalized_quaternion(_quaternion_product(first_q, second_q))
    return result


def inverse_pose(pose: np.ndarray) -> np.ndarray:
    """Invert one scalar-last rigid pose."""

    value = pose_array(pose)
    quaternion = _normalized_quaternion(value[3:])
    inverse_quaternion = np.array(
        [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]],
        dtype=float,
    )
    result = np.empty(7, dtype=float)
    result[:3] = -_rotation_matrix(inverse_quaternion) @ value[:3]
    result[3:] = inverse_quaternion
    return result


def _normalized_quaternion(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("pose quaternion must have non-zero norm")
    return quaternion / norm


def _quaternion_product(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = first
    x2, y2, z2, w2 = second
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=float,
    )


def _rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )
