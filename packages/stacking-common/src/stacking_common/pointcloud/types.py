"""Immutable array representation of a point cloud."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PointCloudData:
    """Finite XYZ points independent of a sensor or rendering library."""

    points: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("point-cloud points must have shape (N, 3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("point-cloud points must be finite")
        points = points.copy()
        points.setflags(write=False)
        object.__setattr__(self, "points", points)

    def __len__(self) -> int:
        return len(self.points)

    def transformed(self, transform: np.ndarray) -> "PointCloudData":
        """Return points transformed by a rigid homogeneous matrix."""

        value = validate_rigid_transform(transform)
        return PointCloudData(
            self.points @ value[:3, :3].T + value[:3, 3]
        )


def validate_rigid_transform(
    transform: np.ndarray,
    name: str = "transform",
) -> np.ndarray:
    """Return a validated copy of a rigid 4-by-4 transform."""

    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation must be proper")
    return value.copy()
