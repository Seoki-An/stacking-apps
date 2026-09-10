"""Target-overlap evaluation used by environment rewards."""

from typing import Protocol

import numpy as np

from stacking_common.domain import StackObject, TargetGeometry
from stacking_common.domain.geometry import inverse_transform_points


class OverlapEvaluator(Protocol):
    """Evaluate object containment and target coverage."""

    def object_containment(
        self, stack_object: StackObject, pose: np.ndarray
    ) -> float:
        """Return the fraction of an object inside the target."""

    def target_coverage(
        self, stack_object: StackObject, pose: np.ndarray
    ) -> float:
        """Return the fraction of the target occupied by an object."""


class VoxelOverlapEvaluator:
    """Approximate overlap on an explicit platform-tuned voxel grid."""

    def __init__(self, target: TargetGeometry, resolution: float) -> None:
        self.target = target
        self.resolution = float(resolution)
        if not np.isfinite(self.resolution) or self.resolution <= 0.0:
            raise ValueError("overlap resolution must be positive and finite")
        self._grid_points = self._voxel_centers()

    def object_containment(
        self, stack_object: StackObject, pose: np.ndarray
    ) -> float:
        intersection = self._intersection_volume(stack_object, pose)
        return float(np.clip(intersection / stack_object.volume, 0.0, 1.0))

    def target_coverage(
        self, stack_object: StackObject, pose: np.ndarray
    ) -> float:
        intersection = self._intersection_volume(stack_object, pose)
        return float(np.clip(intersection / self.target.volume, 0.0, 1.0))

    def _voxel_centers(self) -> np.ndarray:
        points = self.target.points
        minimum = np.min(points, axis=0)
        maximum = np.max(points, axis=0)
        if np.any(maximum - minimum < self.resolution):
            raise ValueError(
                "overlap resolution must not exceed a target dimension"
            )
        axes = [
            np.arange(low + self.resolution / 2.0, high, self.resolution)
            for low, high in zip(minimum, maximum)
        ]
        grid = np.meshgrid(*axes, indexing="ij")
        candidates = np.stack(grid, axis=-1).reshape(-1, 3)
        return candidates[self.target.contains(candidates)]

    def _intersection_volume(
        self, stack_object: StackObject, pose: np.ndarray
    ) -> float:
        if len(self._grid_points) == 0:
            return 0.0
        local_points = inverse_transform_points(self._grid_points, pose)
        inside = stack_object.contains_local_points(
            local_points, tolerance=self.resolution * 1e-6
        )
        return float(np.count_nonzero(inside)) * self.resolution**3
