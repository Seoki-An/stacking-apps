"""Target-overlap evaluation used by environment rewards."""

from collections import OrderedDict
from threading import Lock
from typing import Protocol

import numpy as np

from stacking_common.domain import StackObject, TargetGeometry
from stacking_common.domain.geometry import inverse_transform_points, transform_points


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

    _CACHE_SIZE = 1024

    def __init__(self, target: TargetGeometry, resolution: float) -> None:
        self.target = target
        self.resolution = float(resolution)
        if not np.isfinite(self.resolution) or self.resolution <= 0.0:
            raise ValueError("overlap resolution must be positive and finite")
        self._axes, self._lattice_indices, self._grid_points = self._voxel_centers()
        # Geometry and the target grid are fixed for this evaluator. Keep exact
        # object/pose intersections across CEM re-ranking and reward queries.
        self._intersections = OrderedDict()
        self._cache_lock = Lock()

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
        inside = self.target.contains(candidates)
        # Each lattice cell's row in the retained points, or -1 outside.
        indices = np.full(len(candidates), -1, dtype=np.int64)
        indices[inside] = np.arange(np.count_nonzero(inside))
        return axes, indices.reshape(grid[0].shape), candidates[inside]

    def _intersection_volume(
        self, stack_object: StackObject, pose: np.ndarray
    ) -> float:
        if len(self._grid_points) == 0:
            return 0.0
        pose = np.array(pose, dtype=float, copy=True)
        key = (id(stack_object), pose.shape, pose.tobytes())
        with self._cache_lock:
            cached = self._intersections.get(key)
            if cached is not None:
                self._intersections.move_to_end(key)
                return cached[1]

        volume = self._compute_intersection_volume(stack_object, pose)
        with self._cache_lock:
            # Retain the object itself so its identity cannot be reused while
            # cached. The lock also protects shared action-filter workers.
            self._intersections[key] = (stack_object, volume)
            self._intersections.move_to_end(key)
            if len(self._intersections) > self._CACHE_SIZE:
                self._intersections.popitem(last=False)
        return volume

    def _compute_intersection_volume(
        self, stack_object: StackObject, pose: np.ndarray
    ) -> float:
        points = self._candidate_grid_points(stack_object, pose)
        local_points = inverse_transform_points(points, pose)
        inside = stack_object.contains_local_points(
            local_points, tolerance=self.resolution * 1e-6
        )
        return float(np.count_nonzero(inside)) * self.resolution**3

    def _candidate_grid_points(
        self, stack_object: StackObject, pose: np.ndarray
    ) -> np.ndarray:
        """Conservatively crop the original grid, preserving its voxel centers."""
        lower = np.full(3, np.inf)
        upper = np.full(3, -np.inf)
        tolerance = self.resolution * 1e-6
        for geometry in stack_object.geometries:
            center = np.mean(geometry.points, axis=0)
            face_centers, normals, _ = geometry.face_data()
            margin = float(np.min(np.einsum(
                'ij,ij->i', normals, face_centers - center)))
            if margin <= 0.0:
                return self._grid_points
            # A plane tolerance can extend acute corners much farther than
            # the tolerance itself. Expanding about an interior point by
            # 1 + tolerance/min_face_distance encloses the relaxed polytope.
            # Double that allowance and pad by a voxel for floating roundoff.
            expanded = center + (geometry.points - center) * (
                1.0 + 2.0 * tolerance / margin)
            world = transform_points(expanded, pose)
            if not np.all(np.isfinite(world)):
                return self._grid_points
            lower = np.minimum(lower, np.min(world, axis=0))
            upper = np.maximum(upper, np.max(world, axis=0))
        lower -= self.resolution
        upper += self.resolution
        # The points within these bounds form one lattice sub-block. Its kept
        # rows, in lattice order, equal the full-grid mask in the same order.
        block = tuple(
            slice(np.searchsorted(axis, low, side="left"),
                  np.searchsorted(axis, high, side="right"))
            for axis, low, high in zip(self._axes, lower, upper)
        )
        rows = self._lattice_indices[block].ravel()
        return self._grid_points[rows[rows >= 0]]
