"""Target-overlap evaluation used by environment rewards."""

from collections import OrderedDict
from threading import Lock
from typing import Protocol

import numpy as np

from stacking_common.domain import CompositeTargetGeometry, StackObject, TargetRegion
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

    def __init__(self, target: TargetRegion, resolution: float) -> None:
        self.target = target
        self.resolution = float(resolution)
        if not np.isfinite(self.resolution) or self.resolution <= 0.0:
            raise ValueError("overlap resolution must be positive and finite")
        self._composite = isinstance(target, CompositeTargetGeometry)
        if self._composite:
            self._grid_points = self._union_voxel_centers()
            self.target_volume = len(self._grid_points) * self.resolution**3
        else:
            self._axes, self._lattice_indices, self._grid_points = self._voxel_centers()
            self.target_volume = float(target.volume)
        # Geometry and the target grid are fixed for this evaluator. Keep exact
        # object/pose intersections across CEM re-ranking and reward queries.
        self._intersections = OrderedDict()
        self._occupancies = OrderedDict()
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
        return float(np.clip(intersection / self.target_volume, 0.0, 1.0))

    def scene_coverage(self, placements) -> float:
        """Union coverage of (object, pose) pairs; overlapping stones count once."""
        occupied = np.zeros(len(self._grid_points), dtype=bool)
        for obj, pose in placements:
            occupied[self._occupied_rows(obj, pose)] = True
        return float(np.clip(occupied.sum() * self.resolution**3 / self.target_volume, 0., 1.))

    def _union_voxel_centers(self):
        """One shared lattice, enumerated only inside component bounding boxes.

        Integer-cell deduplication makes overlap and decomposition irrelevant.
        Never allocate the potentially enormous empty box between components.
        """
        origin = self.target.bounds[0]
        retained = []
        blocks = []
        for component in self.target.components:
            lower, upper = component.bounds
            if np.any(upper - lower < self.resolution):
                raise ValueError("overlap resolution must not exceed a target component dimension")
            start = np.ceil((lower - origin) / self.resolution - .5 - 1e-9).astype(np.int64)
            stop = np.floor((upper - origin) / self.resolution - .5 + 1e-9).astype(np.int64) + 1
            blocks.append((start, stop))
            shape = stop - start
            size = int(shape[0]) * int(shape[1]) * int(shape[2])
            count = 0
            for offset in range(0, size, 65536):
                flat = np.arange(offset, min(offset + 65536, size))
                cells = np.column_stack(np.unravel_index(flat, tuple(shape))) + start
                points = origin + (cells + .5) * self.resolution
                cells = cells[component.contains(points)]
                retained.append(cells)
                count += len(cells)
            if not count:
                raise ValueError("overlap grid misses a target component; reduce overlap resolution")
        cells = np.unique(np.concatenate(retained), axis=0)
        self._union_blocks = self._index_union_cells(origin, cells, blocks)
        return origin + (cells + .5) * self.resolution

    def _index_union_cells(self, origin, cells, blocks):
        """Local 3D indices into the unchanged, deduplicated union lattice.

        Allocate only component bounding boxes, never the empty space between
        disconnected components. Overlapping blocks reference the same global
        row IDs; query results are deduplicated and retain global lattice order.
        """
        indexed = []
        for start, stop in blocks:
            first = np.searchsorted(cells[:, 0], start[0], side="left")
            last = np.searchsorted(cells[:, 0], stop[0], side="left")
            rows = np.arange(first, last)
            inside = np.all((cells[rows] >= start) & (cells[rows] < stop), axis=1)
            rows = rows[inside]
            local = cells[rows] - start
            indices = np.full(tuple(stop - start), -1, dtype=np.intp)
            indices[tuple(local.T)] = rows
            axes = tuple(origin[axis] + (np.arange(start[axis], stop[axis]) + .5)
                         * self.resolution for axis in range(3))
            indices.setflags(write=False)
            for axis in axes:
                axis.setflags(write=False)
            indexed.append((axes, indices))
        return tuple(indexed)

    def _union_grid_rows(self, lower, upper):
        chunks = []
        for axes, indices in self._union_blocks:
            block = tuple(slice(np.searchsorted(axis, low, side="left"),
                                np.searchsorted(axis, high, side="right"))
                          for axis, low, high in zip(axes, lower, upper))
            rows = indices[block].ravel()
            rows = rows[rows >= 0]
            if len(rows):
                chunks.append(rows)
        if not chunks:
            return np.empty(0, dtype=np.intp)
        if len(chunks) == 1:
            return chunks[0]
        return np.unique(np.concatenate(chunks))

    def _occupied_rows(self, stack_object, pose):
        pose = np.array(pose, dtype=float, copy=True)
        key = (id(stack_object), pose.shape, pose.tobytes())
        with self._cache_lock:
            cached = self._occupancies.get(key)
            if cached is not None:
                self._occupancies.move_to_end(key)
                return cached[1]
        rows = self._candidate_grid_rows(stack_object, pose)
        local = inverse_transform_points(self._grid_points[rows], pose)
        rows = rows[stack_object.contains_local_points(local, tolerance=self.resolution * 1e-6)]
        rows.setflags(write=False)
        with self._cache_lock:
            self._occupancies[key] = (stack_object, rows)
            self._occupancies.move_to_end(key)
            if len(self._occupancies) > self._CACHE_SIZE:
                self._occupancies.popitem(last=False)
        return rows

    def _voxel_centers(self):
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
        if self._composite:
            return len(self._occupied_rows(stack_object, pose)) * self.resolution**3
        points = self._candidate_grid_points(stack_object, pose)
        local_points = inverse_transform_points(points, pose)
        inside = stack_object.contains_local_points(
            local_points, tolerance=self.resolution * 1e-6
        )
        return float(np.count_nonzero(inside)) * self.resolution**3

    def _candidate_grid_points(
        self, stack_object: StackObject, pose: np.ndarray
    ) -> np.ndarray:
        return self._grid_points[self._candidate_grid_rows(stack_object, pose)]

    def _candidate_grid_rows(self, stack_object, pose):
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
                return np.arange(len(self._grid_points))
            # A plane tolerance can extend acute corners much farther than
            # the tolerance itself. Expanding about an interior point by
            # 1 + tolerance/min_face_distance encloses the relaxed polytope.
            # Double that allowance and pad by a voxel for floating roundoff.
            expanded = center + (geometry.points - center) * (
                1.0 + 2.0 * tolerance / margin)
            world = transform_points(expanded, pose)
            if not np.all(np.isfinite(world)):
                return np.arange(len(self._grid_points))
            lower = np.minimum(lower, np.min(world, axis=0))
            upper = np.maximum(upper, np.max(world, axis=0))
        lower -= self.resolution
        upper += self.resolution
        if self._composite:
            return self._union_grid_rows(lower, upper)
        # The points within these bounds form one lattice sub-block. Its kept
        # rows, in lattice order, equal the full-grid mask in the same order.
        block = tuple(
            slice(np.searchsorted(axis, low, side="left"),
                  np.searchsorted(axis, high, side="right"))
            for axis, low, high in zip(self._axes, lower, upper)
        )
        rows = self._lattice_indices[block].ravel()
        return rows[rows >= 0]
