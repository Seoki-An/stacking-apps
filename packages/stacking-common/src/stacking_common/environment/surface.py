"""Bounded, branch-local caching of a coarse top surface for rewards."""

from collections import OrderedDict

import numpy as np

from stacking_common.domain import CompositeTargetGeometry
from stacking_common.domain.geometry import transform_points
from stacking_common.visualization.heightmap import _rasterize_triangles


class SurfaceFlatness:
    """Measure area-weighted height statistics over the target's floor footprint.

    Empty cells have zero height. The fixed 32 x 32 cell-center grid measures
    the upper envelope, not internal cavities or individual stone centers.
    Each reward worker owns its caches; no mutable state is shared by branches.
    """

    def __init__(self, inventory, *, ground_z=0.):
        self.inventory = inventory
        target = inventory.target
        self._grids = None
        if isinstance(target, CompositeTargetGeometry):
            from stacking_common.actions.target_regions import TargetRegions
            regions = TargetRegions(target, ground_z)
            self.floor_z = regions.ground_z
            self._grids, weights = [], []
            for region in regions.regions:
                lower, upper, xx, yy = region.grid((32, 32))
                mask = region.mask(xx, yy, self.floor_z).ravel()
                if not mask.any():
                    raise ValueError("surface reward grid misses a composite target region")
                self._grids.append((xx[0], yy[:, 0], mask))
                weights.extend([float(np.prod(upper - lower)) / 1024] * int(mask.sum()))
            self._weights = np.asarray(weights)
            self._size = len(weights)
        else:
            self.floor_z = float(target.origin[2])
            cells = (np.arange(32) + .5) / 32 - .5
            self.x = target.origin[0] + target.width * cells
            self.y = target.origin[1] + target.length * cells
            self._size = len(self.x) * len(self.y)
            self._weights = np.ones(self._size)
        self._weight_sum = float(self._weights.sum())
        self._objects = OrderedDict()
        self._scenes = OrderedDict()

    @staticmethod
    def _remember(cache, key, value, limit):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)
        return value

    def _object_surface(self, key):
        cached = self._objects.get(key)
        if cached is not None:
            self._objects.move_to_end(key)
            return cached
        index, pose_bytes = key
        pose = np.frombuffer(pose_bytes, dtype=np.float64)
        points, faces = self.inventory.objects[index].mesh_arrays()
        world = transform_points(points, pose)
        world[:, 2] -= self.floor_z
        if self._grids is None:
            surface = np.zeros(self._size)
            _rasterize_triangles(surface, self.x, self.y, world[faces].reshape(-1, 3, 3))
        else:
            pieces = []
            for x, y, mask in self._grids:
                values = np.zeros(len(x) * len(y))
                _rasterize_triangles(values, x, y, world[faces].reshape(-1, 3, 3))
                pieces.append(values[mask])
            surface = np.concatenate(pieces)
        return self._remember(self._objects, key, surface, 128)

    def measure(self, state):
        return self.statistics(state)[:2]

    def measure_poses(self, placed_indices, object_poses):
        """Retain the max/mean diagnostic API for existing callers."""
        return self.statistics_poses(placed_indices, object_poses)[:2]

    def statistics(self, state):
        return self.statistics_poses(state.placed_object_indices, state.object_poses)

    def statistics_poses(self, placed_indices, object_poses):
        """Return maximum, area-weighted mean and variance (m, m, m²)."""
        keys = []
        for index in placed_indices:
            oid = self.inventory.objects[index].object_id
            pose = object_poses.get(oid)
            if pose is None:
                pose = self.inventory.initial_pose(oid)
            pose = np.asarray(pose, dtype=np.float64)
            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise ValueError("surface reward requires finite settled poses")
            keys.append((index, pose.tobytes()))
        key = tuple(keys)
        cached = self._scenes.get(key)
        if cached is not None:
            self._scenes.move_to_end(key)
            _, total, maximum, variance = cached
            return maximum, total / self._weight_sum, variance

        parent = self._scenes.get(key[:-1]) if key else None
        if parent is not None:
            # Exact pose keys ensure this path is used only when all previous
            # stones are unchanged. A sibling never mutates its parent's grid.
            previous, total, maximum, _ = parent
            added = self._object_surface(key[-1])
            increase = np.maximum(added - previous, 0.)
            surface = previous + increase
            total += float((increase * self._weights).sum())
            maximum = max(maximum, float(added.max()))
        else:
            # Motion can expose previously hidden surfaces. Recompose the
            # envelope from current poses, reusing unchanged object projections.
            surface = np.zeros(self._size)
            for object_key in key:
                np.maximum(surface, self._object_surface(object_key), out=surface)
            total, maximum = float((surface * self._weights).sum()), float(surface.max())
        mean = total / self._weight_sum
        # Center first instead of E[h²] - E[h]² to avoid cancellation on tall,
        # nearly flat structures. The existing grid and area weights are reused.
        variance = float(np.dot(self._weights, (surface - mean) ** 2) / self._weight_sum)
        self._remember(self._scenes, key, (surface, total, maximum, variance), 64)
        return maximum, mean, variance
