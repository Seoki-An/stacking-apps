"""Bounded, branch-local caching of a coarse top surface for rewards."""

from collections import OrderedDict

import numpy as np

from stacking_common.domain.geometry import transform_points
from stacking_common.visualization.heightmap import _rasterize_triangles


class SurfaceFlatness:
    """Measure max-minus-mean height over the target's floor footprint.

    Empty cells have zero height. The fixed 32 x 32 cell-center grid measures
    the upper envelope, not internal cavities or individual stone centers.
    Each reward worker owns its caches; no mutable state is shared by branches.
    """

    def __init__(self, inventory):
        self.inventory = inventory
        target = inventory.target
        self.floor_z = float(target.origin[2])
        cells = (np.arange(32) + .5) / 32 - .5
        self.x = target.origin[0] + target.width * cells
        self.y = target.origin[1] + target.length * cells
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
        surface = np.zeros(len(self.x) * len(self.y))
        _rasterize_triangles(surface, self.x, self.y, world[faces].reshape(-1, 3, 3))
        return self._remember(self._objects, key, surface, 128)

    def measure(self, state):
        keys = []
        for index in state.placed_object_indices:
            oid = self.inventory.objects[index].object_id
            pose = state.object_poses.get(oid)
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
            _, total, maximum = cached
            return maximum, total / (len(self.x) * len(self.y))

        parent = self._scenes.get(key[:-1]) if key else None
        if parent is not None:
            # Exact pose keys ensure this path is used only when all previous
            # stones are unchanged. A sibling never mutates its parent's grid.
            previous, total, maximum = parent
            added = self._object_surface(key[-1])
            increase = np.maximum(added - previous, 0.)
            surface = previous + increase
            total += float(increase.sum())
            maximum = max(maximum, float(added.max()))
        else:
            # Motion can expose previously hidden surfaces. Recompose the
            # envelope from current poses, reusing unchanged object projections.
            surface = np.zeros(len(self.x) * len(self.y))
            for object_key in key:
                np.maximum(surface, self._object_surface(object_key), out=surface)
            total, maximum = float(surface.sum()), float(surface.max())
        self._remember(self._scenes, key, (surface, total, maximum), 64)
        return maximum, total / surface.size
