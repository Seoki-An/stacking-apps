"""Connected target regions for proposal and floor policies, not separate goals."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.spatial import ConvexHull

from stacking_common.domain import CompositeTargetGeometry
from stacking_common.domain.geometry import transform_points


def ground_height(inventory, explicit=None):
    value = ((0. if explicit is None else explicit)
             if isinstance(inventory.target, CompositeTargetGeometry) else inventory.target.origin[2])
    if not np.isfinite(value):
        raise ValueError("placement ground height must be finite")
    return float(value)


def polygon(points):
    points = np.asarray(points, dtype=float)
    return points[ConvexHull(points).vertices]


def polygons_overlap(first, second):
    """Separating-axis intersection, including touching convex polygons."""
    for vertices in (first, second):
        edges = np.roll(vertices, -1, axis=0) - vertices
        normals = np.column_stack((-edges[:, 1], edges[:, 0]))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        a, b = first @ normals.T, second @ normals.T
        if np.any((a.max(0) < b.min(0) - 1e-9) | (b.max(0) < a.min(0) - 1e-9)):
            return False
    return True


def horizontal_polygon(geometry, z):
    """Intersect a convex mesh with a horizontal plane without convexifying a union."""
    points = geometry.points
    near = np.abs(points[:, 2] - z) <= 1e-9
    values = [points[near, :2]]
    edges = np.unique(np.sort(np.concatenate(
        [geometry.faces[:, [0, 1]], geometry.faces[:, [1, 2]], geometry.faces[:, [2, 0]]]), axis=1), axis=0)
    a, b = points[edges[:, 0]], points[edges[:, 1]]
    crosses = ((a[:, 2] < z) & (b[:, 2] > z)) | ((a[:, 2] > z) & (b[:, 2] < z))
    a, b = a[crosses], b[crosses]
    if len(a):
        t = (z - a[:, 2]) / (b[:, 2] - a[:, 2])
        values.append((a + t[:, None] * (b - a))[:, :2])
    xy = np.unique(np.concatenate(values), axis=0)
    if len(xy) < 3 or np.linalg.matrix_rank(xy - xy.mean(0), tol=1e-9) < 2:
        raise ValueError("composite sampling requires every component to meet the common ground plane over an area")
    return polygon(xy)


@dataclass(frozen=True)
class SamplingRegion:
    target: CompositeTargetGeometry

    @property
    def region_id(self):
        return self.target.component_ids

    def mask(self, xx, yy, z):
        return self.target.horizontal_mask(np.c_[xx.ravel(), yy.ravel()], z).reshape(xx.shape)

    def grid(self, shape):
        lower, upper = self.target.bounds[:, :2]
        rows, columns = shape
        x = lower[0] + (np.arange(columns) + .5) / columns * (upper[0] - lower[0])
        y = lower[1] + (np.arange(rows) + .5) / rows * (upper[1] - lower[1])
        xx, yy = np.meshgrid(x, y)
        return lower, upper, xx, yy


class TargetRegions:
    """Group touching ground sections geometrically, independently of raster resolution.

    Overlapping DSF pieces share one floor. Disjoint regions can be started
    independently. All components must meet the same physical ground plane;
    elevated target components need a separate support policy.
    """

    def __init__(self, target, ground_z=0.):
        self.target = target
        self.ground_z = float(ground_z)
        if not np.isfinite(self.ground_z):
            raise ValueError("placement ground height must be finite")
        components = sorted(target.components, key=lambda item: item.component_id)
        sections = [horizontal_polygon(item.world_geometry, self.ground_z) for item in components]
        groups = []
        for index in range(len(components)):
            joined = [group for group in groups
                      if any(polygons_overlap(sections[index], sections[other]) for other in group)]
            for group in joined:
                groups.remove(group)
            groups.append([index, *(other for group in joined for other in group)])
        self.regions = tuple(SamplingRegion(CompositeTargetGeometry(tuple(
            components[index] for index in sorted(group)))) for group in groups)
        self.regions = tuple(sorted(self.regions, key=lambda item: item.region_id))
        self._footprints = {
            item.component_id: polygon(item.world_geometry.points[:, :2]) for item in components}

    def pose_metadata(self, inventory, object_index, pose):
        points, _ = inventory[object_index].mesh_arrays()
        footprint = polygon(transform_points(points, pose)[:, :2])
        component_ids = tuple(sorted(key for key, value in self._footprints.items()
                                     if polygons_overlap(footprint, value)))
        region_ids = tuple(region.region_id for region in self.regions
                           if set(region.region_id).intersection(component_ids))
        return {"target_component_ids": component_ids, "target_region_ids": region_ids,
                "target_region_id": region_ids[0] if region_ids else None}


class CachedAnchorContainment:
    """Sampler-local half-spaces and bounded orientation-dependent offsets.

    Target geometry is immutable. Heights/XY positions remain query inputs,
    while vertex content, inset and margin key the reusable support offsets.
    The LRU bound prevents arbitrary CEM rotations growing the cache forever.
    """

    def __init__(self, region):
        planes = []
        for component in region.target.components:
            centers, normals, _ = component.world_geometry.face_data()
            bounds = np.einsum("ij,ij->i", centers, normals)
            xy_norm = np.linalg.norm(normals[:, :2], axis=1)
            for array in (normals, bounds, xy_norm):
                array.setflags(write=False)
            planes.append((normals, bounds, xy_norm))
        self.planes = tuple(planes)

        @lru_cache(maxsize=1024)
        def offsets(vertex_bytes, inset_scale, margin):
            points = np.frombuffer(vertex_bytes, dtype=float).reshape(-1, 3).copy()
            points[:, :2] *= inset_scale
            result = []
            for normals, bounds, xy_norm in self.planes:
                support = (points @ normals.T).max(0)
                values = bounds - support - margin * xy_norm
                values.setflags(write=False)
                result.append(values)
            return tuple(result)

        self.offsets = offsets

    def __call__(self, xy, rotated_points, placement_z, *, inset_scale=1., margin=0.):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        positions = np.c_[xy, np.broadcast_to(placement_z, (len(xy),))]
        offsets = self.offsets(np.asarray(rotated_points, dtype=float).tobytes(),
                               inset_scale, margin)
        return np.column_stack([
            np.all(positions @ normals.T <= values + 1e-9, axis=1)
            for (normals, _, _), values in zip(self.planes, offsets)])


def anchor_components(region, xy, rotated_points, placement_z, *, inset_scale=1., margin=0.):
    """Orientation-aware convex containment for each component and anchor.

    Match native posegen's conservative whole-stone-in-one-component semantics.
    XY inset scaling is a sampling heuristic; vertical extent is never scaled.
    The scaled vertices must satisfy every component half-space at the sampled
    placement height. No bounding rectangle is used as a feasibility test.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    positions = np.c_[xy, np.broadcast_to(placement_z, (len(xy),))]
    points = np.array(rotated_points, dtype=float, copy=True)
    points[:, :2] *= inset_scale
    membership = []
    for component in region.target.components:
        centers, normals, _ = component.world_geometry.face_data()
        bounds = np.einsum("ij,ij->i", centers, normals)
        support = (points @ normals.T).max(0)
        # A physical XY margin, not a Z offset or a scale-dependent plane value.
        offsets = bounds - support - margin * np.linalg.norm(normals[:, :2], axis=1)
        membership.append(np.all(positions @ normals.T <= offsets + 1e-9, axis=1))
    return np.column_stack(membership)
