"""Optional two-sided wall geometry; no simulator or platform dependencies."""

from collections import OrderedDict
from dataclasses import dataclass
from collections.abc import Mapping
from threading import RLock

import numpy as np

from stacking_common.domain.geometry import quaternion_matrix


@dataclass(frozen=True)
class ConstructionConfig:
    mode: str = "volume_fill"
    surface_tolerance: float | None = None
    proposal_weight: float = 2.0
    reward_weight: float = 20.0

    def __post_init__(self):
        if self.mode not in {"volume_fill", "two_sided_wall"}:
            raise ValueError("construction.mode must be volume_fill or two_sided_wall")
        if self.surface_tolerance is not None and (
            not np.isfinite(self.surface_tolerance) or self.surface_tolerance <= 0
        ):
            raise ValueError("construction.surface_tolerance must be positive metres")
        if self.enabled and self.surface_tolerance is None:
            raise ValueError("two_sided_wall requires construction.surface_tolerance")
        for name in ("proposal_weight", "reward_weight"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"construction.{name} must be finite and nonnegative")

    @property
    def enabled(self):
        return self.mode == "two_sided_wall"

    @classmethod
    def from_mapping(cls, values):
        if not isinstance(values, Mapping):
            raise TypeError("construction must be a mapping")
        unknown = set(values) - {"mode", "surface_tolerance", "proposal_weight", "reward_weight"}
        if unknown:
            raise ValueError(f"unknown construction settings: {sorted(unknown)}")
        return cls(**values)


class WallSurfaces:
    """Rasterize the outer union of convex components onto the two YZ faces.

    Ray/half-space intersections avoid counting internal component faces. Caches
    are bounded and keyed by exact poses; they never encode a row cursor.
    """

    def __init__(self, inventory, config, shape=(32, 32)):
        if not config.enabled:
            raise ValueError("WallSurfaces requires two_sided_wall mode")
        self.inventory, self.config = inventory, config
        self.target = target = inventory.target
        nz, ny = shape
        if min(nz, ny) < 2:
            raise ValueError("wall surface grids require at least two cells per axis")
        self.y = target.origin[1] + ((np.arange(ny) + .5) / ny - .5) * target.length
        self.z = target.origin[2] + (np.arange(nz) + .5) / nz * target.height
        self.yy, self.zz = np.meshgrid(self.y, self.z)
        z = self.zz - target.origin[2]
        taper = target.taper
        self.slopes = np.tan([taper.width_negative, taper.width_positive])
        self.normals = np.array([[-1., 0., self.slopes[0]], [1., 0., self.slopes[1]]])
        self.plane_offsets = target.width / 2 + self.normals @ target.origin
        self.face_x = np.stack((
            target.origin[0] - target.width / 2 + self.slopes[0] * z,
            target.origin[0] + target.width / 2 - self.slopes[1] * z))
        self.valid = ((self.yy >= target.origin[1] - target.length / 2 + np.tan(taper.length_negative) * z)
                      & (self.yy <= target.origin[1] + target.length / 2 - np.tan(taper.length_positive) * z))
        area = target.length / ny * target.height / nz
        self.weights = (np.sqrt(1 + self.slopes ** 2)[:, None, None] * area
                        * (1 - z / target.height)[None] * self.valid[None])
        self.total_weight = float(self.weights.sum())
        if self.total_weight <= 0:
            raise ValueError("wall target has no surface grid cells")
        self._geometry = []
        self.points = []
        for obj in inventory:
            planes = []
            for component in obj.geometries:
                centers, normals, _ = component.face_data()
                planes.append((normals, np.einsum("ij,ij->i", centers, normals)))
            self._geometry.append(planes)
            self.points.append(obj.mesh_arrays()[0])
        self._objects = OrderedDict()
        self._states = OrderedDict()
        self._samples = OrderedDict()
        self._lock = RLock()

    def _remember(self, cache, key, value, limit):
        with self._lock:
            cache[key] = value
            cache.move_to_end(key)
            if len(cache) > limit:
                cache.popitem(last=False)
        return value

    def empty_envelope(self):
        result = np.empty((2, *self.yy.shape))
        result[0] = np.inf
        result[1] = -np.inf
        return result

    def object_envelope(self, index, pose):
        pose = np.asarray(pose, dtype=float)
        key = (index, pose.tobytes())
        with self._lock:
            if key in self._objects:
                self._objects.move_to_end(key)
                return self._objects[key]
        rotation = quaternion_matrix(pose[3:])
        points = self.points[index] @ rotation.T + pose[:3]
        lo, hi = points.min(axis=0), points.max(axis=0)
        mask = (self.valid & (self.yy >= lo[1]) & (self.yy <= hi[1])
                & (self.zz >= lo[2]) & (self.zz <= hi[2]))
        rows, cols = np.nonzero(mask)
        envelope = self.empty_envelope()
        y, z = self.yy[mask], self.zz[mask]
        low, high = self._ray_bounds(index, pose, y, z)
        envelope[0, rows, cols] = low
        envelope[1, rows, cols] = high
        envelope.setflags(write=False)
        return self._remember(self._objects, key, envelope, 256)

    def _ray_bounds(self, index, pose, y, z):
        rotation = quaternion_matrix(pose[3:])
        outer_low, outer_high = np.full(len(y), np.inf), np.full(len(y), -np.inf)
        for normals, offsets in self._geometry[index]:
            a = normals @ rotation.T
            b = offsets + a @ pose[:3]
            rhs = b[:, None] - a[:, 1, None] * y - a[:, 2, None] * z
            lower, upper = a[:, 0] < -1e-10, a[:, 0] > 1e-10
            parallel = ~(lower | upper)
            low = np.max(rhs[lower] / a[lower, 0, None], axis=0) if lower.any() else np.full(len(y), -np.inf)
            high = np.min(rhs[upper] / a[upper, 0, None], axis=0) if upper.any() else np.full(len(y), np.inf)
            hit = (low <= high + 1e-9) & np.all(rhs[parallel] >= -1e-9, axis=0)
            outer_low[hit] = np.minimum(outer_low[hit], low[hit])
            outer_high[hit] = np.maximum(outer_high[hit], high[hit])
        return outer_low, outer_high

    def surface_samples(self, index, quaternion, side):
        """Cached exterior samples for cheap proposal estimates, not final credit."""
        key = (index, np.asarray(quaternion).tobytes(), side)
        with self._lock:
            if key in self._samples:
                self._samples.move_to_end(key)
                return self._samples[key]
        points = self.points[index] @ quaternion_matrix(quaternion).T
        lo, hi = points.min(axis=0), points.max(axis=0)
        spacing = np.array([self.target.length / len(self.y), self.target.height / len(self.z)])
        ny, nz = np.clip(np.ceil((hi - lo)[1:] / spacing).astype(int), 2, 16)
        y = lo[1] + (np.arange(ny) + .5) * (hi[1] - lo[1]) / ny
        z = lo[2] + (np.arange(nz) + .5) * (hi[2] - lo[2]) / nz
        yy, zz = np.meshgrid(y, z)
        x = self._ray_bounds(index, np.r_[0., 0., 0., quaternion], yy.ravel(), zz.ravel())[side]
        valid = np.isfinite(x)
        samples = np.column_stack((x[valid], yy.ravel()[valid], zz.ravel()[valid]))
        area = (hi[1] - lo[1]) * (hi[2] - lo[2]) / (ny * nz)
        samples.setflags(write=False)
        return self._remember(self._samples, key, (samples, area), 256)

    def proposal_gains(self, envelope, index, quaternion, translations, side, *,
                       surface_samples=None, quality=None):
        """Batch quadrature over each candidate's outward-facing surface."""
        samples, area = (self.surface_samples(index, quaternion, side)
                         if surface_samples is None else surface_samples)
        points = np.asarray(translations)[:, None, :] + samples[None]
        target = self.target
        columns = np.floor((points[..., 1] - target.origin[1] + target.length / 2)
                           / target.length * len(self.y)).astype(int)
        rows = np.floor((points[..., 2] - target.origin[2]) / target.height * len(self.z)).astype(int)
        valid = (columns >= 0) & (columns < len(self.y)) & (rows >= 0) & (rows < len(self.z))
        rows, columns = np.clip(rows, 0, len(self.z) - 1), np.clip(columns, 0, len(self.y) - 1)
        valid &= self.valid[rows, columns]
        old_x = envelope[side, rows, columns]
        new_x = (np.minimum(old_x, points[..., 0]) if side == 0 else np.maximum(old_x, points[..., 0]))
        # Evaluate the analytic plane at the quadrature point, not at its cell center.
        face_x = (target.origin[0] + (-1 if side == 0 else 1)
                  * (target.width / 2 - self.slopes[side] * (points[..., 2] - target.origin[2])))
        scale = np.sqrt(1 + self.slopes[side] ** 2)
        old_q = (self.quality(envelope) if quality is None else quality)[side, rows, columns]
        new_q = np.clip(1 - np.abs(new_x - face_x) / scale / self.config.surface_tolerance, 0, 1)
        own_q = np.clip(1 - np.abs(points[..., 0] - face_x) / scale / self.config.surface_tolerance, 0, 1)
        weights = area * scale * np.clip(1 - (points[..., 2] - target.origin[2]) / target.height, 0, 1)
        return (np.sum(np.where(valid, (new_q - old_q) * weights, 0.), axis=1) / self.total_weight,
                np.any(valid & (own_q > 0), axis=1))

    @staticmethod
    def union(first, second):
        return np.stack((np.minimum(first[0], second[0]), np.maximum(first[1], second[1])))

    def envelope(self, state):
        key = tuple((i, np.asarray(state.object_poses[state.object_ids[i]], dtype=float).tobytes())
                    for i in sorted(state.placed_object_indices))
        with self._lock:
            if key in self._states:
                self._states.move_to_end(key)
                return self._states[key]
        result = self.empty_envelope()
        for i, _ in key:
            result = self.union(result, self.object_envelope(i, state.object_poses[state.object_ids[i]]))
        result.setflags(write=False)
        return self._remember(self._states, key, result, 16)

    def quality(self, envelope):
        distance = np.abs(envelope - self.face_x) / np.sqrt(1 + self.slopes ** 2)[:, None, None]
        return np.where(self.valid[None], np.clip(1 - distance / self.config.surface_tolerance, 0, 1), 0.)

    def progress(self, envelope):
        return float(np.sum(self.weights * self.quality(envelope)) / self.total_weight)

    def remaining(self, envelope):
        return np.sum(self.weights * (1 - self.quality(envelope)), axis=(1, 2))

    def gain(self, envelope, index, pose):
        return self.progress(self.union(envelope, self.object_envelope(index, pose))) - self.progress(envelope)

    def tangent_x(self, index, quaternion, side, placement_z):
        normal = self.normals[side]
        rotated = self.points[index] @ quaternion_matrix(quaternion).T
        support = np.max(rotated @ normal)
        return (self.plane_offsets[side] - support - normal[2] * placement_z) / normal[0]

    def diagnostics(self, state):
        envelope = self.envelope(state)
        quality = self.quality(envelope)
        coverage = np.sum(self.weights * quality, axis=(1, 2)) / self.weights.sum(axis=(1, 2))
        memberships = {}
        for index in state.placed_object_indices:
            q = self.quality(self.object_envelope(index, state.object_poses[state.object_ids[index]]))
            memberships[index] = set(np.flatnonzero(np.any(q > 0, axis=(1, 2))).tolist())
        # Ground is deliberately excluded: a shared foundation is not a row bond.
        graph = {i: set() for i in memberships}
        for contact in state.contact_points:
            if contact.ground_first or contact.ground_second:
                continue
            a, b = contact.object_index_first, contact.object_index_second
            if a in graph and b in graph and a != b:
                graph[a].add(b)
                graph[b].add(a)
        unseen = set(graph)
        connected = False
        while unseen:
            pending, faces = [unseen.pop()], set()
            while pending:
                i = pending.pop()
                faces.update(memberships[i])
                neighbors = graph[i] & unseen
                unseen.difference_update(neighbors)
                pending.extend(neighbors)
            connected |= len(faces) == 2
        return {"construction_mode": "two_sided_wall",
                "wall_surface_progress": self.progress(envelope),
                "wall_negative_x_coverage": float(coverage[0]),
                "wall_positive_x_coverage": float(coverage[1]),
                "wall_cross_row_connected": bool(connected),
                "wall_contact_data_available": bool(state.contact_points)}
