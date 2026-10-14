"""Opt-in proposals along the two exterior faces of one thick wall."""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull

from .scored import ScoredPlanarPoseSampler
from ._support_maps import BottomHeightKernel, bottom_height_kernel
from . import _scored_floor as floor
from .types import PlanarPlacement, PlacementFilterDecision
from stacking_common.wall import WallSurfaces
from stacking_common.domain.geometry import quaternion_matrix


@dataclass(frozen=True)
class _OrientationGeometry:
    quaternion: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    target_extent: np.ndarray
    side_support: np.ndarray
    bottom_kernel: BottomHeightKernel
    surface_samples: tuple


class WallPlanarPoseSampler(ScoredPlanarPoseSampler):
    """Reuse orientation and sampling contracts, with no global active floor."""

    def __init__(self, *args, construction, **kwargs):
        super().__init__(*args, **kwargs)
        self.wall = WallSurfaces(self.inventory, construction, self.config.grid_shape)
        self._wall_context = None
        self._wall_observation = None
        # Finite proposal banks live separately from arbitrary CEM rotations.
        # Keep only the latest bank/grid spacing per object.
        self._orientation_geometry = {}
        self._wall_kernels = OrderedDict()
        self._target_planes = ConvexHull(self.inventory.target.points).equations

    @staticmethod
    def _spacing(observation):
        height = np.asarray(observation.scene_height_map)
        dx = np.diff(observation.height_map_xlim)[0] / max(height.shape[1] - 1, 1)
        dy = np.diff(observation.height_map_ylim)[0] / max(height.shape[0] - 1, 1)
        return float(dx), float(dy)

    def _geometry(self, state, observation, index):
        quaternions = self.quaternions(state, observation, index)
        dx, dy = self._spacing(observation)
        key = (tuple(q.tobytes() for q in quaternions), dx, dy)
        cached = self._orientation_geometry.get(index)
        if cached is not None and cached[0] == key:
            return cached[1]
        geometries = []
        for q in quaternions:
            rotated = Rotation.from_quat(q).apply(self.wall.points[index])
            # Use the same plane support arithmetic as WallSurfaces.tangent_x.
            plane_points = self.wall.points[index] @ quaternion_matrix(q).T
            geometries.append(_OrientationGeometry(
                q, rotated.min(axis=0), rotated.max(axis=0),
                np.max(rotated @ self._target_planes[:, :3].T, axis=0),
                np.array([np.max(plane_points @ normal) for normal in self.wall.normals]),
                bottom_height_kernel(self.stones[index].points, q, dx, dy),
                tuple(self.wall.surface_samples(index, q, side) for side in (0, 1)),
            ))
        self._orientation_geometry[index] = (key, tuple(geometries))
        return self._orientation_geometry[index][1]

    def _support(self, observation, index, quaternion, xy, *, kernel=None):
        height = np.asarray(observation.scene_height_map)
        dx, dy = self._spacing(observation)
        if kernel is None:
            key = (index, np.asarray(quaternion).tobytes(), dx, dy)
            if key not in self._wall_kernels:
                self._wall_kernels[key] = bottom_height_kernel(self.stones[index].points, quaternion, dx, dy)
                if len(self._wall_kernels) > 256:
                    self._wall_kernels.popitem(last=False)
            self._wall_kernels.move_to_end(key)
            kernel = self._wall_kernels[key]
        xy = np.atleast_2d(xy)
        cols = (xy[:, 0, None] - observation.height_map_xlim[0]) / dx + kernel.x_offsets
        rows = (xy[:, 1, None] - observation.height_map_ylim[0]) / dy + kernel.y_offsets
        sampled = map_coordinates(height, [rows, cols], order=0, mode="constant", cval=np.nan)
        sampled = np.maximum(sampled, self.inventory.target.origin[2])
        clearance = sampled - kernel.bottom_height
        placement = np.max(clearance, axis=1)
        contact = clearance >= placement[:, None] - self.settings["contact_gap_tolerance"]
        support = np.mean(contact, axis=1)
        contact_height = np.sum(np.where(contact, sampled, 0.), axis=1) / np.maximum(contact.sum(axis=1), 1)
        return placement, support, contact_height

    def _context(self, state, observation):
        height = np.asarray(observation.scene_height_map)
        key = (tuple(state.placed_object_indices),
               tuple((i, np.asarray(state.object_poses[state.object_ids[i]]).tobytes())
                     for i in state.placed_object_indices), height.shape, height.tobytes(),
               tuple(observation.height_map_xlim), tuple(observation.height_map_ylim),
               self.relax_corner_seed)
        if key != self._wall_context:
            self._scores.clear()
            self._wall_context = key
            self._envelope = self.wall.envelope(state)
            self._quality = self.wall.quality(self._envelope)
            self._remaining = np.sum(self.wall.weights * (1 - self._quality), axis=(1, 2))
            self._progress = float(np.sum(self.wall.weights * self._quality) / self.wall.total_weight)
            self._represented = np.any(self._quality > 0, axis=(1, 2))
            self._neighbors = []
            for i in state.placed_object_indices:
                pose = state.object_poses[state.object_ids[i]]
                points = Rotation.from_quat(pose[3:]).apply(self.wall.points[i]) + pose[:3]
                self._neighbors.append((points.min(axis=0), points.max(axis=0)))
        self._wall_observation = observation

    def _eligible(self, index, pose, sides):
        points = Rotation.from_quat(pose[3:]).apply(self.wall.points[index]) + pose[:3]
        lo, hi = points.min(axis=0), points.max(axis=0)
        ground = lo[2] <= self.inventory.target.origin[2] + self.settings["floor"]["bottom_tolerance"]
        if ground and any(not self._represented[side] for side in sides):
            return True
        # Conservative geometric adjacency; nominal physics remains authoritative.
        gap = self.settings["contact_gap_tolerance"]
        return any(np.all(lo <= upper + gap) and np.all(hi >= lower - gap)
                   for lower, upper in self._neighbors)

    def _candidate(self, observation, index, pose, side=None):
        placement, support, contact_height = self._support(observation, index, pose[3:], pose[:2])
        pose = pose.copy()
        pose[2] = placement[0]
        if not np.all(np.isfinite(pose)):
            return None
        points = Rotation.from_quat(pose[3:]).apply(self.wall.points[index]) + pose[:3]
        # Full target geometry, including Y ends and all tapers; no center split.
        if not np.all(self.inventory.target.contains(points, tolerance=1e-7)):
            return None
        surface = self.wall.object_envelope(index, pose)
        represented = np.flatnonzero(np.any(self.wall.quality(surface) > 0, axis=(1, 2)))
        if not len(represented) or (side is not None and side not in represented):
            return None
        if not self._eligible(index, pose, represented):
            return None
        gain = self.wall.progress(self.wall.union(self._envelope, surface)) - self._progress
        # Height uses fixed target units so both sides and stones are comparable.
        height = np.clip((contact_height[0] - self.inventory.target.origin[2]) / self.inventory.target.height, 0, 1)
        weights = self.settings["score"]
        score = (float(weights["height"]) * height + float(weights["support_likelihood"]) * support[0]
                 + self.wall.config.proposal_weight * gain)
        return dict(xy=pose[:2], pose=pose, score=float(score), height=float(contact_height[0]),
                    support_likelihood=float(support[0]), wall_surface_gain=float(gain),
                    side=int(represented[0] if side is None else side), sides=represented.tolist())

    def score_map(self, state, observation, object_index):
        self._context(state, observation)
        if object_index in self._scores:
            return self._scores[object_index][1]
        target = self.inventory.target
        candidates = []
        y_grid = np.linspace(target.origin[1] - target.length / 2,
                             target.origin[1] + target.length / 2, self.config.grid_shape[1])
        for orientation_index, geometry in enumerate(self._geometry(state, observation, object_index)):
            q = geometry.quaternion
            ys = y_grid[(y_grid + geometry.lower[1] >= target.origin[1] - target.length / 2)
                        & (y_grid + geometry.upper[1] <= target.origin[1] + target.length / 2)]
            if not len(ys):
                continue
            # Both faces share this orientation's bottom kernel. Flatten them
            # into one interpolation call per support/taper update.
            z = np.full((2, len(ys)), target.origin[2] - geometry.lower[2])
            xy = np.zeros((2, len(ys), 2))
            xy[:, :, 1] = ys
            for _ in range(3):
                xy[:, :, 0] = (self.wall.plane_offsets[:, None] - geometry.side_support[:, None]
                               - self.wall.normals[:, 2, None] * z) / self.wall.normals[:, 0, None]
                z, support, height = (values.reshape(2, len(ys)) for values in self._support(
                    observation, object_index, q, xy.reshape(-1, 2), kernel=geometry.bottom_kernel))
            for side in (0, 1):
                translations = np.column_stack((xy[side], z[side]))
                a, b = self._target_planes[:, :3], -self._target_planes[:, 3]
                valid = np.all(translations @ a.T + geometry.target_extent <= b + 1e-7, axis=1)
                lo, hi = translations + geometry.lower, translations + geometry.upper
                adjacent = np.zeros(len(ys), dtype=bool)
                gap = self.settings["contact_gap_tolerance"]
                for lower, upper in self._neighbors:
                    adjacent |= np.all(lo <= upper + gap, axis=1) & np.all(hi >= lower - gap, axis=1)
                if not self._represented[side]:
                    adjacent |= lo[:, 2] <= target.origin[2] + self.settings["floor"]["bottom_tolerance"]
                valid &= adjacent & np.all(np.isfinite(translations), axis=1)
                indices = np.flatnonzero(valid)
                if not len(indices):
                    continue
                gains, fits = self.wall.proposal_gains(
                    self._envelope, object_index, q, translations[indices], side,
                    surface_samples=geometry.surface_samples[side], quality=self._quality)
                weights = self.settings["score"]
                normalized_height = np.clip((height[side] - target.origin[2]) / target.height, 0, 1)
                score = (float(weights["height"]) * normalized_height[indices]
                         + float(weights["support_likelihood"]) * support[side, indices]
                         + self.wall.config.proposal_weight * gains)
                items = []
                poses = np.empty((len(ys), 7))
                poses[:, :3], poses[:, 3:] = translations, q
                for j, i in enumerate(indices):
                    if fits[j]:
                        items.append(dict(xy=xy[side, i].copy(), pose=poses[i].copy(),
                            score=float(score[j]), height=float(height[side, i]),
                            support_likelihood=float(support[side, i]), wall_surface_gain=float(gains[j]),
                            side=side, orientation_index=orientation_index))
                if not self._represented[side] and items and not self.relax_corner_seed:
                    # Seed an unstarted face from either end; no persistent cursor.
                    items = [items[0]] if len(items) == 1 else [items[0], items[-1]]
                candidates.extend(items)
        values = np.asarray([item["score"] for item in candidates])
        minimum = float(values.min()) if len(values) else 0.
        span = max(float(np.ptp(values)), 1e-12) if len(values) else 1.
        for item in candidates:
            item["normalized_score"] = float((item["score"] - minimum) / span)
        scores = {"candidates": candidates, "wall_remaining": self._remaining,
                  "wall_quality": self._quality}
        self._scores[object_index] = (observation, scores)
        return scores

    def sample(self, inventory, state, observation, object_index, count, sampling_priority):
        scores = self.score_map(state, observation, object_index)
        pools = [[c for c in scores["candidates"] if c["side"] == side] for side in (0, 1)]
        view = self.view(state, observation)
        # Draw each side's elite/Boltzmann sequence once, then interleave by
        # remaining demand. Prefixes retain opportunities for both sides.
        queues = [list(floor.sample_scored_candidates(view, pool, min(count, len(pool)), True, rng=self.rng))
                  if pool else [] for pool in pools]
        selected, seen = [], set()
        while len(selected) < count and any(queues):
            demand = np.where([bool(pool) for pool in queues], scores["wall_remaining"], 0.)
            if demand.sum() <= 0:
                demand = np.array([bool(pool) for pool in queues], dtype=float)
            side = int(self.rng.choice(2, p=demand / demand.sum()))
            item = queues[side].pop(0)
            key = tuple(np.round(item["pose"], 7))
            if key not in seen:
                seen.add(key)
                selected.append(item)
        return tuple(PlanarPlacement(
            pose=item["pose"], score=item["score"], diagnostics={
                "source": "two_sided_wall", "sampling_policy": "two_sided_wall",
                "wall_proposal_side": ("negative_x", "positive_x")[item["side"]],
                "wall_predicted_surface_gain": item["wall_surface_gain"],
                "orientation_index": item["orientation_index"],
                "height_map_score": item["score"], "height_map_score_normalized": item["normalized_score"],
            }) for item in selected)

    def score_at_object_pose(self, observation, pose, object_index):
        if observation is not self._wall_observation:
            return 0., 0.
        item = self._candidate(observation, object_index, np.asarray(pose, dtype=float))
        if item is None:
            return 0., 0.
        candidates = self._scores.get(object_index, (None, {}))[1].get("candidates", [])
        values = [c["score"] for c in candidates]
        normalized = (np.clip((item["score"] - min(values)) / max(np.ptp(values), 1e-12), 0, 1)
                      if values else 0.)
        return item["score"], float(normalized)


class WallPlacementFilter:
    """Measure solver drift without locking the optimized pose to its seed side."""

    def __init__(self, wall):
        self.wall = wall

    def evaluate(self, inventory, state, object_index, initial, solved):
        return self.check(state, object_index, solved.pose)

    def check(self, state, object_index, pose):
        surface = self.wall.object_envelope(object_index, pose)
        sides = np.any(self.wall.quality(surface) > 0, axis=(1, 2))
        diagnostics = {"wall_solved_sides": [name for name, present in
            zip(("negative_x", "positive_x"), sides) if present],
            "wall_solved_surface_gain": self.wall.gain(self.wall.envelope(state), object_index, pose)}
        if not sides.any():
            return PlacementFilterDecision.reject("wall_surface_miss", **diagnostics)
        return PlacementFilterDecision.accept(**diagnostics)
