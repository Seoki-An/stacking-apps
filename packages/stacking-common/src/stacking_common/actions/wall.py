"""Opt-in proposals along the two exterior faces of one thick wall."""

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull

from .scored import ScoredPlanarPoseSampler
from ._support_maps import bottom_height_kernel
from . import _scored_floor as floor
from .types import PlanarPlacement, PlacementFilterDecision
from stacking_common.wall import WallSurfaces


class WallPlanarPoseSampler(ScoredPlanarPoseSampler):
    """Reuse orientation and sampling contracts, with no global active floor."""

    def __init__(self, *args, construction, **kwargs):
        super().__init__(*args, **kwargs)
        self.wall = WallSurfaces(self.inventory, construction, self.config.grid_shape)
        self._wall_context = None
        self._wall_observation = None
        self._wall_kernels = {}
        self._target_planes = ConvexHull(self.inventory.target.points).equations

    def _support(self, observation, index, quaternion, xy):
        height = np.asarray(observation.scene_height_map)
        dx = np.diff(observation.height_map_xlim)[0] / max(height.shape[1] - 1, 1)
        dy = np.diff(observation.height_map_ylim)[0] / max(height.shape[0] - 1, 1)
        key = (index, np.asarray(quaternion).tobytes(), float(dx), float(dy))
        if key not in self._wall_kernels:
            if len(self._wall_kernels) >= 256:
                self._wall_kernels.clear()
            self._wall_kernels[key] = bottom_height_kernel(self.stones[index].points, quaternion, dx, dy)
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
            self._represented = np.any(self.wall.quality(self._envelope) > 0, axis=(1, 2))
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
        gain = self.wall.progress(self.wall.union(self._envelope, surface)) - self.wall.progress(self._envelope)
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
        for orientation_index, q in enumerate(self.quaternions(state, observation, object_index)):
            rotated = Rotation.from_quat(q).apply(self.wall.points[object_index])
            ys = y_grid[(y_grid + rotated[:, 1].min() >= target.origin[1] - target.length / 2)
                        & (y_grid + rotated[:, 1].max() <= target.origin[1] + target.length / 2)]
            for side in (0, 1):
                if not len(ys):
                    continue
                z = np.full(len(ys), target.origin[2] - rotated[:, 2].min())
                xy = np.column_stack((np.zeros(len(ys)), ys))
                # Bounded support/taper updates. Final geometry checks reject a
                # seed if this approximation has not reached a feasible pose.
                for _ in range(3):
                    xy[:, 0] = self.wall.tangent_x(object_index, q, side, z)
                    z, support, height = self._support(observation, object_index, q, xy)
                translations = np.column_stack((xy, z))
                a, b = self._target_planes[:, :3], -self._target_planes[:, 3]
                extent = np.max(rotated @ a.T, axis=0)
                valid = np.all(translations @ a.T + extent <= b + 1e-7, axis=1)
                lo, hi = translations + rotated.min(axis=0), translations + rotated.max(axis=0)
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
                gains, fits = self.wall.proposal_gains(self._envelope, object_index, q, translations[indices], side)
                weights = self.settings["score"]
                normalized_height = np.clip((height - target.origin[2]) / target.height, 0, 1)
                score = (float(weights["height"]) * normalized_height[indices]
                         + float(weights["support_likelihood"]) * support[indices]
                         + self.wall.config.proposal_weight * gains)
                items = []
                for j, i in enumerate(indices):
                    if fits[j]:
                        items.append(dict(xy=xy[i].copy(), pose=np.r_[translations[i], q],
                            score=float(score[j]), height=float(height[i]),
                            support_likelihood=float(support[i]), wall_surface_gain=float(gains[j]),
                            side=side, orientation_index=orientation_index))
                if not self._represented[side] and items and not self.relax_corner_seed:
                    # Seed an unstarted face from either end; no persistent cursor.
                    items = [items[0]] if len(items) == 1 else [items[0], items[-1]]
                candidates.extend(items)
        values = np.asarray([item["score"] for item in candidates])
        for item in candidates:
            item["normalized_score"] = float((item["score"] - values.min()) / max(np.ptp(values), 1e-12))
        scores = {"candidates": candidates, "wall_remaining": self.wall.remaining(self._envelope),
                  "wall_quality": self.wall.quality(self._envelope)}
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
