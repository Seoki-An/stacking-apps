"""Scored placement proposals over canonical scenes, with a private RNG.

The array scoring kernels preserve stacking-planner's score-mode behavior.
The small views below adapt canonical objects to those kernels; no legacy or
platform package is imported. Both platforms use this sampler with their own
physical limits and orientation providers.
"""

from types import SimpleNamespace
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import random
import time

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from . import _scored_floor as floor
from ._support_maps import bottom_height_kernel
from .types import PlanarPlacement, PlacementFilterDecision
from .generator import PlacementActionGenerator
from .filtering import ActiveFloorFilter
from stacking_common.search.types import ActionGenerationResult


class _StoneView:
    def __init__(self, obj, pose):
        self.id, self.pose = obj.object_id, pose
        points, _ = obj.mesh_arrays()
        hull = ConvexHull(points)
        self.points = points[hull.vertices]
        self.geometries = []
        for geometry in obj.geometries:
            equations = ConvexHull(geometry.points).equations
            self.geometries.append(SimpleNamespace(
                support_planes=(equations[:, :3], -equations[:, 3])))

    def get_global_hull_mesh_array(self):
        return self.points, None

    def local_aabb_extent(self):
        return np.ptp(self.points, axis=0)


class ScoredPlanarPoseSampler:
    """Legacy score-mode geometry, representatives, and Boltzmann selection."""

    def __init__(self, inventory, config, orientation_provider, settings, *, seed=0, floor_policy=None):
        if config.minimum_xy_separation != 0:
            raise ValueError("scored selection requires minimum_xy_separation: 0")
        self.floor_policy = floor_policy
        self.relax_floor_fill = False
        self._active_floor = None
        self.inventory = inventory
        self.config = config
        self.orientation_provider = orientation_provider
        self.settings = settings
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.stones = [_StoneView(obj, inventory.initial_pose(obj.object_id)) for obj in inventory]
        self.target_wall = SimpleNamespace(width=inventory.target.width, length=inventory.target.length,
            height=inventory.target.height, origin=inventory.target.origin,
            taper=asdict(inventory.target.taper))
        self._kernels = {}
        self._quaternions = {}
        self._scores = {}
        self._sampled_scores = {}
        self._state_map_bundle = None
        self._score_context = None
        self._score_keys = {}

    def view(self, state, observation):
        planar = {
            **self.settings,
            "score_map_size": list(reversed(self.config.grid_shape)),
            "floor_fill": {**self.settings["floor_fill"],
                           "boundary_inset_radius_scale": self.config.target_anchor_inset_scale},
        }
        return SimpleNamespace(
            stones=self.stones, stone_set=self.inventory.object_ids,
            target_wall=self.target_wall,
            xlim=observation.height_map_xlim, ylim=observation.height_map_ylim,
            cfg=SimpleNamespace(action=SimpleNamespace(planar=planar)),
            support_settings=SimpleNamespace(
                ground_z=self.inventory.target.origin[2],
                z_tolerance=self.settings["floor_fill"]["lower_floor_bottom_z_tolerance"],
                contact_gap_tolerance=self.settings["contact_gap_tolerance"]),
        )

    @staticmethod
    def state_view(state):
        return SimpleNamespace(stone_seq=state.placed_object_indices,
                               stone_set=state.object_ids, stone_poses=state.object_poses)

    def quaternions(self, state, observation, object_index):
        provider = self.orientation_provider
        variants = getattr(provider, "quaternions", None)
        bases = (variants(self.inventory, state, observation, object_index) if callable(variants)
                 else (provider.quaternion(self.inventory, state, observation, object_index),))
        # Yaw products depend only on the bases; reuse them, returning copies.
        key = (object_index, tuple(np.asarray(base, dtype=float).tobytes() for base in bases))
        if key not in self._quaternions:
            rotations = [Rotation.from_euler("z", yaw) * Rotation.from_quat(base)
                         for base in bases for yaw in self.config.yaw_angles]
            unique = {}
            for rotation in rotations:
                unique.setdefault(tuple(np.round(rotation.as_matrix().ravel(), 7)), rotation.as_quat())
            self._quaternions[key] = tuple(unique.values())
        return tuple(value.copy() for value in self._quaternions[key])

    def score_map(self, state, observation, object_index):
        # State/observation arrays are mutable. Key their contents, not just
        # their identities, and retain maps only for the current scene.
        height = np.asarray(observation.scene_height_map)
        context = (
            tuple(state.object_ids), tuple(state.placed_object_indices),
            tuple((object_id, np.asarray(pose).tobytes())
                  for object_id, pose in sorted(state.object_poses.items())),
            height.shape, height.dtype.str, height.tobytes(),
            tuple(observation.height_map_xlim), tuple(observation.height_map_ylim),
            self.config, repr(self.settings), self.relax_floor_fill,
        )
        if context != self._score_context:
            self._scores.clear()
            self._score_keys.clear()
            self._sampled_scores.clear()
            self._state_map_bundle = None
            self._score_context = context
            self._active_floor = (None if self.floor_policy is None
                                  else self.floor_policy.context_details(state))
        view = self.view(state, observation)
        quaternions = self.quaternions(state, observation, object_index)
        rows, columns = self.config.grid_shape
        dx = np.diff(view.xlim)[0] / max(columns - 1, 1)
        dy = np.diff(view.ylim)[0] / max(rows - 1, 1)
        key = (object_index, float(dx), float(dy), tuple(tuple(q) for q in quaternions))
        if self._score_keys.get(object_index) == key:
            scores = self._scores[object_index][1]
            # CEM lookup uses the current observation's identity, including
            # when an equivalent observation was reconstructed from a state.
            self._scores[object_index] = (observation, scores)
            return scores
        if key not in self._kernels:
            self._kernels[key] = [bottom_height_kernel(self.stones[object_index].points, q, dx, dy)
                                  for q in quaternions]
        if self._state_map_bundle is None:
            shape = floor.score_grid_shape(view, height)
            xy_grid = floor.xy_grid_for_shape(view, shape)
            lower, upper = floor.target_anchor_bounds(view, None)
            valid = np.all((xy_grid >= lower) & (xy_grid <= upper), axis=-1)
            self._state_map_bundle = floor.build_state_map_bundle(
                height, shape, view.support_settings.ground_z, valid,
                smooth_kernel_size=floor.HEIGHT_SCORE_SMOOTH_KERNEL_SIZE,
            )
        scores = floor.score_xy_debug_map(view, self.state_view(state), object_index,
            scene_height_map=observation.scene_height_map, support_kernels=self._kernels[key],
            active_floor=self._active_floor, state_map_bundle=self._state_map_bundle,
            relax_floor_fill=self.relax_floor_fill)
        self._scores[object_index] = (observation, scores)
        self._score_keys[object_index] = key
        self._sampled_scores.pop(object_index, None)
        return scores

    def sample(self, inventory, state, observation, object_index, count, sampling_priority):
        scores = self.score_map(state, observation, object_index)
        candidates = scores.get("candidates", [])
        if not candidates:
            return ()
        view = self.view(state, observation)
        pools = [candidates]
        if (not self.relax_floor_fill and sampling_priority == "lower_floor_first"
                and state.placed_object_indices):
            layer = (self._active_floor if self.floor_policy is not None
                     else floor.active_floor_context(view, self.state_view(state)))
            if layer is not None:
                ceiling = layer["support_z"] + layer["bottom_tol"]
                # Preserve score/diversity sampling within each pool, but spend
                # the proposal budget on the unfilled active floor first.
                pools = [
                    [item for item in candidates if item["height"] <= ceiling],
                    [item for item in candidates if item["height"] > ceiling],
                ]
        selected = []
        for pool in pools:
            remaining = count - len(selected)
            if pool and remaining > 0:
                selected.extend(floor.sample_scored_candidates(
                    view, pool, remaining, True, rng=self.rng))
        quaternions = self.quaternions(state, observation, object_index)
        proposals = tuple(PlanarPlacement(
            pose=np.r_[item["xy"], 0., quaternions[item["orientation_index"]]],
            score=float(item["score"]),
            diagnostics={"source": "scored_floor", "orientation_index": item["orientation_index"],
                         "height_map_score": float(item["score"]),
                         "height_map_score_normalized": float(item.get("normalized_score", 0.))},
        ) for item in selected)
        self._sampled_scores[object_index] = {
            self._pose_key(p.pose): (p.score, p.diagnostics["height_map_score_normalized"])
            for p in proposals}
        return proposals

    @staticmethod
    def _pose_key(pose):
        return tuple(np.round(np.r_[pose[:2], pose[3:]], 7))

    def score_at_object_pose(self, observation, pose, object_index):
        # Mutations must retain the parent object's orientation-aware score map.
        entry = self._scores.get(object_index)
        if entry is None or entry[0] is not observation:
            return 0., 0.
        sampled = self._sampled_scores.get(object_index, {}).get(self._pose_key(pose))
        if sampled is not None:
            return sampled
        scores = entry[1]
        candidates = scores.get("candidates", [])
        if not candidates:
            return 0., 0.
        # Match legacy pose score lookup: sampled poses have exact keys; an
        # unsampled mutation falls back to the nearest XY score-map cell.
        xs, ys = scores["x_coords"], scores["y_coords"]
        col = int(np.argmin(np.abs(xs - pose[0])))
        row = int(np.argmin(np.abs(ys - pose[1])))
        raw = float(scores["scores"][row, col])
        valid = scores["scores"][scores["candidate_mask"] & np.isfinite(scores["scores"])]
        if not scores["candidate_mask"][row, col] or not np.isfinite(raw) or not len(valid):
            return 0., 0.
        span = float(valid.max() - valid.min())
        return raw, (0. if span <= 1e-12 else float((raw - valid.min()) / span))


class ScoredPlacementActionGenerator(PlacementActionGenerator):
    """Allocate each fresh request as legacy action sampling did.

The common deterministic sampler pages through a fixed proposal stream. This
policy instead samples a new population on every call, using private seeded
Python/NumPy RNGs. Retry offsets therefore do not skip a newly drawn prefix.
"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rng = random.Random(self.planar_sampler.seed)
        self._sampling_calls = 0

    @contextmanager
    def floor_fill_fallback(self):
        """Temporarily admit every floor after a strict root search exhausts."""
        filters = self.filters
        if not any(isinstance(item, ActiveFloorFilter) for item in filters):
            yield False
            return
        previous = self.planar_sampler.relax_floor_fill
        self.filters = tuple(
            ActiveFloorFilter(item.policy, relaxed=True)
            if isinstance(item, ActiveFloorFilter) else item for item in filters
        )
        self.planar_sampler.relax_floor_fill = True
        try:
            yield True
        finally:
            self.filters = filters
            self.planar_sampler.relax_floor_fill = previous

    def generate_batch(self, states, observations, requests):
        if not (len(states) == len(observations) == len(requests)):
            raise ValueError("batched action-generation inputs must align")
        jobs = []
        exhausted = []
        attempts = [0 for _ in states]
        for batch, (state, observation, request) in enumerate(zip(states, observations, requests)):
            self._validate_state(state)
            self._sampling_calls += 1
            available = [i for i, object_id in enumerate(state.object_ids)
                         if i not in state.placed_object_indices
                         and object_id not in self.config.banned_object_ids]
            exhausted.append(not available)
            priority = request.sampling_priority == "lower_floor_first" and bool(state.placed_object_indices)
            if not priority:
                self.rng.shuffle(available)
            elif available and request.max_candidates < len(available):
                offset = (self._sampling_calls - 1) % len(available)
                available = available[offset:] + available[:offset]
            counts = dict.fromkeys(available, 0)
            remaining = request.max_candidates
            for _ in range(self.config.max_poses_per_object):
                for index in available:
                    if remaining <= 0:
                        break
                    counts[index] += 1
                    remaining -= 1
                if remaining <= 0:
                    break
            for index, count in counts.items():
                if not count or (request.deadline is not None and time.monotonic() >= request.deadline):
                    continue
                proposals = self.planar_sampler.sample(self.inventory, state, observation,
                    index, self.config.max_poses_per_object, request.sampling_priority)
                count = min(count, len(proposals))
                chosen = proposals[:count] if priority else self.rng.sample(proposals, count)
                jobs.extend((batch, state, index, proposal) for proposal in chosen)
                attempts[batch] += count
        candidates = [[] for _ in states]
        rejections = [Counter() for _ in states]
        for (batch, *_), (candidate, reason) in zip(jobs, self._build_jobs(jobs)):
            if reason is not None:
                rejections[batch][reason] += 1
            elif candidate is not None:
                candidates[batch].append(candidate)
        return tuple(ActionGenerationResult(tuple(candidates[i]), attempts[i], exhausted[i], rejections[i])
                     for i in range(len(states)))


class TargetXYOverlapFilter:
    """Action-build gate; volumetric containment remains a simulation outcome."""

    def evaluate(self, inventory, state, object_index, initial, solved):
        points, _ = inventory[object_index].mesh_arrays()
        points = Rotation.from_quat(solved.pose[3:]).apply(points) + solved.pose[:3]
        target = inventory.target
        half = np.array([target.width, target.length]) / 2.
        overlap = bool(np.all(points[:, :2].max(axis=0) >= target.origin[:2] - half)
                       and np.all(points[:, :2].min(axis=0) <= target.origin[:2] + half))
        diagnostics = {"target_structure_xy_overlap": overlap}
        return (PlacementFilterDecision.accept(**diagnostics) if overlap else
                PlacementFilterDecision.reject("final_pose_outside_target", **diagnostics))
