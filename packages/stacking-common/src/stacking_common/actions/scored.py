"""Scored placement proposals over canonical scenes, with a private RNG.

The array scoring kernels preserve stacking-planner's score-mode behavior.
The small views below adapt canonical objects to those kernels; no legacy or
platform package is imported. Both platforms use this sampler with their own
physical limits and orientation providers.
"""

from types import SimpleNamespace
from contextlib import contextmanager
from dataclasses import asdict
import random
import time

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .connectivity import PreviousActionConnectivity, projected_footprint
from . import _scored_floor as floor
from ._support_maps import bottom_height_kernel
from .types import PlanarPlacement, PlacementFilterDecision
from .generator import PlacementActionGenerator
from stacking_common.domain import CompositeTargetGeometry


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
    """Orientation-aware scores, empty-scene corner seeds, and elite/Boltzmann draws."""

    def __init__(self, inventory, config, orientation_provider, settings, *, seed=0, floor_policy=None, connectivity_enabled=True):
        self.floor_policy = floor_policy
        self.relax_corner_seed = False
        self._active_floor = None
        self.inventory = inventory
        self.config = config
        self.orientation_provider = orientation_provider
        self.connectivity_enabled = connectivity_enabled
        self.settings = settings
        if settings.get("connectivity", "active_floor") not in ("active_floor", "previous_action", "previous_action_soft"):
            raise ValueError("planar.connectivity must be active_floor, previous_action, or previous_action_soft")
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.stones = [_StoneView(obj, inventory.initial_pose(obj.object_id)) for obj in inventory]
        # Fixed across states/orientations, so exhausting the inventory cannot
        # silently change the distance preference's physical scale.
        self.connectivity_length = max(float(np.cbrt(np.mean(
            [obj.volume for obj in inventory]))), 1e-12) if self.stones else 1.

        composite = isinstance(inventory.target, CompositeTargetGeometry)
        self.target_wall = None if composite else SimpleNamespace(width=inventory.target.width, length=inventory.target.length,
            height=inventory.target.height, origin=inventory.target.origin,
            taper=asdict(inventory.target.taper))
        self._kernels = {}
        self._footprint_bounds = {}
        self._footprint_polygons = {}
        self._quaternions = {}
        self._scores = {}
        self._sampled_scores = {}
        self._state_map_bundle = None
        self._score_context = None
        self._score_keys = {}
        self._composite = None
        if composite:
            from .floor import ActiveFloorPolicy, floor_policy_settings
            from .composite_sampling import CompositePlanarSampler
            if self.floor_policy is None:
                self.floor_policy = ActiveFloorPolicy(inventory, **floor_policy_settings(settings["floor"]))
            self._composite = CompositePlanarSampler(self)

    def view(self, state, observation):
        planar = {
            **self.settings,
            "connectivity_enabled": self.connectivity_enabled,
            "connectivity_length": self.connectivity_length,
            "score_map_size": list(reversed(self.config.grid_shape)),
            "floor": {**self.settings["floor"],
                           "boundary_inset_radius_scale": self.config.target_anchor_inset_scale},
        }
        return SimpleNamespace(
            stones=self.stones, stone_set=self.inventory.object_ids,
            target_wall=self.target_wall,
            xlim=observation.height_map_xlim, ylim=observation.height_map_ylim,
            cfg=SimpleNamespace(action=SimpleNamespace(planar=planar)),
            support_settings=SimpleNamespace(
                ground_z=self.inventory.target.origin[2],
                z_tolerance=self.settings["floor"]["bottom_tolerance"],
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
        if self._composite is not None:
            return self._composite.score_map(state, observation, object_index)
        # State/observation arrays are mutable. Key their contents, not just
        # their identities, and retain maps only for the current scene.
        height = np.asarray(observation.scene_height_map)
        context = (
            tuple(state.object_ids), tuple(state.placed_object_indices),
            tuple((object_id, np.asarray(pose).tobytes())
                  for object_id, pose in sorted(state.object_poses.items())),
            height.shape, height.dtype.str, height.tobytes(),
            tuple(observation.height_map_xlim), tuple(observation.height_map_ylim),
            self.config, repr(self.settings),
            0 if self.floor_policy is None else self.floor_policy.fallback_level(state),
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
        self._score_state = self.state_view(state)
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
            rotations = Rotation.from_quat(quaternions).as_matrix()
            rotated = self.stones[object_index].points @ rotations.transpose(0, 2, 1)
            self._footprint_polygons[key] = [projected_footprint(points) for points in rotated]
            self._footprint_bounds[key] = np.stack(
                (rotated[:, :, :2].min(axis=1), rotated[:, :, :2].max(axis=1)), axis=1)
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
            footprint_bounds=self._footprint_bounds[key],
            footprint_polygons=self._footprint_polygons[key],
            active_floor=self._active_floor, state_map_bundle=self._state_map_bundle)
        if not state.placed_object_indices:
            # Corner seeds use full footprints even when ordinary sampling
            # permits overhang via target_anchor_inset_scale < 1.
            corner_bounds = []
            for bounds in self._footprint_bounds[key]:
                static = floor.target_anchor_static(view, object_index, xy_bounds=bounds)
                static = (*static[:-1], max(self.config.target_anchor_inset_scale, 1.) * bounds)
                corner_bounds.append(floor.target_anchor_bounds(view, object_index, _static=static))
            scores["corner_candidates"] = floor.corner_seed_candidates(
                scores.get("candidates", []), corner_bounds, np.array([dx, dy]))
        self._scores[object_index] = (observation, scores)
        self._score_keys[object_index] = key
        self._sampled_scores.pop(object_index, None)
        return scores

    def sample(self, inventory, state, observation, object_index, count, sampling_priority):
        if self._composite is not None:
            return self._composite.sample(state, observation, object_index, count, sampling_priority)
        scores = self.score_map(state, observation, object_index)
        candidates = scores.get("candidates", [])
        if not candidates:
            return ()
        view = self.view(state, observation)
        corner_seeds = (scores.get("corner_candidates", [])
                        if not state.placed_object_indices and not self.relax_corner_seed else [])
        pools = [] if corner_seeds else [np.arange(len(candidates))]
        if sampling_priority == "lower_floor_first" and state.placed_object_indices:
            layer = (self._active_floor if self.floor_policy is not None
                     else floor.active_floor_context(view, self.state_view(state)))
            if layer is not None:
                ceiling = layer["support_z"] + layer["bottom_tol"]
                # Preserve score/diversity sampling within each pool, but spend
                # the proposal budget on the unfilled active floor first.
                heights = candidates.columns["height"]
                pools = [np.flatnonzero(heights <= ceiling), np.flatnonzero(heights > ceiling)]
        selected = (floor.sample_corner_candidates(view, corner_seeds, count, rng=self.rng)
                    if corner_seeds else [])
        for pool in pools:
            remaining = count - len(selected)
            if len(pool) and remaining > 0:
                chosen = floor.sample_scored_indices(
                    view.cfg.action.planar, candidates.columns["score"][pool],
                    remaining, True, rng=self.rng)
                selected.extend(candidates[pool[index]] for index in chosen)
        quaternions = self.quaternions(state, observation, object_index)
        proposals = tuple(PlanarPlacement(
            pose=np.r_[item["xy"], 0., quaternions[item["orientation_index"]]],
            score=float(item["score"]),
            diagnostics={"source": "scored_floor", "orientation_index": item["orientation_index"],
                         "sampling_policy": ("corner_seed" if corner_seeds else "scored"),
                         **({"seed_corner": item["seed_corner"]} if "seed_corner" in item else {}),
                         **({"corner_seed_fallback": True}
                            if not state.placed_object_indices and not corner_seeds else {}),
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
        if self._composite is not None:
            return self._composite.score_at_pose(observation, pose, object_index)
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
        # The displayed XY map contains the best rotation at each cell. A CEM
        # mutation must instead evaluate its own (possibly off-bank) rotation.
        # Freeze the fresh bank's scales so its normalized score is comparable.
        view = self.view(self._score_state, observation)
        xs, ys = scores["x_coords"], scores["y_coords"]
        dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
        points = self.stones[object_index].points
        rotated = Rotation.from_quat(pose[3:]).apply(points)
        bounds = np.array([rotated[:, :2].min(0), rotated[:, :2].max(0)])
        scores = floor.score_xy_debug_map(
            view, self._score_state, object_index,
            scene_height_map=observation.scene_height_map,
            support_kernels=[bottom_height_kernel(points, pose[3:], dx, dy)],
            footprint_bounds=bounds[None], footprint_polygons=[projected_footprint(rotated)],
            active_floor=self._active_floor,
            state_map_bundle=self._state_map_bundle, normalization=scores["normalization"])
        col = int(np.argmin(np.abs(xs - pose[0])))
        row = int(np.argmin(np.abs(ys - pose[1])))
        raw = float(scores["scores"][row, col])
        if not scores["candidate_mask"][row, col] or not np.isfinite(raw):
            return 0., 0.
        reference = scores["normalization"]
        return raw, float(np.clip((raw - reference["score_min"]) / reference["score_span"], 0., 1.))

    def region_metadata(self, object_index, pose):
        return {} if self._composite is None else self._composite.metadata(object_index, pose)

    def allows_initial_pose(self, state, object_index, pose):
        """Apply the optional previous-action mask to continuous CEM seeds too."""
        if (not self.connectivity_enabled
                or self.settings.get("connectivity", "active_floor") != "previous_action"
                or not state.placed_object_indices):
            return True
        check = PreviousActionConnectivity(self.inventory, self.settings["contact_gap_tolerance"])
        return check.check(state, object_index, pose).accepted



class ScoredPlacementActionGenerator(PlacementActionGenerator):
    """Draw fresh pose populations with optional volume-biased stone budgets.

    Private seeded RNGs keep runs reproducible. Retry offsets do not skip a
    newly drawn prefix; a zero volume exponent preserves legacy allocation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if isinstance(getattr(self.inventory, "target", None), CompositeTargetGeometry):
            # Mutations can cross gaps even when all fresh seeds are valid.
            self.filters += (TargetXYOverlapFilter(),)
        if getattr(self.planar_sampler, "settings", {}).get("connectivity") == "previous_action":
            from .connectivity import RootConnectivityPreference
            policy = (RootConnectivityPreference if getattr(self.planar_sampler, "root_global_fraction", 0.) > 0.
                      else PreviousActionConnectivity)
            self.filters += (policy(self.inventory, self.planar_sampler.settings["contact_gap_tolerance"]),)
        self.root_global_fraction = getattr(self.planar_sampler, "root_global_fraction", 0.)
        self.rng = random.Random(self.planar_sampler.seed)
        self._sampling_calls = 0
        self._stone_weights = None
        if self.config.stone_volume_exponent > 0:
            volumes = np.asarray([obj.volume for obj in self.inventory], dtype=float)
            if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
                raise ValueError("volume-biased sampling requires finite positive stone volumes")
            if len(volumes):
                weights = (volumes / volumes.max()) ** self.config.stone_volume_exponent
                self._stone_weights = np.maximum(weights, np.finfo(float).tiny).tolist()

    def _initialize_candidate(self, state, object_index, proposal, *, scene_contexts=None):
        allows = getattr(self.planar_sampler, "allows_initial_pose", None)
        if callable(allows) and not allows(state, object_index, proposal.pose):
            return None, "previous_action_disconnected"
        return super()._initialize_candidate(state, object_index, proposal,
                                             scene_contexts=scene_contexts)

    def _proposal_counts(self, available, budget, priority):
        """Allocate fresh proposals, retaining the per-stone attempt cap."""
        counts = dict.fromkeys(available, 0)
        if self._stone_weights is not None:
            pool = list(available)
            weights = [self._stone_weights[index] for index in pool]
            for _ in range(min(budget, len(pool) * self.config.max_poses_per_object)):
                slot = self.rng.choices(range(len(pool)), weights=weights, k=1)[0]
                index = pool[slot]
                counts[index] += 1
                if counts[index] == self.config.max_poses_per_object:
                    pool.pop(slot)
                    weights.pop(slot)
            return counts
        # Preserve the previous allocation and RNG sequence when bias is disabled.
        available = list(available)
        if not priority:
            self.rng.shuffle(available)
        elif available and budget < len(available):
            offset = (self._sampling_calls - 1) % len(available)
            available = available[offset:] + available[:offset]
        counts = dict.fromkeys(available, 0)
        for _ in range(self.config.max_poses_per_object):
            for index in available:
                if budget <= 0:
                    break
                counts[index] += 1
                budget -= 1
            if budget <= 0:
                break
        return counts

    @contextmanager
    def corner_seed_fallback(self):
        """Retry an exhausted empty root using ordinary scored sampling."""
        previous = self.planar_sampler.relax_corner_seed
        self.planar_sampler.relax_corner_seed = True
        try:
            yield True
        finally:
            self.planar_sampler.relax_corner_seed = previous

    def select_proposals(self, state, observation, request):
        """Draw a fresh population; retry offsets do not skip its prefix."""
        self._validate_state(state)
        self._sampling_calls += 1
        available = [i for i, object_id in enumerate(state.object_ids)
                     if i not in state.placed_object_indices
                     and object_id not in self.config.banned_object_ids
                     and object_id not in request.execution_rejections.object_ids]
        priority = request.sampling_priority == "lower_floor_first" and bool(state.placed_object_indices)
        counts = self._proposal_counts(available, request.max_candidates, priority)
        selected = []
        for index, count in counts.items():
            if not count or (request.deadline is not None and time.monotonic() >= request.deadline):
                continue
            # Compute the elite/random mixture for the allocated budget.
            # Oversampling then taking a prefix can retain only elites.
            proposals = self.planar_sampler.sample(self.inventory, state, observation,
                index, count, request.sampling_priority)
            selected.extend((index, proposal) for proposal in proposals)
        return tuple(selected), not available


class TargetXYOverlapFilter:
    """Action-build gate; volumetric containment remains a simulation outcome."""

    def evaluate(self, inventory, state, object_index, initial, solved):
        points, _ = inventory[object_index].mesh_arrays()
        points = Rotation.from_quat(solved.pose[3:]).apply(points) + solved.pose[:3]
        target = inventory.target
        if isinstance(target, CompositeTargetGeometry):
            from .target_regions import polygon, polygons_overlap
            footprint = polygon(points[:, :2])
            overlap = any(polygons_overlap(footprint, polygon(component.world_geometry.points[:, :2]))
                          for component in target.components)
            diagnostics = {"target_structure_xy_overlap": overlap}
            return (PlacementFilterDecision.accept(**diagnostics) if overlap else
                    PlacementFilterDecision.reject("final_pose_outside_target", **diagnostics))
        half = np.array([target.width, target.length]) / 2.
        overlap = bool(np.all(points[:, :2].max(axis=0) >= target.origin[:2] - half)
                       and np.all(points[:, :2].min(axis=0) <= target.origin[:2] + half))
        diagnostics = {"target_structure_xy_overlap": overlap}
        return (PlacementFilterDecision.accept(**diagnostics) if overlap else
                PlacementFilterDecision.reject("final_pose_outside_target", **diagnostics))
