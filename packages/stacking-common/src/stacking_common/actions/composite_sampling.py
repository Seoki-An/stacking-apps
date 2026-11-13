"""Region-balanced proposals using actual convex-component containment masks."""

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from types import MappingProxyType
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation

from stacking_common.domain.geometry import transform_points
from stacking_common.visualization.heightmap import _rasterize_triangles

from . import _scored_floor as floor
from ._support_maps import bottom_height_kernel, support_likelihood_maps_by_kernel
from .target_regions import CachedAnchorContainment, SamplingRegion
from .types import PlanarPlacement
from .connectivity import bounded_distance_penalty, footprints_within_gap, projected_footprint


class _CandidateRows(Sequence):
    """Read-only diagnostic row view; search keeps candidates in column arrays.

    A dict is constructed only when a caller requests a row (or explicitly
    iterates the diagnostic view), not for every feasible grid location.
    """

    def __init__(self, batches):
        self.batches = tuple(batches)
        self.ends = np.cumsum([len(columns["score"]) for _, _, columns in self.batches])

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[row] for row in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        batch = int(np.searchsorted(self.ends, index, side="right"))
        row = index - (int(self.ends[batch - 1]) if batch else 0)
        region_id, component_ids, columns = self.batches[batch]
        item = {key: values[row].item() for key, values in columns.items()
                if key not in ("xy", "membership")}
        item.update(xy=columns["xy"][row].copy(), target_region_id=region_id,
                    target_component_ids=tuple(component_ids[columns["membership"][row]].tolist()))
        return item


def _readonly(value, memo=None):
    """Freeze shared diagnostic containers without copying their array storage."""
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
    elif isinstance(value, dict):
        frozen = {key: _readonly(item, memo) for key, item in value.items()}
        value_id = id(value)
        value = MappingProxyType(frozen)
        memo[value_id] = value
    elif isinstance(value, (tuple, list)):
        value = tuple(_readonly(item, memo) for item in value)
    elif isinstance(value, _CandidateRows):
        value.batches = _readonly(value.batches, memo)
        value.ends.setflags(write=False)
    return value


def _cache_bytes(value, seen=None):
    """Count retained containers and backing arrays once within a state bundle.

    Immutable target/region geometry belongs to the sampler, not this cache.
    Counting shared grids again in different states is intentionally conservative.
    """
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return size + (_cache_bytes(value.base, seen) if value.base is not None else 0)
    if isinstance(value, Mapping):
        # A mapping proxy also retains its underlying dict.
        if isinstance(value, MappingProxyType):
            size += sys.getsizeof(dict(value))
        return size + sum(_cache_bytes(k, seen) + _cache_bytes(v, seen)
                          for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return size + sum(_cache_bytes(item, seen) for item in value)
    if isinstance(value, _CandidateRows):
        return size + _cache_bytes(vars(value), seen)
    return size


class CompositePlanarSampler:
    """Keep the original scorer untouched; share its kernels and draw policy.

    The configured grid size is shared across the target's XY bounds. Regions
    retain their own floor policy and sampling pools. If the shared grid yields
    no candidates for a region, retry its original local grid so narrow or
    distant components do not silently disappear. All placed meshes contribute
    to the physical surface, including geometry outside the target footprint.
    """

    # Bound additional retained states; the working state exists independently.
    # These are implementation limits, not planning/sampling parameters.
    _cache_max_bytes = 64 * 1024 * 1024
    _cache_max_states = 16

    def __init__(self, owner):
        self.owner = owner
        self.policy = owner.floor_policy
        self.regions = self.policy.target_regions
        self._context = None
        self._maps = {}
        self._latest_maps = {}
        self._surfaces = {}
        self._states = OrderedDict()
        self._states_bytes = 0
        self._cursor = 0
        self._kernels = {}
        self._rotated = {}
        self._containment = {region.region_id: CachedAnchorContainment(region)
                             for region in self.regions.regions}

    def metadata(self, object_index, pose):
        return self.regions.pose_metadata(self.owner.inventory, object_index, pose)

    def _activate(self, context, state):
        if context == self._context:
            return
        saved = self._states.pop(context, None)
        if saved is not None:
            self._states_bytes -= saved[0]
        if self._context is not None:
            bundle = (self._maps, self._surfaces, self._layers)
            size = _cache_bytes((self._context, bundle))
            if size <= self._cache_max_bytes and self._cache_max_states:
                self._states[self._context] = (size, bundle)
                self._states_bytes += size
        while self._states and (self._states_bytes > self._cache_max_bytes or
                                len(self._states) > self._cache_max_states):
            _, (size, _) = self._states.popitem(last=False)
            self._states_bytes -= size
        self._context = context
        # Only maps requested during this visit may be used to score mutations.
        self._latest_maps = {}
        if saved is None:
            self._maps, self._surfaces = {}, {}
            self._layers = _readonly(self.policy.context_details(state)["regions"])
        else:
            self._maps, self._surfaces, self._layers = saved[1]

    def score_map(self, state, observation, object_index):
        owner = self.owner
        self._state = state
        quaternions = owner.quaternions(state, observation, object_index)
        # Inventory/target geometry is immutable and scoped to this sampler.
        # Rotations may depend on observation; their exact values key each map.
        context = (tuple(state.object_ids), tuple(state.placed_object_indices),
                   tuple((oid, np.asarray(pose).shape, np.asarray(pose).dtype.str,
                          np.asarray(pose).tobytes()) for oid, pose in sorted(state.object_poses.items())),
                   owner.config, repr(owner.settings), owner.relax_corner_seed,
                   owner.connectivity_enabled, owner.connectivity_length,
                   self.policy.bottom_tolerance, self.policy.fill_ratio,
                   self.policy.coverage_mode, self.policy.grid_size)
        self._activate(context, state)
        key = (object_index, tuple(tuple(q) for q in quaternions))
        if key in self._maps:
            self._latest_maps[object_index] = self._maps[key]
            return self._maps[key]
        batches, maps = [], {}
        shared = self._support_maps(state, object_index, quaternions,
                                    SamplingRegion(owner.inventory.target))
        for layer in self._layers:
            result = self._region_scores(layer, shared)
            result["grid_source"] = "shared"
            if (not result["candidates"] and
                    not np.array_equal(layer["region"].target.bounds,
                                       owner.inventory.target.bounds)):
                local = self._support_maps(state, object_index, quaternions, layer["region"])
                result = self._region_scores(layer, local)
                result["grid_source"] = "region_fallback"
            maps[layer["region_id"]] = result
            batches.extend(result["candidates"].batches)
        candidates = _CandidateRows(batches)
        low, span = 0., 1.
        if candidates:
            scores = np.concatenate([columns["score"] for _, _, columns in batches])
            low, span = scores.min(), max(float(np.ptp(scores)), 1e-12)
            for _, _, columns in batches:
                columns["normalized_score"] = (columns["score"] - low) / span
        for _, _, columns in batches:
            for values in columns.values():
                values.setflags(write=False)
        result = _readonly({"regions": maps, "candidates": candidates,
                  "normalization": {"score_min": float(low), "score_span": float(span)}})
        self._maps[key] = result
        self._latest_maps[object_index] = result
        return result

    def diagnostic_maps(self, state, observation, object_index):
        """Reconstruct score terms on each production region's chosen grid.

        Only the viewer calls this on its separate fixed-orientation sampler;
        ordinary search maps retain their compact column storage.
        """
        production = self.score_map(state, observation, object_index)
        quaternions = self.owner.quaternions(state, observation, object_index)
        shared = self._support_maps(state, object_index, quaternions,
                                    SamplingRegion(self.owner.inventory.target))
        maps = {}
        for layer in self._layers:
            region_id = layer["region_id"]
            source = production["regions"][region_id]["grid_source"]
            support = (self._support_maps(state, object_index, quaternions, layer["region"])
                       if source == "region_fallback" else shared)
            result = self._region_scores(layer, support, include_terms=True)
            result["grid_source"] = source
            maps[region_id] = result
        return maps

    def _support_maps(self, state, object_index, quaternions, region, *, cache_geometry=True):
        """Compute the physical surface/support once, before target masking."""
        owner = self.owner
        shape = tuple(max(2, value) for value in owner.config.grid_shape)
        _, _, xx, yy = region.grid(shape)
        x, y = xx[0], yy[:, 0]
        dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
        xy = np.c_[xx.ravel(), yy.ravel()]
        grid_key = (shape, tuple(x), tuple(y))
        if grid_key not in self._surfaces:
            # Gaps in the target still have physical ground. Masking them with
            # NaNs here would invalidate support kernels near target edges.
            surface = np.full(xx.size, self.regions.ground_z)
            for index in state.placed_object_indices:
                obj = owner.inventory[index]
                points, faces = obj.mesh_arrays()
                pose = state.object_poses.get(obj.object_id, owner.inventory.initial_pose(obj.object_id))
                world = transform_points(points, pose)
                _rasterize_triangles(surface, x, y, world[faces])
            self._surfaces[grid_key] = surface.reshape(shape)
        surface = self._surfaces[grid_key]
        source = owner.stones[object_index].points
        orientations = []
        for q in quaternions:
            rotation_key = (object_index, tuple(q))
            rotation = self._rotated.get(rotation_key)
            if rotation is None:
                rotated = Rotation.from_quat(q).apply(source)
                rotated.setflags(write=False)
                rotation = (rotated, _readonly(projected_footprint(rotated)))
                if cache_geometry:
                    self._rotated[rotation_key] = rotation
            rotated, footprint = rotation
            kernel_key = (object_index, dx, dy, tuple(q))
            kernel = self._kernels.get(kernel_key)
            if kernel is None:
                kernel = bottom_height_kernel(source, q, dx, dy)
                if cache_geometry:
                    self._kernels[kernel_key] = kernel
            support = support_likelihood_maps_by_kernel(
                surface, [kernel], owner.settings["contact_gap_tolerance"])
            # Raster kernels may miss the lowest vertex of an irregular stone.
            placement = np.maximum(support.placement_height[0].ravel(),
                                   self.regions.ground_z - rotated[:, 2].min())
            orientations.append((rotated, footprint, support.contact_height[0].ravel(),
                                 placement, support.support[0].ravel()))
        return x, y, xy, shape, orientations

    def _region_scores(self, layer, support_maps, *, include_terms=False):
        owner = self.owner
        region = layer["region"]
        region_id = region.region_id
        component_ids = np.array(region.target.component_ids)
        x, y, xy, shape, orientations = support_maps
        dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
        score_cfg = owner.settings.get("score", {})
        cfg = owner.settings["floor"]
        scale = owner.config.target_anchor_inset_scale
        scale = .6 if scale is None else scale
        margin = float(owner.settings.get("target_mask_margin", 0.))
        chunks = []
        containment = self._containment[region_id]
        score_grid = np.full(shape, np.nan)
        mask = np.zeros(shape, dtype=bool)
        terms = ({name: np.full(shape, np.nan) for name in (
            "height", "support_likelihood", "fill_area", "target_boundary",
            "planar_bias", "previous_action_distance", "placement_height")}
            if include_terms else None)
        previous_entry = None
        if (owner.connectivity_enabled and owner.settings.get("connectivity", "active_floor") in ("previous_action", "previous_action_soft")
                and self._state.placed_object_indices):
            index = self._state.placed_object_indices[-1]
            previous_entry = floor.floor_entry(owner.stones[index],
                self._state.object_poses[self._state.object_ids[index]], include_volumes=False)
        height_scale = max(float(region.target.bounds[1, 2] - self.regions.ground_z), 1e-12)
        bias_xy = floor.score_planar_bias_xy(score_cfg)
        axis = str(score_cfg.get("planar_bias_axis", score_cfg.get("excavator_distance_axis", "xy"))).lower()
        extent = np.ptp(owner.inventory.target.bounds[:, :2], axis=0)
        footprint_scale = float(cfg.get("footprint_scale", .9))
        xs, ys = layer["xx"][0], layer["yy"][:, 0]
        # Layer masks are shared by every orientation.
        unfilled_table = floor.cell_count_table(layer["valid"] & ~layer["occupied"])
        contact_table = floor.cell_count_table(layer["occupied_neighbor"])
        for orientation, (rotated, footprint, height, placement, likelihood) in enumerate(orientations):
            finite = np.isfinite(height) & np.isfinite(placement) & np.isfinite(likelihood)
            indices = np.flatnonzero(finite)
            membership = containment(xy[indices], rotated, placement[indices],
                                     inset_scale=scale, margin=margin)
            valid = membership.any(axis=1)
            indices, membership = indices[valid], membership[valid]
            # A scaled footprint may overhang, but its anchor cannot be in a gap.
            valid = region.target.contains(np.column_stack((xy[indices], height[indices])))
            indices, membership = indices[valid], membership[valid]
            if not len(indices):
                continue
            target_valid = np.zeros(shape, dtype=bool)
            target_valid.ravel()[indices] = True
            distance = distance_transform_edt(np.pad(target_valid, 1), sampling=(dy, dx))[1:-1, 1:-1]
            bounds = footprint_scale * np.stack((rotated[:, :2].min(0), rotated[:, :2].max(0)))
            anchors = xy[indices]
            rectangles = (
                np.searchsorted(xs, anchors[:, 0] + bounds[0, 0]),
                np.searchsorted(xs, anchors[:, 0] + bounds[1, 0], side="right"),
                np.searchsorted(ys, anchors[:, 1] + bounds[0, 1]),
                np.searchsorted(ys, anchors[:, 1] + bounds[1, 1], side="right"))
            unfilled = floor.table_cell_counts(unfilled_table, rectangles)
            contacts = floor.table_cell_counts(contact_table, rectangles)
            connected = (contacts > 0) if layer["occupied"].any() else np.ones(len(indices), dtype=bool)
            if not owner.connectivity_enabled:
                connected[:] = True
            distance_penalty = np.zeros(len(indices))
            if owner.settings.get("connectivity") == "previous_action_soft":
                connected[:] = True
                if previous_entry is not None:
                    distance_penalty = bounded_distance_penalty(
                        previous_entry["xy_polygon"], footprint, anchors,
                        float(owner.settings["contact_gap_tolerance"]), owner.connectivity_length)
            elif previous_entry is not None:
                connected = footprints_within_gap(previous_entry["xy_polygon"],
                    footprint, anchors,
                    float(owner.settings["contact_gap_tolerance"]) + 1e-9)
            keep = connected
            indices, membership, unfilled = indices[keep], membership[keep], unfilled[keep]
            if not len(indices):
                continue
            boundary = 1. - distance.ravel()[indices] / max(float(distance.max()), 1e-12)
            relative_height = np.maximum(height[indices] - self.regions.ground_z, 0.)
            delta = np.abs(xy[indices] - bias_xy)
            bias = (delta[:, 0] / max(extent[0], 1e-12) if axis in ("x", "0") else
                    delta[:, 1] / max(extent[1], 1e-12) if axis in ("y", "1") else
                    np.linalg.norm(delta, axis=1) / max(float(np.linalg.norm(extent)), 1e-12))
            scores = (float(score_cfg.get("height", -1.5)) * relative_height / height_scale
                      + float(score_cfg.get("support_likelihood", 2.)) * likelihood[indices]
                      + float(score_cfg.get("fill_area", 0.)) * unfilled / max(int(unfilled.max()), 1)
                      + float(score_cfg.get("target_boundary", 0.)) * boundary
                      + float(score_cfg.get("planar_bias", score_cfg.get("excavator_distance", .25))) * np.minimum(bias, 1.))
            scores -= float(score_cfg.get("previous_action_distance", 1.)) * distance_penalty[keep]
            if terms is not None:
                wins = ~np.isfinite(score_grid.ravel()[indices]) | (scores > score_grid.ravel()[indices])
                values = dict(height=relative_height / height_scale,
                    support_likelihood=likelihood[indices],
                    fill_area=unfilled / max(int(unfilled.max()), 1), target_boundary=boundary,
                    planar_bias=np.minimum(bias, 1.),
                    previous_action_distance=distance_penalty[keep], placement_height=placement[indices])
                for name, value in values.items():
                    terms[name].ravel()[indices[wins]] = value[wins]
            full = containment(xy[indices], rotated, placement[indices],
                               inset_scale=max(scale, 1.), margin=margin).any(axis=1)
            chunks.append(dict(xy=xy[indices], score=scores,
                height=height[indices], placement_height=placement[indices],
                orientation_index=np.full(len(indices), orientation, dtype=int),
                membership=membership,
                lower_floor=height[indices] <= layer["support_z"] + layer["bottom_tol"],
                full_footprint=full, seed=np.zeros(len(indices), dtype=bool)))
            score_grid.ravel()[indices] = np.fmax(score_grid.ravel()[indices], scores)
            mask.ravel()[indices] = True
        columns = ({key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
                   if chunks else {"score": np.empty(0)})
        if chunks and layer["empty"] and not owner.relax_corner_seed:
            eligible = np.flatnonzero(columns["full_footprint"])
            if len(eligible):
                centers = columns["xy"][eligible]
                # Feasible extrema, not corners of a bounding rectangle or notch.
                for direction in ((-1., -1.), (1., -1.), (1., 1.), (-1., 1.)):
                    columns["seed"][eligible[int(np.argmax(centers @ direction))]] = True
        result = {"x_coords": x, "y_coords": y, "scores": score_grid,
                "candidate_mask": mask, "columns": columns,
                "candidates": _CandidateRows([(region_id, component_ids, columns)])}
        if terms is not None:
            result["terms"] = terms
            result["weights"] = dict(
                height=float(score_cfg.get("height", -1.5)),
                support_likelihood=float(score_cfg.get("support_likelihood", 2.)),
                fill_area=float(score_cfg.get("fill_area", 0.)),
                target_boundary=float(score_cfg.get("target_boundary", 0.)),
                planar_bias=float(score_cfg.get("planar_bias", score_cfg.get("excavator_distance", .25))),
                previous_action_distance=-float(score_cfg.get("previous_action_distance", 1.))
                    if owner.settings.get("connectivity") == "previous_action_soft" else 0.)
        return result

    def sample(self, state, observation, object_index, count, sampling_priority):
        if count <= 0:
            return ()
        scores = self.score_map(state, observation, object_index)
        pools = {key: np.arange(len(value["candidates"]))
                 for key, value in scores["regions"].items() if value["candidates"]}
        if not pools:
            return ()
        keys = sorted(pools)
        cursor = self._cursor % len(keys)
        # Allocate complete per-region budgets first. Drawing k=1 repeatedly
        # rounds every positive elite fraction up to an entirely greedy draw.
        schedule, budgets = [], dict.fromkeys(keys, 0)
        while pools and len(schedule) < count:
            key = keys[cursor % len(keys)]
            cursor += 1
            if key not in pools:
                continue
            schedule.append(key)
            budgets[key] += 1
            if budgets[key] == len(pools[key]):
                del pools[key]
        self._cursor = cursor % len(keys)
        chosen_by_region = {}
        for key, budget in budgets.items():
            region_map = scores["regions"][key]
            columns = region_map["columns"]
            groups = [columns["seed"], ~columns["seed"]]
            if sampling_priority == "lower_floor_first":
                groups = [group & lower for group in groups
                          for lower in (columns["lower_floor"], ~columns["lower_floor"])]
            chosen = []
            for group in groups:
                eligible = np.flatnonzero(group)
                remaining = min(budget - len(chosen), len(eligible))
                if remaining <= 0:
                    continue
                indices = floor.sample_scored_indices(
                    self.owner.settings, columns["score"][eligible], remaining, True, rng=self.owner.rng)
                chosen.extend(eligible[indices].tolist())
            chosen_by_region[key] = iter(chosen)
        # Keep the region-balanced output order used by downstream budgets.
        selected = [scores["regions"][key]["candidates"][next(chosen_by_region[key])]
                    for key in schedule]
        quaternions = self.owner.quaternions(state, observation, object_index)
        return tuple(PlanarPlacement(np.r_[item["xy"], 0., quaternions[item["orientation_index"]]],
            score=item["score"], diagnostics={
                "source": "composite_scored_floor", "sampling_policy": "corner_seed" if item["seed"] else "scored",
                "orientation_index": item["orientation_index"], "target_region_id": item["target_region_id"],
                "target_region_ids": (item["target_region_id"],), "target_component_ids": item["target_component_ids"],
                "height_map_score": item["score"], "height_map_score_normalized": item["normalized_score"],
            }) for item in selected)

    def score_at_pose(self, observation, pose, object_index):
        metadata = self.metadata(object_index, pose)
        scores = self._latest_maps.get(object_index)
        if scores is None or not scores["candidates"]:
            return 0., 0.
        # Evaluate the actual rotation, including off-bank CEM yaw mutations.
        # Transient rotations must not grow the persistent geometry caches.
        shared = None
        nearest = None
        for layer in self._layers:
            region_id = layer["region_id"]
            if region_id not in metadata["target_region_ids"]:
                continue
            base_map = scores["regions"][region_id]
            if base_map["grid_source"] == "region_fallback":
                support = self._support_maps(self._state, object_index, [pose[3:]], layer["region"],
                                             cache_geometry=False)
            else:
                if shared is None:
                    shared = self._support_maps(self._state, object_index, [pose[3:]],
                        SamplingRegion(self.owner.inventory.target), cache_geometry=False)
                support = shared
            region_map = self._region_scores(layer, support)
            xs, ys = region_map["x_coords"], region_map["y_coords"]
            col, row = int(np.argmin(abs(xs - pose[0]))), int(np.argmin(abs(ys - pose[1])))
            if not region_map["candidate_mask"][row, col]:
                continue
            columns = region_map["columns"]
            cell = np.array([xs[col], ys[row]])
            index = int(np.flatnonzero(np.all(columns["xy"] == cell, axis=1))[0])
            rotated = support[-1][0][0]
            scale = self.owner.config.target_anchor_inset_scale
            if not self._containment[region_id](
                    [pose[:2]], rotated, columns["placement_height"][index],
                    inset_scale=.6 if scale is None else scale,
                    margin=float(self.owner.settings.get("target_mask_margin", 0.))).any():
                continue
            distance = float(np.linalg.norm(cell - pose[:2]))
            if nearest is None or distance < nearest[0]:
                nearest = (distance, float(columns["score"][index]))
        if nearest is None:
            return 0., 0.
        raw = nearest[1]
        reference = scores["normalization"]
        return raw, float(np.clip((raw - reference["score_min"]) / reference["score_span"], 0., 1.))
