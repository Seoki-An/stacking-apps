"""Region-balanced proposals using actual convex-component containment masks."""

from types import SimpleNamespace

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation

from stacking_common.domain.geometry import transform_points
from stacking_common.visualization.heightmap import _rasterize_triangles

from . import _scored_floor as floor
from ._support_maps import bottom_height_kernel, support_likelihood_maps_by_kernel
from .target_regions import anchor_components
from .types import PlanarPlacement


class CompositePlanarSampler:
    """Keep the original scorer untouched; share its kernels and draw policy.

    Each connected region gets a local grid of the configured size. Empty space
    between distant targets therefore neither consumes resolution nor dilutes
    floor coverage. All placed meshes, not only the region's stones, contribute
    to the local top surface and remain in native collision/force scenes.
    """

    def __init__(self, owner):
        self.owner = owner
        self.policy = owner.floor_policy
        self.regions = self.policy.target_regions
        self._context = None
        self._maps = {}
        self._latest_maps = {}
        self._surfaces = {}
        self._cursor = 0
        self._kernels = {}

    def metadata(self, object_index, pose):
        return self.regions.pose_metadata(self.owner.inventory, object_index, pose)

    def score_map(self, state, observation, object_index):
        owner = self.owner
        quaternions = owner.quaternions(state, observation, object_index)
        context = (tuple(state.placed_object_indices),
                   tuple((oid, np.asarray(pose).tobytes()) for oid, pose in sorted(state.object_poses.items())),
                   owner.config, repr(owner.settings), owner.relax_floor_fill, owner.relax_corner_seed)
        if context != self._context:
            self._context = context
            self._maps.clear()
            self._latest_maps.clear()
            self._surfaces.clear()
            self._layers = self.policy.context_details(state)["regions"]
        key = (object_index, tuple(tuple(q) for q in quaternions))
        if key in self._maps:
            self._latest_maps[object_index] = self._maps[key]
            return self._maps[key]
        candidates, maps = [], {}
        for layer in self._layers:
            result = self._region_scores(state, object_index, quaternions, layer)
            maps[layer["region_id"]] = result
            candidates.extend(result["candidates"])
        if candidates:
            scores = np.array([item["score"] for item in candidates])
            low, span = scores.min(), max(float(np.ptp(scores)), 1e-12)
            for item in candidates:
                item["normalized_score"] = float((item["score"] - low) / span)
        result = {"regions": maps, "candidates": candidates}
        self._maps[key] = result
        self._latest_maps[object_index] = result
        return result

    def _region_scores(self, state, object_index, quaternions, layer):
        owner = self.owner
        region = layer["region"]
        shape = tuple(max(2, value) for value in owner.config.grid_shape)
        _, _, xx, yy = region.grid(shape)
        x, y = xx[0], yy[:, 0]
        dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
        xy = np.c_[xx.ravel(), yy.ravel()]
        if region.region_id not in self._surfaces:
            surface = np.full(xx.size, self.regions.ground_z)
            for index in state.placed_object_indices:
                obj = owner.inventory[index]
                points, faces = obj.mesh_arrays()
                pose = state.object_poses.get(obj.object_id, owner.inventory.initial_pose(obj.object_id))
                world = transform_points(points, pose)
                _rasterize_triangles(surface, x, y, world[faces])
            self._surfaces[region.region_id] = surface.reshape(shape)
        surface = self._surfaces[region.region_id]
        score_cfg = owner.settings.get("score", {})
        cfg = owner.settings["floor"]
        scale = owner.config.target_anchor_inset_scale
        scale = .6 if scale is None else scale
        margin = float(owner.settings.get("target_mask_margin", 0.))
        source = owner.stones[object_index].points
        candidates = []
        score_grid = np.full(shape, np.nan)
        mask = np.zeros(shape, dtype=bool)
        for orientation, q in enumerate(quaternions):
            rotated = Rotation.from_quat(q).apply(source)
            kernel_key = (object_index, region.region_id, shape, tuple(q))
            if kernel_key not in self._kernels:
                self._kernels[kernel_key] = bottom_height_kernel(source, q, dx, dy)
            kernel = self._kernels[kernel_key]
            support = support_likelihood_maps_by_kernel(surface, [kernel], owner.settings["contact_gap_tolerance"])
            height = support.contact_height[0].ravel()
            # The raster kernel can miss the lowest vertex of an irregular or
            # tilted stone. Do not let that discretization error put its mesh
            # below the physical ground and reject every containment anchor.
            placement = np.maximum(support.placement_height[0].ravel(),
                                   self.regions.ground_z - rotated[:, 2].min())
            likelihood = support.support[0].ravel()
            finite = np.isfinite(height) & np.isfinite(placement) & np.isfinite(likelihood)
            indices = np.flatnonzero(finite)
            membership = anchor_components(region, xy[indices], rotated, placement[indices],
                                           inset_scale=scale, margin=margin)
            valid = membership.any(axis=1)
            indices, membership = indices[valid], membership[valid]
            # A scaled footprint may overhang, but its anchor cannot be in a gap.
            valid = region.target.contains(np.c_[xy[indices], height[indices]])
            indices, membership = indices[valid], membership[valid]
            if not len(indices):
                continue
            target_valid = np.zeros(shape, dtype=bool)
            target_valid.ravel()[indices] = True
            distance = distance_transform_edt(np.pad(target_valid, 1), sampling=(dy, dx))[1:-1, 1:-1]
            bounds = float(cfg.get("footprint_scale", .9)) * np.stack(
                (rotated[:, :2].min(0), rotated[:, :2].max(0)))
            xs, ys = layer["xx"][0], layer["yy"][:, 0]
            anchors = xy[indices]
            rectangles = (
                np.searchsorted(xs, anchors[:, 0] + bounds[0, 0]),
                np.searchsorted(xs, anchors[:, 0] + bounds[1, 0], side="right"),
                np.searchsorted(ys, anchors[:, 1] + bounds[0, 1]),
                np.searchsorted(ys, anchors[:, 1] + bounds[1, 1], side="right"))
            unfilled = floor.rectangle_cell_counts(layer["valid"] & ~layer["occupied"], rectangles)
            contacts = floor.rectangle_cell_counts(layer["occupied_neighbor"], rectangles)
            connected = (contacts > 0) if layer["occupied"].any() else np.ones(len(indices), dtype=bool)
            keep = connected & ((unfilled > 0) | owner.relax_floor_fill)
            indices, membership, unfilled = indices[keep], membership[keep], unfilled[keep]
            if not len(indices):
                continue
            boundary = 1. - distance.ravel()[indices] / max(float(distance.max()), 1e-12)
            relative_height = np.maximum(height[indices] - self.regions.ground_z, 0.)
            height_scale = max(float(region.target.bounds[1, 2] - self.regions.ground_z), 1e-12)
            bias_xy = floor.score_planar_bias_xy(score_cfg)
            axis = str(score_cfg.get("planar_bias_axis", score_cfg.get("excavator_distance_axis", "xy"))).lower()
            delta = np.abs(xy[indices] - bias_xy)
            extent = np.ptp(owner.inventory.target.bounds[:, :2], axis=0)
            bias = (delta[:, 0] / max(extent[0], 1e-12) if axis in ("x", "0") else
                    delta[:, 1] / max(extent[1], 1e-12) if axis in ("y", "1") else
                    np.linalg.norm(delta, axis=1) / max(float(np.linalg.norm(extent)), 1e-12))
            scores = (float(score_cfg.get("height", -1.5)) * relative_height / height_scale
                      + float(score_cfg.get("support_likelihood", 2.)) * likelihood[indices]
                      + float(score_cfg.get("fill_area", 0.)) * unfilled / max(int(unfilled.max()), 1)
                      + float(score_cfg.get("target_boundary", 0.)) * boundary
                      + float(score_cfg.get("planar_bias", score_cfg.get("excavator_distance", .25))) * np.minimum(bias, 1.))
            full = anchor_components(region, xy[indices], rotated, placement[indices],
                                     inset_scale=max(scale, 1.), margin=margin).any(axis=1)
            for row, index in enumerate(indices):
                candidates.append(dict(xy=xy[index].copy(), score=float(scores[row]),
                    height=float(height[index]), placement_height=float(placement[index]),
                    orientation_index=orientation, target_region_id=region.region_id,
                    target_component_ids=tuple(np.array(region.target.component_ids)[membership[row]].tolist()),
                    lower_floor=bool(height[index] <= layer["support_z"] + layer["bottom_tol"]),
                    full_footprint=bool(full[row]), seed=False))
                previous = score_grid.ravel()[index]
                if not np.isfinite(previous) or scores[row] > previous:
                    score_grid.ravel()[index] = scores[row]
                mask.ravel()[index] = True
        if layer["empty"] and not owner.relax_corner_seed:
            eligible = [item for item in candidates if item["full_footprint"]]
            if eligible:
                centers = np.array([item["xy"] for item in eligible])
                # Feasible extrema, not corners of a bounding rectangle or notch.
                for direction in ((-1., -1.), (1., -1.), (1., 1.), (-1., 1.)):
                    eligible[int(np.argmax(centers @ direction))]["seed"] = True
        return {"x_coords": x, "y_coords": y, "scores": score_grid,
                "candidate_mask": mask, "candidates": candidates}

    def sample(self, state, observation, object_index, count, sampling_priority):
        if count <= 0:
            return ()
        scores = self.score_map(state, observation, object_index)
        pools = {key: list(value["candidates"]) for key, value in scores["regions"].items() if value["candidates"]}
        if not pools:
            return ()
        keys = sorted(pools)
        cursor = self._cursor % len(keys)
        selected = []
        view = SimpleNamespace(cfg=SimpleNamespace(action=SimpleNamespace(planar=self.owner.settings)))
        while pools and len(selected) < count:
            key = keys[cursor % len(keys)]
            cursor += 1
            if key not in pools:
                continue
            pool = pools[key]
            seeds = [item for item in pool if item["seed"]]
            eligible = seeds or pool
            if sampling_priority == "lower_floor_first":
                lower = [item for item in eligible if item["lower_floor"]]
                eligible = lower or eligible
            item = floor.sample_scored_candidates(view, eligible, 1, True, rng=self.owner.rng)[0]
            selected.append(item)
            pools[key] = [other for other in pool if other is not item]
            if not pools[key]:
                del pools[key]
        self._cursor = cursor % len(keys)
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
        scores = self._latest_maps.get(object_index, {"candidates": ()})
        candidates = [item for item in scores["candidates"]
                      if item["target_region_id"] in metadata["target_region_ids"]]
        if not candidates:
            return 0., 0.
        item = min(candidates, key=lambda item: float(np.linalg.norm(item["xy"] - pose[:2])))
        region = next(region for region in self.regions.regions if region.region_id == item["target_region_id"])
        rotated = Rotation.from_quat(pose[3:]).apply(self.owner.stones[object_index].points)
        scale = self.owner.config.target_anchor_inset_scale
        if not anchor_components(region, [pose[:2]], rotated, item["placement_height"],
                                 inset_scale=.6 if scale is None else scale,
                                 margin=float(self.owner.settings.get("target_mask_margin", 0.))).any():
            return 0., 0.
        return item["score"], item["normalized_score"]
