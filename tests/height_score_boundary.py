"""Display-only continuation of the placement prior beyond admitted anchors."""

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from stacking_common.actions.config import PlanarSamplingConfig
from stacking_common.actions.orientation import MappingOrientationProvider
from stacking_common.actions.scored import ScoredPlanarPoseSampler
from stacking_common.actions import _scored_floor as floor
from stacking_common.actions._support_maps import (
    bottom_height_kernel, build_state_map_bundle, support_likelihood_maps_by_kernel,
)


def extended_score_fields(view, boundary_display=None):
    """Continue evaluated terms to the target edge, preserving source arrays.

    Nearest evaluated anchors supply missing non-boundary terms. The boundary
    term uses its explicit clamped continuation when available. Recombine the
    extended terms with the original weights for a consistent display total.
    These are display extrapolations, not evaluations or admitted placements.
    """
    from scipy.ndimage import distance_transform_edt

    names = tuple(view.weights)
    source = np.isfinite(view.fields["score"])
    for name in names:
        source &= np.isfinite(view.fields[name])
    if not source.any() or not names:
        return {name: np.full(view.valid.shape, np.nan) for name in (*names, "score")}
    dx, dy = abs(float(np.diff(view.x).mean())), abs(float(np.diff(view.y).mean()))
    nearest = distance_transform_edt(~source, sampling=(dy, dx),
                                    return_distances=False, return_indices=True)
    xy = np.stack(np.meshgrid(view.x, view.y), axis=-1)
    lower, upper = view.target_bounds
    # Include cells intersecting the target; the plot clips them to the exact
    # rectangle so coarse raster centers cannot leave a black strip at its edge.
    half_cell = np.array([dx, dy]) / 2
    in_target = np.all((xy + half_cell >= lower) & (xy - half_cell <= upper), axis=-1)
    extended = {}
    for name in names:
        original = view.fields[name]
        values = original[tuple(nearest)].copy()
        known = np.isfinite(original)
        if name == "target_boundary" and boundary_display is not None:
            values[:] = np.where(np.isfinite(boundary_display["score"]), boundary_display["score"], 1.)
        values[known] = original[known]
        values[~in_target] = np.nan
        extended[name] = values
    extended["score"] = sum(extended[name] * view.weights[name] for name in names)
    known_total = np.isfinite(view.fields["score"]) & in_target
    extended["score"][known_total] = view.fields["score"][known_total]
    return extended


def boundary_display_map(inventory, state, object_index, pose, view, planar_config, settings, floor_settings=None):
    """Extend the boundary trend over the target base without admitting seeds.

    Retain all returned production values exactly. At unscored anchors, use
    the same footprint/taper bounds and contact-height calculation, projecting
    points outside the anchor rectangle onto its nearest boundary. Those
    extensions saturate at one and are diagnostics, not planner scores.
    """
    if "floor" not in settings:
        legacy = settings.get("floor_fill", {})
        settings = {**settings, "floor": {**floor_settings,
            "footprint_scale": legacy.get("u_shape_footprint_scale", .9)}}
    config = PlanarSamplingConfig(view.valid.shape, (0.,),
        target_anchor_inset_scale=planar_config.get("target_anchor_inset_scale"))
    provider = MappingOrientationProvider({i: pose[3:] for i in inventory.object_ids})
    sampler = ScoredPlanarPoseSampler(inventory, config, provider, settings)
    observation = SimpleNamespace(scene_height_map=view.scene_height,
        height_map_xlim=view.scene_xlim, height_map_ylim=view.scene_ylim)
    native_view = sampler.view(state, observation)
    xy = np.stack(np.meshgrid(view.x, view.y), axis=-1)
    state_lower, state_upper = floor.target_anchor_bounds(native_view, None)
    state_valid = np.all((xy >= state_lower) & (xy <= state_upper), axis=-1)
    bundle = build_state_map_bundle(view.scene_height, view.valid.shape,
        inventory.target.origin[2], state_valid, floor.HEIGHT_SCORE_SMOOTH_KERNEL_SIZE)
    points = sampler.stones[object_index].points
    rotated = Rotation.from_quat(pose[3:]).apply(points)
    offsets = np.array([rotated[:, :2].min(axis=0), rotated[:, :2].max(axis=0)])
    dx, dy = float(np.diff(view.x).mean()), float(np.diff(view.y).mean())
    kernel = bottom_height_kernel(points, pose[3:], dx, dy)
    support = support_likelihood_maps_by_kernel(bundle.surface_height, [kernel],
                                                settings["contact_gap_tolerance"])
    heights = support.contact_height[0].copy()
    for row, column in np.argwhere(~np.isfinite(heights)):
        heights[row, column] = floor.mean_top_height_at_anchor(native_view, bundle.smoothed_height,
            xy, xy[row, column], object_index,
            _bounds=settings["floor"].get("footprint_scale", .9) * offsets)
    static = floor.target_anchor_static(native_view, object_index, xy_bounds=offsets)
    lower, upper = floor.target_anchor_bounds(native_view, object_index, z=heights, _static=static)
    original = view.fields["target_boundary"]
    known = np.isfinite(original)
    recomputed = floor.target_boundary_proximity(xy, lower, upper)
    if not np.allclose(recomputed[known], original[known], rtol=1e-9, atol=1e-9):
        raise ValueError("boundary reconstruction does not match recorded diagnostic scores")
    projected = np.minimum(np.maximum(xy, lower), upper)
    extended = floor.target_boundary_proximity(projected, lower, upper)
    base_lower, base_upper = view.target_bounds
    in_target = np.all((xy >= base_lower) & (xy <= base_upper), axis=-1)
    extended[~in_target | np.any(upper <= lower, axis=-1)] = np.nan
    extended[known] = original[known]
    base_bounds = np.array(floor.target_anchor_bounds(native_view, object_index, _static=static))
    return dict(score=extended, lower=lower, upper=upper, contact_height=heights,
                footprint_offsets=offsets, base_anchor_bounds=base_bounds)
