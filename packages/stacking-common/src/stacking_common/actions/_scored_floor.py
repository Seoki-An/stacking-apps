"""Pure array score policy migrated from stacking-planner floor_fill.py.

Only the dependencies of scored sampling are retained; platform inputs arrive
through the canonical inventory adapter in scored.py.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from scipy.spatial import ConvexHull, QhullError

from scipy.spatial.transform import Rotation

from ._support_maps import (
    BottomHeightKernel,
    StateMapBundle,
    build_state_map_bundle,
    rectangular_bottom_kernel,
    support_likelihood_maps_by_kernel,
    transformed_stone_volumes,
    volume_slice_mask,
)

HEIGHT_SCORE_SMOOTH_KERNEL_SIZE = 5

def active_floor_context(inventory, state, min_count: Optional[int] = None) -> Optional[dict]:
    if min_count is not None and len(state.stone_seq) >= min_count:
        return None

    floor_cfg = inventory.cfg.action.planar.get("floor", {})
    support = inventory.support_settings
    support_z = support.ground_z
    bottom_tol = float(
        floor_cfg.get("bottom_tolerance", support.z_tolerance)
    )
    fill_ratio = float(floor_cfg.get("fill_ratio", 0.65))
    grid_size = max(int(floor_cfg.get("grid_size", 32)), 4)

    entries = placed_floor_entries(inventory, state)
    used: set[int] = set()
    # A completed layer consumes stones; an empty/incomplete layer returns.
    while True:
        layer_items = [
            (idx, entry)
            for idx, entry in enumerate(entries)
            if idx not in used and entry["bottom"] <= support_z + bottom_tol
        ]
        layer = [entry for _, entry in layer_items]
        layer_top = max((entry["top"] for entry in layer), default=support_z)
        context = occupancy_context(inventory, layer, support_z, bottom_tol, grid_size)
        if context is None:
            return None
        if not layer_items or context["occupancy"] < fill_ratio:
            return context

        for idx, _ in layer_items:
            used.add(idx)
        support_z = layer_top
        if len(used) >= len(entries):
            return occupancy_context(inventory, [], support_z, bottom_tol, grid_size)


def placed_floor_entries(inventory, state) -> list[dict]:
    entries = []
    for idx in getattr(state, "stone_seq", []) or []:
        resolved = state_stone(inventory, state, idx)
        if resolved is None:
            continue
        stone, stone_id = resolved
        pose = np.asarray(state.stone_poses.get(stone_id, stone.pose), dtype=float)
        if pose.shape[0] < 7 or not np.all(np.isfinite(pose[:7])):
            continue
        entries.append(floor_entry(stone, pose))
    entries.sort(key=lambda item: (item["bottom"], item["top"]))
    return entries

def state_stone(inventory, state, entry):
    idx = int(entry)
    id_to_stone = {int(stone.id): stone for stone in inventory.stones}
    stone_set = getattr(state, "stone_set", None)
    if stone_set is not None and 0 <= idx < len(stone_set):
        stone_id = int(stone_set[idx])
        stone = id_to_stone.get(stone_id)
        if stone is not None:
            return stone, stone_id
        return None
    if 0 <= idx < len(inventory.stones):
        stone = inventory.stones[idx]
        return stone, int(stone.id)
    stone = id_to_stone.get(idx)
    if stone is not None:
        return stone, idx
    return None

def floor_entry(stone, pose: np.ndarray, include_volumes: bool = True) -> dict:
    vertices = stone_world_vertices(stone, pose)
    xy = vertices[:, :2]
    xy_lower = np.min(xy, axis=0)
    xy_upper = np.max(xy, axis=0)
    z = vertices[:, 2]
    entry = {
        "xy": np.asarray(pose[:2], dtype=float).copy(),
        "half": np.maximum(0.5 * (xy_upper - xy_lower), 1e-6),
        "extent": np.maximum(xy_upper - xy_lower, 1e-6),
        "bottom": float(np.min(z)),
        "top": float(np.max(z)),
        "xy_aabb": (xy_lower, xy_upper),
        "xy_polygon": convex_xy_polygon(xy),
    }
    if include_volumes:
        entry["volumes"] = transformed_stone_volumes(stone, pose)
    return entry

def stone_world_vertices(stone, pose: np.ndarray) -> np.ndarray:
    verts_local, _ = stone.get_global_hull_mesh_array()
    rot = Rotation.from_quat(pose[3:]).as_matrix()
    return verts_local @ rot.T + pose[:3]

def convex_xy_polygon(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 3:
        return xy
    try:
        hull = ConvexHull(xy)
    except QhullError:
        return xy
    return xy[hull.vertices]

def occupancy_context(
    inventory,
    entries: list[dict],
    support_z: float,
    bottom_tol: float,
    grid_size: int,
) -> Optional[dict]:
    grid = occupancy_grid(inventory, grid_size)
    if grid is None:
        return None
    lower, upper, xx, yy = grid
    occupied = entries_occupancy_mask(
        entries,
        lower,
        upper,
        xx,
        yy,
        support_z,
        support_z + bottom_tol,
    )
    return {
        "support_z": float(support_z),
        "bottom_tol": float(bottom_tol),
        "lower": lower,
        "upper": upper,
        "xx": xx,
        "yy": yy,
        "occupied": occupied,
        "occupied_neighbor": neighbor_mask(occupied),
        "occupancy": float(np.mean(occupied)),
    }

def occupancy_grid(inventory, grid_size: int):
    wall = inventory.target_wall
    origin = np.asarray(wall.origin[:2], dtype=float)
    half = np.asarray([wall.width, wall.length], dtype=float) / 2.0
    lower = origin - half
    upper = origin + half
    if np.any(upper <= lower):
        return None

    xs = np.linspace(lower[0], upper[0], max(int(grid_size), 4))
    ys = np.linspace(lower[1], upper[1], max(int(grid_size), 4))
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return lower, upper, xx, yy

def entries_occupancy_mask(
    entries: list[dict],
    lower: np.ndarray,
    upper: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    z_lower: float,
    z_upper: float,
) -> np.ndarray:
    occupied = np.zeros(xx.shape, dtype=bool)
    for entry in entries:
        occupied |= floor_slice_mask(
            entry,
            lower,
            upper,
            xx,
            yy,
            z_lower,
            z_upper,
        )
    return occupied

def floor_slice_mask(
    entry: dict,
    lower: np.ndarray,
    upper: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    z_lower: float,
    z_upper: float,
) -> np.ndarray:
    in_target = (
        (xx >= float(lower[0]))
        & (xx <= float(upper[0]))
        & (yy >= float(lower[1]))
        & (yy <= float(upper[1]))
    )
    volumes = tuple(entry.get("volumes", ()))
    if not volumes:
        return in_target & footprint_mask(entry, lower, upper, xx, yy)
    return in_target & volume_slice_mask(
        volumes,
        xx,
        yy,
        z_lower,
        z_upper,
    )

def footprint_mask(
    entry: dict,
    lower: np.ndarray,
    upper: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
) -> np.ndarray:
    aabb_lower, aabb_upper = entry["xy_aabb"]
    aabb_lower = np.maximum(np.asarray(aabb_lower, dtype=float), lower)
    aabb_upper = np.minimum(np.asarray(aabb_upper, dtype=float), upper)
    if np.any(aabb_upper < aabb_lower):
        return np.zeros(xx.shape, dtype=bool)

    in_aabb = (
        (xx >= aabb_lower[0])
        & (xx <= aabb_upper[0])
        & (yy >= aabb_lower[1])
        & (yy <= aabb_upper[1])
    )
    polygon = np.asarray(entry.get("xy_polygon", []), dtype=float)
    if len(polygon) < 3:
        return in_aabb
    return in_aabb & points_in_convex_polygon(xx, yy, polygon)

def points_in_convex_polygon(xx: np.ndarray, yy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    inside = np.ones(xx.shape, dtype=bool)
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        edge = b - a
        cross = edge[0] * (yy - a[1]) - edge[1] * (xx - a[0])
        inside &= cross >= -1e-9
    return inside

def neighbor_mask(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    neighbors = np.zeros(mask.shape, dtype=bool)
    for dx in range(3):
        for dy in range(3):
            if dx == 1 and dy == 1:
                continue
            neighbors |= padded[dx : dx + mask.shape[0], dy : dy + mask.shape[1]]
    return neighbors

def score_xy_debug_map(
    inventory,
    state,
    stone_idx: Optional[int],
    scene_height_map: Optional[np.ndarray] = None,
    active_floor: Optional[dict] = None,
    support_kernels: Optional[list[BottomHeightKernel]] = None,
    state_map_bundle: Optional[StateMapBundle] = None,
    relax_floor_fill: bool = False,
    footprint_bounds: Optional[np.ndarray] = None,
) -> dict:
    if scene_height_map is None:
        scene_height_map = inventory.get_height_map(state)
    height_map = np.asarray(scene_height_map)
    shape = score_grid_shape(inventory, height_map)
    xy_grid = xy_grid_for_shape(inventory, shape)
    support = inventory.support_settings
    if state_map_bundle is None:
        state_lower, state_upper = target_anchor_bounds(inventory, None)
        state_valid = np.all(
            (xy_grid >= state_lower) & (xy_grid <= state_upper),
            axis=-1,
        )
        state_map_bundle = build_state_map_bundle(
            height_map,
            shape,
            support.ground_z,
            state_valid,
            smooth_kernel_size=HEIGHT_SCORE_SMOOTH_KERNEL_SIZE,
        )
    bundle = state_map_bundle
    heights = bundle.smoothed_height
    grid_dx = float(np.mean(np.diff(xy_grid[0, :, 0])))
    grid_dy = float(np.mean(np.diff(xy_grid[:, 0, 1])))
    if support_kernels is None:
        support_kernels = [
            rectangular_bottom_kernel(
                stone_xy_extent(inventory, stone_idx),
                grid_dx,
                grid_dy,
            )
        ]
    if footprint_bounds is None:
        half = 0.5 * stone_xy_extent(inventory, stone_idx)
        footprint_bounds = np.tile(np.array([-half, half]), (len(support_kernels), 1, 1))
    footprint_bounds = np.asarray(footprint_bounds, dtype=float)
    if footprint_bounds.shape != (len(support_kernels), 2, 2):
        raise ValueError("footprint bounds must contain XY minima and maxima for each orientation")
    target_statics = [target_anchor_static(inventory, stone_idx, xy_bounds=bounds)
                      for bounds in footprint_bounds]
    target_bounds = np.array([target_anchor_bounds(inventory, stone_idx, _static=static)
                              for static in target_statics]).reshape(-1, 2, 2)
    valid_by_orientation = np.all(
        (xy_grid[None] >= target_bounds[:, 0, None, None, :])
        & (xy_grid[None] <= target_bounds[:, 1, None, None, :]), axis=-1)
    valid = np.any(valid_by_orientation, axis=0)
    score_map = np.full(shape, np.nan, dtype=float)
    candidate_mask = np.zeros(shape, dtype=bool)
    connectivity_mask = valid.copy()
    height_score_map = np.full(shape, np.nan, dtype=float)
    support_likelihood_map = np.full(shape, np.nan, dtype=float)
    fill_area_map = np.full(shape, np.nan, dtype=float)
    target_boundary_map = np.full(shape, np.nan, dtype=float)
    planar_bias_map = np.full(shape, np.nan, dtype=float)
    placement_height_map = np.full(shape, np.nan, dtype=float)
    orientation_index_map = np.full(shape, -1, dtype=int)
    if not np.any(valid):
        return {
            "x_coords": xy_grid[0, :, 0].astype(float).copy(),
            "y_coords": xy_grid[:, 0, 1].astype(float).copy(),
            "scores": score_map,
            "valid": valid.copy(),
            "candidate_mask": candidate_mask,
            "connectivity_mask": connectivity_mask,
            "candidates": [],
        }

    cfg = inventory.cfg.action.planar.get("floor", {})
    score_cfg = inventory.cfg.action.planar.get("score", {})
    w_height = float(score_cfg.get("height", -1.5))
    w_support_likelihood = float(score_cfg.get("support_likelihood", 2.0))
    w_fill_area = float(score_cfg.get("fill_area", 0.0))
    w_target_boundary = float(score_cfg.get("target_boundary", 0.0))
    # Older .stack contexts recorded these settings under excavator-specific names.
    w_planar_bias = float(score_cfg.get("planar_bias", score_cfg.get("excavator_distance", 0.25)))
    planar_bias_xy = score_planar_bias_xy(score_cfg)
    planar_bias_axis = str(
        score_cfg.get("planar_bias_axis", score_cfg.get("excavator_distance_axis", "xy"))
    ).lower()

    # Connectivity remains active during floor-fill retries. An empty active
    # layer needs a seed placement, so its connectivity mask is unrestricted.
    layer = active_floor if active_floor is not None else active_floor_context(inventory, state)
    has_occupied = layer is not None and np.any(layer["occupied"])
    if has_occupied:
        occupied_dilated = dilated_mask(layer["occupied"])

    # Each row is an orientation/anchor pair, in orientation-major order.
    # This keeps footprint and support computations aligned without a K x K
    # product between candidate orientations and bottom kernels.
    orientations, rows, columns = np.nonzero(valid_by_orientation)
    grid_index = np.column_stack((rows, columns))
    scaled_bounds = float(cfg.get("footprint_scale", 0.9)) * footprint_bounds
    bounds = scaled_bounds[orientations]
    cell_xy = xy_grid[grid_index[:, 0], grid_index[:, 1]]
    count = len(grid_index)
    keep = np.ones(count, dtype=bool)
    connected = np.ones(count, dtype=bool)
    unfilled_cells = np.zeros(count, dtype=int)
    max_unfilled_cells = 1
    if has_occupied:
        # Summed-area queries avoid an orientations x anchors x floor-cells
        # tensor. Each rectangle needs four lookups per occupancy mask.
        xs, ys = layer["xx"][0], layer["yy"][:, 0]
        rectangles = (
            np.searchsorted(xs, cell_xy[:, 0] + bounds[:, 0, 0], side="left"),
            np.searchsorted(xs, cell_xy[:, 0] + bounds[:, 1, 0], side="right"),
            np.searchsorted(ys, cell_xy[:, 1] + bounds[:, 0, 1], side="left"),
            np.searchsorted(ys, cell_xy[:, 1] + bounds[:, 1, 1], side="right"),
        )
        contact_cells = rectangle_cell_counts(occupied_dilated, rectangles)
        connected = contact_cells > 0
        connectivity_mask[:] = False
        np.logical_or.at(connectivity_mask, (rows, columns), connected)
    if has_occupied:
        unfilled_cells = rectangle_cell_counts(~layer["occupied"], rectangles)
        max_unfilled_cells = max(1, int(np.max(unfilled_cells, initial=0)))
        # Relax only the admission mask. Empty floor remains valuable when
        # upper placements are needed to continue the search.
        if not relax_floor_fill:
            keep = unfilled_cells > 0
    planar_bias = planar_bias_grid(
        inventory, xy_grid, planar_bias_xy, planar_bias_axis,
    )[grid_index[:, 0], grid_index[:, 1]]

    base = np.flatnonzero(keep)
    base_indices = grid_index[base]
    base_xy = cell_xy[base]
    base_xy_items = [xy.copy() for xy in base_xy]
    parts = []
    for orientation_index, kernel in enumerate(support_kernels):
        group = np.flatnonzero(orientations[base] == orientation_index)
        indices = base_indices[group]
        if not len(indices):
            continue
        kernel_support = support_likelihood_maps_by_kernel(
            bundle.surface_height, [kernel], support.contact_gap_tolerance,
            cells=(indices[:, 0], indices[:, 1]),
        )
        support_scores = kernel_support.support[0]
        selected = np.flatnonzero(np.isfinite(support_scores))
        items = group[selected]
        contact_heights = kernel_support.contact_height[0][selected].copy()
        for index in np.flatnonzero(~np.isfinite(contact_heights)):
            contact_heights[index] = mean_top_height_at_anchor(
                inventory, heights, xy_grid, base_xy[items[index]], stone_idx,
                _bounds=scaled_bounds[orientation_index],
            )
        boundary_lower, boundary_upper = target_anchor_bounds(
            inventory, stone_idx, z=contact_heights, _static=target_statics[orientation_index],
        )
        boundary_scores = target_boundary_proximity(
            base_xy[items], boundary_lower, boundary_upper,
        )
        parts.append((
            items,
            np.full(len(selected), orientation_index),
            contact_heights,
            support_scores[selected],
            kernel_support.placement_height[0][selected],
            boundary_scores,
        ))

    # Orientation-major, row-major order for tie breaking and RNG draws.
    item_base, item_orientation, item_height, item_support, item_placement, item_boundary = (
        np.concatenate(values) for values in zip(*parts)
    ) if parts else (np.zeros(0, dtype=int),) * 6
    if not len(item_base):
        return {
            "x_coords": xy_grid[0, :, 0].astype(float).copy(),
            "y_coords": xy_grid[:, 0, 1].astype(float).copy(),
            "scores": score_map,
            "valid": valid.copy(),
            "candidate_mask": candidate_mask,
            "connectivity_mask": connectivity_mask,
            "height": heights.copy(),
            "candidates": [],
        }
    item_base = item_base.astype(int)
    cells = base[item_base]
    h_min = float(np.min(item_height))
    h_span = max(float(np.max(item_height) - h_min), 1e-6)
    height_term = (item_height - h_min) / h_span
    fill_area = unfilled_cells[cells] / max_unfilled_cells
    candidate_scores = (
        w_height * height_term
        + w_support_likelihood * item_support
        + w_fill_area * fill_area
        + w_target_boundary * item_boundary
        + w_planar_bias * planar_bias[cells]
    )

    # Prefer an eligible orientation when a cell has both connected and
    # disconnected poses. Keep raw scores for cells with no eligible pose.
    flat_cell = grid_index[cells, 0] * shape[1] + grid_index[cells, 1]
    order = np.lexsort((np.arange(len(cells)), -candidate_scores, ~connected[cells], flat_cell))
    winners = order[np.r_[True, flat_cell[order][1:] != flat_cell[order][:-1]]]
    rows, columns = grid_index[cells[winners], 0], grid_index[cells[winners], 1]
    score_map[rows, columns] = candidate_scores[winners]
    candidate_mask[rows, columns] = connected[cells[winners]]
    height_score_map[rows, columns] = height_term[winners]
    support_likelihood_map[rows, columns] = item_support[winners]
    fill_area_map[rows, columns] = fill_area[winners]
    target_boundary_map[rows, columns] = item_boundary[winners]
    planar_bias_map[rows, columns] = planar_bias[cells[winners]]
    placement_height_map[rows, columns] = item_placement[winners]
    orientation_index_map[rows, columns] = item_orientation[winners]

    # Scores are independent of connectivity. Apply the hard mask before
    # constructing the pool used by every elite/Boltzmann draw.
    admitted_scores = candidate_scores[connected[cells]]
    candidate_score_min = float(np.min(admitted_scores)) if len(admitted_scores) else 0.
    candidate_score_span = max(float(np.ptp(admitted_scores)), 1e-12) if len(admitted_scores) else 1.
    normalized_scores = np.clip(
        (candidate_scores - candidate_score_min) / candidate_score_span, 0.0, 1.0,
    )
    candidates = [
        {
            "xy": base_xy_items[item],
            "score": score,
            "height": height,
            "support_likelihood": support,
            "placement_height": placement,
            "orientation_index": orientation,
            "target_boundary": boundary,
            "normalized_score": normalized,
        }
        for item, cell, score, height, support, placement, orientation, boundary, normalized in zip(
            item_base.tolist(), cells.tolist(), candidate_scores.tolist(), item_height.tolist(),
            item_support.tolist(), item_placement.tolist(), item_orientation.astype(int).tolist(),
            item_boundary.tolist(), normalized_scores.tolist(),
        ) if connected[cell]
    ]

    return {
        "x_coords": xy_grid[0, :, 0].astype(float).copy(),
        "y_coords": xy_grid[:, 0, 1].astype(float).copy(),
        "scores": score_map,
        "valid": valid.copy(),
        "candidate_mask": candidate_mask,
        "connectivity_mask": connectivity_mask,
        "height": heights.copy(),
        "height_term": height_score_map,
        "support_likelihood": support_likelihood_map,
        "fill_area": fill_area_map,
        "target_boundary": target_boundary_map,
        "planar_bias": planar_bias_map,
        "placement_height": placement_height_map,
        "orientation_index": orientation_index_map,
        "weights": {
            "height": w_height,
            "support_likelihood": w_support_likelihood,
            "fill_area": w_fill_area,
            "target_boundary": w_target_boundary,
            "planar_bias": w_planar_bias,
        },
        "h_min": h_min,
        "h_span": h_span,
        "candidates": candidates,
    }

def rectangle_cell_counts(mask: np.ndarray, rectangles) -> np.ndarray:
    """Count true grid cells in rectangles with exclusive upper indices."""
    x0, x1, y0, y1 = rectangles
    summed = np.pad(np.cumsum(np.cumsum(mask, axis=0), axis=1), ((1, 0), (1, 0)))
    return summed[y1, x1] - summed[y0, x1] - summed[y1, x0] + summed[y0, x0]


def score_grid_shape(inventory, height_map: Optional[np.ndarray]) -> tuple[int, int]:
    size = inventory.cfg.action.planar.get("score_map_size", None)
    if size is None:
        if height_map is None:
            return (32, 32)
        return tuple(np.asarray(height_map).shape[:2])
    if isinstance(size, int):
        width = height = max(int(size), 2)
    else:
        width = max(int(size[0]), 2)
        height = max(int(size[1]), 2)
    return height, width

def xy_grid_for_shape(inventory, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    x = np.linspace(float(inventory.xlim[0]), float(inventory.xlim[1]), width)
    y = np.linspace(float(inventory.ylim[0]), float(inventory.ylim[1]), height)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.stack([xx, yy], axis=-1)

def target_diagonal(inventory) -> float:
    wall = inventory.target_wall
    return max(float(np.hypot(wall.width, wall.length)), 1e-6)

def target_axis_span(inventory, axis: int) -> float:
    bounds = inventory.xlim if axis == 0 else inventory.ylim
    return max(float(bounds[1] - bounds[0]), 1e-6)

def score_planar_bias_xy(score_cfg) -> np.ndarray:
    xy = score_cfg.get("planar_bias_xy", score_cfg.get("excavator_xy"))
    if xy is None:
        return np.zeros(2, dtype=float)
    arr = np.asarray(xy, dtype=float).reshape(-1)
    if arr.shape[0] < 2 or not np.all(np.isfinite(arr[:2])):
        return np.zeros(2, dtype=float)
    return arr[:2].copy()

def score_planar_bias(
    inventory,
    xy: np.ndarray,
    planar_bias_xy: np.ndarray,
    axis: str,
) -> float:
    if axis in ("x", "0"):
        distance = abs(float(xy[0] - planar_bias_xy[0]))
        scale = target_axis_span(inventory, 0)
    elif axis in ("y", "1"):
        distance = abs(float(xy[1] - planar_bias_xy[1]))
        scale = target_axis_span(inventory, 1)
    else:
        distance = float(np.linalg.norm(xy - planar_bias_xy))
        scale = target_diagonal(inventory)
    return min(distance / scale, 1.0)

_PLANAR_BIAS_GRIDS: dict = {}

def planar_bias_grid(
    inventory,
    xy_grid: np.ndarray,
    planar_bias_xy: np.ndarray,
    axis: str,
) -> np.ndarray:
    """score_planar_bias for every grid cell, reused while inputs match.

    The scalar function is evaluated per cell so values stay bit-identical.
    """
    wall = inventory.target_wall
    key = (xy_grid.shape, xy_grid.tobytes(), np.asarray(planar_bias_xy, dtype=float).tobytes(), axis,
           tuple(inventory.xlim), tuple(inventory.ylim), float(wall.width), float(wall.length))
    grid = _PLANAR_BIAS_GRIDS.get(key)
    if grid is None:
        grid = np.array([
            [score_planar_bias(inventory, xy, planar_bias_xy, axis) for xy in row]
            for row in xy_grid
        ]).reshape(xy_grid.shape[:2])
        if len(_PLANAR_BIAS_GRIDS) >= 8:
            _PLANAR_BIAS_GRIDS.clear()
        _PLANAR_BIAS_GRIDS[key] = grid
    return grid

def mean_top_height_at_anchor(
    inventory,
    top_heights: np.ndarray,
    xy_grid: np.ndarray,
    anchor: np.ndarray,
    stone_idx: Optional[int],
    _bounds: Optional[np.ndarray] = None,
) -> float:
    footprint = anchor_grid_mask(inventory, xy_grid, anchor, stone_idx, _bounds=_bounds)
    values = np.asarray(top_heights[footprint], dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        xy = np.asarray(anchor, dtype=float)
        distances = np.linalg.norm(xy_grid - xy, axis=-1)
        y_idx, x_idx = np.unravel_index(int(np.argmin(distances)), distances.shape)
        return float(top_heights[y_idx, x_idx])
    return float(np.mean(values))

def target_anchor_static(inventory, stone_idx: Optional[int], *, xy_bounds=None):
    """Precompute the per-(inventory, stone_idx) terms of ``target_anchor_bounds``
    that do not depend on ``z``. Callers that evaluate many z values for the same
    stone (e.g. a per-orientation/per-cell scoring loop) should compute this once
    and pass it back in via ``target_anchor_bounds(..., _static=...)`` instead of
    re-resolving config lookups and re-querying stone mesh extents on every call.
    """
    wall = inventory.target_wall
    origin = np.asarray(wall.origin[:2], dtype=float)
    half_extent = np.asarray([wall.width, wall.length], dtype=float) / 2.0
    margin = float(inventory.cfg.action.planar.get("target_mask_margin", 0.0))
    if xy_bounds is None:
        half = 0.5 * stone_xy_extent(inventory, stone_idx)
        xy_bounds = np.array([-half, half])
    scale = float(inventory.cfg.action.planar.get("floor", {}).get("boundary_inset_radius_scale", 0.6))
    offsets = scale * np.asarray(xy_bounds, dtype=float)
    return wall, origin, half_extent, margin, offsets

def target_anchor_bounds(
    inventory,
    stone_idx: Optional[int],
    z: Optional[float | np.ndarray] = None,
    _static=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return XY bounds for a scalar height or an array of heights."""
    wall, origin, half_extent, margin, offsets = (
        target_anchor_static(inventory, stone_idx) if _static is None else _static
    )
    lower = origin - half_extent
    upper = origin + half_extent
    if z is not None:
        relative_z = np.clip(
            np.asarray(z, dtype=float) - float(wall.origin[2]), 0.0, wall.height,
        )
        lower = lower + relative_z[..., None] * np.tan(
            [
                max(float(wall.taper["width_negative"]), 0.0),
                max(float(wall.taper["length_negative"]), 0.0),
            ]
        )
        upper = upper - relative_z[..., None] * np.tan(
            [
                max(float(wall.taper["width_positive"]), 0.0),
                max(float(wall.taper["length_positive"]), 0.0),
            ]
        )
    lower += margin - offsets[0]
    upper -= margin + offsets[1]
    return lower, upper

def target_boundary_proximity(
    xy: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float | np.ndarray:
    """Score one XY point or arrays of points/bounds along the last axis."""
    xy = np.asarray(xy, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    valid = ~np.any((upper <= lower) | (xy < lower) | (xy > upper), axis=-1)
    center = 0.5 * (lower + upper)
    half_extent = 0.5 * (upper - lower)
    normalized = np.zeros_like(xy - center)
    np.divide(xy - center, half_extent, out=normalized, where=valid[..., None])
    scores = np.clip(np.max(np.abs(normalized), axis=-1), 0.0, 1.0)
    return float(scores) if scores.ndim == 0 else scores

def anchor_grid_mask(
    inventory,
    xy_grid: np.ndarray,
    anchor: np.ndarray,
    stone_idx: Optional[int],
    _bounds: Optional[np.ndarray] = None,
) -> np.ndarray:
    if _bounds is None:
        half = footprint_half_extent(inventory, stone_idx)
        _bounds = np.array([-half, half])
    xy = np.asarray(anchor, dtype=float)
    xx = xy_grid[..., 0]
    yy = xy_grid[..., 1]
    return (
        (xx >= xy[0] + _bounds[0, 0])
        & (xx <= xy[0] + _bounds[1, 0])
        & (yy >= xy[1] + _bounds[0, 1])
        & (yy <= xy[1] + _bounds[1, 1])
    )

def corner_seed_candidates(candidates, anchor_bounds, grid_spacing):
    """Eligible grid cells within one grid interval of a feasible corner.

    Bounds include the full rotated footprint, so a seed's center is inset.
    A candidate belongs to its nearest corner only, even for narrow targets.
    Existing score values and orientation indices are preserved.
    """
    if not candidates:
        return []
    xy = np.asarray([item["xy"] for item in candidates])
    orientations = np.asarray([item["orientation_index"] for item in candidates], dtype=int)
    bounds = np.asarray(anchor_bounds)[orientations]
    lower, upper = bounds[:, 0], bounds[:, 1]
    valid = np.all((lower <= upper) & (xy >= lower) & (xy <= upper), axis=1)
    # Counterclockwise starting from the lower-X/lower-Y corner.
    sides = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=bool)
    corners = np.where(sides[None], upper[:, None], lower[:, None])
    distance = np.abs(xy[:, None] - corners) / np.maximum(np.abs(grid_spacing), 1e-12)
    nearest = np.argmin(np.sum(distance ** 2, axis=2), axis=1)
    close = np.all(distance[np.arange(len(xy)), nearest] <= 1. + 1e-9, axis=1)
    return [dict(item, seed_corner=int(corner))
            for item, corner, keep in zip(candidates, nearest, valid & close) if keep]


def sample_corner_candidates(inventory, candidates, count, *, rng):
    """Give available corners a first draw, then fill from the same seed pool."""
    selected = []
    corners = np.unique([item["seed_corner"] for item in candidates])
    rng.shuffle(corners)
    for corner in corners[:max(count, 0)]:
        pool = [item for item in candidates if item["seed_corner"] == corner]
        selected.extend(sample_scored_candidates(inventory, pool, 1, True, rng=rng))
    if len(selected) < count:
        chosen = {id(item) for item in selected}
        remaining = [item for item in candidates if id(item) not in chosen]
        if remaining:
            selected.extend(sample_scored_candidates(
                inventory, remaining, count - len(selected), True, rng=rng))
    return selected


def sample_scored_candidates(
    inventory,
    candidates: list[dict],
    k: int,
    random_start: bool,
    *, rng,
) -> list[dict]:
    """Select highest-scoring elites, then Boltzmann draws without replacement.

    Candidates must be distinct objects.
    """
    indices = sample_scored_indices(
        inventory.cfg.action.planar, [item["score"] for item in candidates],
        k, random_start, rng=rng)
    return [candidates[index] for index in indices]


def sample_scored_indices(settings, scores, k, random_start, *, rng):
    """Array counterpart of the shared elite/Boltzmann selection policy."""
    k = max(int(k), 1)
    temperature = float(settings.get("boltzmann_temperature", 1.0))
    if temperature <= 0.0:
        raise ValueError("action.planar.boltzmann_temperature must be positive")
    min_probability = float(
        settings.get("boltzmann_min_probability", 0.0)
    )
    if min_probability < 0.0:
        raise ValueError("action.planar.boltzmann_min_probability must be >= 0")
    score_cfg = settings.get("score", {})
    elite_fraction = float(score_cfg.get("elite_fraction", 0.0))
    if elite_fraction < 0.0:
        raise ValueError("action.planar.score.elite_fraction must be >= 0")
    scores = np.asarray(scores, dtype=float)
    chosen = np.zeros(len(scores), dtype=bool)
    selected: list[int] = []

    def choose(indices) -> bool:
        index = _weighted_choice(indices, chosen, scores, temperature, min_probability, rng=rng)
        if index is None:
            return False
        selected.append(index)
        chosen[index] = True
        return True

    elite_count = min(int(np.ceil(k * min(elite_fraction, 1.0))), len(scores))
    if elite_count > 0:
        # Stable descending order, as sorted(..., reverse=True) on finite scores.
        for index in np.argsort(-scores, kind="stable")[:elite_count]:
            if len(selected) >= k:
                break
            if not chosen[index]:
                selected.append(index)
                chosen[index] = True

    remaining = np.flatnonzero(~chosen)
    while len(selected) < k and choose(remaining):
        pass

    if not random_start:
        selected.sort(key=scores.__getitem__, reverse=True)
    return selected

def _weighted_choice(
    indices: np.ndarray,
    chosen: np.ndarray,
    scores: np.ndarray,
    temperature: float,
    min_probability: float,
    *, rng,
) -> Optional[int]:
    pool = indices[~chosen[indices]]
    if not len(pool):
        return None
    logits = scores[pool] / temperature
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    if min_probability > 0.0:
        probs = np.maximum(probs, min_probability)
    probs = probs / np.sum(probs)
    return int(pool[int(rng.choice(len(pool), p=probs))])

def footprint_half_extent(inventory, stone_idx: Optional[int], _floor_cfg=None) -> np.ndarray:
    """Shared half-footprint extent for sampling and anchor-grid lookup."""
    cfg = (
        inventory.cfg.action.planar.get("floor", {})
        if _floor_cfg is None
        else _floor_cfg
    )
    scale = float(cfg.get("footprint_scale", 0.9))
    return 0.5 * scale * stone_xy_extent(inventory, stone_idx)

def dilated_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    return (
        padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
        | padded[:-2, :-2]
        | padded[:-2, 2:]
        | padded[2:, :-2]
        | padded[2:, 2:]
    )

def stone_xy_extent(inventory, stone_idx: Optional[int]) -> np.ndarray:
    if stone_idx is None:
        extents = [
            np.asarray(stone.local_aabb_extent()[:2], dtype=float)
            for stone in inventory.stones
        ]
        return np.mean(extents, axis=0) if extents else np.ones(2, dtype=float)
    return np.asarray(
        inventory.stones[int(stone_idx)].local_aabb_extent()[:2],
        dtype=float,
    )
