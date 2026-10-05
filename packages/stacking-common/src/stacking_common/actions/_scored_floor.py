"""Pure array score policy migrated from stacking-planner floor_fill.py.

Only the dependencies of scored sampling are retained; platform inputs arrive
through the canonical inventory adapter in scored.py.
"""

from __future__ import annotations

from collections import deque

from typing import Optional

import numpy as np

from scipy.spatial import ConvexHull, QhullError

from scipy.spatial.transform import Rotation

from ._support_maps import (
    BottomHeightKernel,
    StateMapBundle,
    build_state_map_bundle,
    masked_mean_filter,
    rectangular_bottom_kernel,
    support_likelihood_maps_by_kernel,
    transformed_stone_volumes,
    volume_slice_mask,
)

LOWER_FLOOR_FILL_REJECT_STACKED_KEY = "lower_floor_fill_reject_stacked"

LEGACY_GROUND_FILL_REJECT_STACKED_KEY = "ground_fill_reject_stacked"

HEIGHT_SCORE_SMOOTH_KERNEL_SIZE = 5

def active_floor_context(inventory, state, min_count: Optional[int] = None) -> Optional[dict]:
    if min_count is not None and len(state.stone_seq) >= min_count:
        return None

    floor_fill_cfg = inventory.cfg.action.planar.get("floor_fill", {})
    support = inventory.support_settings
    support_z = support.ground_z
    bottom_tol = float(
        floor_fill_cfg.get("lower_floor_bottom_z_tolerance", support.z_tolerance)
    )
    fill_ratio = float(floor_fill_cfg.get("lower_floor_fill_ratio", 0.65))
    grid_size = max(int(floor_fill_cfg.get("lower_floor_occupancy_grid", 32)), 4)
    max_layers = int(floor_fill_cfg.get("lower_floor_max_layers", 8))

    entries = placed_floor_entries(inventory, state)
    last_top = support_z
    used: set[int] = set()
    for _ in range(max(max_layers, 1)):
        layer_items = [
            (idx, entry)
            for idx, entry in enumerate(entries)
            if idx not in used and entry["bottom"] <= support_z + bottom_tol
        ]
        layer = [entry for _, entry in layer_items]
        layer_top = max((entry["top"] for entry in layer), default=support_z)
        last_top = layer_top
        context = occupancy_context(inventory, layer, support_z, bottom_tol, grid_size)
        if context is None:
            return None
        if context["occupancy"] < fill_ratio:
            return context

        for idx, _ in layer_items:
            used.add(idx)
        support_z = layer_top
        if len(used) >= len(entries):
            return occupancy_context(inventory, [], support_z, bottom_tol, grid_size)

    return occupancy_context(inventory, [], last_top, bottom_tol, grid_size)

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

def target_structure_xy_overlap(
    inventory,
    stone_idx: int,
    pose: np.ndarray,
) -> bool:
    """Return whether the oriented stone footprint overlaps the target base."""
    pose = np.asarray(pose, dtype=float)
    if (
        not 0 <= int(stone_idx) < len(inventory.stones)
        or pose.shape[0] < 7
        or not np.all(np.isfinite(pose[:7]))
    ):
        return False

    stone_lower, stone_upper = floor_entry(
        inventory.stones[int(stone_idx)],
        pose,
        include_volumes=False,
    )["xy_aabb"]
    wall = inventory.target_wall
    target_half = 0.5 * np.asarray([wall.width, wall.length], dtype=float)
    target_origin = np.asarray(wall.origin[:2], dtype=float)
    target_lower = target_origin - target_half
    target_upper = target_origin + target_half
    return bool(
        np.all(stone_upper >= target_lower)
        and np.all(stone_lower <= target_upper)
    )

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

def connected_component_labels(mask: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    labels = np.full(mask.shape, -1, dtype=int)
    sizes: dict[int, int] = {}
    component_id = 0
    height, width = mask.shape
    for start_y, start_x in np.argwhere(mask):
        start_y = int(start_y)
        start_x = int(start_x)
        if labels[start_y, start_x] >= 0:
            continue
        queue = deque([(start_y, start_x)])
        labels[start_y, start_x] = component_id
        size = 0
        while queue:
            y, x = queue.popleft()
            size += 1
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if not mask[ny, nx] or labels[ny, nx] >= 0:
                    continue
                labels[ny, nx] = component_id
                queue.append((ny, nx))
        sizes[component_id] = size
        component_id += 1
    return labels, sizes

_GAP_COMPONENTS: dict = {}

def gap_components(occupied: np.ndarray):
    """Gap labels, sizes, centers and dilated occupancy for one floor grid.

    Every object scored on the same state shares this result, so it is reused
    while the occupancy bytes match. Returned arrays must be treated read-only.
    """
    occupied = np.asarray(occupied, dtype=bool)
    key = (occupied.shape, occupied.tobytes())
    result = _GAP_COMPONENTS.get(key)
    if result is None:
        labels, sizes = connected_component_labels(~occupied)
        result = (labels, sizes, component_centers(labels, sizes), dilated_mask(occupied))
        if len(_GAP_COMPONENTS) >= 64:
            _GAP_COMPONENTS.clear()
        _GAP_COMPONENTS[key] = result
    return result

def component_centers(labels: np.ndarray, sizes: dict[int, int]) -> dict[int, np.ndarray]:
    centers: dict[int, np.ndarray] = {}
    for component_id in sizes:
        coords = np.argwhere(labels == component_id)
        if len(coords) == 0:
            continue
        centers[int(component_id)] = np.mean(coords, axis=0)
    return centers

def active_layer_fill_metrics(
    inventory,
    layer: dict,
    stone_idx: int,
    pose: np.ndarray,
) -> dict:
    entry = floor_entry(inventory.stones[int(stone_idx)], np.asarray(pose, dtype=float))
    above_active_layer = entry["bottom"] > (
        float(layer["support_z"]) + float(layer["bottom_tol"])
    )
    metrics = {
        "fill_score": 0.0,
        "above_active_layer": bool(above_active_layer),
        "cell_count": 0,
        "unfilled_cells": 0,
        "contact_cells": 0,
        "overlap_cells": 0,
    }
    if above_active_layer:
        return metrics

    candidate_cells = floor_slice_mask(
        entry,
        layer["lower"],
        layer["upper"],
        layer["xx"],
        layer["yy"],
        float(layer["support_z"]),
        float(layer["support_z"]) + float(layer["bottom_tol"]),
    )
    cell_count = int(np.count_nonzero(candidate_cells))
    metrics["cell_count"] = cell_count
    if cell_count == 0:
        return metrics
    metrics["overlap_cells"] = int(np.count_nonzero(candidate_cells & layer["occupied"]))
    unfilled = candidate_cells & ~layer["occupied"]
    unfilled_cells = int(np.count_nonzero(unfilled))
    metrics["unfilled_cells"] = unfilled_cells
    contact_cells = int(np.count_nonzero(candidate_cells & layer["occupied_neighbor"]))
    metrics["contact_cells"] = contact_cells
    if unfilled_cells == 0:
        return metrics
    unfilled_ratio = float(np.count_nonzero(unfilled) / cell_count)
    adjacent_gap = unfilled & layer["occupied_neighbor"]
    adjacent_ratio = float(np.count_nonzero(adjacent_gap) / cell_count)
    metrics["fill_score"] = min(unfilled_ratio + 0.5 * adjacent_ratio, 1.0)
    return metrics

def score_xy_debug_map(
    inventory,
    state,
    stone_idx: Optional[int],
    scene_height_map: Optional[np.ndarray] = None,
    active_floor: Optional[dict] = None,
    support_kernels: Optional[list[BottomHeightKernel]] = None,
    state_map_bundle: Optional[StateMapBundle] = None,
    relax_floor_fill: bool = False,
) -> dict:
    if scene_height_map is None:
        scene_height_map = inventory.get_height_map(state)
    height_map = np.asarray(scene_height_map)
    shape = score_grid_shape(inventory, height_map)
    xy_grid = xy_grid_for_shape(inventory, shape)
    target_lower, target_upper = target_anchor_bounds(inventory, stone_idx)
    valid = np.all(
        (xy_grid >= target_lower) & (xy_grid <= target_upper),
        axis=-1,
    )
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
    score_map = np.full(shape, np.nan, dtype=float)
    candidate_mask = np.zeros(shape, dtype=bool)
    height_score_map = np.full(shape, np.nan, dtype=float)
    support_likelihood_map = np.full(shape, np.nan, dtype=float)
    open_area_map = np.full(shape, np.nan, dtype=float)
    fill_area_map = np.full(shape, np.nan, dtype=float)
    frontier_map = np.full(shape, np.nan, dtype=float)
    target_boundary_map = np.full(shape, np.nan, dtype=float)
    excavator_distance_map = np.full(shape, np.nan, dtype=float)
    placement_height_map = np.full(shape, np.nan, dtype=float)
    orientation_index_map = np.full(shape, -1, dtype=int)
    if not np.any(valid):
        return {
            "x_coords": xy_grid[0, :, 0].astype(float).copy(),
            "y_coords": xy_grid[:, 0, 1].astype(float).copy(),
            "scores": score_map,
            "valid": valid.copy(),
            "candidate_mask": candidate_mask,
            "candidates": [],
        }

    cfg = inventory.cfg.action.planar.get("floor_fill", {})
    score_cfg = inventory.cfg.action.planar.get("score", {})
    w_height = float(score_cfg.get("height", -1.5))
    w_support_likelihood = float(score_cfg.get("support_likelihood", 2.0))
    w_open_area = float(score_cfg.get("open_area", 1.0))
    w_fill_area = float(score_cfg.get("fill_area", 0.0))
    w_frontier = float(score_cfg.get("frontier", 1.0))
    w_target_boundary = float(score_cfg.get("target_boundary", 0.0))
    w_excavator_distance = float(score_cfg.get("excavator_distance", 0.25))
    excavator_xy = score_excavator_xy(score_cfg)
    excavator_distance_axis = str(
        score_cfg.get("excavator_distance_axis", "xy")
    ).lower()

    layer = None
    if not relax_floor_fill:
        layer = active_floor if active_floor is not None else active_floor_context(inventory, state)
    gap_labels = None
    gap_sizes: dict[int, int] = {}
    gap_centers: dict[int, np.ndarray] = {}
    connected_after = int(cfg.get("connected_after", 1))
    require_connected = (
        layer is not None
        and np.any(layer["occupied"])
        and len(state.stone_seq) >= connected_after
    )
    min_contact = max(int(cfg.get("u_shape_min_frontier_contact_cells", 1)), 0)
    has_occupied = layer is not None and np.any(layer["occupied"])
    if has_occupied:
        gap_labels, gap_sizes, gap_centers, occupied_dilated = gap_components(layer["occupied"])

    footprint_half = u_shape_footprint_half(inventory, stone_idx, _floor_cfg=cfg)
    # Anchor cells in row-major order. Every per-cell quantity below is an
    # array over these cells; filtering keeps that order.
    grid_index = np.argwhere(valid)
    cell_xy = xy_grid[grid_index[:, 0], grid_index[:, 1]]
    count = len(grid_index)
    keep = np.ones(count, dtype=bool)
    frontier = np.zeros(count)
    open_area = np.ones(count)
    unfilled_cells = np.zeros(count, dtype=int)
    gap_component = np.full(count, -1, dtype=int)
    gap_component_size = np.zeros(count, dtype=int)
    gap_region: list = [None] * count
    max_unfilled_cells = 1
    if has_occupied:
        # Footprint masks on the floor grid, one row per anchor cell, exactly
        # as anchor_occupancy_mask evaluates them.
        xx, yy = layer["xx"].ravel(), layer["yy"].ravel()
        footprints = (
            (xx >= (cell_xy[:, 0] - footprint_half[0])[:, None])
            & (xx <= (cell_xy[:, 0] + footprint_half[0])[:, None])
            & (yy >= (cell_xy[:, 1] - footprint_half[1])[:, None])
            & (yy <= (cell_xy[:, 1] + footprint_half[1])[:, None])
        )
        cell_count = np.count_nonzero(footprints, axis=1)
        unfilled = footprints & ~layer["occupied"].ravel()
        unfilled_cells = np.count_nonzero(unfilled, axis=1)
        keep = cell_count > 0
        open_area = np.divide(unfilled_cells, cell_count, out=np.zeros(count), where=keep)
        keep &= open_area > 0.0
        # Unfilled cells are never occupied, so all carry a gap label. The
        # most common label wins; ties keep the smallest, as np.unique does.
        labels = gap_labels.ravel()
        label_counts = np.stack(
            [np.count_nonzero(unfilled & (labels == label), axis=1) for label in range(len(gap_sizes))],
            axis=1,
        ) if gap_sizes else np.zeros((count, 0), dtype=int)
        kept = np.flatnonzero(keep)
        if len(kept):
            components = np.argmax(label_counts[kept], axis=1)
            centers = np.array([gap_centers[label] for label in range(len(gap_sizes))])[components]
            gap_component[kept] = components
            gap_component_size[kept] = np.array([gap_sizes[label] for label in range(len(gap_sizes))])[components]
            for index, component, above_y, above_x in zip(
                kept.tolist(), components.tolist(),
                (grid_index[kept, 0] >= centers[:, 0]).tolist(),
                (grid_index[kept, 1] >= centers[:, 1]).tolist(),
            ):
                gap_region[index] = (component, int(above_y), int(above_x))
        if np.any(keep):
            max_unfilled_cells = max(max_unfilled_cells, int(np.max(unfilled_cells[keep])))
        contact_cells = np.count_nonzero(footprints & occupied_dilated.ravel(), axis=1)
        frontier = np.minimum(
            np.divide(contact_cells, cell_count, out=np.zeros(count), where=cell_count > 0), 1.0,
        )
        if require_connected and min_contact > 0:
            keep &= contact_cells >= min_contact
        if bool(cfg.get("u_shape_filter", True)):
            for index in np.flatnonzero(keep):
                keep[index] = not creates_u_shape_pocket(
                    inventory, layer, cell_xy[index], stone_idx, _floor_cfg=cfg,
                    _half=footprint_half, _candidate=footprints[index].reshape(layer["xx"].shape),
                )
    excavator_distance = excavator_distance_grid(
        inventory, xy_grid, excavator_xy, excavator_distance_axis,
    )[grid_index[:, 0], grid_index[:, 1]]

    base = np.flatnonzero(keep)
    base_indices = grid_index[base]
    base_xy = cell_xy[base]
    # Orientations of one anchor share its xy array, as the dict spread did.
    base_xy_items = [xy.copy() for xy in base_xy]
    target_anchor_base = target_anchor_static(inventory, stone_idx)
    # Support is only read at the kept anchor cells; rows align with base.
    kernel_support = support_likelihood_maps_by_kernel(
        bundle.surface_height,
        support_kernels,
        support.contact_gap_tolerance,
        cells=(base_indices[:, 0], base_indices[:, 1]),
    )
    parts = []
    for orientation_index in range(kernel_support.support.shape[0]):
        support_scores = kernel_support.support[orientation_index]
        selected = np.flatnonzero(np.isfinite(support_scores))
        contact_heights = kernel_support.contact_height[orientation_index][selected].copy()
        for index in np.flatnonzero(~np.isfinite(contact_heights)):
            contact_heights[index] = mean_top_height_at_anchor(
                inventory, heights, xy_grid, base_xy[selected[index]], stone_idx,
                _half=footprint_half,
            )
        boundary_lower, boundary_upper = target_anchor_bounds(
            inventory, stone_idx, z=contact_heights, _static=target_anchor_base,
        )
        boundary_scores = target_boundary_proximity(
            base_xy[selected], boundary_lower, boundary_upper,
        )
        parts.append((
            selected,
            np.full(len(selected), orientation_index),
            contact_heights,
            support_scores[selected],
            kernel_support.placement_height[orientation_index][selected],
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
            "height": heights.copy(),
            "candidates": [],
        }
    item_base = item_base.astype(int)
    cells = base[item_base]
    h_min = float(np.min(item_height))
    h_span = max(float(np.max(item_height) - h_min), 1e-6)
    height_term = (item_height - h_min) / h_span
    fill_area = unfilled_cells[cells] / max_unfilled_cells
    # Same terms, in the same left-to-right order, as the scalar sum.
    candidate_scores = (
        w_height * height_term
        + w_support_likelihood * item_support
        + w_open_area * open_area[cells]
        + w_fill_area * fill_area
        + w_frontier * frontier[cells]
        + w_target_boundary * item_boundary
        + w_excavator_distance * excavator_distance[cells]
    )

    # Each map cell shows its first maximum-score candidate.
    flat_cell = grid_index[cells, 0] * shape[1] + grid_index[cells, 1]
    order = np.lexsort((np.arange(len(cells)), -candidate_scores, flat_cell))
    winners = order[np.r_[True, flat_cell[order][1:] != flat_cell[order][:-1]]]
    rows, columns = grid_index[cells[winners], 0], grid_index[cells[winners], 1]
    score_map[rows, columns] = candidate_scores[winners]
    candidate_mask[rows, columns] = True
    height_score_map[rows, columns] = height_term[winners]
    support_likelihood_map[rows, columns] = item_support[winners]
    open_area_map[rows, columns] = open_area[cells[winners]]
    fill_area_map[rows, columns] = fill_area[winners]
    frontier_map[rows, columns] = frontier[cells[winners]]
    target_boundary_map[rows, columns] = item_boundary[winners]
    excavator_distance_map[rows, columns] = excavator_distance[cells[winners]]
    placement_height_map[rows, columns] = item_placement[winners]
    orientation_index_map[rows, columns] = item_orientation[winners]

    candidate_score_min = float(np.min(candidate_scores))
    candidate_score_span = max(
        float(np.max(candidate_scores) - candidate_score_min),
        1e-12,
    )
    normalized_scores = np.clip(
        (candidate_scores - candidate_score_min) / candidate_score_span, 0.0, 1.0,
    )
    gap_component = gap_component.tolist()
    gap_component_size = gap_component_size.tolist()
    candidates = [
        {
            "xy": base_xy_items[item],
            "score": score,
            "height": height,
            "support_likelihood": support,
            "placement_height": placement,
            "orientation_index": orientation,
            "gap_component": gap_component[cell],
            "gap_component_size": gap_component_size[cell],
            "gap_region": gap_region[cell],
            "target_boundary": boundary,
            "normalized_score": normalized,
        }
        for item, cell, score, height, support, placement, orientation, boundary, normalized in zip(
            item_base.tolist(), cells.tolist(), candidate_scores.tolist(), item_height.tolist(),
            item_support.tolist(), item_placement.tolist(), item_orientation.astype(int).tolist(),
            item_boundary.tolist(), normalized_scores.tolist(),
        )
    ]

    return {
        "x_coords": xy_grid[0, :, 0].astype(float).copy(),
        "y_coords": xy_grid[:, 0, 1].astype(float).copy(),
        "scores": score_map,
        "valid": valid.copy(),
        "candidate_mask": candidate_mask,
        "height": heights.copy(),
        "height_term": height_score_map,
        "support_likelihood": support_likelihood_map,
        "open_area": open_area_map,
        "fill_area": fill_area_map,
        "frontier": frontier_map,
        "target_boundary": target_boundary_map,
        "excavator_distance": excavator_distance_map,
        "placement_height": placement_height_map,
        "orientation_index": orientation_index_map,
        "weights": {
            "height": w_height,
            "support_likelihood": w_support_likelihood,
            "open_area": w_open_area,
            "fill_area": w_fill_area,
            "frontier": w_frontier,
            "target_boundary": w_target_boundary,
            "excavator_distance": w_excavator_distance,
        },
        "h_min": h_min,
        "h_span": h_span,
        "candidates": candidates,
    }

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

def score_excavator_xy(score_cfg) -> np.ndarray:
    xy = score_cfg.get("excavator_xy", None)
    if xy is None:
        return np.zeros(2, dtype=float)
    arr = np.asarray(xy, dtype=float).reshape(-1)
    if arr.shape[0] < 2 or not np.all(np.isfinite(arr[:2])):
        return np.zeros(2, dtype=float)
    return arr[:2].copy()

def score_excavator_distance(
    inventory,
    xy: np.ndarray,
    excavator_xy: np.ndarray,
    axis: str,
) -> float:
    if axis in ("x", "0"):
        distance = abs(float(xy[0] - excavator_xy[0]))
        scale = target_axis_span(inventory, 0)
    elif axis in ("y", "1"):
        distance = abs(float(xy[1] - excavator_xy[1]))
        scale = target_axis_span(inventory, 1)
    else:
        distance = float(np.linalg.norm(xy - excavator_xy))
        scale = target_diagonal(inventory)
    return min(distance / scale, 1.0)

_EXCAVATOR_DISTANCE_GRIDS: dict = {}

def excavator_distance_grid(
    inventory,
    xy_grid: np.ndarray,
    excavator_xy: np.ndarray,
    axis: str,
) -> np.ndarray:
    """score_excavator_distance for every grid cell, reused while inputs match.

    The scalar function is evaluated per cell so values stay bit-identical.
    """
    wall = inventory.target_wall
    key = (xy_grid.shape, xy_grid.tobytes(), np.asarray(excavator_xy, dtype=float).tobytes(), axis,
           tuple(inventory.xlim), tuple(inventory.ylim), float(wall.width), float(wall.length))
    grid = _EXCAVATOR_DISTANCE_GRIDS.get(key)
    if grid is None:
        grid = np.array([
            [score_excavator_distance(inventory, xy, excavator_xy, axis) for xy in row]
            for row in xy_grid
        ]).reshape(xy_grid.shape[:2])
        if len(_EXCAVATOR_DISTANCE_GRIDS) >= 8:
            _EXCAVATOR_DISTANCE_GRIDS.clear()
        _EXCAVATOR_DISTANCE_GRIDS[key] = grid
    return grid

def mean_top_height_at_anchor(
    inventory,
    top_heights: np.ndarray,
    xy_grid: np.ndarray,
    anchor: np.ndarray,
    stone_idx: Optional[int],
    _half: Optional[np.ndarray] = None,
) -> float:
    footprint = anchor_grid_mask(inventory, xy_grid, anchor, stone_idx, _half=_half)
    values = np.asarray(top_heights[footprint], dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        xy = np.asarray(anchor, dtype=float)
        distances = np.linalg.norm(xy_grid - xy, axis=-1)
        y_idx, x_idx = np.unravel_index(int(np.argmin(distances)), distances.shape)
        return float(top_heights[y_idx, x_idx])
    return float(np.mean(values))

def target_anchor_static(inventory, stone_idx: Optional[int]):
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
    inset = anchor_inset_xy(inventory, stone_idx)
    return wall, origin, half_extent, margin, inset

def target_anchor_bounds(
    inventory,
    stone_idx: Optional[int],
    z: Optional[float | np.ndarray] = None,
    _static=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return XY bounds for a scalar height or an array of heights."""
    wall, origin, half_extent, margin, inset = (
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
    lower += margin + inset
    upper -= margin + inset
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
    _half: Optional[np.ndarray] = None,
) -> np.ndarray:
    half = u_shape_footprint_half(inventory, stone_idx) if _half is None else _half
    xy = np.asarray(anchor, dtype=float)
    xx = xy_grid[..., 0]
    yy = xy_grid[..., 1]
    return (
        (xx >= xy[0] - half[0])
        & (xx <= xy[0] + half[0])
        & (yy >= xy[1] - half[1])
        & (yy <= xy[1] + half[1])
    )

def sample_scored_candidates(
    inventory,
    candidates: list[dict],
    k: int,
    random_start: bool,
    *, rng,
) -> list[dict]:
    """Boltzmann/diversity selection over candidate indices.

    Pools, their order and every ``rng.choice(n, p=...)`` call match the
    original per-dict implementation, so seeded draws are unchanged.
    Candidates must be distinct objects.
    """
    k = max(int(k), 1)
    temperature = float(inventory.cfg.action.planar.get("boltzmann_temperature", 1.0))
    if temperature <= 0.0:
        raise ValueError("action.planar.boltzmann_temperature must be positive")
    min_probability = float(
        inventory.cfg.action.planar.get("boltzmann_min_probability", 0.0)
    )
    if min_probability < 0.0:
        raise ValueError("action.planar.boltzmann_min_probability must be >= 0")
    score_cfg = inventory.cfg.action.planar.get("score", {})
    elite_fraction = float(score_cfg.get("elite_fraction", 0.0))
    if elite_fraction < 0.0:
        raise ValueError("action.planar.score.elite_fraction must be >= 0")
    scores = np.asarray([item["score"] for item in candidates], dtype=float)
    heights = np.asarray([item["height"] for item in candidates], dtype=float)
    xy = np.asarray([item["xy"] for item in candidates], dtype=float).reshape(len(candidates), -1)
    h_min = float(np.min(heights))
    h_span = max(float(np.max(heights) - h_min), 1e-6)
    origin = np.asarray(inventory.target_wall.origin[:2], dtype=float)
    h_bin = np.clip(np.floor(3.0 * (heights - h_min) / h_span), 0, 2).astype(int)
    quadrant = (xy[:, 0] >= origin[0]).astype(int) + 2 * (xy[:, 1] >= origin[1]).astype(int)
    by_mode = _index_groups(4 * h_bin + quadrant)

    chosen = np.zeros(len(candidates), dtype=bool)
    selected: list[int] = []

    def choose(indices) -> bool:
        index = _weighted_choice(indices, chosen, scores, temperature, min_probability, rng=rng)
        if index is None:
            return False
        selected.append(index)
        chosen[index] = True
        return True

    # Representatives of the upper, middle and lower thirds of the Y range.
    y_min, y_max = float(np.min(xy[:, 1])), float(np.max(xy[:, 1]))
    if y_max > y_min:
        region = np.clip(np.floor(3.0 * (xy[:, 1] - y_min) / (y_max - y_min)), 0, 2)
        for value in (2, 1, 0):
            if len(selected) >= k:
                break
            choose(np.flatnonzero(region == value))

    # Gap representatives, largest components first; not capped by k.
    gap_keys: dict[tuple, int] = {}
    gap_ids = np.full(len(candidates), -1, dtype=int)
    gap_sizes = np.zeros(len(candidates), dtype=int)
    for index, item in enumerate(candidates):
        gap_region = item.get("gap_region", None)
        if gap_region is not None:
            gap_ids[index] = gap_keys.setdefault(tuple(gap_region), len(gap_keys))
            gap_sizes[index] = int(item.get("gap_component_size", 0))
    if gap_keys:
        has_gap = np.flatnonzero(gap_ids >= 0)
        gap_groups = sorted(
            (has_gap[group] for group in _index_groups(gap_ids[has_gap])),
            key=lambda group: (int(np.max(gap_sizes[group])), float(np.max(scores[group]))),
            reverse=True,
        )
        for group in gap_groups[: min(k, len(gap_groups))]:
            choose(group)

    elite_count = min(int(np.ceil(k * min(elite_fraction, 1.0))), len(candidates))
    if elite_count > 0:
        # Stable descending order, as sorted(..., reverse=True) on finite scores.
        for index in np.argsort(-scores, kind="stable")[:elite_count]:
            if len(selected) >= k:
                break
            if not chosen[index]:
                selected.append(index)
                chosen[index] = True

    modes = sorted(by_mode, key=lambda group: float(np.max(scores[group])), reverse=True)
    for group in modes:
        if len(selected) >= k:
            break
        choose(group)

    remaining = np.flatnonzero(~chosen)
    while len(selected) < k and choose(remaining):
        pass

    if not random_start:
        selected.sort(key=scores.__getitem__, reverse=True)
    return [candidates[index] for index in selected]

def _index_groups(keys: np.ndarray) -> list[np.ndarray]:
    """Ascending indices for each key, ordered by the key's first appearance."""
    if not len(keys):
        return []
    order = np.argsort(keys, kind="stable")
    starts = np.flatnonzero(np.r_[True, keys[order][1:] != keys[order][:-1]])
    groups = np.split(order, starts[1:])
    groups.sort(key=lambda group: group[0])
    return groups

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

def creates_u_shape_pocket(
    inventory,
    layer: dict,
    anchor: np.ndarray,
    stone_idx: Optional[int],
    _floor_cfg=None,
    _half=None,
    _candidate: Optional[np.ndarray] = None,
) -> bool:
    cfg = (
        inventory.cfg.action.planar.get("floor_fill", {})
        if _floor_cfg is None
        else _floor_cfg
    )
    if not bool(cfg.get("u_shape_filter", True)):
        return False

    candidate = (
        anchor_occupancy_mask(inventory, layer, anchor, stone_idx, _half=_half)
        if _candidate is None
        else _candidate
    )
    if not np.any(candidate & ~layer["occupied"]):
        return False

    if bool(cfg.get("u_shape_require_frontier_contact", True)):
        contact = candidate & dilated_mask(layer["occupied"])
        min_contact = max(int(cfg.get("u_shape_min_frontier_contact_cells", 1)), 0)
        if min_contact > 0 and int(np.count_nonzero(contact)) < min_contact:
            return True

    occupied = layer["occupied"] | candidate
    return has_u_shape_component(inventory, ~occupied, occupied, candidate, _floor_cfg=cfg)

def has_u_shape_component(
    inventory,
    empty: np.ndarray,
    occupied: np.ndarray,
    candidate: np.ndarray,
    _floor_cfg=None,
) -> bool:
    cfg = (
        inventory.cfg.action.planar.get("floor_fill", {})
        if _floor_cfg is None
        else _floor_cfg
    )
    min_empty = int(cfg.get("u_shape_min_empty_cells", 4))
    min_contact = int(cfg.get("u_shape_min_contact_cells", 6))
    max_opening_ratio = float(cfg.get("u_shape_max_opening_ratio", 0.40))

    for component in empty_components(empty):
        if len(component) < min_empty:
            continue
        component_mask = np.zeros(empty.shape, dtype=bool)
        ys, xs = zip(*component)
        component_mask[np.asarray(ys), np.asarray(xs)] = True
        if not np.any(component_mask & candidate) and not adjacent_to(
            component_mask,
            candidate,
        ):
            continue

        contact = int(np.count_nonzero(adjacent_to_mask(component_mask, occupied)))
        if contact < min_contact:
            continue

        opening = int(
            np.count_nonzero(grid_boundary_mask(component_mask.shape) & component_mask)
        )
        if opening / max(contact, 1) <= max_opening_ratio:
            return True
    return False

def u_shape_footprint_half(inventory, stone_idx: Optional[int], _floor_cfg=None) -> np.ndarray:
    """The ``anchor_occupancy_mask``/``anchor_grid_mask`` half-footprint extent.
    Both derive it identically from the same stone mesh extent and config key;
    hoist it once per stone_idx instead of re-deriving it on every grid cell.
    """
    cfg = (
        inventory.cfg.action.planar.get("floor_fill", {})
        if _floor_cfg is None
        else _floor_cfg
    )
    scale = float(cfg.get("u_shape_footprint_scale", 0.9))
    return 0.5 * scale * stone_xy_extent(inventory, stone_idx)

def anchor_occupancy_mask(
    inventory,
    layer: dict,
    anchor: np.ndarray,
    stone_idx: Optional[int],
    _half: Optional[np.ndarray] = None,
) -> np.ndarray:
    half = u_shape_footprint_half(inventory, stone_idx) if _half is None else _half
    xy = np.asarray(anchor, dtype=float)
    xx = layer["xx"]
    yy = layer["yy"]
    return (
        (xx >= xy[0] - half[0])
        & (xx <= xy[0] + half[0])
        & (yy >= xy[1] - half[1])
        & (yy <= xy[1] + half[1])
    )

def empty_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    visited = np.zeros(mask.shape, dtype=bool)
    components = []
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue
            queue = deque([(y, x)])
            visited[y, x] = True
            component = []
            while queue:
                cy, cx = queue.popleft()
                component.append((cy, cx))
                for ny, nx in (
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                ):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not visited[ny, nx]
                        and mask[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            components.append(component)
    return components

def adjacent_to(mask: np.ndarray, other: np.ndarray) -> bool:
    return bool(np.any(adjacent_to_mask(mask, other)))

def adjacent_to_mask(mask: np.ndarray, other: np.ndarray) -> np.ndarray:
    return mask & dilated_mask(other)

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

def grid_boundary_mask(shape: tuple[int, int]) -> np.ndarray:
    boundary = np.zeros(shape, dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    return boundary

def anchor_inset_xy(inventory, stone_idx: Optional[int]) -> np.ndarray:
    cfg = inventory.cfg.action.planar.get("floor_fill", {})
    scale = float(cfg.get("boundary_inset_radius_scale", 0.6))
    return np.maximum(scale * 0.5 * stone_xy_extent(inventory, stone_idx), 0.0)

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
