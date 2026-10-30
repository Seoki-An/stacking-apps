"""Platform-neutral active-floor geometry used by sampling and orientation diagnostics."""

from __future__ import annotations

import numpy as np

from stacking_common.domain import Action, CompositeTargetGeometry, ObjectInventory, State
from stacking_common.domain.geometry import quaternion_matrix, transform_points
from .target_regions import TargetRegions, ground_height


def floor_policy_settings(settings) -> dict:
    """Extract layer geometry from the unified floor block."""
    return {key: settings[key] for key in (
        "coverage_mode", "bottom_tolerance", "fill_ratio", "grid_size",
    )}


class ActiveFloorPolicy:
    """Track occupied layers and measure candidate floor overlap."""

    def __init__(
        self, inventory: ObjectInventory, *, bottom_tolerance: float,
        fill_ratio: float, grid_size: int,
        coverage_mode: str = "bottom_slice",
        ground_z: float | None = None,
    ) -> None:
        if not np.isfinite(bottom_tolerance) or bottom_tolerance <= 0:
            raise ValueError("floor bottom tolerance must be positive")
        if not 0 < fill_ratio <= 1 or grid_size < 2:
            raise ValueError("invalid active-floor limits")
        if coverage_mode not in {"bottom_slice", "projected_footprint"}:
            raise ValueError("unsupported floor coverage mode")
        self.coverage_mode = coverage_mode
        self.inventory = inventory
        self.bottom_tolerance = float(bottom_tolerance)
        self.fill_ratio = float(fill_ratio)
        self.grid_size = int(grid_size)
        self.ground_z = ground_height(inventory, ground_z)
        self.target_regions = (TargetRegions(inventory.target, self.ground_z)
                               if isinstance(inventory.target, CompositeTargetGeometry) else None)
        self._region_grids = ({region.region_id: tuple(_readonly(value) for value in
                              region.grid((self.grid_size, self.grid_size)))
                              for region in self.target_regions.regions}
                             if self.target_regions is not None else {})
        # Search states share most placed stones with their parents. Entries and
        # masks are pure functions of (object, exact pose), so they are reused;
        # cached arrays are read-only.
        self._grid_values = None
        self._entries: dict = {}
        self._masks: dict = {}

    def pose_diagnostics(
        self,
        state: State,
        action: Action,
        *, context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Describe candidate overlap without rejecting its solved pose."""

        if context is None:
            context = self.context_details(state)
        if self.target_regions is not None:
            return self._composite_diagnostics(action, context)
        diagnostics = self._layer_diagnostics(
            action, context["support_z"], context["occupied"],
            neighbors=context["occupied_neighbor"], grid=(context["xx"], context["yy"]))
        diagnostics["fixed_scene_active_layer_coverage_mode"] = self.coverage_mode
        diagnostics["fixed_scene_completed_layers"] = context["completed_layers"]
        return diagnostics

    def context(self, state: State) -> tuple[float, np.ndarray]:
        """Find the first layer below the configured coverage threshold."""
        context = self.context_details(state)
        if self.target_regions is not None:
            raise ValueError("composite targets have per-region floors; use context_details()['regions']")
        return context["support_z"], context["occupied"]

    def context_details(self, state: State) -> dict[str, object]:
        """Share active-layer geometry with sampling and orientation diagnostics."""
        if self.target_regions is not None:
            return self._composite_context(state)
        entries = [
            self._entry(index, state.object_poses[state.object_ids[index]])
            for index in state.placed_object_indices
        ]
        support_z = self.ground_z
        remaining = list(entries)
        layer_entries: list[dict[str, object]] = []
        occupied = self._empty_mask()
        completed_layers = []
        # Every completed layer consumes at least one remaining stone.
        while remaining:
            layer_entries = [
                entry
                for entry in remaining
                if float(entry["bottom"]) <= support_z + self.bottom_tolerance
            ]
            occupied = self._occupancy(layer_entries, support_z)
            if float(np.mean(occupied)) < self.fill_ratio:
                break
            if not layer_entries:
                break
            completed_layers.append({"support_z": support_z,
                                     "coverage": float(np.mean(occupied))})
            used = {id(entry) for entry in layer_entries}
            remaining = [entry for entry in remaining if id(entry) not in used]
            support_z = max(float(entry["top"]) for entry in layer_entries)
            # The completed floor's footprint must not occupy the new layer.
            occupied = self._empty_mask()

        lower, upper, xx, yy = self._grid()
        return {
            "support_z": support_z, "bottom_tol": self.bottom_tolerance,
            "lower": lower, "upper": upper, "xx": xx, "yy": yy,
            "occupied": occupied, "occupied_neighbor": _neighbor_mask(occupied),
            "occupancy": float(np.mean(occupied)), "coverage_mode": self.coverage_mode,
            "completed_layers": completed_layers,
        }

    def _layer_diagnostics(self, action, support_z, occupied, *, neighbors=None, grid=None, valid=None):
        candidate = self._entry(action.object_index, action.pose)
        above = float(candidate["bottom"]) > support_z + self.bottom_tolerance
        candidate_cells = (self._mask(candidate, support_z, grid=grid) if valid is None else
                           self._compute_mask(candidate, support_z, grid) & valid)
        if neighbors is None:
            neighbors = _neighbor_mask(occupied)
        cell_count = int(np.count_nonzero(candidate_cells))
        overlap = int(np.count_nonzero(candidate_cells & occupied))
        unfilled = candidate_cells & ~occupied
        unfilled_count = int(np.count_nonzero(unfilled))
        contact = int(np.count_nonzero(candidate_cells & neighbors))
        fill_score = (
            0.0
            if cell_count == 0 or unfilled_count == 0
            else min(
                unfilled_count / cell_count
                + 0.5 * np.count_nonzero(unfilled & neighbors) / cell_count,
                1.0,
            )
        )
        diagnostics = {
            "fixed_scene_active_layer_bottom": float(candidate["bottom"]),
            "fixed_scene_active_layer_support_z": support_z,
            "fixed_scene_active_layer_fill_score": float(fill_score),
            "fixed_scene_active_layer_above": bool(above),
            "fixed_scene_active_layer_contact_cells": contact,
            "fixed_scene_active_layer_min_contact_cells": 1,
            "fixed_scene_active_layer_unfilled_cells": unfilled_count,
            "fixed_scene_active_layer_overlap_cells": overlap,
            "fixed_scene_active_layer_occupancy": (float(np.mean(occupied)) if valid is None else
                                                    float(np.count_nonzero(occupied) / max(np.count_nonzero(valid), 1))),
        }
        return diagnostics

    def _composite_context(self, state):
        entries = [self._entry(index, state.object_poses[state.object_ids[index]])
                   for index in state.placed_object_indices]
        contexts = []
        for region in self.target_regions.regions:
            lower, upper, xx, yy = self._region_grids[region.region_id]
            ground_mask = region.mask(xx, yy, self.ground_z)
            if not ground_mask.any():
                raise ValueError("floor grid misses a composite target region; increase floor.grid_size")
            # Only stones actually overlapping this region contribute to its layers.
            projected_target = (region.target.footprint_distance(np.c_[xx.ravel(), yy.ravel()]) <= 1e-9).reshape(xx.shape)
            remaining = [entry for entry in entries if np.any(projected_target & self._projected_mask(entry, xx, yy))]
            empty = not remaining
            support_z, completed = self.ground_z, []
            while True:
                valid = region.mask(xx, yy, support_z)
                layer = [entry for entry in remaining
                         if entry["bottom"] <= support_z + self.bottom_tolerance
                         and np.any(self._compute_mask(entry, support_z, (xx, yy)) & valid)]
                occupied = np.zeros(xx.shape, dtype=bool)
                for entry in layer:
                    occupied |= self._compute_mask(entry, support_z, (xx, yy))
                occupied &= valid
                occupancy = np.count_nonzero(occupied) / max(np.count_nonzero(valid), 1)
                if not valid.any() or not layer or occupancy < self.fill_ratio:
                    break
                completed.append({"support_z": support_z, "coverage": float(occupancy)})
                consumed = {id(entry) for entry in layer}
                remaining = [entry for entry in remaining if id(entry) not in consumed]
                support_z = max(entry["top"] for entry in layer)
            contexts.append({
                "region_id": region.region_id, "region": region, "empty": empty,
                "support_z": support_z, "bottom_tol": self.bottom_tolerance,
                "lower": lower, "upper": upper, "xx": xx, "yy": yy, "valid": valid,
                "occupied": occupied, "occupied_neighbor": _neighbor_mask(occupied) & valid,
                "occupancy": float(occupancy), "coverage_mode": self.coverage_mode,
                "completed_layers": completed,
            })
        return {"regions": tuple(contexts), "ground_z": self.ground_z}

    @staticmethod
    def _projected_mask(entry, xx, yy):
        mask = np.zeros(xx.shape, dtype=bool)
        for normals, offsets in entry["volumes"]:
            mask |= _volume_columns(normals, offsets, xx, yy)[0]
        return mask

    def _composite_diagnostics(self, action, context):
        metadata = self.target_regions.pose_metadata(self.inventory, action.object_index, action.pose)
        candidate = self._entry(action.object_index, action.pose)
        regions = [region for region in context["regions"] if region["region_id"] in metadata["target_region_ids"]]
        if not regions:
            return {**metadata,
                "fixed_scene_active_layer_fill_score": 0., "fixed_scene_active_layer_above": False}
        # Resolve the placement's region from geometry, not a stale CEM label.
        region = max(regions, key=lambda item: np.count_nonzero(
            self._projected_mask(candidate, item["xx"], item["yy"]) & item["valid"]))
        diagnostics = self._layer_diagnostics(action, region["support_z"], region["occupied"],
            neighbors=region["occupied_neighbor"], grid=(region["xx"], region["yy"]), valid=region["valid"])
        diagnostics.update(metadata, target_region_id=region["region_id"],
            fixed_scene_active_layer_coverage_mode=self.coverage_mode,
            fixed_scene_completed_layers=region["completed_layers"])
        return diagnostics

    def _entry(self, object_index: int, pose: np.ndarray) -> dict[str, object]:
        pose = np.asarray(pose, dtype=float)
        key = (int(object_index), pose.tobytes())
        entry = self._entries.get(key)
        if entry is None:
            if len(self._entries) >= 4096:
                self._entries.clear()
                self._masks.clear()
            entry = self._entries[key] = self._compute_entry(object_index, pose)
        return entry

    def _compute_entry(self, object_index: int, pose: np.ndarray) -> dict[str, object]:
        stack_object = self.inventory[object_index]
        points, _ = stack_object.mesh_arrays()
        world = transform_points(points, pose)
        rotation = quaternion_matrix(np.asarray(pose, dtype=float)[3:])
        translation = np.asarray(pose, dtype=float)[:3]
        volumes = []
        for geometry in stack_object.geometries:
            centers, normals, _ = geometry.face_data()
            world_normals = normals @ rotation.T
            volumes.append(
                (
                    _readonly(world_normals),
                    _readonly(np.einsum("ij,ij->i", normals, centers)
                              + world_normals @ translation),
                )
            )
        return {
            "volumes": tuple(volumes),
            "bottom": float(np.min(world[:, 2])),
            "top": float(np.max(world[:, 2])),
        }

    def _occupancy(self, entries, support_z: float) -> np.ndarray:
        occupied = self._empty_mask()
        for entry in entries:
            occupied |= self._mask(entry, support_z)
        return occupied

    def _empty_mask(self) -> np.ndarray:
        return np.zeros((self.grid_size, self.grid_size), dtype=bool)

    def _grid(self):
        if self._grid_values is None:
            self._grid_values = tuple(_readonly(values) for values in self._compute_grid())
        return self._grid_values

    def _compute_grid(self):
        target = self.inventory.target
        lower = target.origin[:2] - np.array([target.width, target.length]) / 2.0
        upper = target.origin[:2] + np.array([target.width, target.length]) / 2.0
        if self.coverage_mode == "projected_footprint":
            # Cell centers give equal-area samples without overweighting edges.
            fractions = (np.arange(self.grid_size) + 0.5) / self.grid_size
            xs = lower[0] + fractions * (upper[0] - lower[0])
            ys = lower[1] + fractions * (upper[1] - lower[1])
        else:
            xs = np.linspace(lower[0], upper[0], self.grid_size)
            ys = np.linspace(lower[1], upper[1], self.grid_size)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        return lower, upper, xx, yy

    def _mask(self, entry, support_z: float, *, grid=None) -> np.ndarray:
        default = self._grid()[2:]
        if grid is not None and not (grid[0] is default[0] and grid[1] is default[1]):
            return self._compute_mask(entry, support_z, grid)
        # Projected footprints do not depend on the support height.
        key = (id(entry), None if self.coverage_mode == "projected_footprint" else float(support_z))
        mask = self._masks.get(key)
        if mask is None or mask[0] is not entry:
            mask = self._masks[key] = (entry, _readonly(self._compute_mask(entry, support_z, default)))
        return mask[1]

    def _compute_mask(self, entry, support_z: float, grid) -> np.ndarray:
        xx, yy = grid
        occupied = np.zeros(xx.shape, dtype=bool)
        for normals, offsets in entry["volumes"]:
            if self.coverage_mode == "projected_footprint":
                # Union the convex components, preserving gaps between them.
                occupied |= _volume_columns(normals, offsets, xx, yy)[0]
            else:
                occupied |= _volume_slice_mask(
                    normals, offsets, xx, yy, support_z,
                    support_z + self.bottom_tolerance)
        return occupied


def _readonly(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    values.setflags(write=False)
    return values


def _volume_slice_mask(
    normals: np.ndarray,
    offsets: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    z_lower: float,
    z_upper: float,
) -> np.ndarray:
    inside, column_lower, column_upper = _volume_columns(normals, offsets, xx, yy)
    return (
        inside
        & (column_upper >= float(z_lower) - 1e-8)
        & (column_lower <= float(z_upper) + 1e-8)
    )


def _volume_columns(normals, offsets, xx, yy):
    rhs = (
        offsets[:, None, None]
        - normals[:, 0, None, None] * xx[None, ...]
        - normals[:, 1, None, None] * yy[None, ...]
    )
    z_coefficients = normals[:, 2]
    lower_faces = z_coefficients < -1e-10
    upper_faces = z_coefficients > 1e-10
    vertical_faces = ~(lower_faces | upper_faces)
    column_lower = np.full(xx.shape, -np.inf, dtype=float)
    column_upper = np.full(xx.shape, np.inf, dtype=float)
    if np.any(lower_faces):
        column_lower = np.max(
            rhs[lower_faces] / z_coefficients[lower_faces, None, None],
            axis=0,
        )
    if np.any(upper_faces):
        column_upper = np.min(
            rhs[upper_faces] / z_coefficients[upper_faces, None, None],
            axis=0,
        )
    inside = column_lower <= column_upper + 1e-8
    if np.any(vertical_faces):
        inside &= np.all(rhs[vertical_faces] >= -1e-8, axis=0)
    return inside, column_lower, column_upper


def _neighbor_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    neighbors = np.zeros(mask.shape, dtype=bool)
    for row in range(3):
        for column in range(3):
            if row == 1 and column == 1:
                continue
            neighbors |= padded[
                row : row + mask.shape[0],
                column : column + mask.shape[1],
            ]
    return neighbors
