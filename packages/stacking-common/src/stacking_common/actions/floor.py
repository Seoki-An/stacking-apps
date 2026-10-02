"""Platform-neutral active-floor geometry shared by sampling and validation."""

from __future__ import annotations

import numpy as np

from stacking_common.domain import Action, ObjectInventory, State
from stacking_common.domain.geometry import quaternion_matrix, transform_points


class ActiveFloorPolicy:
    """Enforce lower-floor filling with limits supplied by the platform."""

    def __init__(
        self, inventory: ObjectInventory, *, bottom_tolerance: float,
        fill_ratio: float, grid_size: int, max_layers: int,
        minimum_contact_cells: int,
        score_weights: tuple[float, float, float, float, float, float],
        coverage_mode: str = "bottom_slice",
    ) -> None:
        if not np.isfinite(bottom_tolerance) or bottom_tolerance <= 0:
            raise ValueError("floor bottom tolerance must be positive")
        if not 0 < fill_ratio <= 1 or grid_size < 2 or max_layers < 1 or minimum_contact_cells < 0:
            raise ValueError("invalid active-floor limits")
        if coverage_mode not in {"bottom_slice", "projected_footprint"}:
            raise ValueError("unsupported floor coverage mode")
        self.coverage_mode = coverage_mode
        self.inventory = inventory
        self.bottom_tolerance = float(bottom_tolerance)
        self.fill_ratio = float(fill_ratio)
        self.grid_size = int(grid_size)
        self.max_layers = int(max_layers)
        self.minimum_contact_cells = int(minimum_contact_cells)
        if len(score_weights) != 6 or not np.all(np.isfinite(score_weights)):
            raise ValueError("floor scoring requires six finite weights")
        self.score_weights = tuple(float(value) for value in score_weights)

    def active_floor_rejection(
        self,
        state: State,
        action: Action,
        *, context: dict[str, object] | None = None,
    ) -> tuple[str | None, dict[str, object]]:
        """Reject a settled pose above or disconnected from an unfilled layer."""

        if context is None:
            context = self.context_details(state)
        reason, diagnostics = self._reject_at_layer(
            action, context["support_z"], context["occupied"],
            neighbors=context["occupied_neighbor"], grid=(context["xx"], context["yy"]))
        diagnostics["fixed_scene_active_layer_coverage_mode"] = self.coverage_mode
        diagnostics["fixed_scene_completed_layers"] = context["completed_layers"]
        return reason, diagnostics

    def context(self, state: State) -> tuple[float, np.ndarray]:
        """Find the first layer below the configured coverage threshold."""
        context = self.context_details(state)
        return context["support_z"], context["occupied"]

    def context_details(self, state: State) -> dict[str, object]:
        """Share the active layer's geometry between sampling and validation."""
        entries = [
            self._entry(index, state.object_poses[state.object_ids[index]])
            for index in state.placed_object_indices
        ]
        support_z = float(self.inventory.target.origin[2])
        remaining = list(entries)
        layer_entries: list[dict[str, object]] = []
        occupied = self._empty_mask()
        completed_layers = []
        for _ in range(self.max_layers):
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

    def _reject_at_layer(self, action, support_z, occupied, *, neighbors=None, grid=None):
        candidate = self._entry(action.object_index, action.pose)
        above = float(candidate["bottom"]) > support_z + self.bottom_tolerance
        candidate_cells = self._mask(candidate, support_z, grid=grid)
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
            "fixed_scene_active_layer_support_z": support_z,
            "fixed_scene_active_layer_fill_score": float(fill_score),
            "fixed_scene_active_layer_above": bool(above),
            "fixed_scene_active_layer_contact_cells": contact,
            "fixed_scene_active_layer_min_contact_cells": (
                self.minimum_contact_cells
            ),
            "fixed_scene_active_layer_unfilled_cells": unfilled_count,
            "fixed_scene_active_layer_overlap_cells": overlap,
            "fixed_scene_active_layer_occupancy": float(np.mean(occupied)),
        }
        if above:
            diagnostics["active_floor_reject_reason"] = "above_active_layer"
            return "above_active_layer", diagnostics
        if (
            np.any(occupied)
            and fill_score > 0.0
            and contact < self.minimum_contact_cells
        ):
            diagnostics["active_floor_reject_reason"] = (
                "disconnected_active_layer"
            )
            return "disconnected_active_layer", diagnostics
        return None, diagnostics

    def _entry(self, object_index: int, pose: np.ndarray) -> dict[str, object]:
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
                    world_normals,
                    np.einsum("ij,ij->i", normals, centers)
                    + world_normals @ translation,
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
        xx, yy = self._grid()[2:] if grid is None else grid
        occupied = self._empty_mask()
        for normals, offsets in entry["volumes"]:
            if self.coverage_mode == "projected_footprint":
                # Union the convex components, preserving gaps between them.
                occupied |= _volume_columns(normals, offsets, xx, yy)[0]
            else:
                occupied |= _volume_slice_mask(
                    normals, offsets, xx, yy, support_z,
                    support_z + self.bottom_tolerance)
        return occupied


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


class FloorProposalRanker:
    """Orientation-aware footprint support and open-floor frontier scoring.

    Rasterize each orientation once, then translate its sparse footprint over
    the scene grid. This avoids a convex-volume query for every grid anchor.
    These are proposal heuristics; solved poses still require exact filtering.
    """

    def __init__(self, policy, state, xs, ys, heights):
        self.policy = policy
        self.xs, self.ys, self.heights = xs, ys, heights
        self.support_z, occupied = policy.context(state)
        target = policy.inventory.target
        lower = target.origin[:2] - np.array([target.width, target.length]) / 2
        span = np.array([target.width, target.length])
        upper = lower + span
        self.in_target = (
            ((ys >= lower[1]) & (ys <= upper[1]))[:, None]
            & ((xs >= lower[0]) & (xs <= upper[0]))[None, :]
        )
        columns = np.clip(
            np.rint((xs - lower[0]) / span[0] * (policy.grid_size - 1)).astype(int),
            0, policy.grid_size - 1,
        )
        rows = np.clip(
            np.rint((ys - lower[1]) / span[1] * (policy.grid_size - 1)).astype(int),
            0, policy.grid_size - 1,
        )
        self.occupied = occupied[np.ix_(rows, columns)]
        self.neighbors = _neighbor_mask(occupied)[np.ix_(rows, columns)]
        self.has_occupied = bool(np.any(occupied))
        self.occupancy = float(np.mean(occupied))
        self.gap_labels, self.gap_centers = _gap_regions(self.in_target & ~self.occupied)

    def kernel(self, object_index, quaternion):
        entry = self.policy._entry(object_index, np.r_[np.zeros(3), quaternion])
        points, _ = self.policy.inventory[object_index].mesh_arrays()
        rotated = points @ quaternion_matrix(quaternion).T
        dx = self.xs[1] - self.xs[0] if len(self.xs) > 1 else self.policy.inventory.target.width
        dy = self.ys[1] - self.ys[0] if len(self.ys) > 1 else self.policy.inventory.target.length
        columns = np.arange(int(np.floor(rotated[:, 0].min() / dx)), int(np.ceil(rotated[:, 0].max() / dx)) + 1)
        rows = np.arange(int(np.floor(rotated[:, 1].min() / dy)), int(np.ceil(rotated[:, 1].max() / dy)) + 1)
        cc, rr = np.meshgrid(columns, rows)
        lower = np.full(cc.shape, np.inf)
        upper = np.full(cc.shape, -np.inf)
        for normals, offsets in entry["volumes"]:
            inside, lo, hi = _volume_columns(normals, offsets, cc * dx, rr * dy)
            lower = np.minimum(lower, np.where(inside, lo, np.inf))
            upper = np.maximum(upper, np.where(inside, hi, -np.inf))
        valid = np.isfinite(lower) & np.isfinite(upper)
        return rr[valid], cc[valid], lower[valid], upper[valid], entry["bottom"]

    def metrics(self, kernel, row, column):
        rr, cc, lower, upper, bottom = kernel
        rows, columns = rr + row, cc + column
        valid = (rows >= 0) & (rows < len(self.ys)) & (columns >= 0) & (columns < len(self.xs))
        if not np.any(valid):
            return None
        rows, columns, lower, upper = rows[valid], columns[valid], lower[valid], upper[valid]
        # The kernel includes the whole footprint for support, but only its
        # bottom slice contributes to filling the active floor.
        floor_cells = (lower <= bottom + self.policy.bottom_tolerance) & (upper >= bottom)
        # Height maps include an exterior margin for collision/support queries;
        # that margin must never be scored as empty target-floor space.
        floor_cells &= self.in_target[rows, columns]
        occupied = self.occupied[rows, columns]
        frontier = self.neighbors[rows, columns]
        unfilled = floor_cells & ~occupied
        cells = int(np.count_nonzero(floor_cells))
        open_cells = int(np.count_nonzero(unfilled))
        contact = int(np.count_nonzero(floor_cells & frontier))
        if cells == 0 or open_cells == 0:
            return None
        if self.has_occupied and contact < self.policy.minimum_contact_cells:
            return None
        heights = np.maximum(self.heights[rows, columns], self.support_z)
        translation_z = float(np.max(heights - lower))
        gaps = translation_z + lower - heights
        labels, counts = np.unique(self.gap_labels[rows[unfilled], columns[unfilled]], return_counts=True)
        gap = int(labels[np.argmax(counts)])
        center_row, center_column = self.gap_centers[gap]
        return {
            "floor_gap_region": (gap, int(row >= center_row), int(column >= center_column)),
            "active_floor_support_z": self.support_z,
            "active_floor_occupancy": self.occupancy,
            "active_floor_unfilled_cells": open_cells,
            "active_floor_contact_cells": contact,
            "active_floor_open_fraction": open_cells / cells,
            "active_floor_frontier_fraction": float(np.count_nonzero(unfilled & frontier)) / cells,
            "footprint_support_height": translation_z + float(bottom),
            "footprint_support_likelihood": float(np.mean(gaps <= self.policy.bottom_tolerance)),
        }


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


def _gap_regions(mask: np.ndarray):
    """Label open-floor components; their quadrants also split large gaps."""
    labels = np.full(mask.shape, -1, dtype=int)
    centers = {}
    for row, column in np.argwhere(mask):
        if labels[row, column] >= 0:
            continue
        label = len(centers)
        pending = [(int(row), int(column))]
        labels[row, column] = label
        cells = []
        while pending:
            r, c = pending.pop()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]
                        and mask[nr, nc] and labels[nr, nc] < 0):
                    labels[nr, nc] = label
                    pending.append((nr, nc))
        centers[label] = np.mean(cells, axis=0)
    return labels, centers
