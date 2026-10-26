"""Geometry-only initialization of placement height."""

import numpy as np

from stacking_common.domain import ObjectInventory, State
from stacking_common.domain.geometry import transform_points

from .types import InitializedPlacement, PlanarPlacement
from .target_regions import ground_height


class MeshHeightInitializer:
    """Place a candidate above overlapping mesh bounds and the ground plane."""

    def __init__(self, clearance: float, xy_padding: float, *, ground_z: float | None = None) -> None:
        if not np.isfinite(clearance) or clearance < 0.0:
            raise ValueError("initial height clearance must be non-negative")
        if not np.isfinite(xy_padding) or xy_padding < 0.0:
            raise ValueError("initial height XY padding must be non-negative")
        self.clearance = float(clearance)
        self.xy_padding = float(xy_padding)
        self.ground_z = ground_z

    def initialize(
        self,
        inventory: ObjectInventory,
        state: State,
        object_index: int,
        proposal: PlanarPlacement,
    ) -> InitializedPlacement:
        if object_index < 0 or object_index >= len(inventory):
            raise ValueError("placement object index is outside the inventory")
        candidate_points, _ = inventory[object_index].mesh_arrays()
        zero_height_pose = proposal.pose.copy()
        zero_height_pose[2] = 0.0
        candidate_world = transform_points(candidate_points, zero_height_pose)
        candidate_min_xy = np.min(candidate_world[:, :2], axis=0)
        candidate_max_xy = np.max(candidate_world[:, :2], axis=0)

        support_height = ground_height(inventory, self.ground_z)
        support_indices: list[int] = []
        for placed_index in state.placed_object_indices:
            placed = inventory[placed_index]
            placed_pose = state.object_poses.get(
                placed.object_id,
                inventory.initial_pose(placed.object_id),
            )
            placed_points, _ = placed.mesh_arrays()
            placed_world = transform_points(placed_points, placed_pose)
            placed_min_xy = np.min(placed_world[:, :2], axis=0)
            placed_max_xy = np.max(placed_world[:, :2], axis=0)
            if not _xy_overlaps(
                candidate_min_xy,
                candidate_max_xy,
                placed_min_xy,
                placed_max_xy,
                self.xy_padding,
            ):
                continue
            top = float(np.max(placed_world[:, 2]))
            if top > support_height:
                support_height = top
                support_indices = [int(placed_index)]
            elif np.isclose(top, support_height):
                support_indices.append(int(placed_index))

        pose = zero_height_pose.copy()
        pose[2] = support_height + self.clearance - float(
            np.min(candidate_world[:, 2])
        )
        return InitializedPlacement(
            pose=pose,
            support_height=support_height,
            support_object_indices=tuple(support_indices),
            diagnostics={
                "initial_height_clearance": self.clearance,
                "initial_height_xy_padding": self.xy_padding,
            },
        )


def _xy_overlaps(
    first_min: np.ndarray,
    first_max: np.ndarray,
    second_min: np.ndarray,
    second_max: np.ndarray,
    padding: float,
) -> bool:
    return bool(
        np.all(first_max + padding >= second_min)
        and np.all(second_max + padding >= first_min)
    )

