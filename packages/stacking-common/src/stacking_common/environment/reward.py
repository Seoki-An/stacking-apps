"""Concrete reward calculation migrated from the tabletop environment."""

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from stacking_common.domain import StackObject, State

from ._geometry import exposed_surface_normal, quaternion_matrix
from .overlap import OverlapEvaluator


class RewardTarget(Protocol):
    origin: np.ndarray
    height: float
    volume: float


class RewardInventory(Protocol):
    objects: Sequence[StackObject]
    target: RewardTarget

    def initial_pose(self, object_id: int) -> np.ndarray:
        """Return an object's pose before it enters the placed scene."""


SupportEvaluator = Callable[
    [RewardInventory, State, int, np.ndarray], tuple[int, bool]
]


@dataclass(frozen=True)
class RewardWeights:
    stability: float
    object_containment: float
    target_overlap_increment: float
    place_stability: float
    large_object_lower: float
    inward_orientation: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.stability,
                self.object_containment,
                self.target_overlap_increment,
                self.place_stability,
                self.large_object_lower,
                self.inward_orientation,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("reward weights must be finite")


@dataclass(frozen=True)
class RewardConfig:
    posegen_threshold: float
    containment_threshold: float
    weights: RewardWeights
    nonfinite_displacement: float = 2.0
    max_displacement: float | None = None

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.posegen_threshold)
            or self.posegen_threshold <= 0.0
        ):
            raise ValueError("posegen threshold must be positive and finite")
        if (
            not np.isfinite(self.containment_threshold)
            or not 0.0 <= self.containment_threshold <= 1.0
        ):
            raise ValueError("containment threshold must be between zero and one")
        if (
            not np.isfinite(self.nonfinite_displacement)
            or self.nonfinite_displacement < 0.0
        ):
            raise ValueError(
                "nonfinite displacement must be finite and non-negative"
            )
        if self.max_displacement is not None and (
            not np.isfinite(self.max_displacement)
            or self.max_displacement < 0.0
        ):
            raise ValueError(
                "maximum displacement must be finite and non-negative"
            )


class StackingReward:
    """Evaluate one placed object using explicit platform-tuned weights."""

    def __init__(
        self,
        inventory: RewardInventory,
        overlap_evaluator: OverlapEvaluator,
        config: RewardConfig,
        support_evaluator: SupportEvaluator | None = None,
    ) -> None:
        self.inventory = inventory
        self.overlap_evaluator = overlap_evaluator
        self.config = config
        self.support_evaluator = support_evaluator or _contact_support_summary
        # Fixed for this inventory, not recomputed from the remaining stones.
        # One fully contained mean-sized stone contributes one normalized unit.
        volumes = np.asarray([obj.volume for obj in inventory.objects], dtype=float)
        if not len(volumes) or np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
            raise ValueError("reward normalization requires finite positive inventory volumes")
        target_volume = float(inventory.target.volume)
        if not np.isfinite(target_volume) or target_volume <= 0:
            raise ValueError("reward normalization requires a finite positive target volume")
        self.mean_object_volume = float(volumes.mean())
        self.target_overlap_increment_scale = target_volume / self.mean_object_volume

    def compute(self, state: State) -> tuple[float, dict[str, object]]:
        if not state.placed_object_indices or not state.action_history:
            raise ValueError("reward requires one completed placement")

        inventory = self.inventory
        objects = inventory.objects
        inventory_ids = [int(stack_object.object_id) for stack_object in objects]
        if state.object_ids != inventory_ids:
            raise ValueError("state inventory does not match reward inventory")
        target = inventory.target
        overlap_evaluator = self.overlap_evaluator
        sequence = state.placed_object_indices
        action = state.action_history[-1]
        placed_index = sequence[-1]
        placed_id = inventory_ids[placed_index]
        if action.object_index != placed_index or action.object_id not in (
            -1,
            placed_id,
        ):
            raise ValueError("latest action does not match the latest placement")

        displacement = float(action.place_robustness_displacement)
        nonfinite = not np.isfinite(displacement)
        if nonfinite:
            displacement = self.config.nonfinite_displacement
        clipped = False
        if (
            self.config.max_displacement is not None
            and displacement > self.config.max_displacement
        ):
            displacement = self.config.max_displacement
            clipped = True

        previous_overlap = 0.0
        for index in sequence[:-1]:
            stack_object = objects[index]
            pose = state.object_poses.get(
                stack_object.object_id,
                inventory.initial_pose(stack_object.object_id),
            )
            previous_overlap += float(
                overlap_evaluator.target_coverage(stack_object, pose)
            )

        placed_object = objects[placed_index]
        placed_pose = state.object_poses.get(
            placed_object.object_id,
            inventory.initial_pose(placed_object.object_id),
        )
        containment = float(
            overlap_evaluator.object_containment(placed_object, placed_pose)
        )
        placed_overlap = containment * placed_object.volume / target.volume
        target_overlap = previous_overlap + placed_overlap
        overlap_delta = target_overlap - previous_overlap
        overlap_increment = max(overlap_delta, 0.0)
        normalized_overlap_increment = overlap_increment * self.target_overlap_increment_scale

        stability = -min(
            action.c_feq / self.config.posegen_threshold, 1.0
        )
        place_stability = -displacement
        large_object_lower = self._large_object_lower_score(state)
        inward_orientation = self._inward_orientation_score(
            placed_index, placed_pose
        )
        support_count, support_has_ground = self.support_evaluator(
            inventory, state, placed_index, placed_pose
        )
        containment_reward = float(
            np.clip(
                containment - self.config.containment_threshold,
                0.0,
                1.0,
            )
        )

        weights = self.config.weights
        reward = (
            weights.stability * stability
            + weights.object_containment * containment_reward
            + weights.target_overlap_increment * normalized_overlap_increment
            + weights.place_stability * place_stability
            + weights.large_object_lower * large_object_lower
            + weights.inward_orientation * inward_orientation
        )
        return float(reward), {
            "stability": stability,
            "target_IoU": target_overlap,
            "previous_target_IoU": previous_overlap,
            "target_IoU_delta": overlap_delta,
            "target_IoU_increment": overlap_increment,
            "target_IoU_increment_normalized": normalized_overlap_increment,
            "target_overlap_increment_scale": self.target_overlap_increment_scale,
            "mean_object_volume": self.mean_object_volume,
            "target_overlap_increment_reward": weights.target_overlap_increment * normalized_overlap_increment,
            "object_IoU": containment,
            "object_IoU_reward": containment_reward,
            "target_containment_fraction": containment,
            "target_containment_threshold": self.config.containment_threshold,
            "place_stability": place_stability,
            "place_robustness_displacement": displacement,
            "place_robustness_nonfinite": nonfinite,
            "place_robustness_clipped": clipped,
            "large_object_lower": large_object_lower,
            "inward_orientation": inward_orientation,
            "support_count": support_count,
            "support_has_ground": support_has_ground,
        }

    def _large_object_lower_score(self, state: State) -> float:
        objects = self.inventory.objects
        volumes = np.asarray(
            [stack_object.volume for stack_object in objects], dtype=float
        )
        volume_range = float(volumes.max() - volumes.min())
        placed_object = objects[state.placed_object_indices[-1]]
        volume_score = (
            0.5
            if volume_range <= 0.0
            else (placed_object.volume - float(volumes.min())) / volume_range
        )
        pose = state.object_poses.get(
            placed_object.object_id,
            self.inventory.initial_pose(placed_object.object_id),
        )
        target = self.inventory.target
        z = float(pose[2]) - float(target.origin[2])
        lowness = 1.0 - float(
            np.clip(z / max(float(target.height), 1e-6), 0.0, 1.0)
        )
        return float(volume_score * lowness)

    def _inward_orientation_score(
        self, object_index: int, pose: np.ndarray
    ) -> float:
        normals, areas = self.inventory.objects[
            object_index
        ].representative_face_normals()
        if len(normals) == 0:
            return 0.0
        world_normals = np.asarray(normals) @ quaternion_matrix(pose[3:]).T
        normal = exposed_surface_normal(world_normals, areas)
        if normal is None:
            return 0.0

        upward = float(
            np.clip((float(normal[2]) - 0.7) / 0.3, 0.0, 1.0)
        )
        heading = normal[:2]
        heading_norm = float(np.linalg.norm(heading))
        if heading_norm < 0.05:
            inward = 0.5
        else:
            direction = np.asarray(
                self.inventory.target.origin[:2], dtype=float
            ) - np.asarray(pose[:2], dtype=float)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm < 1e-9:
                inward = 0.5
            else:
                minimum = float(np.cos(np.deg2rad(40.0)))
                dot = float(
                    np.dot(heading / heading_norm, direction / direction_norm)
                )
                inward = float(
                    np.clip((dot - minimum) / (1.0 - minimum), 0.0, 1.0)
                )
        return upward * inward


def _contact_support_summary(
    inventory: RewardInventory,
    state: State,
    object_index: int,
    pose: np.ndarray,
) -> tuple[int, bool]:
    del inventory, pose
    sources: set[int | str] = set()
    for contact in state.contact_points:
        first = contact.object_index_first
        second = contact.object_index_second
        if first == object_index:
            if contact.ground_second:
                sources.add("ground")
            elif second is not None:
                sources.add(int(second))
        elif second == object_index:
            if contact.ground_first:
                sources.add("ground")
            elif first is not None:
                sources.add(int(first))
    return len(sources), "ground" in sources
