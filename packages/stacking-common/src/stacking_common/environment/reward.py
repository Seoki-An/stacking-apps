"""Concrete reward calculation migrated from the tabletop environment."""

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np

from stacking_common.domain import CompositeTargetGeometry, StackObject, State

from ._geometry import exposed_surface_normal, quaternion_matrix
from .overlap import OverlapEvaluator


class RewardTarget(Protocol):
    origin: np.ndarray
    height: float
    volume: float


class RewardInventory(Protocol):
    objects: Sequence[StackObject]
    target: RewardTarget | CompositeTargetGeometry

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
    surface_flatness: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.stability,
                self.object_containment,
                self.target_overlap_increment,
                self.place_stability,
                self.large_object_lower,
                self.inward_orientation,
                self.surface_flatness,
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
    displacement_reference: float = 1.0
    # Recorded in planning contexts so cached returns from older reward
    # definitions cannot silently be reused. This is not a tuning option.
    surface_flatness_metric: str = field(
        default="area_weighted_height_variance", init=False
    )

    def __post_init__(self) -> None:
        if not np.isfinite(self.displacement_reference) or self.displacement_reference <= 0.0:
            raise ValueError("displacement reference must be positive and finite")
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


@dataclass(frozen=True)
class PreparedRewardGeometry:
    """Owned scalar geometry terms for one nominal placement, independent of robustness."""

    previous_overlap: float
    target_overlap: float
    containment: float
    surface: tuple[float, float, float] | None
    previous_surface: tuple[float, float, float] | None


class StackingReward:
    """Evaluate one placed object using explicit platform-tuned weights."""

    def __init__(
        self,
        inventory: RewardInventory,
        overlap_evaluator: OverlapEvaluator,
        config: RewardConfig,
        support_evaluator: SupportEvaluator | None = None,
        *,
        ground_z: float = 0.,
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
        self._composite = isinstance(inventory.target, CompositeTargetGeometry)
        if self._composite and (not hasattr(overlap_evaluator, "target_volume")
                                or not callable(getattr(overlap_evaluator, "scene_coverage", None))):
            raise TypeError("composite rewards require union target_volume and scene_coverage evaluation")
        target_volume = float(overlap_evaluator.target_volume if self._composite else inventory.target.volume)
        if not np.isfinite(target_volume) or target_volume <= 0:
            raise ValueError("reward normalization requires a finite positive target volume")
        self.mean_object_volume = float(volumes.mean())
        self.target_volume = target_volume
        self.target_overlap_increment_scale = target_volume / self.mean_object_volume
        self._surface = None
        if config.weights.surface_flatness != 0.:
            from .surface import SurfaceFlatness
            self._surface = (SurfaceFlatness(inventory, ground_z=ground_z) if self._composite
                             else SurfaceFlatness(inventory))

    def prepare_reward_action(self, parent: State, action):
        """Carry the true pre-placement surface through physics and revalidation.

        Only scalar diagnostics are copied into the candidate. The caller's
        action and parent remain unchanged, and no trajectory capture is needed.
        """
        if self._surface is None:
            return action
        maximum, mean, variance = self._surface.statistics(parent)
        candidate = action.copy()
        candidate.diagnostics.update(
            surface_previous_height_max=maximum,
            surface_previous_height_mean=mean,
            surface_previous_height_variance=variance,
        )
        return candidate

    def prepare_geometry(self, sequence, object_poses) -> PreparedRewardGeometry:
        """Prepare expensive terms from nominal poses; no provisional reward is returned."""
        inventory = self.inventory
        objects = inventory.objects
        overlap_evaluator = self.overlap_evaluator
        placed_index = sequence[-1]
        previous_overlap = 0.0
        previous_placements = []
        for index in sequence[:-1]:
            stack_object = objects[index]
            pose = object_poses.get(
                stack_object.object_id,
                inventory.initial_pose(stack_object.object_id),
            )
            if self._composite:
                previous_placements.append((stack_object, pose))
            else:
                previous_overlap += float(overlap_evaluator.target_coverage(stack_object, pose))

        placed_object = objects[placed_index]
        placed_pose = object_poses.get(
            placed_object.object_id,
            inventory.initial_pose(placed_object.object_id),
        )
        containment = float(
            overlap_evaluator.object_containment(placed_object, placed_pose)
        )
        if self._composite:
            previous_overlap = overlap_evaluator.scene_coverage(previous_placements)
            target_overlap = overlap_evaluator.scene_coverage([*previous_placements, (placed_object, placed_pose)])
        else:
            placed_overlap = containment * placed_object.volume / self.target_volume
            target_overlap = previous_overlap + placed_overlap
        previous_surface = (self._surface.statistics_poses(sequence[:-1], object_poses)
                            if self._surface is not None else None)
        surface = (self._surface.statistics_poses(sequence, object_poses)
                   if self._surface is not None else None)
        return PreparedRewardGeometry(previous_overlap, target_overlap, containment, surface, previous_surface)

    def compute(self, state: State) -> tuple[float, dict[str, object]]:
        return self.compute_prepared(state, None)

    def compute_prepared(self, state: State, geometry: PreparedRewardGeometry | None):
        """Finalize with the actual validation result and unchanged reward arithmetic."""
        if not state.placed_object_indices or not state.action_history:
            raise ValueError("reward requires one completed placement")

        inventory = self.inventory
        objects = inventory.objects
        inventory_ids = [int(stack_object.object_id) for stack_object in objects]
        if state.object_ids != inventory_ids:
            raise ValueError("state inventory does not match reward inventory")
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

        if geometry is None:
            geometry = self.prepare_geometry(sequence, state.object_poses)
        previous_overlap = geometry.previous_overlap
        target_overlap = geometry.target_overlap
        containment = geometry.containment
        placed_object = objects[placed_index]
        placed_pose = state.object_poses.get(
            placed_object.object_id, inventory.initial_pose(placed_object.object_id))
        overlap_delta = target_overlap - previous_overlap
        overlap_increment = max(overlap_delta, 0.0)
        normalized_overlap_increment = overlap_increment * self.target_overlap_increment_scale

        equilibrium = action.diagnostics.get("posegen_equilibrium_residual_normalized")
        legacy_equilibrium = equilibrium is None
        if legacy_equilibrium:
            # Historical artifacts do not contain separate force/torque residuals.
            equilibrium = action.c_feq / self.config.posegen_threshold
        equilibrium = float(equilibrium)
        stability = -min(max(equilibrium, 0.0), 1.0)
        place_stability = -displacement / self.config.displacement_reference
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
        flatness = 0.0
        surface_info = {}
        if self._surface is not None:
            maximum, mean, variance = geometry.surface
            previous_maximum, previous_mean, previous_variance = geometry.previous_surface
            # Old actions recorded max/mean only; those cannot reconstruct a
            # variance. Use the geometric fallback consistently for that case.
            baseline_recorded = "surface_previous_height_variance" in action.diagnostics
            if baseline_recorded:
                previous_maximum = float(action.diagnostics["surface_previous_height_max"])
                previous_mean = float(action.diagnostics["surface_previous_height_mean"])
                previous_variance = float(action.diagnostics["surface_previous_height_variance"])
                if (not np.isfinite([previous_maximum, previous_mean, previous_variance]).all()
                        or previous_variance < 0.
                        or previous_mean < 0.
                        or previous_mean > previous_maximum + 1e-12 * max(1., abs(previous_maximum))):
                    raise ValueError("invalid pre-placement surface baseline")
            reference = float(np.cbrt(self.mean_object_volume))
            relief = max(maximum - mean, 0.)
            previous_relief = max(previous_maximum - previous_mean, 0.)
            flatness = (previous_variance - variance) / reference ** 2
            surface_info = {
                "surface_flatness_metric": self.config.surface_flatness_metric,
                "surface_height_variance": variance,
                "surface_previous_height_variance": previous_variance,
                "surface_height_variance_change": variance - previous_variance,
                "surface_height_variance_normalized": variance / reference ** 2,
                "surface_previous_height_variance_normalized": previous_variance / reference ** 2,
                "surface_height_max": maximum,
                "surface_height_mean": mean,
                "surface_height_relief": relief,
                "surface_height_reference": reference,
                "surface_previous_height_max": previous_maximum,
                "surface_previous_height_mean": previous_mean,
                "surface_previous_height_relief": previous_relief,
                "surface_height_relief_change": relief - previous_relief,
                "surface_baseline_source": "parent_state" if baseline_recorded else "current_poses_without_candidate",
            }
        reward = (
            weights.stability * stability
            + weights.object_containment * containment_reward
            + weights.target_overlap_increment * normalized_overlap_increment
            + weights.place_stability * place_stability
            + weights.large_object_lower * large_object_lower
            + weights.inward_orientation * inward_orientation
            + weights.surface_flatness * flatness
        )
        return float(reward), {
            **surface_info,
            "surface_flatness": flatness,
            "surface_flatness_reward": weights.surface_flatness * flatness,
            "stability": stability,
            "equilibrium_residual": equilibrium,
            "equilibrium_legacy_scaling": legacy_equilibrium,
            "target_IoU": target_overlap,
            "previous_target_IoU": previous_overlap,
            "target_IoU_delta": overlap_delta,
            "target_IoU_increment": overlap_increment,
            "target_IoU_increment_normalized": normalized_overlap_increment,
            "target_overlap_increment_scale": self.target_overlap_increment_scale,
            "target_volume": self.target_volume,
            "target_volume_method": "voxel_union" if self._composite else "analytic",
            "mean_object_volume": self.mean_object_volume,
            "target_overlap_increment_reward": weights.target_overlap_increment * normalized_overlap_increment,
            "object_IoU": containment,
            "object_IoU_reward": containment_reward,
            "target_containment_fraction": containment,
            "target_containment_threshold": self.config.containment_threshold,
            "place_stability": place_stability,
            "place_robustness_displacement": displacement,
            "place_robustness_displacement_reference": self.config.displacement_reference,
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
        origin, height = self._target_reference(pose)
        z = float(pose[2]) - float(origin[2])
        lowness = 1.0 - float(
            np.clip(z / max(float(height), 1e-6), 0.0, 1.0)
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
            origin, _ = self._target_reference(pose)
            direction = np.asarray(origin[:2], dtype=float) - np.asarray(pose[:2], dtype=float)
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

    def _target_reference(self, pose):
        target = self.inventory.target
        if not self._composite:
            return target.origin, target.height
        # A convex component's vertex centroid lies inside it. An aggregate
        # center can lie in a gap/notch and incorrectly reward outward tilts.
        component = min(target.components, key=lambda item: (
            float(item.footprint_distance([pose[:2]])[0]),
            float(np.linalg.norm(item.world_geometry.points.mean(0)[:2] - pose[:2])),
            item.component_id))
        lower, upper = component.bounds
        center = component.world_geometry.points.mean(0)
        return np.r_[center[:2], lower[2]], upper[2] - lower[2]


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
