"""Compose shared search with independently resolved excavator geometry."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from stacking_common.actions.floor import ActiveFloorPolicy, floor_policy_settings
from stacking_common.actions.orientation import InventoryOrientationProvider, PrincipalAxisOrientationProvider
from stacking_common.assets import ObjectAssetLoader
from stacking_common.environment import CoreSimulationProfile, EnvironmentProfile
from stacking_common.actions import CorePosegenProfile
from stacking_common.actions.scored import TargetXYOverlapFilter
from stacking_common.actions.ply_height import PlyHeightInitializer
from stacking_common.planning import PlacementPlanningProblem, SolverProfileFiles
from stacking_common.profiles import (
    ActionGenerationProfile, InventoryProfile, MCTSProfile,
    load_action_generation_profile, load_inventory_profile, load_mcts_profile,
    load_environment_profile, load_posegen_profile, load_simulation_profile,
)
from stacking_common.profiles.planning import PlanningProfile, load_planning_profile
from stacking_common.profiles.native import load_native_solver_profile
from stacking_common.profiles.action import resolve_scored_sampling_settings
from stacking_common.workflows import MCTSPlanningSequenceCoordinator
from stacking_common.workflows.search import StackingSearchApplication


@dataclass(frozen=True)
class ExcavatorSettings:
    pick_ground_height: float
    place_ground_height: float
    handoff_ground_height: float
    place_z_offset: float
    handoff_positions: tuple[tuple[float, float, float], ...]

    def __post_init__(self):
        if not np.all(np.isfinite([self.pick_ground_height, self.place_ground_height,
                                   self.handoff_ground_height, self.place_z_offset])):
            raise ValueError("excavator support heights and lift must be finite")
        if self.place_z_offset < 0:
            raise ValueError("placement lift must be nonnegative")
        positions = np.asarray(self.handoff_positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions) or not np.all(np.isfinite(positions)):
            raise ValueError("handoff positions must be finite XYZ values")
        if not np.allclose(positions[:, 2], self.handoff_ground_height):
            raise ValueError("handoff positions must use the handoff support height")
        object.__setattr__(self, "handoff_positions", tuple(map(tuple, positions)))


@dataclass(frozen=True)
class ExcavatorApplication(StackingSearchApplication):
    repository_root: Path
    solver_profiles: SolverProfileFiles
    environment_profile: EnvironmentProfile
    simulation_profile: CoreSimulationProfile
    posegen_profile: CorePosegenProfile
    inventory_profile: InventoryProfile
    action_generation_profile: ActionGenerationProfile
    mcts_profile: MCTSProfile
    planning_profile: PlanningProfile
    excavator_profile: ExcavatorSettings
    floor_settings: dict
    pose_from_scan: bool
    roll_angles: tuple[float, ...]
    sampling_settings: dict

    @classmethod
    def from_repository(cls, repository, *, mcts_config=None, inventory_name=None,
                        target_shape=None, recorded_target=None):
        root = Path(repository).resolve()
        config = root / "apps/excavator/config"
        solvers = SolverProfileFiles.from_directory(config / "solvers")
        environment = load_environment_profile(config / "environment.yml")
        planning = load_planning_profile(config / "planning.yml")
        values = yaml.safe_load((config / "planning.yml").read_text())["excavator"]
        settings = ExcavatorSettings(**values)
        # Search is local to the stack frame; support height is a world-frame value.
        if not np.isclose(planning.world_T_stack.parent_T_child[2], settings.place_ground_height):
            raise ValueError("world_T_stack Z must equal the placement support height")
        action = yaml.safe_load((config / "action_generation.yml").read_text())
        if "floor" in action or "scored_sampling" in action:
            raise ValueError("excavator sampling settings belong under planar; "
                             "use planar.floor and planar.score")
        planar = action["planar"]
        floor_settings = planar["floor"]
        if "score_weights" in floor_settings or "score_weights" in planar:
            raise ValueError("use named weights under planar.score; "
                             "score_weights is no longer an excavator setting")
        sampling_settings = resolve_scored_sampling_settings(planar)
        scan = action["orientation"]["pose_from_scan"]
        if not isinstance(scan, bool):
            raise TypeError("pose_from_scan must be boolean")
        return cls(root, solvers, environment,
                   load_simulation_profile(solvers.simulation),
                   load_posegen_profile(solvers.posegen),
                   load_inventory_profile(config / "inventory.yml", inventory_name=inventory_name,
                       target_shape=target_shape, recorded_target=recorded_target,
                       recorded_inventory=(recorded_target if recorded_target is not None
                           and "model_directory" in recorded_target else None)),
                   load_action_generation_profile(config / "action_generation.yml",
                       max_poses_per_object=environment.observation.max_poses_per_object,
                       banned_object_ids=environment.observation.banned_object_ids),
                   load_mcts_profile(mcts_config or config / "mcts_cpu.yml"), planning, settings,
                   floor_policy_settings(sampling_settings["floor"]), scan,
                   tuple(np.deg2rad(action["orientation"]["roll_angles_degrees"])),
                   sampling_settings)

    def load_inventory(self, object_ids, initial_poses):
        profile = self.inventory_profile
        unknown = set(object_ids) - set(profile.object_ids)
        if unknown:
            raise ValueError(f"objects outside the excavator catalog: {sorted(unknown)}")
        return ObjectAssetLoader(self.repository_root / profile.model_directory,
                                 profile.density, profile.target.geometry(),
                                 dict(profile.density_overrides)).load(object_ids, initial_poses)

    def build_floor_policy(self, inventory):
        return ActiveFloorPolicy(inventory, **self.floor_settings,
            ground_z=float(self.posegen_profile.objective.get("ground_height", 0.)))

    def build_orientation_provider(self):
        # The scored sampler applies yaw to this base for both support kernels
        # and final proposals. Excavator scan mode retains the measured heading.
        return InventoryOrientationProvider() if self.pose_from_scan else PrincipalAxisOrientationProvider(self.roll_angles)

    def build_orientation_filters(self, floor_policy):
        from stacking_common.actions import ScanOrientationFilter
        return (ScanOrientationFilter(floor_policy),) if self.pose_from_scan else ()

    def build_target_filters(self, inventory, overlap_evaluator):
        return (TargetXYOverlapFilter(),)

    def additional_context_assets(self, inventory):
        directory = self.repository_root / self.inventory_profile.model_directory
        return {f"object_surface/{object_id}": path for object_id in inventory.object_ids
                if (path := directory / f"model_{object_id}_mesh.ply").is_file()}

    def build_height_initializer(self):
        profile = self.action_generation_profile.generator
        return PlyHeightInitializer(self.repository_root / self.inventory_profile.model_directory,
            clearance=profile.initial_height_clearance,
            xy_padding=profile.initial_height_xy_padding,
            ground_z=float(self.posegen_profile.objective.get("ground_height", 0.)))

    def build_placement_planning_problem(self, state, action, *, pick_pose,
                                         pick_scene_poses=None, metadata=None):
        transform = self.planning_profile.world_T_stack
        problem = PlacementPlanningProblem.from_search(
            state, action, pick_pose=pick_pose, world_T_stack=transform,
            pick_scene_poses=pick_scene_poses,
            metadata={**(metadata or {}), "planning_T_search": transform.parent_T_child,
                      "search_frame_id": transform.child_frame,
                      "place_z_offset": self.excavator_profile.place_z_offset})
        lifted = problem.place_pose.copy()
        lifted[2] += self.excavator_profile.place_z_offset
        place_objects = tuple(replace(obj, pose=lifted)
                           if obj.object_id == action.object_id else obj
                           for obj in problem.place_scene_objects)
        return replace(problem, place_pose=lifted, scene_objects=place_objects,
                       place_scene_objects=place_objects)

    def build_native_motion_planner(self, inventory, *, solver_profile=None, core_module=None):
        from .planning.native import ExcavatorNativePlanningBackend
        solver = solver_profile or load_native_solver_profile(self.solver_profiles.grasp, self.solver_profiles.motion)
        backend = ExcavatorNativePlanningBackend(self, inventory, solver, core_module=core_module)
        return backend

    def build_planning_coordinator(self, inventory, *, solver_profile=None,
                                    core_module=None, progress=None, context_sha256=None):
        from .planning.coordinator import ExcavatorPlanningCoordinator
        return ExcavatorPlanningCoordinator(
            self, inventory, self.build_native_motion_planner(inventory,
                solver_profile=solver_profile, core_module=core_module),
            progress=progress, context_sha256=context_sha256)

    def build_planning_sequence(self, inventory, session, request, *, writer=None,
                                progress=None, solver_profile=None, context_sha256=None,
                                checkpoint_builder=None):
        return MCTSPlanningSequenceCoordinator(
            session, request, self.build_planning_coordinator(inventory,
                solver_profile=solver_profile, progress=progress, context_sha256=context_sha256),
            frame_id=self.planning_profile.world_T_stack.child_frame,
            writer=writer, progress=progress, checkpoint_builder=checkpoint_builder,
            committed_state_evaluator=self.build_committed_state_evaluator(inventory))
