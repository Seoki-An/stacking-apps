"""Tabletop application composition root."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stacking_common.actions.orientation import PrincipalAxisOrientationProvider

from stacking_common.actions import (
    CorePosegenProfile,
    ScanAlignedOrientationProvider,
    ScanOrientationFilter,
)
from stacking_common.domain import ObjectInventory
from stacking_common.environment import (
    CorePlacementBackend,
    CoreSimulationProfile,
    EnvironmentProfile,
    Observation,
)
from stacking_common.planning import PlacementPlanningProblem, SolverProfileFiles
from stacking_common.workflows import MCTSPlanningSequenceCoordinator
from stacking_common.pointcloud import MultiViewVoxelMerge, VoxelMergeConfig

from stacking_common.workflows.search import StackingSearchApplication

from .config import (
    TabletopActionGenerationProfile,
    TabletopInventoryProfile,
    TabletopMCTSProfile,
    TabletopPerceptionProfile,
    TabletopPlanningProfile,
    load_action_generation_profile,
    load_environment_profile,
    load_inventory_profile,
    load_mcts_profile,
    load_perception_profile,
    load_planning_profile,
    load_posegen_profile,
    load_simulation_profile,
)
from .environment import TabletopObjectAssetLoader
from .perception import (
    DualFemtoFusion,
    Open3dRegistrationBackend,
    TabletopDsfExporter,
    TabletopCaptureBuffer,
    TabletopFieldMatcher,
    TabletopFieldSegmenter,
    TabletopObjectRegistrar,
    TabletopPcdCatalog,
    TabletopMeshReconstructor,
    TabletopPoseGraphRefiner,
)
from .planning import (
    TabletopMotionPlannerAdapter,
    TabletopNativePlanningBackend,
    TabletopPlacementSafety,
    TabletopPlanningCoordinator,
    load_native_solver_profile,
)


@dataclass(frozen=True)
class TabletopApplication(StackingSearchApplication):
    """Top-level tabletop dependencies assembled at the platform boundary."""

    solver_profiles: SolverProfileFiles
    environment_profile: EnvironmentProfile
    simulation_profile: CoreSimulationProfile
    short_simulation_profile: CoreSimulationProfile
    posegen_profile: CorePosegenProfile
    mcts_profile: TabletopMCTSProfile
    inventory_profile: TabletopInventoryProfile
    perception_profile: TabletopPerceptionProfile
    action_generation_profile: TabletopActionGenerationProfile
    planning_profile: TabletopPlanningProfile
    repository_root: Path

    @classmethod
    def from_repository(
        cls, repository: str | Path, *, mcts_config: str | Path | None = None
    ) -> "TabletopApplication":
        root = Path(repository)
        profiles = SolverProfileFiles.from_directory(
            root / "apps" / "tabletop" / "config" / "solvers"
        )
        environment_profile_path = (
            root / "apps" / "tabletop" / "config" / "environment.yml"
        )
        inventory_profile_path = (
            root / "apps" / "tabletop" / "config" / "inventory.yml"
        )
        perception_profile_path = (
            root / "apps" / "tabletop" / "config" / "perception.yml"
        )
        environment_profile = load_environment_profile(
            environment_profile_path
        )
        action_generation_profile = load_action_generation_profile(
            root / "apps" / "tabletop" / "config" / "action_generation.yml",
            max_poses_per_object=(
                environment_profile.observation.max_poses_per_object
            ),
            banned_object_ids=environment_profile.observation.banned_object_ids,
        )
        mcts_profile_path = (
            Path(mcts_config)
            if mcts_config is not None
            else root / "apps" / "tabletop" / "config" / "mcts_cpu.yml"
        )
        return cls(
            solver_profiles=profiles,
            environment_profile=environment_profile,
            simulation_profile=load_simulation_profile(profiles.simulation),
            short_simulation_profile=load_simulation_profile(
                profiles.simulation,
                fidelity="short",
            ),
            posegen_profile=load_posegen_profile(profiles.posegen),
            mcts_profile=load_mcts_profile(mcts_profile_path),
            inventory_profile=load_inventory_profile(
                inventory_profile_path
            ),
            perception_profile=load_perception_profile(
                perception_profile_path
            ),
            action_generation_profile=action_generation_profile,
            planning_profile=load_planning_profile(
                root / "apps" / "tabletop" / "config" / "planning.yml"
            ),
            repository_root=root.resolve(),
        )

    def build_placement_planning_problem(
        self,
        state,
        action,
        *,
        pick_pose: np.ndarray,
        pick_scene_poses: Mapping[int, np.ndarray] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> PlacementPlanningProblem:
        """Move one stack-local search choice across the planning boundary."""

        world_T_stack = self.planning_profile.world_T_stack
        details = dict(metadata or {})
        details.update(
            {
                "search_frame_id": world_T_stack.child_frame,
                "planning_T_search": world_T_stack.parent_T_child.copy(),
            }
        )
        return PlacementPlanningProblem.from_search(
            state,
            action,
            pick_pose=pick_pose,
            world_T_stack=world_T_stack,
            pick_scene_poses=pick_scene_poses,
            metadata=details,
        )

    def build_motion_planner(self, solve) -> TabletopMotionPlannerAdapter:
        """Adapt the platform's native pick-place solve callback."""

        return TabletopMotionPlannerAdapter(solve)

    def build_stacking_core_motion_planner(
        self,
        problem_builder,
        *,
        config=None,
        core_module=None,
    ) -> TabletopMotionPlannerAdapter:
        """Compose the nanobind planner once its native types are exported."""

        return TabletopMotionPlannerAdapter.from_stacking_core(
            problem_builder,
            config=config,
            core_module=core_module,
        )

    def build_native_motion_planner(
        self,
        inventory: ObjectInventory,
        *,
        core_module=None,
    ) -> TabletopMotionPlannerAdapter:
        """Build the configured UR5e/SR-gripper native planning boundary."""

        solver = load_native_solver_profile(
            self.solver_profiles.grasp,
            self.solver_profiles.motion,
        )
        backend = TabletopNativePlanningBackend(
            inventory,
            self.planning_profile,
            solver,
            self.repository_root,
            core_module=core_module,
        )
        return TabletopMotionPlannerAdapter(backend.solve)

    def build_planning_coordinator(
        self,
        inventory: ObjectInventory,
        *,
        core_module=None,
        progress=None,
        solver_profile=None,
        context_sha256=None,
    ) -> TabletopPlanningCoordinator:
        """Compose native planning with common problem and artifact contracts."""

        solver = solver_profile or load_native_solver_profile(
            self.solver_profiles.grasp,
            self.solver_profiles.motion,
        )
        native_backend = TabletopNativePlanningBackend(
            inventory,
            self.planning_profile,
            solver,
            self.repository_root,
            core_module=core_module,
        )
        placement_backend = CorePlacementBackend.from_profile(
            inventory.models,
            self.short_simulation_profile,
            place_stability=self.environment_profile.place_stability,
            core_module=core_module,
        )
        placement_backend.reset(inventory.object_ids)

        def build_problem(state, action, *, pick_pose, metadata):
            pick_scene_poses = {
                object_id: inventory.initial_pose(object_id)
                for index, object_id in enumerate(state.object_ids)
                if index not in state.placed_object_indices
            }
            return self.build_placement_planning_problem(
                state,
                action,
                pick_pose=pick_pose,
                pick_scene_poses=pick_scene_poses,
                metadata=metadata,
            )

        return TabletopPlanningCoordinator(
            inventory,
            build_problem,
            TabletopMotionPlannerAdapter(native_backend.solve),
            placement_backend=placement_backend,
            construction_filter=self.build_construction_filter(inventory),
            safety=(
                self.build_floor_policy(inventory)
                if self.action_generation_profile.use_floor_filter
                else None
            ),
            world_T_stack=self.planning_profile.world_T_stack,
            progress=progress,
            context_sha256=context_sha256,
        )

    def build_planning_sequence(
        self,
        inventory: ObjectInventory,
        session,
        request,
        *,
        writer=None,
        core_module=None,
        progress=None,
        solver_profile=None,
        context_sha256=None,
        checkpoint_builder=None,
    ) -> MCTSPlanningSequenceCoordinator:
        """Compose reusable search retries with tabletop motion planning."""

        return MCTSPlanningSequenceCoordinator(
            session,
            request,
            self.build_planning_coordinator(
                inventory,
                core_module=core_module,
                progress=progress,
                solver_profile=solver_profile,
                context_sha256=context_sha256,
            ),
            frame_id=self.planning_profile.world_T_stack.child_frame,
            writer=writer,
            progress=progress,
            checkpoint_builder=checkpoint_builder,
            committed_state_evaluator=self.build_committed_state_evaluator(inventory),
        )

    def load_inventory(
        self,
        object_ids: Sequence[int],
        initial_poses: Mapping[int, np.ndarray],
    ) -> ObjectInventory:
        """Load configured tabletop object assets and initial poses."""

        profile = self.inventory_profile
        resolved_ids = tuple(int(object_id) for object_id in object_ids)
        unknown_ids = sorted(set(resolved_ids) - set(profile.object_ids))
        if unknown_ids:
            raise ValueError(
                f"objects are not in the tabletop catalog: {unknown_ids}"
            )
        loader = TabletopObjectAssetLoader(
            self.repository_root / profile.model_directory,
            density=profile.density,
            density_overrides=dict(profile.density_overrides),
            target=profile.target.geometry(),
        )
        return loader.load(resolved_ids, initial_poses)

    def build_pcd_catalog(self) -> TabletopPcdCatalog:
        """Build the app-owned measured model-cloud catalog."""

        profile = self.inventory_profile
        return TabletopPcdCatalog(
            self.repository_root / profile.model_directory,
            profile.object_ids,
        )

    def build_field_registrar(self) -> TabletopObjectRegistrar:
        """Build registration for field-scan model identification."""

        return TabletopObjectRegistrar(
            self.build_pcd_catalog(),
            Open3dRegistrationBackend(
                self.perception_profile.field_registration
            ),
            self.perception_profile.field_quality,
        )

    def build_in_hand_registrar(self) -> TabletopObjectRegistrar:
        """Build independently tuned in-hand object registration."""

        return TabletopObjectRegistrar(
            self.build_pcd_catalog(),
            Open3dRegistrationBackend(
                self.perception_profile.in_hand_registration
            ),
            self.perception_profile.in_hand_quality,
        )

    def build_field_segmenter(self) -> TabletopFieldSegmenter:
        return TabletopFieldSegmenter(self.perception_profile.segmentation)

    def build_field_matcher(self) -> TabletopFieldMatcher:
        return TabletopFieldMatcher(
            self.build_pcd_catalog(),
            Open3dRegistrationBackend(
                self.perception_profile.field_registration
            ),
            self.perception_profile.field_quality,
            self.perception_profile.assignment,
        )

    def build_dual_femto_fusion(self) -> DualFemtoFusion:
        return DualFemtoFusion(
            self.perception_profile.dual_femto_fusion,
            self.perception_profile.segmentation,
        )

    def build_primary_capture_buffer(self) -> TabletopCaptureBuffer:
        return TabletopCaptureBuffer(
            self.perception_profile.primary_camera,
            self.perception_profile.capture,
        )

    def build_reconstruction_merger(self) -> MultiViewVoxelMerge:
        profile = self.perception_profile.reconstruction
        return MultiViewVoxelMerge(
            VoxelMergeConfig(
                profile.merge_voxel_size,
                profile.minimum_view_support,
            )
        )

    def build_pose_graph_refiner(self) -> TabletopPoseGraphRefiner:
        return TabletopPoseGraphRefiner(
            self.perception_profile.reconstruction.pose_graph
        )

    def build_mesh_reconstructor(self) -> TabletopMeshReconstructor:
        return TabletopMeshReconstructor(
            self.perception_profile.reconstruction.mesh
        )

    def build_dsf_exporter(self) -> TabletopDsfExporter:
        return TabletopDsfExporter(
            self.perception_profile.reconstruction.dsf
        )


    def build_floor_policy(self, inventory):
        return TabletopPlacementSafety(inventory, **self.floor_settings)

    @property
    def floor_settings(self):
        from stacking_common.actions.floor import floor_policy_settings
        return floor_policy_settings(self.sampling_settings["floor"])

    @property
    def sampling_settings(self):
        return self.action_generation_profile.sampling_settings

    def build_orientation_provider(self):
        return (ScanAlignedOrientationProvider() if self.action_generation_profile.pose_from_scan
                else PrincipalAxisOrientationProvider((0., np.pi)))

    def build_orientation_filters(self, floor_policy):
        return (ScanOrientationFilter(floor_policy),) if self.action_generation_profile.pose_from_scan else ()
