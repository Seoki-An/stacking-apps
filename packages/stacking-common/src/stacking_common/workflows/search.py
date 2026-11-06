"""Shared composition of native placement search workflows."""

from collections.abc import Callable, Mapping, Sequence

from stacking_common.actions import (
    CEMRootActionGenerator,
    CorePlacementPoseSolver,
    MeshHeightInitializer,
    ObjectOrientationProvider,
    PlacementActionGenerator,
    PlacementFilter,
    PlacementPoseSolver,
    TargetContainmentFilter,
)
from stacking_common.actions.scored import ScoredPlanarPoseSampler, ScoredPlacementActionGenerator
from stacking_common.domain import CompositeTargetGeometry, ObjectInventory, State
from stacking_common.environment import (
    ArrayObservationBuilder,
    CorePlacementBackend,
    InventoryObservationAdapter,
    Observation,
    PlacementBackend,
    SimulationTransition,
    StackingEnvironment,
    StackingReward,
    VoxelOverlapEvaluator,
)
from stacking_common.search import (
    EvaluationFidelity,
    HorizonValueEstimator,
    MCTSSearchRequest,
    MonteCarloTreeSearch,
    ParallelTransitionEvaluator,
    PlacementPriorScorer,
    StackingEnvironmentEvaluator,
)

from stacking_common.visualization.heightmap import HeightMapRenderer


def require_search_target(target, *, ground_z: float = 0.) -> None:
    """Validate composite support assumptions before allocating native workers.

    Components may overlap or be disconnected, but must intersect the shared
    physical ground plane over an area. Elevated regions need a separate
    support policy; accepting them here would silently omit their proposals.
    """

    if isinstance(target, CompositeTargetGeometry):
        from stacking_common.actions.target_regions import TargetRegions

        TargetRegions(target, ground_z=ground_z)


class StackingSearchApplication:
    """Compose search from platform profiles and explicit geometry policies."""

    def build_core_backend(
        self,
        inventory: ObjectInventory,
        fidelity: EvaluationFidelity = EvaluationFidelity.LONG,
    ) -> CorePlacementBackend:
        """Build the platform stacking-core placement backend."""

        fidelity = EvaluationFidelity(fidelity)
        profile = self.simulation_profile
        backend_type = CorePlacementBackend
        if self.mcts_profile.parallel.simulation_backend == "cuda":
            from stacking_common.environment.cuda_backend import CudaPlacementBackend
            backend_type = CudaPlacementBackend
        return backend_type.from_profile(
            inventory.models,
            profile,
            place_stability=self.environment_profile.place_stability,
        )

    def build_pose_solver(
        self,
        inventory: ObjectInventory,
        *,
        core_module=None,
    ) -> CorePlacementPoseSolver:
        """Build native pose generation for the platform target and objects."""

        solver_type = CorePlacementPoseSolver
        options = {}
        if self.mcts_profile.parallel.posegen_backend == "cuda":
            from stacking_common.actions.cuda_solver import CudaPlacementPoseSolver
            solver_type = CudaPlacementPoseSolver
            options["capacity"] = self.mcts_profile.parallel.posegen_capacity
        return solver_type(
            inventory,
            self.posegen_profile,
            core_module=core_module,
            max_initial_xy_displacement_ratio=self.action_generation_profile.generator.max_initial_xy_displacement_ratio,
            **options,
        )

    def build_search_environments(
        self,
        inventory: ObjectInventory,
    ) -> dict[EvaluationFidelity, StackingEnvironment[Observation]]:
        """Build the single settling-then-full-simulation environment."""

        require_search_target(inventory.target,
            ground_z=float(self.posegen_profile.objective.get("ground_height", 0.)))
        return {
            fidelity: self.build_environment(
                inventory,
                self.build_core_backend(inventory, fidelity),
                defer_observations=True,
                result_workers=(self.mcts_profile.parallel.effective_workers
                                if self.mcts_profile.parallel.simulation_backend == "cuda" else 1),
            )
            for fidelity in (EvaluationFidelity.LONG,)
        }

    def build_mcts(
        self,
        inventory: ObjectInventory,
        environments: Mapping[
            EvaluationFidelity,
            StackingEnvironment[Observation],
        ] | None = None,
        *,
        transition_evaluator=None,
        pose_solver: PlacementPoseSolver | None = None,
        orientation_provider: ObjectOrientationProvider[Observation] | None = None,
        prior_scorer=None,
        value_estimator=None,
        rng=None,
    ) -> MonteCarloTreeSearch[Observation]:
        """Compose the native common platform MCTS path."""

        require_search_target(inventory.target,
            ground_z=float(self.posegen_profile.objective.get("ground_height", 0.)))
        if environments is not None and transition_evaluator is not None:
            raise ValueError(
                "provide MCTS environments or a transition evaluator, not both"
            )
        profile = self.mcts_profile
        worker_count = profile.parallel.effective_workers
        if transition_evaluator is None:
            cuda_simulation = profile.parallel.simulation_backend == "cuda"
            if environments is not None or worker_count == 1 or cuda_simulation:
                transition_evaluator = StackingEnvironmentEvaluator(
                    environments or self.build_search_environments(inventory),
                    connectivity_check=self.build_connectivity_check(inventory),
                    allow_root_jumps=self.root_jumps_enabled(),
                    initial_xy_displacement_filter=self.build_initial_xy_displacement_filter(inventory),
                )
            else:
                transition_evaluator = self.build_parallel_transition_evaluator(
                    inventory, worker_count
                )
        reward = self.environment_profile.reward
        overlap_evaluator = (
            VoxelOverlapEvaluator(
                inventory.target,
                resolution=self.environment_profile.overlap_resolution,
            )
            if isinstance(inventory, ObjectInventory)
            else None
        )
        solvers = (
            [pose_solver]
            if pose_solver is not None
            else [self.build_pose_solver(inventory) for _ in range(
                1 if self.mcts_profile.parallel.posegen_backend == "cuda" else worker_count
            )]
        )
        generator = self.build_action_generator(
            inventory,
            solvers[0],
            orientation_provider=(
                self.build_orientation_provider()
                if orientation_provider is None
                else orientation_provider
            ),
            worker_pose_solvers=solvers[1:],
            filters=(
                ()
                if overlap_evaluator is None
                else (
                    *self.build_target_filters(inventory, overlap_evaluator),
                )
            ),
        )
        if prior_scorer is None:
            if overlap_evaluator is None:
                raise TypeError("native MCTS requires an ObjectInventory")
            prior = PlacementPriorScorer(
                inventory,
                overlap_evaluator,
                posegen_threshold=reward.posegen_threshold,
                stability_weight=reward.weights.stability,
                containment_weight=reward.weights.object_containment,
            )
        else:
            prior = prior_scorer
        root_generator = generator
        if self.root_jumps_enabled():
            root_generator = self.build_action_generator(
                inventory, solvers[0],
                orientation_provider=orientation_provider or self.build_orientation_provider(),
                worker_pose_solvers=solvers[1:],
                filters=(() if overlap_evaluator is None else self.build_target_filters(inventory, overlap_evaluator)),
                root=True)
        value = value_estimator or HorizonValueEstimator(
            profile.bootstrap_mean_reward,
            profile.bootstrap_horizon,
            profile.tree.discount,
        )
        return MonteCarloTreeSearch(
            profile.tree,
            generator,
            prior,
            transition_evaluator,
            value,
            root_action_generator=CEMRootActionGenerator(
                root_generator,
                prior,
                profile.root_cem,
            ),
            rng=rng,
        )

    def build_parallel_transition_evaluator(
        self,
        inventory: ObjectInventory,
        worker_count: int,
    ) -> ParallelTransitionEvaluator[Observation]:
        """Build independent simulators for sibling batches."""

        if worker_count <= 0:
            raise ValueError("MCTS transition worker count must be positive")
        return ParallelTransitionEvaluator(
            [
                StackingEnvironmentEvaluator(
                    self.build_search_environments(inventory),
                    connectivity_check=self.build_connectivity_check(inventory),
                    allow_root_jumps=self.root_jumps_enabled(),
                    initial_xy_displacement_filter=self.build_initial_xy_displacement_filter(inventory),
                )
                for _ in range(worker_count)
            ],
            max_workers=worker_count,
        )

    def default_mcts_request(self) -> MCTSSearchRequest:
        """Return the immutable platform search budget and fidelity schedule."""

        return self.mcts_profile.request

    def build_connectivity_check(self, inventory):
        if self.sampling_settings.get("connectivity", "active_floor") != "previous_action":
            return None
        from stacking_common.actions.connectivity import PreviousActionConnectivity
        return PreviousActionConnectivity(inventory, self.sampling_settings["contact_gap_tolerance"])

    def build_initial_xy_displacement_filter(self, inventory):
        from stacking_common.actions.filtering import InitialXYDisplacementFilter
        ratio = self.action_generation_profile.generator.max_initial_xy_displacement_ratio
        return None if ratio is None else InitialXYDisplacementFilter(inventory, ratio)

    def root_jumps_enabled(self):
        return (self.sampling_settings.get("connectivity", "active_floor") == "previous_action"
                and self.sampling_settings.get("root_global_fraction", 0.) > 0.)

    def build_environment(
        self,
        inventory: ObjectInventory,
        backend: PlacementBackend,
        *,
        result_workers: int = 1,
        defer_observations: bool = False,
    ) -> StackingEnvironment[Observation]:
        """Compose the common environment around platform dependencies."""

        if result_workers < 1:
            raise ValueError("result worker count must be positive")
        observation_builder, reward_evaluator = self._build_state_evaluators(inventory)
        return StackingEnvironment(
            SimulationTransition(inventory.object_ids, backend),
            observation_builder,
            reward_evaluator,
            defer_observations=defer_observations,
            worker_evaluators=[self._build_state_evaluators(inventory)
                               for _ in range(result_workers - 1)],
        )

    def build_committed_state_evaluator(
        self,
        inventory: ObjectInventory,
    ) -> Callable[[State], tuple[Observation, float, Mapping[str, object]]]:
        """Observe and score an approved scene without moving it in simulation."""

        observation_builder, reward_evaluator = self._build_state_evaluators(inventory)

        def evaluate(state: State):
            reward, info = reward_evaluator.compute(state)
            return observation_builder.build(state), float(reward), info

        return evaluate

    def _build_state_evaluators(self, inventory: ObjectInventory):
        """Use identical observation and reward profiles for search and commit."""

        profile = self.environment_profile
        renderer = HeightMapRenderer(
            inventory,
            resolution=profile.observation.height_map_resolution,
            margin=profile.height_map_margin,
        )
        observation_source = InventoryObservationAdapter(
            inventory,
            surface_sample_count=profile.surface_sample_count,
            height_map_renderer=renderer,
        )
        observation_builder = ArrayObservationBuilder(
            observation_source,
            profile.observation,
        )
        overlap_evaluator = VoxelOverlapEvaluator(
            inventory.target,
            resolution=profile.overlap_resolution,
        )
        reward_evaluator = StackingReward(
            inventory,
            overlap_evaluator,
            profile.reward,
            ground_z=float(self.posegen_profile.objective.get("ground_height", 0.)),
        )
        return observation_builder, reward_evaluator

    def build_action_generator(
        self,
        inventory: ObjectInventory,
        pose_solver: PlacementPoseSolver,
        orientation_provider: ObjectOrientationProvider[Observation] | None = None,
        worker_pose_solvers: Sequence[PlacementPoseSolver] = (),
        filters: Sequence[PlacementFilter] = (),
        *, root: bool = False,
    ) -> PlacementActionGenerator[Observation]:
        """Build both platforms' scored proposals around the selected solver."""

        profile = self.action_generation_profile
        floor_policy = self.build_floor_policy(inventory)
        layer_filters = ()
        if profile.generator.reject_above_active_layer:
            from stacking_common.actions.filtering import ActiveLayerAdmissionFilter
            layer_filters = (ActiveLayerAdmissionFilter(floor_policy),)
        sampler = ScoredPlanarPoseSampler
        if root and self.root_jumps_enabled():
            from stacking_common.actions.root_sampling import RootMixturePlanarSampler
            sampler = RootMixturePlanarSampler
        return ScoredPlacementActionGenerator(
            inventory=inventory,
            config=profile.generator,
            planar_sampler=sampler(
                inventory, profile.planar,
                orientation_provider or self.build_orientation_provider(),
                self.sampling_settings,
                seed=self.mcts_profile.root_cem.seed,
                floor_policy=floor_policy,
            ),
            initializer=self.build_height_initializer(),
            pose_solver=pose_solver,
            worker_pose_solvers=worker_pose_solvers,
            filters=(
                *layer_filters,
                *self.build_orientation_filters(floor_policy),
                *filters,
            ),
        )

    def build_height_initializer(self):
        profile = self.action_generation_profile.generator
        return MeshHeightInitializer(
            clearance=profile.initial_height_clearance,
            xy_padding=profile.initial_height_xy_padding,
            ground_z=float(self.posegen_profile.objective.get("ground_height", 0.)),
        )


    def build_floor_policy(self, inventory):
        raise NotImplementedError

    def build_target_filters(self, inventory, overlap_evaluator):
        return (TargetContainmentFilter(overlap_evaluator,
                    self.environment_profile.reward.containment_threshold),)

    def build_orientation_provider(self):
        raise NotImplementedError

    def build_orientation_filters(self, floor_policy):
        raise NotImplementedError
