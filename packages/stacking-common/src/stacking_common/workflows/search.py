"""Shared composition of native placement search workflows."""

from collections.abc import Callable, Mapping, Sequence

from stacking_common.actions import (
    ActiveFloorFilter,
    CEMRootActionGenerator,
    CorePlacementPoseSolver,
    HeightMapPlanarPoseSampler,
    MeshHeightInitializer,
    ObjectOrientationProvider,
    PlacementActionGenerator,
    PlacementFilter,
    PlacementPoseSolver,
    TargetContainmentFilter,
)
from stacking_common.domain import ObjectInventory, State
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


class StackingSearchApplication:
    """Compose search from platform profiles and explicit geometry policies."""

    def build_core_backend(
        self,
        inventory: ObjectInventory,
        fidelity: EvaluationFidelity = EvaluationFidelity.LONG,
    ) -> CorePlacementBackend:
        """Build the platform stacking-core placement backend."""

        fidelity = EvaluationFidelity(fidelity)
        profile = (
            self.short_simulation_profile
            if fidelity is EvaluationFidelity.SHORT
            else self.simulation_profile
        )
        return CorePlacementBackend.from_profile(
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
        if self.posegen_profile.backend == "cuda":
            from stacking_common.actions.cuda_solver import CudaPlacementPoseSolver
            solver_type = CudaPlacementPoseSolver
        return solver_type(
            inventory,
            self.posegen_profile,
            core_module=core_module,
        )

    def build_search_environments(
        self,
        inventory: ObjectInventory,
    ) -> dict[EvaluationFidelity, StackingEnvironment[Observation]]:
        """Build independent short-search and long-validation environments."""

        return {
            fidelity: self.build_environment(
                inventory,
                self.build_core_backend(inventory, fidelity),
            )
            for fidelity in (
                EvaluationFidelity.SHORT,
                EvaluationFidelity.LONG,
            )
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

        if environments is not None and transition_evaluator is not None:
            raise ValueError(
                "provide MCTS environments or a transition evaluator, not both"
            )
        profile = self.mcts_profile
        worker_count = profile.parallel.effective_workers
        if transition_evaluator is None:
            if environments is not None or worker_count == 1:
                transition_evaluator = StackingEnvironmentEvaluator(
                    environments or self.build_search_environments(inventory)
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
                1 if self.posegen_profile.backend == "cuda" else worker_count
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
                generator,
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
        """Build independent short/long simulator pairs for sibling batches."""

        if worker_count <= 0:
            raise ValueError("MCTS transition worker count must be positive")
        return ParallelTransitionEvaluator(
            [
                StackingEnvironmentEvaluator(
                    self.build_search_environments(inventory)
                )
                for _ in range(worker_count)
            ],
            max_workers=worker_count,
        )

    def default_mcts_request(self) -> MCTSSearchRequest:
        """Return the immutable platform search budget and fidelity schedule."""

        return self.mcts_profile.request

    def build_environment(
        self,
        inventory: ObjectInventory,
        backend: PlacementBackend,
    ) -> StackingEnvironment[Observation]:
        """Compose the common environment around platform dependencies."""

        observation_builder, reward_evaluator = self._build_state_evaluators(inventory)
        return StackingEnvironment(
            SimulationTransition(inventory.object_ids, backend),
            observation_builder,
            reward_evaluator,
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
        )
        return observation_builder, reward_evaluator

    def build_action_generator(
        self,
        inventory: ObjectInventory,
        pose_solver: PlacementPoseSolver,
        orientation_provider: ObjectOrientationProvider[Observation] | None = None,
        worker_pose_solvers: Sequence[PlacementPoseSolver] = (),
        filters: Sequence[PlacementFilter] = (),
    ) -> PlacementActionGenerator[Observation]:
        """Build the new common generator around the selected pose solver."""

        profile = self.action_generation_profile
        floor_policy = self.build_floor_policy(inventory)
        return PlacementActionGenerator(
            inventory=inventory,
            config=profile.generator,
            planar_sampler=HeightMapPlanarPoseSampler(
                profile.planar,
                orientation_provider=orientation_provider,
                floor_policy=floor_policy,
            ),
            initializer=MeshHeightInitializer(
                clearance=profile.generator.initial_height_clearance,
                xy_padding=profile.generator.initial_height_xy_padding,
            ),
            pose_solver=pose_solver,
            worker_pose_solvers=worker_pose_solvers,
            filters=(
                *((ActiveFloorFilter(floor_policy),) if profile.strict_floor_first else ()),
                *self.build_orientation_filters(floor_policy),
                *filters,
            ),
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
