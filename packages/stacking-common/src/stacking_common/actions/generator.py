"""Environment-independent coordinator for placement action generation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Generic, TypeVar

from stacking_common.domain import Action, ObjectInventory, State
from stacking_common.search.types import (
    ActionCandidate,
    ActionGenerationRequest,
    ActionGenerationResult,
)

from .config import PlacementActionGeneratorConfig
from .contracts import (
    PlacementFilter,
    PlacementInitializer,
    PlacementPoseSolver,
    PlanarPoseSampler,
)
from .filtering import PosegenFeasibilityFilter
from .types import PlanarPlacement


ObservationT = TypeVar("ObservationT")


class PlacementActionGenerator(Generic[ObservationT]):
    """Build bounded common actions from injected proposal and solve services."""

    def __init__(
        self,
        inventory: ObjectInventory,
        config: PlacementActionGeneratorConfig,
        planar_sampler: PlanarPoseSampler[ObservationT],
        initializer: PlacementInitializer,
        pose_solver: PlacementPoseSolver,
        filters: Sequence[PlacementFilter] = (),
        worker_pose_solvers: Sequence[PlacementPoseSolver] = (),
    ) -> None:
        self.inventory = inventory
        self.config = config
        self.planar_sampler = planar_sampler
        self.initializer = initializer
        self.pose_solver = pose_solver
        self.pose_solvers = (pose_solver, *tuple(worker_pose_solvers))
        self._executor: ThreadPoolExecutor | None = None
        self.filters = (
            PosegenFeasibilityFilter(
                config.posegen_gap_threshold,
                config.posegen_equilibrium_threshold,
                require_solver_convergence=config.require_solver_convergence,
                reject_equilibrium_at_threshold=config.reject_equilibrium_at_threshold,
            ),
            *tuple(filters),
        )

    def generate(
        self,
        state: State,
        observation: ObservationT,
        request: ActionGenerationRequest,
    ) -> ActionGenerationResult:
        return self.generate_batch((state,), (observation,), (request,))[0]

    def generate_batch(
        self,
        states: Sequence[State],
        observations: Sequence[ObservationT],
        requests: Sequence[ActionGenerationRequest],
    ) -> tuple[ActionGenerationResult, ...]:
        """Generate for several nodes using one shared pose-solver work queue."""

        states = tuple(states)
        observations = tuple(observations)
        requests = tuple(requests)
        if not (len(states) == len(observations) == len(requests)):
            raise ValueError("batched action-generation inputs must align")
        if not states:
            return ()

        selected_by_request = []
        exhausted_by_request = []
        jobs = []
        for request_index, (state, observation, request) in enumerate(
            zip(states, observations, requests)
        ):
            self._validate_state(state)
            proposals = self._proposal_stream(
                state,
                observation,
                request.sampling_priority,
            )
            start = min(request.attempt_offset, len(proposals))
            selected = tuple(
                proposals[start : start + request.max_candidates]
            )
            selected_by_request.append(selected)
            exhausted_by_request.append(start + len(selected) >= len(proposals))
            jobs.extend(
                (request_index, state, object_index, proposal)
                for object_index, proposal in selected
            )

        built = self._build_jobs(jobs)
        candidates_by_request: list[list[ActionCandidate]] = [
            [] for _ in states
        ]
        rejections_by_request = [Counter() for _ in states]
        for (request_index, *_), (candidate, rejection_reason) in zip(
            jobs, built
        ):
            if rejection_reason is not None:
                rejections_by_request[request_index][rejection_reason] += 1
            elif candidate is not None:
                candidates_by_request[request_index].append(candidate)

        return tuple(
            ActionGenerationResult(
                candidates=tuple(candidates),
                attempts=len(selected),
                exhausted=exhausted,
                rejection_counts=rejections,
            )
            for candidates, selected, exhausted, rejections in zip(
                candidates_by_request,
                selected_by_request,
                exhausted_by_request,
                rejections_by_request,
            )
        )

    def build_candidate(
        self,
        state: State,
        object_index: int,
        proposal: PlanarPlacement,
    ) -> tuple[ActionCandidate | None, str | None]:
        """Initialize, solve, and filter one explicit planar proposal."""

        return self._build_candidate_with_solver(
            self.pose_solver, state, object_index, proposal
        )

    def build_candidates(
        self,
        state: State,
        proposals: Sequence[tuple[int, PlanarPlacement]],
    ) -> tuple[tuple[ActionCandidate | None, str | None], ...]:
        """Build an ordered batch across independent pose-solver workers."""

        jobs = tuple(
            (0, state, object_index, proposal)
            for object_index, proposal in proposals
        )
        return self._build_jobs(jobs)

    def _build_jobs(self, jobs):
        """Solve ordered, potentially multi-state jobs on owned workers."""

        jobs = tuple(jobs)
        if len(self.pose_solvers) == 1 or len(jobs) <= 1:
            return tuple(
                self._build_candidate_with_solver(
                    self.pose_solver, state, object_index, proposal
                )
                for _, state, object_index, proposal in jobs
            )

        buckets = [[] for _ in self.pose_solvers]
        for index, (_, state, object_index, proposal) in enumerate(jobs):
            buckets[index % len(buckets)].append(
                (index, state, object_index, proposal)
            )

        def build_bucket(solver, bucket):
            return [
                (
                    index,
                    self._build_candidate_with_solver(
                        solver, state, object_index, proposal
                    ),
                )
                for index, state, object_index, proposal in bucket
            ]

        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=len(self.pose_solvers))
        futures = [
            self._executor.submit(build_bucket, solver, bucket)
            for solver, bucket in zip(self.pose_solvers, buckets)
            if bucket
        ]
        results = [item for future in futures for item in future.result()]
        results.sort(key=lambda item: item[0])
        return tuple(result for _, result in results)

    def close(self) -> None:
        """Release the lazily-created pose-generation worker pool."""

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _build_candidate_with_solver(
        self,
        pose_solver: PlacementPoseSolver,
        state: State,
        object_index: int,
        proposal: PlanarPlacement,
    ) -> tuple[ActionCandidate | None, str | None]:
        self._validate_state(state)
        if object_index in state.placed_object_indices:
            return None, "object_already_placed"
        stack_object = self.inventory[object_index]
        if stack_object.object_id in self.config.banned_object_ids:
            return None, "banned_object"
        initial = self.initializer.initialize(
            self.inventory, state, object_index, proposal
        )
        solved = pose_solver.solve(self.inventory, state, object_index, initial)
        filter_diagnostics: dict[str, object] = {}
        for placement_filter in self.filters:
            decision = placement_filter.evaluate(
                self.inventory,
                state,
                object_index,
                initial,
                solved,
            )
            filter_diagnostics.update(decision.diagnostics)
            if not decision.accepted:
                return None, decision.reason

        diagnostics = dict(proposal.diagnostics)
        diagnostics.update(initial.diagnostics)
        diagnostics.update(solved.diagnostics)
        diagnostics.update(filter_diagnostics)
        diagnostics["planar_score"] = proposal.score
        action = Action(
            object_index=object_index,
            object_id=stack_object.object_id,
            pose=solved.pose,
            init_pose=initial.pose,
            solved_pose=solved.pose,
            c_feq=solved.c_feq,
            c_gap=solved.c_gap,
            diagnostics=diagnostics,
        )
        return ActionCandidate(action=action, diagnostics=diagnostics), None

    def _proposal_stream(
        self,
        state: State,
        observation: ObservationT,
        sampling_priority: str | None,
    ) -> list[tuple[int, PlanarPlacement]]:
        available = [
            index
            for index, object_id in enumerate(state.object_ids)
            if index not in state.placed_object_indices
            and object_id not in self.config.banned_object_ids
        ]
        by_object = {
            index: tuple(
                self.planar_sampler.sample(
                    self.inventory,
                    state,
                    observation,
                    index,
                    self.config.max_poses_per_object,
                    sampling_priority,
                )
            )
            for index in available
        }
        stream: list[tuple[int, PlanarPlacement]] = []
        for pose_index in range(self.config.max_poses_per_object):
            for object_index in available:
                proposals = by_object[object_index]
                if pose_index < len(proposals):
                    stream.append((object_index, proposals[pose_index]))
        return stream

    def _validate_state(self, state: State) -> None:
        if tuple(state.object_ids) != self.inventory.object_ids:
            raise ValueError("state inventory does not match action inventory")
