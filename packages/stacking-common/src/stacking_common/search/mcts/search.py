"""Environment-independent Monte Carlo tree-search coordination."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field, replace
from typing import Generic, TypeVar

import numpy as np

from stacking_common.errors import BackendCapacityError
from stacking_common.domain import Action
from stacking_common.search.contracts import (
    ActionGenerator,
    PriorScorer,
    TransitionEvaluator,
    ValueEstimator,
)
from stacking_common.search.types import (
    ActionCandidate,
    ActionGenerationRequest,
    EvaluationFidelity,
    ExecutionRejections,
)

from .backup import backpropagate
from .config import MCTSConfig
from .node import MCTSNode
from .tree_policy import can_expand, expansion_room, select_child


ObservationT = TypeVar("ObservationT")


@dataclass(frozen=True)
class MCTSSearchRequest:
    """Runtime controls for one bounded MCTS invocation."""

    iterations: int
    evaluation_fidelity: EvaluationFidelity = EvaluationFidelity.LONG
    epsilon: float = 0.0
    reset_statistics: bool = True
    root_sampling_priority: str | None = None
    deeper_sampling_priority: str | None = None
    max_time_seconds: float | None = None
    execution_rejections: ExecutionRejections = field(
        default_factory=ExecutionRejections
    )
    seed_actions: tuple[Action, ...] = ()

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("MCTS search iterations must be positive")
        if not math.isfinite(self.epsilon) or not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("MCTS search epsilon must be between zero and one")
        if self.max_time_seconds is not None and (
            not math.isfinite(self.max_time_seconds)
            or self.max_time_seconds <= 0.0
        ):
            raise ValueError("MCTS maximum search time must be positive")
        object.__setattr__(
            self, "evaluation_fidelity", EvaluationFidelity(self.evaluation_fidelity)
        )
        object.__setattr__(self, "epsilon", float(self.epsilon))
        object.__setattr__(self, "reset_statistics", bool(self.reset_statistics))
        if not isinstance(self.execution_rejections, ExecutionRejections):
            raise TypeError("MCTS execution rejections must be ExecutionRejections")
        seeds = tuple(self.seed_actions)
        if not all(isinstance(action, Action) for action in seeds):
            raise TypeError("MCTS seed actions must be Action values")
        object.__setattr__(
            self,
            "seed_actions",
            tuple(action.copy() for action in seeds),
        )
        object.__setattr__(
            self,
            "root_sampling_priority",
            _optional_priority(self.root_sampling_priority, "root"),
        )
        object.__setattr__(
            self,
            "deeper_sampling_priority",
            _optional_priority(self.deeper_sampling_priority, "deeper"),
        )


@dataclass(frozen=True)
class SubtreeRevalidationResult:
    """Accounting for authoritative checks of preserved root children."""

    attempted: int = 0
    accepted: int = 0
    rejected: int = 0
    cleared_subtrees: int = 0
    timed_out: bool = False


@dataclass(frozen=True)
class RootValidationResult:
    """Accounting for authoritative root admission before lookahead."""

    attempted: int = 0
    accepted: int = 0
    rejected: int = 0
    timed_out: bool = False


@dataclass(frozen=True)
class RootLookaheadResult:
    """Coverage of authoritative roots by at least one descendant attempt."""

    eligible: int = 0
    attempted: int = 0
    generated: int = 0
    evaluated: int = 0


@dataclass(frozen=True)
class RootRetryResult:
    """Diversified retries after authoritative root rejection."""

    rounds: int = 0
    retired: int = 0
    generated: int = 0
    accepted: int = 0


@dataclass(frozen=True)
class FloorFillFallbackResult:
    """Outcome of the final sampling fallback (legacy result field name)."""

    attempted: bool = False
    pools: int = 0
    generated: int = 0
    accepted: int = 0
    selected: bool = False

    policy: str = "corner_seed"


@dataclass(frozen=True)
class MCTSSearchResult(Generic[ObservationT]):
    """Tree and compact accounting produced by one MCTS invocation."""

    root: MCTSNode[ObservationT]
    selected: MCTSNode[ObservationT] | None
    completed_iterations: int
    expansion_calls: int
    generated_candidates: int
    evaluated_candidates: int
    failed_evaluations: int
    effective_max_depth: int
    ranked_candidates: tuple[MCTSNode[ObservationT], ...] = ()
    timed_out: bool = False
    # Legacy result columns retained for existing artifact/consumer compatibility.
    final_validation_attempts: int = 0
    final_validation_failures: int = 0
    final_validation_timed_out: bool = False
    short_simulation_batches: int = 0
    short_simulation_jobs: int = 0
    short_parallel_batches: int = 0
    short_parallel_jobs: int = 0
    long_simulation_batches: int = 0
    long_simulation_jobs: int = 0
    short_max_batch_size: int = 0
    max_evaluation_batch_size: int = 0
    transition_workers: int = 1
    pose_generation_seconds: float = 0.0
    pose_generation_batches: int = 0
    pose_generation_node_requests: int = 0
    pose_generation_max_node_batch: int = 0
    descendant_frontier_batches: int = 0
    descendant_frontier_iterations: int = 0
    short_simulation_seconds: float = 0.0
    long_simulation_seconds: float = 0.0
    short_settling_job_seconds: float = 0.0
    short_structural_job_seconds: float = 0.0
    long_settling_job_seconds: float = 0.0
    long_structural_job_seconds: float = 0.0
    revalidation: SubtreeRevalidationResult = SubtreeRevalidationResult()
    root_validation: RootValidationResult = RootValidationResult()
    root_lookahead: RootLookaheadResult = RootLookaheadResult()
    root_retry: RootRetryResult = RootRetryResult()
    floor_fill_fallback: FloorFillFallbackResult = FloorFillFallbackResult()
    descendant_validation_attempts: int = 0
    descendant_validation_failures: int = 0
    final_selection_max_weight: float = 0.5
    seeded_candidates: int = 0
    execution_rejected_candidates: int = 0
    backend_failure: dict[str, object] | None = None

    @property
    def failure_reason(self):
        if self.backend_failure is not None:
            return self.backend_failure["reason"]
        return "search_exhausted" if self.selected is None else None


@dataclass
class _SearchCounters:
    generation_deadline: float | None = None
    completed_iterations: int = 0
    expansion_calls: int = 0
    generated_candidates: int = 0
    evaluated_candidates: int = 0
    failed_evaluations: int = 0
    short_simulation_batches: int = 0
    short_simulation_jobs: int = 0
    short_parallel_batches: int = 0
    short_parallel_jobs: int = 0
    long_simulation_batches: int = 0
    long_simulation_jobs: int = 0
    short_max_batch_size: int = 0
    max_evaluation_batch_size: int = 0
    root_validation_attempts: int = 0
    root_validation_failures: int = 0
    root_validation_timed_out: bool = False
    root_lookahead_eligible: int = 0
    root_lookahead_attempts: int = 0
    root_lookahead_generated: int = 0
    root_lookahead_evaluated: int = 0
    root_retry_rounds: int = 0
    root_retry_retired: int = 0
    root_retry_generated: int = 0
    root_retry_accepted: int = 0
    seeded_candidates: int = 0
    execution_rejected_candidates: int = 0
    pose_generation_seconds: float = 0.0
    pose_generation_batches: int = 0
    pose_generation_node_requests: int = 0
    pose_generation_max_node_batch: int = 0
    descendant_frontier_batches: int = 0
    descendant_frontier_iterations: int = 0
    short_simulation_seconds: float = 0.0
    long_simulation_seconds: float = 0.0
    short_settling_job_seconds: float = 0.0
    short_structural_job_seconds: float = 0.0
    long_settling_job_seconds: float = 0.0
    long_structural_job_seconds: float = 0.0


class MonteCarloTreeSearch(Generic[ObservationT]):
    """Coordinate MCTS through injected, framework-independent services."""

    def __init__(
        self,
        config: MCTSConfig,
        action_generator: ActionGenerator[ObservationT],
        prior_scorer: PriorScorer[ObservationT],
        transition_evaluator: TransitionEvaluator[ObservationT],
        value_estimator: ValueEstimator[ObservationT],
        *,
        root_action_generator: ActionGenerator[ObservationT] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config
        self.action_generator = action_generator
        self.root_action_generator = root_action_generator
        self.prior_scorer = prior_scorer
        self.transition_evaluator = transition_evaluator
        self.value_estimator = value_estimator
        self.rng = rng if rng is not None else random.Random()

    def search(
        self,
        root: MCTSNode[ObservationT],
        request: MCTSSearchRequest,
    ) -> MCTSSearchResult[ObservationT]:
        """Run a single-fidelity search, then rank its existing evidence."""

        if not root.evaluated:
            raise ValueError("MCTS search root must contain an evaluated state")
        effective_depth = self.config.effective_max_depth(
            len(root.state.placed_object_indices)
        )
        if effective_depth != self.config.max_depth:
            bounded = MonteCarloTreeSearch(
                replace(self.config, max_depth=effective_depth),
                self.action_generator,
                self.prior_scorer,
                self.transition_evaluator,
                self.value_estimator,
                root_action_generator=self.root_action_generator,
                rng=self.rng,
            )
            return bounded.search(root, request)
        deadline = (
            None
            if request.max_time_seconds is None
            else time.monotonic() + request.max_time_seconds
        )
        iteration_deadline = deadline
        counters = _SearchCounters(generation_deadline=iteration_deadline)
        root.detach()
        begin_evaluation = getattr(self.transition_evaluator, "begin_search", None)
        if callable(begin_evaluation):
            begin_evaluation(root.state)
        self._apply_execution_rejections(root, request, counters)
        selected, ranked_candidates = None, ()
        fallback = FloorFillFallbackResult()
        backend_failure = None
        try:
            self._seed_root(root, request, counters)
            if request.reset_statistics:
                root.reset_statistics(recursive=True)
            reset_generator_statistics = getattr(
                self.root_action_generator, "reset_statistics", None
            )
            if callable(reset_generator_statistics):
                reset_generator_statistics()

            self._prepare_root(
                root,
                request,
                counters,
                iteration_deadline,
                reseed=request.reset_statistics,
            )
            for child in root.evaluated_children():
                if not child.children:
                    child.info.pop("root_lookahead_attempted", None)
            counters.root_lookahead_eligible = len(
                self._root_lookahead_eligible_children(root)
            )
            self._run_search_iterations(root, request, counters)

            selected, ranked_candidates = self._final_select(root)
            excluded_actions = [
                child.action
                for child in root.children
                if child.action is not None
            ]
            for _ in range(self.config.rejected_root_retry_rounds):
                if selected is not None:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    break
                retired = root.retire_failed_children()
                cem = getattr(self.root_action_generator, "last_statistics", None)
                fresh_remaining = cem is not None and not cem.fresh_exhausted
                if not retired and not fresh_remaining:
                    break
                if not retired:
                    root.expansion_attempts = 0
                    root.expansion_exhausted = False
                counters.generation_deadline = deadline
                counters.root_retry_rounds += 1
                counters.root_retry_retired += len(retired)
                begin_retry = getattr(self.root_action_generator, "begin_retry", None)
                if callable(begin_retry):
                    begin_retry(excluded_actions)
                generated_before = counters.generated_candidates
                accepted_before = counters.root_validation_attempts - (
                    counters.root_validation_failures
                )
                self._prepare_root(
                    root,
                    request,
                    counters,
                    deadline,
                )
                counters.root_lookahead_eligible += len(
                    self._root_lookahead_eligible_children(root)
                )
                counters.root_retry_generated += (
                    counters.generated_candidates - generated_before
                )
                accepted_after = counters.root_validation_attempts - (
                    counters.root_validation_failures
                )
                counters.root_retry_accepted += max(
                    accepted_after - accepted_before,
                    0,
                )
                self._run_search_iterations(root, request, counters)
                excluded_actions.extend(retired)
                excluded_actions.extend(
                    child.action
                    for child in root.children
                    if child.action is not None
                )
                selected, ranked_candidates = self._final_select(root)
            if selected is None:
                selected, ranked_candidates, fallback = self._run_corner_seed_fallback(
                    root, request, counters, deadline
                )
        except BackendCapacityError as error:
            # Resource exhaustion is independent of sampling retries or floor
            # relaxation. Preserve the partial tree and stop this search now.
            selected, ranked_candidates = None, ()
            backend_failure = dict(error.diagnostics)
        deadline_exceeded = (
            deadline is not None and time.monotonic() >= deadline
        )
        return MCTSSearchResult(
            root=root,
            selected=selected,
            completed_iterations=counters.completed_iterations,
            expansion_calls=counters.expansion_calls,
            generated_candidates=counters.generated_candidates,
            evaluated_candidates=counters.evaluated_candidates,
            failed_evaluations=counters.failed_evaluations,
            effective_max_depth=self.config.max_depth,
            ranked_candidates=ranked_candidates,
            floor_fill_fallback=fallback,
            backend_failure=backend_failure,
            timed_out=backend_failure is None and (
                counters.root_validation_timed_out
                or deadline_exceeded
            ),
            short_simulation_batches=counters.short_simulation_batches,
            short_simulation_jobs=counters.short_simulation_jobs,
            short_parallel_batches=counters.short_parallel_batches,
            short_parallel_jobs=counters.short_parallel_jobs,
            long_simulation_batches=counters.long_simulation_batches,
            long_simulation_jobs=counters.long_simulation_jobs,
            short_max_batch_size=counters.short_max_batch_size,
            max_evaluation_batch_size=counters.max_evaluation_batch_size,
            transition_workers=int(
                getattr(self.transition_evaluator, "worker_count", 1)
            ),
            pose_generation_seconds=counters.pose_generation_seconds,
            pose_generation_batches=counters.pose_generation_batches,
            pose_generation_node_requests=(
                counters.pose_generation_node_requests
            ),
            pose_generation_max_node_batch=(
                counters.pose_generation_max_node_batch
            ),
            descendant_frontier_batches=counters.descendant_frontier_batches,
            descendant_frontier_iterations=(
                counters.descendant_frontier_iterations
            ),
            short_simulation_seconds=counters.short_simulation_seconds,
            long_simulation_seconds=counters.long_simulation_seconds,
            short_settling_job_seconds=counters.short_settling_job_seconds,
            short_structural_job_seconds=counters.short_structural_job_seconds,
            long_settling_job_seconds=counters.long_settling_job_seconds,
            long_structural_job_seconds=counters.long_structural_job_seconds,
            root_validation=RootValidationResult(
                attempted=counters.root_validation_attempts,
                accepted=(
                    counters.root_validation_attempts
                    - counters.root_validation_failures
                ),
                rejected=counters.root_validation_failures,
                timed_out=counters.root_validation_timed_out,
            ),
            root_lookahead=RootLookaheadResult(
                eligible=counters.root_lookahead_eligible,
                attempted=counters.root_lookahead_attempts,
                generated=counters.root_lookahead_generated,
                evaluated=counters.root_lookahead_evaluated,
            ),
            root_retry=RootRetryResult(
                rounds=counters.root_retry_rounds,
                retired=counters.root_retry_retired,
                generated=counters.root_retry_generated,
                accepted=counters.root_retry_accepted,
            ),
            final_selection_max_weight=self.config.final_selection.max_weight,
            seeded_candidates=counters.seeded_candidates,
            execution_rejected_candidates=(
                counters.execution_rejected_candidates
            ),
        )

    def _run_search_iterations(
        self, root: MCTSNode, request: MCTSSearchRequest, counters: _SearchCounters,
    ) -> None:
        """Search each admitted pool using the remaining shared budget."""
        while counters.completed_iterations < request.iterations:
            # Root admission has finished. An empty/rejected pool has no
            # evidence to back up; leave its budget for retries or fallback.
            if not root.selectable_children():
                break
            if (
                counters.generation_deadline is not None
                and time.monotonic() >= counters.generation_deadline
            ):
                break
            coverage = self._root_lookahead_candidates(root)
            if not coverage:
                consumed = self._run_descendant_frontier_batch(
                    root,
                    request,
                    counters,
                    request.iterations - counters.completed_iterations,
                )
                if consumed > 0:
                    counters.completed_iterations += consumed
                else:
                    if (
                        counters.generation_deadline is not None
                        and time.monotonic() >= counters.generation_deadline
                    ):
                        break
                    if not self._run_iteration(root, request, counters):
                        break
                    counters.completed_iterations += 1
            else:
                consumed = self._run_root_lookahead_batch(
                    coverage[: request.iterations - counters.completed_iterations],
                    request,
                    counters,
                )
                if consumed <= 0:
                    break
                counters.completed_iterations += consumed

    def _run_corner_seed_fallback(self, root, request, counters, deadline):
        """Retry empty-scene corner seeds within the original deadline.

        After the first pool, up to rejected_root_retry_rounds fresh pools run,
        each excluding earlier candidates, until one yields a selection.
        """
        empty = root.state is not None and not root.state.placed_object_indices
        root_generator = getattr(self.root_action_generator, "generator", self.action_generator)
        relax = getattr(root_generator, "corner_seed_fallback", None)
        policy = "corner_seed"
        if (not empty or not callable(relax)
                or (deadline is not None and time.monotonic() >= deadline)):
            return None, (), FloorFillFallbackResult()
        with relax() as enabled:
            if not enabled:
                return None, (), FloorFillFallbackResult()
            # Rebuild the empty root with ordinary proposals instead of corner seeds.
            root.clear_children()
            begin_retry = getattr(self.root_action_generator, "begin_retry", None)
            counters.generation_deadline = deadline
            excluded = []
            selected, ranked, pools, generated, accepted = None, (), 0, 0, 0
            for pool in range(self.config.rejected_root_retry_rounds + 1):
                if pool:
                    if deadline is not None and time.monotonic() >= deadline:
                        break
                    retired = root.retire_failed_children()
                    cem = getattr(self.root_action_generator, "last_statistics", None)
                    if not retired and (cem is None or cem.fresh_exhausted):
                        break
                    if not retired:
                        root.expansion_attempts = 0
                        root.expansion_exhausted = False
                if callable(begin_retry):
                    begin_retry(tuple(excluded))
                pools += 1
                generated_before = counters.generated_candidates
                accepted_before = counters.root_validation_attempts - counters.root_validation_failures
                self._prepare_root(root, request, counters, deadline)
                generated += counters.generated_candidates - generated_before
                accepted += (counters.root_validation_attempts - counters.root_validation_failures
                             - accepted_before)
                counters.root_lookahead_eligible += len(self._root_lookahead_eligible_children(root))
                self._run_search_iterations(root, request, counters)
                excluded.extend(child.action for child in root.children if child.action is not None)
                selected, ranked = self._final_select(root)
                if selected is not None:
                    break
            return selected, ranked, FloorFillFallbackResult(
                attempted=True, pools=pools, generated=generated, accepted=accepted,
                selected=selected is not None, policy=policy,
            )

    def close(self) -> None:
        """Release worker pools and environments owned by search collaborators."""

        closed: set[int] = set()
        for collaborator in (
            self.action_generator,
            self.root_action_generator,
            self.transition_evaluator,
        ):
            if collaborator is None or id(collaborator) in closed:
                continue
            close = getattr(collaborator, "close", None)
            if callable(close):
                close()
            closed.add(id(collaborator))

    def _prepare_root(
        self,
        root,
        request,
        counters,
        deadline,
        *,
        reseed=False,
    ) -> None:
        """Generate, simulate, and seed every available root candidate once."""

        while can_expand(root, self.config):
            if deadline is not None and time.monotonic() >= deadline:
                counters.root_validation_timed_out = True
                break
            before = root.expansion_attempts
            self._expand(
                root,
                counters,
                request.root_sampling_priority,
                request,
            )
            if root.expansion_attempts == before:
                break
        pending = [
            child
            for child in root.selectable_children()
            if not child.evaluated
        ]
        self._validate_root_nodes(
            root,
            pending,
            request,
            counters,
            deadline,
        )
        for child in root.evaluated_children():
            if reseed or child.visits == 0:
                self._back_up(child)

    def _root_lookahead_eligible_children(self, root):
        if not self.config.root_lookahead or self.config.max_depth <= 1:
            return []
        if can_expand(root, self.config):
            return []
        return [
            child
            for child in root.evaluated_children()
            if not child.terminal
            and child.depth < self.config.max_depth
        ]

    def _root_lookahead_candidates(self, root):
        return sorted(
            (
                child
                for child in self._root_lookahead_eligible_children(root)
                if not bool(child.info.get("root_lookahead_attempted", False))
                and not child.children
            ),
            key=lambda child: child.prior,
            reverse=True,
        )

    def _run_root_lookahead_batch(
        self,
        children,
        request,
        counters,
    ) -> int:
        """Evaluate independent root continuations in one worker-sized batch."""

        generated_before = counters.generated_candidates
        evaluated_before = counters.evaluated_candidates
        evaluation_limit = self.config.evaluation_batch_size
        selected_children = list(children[:evaluation_limit])
        for child in selected_children:
            child.info["root_lookahead_attempted"] = True
            counters.root_lookahead_attempts += 1

        completed = self._run_continuation_batch(selected_children, request, counters)
        counters.root_lookahead_generated += (
            counters.generated_candidates - generated_before
        )
        counters.root_lookahead_evaluated += (
            counters.evaluated_candidates - evaluated_before
        )
        return completed

    def _run_continuation_batch(self, children, request, counters) -> int:
        """Expand and evaluate one level beneath independent parent nodes."""
        selected_children = list(children)
        evaluation_limit = self.config.evaluation_batch_size
        targets = self._allocate_generation_slots(
            selected_children,
            evaluation_limit - sum(
                len(child.pending_children()) for child in selected_children
            ),
        )
        self._expand_many(
            list(targets),
            counters,
            request.deeper_sampling_priority,
            request,
            new_candidate_targets=targets,
        )
        # A native solve may finish after the deadline. Keep its pending
        # candidates, but do not launch another simulation batch or back up
        # unevaluated placements after the search budget is spent.
        if (counters.generation_deadline is not None
                and time.monotonic() >= counters.generation_deadline):
            return 0

        selected_nodes = []
        for child in selected_children:
            selected = select_child(
                child,
                self.config,
                epsilon=request.epsilon,
                rng=self.rng,
            )
            selected_nodes.append((child, selected))

        evaluation_nodes = [
            selected
            for _, selected in selected_nodes
            if selected is not None and not selected.evaluated
        ]
        selected_ids = {id(node) for node in evaluation_nodes}
        for child, _ in selected_nodes:
            if len(evaluation_nodes) >= evaluation_limit:
                break
            for sibling in child.pending_children():
                if id(sibling) in selected_ids:
                    continue
                evaluation_nodes.append(sibling)
                selected_ids.add(id(sibling))
                if len(evaluation_nodes) >= evaluation_limit:
                    break

        if evaluation_nodes:
            self._evaluate_many(
                evaluation_nodes,
                request.evaluation_fidelity,
                counters,
            )
        for child, selected in selected_nodes:
            if selected is None:
                self._back_up(child)
            else:
                self._back_up(selected)
        return len(selected_children)

    def _run_descendant_frontier_batch(
        self,
        root,
        request,
        counters,
        remaining_iterations,
    ) -> int:
        """Select a PMBS-style batch globally, then expand/evaluate/back up.

        Reservations exist only in this call, never in persistent tree statistics.
        Pending actions are unique. Before poses exist, an expansion reserves at
        most its current generation allowance; multiple slots may share a state.
        """

        limit = min(remaining_iterations, self.config.evaluation_batch_size)
        if (
            limit < 2
            or self.config.max_depth <= 1
            or root.terminal
            or root.failed
            or root.pending_children()
            or can_expand(root, self.config)
        ):
            return 0

        virtual_visits = {}
        unavailable = set()
        expansions = {}
        reservations = []
        for _ in range(limit):
            if (
                counters.generation_deadline is not None
                and time.monotonic() >= counters.generation_deadline
            ):
                break
            selected = self._select_batch_frontier(
                root, request, virtual_visits, unavailable, expansions
            )
            if selected is None:
                break
            node, needs_expansion = selected
            reservations.append(selected)
            if needs_expansion:
                expansions[node] = expansions.get(node, 0) + 1
            else:
                unavailable.add(node)
            ancestor = node
            while ancestor is not None:
                virtual_visits[ancestor] = virtual_visits.get(ancestor, 0) + 1
                ancestor = ancestor.parent

        if not reservations:
            return 0
        # Pending siblings consume spare evaluation capacity before new poses.
        # Retain each state's reservations, then allocate only unused slots.
        pending = {
            child
            for node, needs_expansion in reservations
            if not needs_expansion and node.parent is not None
            for child in node.parent.pending_children()
        }
        targets = self._allocate_generation_slots(
            list(expansions),
            self.config.evaluation_batch_size
            - len(pending) - sum(expansions.values()),
            reserved=expansions,
        )
        self._expand_many(
            list(targets), counters, request.deeper_sampling_priority, request,
            new_candidate_targets=targets,
        )
        # A native solve may finish after the deadline. Keep its pending
        # candidates, but do not launch another simulation batch or back up
        # unevaluated placements after the search budget is spent.
        if (counters.generation_deadline is not None
                and time.monotonic() >= counters.generation_deadline):
            return 0

        selected_nodes = []
        selected_set = set()
        for node, needs_expansion in reservations:
            if needs_expansion:
                pending = [
                    child
                    for child in node.pending_children()
                    if child not in selected_set
                ]
                if pending:
                    node = select_child(
                        node,
                        self.config,
                        epsilon=request.epsilon,
                        rng=self.rng,
                        excluded=selected_set,
                    )
                elif node.selectable_children() or node in selected_set:
                    # Rejected proposals do not create fictitious evaluations
                    # or repeated backups for unfilled expansion slots.
                    continue
            selected_nodes.append(node)
            selected_set.add(node)

        # As in serial search, spare workers may evaluate pending siblings;
        # only reserved selections contribute visits and completed iterations.
        evaluation_nodes = [node for node in selected_nodes if not node.evaluated]
        evaluation_set = set(evaluation_nodes)
        for selected in selected_nodes:
            if len(evaluation_nodes) >= self.config.evaluation_batch_size:
                break
            if selected.parent is None:
                continue
            for sibling in selected.parent.pending_children():
                if sibling in evaluation_set:
                    continue
                evaluation_nodes.append(sibling)
                evaluation_set.add(sibling)
                if len(evaluation_nodes) >= self.config.evaluation_batch_size:
                    break

        if evaluation_nodes:
            self._evaluate_many(evaluation_nodes, request.evaluation_fidelity, counters)

        # All reservations disappear here (and on any exception) without undoing
        # edits to real visits/rewards: the selection overlay is local only.
        for node in selected_nodes:
            self._back_up(node)
        if selected_nodes:
            counters.descendant_frontier_batches += 1
            counters.descendant_frontier_iterations += len(selected_nodes)
        return len(selected_nodes)

    def _select_batch_frontier(
        self, root, request, virtual_visits, unavailable, expansions
    ):
        """Traverse from the root, skipping work already reserved this round."""

        node = root
        while True:
            if (
                not node.evaluated
                or node.terminal
                or node.depth >= self.config.max_depth
            ):
                return node, False
            if not node.pending_children() and can_expand(node, self.config):
                capacity = min(
                    expansion_room(node, self.config),
                    self.config.generation_batch_size,
                )
                if expansions.get(node, 0) < capacity:
                    return node, True
            child = select_child(
                node,
                self.config,
                epsilon=request.epsilon,
                rng=self.rng,
                virtual_visits=virtual_visits,
                excluded=unavailable,
            )
            if child is not None:
                node = child
            elif node is root:
                return None
            elif not node.selectable_children() and node not in expansions:
                return node, False
            else:
                # All work below this node is already reserved. Reconsider the
                # ancestor using global PUCT rather than imposing branch quotas.
                unavailable.add(node)
                node = node.parent

    def _run_root_lookahead_iteration(
        self,
        child,
        request,
        counters,
    ) -> None:
        """Compatibility wrapper for one root-lookahead iteration."""

        self._run_root_lookahead_batch([child], request, counters)

    def _back_up(self, node) -> None:
        bootstrap = 0.0
        if (node.evaluated and not node.terminal and not node.failed
                and node.depth < self.config.max_depth):
            estimate_at_depth = getattr(
                self.value_estimator,
                "estimate_at_depth",
                None,
            )
            if callable(estimate_at_depth):
                remaining_depth = max(
                    self.config.max_depth - node.depth,
                    0,
                )
                bootstrap = float(
                    estimate_at_depth(
                        node.state,
                        node.observation,
                        remaining_depth,
                    )
                )
            else:
                bootstrap = float(
                    self.value_estimator.estimate(node.state, node.observation)
                )
            if not math.isfinite(bootstrap):
                raise ValueError("value estimator returned a non-finite value")
            # backpropagate discounts the future estimate once at the leaf:
            # r_leaf + mean_reward * sum(gamma**i, i=1..remaining_depth).
        backpropagate(node, self.config, bootstrap_value=bootstrap)

    def _run_iteration(self, root, request, counters) -> bool:
        node = root
        while True:
            if node.terminal or node.failed:
                break
            if not node.pending_children() and can_expand(node, self.config):
                priority = (
                    request.root_sampling_priority
                    if node.depth == 0
                    else request.deeper_sampling_priority
                )
                self._expand(node, counters, priority, request)
            if (counters.generation_deadline is not None
                    and time.monotonic() >= counters.generation_deadline):
                return False
            child = select_child(
                node, self.config, epsilon=request.epsilon, rng=self.rng
            )
            if child is None:
                break
            if not child.evaluated:
                fidelity = request.evaluation_fidelity
                batch = [child]
                batch.extend(
                    sibling
                    for sibling in node.pending_children()
                    if sibling is not child
                )
                self._evaluate_many(
                    batch[: self.config.evaluation_batch_size], fidelity, counters
                )
                node = child
                break
            node = child

        self._back_up(node)
        return True

    def _expand(self, node, counters, sampling_priority, request) -> None:
        self._expand_many(
            [node], counters, sampling_priority, request
        )

    def _allocate_generation_slots(self, nodes, spare_slots, *, reserved=None):
        """Share evaluation slots without exceeding the widening allowance."""

        targets = dict(reserved or {})
        capacities = {
            node: min(
                expansion_room(node, self.config), self.config.generation_batch_size
            )
            for node in nodes if can_expand(node, self.config)
        }
        while spare_slots > 0:
            available = [
                node for node, capacity in capacities.items()
                if targets.get(node, 0) < capacity
            ]
            if not available:
                break
            for node in available:
                targets[node] = targets.get(node, 0) + 1
                spare_slots -= 1
                if spare_slots == 0:
                    break
        return targets

    def _expand_many(
        self,
        nodes,
        counters,
        sampling_priority,
        request,
        *,
        new_candidate_targets=None,
    ) -> None:
        """Generate poses, optionally bounded by this batch's accepted demand.

        Targets count new children, not existing selectable children. Rejections
        may be retried within the existing round, attempt and deadline limits.
        Root admission and serial expansion keep their usual widening policy.
        """

        nodes = list(nodes)
        if not nodes:
            return
        if any(node.state is None for node in nodes):
            raise ValueError("cannot expand an MCTS node without state")
        generators = [
            (
                self.root_action_generator
                if node.depth == 0 and self.root_action_generator is not None
                else self.action_generator
            )
            for node in nodes
        ]
        if any(generator is not generators[0] for generator in generators[1:]):
            for node in nodes:
                self._expand_many(
                    [node], counters, sampling_priority, request,
                    new_candidate_targets=new_candidate_targets,
                )
            return
        generator = generators[0]
        initial_children = {node: len(node.children) for node in nodes}
        desired = {
            node: min(
                self.config.min_feasible_children,
                self.config.hard_child_limit(node.depth),
            )
            for node in nodes
        }
        stopped = set()
        for batch_index in range(self.config.expansion_retry_batches):
            if (
                counters.generation_deadline is not None
                and time.monotonic() >= counters.generation_deadline
            ):
                break
            remaining = (
                None if new_candidate_targets is None else {
                    node: new_candidate_targets[node] - (
                        len(node.children) - initial_children[node]
                    )
                    for node in nodes
                }
            )
            active = [
                node
                for node in nodes
                if node not in stopped
                and can_expand(node, self.config)
            ]
            if remaining is not None:
                # Exhausted states cannot use their outstanding slots. Reassign
                # those slots within this selected frontier, without exceeding
                # the original batch budget or starting extra retry rounds.
                refill = {
                    node: min(remaining[node], expansion_room(node, self.config))
                    for node in active
                }
                remaining_slots = sum(remaining.values()) - sum(refill.values())
                remaining = self._allocate_generation_slots(
                    active, remaining_slots, reserved=refill
                )
                new_candidate_targets = {
                    node: len(node.children) - initial_children[node]
                    + remaining.get(node, 0)
                    for node in nodes
                }
                active = [node for node in active if remaining.get(node, 0) > 0]
            elif batch_index > 0:
                active = [
                    node for node in active
                    if len(node.selectable_children()) < desired[node]
                ]
            if not active:
                break
            limits = [
                min(
                    expansion_room(node, self.config),
                    self.config.generation_batch_size,
                )
                for node in active
            ]
            if remaining is not None:
                limits = [
                    min(limit, remaining[node])
                    for node, limit in zip(active, limits)
                ]
            generation_requests = [
                ActionGenerationRequest(
                    max_candidates=limit,
                    sampling_priority=sampling_priority,
                    deadline=counters.generation_deadline,
                    execution_rejections=request.execution_rejections,
                    attempt_offset=max(
                        node.expansion_attempts
                        - int(node.info.get("external_seed_attempts", 0)),
                        0,
                    ),
                )
                for node, limit in zip(active, limits)
            ]
            generation_started = time.perf_counter()
            try:
                generate_batch = getattr(generator, "generate_batch", None)
                if callable(generate_batch) and len(active) > 1:
                    generated_results = tuple(
                        generate_batch(
                            [node.state for node in active],
                            [node.observation for node in active],
                            generation_requests,
                        )
                    )
                else:
                    generated_results = tuple(
                        generator.generate(
                            node.state,
                            node.observation,
                            generation_request,
                        )
                        for node, generation_request in zip(
                            active, generation_requests
                        )
                    )
            finally:
                counters.pose_generation_seconds += (
                    time.perf_counter() - generation_started
                )
            if len(generated_results) != len(active):
                raise ValueError(
                    "batch action generator returned the wrong length"
                )
            counters.pose_generation_batches += 1
            counters.pose_generation_node_requests += len(active)
            counters.pose_generation_max_node_batch = max(
                counters.pose_generation_max_node_batch,
                len(active),
            )
            for node, generated, limit in zip(
                active, generated_results, limits
            ):
                self._record_generation(
                    node,
                    generated,
                    counters,
                    limit,
                    request.execution_rejections,
                )
                if generated.exhausted or generated.attempts == 0:
                    stopped.add(node)

    def _record_generation(
        self,
        node,
        generated,
        counters,
        limit: int,
        execution_rejections: ExecutionRejections,
    ) -> None:
        if generated.attempts > limit:
            raise ValueError("action generator exceeded the requested attempt limit")
        if len(generated.candidates) > limit:
            raise ValueError("action generator exceeded the requested candidate limit")
        candidates = []
        for candidate in generated.candidates:
            reason = execution_rejections.reason(candidate.action)
            if reason is None and self._contains_action(node, candidate.action):
                reason = "duplicate_action"
            if reason is None:
                candidates.append(candidate)
            else:
                node.reject_expansion(reason)
                counters.execution_rejected_candidates += int(
                    reason.startswith("execution_rejected")
                )
        priors = [
            float(value)
            for value in self.prior_scorer.score(
                node.state, node.observation, candidates
            )
        ]
        if len(priors) != len(candidates):
            raise ValueError("prior scorer must return one value per candidate")
        accounted = len(generated.candidates) - len(candidates)
        for candidate, prior in zip(candidates, priors):
            if math.isfinite(prior):
                node.add_candidate(
                    candidate.action, prior=prior, info=candidate.diagnostics
                )
                counters.generated_candidates += 1
            else:
                node.reject_expansion("nonfinite_prior")
            accounted += 1
        for reason, count in generated.rejection_counts.items():
            node.reject_expansion(reason, count=count)
            accounted += count
        if generated.attempts > accounted:
            node.reject_expansion(
                "unclassified_generation",
                count=generated.attempts - accounted,
            )
        if generated.exhausted:
            node.expansion_exhausted = True
        counters.expansion_calls += 1

    def _seed_root(self, root, request, counters) -> None:
        if root.state is None:
            return
        for action in request.seed_actions:
            if expansion_room(root, self.config) <= 0:
                break
            root.info["external_seed_attempts"] = int(
                root.info.get("external_seed_attempts", 0)
            ) + 1
            reason = request.execution_rejections.reason(action)
            if action.object_index in root.state.placed_object_indices:
                reason = "seed_object_already_placed"
            elif action.object_index >= len(root.state.object_ids):
                reason = "seed_object_outside_inventory"
            elif action.object_id not in {
                -1,
                root.state.object_ids[action.object_index],
            }:
                reason = "seed_object_identity_mismatch"
            elif self._contains_action(root, action):
                reason = "duplicate_seed_action"
            if reason is not None:
                root.reject_expansion(reason)
                counters.execution_rejected_candidates += int(
                    reason.startswith("execution_rejected")
                )
                continue
            candidate = ActionCandidate(action)
            scores = tuple(
                self.prior_scorer.score(
                    root.state,
                    root.observation,
                    (candidate,),
                )
            )
            if len(scores) != 1 or not math.isfinite(float(scores[0])):
                root.reject_expansion("nonfinite_seed_prior")
                continue
            root.add_candidate(
                action,
                prior=float(scores[0]),
                info={"external_seed": True},
            )
            counters.seeded_candidates += 1
            counters.generated_candidates += 1

    def _apply_execution_rejections(self, root, request, counters) -> None:
        def apply(node) -> None:
            matched = False
            for child in tuple(node.children):
                apply(child)
                if child.action is None:
                    continue
                reason = request.execution_rejections.reason(child.action)
                if reason is None:
                    continue
                matched = True
                if child.selectable:
                    child.clear_children()
                    child.reject(reason, info=child.info)
                    counters.execution_rejected_candidates += 1
            if matched:
                node.retire_failed_children()

        apply(root)

    @staticmethod
    def _contains_action(node, action: Action) -> bool:
        return any(
            child.action is not None
            and child.action.object_index == action.object_index
            and np.allclose(child.action.pose, action.pose, rtol=0.0, atol=1e-9)
            for child in node.children
        )

    def _evaluate_many(self, nodes, fidelity, counters) -> None:
        if any(
            node.parent is None
            or node.parent.state is None
            or node.action is None
            for node in nodes
        ):
            raise ValueError("MCTS candidate is missing its parent transition data")
        states = [node.parent.state for node in nodes]
        actions = [node.action for node in nodes]
        self._record_evaluation_batch(counters, fidelity, len(actions))
        evaluations = self._evaluate_transitions_timed(
            states, actions, fidelity, counters
        )
        for node, evaluation in zip(nodes, evaluations):
            if evaluation.fidelity is not fidelity:
                raise ValueError("transition evaluator returned the wrong fidelity")
            if evaluation.state.object_ids != node.parent.state.object_ids:
                raise ValueError("transition evaluator changed the object inventory")
            # These evaluations are discarded here, so the node takes the
            # evaluation's private state instead of copying it again.
            self._apply_evaluation(node, evaluation, owned=True)
            counters.evaluated_candidates += 1
            counters.failed_evaluations += int(evaluation.failed)

    def _evaluate_transitions(self, states, actions, fidelity):
        if len(states) != len(actions):
            raise ValueError("transition batch states and actions must align")
        if not actions:
            return ()
        batch = getattr(self.transition_evaluator, "evaluate_batch", None)
        if callable(batch) and len(actions) > 1:
            results = tuple(batch(states, actions, fidelity))
        else:
            results = tuple(
                self.transition_evaluator.evaluate(state, action, fidelity)
                for state, action in zip(states, actions)
            )
        if len(results) != len(actions):
            raise ValueError("batch transition evaluator returned the wrong length")
        return results

    def _evaluate_transitions_timed(
        self, states, actions, fidelity, counters
    ):
        started = time.perf_counter()
        evaluations = self._evaluate_transitions(states, actions, fidelity)
        elapsed = time.perf_counter() - started
        if fidelity is EvaluationFidelity.SHORT:
            counters.short_simulation_seconds += elapsed
            settle_name = "short_settling_job_seconds"
            structural_name = "short_structural_job_seconds"
        elif fidelity is EvaluationFidelity.LONG:
            counters.long_simulation_seconds += elapsed
            settle_name = "long_settling_job_seconds"
            structural_name = "long_structural_job_seconds"
        else:
            return evaluations
        setattr(
            counters,
            settle_name,
            getattr(counters, settle_name)
            + _diagnostic_seconds(
                evaluations, "simulation_settling_wall_seconds"
            ),
        )
        setattr(
            counters,
            structural_name,
            getattr(counters, structural_name)
            + _diagnostic_seconds(
                evaluations, "simulation_structural_wall_seconds"
            ),
        )
        return evaluations

    def _validate_root_nodes(
        self,
        root,
        nodes,
        request,
        counters,
        deadline,
    ) -> None:
        fidelity = request.evaluation_fidelity
        batch_size = self._root_batch_size()
        for start in range(0, len(nodes), batch_size):
            if deadline is not None and time.monotonic() >= deadline:
                counters.root_validation_timed_out = True
                for child in nodes[start:]:
                    child.reject(
                        "root_validation_timeout",
                        info=child.info,
                    )
                break
            batch = nodes[start : start + batch_size]
            self._evaluate_many(batch, fidelity, counters)
            counters.root_validation_attempts += len(batch)
            counters.root_validation_failures += sum(child.failed for child in batch)

    def _final_select(self, root):
        """Mix search means with observed returns on each root's deepest paths.

        Every evaluated feasible root participates, including a root-only path.
        The path component contains no bootstrap and never skips a bad suffix
        in favor of a shallower prefix. Failure penalties remain in the mean.
        """
        scored = []
        weight = self.config.final_selection.max_weight
        for child in root.evaluated_children():
            depth, path = self._deepest_feasible_return(child)
            mean = float(child.mean_value)
            score = (1.0 - weight) * mean + weight * path
            # Remove diagnostics from the old fixed-depth selection on reused trees.
            for key in ("descendant_depth_satisfied", "descendant_depth_fallback",
                        "descendant_depth_eligible", "final_selection_used_mean_fallback"):
                child.info.pop(key, None)
            child.info.update(
                descendant_path_return=path,
                deepest_feasible_depth=depth,
                final_selection_max_weight=weight,
                final_selection_score=float(score),
            )
            if math.isfinite(score):
                scored.append((child, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        ranked = tuple(child for child, _ in scored)
        return (ranked[0] if ranked else None), ranked

    def _deepest_feasible_return(self, node):
        """Return (absolute search depth, discounted observed return).

        Depth takes priority within a root's subtree; return breaks depth ties.
        Pending/rejected transitions never extend a path. A terminal state or
        the current search horizon ends it even if retained children exist.
        """
        depth, value = node.depth, float(node.reward)
        next_index = None
        if not node.terminal and node.depth < self.config.max_depth:
            tails = [(child, self._deepest_feasible_return(child))
                     for child in node.evaluated_children()]
            if tails:
                best, (depth, tail) = max(tails, key=lambda item: item[1])
                next_index = node.children.index(best)
                value += self.config.discount * tail
        node.info.update(deepest_feasible_depth=depth, descendant_path_return=value,
                         deepest_path_child_index=next_index)
        return depth, value

    def _root_batch_size(self) -> int:
        return min(
            self.config.evaluation_batch_size,
            max(int(getattr(self.transition_evaluator, "worker_count", 1)), 1),
        )

    def _record_evaluation_batch(self, counters, fidelity, jobs: int) -> None:
        jobs = int(jobs)
        if jobs <= 0:
            return
        counters.max_evaluation_batch_size = max(
            counters.max_evaluation_batch_size, jobs
        )
        workers = int(getattr(self.transition_evaluator, "worker_count", 1))
        if fidelity is EvaluationFidelity.SHORT:
            counters.short_simulation_batches += 1
            counters.short_simulation_jobs += jobs
            counters.short_max_batch_size = max(
                counters.short_max_batch_size, jobs
            )
            if workers > 1 and jobs > 1:
                counters.short_parallel_batches += 1
                counters.short_parallel_jobs += jobs
        elif fidelity is EvaluationFidelity.LONG:
            counters.long_simulation_batches += 1
            counters.long_simulation_jobs += jobs

    @staticmethod
    def _apply_evaluation(node, evaluation, *, owned=False) -> None:
        info = {**node.info, **evaluation.diagnostics}
        node.set_result(
            evaluation.state,
            evaluation.observation,
            evaluation.reward,
            terminal=evaluation.terminal,
            failed=evaluation.failed,
            rejection_reason=evaluation.rejection_reason,
            info=info,
            owned=owned,
        )



def _optional_priority(value: str | None, level: str) -> str | None:
    if value is None:
        return None
    priority = str(value).strip()
    if not priority:
        raise ValueError(f"{level} action sampling priority must not be empty")
    return priority


def _diagnostic_seconds(evaluations, key: str) -> float:
    total = 0.0
    for evaluation in evaluations:
        try:
            value = float(evaluation.diagnostics.get(key, 0.0))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0.0:
            total += value
    return total
