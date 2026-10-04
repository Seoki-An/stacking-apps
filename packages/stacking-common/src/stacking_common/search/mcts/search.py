"""Environment-independent Monte Carlo tree-search coordination."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field, replace
from typing import Generic, TypeVar

import numpy as np

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
from .tree_policy import best_child, can_expand, expansion_room, select_child


ObservationT = TypeVar("ObservationT")


@dataclass(frozen=True)
class MCTSSearchRequest:
    """Runtime controls for one bounded MCTS invocation."""

    iterations: int
    evaluation_fidelity: EvaluationFidelity = EvaluationFidelity.SHORT
    epsilon: float = 0.0
    reset_statistics: bool = True
    root_sampling_priority: str | None = None
    deeper_sampling_priority: str | None = None
    max_time_seconds: float | None = None
    simulation_depth: int | None = None
    beyond_depth_fidelity: EvaluationFidelity | None = None
    final_validation_fidelity: EvaluationFidelity | None = None
    final_validation_reserve_seconds: float = 0.0
    revalidate_children: bool = False
    revalidation_fidelity: EvaluationFidelity = EvaluationFidelity.LONG
    max_revalidation_drift: float = 0.1
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
        if (
            not math.isfinite(self.final_validation_reserve_seconds)
            or self.final_validation_reserve_seconds < 0.0
        ):
            raise ValueError(
                "MCTS final-validation reserve must be finite and non-negative"
            )
        if self.final_validation_reserve_seconds > 0.0:
            if self.max_time_seconds is None:
                raise ValueError(
                    "MCTS final-validation reserve requires a maximum search time"
                )
            if self.final_validation_reserve_seconds >= self.max_time_seconds:
                raise ValueError(
                    "MCTS final-validation reserve must be smaller than the "
                    "maximum search time"
                )
        if self.simulation_depth is not None and self.simulation_depth < 0:
            raise ValueError("MCTS simulation depth must be non-negative")
        if (
            not math.isfinite(self.max_revalidation_drift)
            or self.max_revalidation_drift < 0.0
        ):
            raise ValueError("MCTS revalidation drift must be non-negative")
        object.__setattr__(
            self, "evaluation_fidelity", EvaluationFidelity(self.evaluation_fidelity)
        )
        for name in ("beyond_depth_fidelity", "final_validation_fidelity"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, EvaluationFidelity(value))
        object.__setattr__(
            self,
            "revalidation_fidelity",
            EvaluationFidelity(self.revalidation_fidelity),
        )
        object.__setattr__(self, "epsilon", float(self.epsilon))
        object.__setattr__(self, "reset_statistics", bool(self.reset_statistics))
        object.__setattr__(self, "revalidate_children", bool(self.revalidate_children))
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
    descendant_validation_attempts: int = 0
    descendant_validation_failures: int = 0
    final_selection_mode: str = "mean"
    seeded_candidates: int = 0
    execution_rejected_candidates: int = 0


@dataclass
class _SearchCounters:
    generation_deadline: float | None = None
    expansion_calls: int = 0
    generated_candidates: int = 0
    evaluated_candidates: int = 0
    failed_evaluations: int = 0
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
    descendant_validation_attempts: int = 0
    descendant_validation_failures: int = 0
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
        """Run bounded selection-to-backup iterations and final validation."""

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
        iteration_deadline = (
            None
            if deadline is None
            else deadline - request.final_validation_reserve_seconds
        )
        counters = _SearchCounters(generation_deadline=iteration_deadline)
        root.detach()
        self._apply_execution_rejections(root, request, counters)
        self._seed_root(root, request, counters)
        revalidation = SubtreeRevalidationResult()
        if request.revalidate_children and root.children:
            revalidation = self.revalidate_children(
                root,
                request.revalidation_fidelity,
                request.max_revalidation_drift,
                deadline=iteration_deadline,
            )
        if request.reset_statistics:
            root.reset_statistics(recursive=True)
        reset_generator_statistics = getattr(
            self.root_action_generator, "reset_statistics", None
        )
        if callable(reset_generator_statistics):
            reset_generator_statistics()

        if self.config.validate_root_before_search:
            self._prepare_authoritative_root(
                root,
                request,
                counters,
                iteration_deadline,
            )
        for child in root.evaluated_children():
            if not child.children:
                child.info.pop("root_lookahead_attempted", None)
        counters.root_lookahead_eligible = len(
            self._root_lookahead_eligible_children(root)
        )
        completed = 0
        while completed < request.iterations:
            if (
                iteration_deadline is not None
                and time.monotonic() >= iteration_deadline
            ):
                break
            coverage = self._root_lookahead_candidates(root)
            if not coverage:
                consumed = self._run_descendant_frontier_batch(
                    root,
                    request,
                    counters,
                    request.iterations - completed,
                )
                if consumed > 0:
                    completed += consumed
                else:
                    if (
                        iteration_deadline is not None
                        and time.monotonic() >= iteration_deadline
                    ):
                        break
                    self._run_iteration(root, request, counters)
                    completed += 1
            else:
                consumed = self._run_root_lookahead_batch(
                    coverage[: request.iterations - completed],
                    request,
                    counters,
                )
                if consumed <= 0:
                    break
                completed += consumed

        selected, ranked_candidates = self._final_select(
            root, request, counters, deadline
        )
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
            self._prepare_authoritative_root(
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
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                coverage = self._root_lookahead_candidates(root)
                if not coverage:
                    break
                self._run_root_lookahead_batch(
                    coverage,
                    request,
                    counters,
                )
            excluded_actions.extend(retired)
            excluded_actions.extend(
                child.action
                for child in root.children
                if child.action is not None
            )
            selected, ranked_candidates = self._final_select(
                root, request, counters, deadline
            )
        deadline_exceeded = (
            deadline is not None and time.monotonic() >= deadline
        )
        return MCTSSearchResult(
            root=root,
            selected=selected,
            completed_iterations=completed,
            expansion_calls=counters.expansion_calls,
            generated_candidates=counters.generated_candidates,
            evaluated_candidates=counters.evaluated_candidates,
            failed_evaluations=counters.failed_evaluations,
            effective_max_depth=self.config.max_depth,
            ranked_candidates=ranked_candidates,
            timed_out=(
                completed < request.iterations
                or counters.final_validation_timed_out
                or counters.root_validation_timed_out
                or revalidation.timed_out
                or deadline_exceeded
            ),
            final_validation_attempts=counters.final_validation_attempts,
            final_validation_failures=counters.final_validation_failures,
            final_validation_timed_out=counters.final_validation_timed_out,
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
            revalidation=revalidation,
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
            descendant_validation_attempts=(
                counters.descendant_validation_attempts
            ),
            descendant_validation_failures=(
                counters.descendant_validation_failures
            ),
            final_selection_mode=self.config.final_selection.mode,
            seeded_candidates=counters.seeded_candidates,
            execution_rejected_candidates=(
                counters.execution_rejected_candidates
            ),
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

    def revalidate_children(
        self,
        root: MCTSNode[ObservationT],
        fidelity: EvaluationFidelity = EvaluationFidelity.LONG,
        max_drift: float = 0.1,
        *,
        deadline: float | None = None,
    ) -> SubtreeRevalidationResult:
        """Re-evaluate preserved children against their authoritative root."""

        if root.state is None:
            raise ValueError("preserved MCTS root must contain a state")
        if not math.isfinite(max_drift) or max_drift < 0.0:
            raise ValueError("MCTS revalidation drift must be non-negative")
        children = root.evaluated_children()
        fidelity = EvaluationFidelity(fidelity)
        batch_size = self._authoritative_batch_size()
        attempted = accepted = rejected = cleared = 0
        timed_out = False
        for start in range(0, len(children), batch_size):
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                for child in children[start:]:
                    child.reject("preserved_revalidation_timeout", info=child.info)
                    rejected += 1
                break
            batch = children[start : start + batch_size]
            actions = [
                child.action for child in batch if child.action is not None
            ]
            evaluations = self._evaluate_transitions(
                [root.state] * len(actions), actions, fidelity
            )
            attempted += len(evaluations)
            for child, evaluation in zip(batch, evaluations):
                previous = child.state
                if previous is not None and not evaluation.failed and (
                    _state_pose_drift(previous, evaluation.state) > max_drift
                ):
                    child.clear_children()
                    cleared += 1
                self._apply_evaluation(child, evaluation, authoritative=True)
                if evaluation.failed:
                    rejected += 1
                else:
                    accepted += 1
        if children and accepted == 0:
            root.expansion_exhausted = False
        return SubtreeRevalidationResult(
            attempted=attempted,
            accepted=accepted,
            rejected=rejected,
            cleared_subtrees=cleared,
            timed_out=timed_out,
        )

    def _prepare_authoritative_root(
        self,
        root,
        request,
        counters,
        deadline,
    ) -> None:
        """Generate, long-validate, and seed every available root candidate."""

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
        fidelity = self._authoritative_fidelity(request)
        pending = [
            child
            for child in root.selectable_children()
            if not self._is_authoritative(child, fidelity)
        ]
        self._validate_root_nodes(
            root,
            pending,
            request,
            counters,
            deadline,
        )
        for child in root.evaluated_children():
            if child.visits == 0:
                self._back_up(child)
                child.info["root_validation_seeded"] = True

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
        attempted = len(selected_children)
        for child in selected_children:
            child.info["root_lookahead_attempted"] = True
            counters.root_lookahead_attempts += 1

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
                self._tree_fidelity(evaluation_nodes[0].depth, request),
                counters,
            )
        for child, selected in selected_nodes:
            if selected is None:
                self._back_up(child)
            else:
                self._back_up(selected)

        counters.root_lookahead_generated += (
            counters.generated_candidates - generated_before
        )
        counters.root_lookahead_evaluated += (
            counters.evaluated_candidates - evaluated_before
        )
        return attempted

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

        by_fidelity = {}
        for node in evaluation_nodes:
            fidelity = self._tree_fidelity(node.depth, request)
            by_fidelity.setdefault(fidelity, []).append(node)
        for fidelity, nodes in by_fidelity.items():
            self._evaluate_many(nodes, fidelity, counters)

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
        if node.evaluated and not node.terminal and not node.failed:
            estimate_at_depth = getattr(
                self.value_estimator,
                "estimate_at_depth",
                None,
            )
            if callable(estimate_at_depth):
                remaining_depth = max(
                    self.config.max_depth + 1 - node.depth,
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
            # The legacy coordinator discounts its leaf estimate once before
            # entering the node-by-node discounted backup loop.
            bootstrap *= self.config.discount
        backpropagate(node, self.config, bootstrap_value=bootstrap)

    def _run_iteration(self, root, request, counters) -> None:
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
            child = select_child(
                node, self.config, epsilon=request.epsilon, rng=self.rng
            )
            if child is None:
                break
            if not child.evaluated:
                fidelity = self._tree_fidelity(child.depth, request)
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
            if reason is None and self._contains_external_seed(
                node, candidate.action
            ):
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

    @staticmethod
    def _contains_external_seed(node, action: Action) -> bool:
        return any(
            bool(child.info.get("external_seed"))
            and child.action is not None
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
        fidelity = self._authoritative_fidelity(request)
        batch_size = self._authoritative_batch_size()
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
            evaluations = self._evaluate_authoritative(
                root,
                batch,
                fidelity,
                request,
                counters,
            )
            counters.root_validation_attempts += len(evaluations)
            counters.root_validation_failures += sum(
                evaluation.failed for evaluation in evaluations
            )
            for child in batch:
                child.info["root_long_validated"] = bool(
                    child.evaluated and child.selectable
                )

    def _final_select(self, root, request, counters, deadline=None):
        if root.state is None:
            return None, ()
        children = root.evaluated_children()
        if not children:
            return None, ()
        ranked = sorted(
            children,
            key=self._provisional_root_score,
            reverse=True,
        )
        fidelity = request.final_validation_fidelity
        limit = self.config.final_validation_candidates
        candidates = ranked if limit == 0 else ranked[:limit]
        if fidelity is not None and limit > 0:
            pending = [
                child
                for child in candidates
                if not self._is_authoritative(child, fidelity)
            ]
            batch_size = self._authoritative_batch_size()
            for start in range(0, len(pending), batch_size):
                if deadline is not None and time.monotonic() >= deadline:
                    counters.final_validation_timed_out = True
                    break
                batch = pending[start : start + batch_size]
                evaluations = self._evaluate_authoritative(
                    root,
                    batch,
                    fidelity,
                    request,
                    counters,
                )
                counters.final_validation_attempts += len(evaluations)
                counters.final_validation_failures += sum(
                    evaluation.failed for evaluation in evaluations
                )
                for child, evaluation in zip(batch, evaluations):
                    child.info["final_validation_failed"] = bool(
                        evaluation.failed
                    )
                    child.info["final_validation_reward"] = float(
                        evaluation.reward
                    )

        scored = []
        for child in candidates:
            if not child.evaluated or not child.selectable:
                continue
            score = self._final_root_score(
                child,
                request,
                counters,
                deadline,
            )
            if math.isfinite(score):
                child.info["final_selection_mode"] = (
                    self.config.final_selection.mode
                )
                child.info["final_selection_score"] = float(score)
                scored.append((child, float(score)))
        if not scored:
            return None, ()
        scored.sort(key=lambda pair: pair[1], reverse=True)
        ranked_candidates = tuple(child for child, _ in scored)
        return ranked_candidates[0], ranked_candidates

    def _evaluate_authoritative(
        self,
        parent,
        nodes,
        fidelity,
        request,
        counters,
    ):
        actions = [node.action for node in nodes if node.action is not None]
        self._record_evaluation_batch(counters, fidelity, len(actions))
        evaluations = self._evaluate_transitions_timed(
            [parent.state] * len(actions),
            actions,
            fidelity,
            counters,
        )
        for node, evaluation in zip(nodes, evaluations):
            if evaluation.fidelity is not fidelity:
                raise ValueError("authoritative evaluator returned the wrong fidelity")
            if evaluation.state.object_ids != parent.state.object_ids:
                raise ValueError("transition evaluator changed the object inventory")
            previous = node.state
            if previous is not None and not evaluation.failed and (
                _state_pose_drift(previous, evaluation.state)
                > request.max_revalidation_drift
            ):
                node.clear_children()
            self._apply_evaluation(node, evaluation, authoritative=True)
            counters.evaluated_candidates += 1
            counters.failed_evaluations += int(evaluation.failed)
        return evaluations

    def _final_root_score(self, child, request, counters, deadline) -> float:
        selection = self.config.final_selection
        mean_score = self._mean_root_score(child)
        weight = self._final_selection_weight()
        if weight <= 0.0:
            return mean_score
        depth = min(
            selection.validate_descendant_depth,
            max(self.config.max_depth - child.depth, 0),
        )
        if depth <= 0:
            path_score = float(child.reward)
        elif selection.validate_descendant_path:
            path_score = self._validated_descendant_return(
                child,
                depth,
                request,
                counters,
                deadline,
            )
        else:
            path_score = self._known_descendant_return(child, depth)
        child.info["validated_descendant_path_available"] = bool(
            path_score is not None and math.isfinite(path_score)
        )
        if path_score is None or not math.isfinite(path_score):
            child.info["final_selection_used_mean_fallback"] = True
            if selection.require_validated_descendant:
                child.reject("missing_validated_descendant", info=child.info)
                return -math.inf
            return mean_score
        child.info["final_selection_used_mean_fallback"] = False
        child.info["validated_descendant_path_return"] = float(path_score)
        return weight * float(path_score) + (1.0 - weight) * mean_score

    def _provisional_root_score(self, child) -> float:
        mean_score = self._mean_root_score(child)
        weight = self._final_selection_weight()
        if weight <= 0.0:
            return mean_score
        path_score = self._known_descendant_return(
            child,
            min(
                self.config.final_selection.validate_descendant_depth,
                max(self.config.max_depth - child.depth, 0),
            ),
        )
        if path_score is None or not math.isfinite(path_score):
            return mean_score
        return weight * path_score + (1.0 - weight) * mean_score

    def _mean_root_score(self, child) -> float:
        if child.info.get("root_validation_seeded", False):
            return float(child.mean_value)
        if "final_validation_reward" in child.info:
            return float(child.info["final_validation_reward"])
        return float(child.mean_value)

    def _final_selection_weight(self) -> float:
        selection = self.config.final_selection
        if selection.mode == "mean":
            return 0.0
        if selection.mode == "max_descendant":
            return 1.0
        return float(selection.max_weight)

    def _known_descendant_return(self, node, remaining_depth):
        if node.failed:
            return None
        reward = float(node.reward)
        if remaining_depth <= 0 or node.terminal:
            return reward
        values = [
            self._known_descendant_return(child, remaining_depth - 1)
            for child in node.evaluated_children()
        ]
        values = [value for value in values if value is not None]
        if not values:
            return None
        return reward + self.config.discount * max(values)

    def _validated_descendant_return(
        self,
        node,
        remaining_depth,
        request,
        counters,
        deadline,
    ):
        if node.failed:
            return None
        reward = float(node.reward)
        if remaining_depth <= 0 or node.terminal:
            return reward
        children = sorted(
            node.children,
            key=lambda child: self._sortable_path_return(
                child, remaining_depth - 1
            ),
            reverse=True,
        )
        fidelity = request.final_validation_fidelity
        if fidelity is None:
            return self._known_descendant_return(node, remaining_depth)
        for child in children:
            if child.action is None:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                counters.final_validation_timed_out = True
                return None
            if not self._is_authoritative(child, fidelity):
                evaluation = self._evaluate_authoritative(
                    node,
                    [child],
                    fidelity,
                    request,
                    counters,
                )[0]
                counters.descendant_validation_attempts += 1
                counters.descendant_validation_failures += int(
                    evaluation.failed
                )
            if not child.selectable or not child.evaluated:
                continue
            tail = self._validated_descendant_return(
                child,
                remaining_depth - 1,
                request,
                counters,
                deadline,
            )
            if tail is not None and math.isfinite(tail):
                return reward + self.config.discount * tail
        return None

    def _sortable_path_return(self, node, remaining_depth) -> float:
        value = self._known_descendant_return(node, remaining_depth)
        return -math.inf if value is None else float(value)

    @staticmethod
    def _is_authoritative(node, fidelity) -> bool:
        return bool(
            node.evaluated
            and node.info.get("authoritative_fidelity") == fidelity.value
        )

    @staticmethod
    def _authoritative_fidelity(request: MCTSSearchRequest):
        return (
            request.final_validation_fidelity
            or EvaluationFidelity.LONG
        )

    def _authoritative_batch_size(self) -> int:
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
    def _apply_evaluation(node, evaluation, *, authoritative=False, owned=False) -> None:
        info = {**node.info, **evaluation.diagnostics}
        if authoritative:
            info["authoritative_fidelity"] = evaluation.fidelity.value
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

    @staticmethod
    def _tree_fidelity(depth: int, request: MCTSSearchRequest):
        if (
            request.simulation_depth is not None
            and depth > request.simulation_depth
            and request.beyond_depth_fidelity is not None
        ):
            return request.beyond_depth_fidelity
        return request.evaluation_fidelity


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


def _state_pose_drift(first, second) -> float:
    common = set(first.object_poses) & set(second.object_poses)
    if not common:
        return 0.0
    return max(
        float(
            np.linalg.norm(
                np.asarray(first.object_poses[object_id])[:3]
                - np.asarray(second.object_poses[object_id])[:3]
            )
        )
        for object_id in common
    )
