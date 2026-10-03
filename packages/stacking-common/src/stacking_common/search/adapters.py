"""Adapters from existing common runtime APIs to search contracts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Generic, Mapping, Protocol, TypeVar

import numpy as np

from stacking_common.domain import Action, State
from stacking_common.environment import StackingEnvironment

from .types import (
    ActionCandidate,
    ActionGenerationRequest,
    ActionGenerationResult,
    EvaluationFidelity,
    TransitionEvaluation,
)


ObservationT = TypeVar("ObservationT")


class MaskedActionSampler(Protocol[ObservationT]):
    """Existing action-sampling shape: actions plus a numeric validity mask."""

    def __call__(
        self,
        state: State,
        observation: ObservationT,
        request: ActionGenerationRequest,
    ) -> tuple[Sequence[Action | None], Sequence[float]]:
        """Return sampled actions and aligned finite-or-negative-infinity mask."""


class MaskedActionGeneratorAdapter(Generic[ObservationT]):
    """Convert an actions-plus-mask sampler into `ActionGenerator`."""

    def __init__(self, sampler: MaskedActionSampler[ObservationT]) -> None:
        self.sampler = sampler

    def generate(
        self,
        state: State,
        observation: ObservationT,
        request: ActionGenerationRequest,
    ) -> ActionGenerationResult:
        actions_value, mask_value = self.sampler(state, observation, request)
        actions = list(actions_value)
        if not all(action is None or isinstance(action, Action) for action in actions):
            raise TypeError(
                "masked action sampler must return Action or None values"
            )
        mask = np.asarray(mask_value, dtype=float)
        if mask.shape != (len(actions),):
            raise ValueError("masked action sampler returned a misaligned mask")

        candidates: list[ActionCandidate] = []
        rejection_counts: Counter[str] = Counter()
        for action, mask_value in zip(actions, mask):
            if action is None:
                rejection_counts["dummy_padding"] += 1
            elif np.isfinite(mask_value):
                candidates.append(
                    ActionCandidate(
                        action=action,
                        diagnostics=action.diagnostics,
                    )
                )
            else:
                rejection_counts[_masked_rejection_reason(action)] += 1

        return ActionGenerationResult(
            candidates=tuple(candidates),
            attempts=len(actions),
            exhausted=len(actions) == 0,
            rejection_counts=rejection_counts,
        )


class StackingEnvironmentEvaluator(Generic[ObservationT]):
    """Evaluate search transitions using configured common environments."""

    def __init__(
        self,
        environments: Mapping[
            EvaluationFidelity,
            StackingEnvironment[ObservationT],
        ],
    ) -> None:
        normalized = {
            EvaluationFidelity(fidelity): environment
            for fidelity, environment in environments.items()
        }
        if not normalized:
            raise ValueError("at least one search evaluation environment is required")
        self.environments = normalized
        self.worker_count = 1
        for environment in normalized.values():
            transition = getattr(environment, "transition", None)
            backend = getattr(transition, "backend", None)
            self.worker_count = max(self.worker_count, getattr(backend, "batch_size", 1))
        self._initialized: set[EvaluationFidelity] = set()

    def evaluate(
        self,
        state: State,
        action: Action,
        fidelity: EvaluationFidelity,
    ) -> TransitionEvaluation[ObservationT]:
        fidelity = EvaluationFidelity(fidelity)
        try:
            environment = self.environments[fidelity]
        except KeyError as error:
            raise ValueError(
                f"no environment is configured for {fidelity.value!r} evaluation"
            ) from error

        if fidelity not in self._initialized:
            environment.reset()
            self._initialized.add(fidelity)
        environment.restore(state)
        result = environment.step(action)
        return self._evaluation(result, fidelity)

    @staticmethod
    def _evaluation(result, fidelity):
        transition_diagnostics = (
            dict(result.state.action_history[-1].diagnostics)
            if result.state.action_history
            else {}
        )
        diagnostics = {**transition_diagnostics, **result.info}
        rejection_reason = _transition_rejection_reason(
            result.state, diagnostics
        )
        failed = bool(result.state.failed or rejection_reason is not None)
        return TransitionEvaluation(
            state=result.state,
            observation=result.observation,
            reward=result.reward,
            fidelity=fidelity,
            terminal=result.state.terminated,
            failed=failed,
            rejection_reason=rejection_reason,
            diagnostics=diagnostics,
        )

    def evaluate_batch(
        self,
        states: Sequence[State],
        actions: Sequence[Action],
        fidelity: EvaluationFidelity,
    ) -> tuple[TransitionEvaluation[ObservationT], ...]:
        """Use batched physics when available, then build each search result."""

        if len(states) != len(actions):
            raise ValueError("transition batch states and actions must align")
        fidelity = EvaluationFidelity(fidelity)
        environment = self.environments.get(fidelity)
        if environment is not None and callable(getattr(environment, "step_batch", None)):
            if not states:
                return ()
            if fidelity not in self._initialized:
                environment.reset()
                self._initialized.add(fidelity)
            return tuple(self._evaluation(result, fidelity)
                         for result in environment.step_batch(states, actions))
        return tuple(self.evaluate(state, action, fidelity) for state, action in zip(states, actions))

    def close(self) -> None:
        """Close every distinct environment owned by this adapter once."""

        closed: set[int] = set()
        for environment in self.environments.values():
            identity = id(environment)
            if identity in closed:
                continue
            environment.close()
            closed.add(identity)


class ParallelTransitionEvaluator(Generic[ObservationT]):
    """Evaluate batches across independent evaluator instances."""

    def __init__(
        self,
        evaluators: Sequence[object],
        *,
        max_workers: int | None = None,
    ) -> None:
        values = tuple(evaluators)
        if not values:
            raise ValueError("parallel transition evaluation requires workers")
        workers = len(values) if max_workers is None else int(max_workers)
        if workers <= 0:
            raise ValueError("parallel transition worker count must be positive")
        self.evaluators = values
        self.worker_count = min(workers, len(values))
        self.executor = ThreadPoolExecutor(max_workers=self.worker_count)

    def evaluate(self, state, action, fidelity):
        return self.evaluators[0].evaluate(state, action, fidelity)

    def evaluate_batch(self, states, actions, fidelity):
        if len(states) != len(actions):
            raise ValueError("transition batch states and actions must align")
        buckets = [[] for _ in self.evaluators]
        for index, pair in enumerate(zip(states, actions)):
            buckets[index % len(buckets)].append((index, *pair))

        def evaluate_bucket(evaluator, jobs):
            return [
                (index, evaluator.evaluate(state, action, fidelity))
                for index, state, action in jobs
            ]

        futures = [
            self.executor.submit(evaluate_bucket, evaluator, jobs)
            for evaluator, jobs in zip(self.evaluators, buckets)
            if jobs
        ]
        results = [item for future in futures for item in future.result()]
        results.sort(key=lambda item: item[0])
        return tuple(value for _, value in results)

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        closed = set()
        for evaluator in self.evaluators:
            if id(evaluator) in closed:
                continue
            close = getattr(evaluator, "close", None)
            if callable(close):
                close()
            closed.add(id(evaluator))


def _masked_rejection_reason(action: Action) -> str:
    diagnostics = action.diagnostics
    reasons = diagnostics.get("action_mask_reasons")
    if isinstance(reasons, Sequence) and not isinstance(reasons, str) and reasons:
        return str(reasons[0])
    if reasons:
        return str(reasons)
    reason = diagnostics.get("action_mask_reason")
    if reason:
        return str(reason)
    if diagnostics.get("posegen_gap_ok", True) is False:
        return "posegen_gap"
    if diagnostics.get("final_pose_ok", True) is False:
        return "final_pose"
    if diagnostics.get("scan_orientation_ok", True) is False:
        return "scan_orientation"
    return "action_mask"


def _transition_rejection_reason(state: State, diagnostics: Mapping) -> str | None:
    """Apply cheap semantic checks after a simulator transition."""

    for key in (
        "rejection_reason",
        "mcts_unselectable_reason",
        "failure_reason",
    ):
        if diagnostics.get(key):
            return str(diagnostics[key])
    if not state.simulation_settled:
        return "simulation_unsettled"
    containment = diagnostics.get("target_containment_fraction")
    threshold = diagnostics.get("target_containment_threshold")
    if containment is not None and threshold is not None:
        containment = float(containment)
        threshold = float(threshold)
        if np.isfinite(containment) and np.isfinite(threshold):
            if containment < threshold:
                return "stone_outside_target"
    if state.failed:
        return "simulation_failure"
    return None
