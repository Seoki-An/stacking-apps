"""Collaborator contracts consumed by platform-neutral candidate search."""

from collections.abc import Sequence
from typing import Protocol, TypeVar

from stacking_common.domain import Action, State

from .types import (
    ActionCandidate,
    ActionGenerationRequest,
    ActionGenerationResult,
    EvaluationFidelity,
    TransitionEvaluation,
)


ObservationT = TypeVar("ObservationT")


class ActionGenerator(Protocol[ObservationT]):
    """Generate a bounded candidate batch for one evaluated state."""

    def generate(
        self,
        state: State,
        observation: ObservationT,
        request: ActionGenerationRequest,
    ) -> ActionGenerationResult:
        """Return candidates plus complete generation-attempt accounting."""


class BatchActionGenerator(ActionGenerator[ObservationT], Protocol):
    """Optionally generate candidates for independent states in one batch."""

    def generate_batch(
        self,
        states: Sequence[State],
        observations: Sequence[ObservationT],
        requests: Sequence[ActionGenerationRequest],
    ) -> Sequence[ActionGenerationResult]:
        """Return results aligned one-to-one with the supplied node requests."""


class PriorScorer(Protocol[ObservationT]):
    """Assign one finite search prior to every candidate in order."""

    def score(
        self,
        state: State,
        observation: ObservationT,
        candidates: Sequence[ActionCandidate],
    ) -> Sequence[float]:
        """Return scores aligned one-to-one with the candidate sequence."""


class TransitionEvaluator(Protocol[ObservationT]):
    """Evaluate an action without exposing simulator internals to search."""

    def evaluate(
        self,
        state: State,
        action: Action,
        fidelity: EvaluationFidelity,
    ) -> TransitionEvaluation[ObservationT]:
        """Apply an action at the requested physical fidelity."""


class BatchTransitionEvaluator(TransitionEvaluator[ObservationT], Protocol):
    """Optionally evaluate independent transitions as one backend batch."""

    def evaluate_batch(
        self,
        states: Sequence[State],
        actions: Sequence[Action],
        fidelity: EvaluationFidelity,
    ) -> Sequence[TransitionEvaluation[ObservationT]]:
        """Return evaluations aligned with the supplied state/action pairs."""


class ValueEstimator(Protocol[ObservationT]):
    """Estimate remaining return from one evaluated search state."""

    def estimate(
        self,
        state: State,
        observation: ObservationT,
    ) -> float:
        """Return a finite bootstrap value for MCTS backup."""
