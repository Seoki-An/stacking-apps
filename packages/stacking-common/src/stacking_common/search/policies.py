"""Small concrete scoring and bootstrap policies for MCTS composition."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Generic, TypeVar

from stacking_common.domain import ObjectInventory, State

from .types import ActionCandidate


ObservationT = TypeVar("ObservationT")


class DiagnosticPriorScorer(Generic[ObservationT]):
    """Read finite prior logits from candidate diagnostics."""

    def __init__(self, key: str = "planar_score", default: float = 0.0) -> None:
        key = str(key).strip()
        if not key:
            raise ValueError("prior diagnostic key must not be empty")
        if not math.isfinite(default):
            raise ValueError("default prior score must be finite")
        self.key = key
        self.default = float(default)

    def score(
        self,
        state: State,
        observation: ObservationT,
        candidates: Sequence[ActionCandidate],
    ) -> list[float]:
        del state, observation
        values = [
            float(candidate.diagnostics.get(self.key, self.default))
            for candidate in candidates
        ]
        return [value if math.isfinite(value) else -math.inf for value in values]


class CallablePriorScorer(Generic[ObservationT]):
    """Adapt an application or learned batch scorer without framework imports."""

    def __init__(
        self,
        function: Callable[
            [State, ObservationT, Sequence[ActionCandidate]],
            Sequence[float],
        ],
    ) -> None:
        self.function = function

    def score(self, state, observation, candidates) -> Sequence[float]:
        return self.function(state, observation, candidates)


class PlacementPriorScorer(Generic[ObservationT]):
    """Score equilibrium and target containment before simulation."""

    def __init__(
        self,
        inventory: ObjectInventory,
        overlap_evaluator,
        *,
        posegen_threshold: float,
        stability_weight: float,
        containment_weight: float,
        planar_weight: float = 0.0,
    ) -> None:
        values = (
            posegen_threshold,
            stability_weight,
            containment_weight,
            planar_weight,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("placement prior settings must be finite")
        if posegen_threshold <= 0.0:
            raise ValueError("placement prior posegen threshold must be positive")
        self.inventory = inventory
        self.overlap_evaluator = overlap_evaluator
        self.posegen_threshold = float(posegen_threshold)
        self.stability_weight = float(stability_weight)
        self.containment_weight = float(containment_weight)
        self.planar_weight = float(planar_weight)

    def score(self, state, observation, candidates) -> list[float]:
        del observation
        if tuple(state.object_ids) != self.inventory.object_ids:
            raise ValueError("state inventory does not match placement prior inventory")
        scores = []
        for candidate in candidates:
            action = candidate.action
            stack_object = self.inventory[action.object_index]
            containment_value = candidate.diagnostics.get(
                "presimulation_target_containment_fraction"
            )
            containment = float(
                self.overlap_evaluator.object_containment(
                    stack_object,
                    action.pose,
                )
                if containment_value is None
                else containment_value
            )
            equilibrium = float(action.diagnostics.get(
                "posegen_equilibrium_residual_normalized",
                action.c_feq / self.posegen_threshold,
            ))
            stability = -min(max(equilibrium, 0.0), 1.0)
            planar = float(candidate.diagnostics.get("planar_score", 0.0))
            score = (
                self.stability_weight * stability
                + self.containment_weight * containment
                + self.planar_weight * planar
            )
            scores.append(score if math.isfinite(score) else -math.inf)
        return scores


class ZeroValueEstimator(Generic[ObservationT]):
    """Disable bootstrap estimation while retaining the value contract."""

    def estimate(self, state: State, observation: ObservationT) -> float:
        del state, observation
        return 0.0


class HorizonValueEstimator(Generic[ObservationT]):
    """Bootstrap with a fixed expected reward over a remaining horizon."""

    def __init__(self, mean_reward: float, horizon: int, discount: float) -> None:
        if not math.isfinite(mean_reward):
            raise ValueError("mean bootstrap reward must be finite")
        if horizon < 0:
            raise ValueError("bootstrap horizon must be non-negative")
        if not math.isfinite(discount) or not 0.0 <= discount <= 1.0:
            raise ValueError("bootstrap discount must be between zero and one")
        self.mean_reward = float(mean_reward)
        self.horizon = int(horizon)
        self.discount = float(discount)

    def estimate(self, state: State, observation: ObservationT) -> float:
        return self.estimate_at_depth(state, observation, self.horizon)

    def estimate_at_depth(
        self,
        state: State,
        observation: ObservationT,
        remaining_depth: int,
    ) -> float:
        """Project only through the remaining legacy search-depth budget."""

        del observation
        if not isinstance(remaining_depth, int) or remaining_depth < 0:
            raise ValueError("remaining search depth must be non-negative")
        remaining = min(
            self.horizon,
            remaining_depth,
            len(state.object_ids) - len(state.placed_object_indices),
        )
        return self.mean_reward * sum(
            self.discount**step for step in range(remaining)
        )


class CallableValueEstimator(Generic[ObservationT]):
    """Adapt an application or learned value callable."""

    def __init__(self, function: Callable[[State, ObservationT], float]) -> None:
        self.function = function

    def estimate(self, state: State, observation: ObservationT) -> float:
        value = float(self.function(state, observation))
        if not math.isfinite(value):
            raise ValueError("value estimator returned a non-finite value")
        return value
