"""Services required to run a stacking environment."""

from typing import Mapping, Protocol, TypeVar

import numpy as np

from stacking_common.domain import Action, State


observation_t = TypeVar("observation_t")


class TransitionModel(Protocol):
    """Owns mutable scene state and applies one placement action."""

    def reset(self) -> State:
        """Reset the scene and return its initial state."""

    def step(self, action: Action) -> State:
        """Apply one action and return the resulting state."""

    def restore(self, state: State) -> None:
        """Replace the mutable scene with a previously captured state."""

    def snapshot(self) -> State:
        """Return the current scene state."""

    def close(self) -> None:
        """Release resources owned by the transition implementation."""


class ObservationBuilder(Protocol[observation_t]):
    """Build a search or learning observation from domain state."""

    def build(self, state: State) -> observation_t:
        """Build an observation without changing the state."""


class HeightMapRenderer(Protocol):
    """Render metric top-surface heights on a fixed world-frame grid."""

    resolution: tuple[int, int]
    xlim: np.ndarray
    ylim: np.ndarray

    def render(self, state: State) -> np.ndarray:
        """Render the placed objects in a state."""

    def render_target(self) -> np.ndarray:
        """Render the fixed target geometry."""


class RewardEvaluator(Protocol):
    """Evaluate the state produced by one transition."""

    def compute(self, state: State) -> tuple[float, Mapping[str, object]]:
        """Return scalar reward and diagnostic values."""
