"""Explicit values returned by environment operations."""

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from stacking_common.domain import State


observation_t = TypeVar("observation_t")


@dataclass(frozen=True)
class ResetResult(Generic[observation_t]):
    state: State
    observation: observation_t


@dataclass(frozen=True)
class StepResult(Generic[observation_t]):
    state: State
    observation: observation_t
    reward: float
    info: dict[str, object] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.state.terminated or self.state.failed
