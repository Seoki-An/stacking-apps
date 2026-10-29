"""Typed values exchanged by candidate search collaborators."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Generic, Mapping, TypeVar

import numpy as np

from stacking_common.domain import Action, State
from stacking_common.domain.action import _read_only


ObservationT = TypeVar("ObservationT")


@dataclass(frozen=True)
class ExecutionRejections:
    """Actions made unavailable by downstream planning or execution."""

    actions: tuple[Action, ...] = ()
    object_ids: frozenset[int] = frozenset()
    xy_radius: float = 0.18
    z_radius: float = 0.25
    same_object_only: bool = True
    use_initial_pose: bool = False

    def __post_init__(self) -> None:
        actions = tuple(self.actions)
        if not all(isinstance(action, Action) for action in actions):
            raise TypeError("execution-rejected actions must be Action values")
        object_ids = frozenset(int(value) for value in self.object_ids)
        if any(value < 0 for value in object_ids):
            raise ValueError("execution-rejected object IDs must be non-negative")
        xy_radius = float(self.xy_radius)
        z_radius = float(self.z_radius)
        if not math.isfinite(xy_radius) or xy_radius < 0.0:
            raise ValueError("execution-rejection XY radius must be non-negative")
        if not math.isfinite(z_radius) or z_radius < 0.0:
            raise ValueError("execution-rejection Z radius must be non-negative")
        object.__setattr__(
            self,
            "actions",
            tuple(action.copy() for action in actions),
        )
        object.__setattr__(self, "object_ids", object_ids)
        object.__setattr__(self, "xy_radius", xy_radius)
        object.__setattr__(self, "z_radius", z_radius)
        object.__setattr__(self, "same_object_only", bool(self.same_object_only))
        object.__setattr__(self, "use_initial_pose", bool(self.use_initial_pose))

    def reason(self, action: Action) -> str | None:
        """Return the execution exclusion matched by an action, if any."""

        if action.object_id in self.object_ids:
            return "execution_rejected_object"
        pose = action.init_pose if self.use_initial_pose else action.pose
        for rejected in self.actions:
            if self.same_object_only and action.object_id != rejected.object_id:
                continue
            rejected_pose = (
                rejected.init_pose if self.use_initial_pose else rejected.pose
            )
            if np.linalg.norm(pose[:2] - rejected_pose[:2]) > self.xy_radius:
                continue
            if self.z_radius > 0.0 and abs(pose[2] - rejected_pose[2]) > self.z_radius:
                continue
            return "execution_rejected_action"
        return None

    def merged(self, other: "ExecutionRejections") -> "ExecutionRejections":
        """Combine accumulated and per-search exclusions."""

        if not isinstance(other, ExecutionRejections):
            raise TypeError("execution rejection merge requires matching values")
        return ExecutionRejections(
            actions=(*self.actions, *other.actions),
            object_ids=self.object_ids | other.object_ids,
            xy_radius=other.xy_radius,
            z_radius=other.z_radius,
            same_object_only=other.same_object_only,
            use_initial_pose=other.use_initial_pose,
        )


class EvaluationFidelity(str, Enum):
    """Physical evaluation level requested by tree search."""

    NONE = "none"
    SHORT = "short"
    LONG = "long"


@dataclass(frozen=True)
class ActionGenerationRequest:
    """One bounded request for action candidates below a search node."""

    max_candidates: int
    sampling_priority: str | None = None
    attempt_offset: int = 0
    # Absolute monotonic deadline; generators stop between native work batches.
    deadline: float | None = None
    execution_rejections: ExecutionRejections = field(default_factory=ExecutionRejections)

    def __post_init__(self) -> None:
        if self.max_candidates <= 0:
            raise ValueError("action candidate limit must be positive")
        if self.attempt_offset < 0:
            raise ValueError("action generation attempt offset must be non-negative")
        if self.deadline is not None and not math.isfinite(self.deadline):
            raise ValueError("generation deadline must be finite")
        if self.sampling_priority is not None:
            priority = str(self.sampling_priority).strip()
            if not priority:
                raise ValueError("action sampling priority must not be empty")
            object.__setattr__(self, "sampling_priority", priority)
        object.__setattr__(self, "attempt_offset", int(self.attempt_offset))


@dataclass(frozen=True)
class ActionCandidate:
    """One generated action and generator-owned diagnostic metadata."""

    action: Action
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.action, Action):
            raise TypeError("search candidate action must be an Action")
        object.__setattr__(self, "action", self.action.copy())
        object.__setattr__(
            self,
            "diagnostics",
            _owned_mapping(self.diagnostics),
        )


@dataclass(frozen=True)
class ActionGenerationResult:
    """Candidates and accounting returned by one generator invocation."""

    candidates: tuple[ActionCandidate, ...]
    attempts: int
    exhausted: bool = False
    rejection_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if not all(isinstance(value, ActionCandidate) for value in candidates):
            raise TypeError("generated candidates must be ActionCandidate values")
        attempts = int(self.attempts)
        if attempts < 0:
            raise ValueError("action generation attempts must be non-negative")

        rejection_counts: dict[str, int] = {}
        for raw_reason, raw_count in self.rejection_counts.items():
            reason = _reason(raw_reason)
            count = int(raw_count)
            if count <= 0:
                raise ValueError("action rejection counts must be positive")
            rejection_counts[reason] = rejection_counts.get(reason, 0) + count

        accounted = len(candidates) + sum(rejection_counts.values())
        if accounted > attempts:
            raise ValueError(
                "accepted and rejected candidates exceed generation attempts"
            )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "exhausted", bool(self.exhausted))
        object.__setattr__(
            self,
            "rejection_counts",
            MappingProxyType(rejection_counts),
        )

    @classmethod
    def empty(cls, *, exhausted: bool) -> ActionGenerationResult:
        """Return an empty result without inventing rejected attempts."""

        return cls(candidates=(), attempts=0, exhausted=exhausted)


@dataclass(frozen=True)
class TransitionEvaluation(Generic[ObservationT]):
    """Evaluated result of applying one action to one parent state."""

    state: State
    observation: ObservationT
    reward: float
    fidelity: EvaluationFidelity
    terminal: bool = False
    failed: bool = False
    rejection_reason: str | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.state, State):
            raise TypeError("transition evaluation state must be a State")
        reward = float(self.reward)
        if not math.isfinite(reward):
            raise ValueError("transition evaluation reward must be finite")
        try:
            fidelity = EvaluationFidelity(self.fidelity)
        except ValueError as error:
            raise ValueError("unknown transition evaluation fidelity") from error

        terminal = bool(self.terminal or self.state.terminated)
        failed = bool(self.failed or self.state.failed)
        rejection_reason = self.rejection_reason
        if rejection_reason is not None:
            rejection_reason = _reason(rejection_reason)
            if not failed:
                raise ValueError(
                    "a rejected transition evaluation must be marked failed"
                )

        object.__setattr__(self, "state", self.state.copy())
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "fidelity", fidelity)
        object.__setattr__(self, "terminal", terminal)
        object.__setattr__(self, "failed", failed)
        object.__setattr__(self, "rejection_reason", rejection_reason)
        object.__setattr__(
            self,
            "diagnostics",
            _owned_mapping(self.diagnostics),
        )

    @property
    def done(self) -> bool:
        """Whether search must stop below this evaluated state."""

        return self.terminal or self.failed


def _owned_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise TypeError("search diagnostics must be a mapping")
    return MappingProxyType({key: _read_only(value) for key, value in values.items()})


def _reason(value: object) -> str:
    reason = str(value).strip()
    if not reason:
        raise ValueError("search rejection reason must not be empty")
    return reason
