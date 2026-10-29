"""Typed values exchanged by placement-generation collaborators."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from stacking_common.domain._validation import pose_array
from stacking_common.domain.action import _read_only


@dataclass(frozen=True)
class PlanarPlacement:
    """One planar position and orientation proposed before height fitting."""

    pose: np.ndarray
    score: float = 0.0
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        score = float(self.score)
        if not math.isfinite(score):
            raise ValueError("planar proposal score must be finite")
        object.__setattr__(self, "pose", pose_array(self.pose, "planar pose"))
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "diagnostics", _owned(self.diagnostics))


@dataclass(frozen=True)
class InitializedPlacement:
    """A planar proposal raised above its inferred support surface."""

    pose: np.ndarray
    support_height: float
    support_object_indices: tuple[int, ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        support_height = float(self.support_height)
        if not math.isfinite(support_height):
            raise ValueError("placement support height must be finite")
        indices = tuple(int(value) for value in self.support_object_indices)
        if any(value < 0 for value in indices):
            raise ValueError("support object indices must be non-negative")
        object.__setattr__(self, "pose", pose_array(self.pose, "initial pose"))
        object.__setattr__(self, "support_height", support_height)
        object.__setattr__(self, "support_object_indices", indices)
        object.__setattr__(self, "diagnostics", _owned(self.diagnostics))


@dataclass(frozen=True)
class PlacementSolveResult:
    """Solver-independent result for one initialized placement."""

    pose: np.ndarray
    # Dimensionless max per-body residual; c_feq is the historical field name.
    c_feq: float
    c_gap: float
    converged: bool = True
    rejection_reason: str | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    # Invalid results carry a reason and placeholder pose/costs only. They must
    # never become actions, even when a platform permits nonconverged solves.
    valid: bool = True

    def __post_init__(self) -> None:
        c_feq = float(self.c_feq)
        c_gap = float(self.c_gap)
        if not math.isfinite(c_feq) or not math.isfinite(c_gap):
            raise ValueError("placement solver costs must be finite")
        reason = self.rejection_reason
        if reason is not None:
            reason = str(reason).strip()
            if not reason:
                raise ValueError("placement rejection reason must not be empty")
        converged = bool(self.converged)
        valid = bool(self.valid)
        if not valid and (converged or reason is None):
            raise ValueError("an invalid placement solve needs nonconvergence and a reason")
        object.__setattr__(self, "valid", valid)
        if reason is not None and converged:
            raise ValueError("a rejected placement solve cannot be converged")
        object.__setattr__(self, "pose", pose_array(self.pose, "solved pose"))
        object.__setattr__(self, "c_feq", c_feq)
        object.__setattr__(self, "c_gap", c_gap)
        object.__setattr__(self, "converged", converged)
        object.__setattr__(self, "rejection_reason", reason)
        object.__setattr__(self, "diagnostics", _owned(self.diagnostics))


@dataclass(frozen=True)
class PlacementFilterDecision:
    """Acceptance decision returned by a placement filter."""

    accepted: bool
    reason: str | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        accepted = bool(self.accepted)
        reason = self.reason
        if reason is not None:
            reason = str(reason).strip()
            if not reason:
                raise ValueError("placement filter reason must not be empty")
        if accepted and reason is not None:
            raise ValueError("an accepted placement cannot have a rejection reason")
        if not accepted and reason is None:
            raise ValueError("a rejected placement requires a reason")
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "diagnostics", _owned(self.diagnostics))

    @classmethod
    def accept(cls, **diagnostics: object) -> PlacementFilterDecision:
        return cls(True, diagnostics=diagnostics)

    @classmethod
    def reject(
        cls, reason: str, **diagnostics: object
    ) -> PlacementFilterDecision:
        return cls(False, reason=reason, diagnostics=diagnostics)


def _owned(values: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise TypeError("placement diagnostics must be a mapping")
    # Nested values become read-only and are shared, not deep-copied again.
    return MappingProxyType({key: _read_only(value) for key, value in values.items()})

