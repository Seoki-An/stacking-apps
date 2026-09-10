"""Collaborator contracts for environment-independent action generation."""

from collections.abc import Sequence
from typing import Protocol, TypeVar

from stacking_common.domain import ObjectInventory, State

from .types import (
    InitializedPlacement,
    PlacementFilterDecision,
    PlacementSolveResult,
    PlanarPlacement,
)


ObservationT = TypeVar("ObservationT")


class PlanarPoseSampler(Protocol[ObservationT]):
    """Propose bounded XY positions and orientations for one object."""

    def sample(
        self,
        inventory: ObjectInventory,
        state: State,
        observation: ObservationT,
        object_index: int,
        count: int,
        sampling_priority: str | None,
    ) -> Sequence[PlanarPlacement]:
        """Return at most ``count`` proposals in preference order."""


class ObjectOrientationProvider(Protocol[ObservationT]):
    """Resolve the measured base orientation for an unplaced object."""

    def quaternion(
        self,
        inventory: ObjectInventory,
        state: State,
        observation: ObservationT,
        object_index: int,
    ):
        """Return a scalar-last XYZW quaternion."""


class PlacementInitializer(Protocol):
    """Raise one planar proposal above its inferred support surface."""

    def initialize(
        self,
        inventory: ObjectInventory,
        state: State,
        object_index: int,
        proposal: PlanarPlacement,
    ) -> InitializedPlacement:
        """Return the initialized pose and inferred support information."""


class PlacementPoseSolver(Protocol):
    """Optimize one initialized placement against the current scene."""

    def solve(
        self,
        inventory: ObjectInventory,
        state: State,
        object_index: int,
        initial: InitializedPlacement,
    ) -> PlacementSolveResult:
        """Return a native-backend-independent pose solve result."""


class PlacementFilter(Protocol):
    """Accept or reject one solved placement with an explicit reason."""

    def evaluate(
        self,
        inventory: ObjectInventory,
        state: State,
        object_index: int,
        initial: InitializedPlacement,
        solved: PlacementSolveResult,
    ) -> PlacementFilterDecision:
        """Return an explicit placement acceptance decision."""
