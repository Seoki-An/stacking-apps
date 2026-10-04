"""Data-only tree nodes for Monte Carlo tree search."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Generic, Mapping, TypeVar

from stacking_common.domain import Action, State


ObservationT = TypeVar("ObservationT")


@dataclass(eq=False)
class MCTSNode(Generic[ObservationT]):
    """An evaluated state or an unevaluated incoming-action candidate."""

    action: Action | None = None
    prior: float = 0.0
    state: State | None = field(default=None, repr=False)
    observation: ObservationT | None = field(default=None, repr=False)
    reward: float = 0.0
    terminal: bool = False
    failed: bool = False
    rejection_reason: str | None = None
    info: dict[str, object] = field(default_factory=dict, repr=False)
    visits: int = 0
    value_sum: float = 0.0
    expansion_attempts: int = 0
    expansion_exhausted: bool = False
    rejected_expansions: Counter[str] = field(
        default_factory=Counter,
        repr=False,
    )
    parent: MCTSNode[ObservationT] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    children: list[MCTSNode[ObservationT]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.prior = float(self.prior)
        self.reward = float(self.reward)
        self.visits = int(self.visits)
        self.value_sum = float(self.value_sum)
        self.expansion_attempts = int(self.expansion_attempts)
        if not math.isfinite(self.prior):
            raise ValueError("MCTS node prior must be finite")
        if not math.isfinite(self.reward):
            raise ValueError("MCTS node reward must be finite")
        if self.visits < 0:
            raise ValueError("MCTS node visits must be non-negative")
        if not math.isfinite(self.value_sum):
            raise ValueError("MCTS node value sum must be finite")
        if self.expansion_attempts < 0:
            raise ValueError("MCTS expansion attempts must be non-negative")
        if self.state is not None:
            self.state = self.state.copy()
            self.terminal = bool(self.terminal or self.state.terminated)
            self.failed = bool(self.failed or self.state.failed)
        if self.rejection_reason is not None:
            self.rejection_reason = _reason(self.rejection_reason)
            self.failed = True
        self.info = dict(self.info)
        self.rejected_expansions = Counter(self.rejected_expansions)

    @property
    def depth(self) -> int:
        """Return depth relative to the currently attached root."""

        result = 0
        parent = self.parent
        while parent is not None:
            result += 1
            parent = parent.parent
        return result

    @property
    def evaluated(self) -> bool:
        """Whether this node has a resulting domain state."""

        return self.state is not None

    @property
    def selectable(self) -> bool:
        """Whether tree selection may consider this node."""

        return not self.failed and self.rejection_reason is None

    @property
    def mean_value(self) -> float:
        """Return the mean backed-up return, or zero before any visit."""

        return self.value_sum / self.visits if self.visits else 0.0

    def attach(self, child: MCTSNode[ObservationT]) -> None:
        """Attach a child while enforcing a single-parent acyclic tree."""

        if child.action is None:
            raise ValueError("an MCTS child requires an incoming action")
        if child.parent is not None:
            raise ValueError("MCTS child is already attached")
        ancestor: MCTSNode[ObservationT] | None = self
        while ancestor is not None:
            if ancestor is child:
                raise ValueError("attaching the MCTS child would create a cycle")
            ancestor = ancestor.parent
        child.parent = self
        self.children.append(child)

    def add_candidate(
        self,
        action: Action,
        *,
        prior: float = 0.0,
        info: Mapping[str, object] | None = None,
    ) -> MCTSNode[ObservationT]:
        """Record one accepted action-generation attempt."""

        child: MCTSNode[ObservationT] = MCTSNode(
            action=action.copy(),
            prior=prior,
            info=dict(info or {}),
        )
        self.attach(child)
        self.expansion_attempts += 1
        return child

    def reject_expansion(self, reason: str, *, count: int = 1) -> None:
        """Record generated actions rejected before child creation."""

        reason = _reason(reason)
        count = int(count)
        if count <= 0:
            raise ValueError("MCTS rejection count must be positive")
        self.expansion_attempts += count
        self.rejected_expansions[reason] += count

    def set_result(
        self,
        state: State,
        observation: ObservationT,
        reward: float,
        *,
        terminal: bool | None = None,
        failed: bool | None = None,
        rejection_reason: str | None = None,
        info: Mapping[str, object] | None = None,
        owned: bool = False,
    ) -> None:
        """Store one evaluated transition result on this node.

        ``owned=True`` takes a state that no other holder will use instead of
        copying it.
        """

        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("MCTS transition reward must be finite")
        self.state = state if owned else state.copy()
        self.observation = observation
        self.reward = reward
        self.terminal = bool(state.terminated or terminal)
        self.failed = bool(state.failed or failed)
        self.rejection_reason = (
            None if rejection_reason is None else _reason(rejection_reason)
        )
        if self.rejection_reason is not None:
            self.failed = True
        self.info = dict(info or {})

    def reject(self, reason: str, *, info: Mapping[str, object] | None = None) -> None:
        """Mark an existing candidate as explicitly unselectable."""

        self.failed = True
        self.rejection_reason = _reason(reason)
        self.info = dict(info or {})

    def detach(self) -> MCTSNode[ObservationT]:
        """Detach this subtree so it becomes a new depth-zero root."""

        if self.parent is not None:
            self.parent.children.remove(self)
            self.parent = None
        return self

    def clear_children(self) -> None:
        """Remove descendants while preserving this evaluated node."""

        for child in self.children:
            child.parent = None
        self.children.clear()
        self.expansion_attempts = 0
        self.expansion_exhausted = False
        self.rejected_expansions.clear()
        self.info.pop("external_seed_attempts", None)

    def retire_failed_children(self) -> tuple[Action, ...]:
        """Detach failed children and reopen bounded generation at this node."""

        retired = [child for child in self.children if not child.selectable]
        if not retired:
            return ()
        actions = tuple(
            child.action.copy() for child in retired if child.action is not None
        )
        for child in retired:
            child.parent = None
        self.children = [child for child in self.children if child.selectable]
        self.expansion_attempts = 0
        self.expansion_exhausted = False
        self.info.pop("external_seed_attempts", None)
        return actions

    def reset_statistics(self, *, recursive: bool = True) -> None:
        """Discard search evidence while retaining evaluated tree structure."""

        self.visits = 0
        self.value_sum = 0.0
        if recursive:
            for child in self.children:
                child.reset_statistics(recursive=True)

    def record_visit(self, value: float) -> None:
        """Accumulate one finite return sample."""

        value = float(value)
        if not math.isfinite(value):
            raise ValueError("MCTS backed-up value must be finite")
        self.visits += 1
        self.value_sum += value

    def selectable_children(self) -> list[MCTSNode[ObservationT]]:
        return [child for child in self.children if child.selectable]

    def pending_children(self) -> list[MCTSNode[ObservationT]]:
        return [child for child in self.selectable_children() if not child.evaluated]

    def evaluated_children(self) -> list[MCTSNode[ObservationT]]:
        return [child for child in self.selectable_children() if child.evaluated]


def _reason(value: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError("MCTS rejection reason must not be empty")
    return result
