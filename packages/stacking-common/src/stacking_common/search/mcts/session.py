"""Multi-step lifecycle for a reusable MCTS tree."""

from __future__ import annotations

from dataclasses import replace
from typing import Generic, TypeVar

from stacking_common.domain import Action, State
from stacking_common.search.types import ExecutionRejections

from .node import MCTSNode
from .search import MCTSSearchRequest, MCTSSearchResult, MonteCarloTreeSearch


ObservationT = TypeVar("ObservationT")


class MCTSSession(Generic[ObservationT]):
    """Own the current root and runtime feedback across placement steps."""

    def __init__(
        self,
        search: MonteCarloTreeSearch[ObservationT],
        root: MCTSNode[ObservationT],
    ) -> None:
        if not root.evaluated:
            raise ValueError("an MCTS session requires an evaluated root")
        self.search_engine = search
        self.root = root.detach()
        self._rejections = ExecutionRejections()
        self._seed_actions: list[Action] = []
        self._revalidate_next = False

    @property
    def execution_rejections(self) -> ExecutionRejections:
        return self._rejections

    def search(self, request: MCTSSearchRequest) -> MCTSSearchResult[ObservationT]:
        """Search from the current root with accumulated runtime feedback."""

        effective_request = replace(
            request,
            execution_rejections=self._rejections.merged(
                request.execution_rejections
            ),
            seed_actions=(*self._seed_actions, *request.seed_actions),
            revalidate_children=(
                request.revalidate_children or self._revalidate_next
            ),
        )
        result = self.search_engine.search(self.root, effective_request)
        self._seed_actions.clear()
        self._revalidate_next = False
        return result

    def promote(
        self,
        selected: MCTSNode[ObservationT],
        *,
        observed_state: State | None = None,
        observed_observation: ObservationT | None = None,
        preserve_subtree: bool = True,
    ) -> MCTSNode[ObservationT]:
        """Commit a selected root child as the next step's root."""

        if selected.parent is not self.root:
            raise ValueError("the promoted node must be a child of the session root")
        selected.detach()
        if observed_state is not None:
            observation = (
                selected.observation
                if observed_observation is None
                else observed_observation
            )
            if observation is None:
                raise ValueError("observed root state requires an observation")
            selected.set_result(
                observed_state,
                observation,
                selected.reward,
                terminal=observed_state.terminated,
                failed=observed_state.failed,
                info=selected.info,
            )
        elif observed_observation is not None:
            selected.observation = observed_observation
        if not preserve_subtree:
            selected.clear_children()
        self.root = selected
        self._revalidate_next = bool(preserve_subtree and selected.children)
        return self.root

    def reject_execution(
        self,
        action: Action,
        *,
        reject_object: bool = False,
    ) -> None:
        """Remember a motion/execution rejection for later searches."""

        if not isinstance(action, Action):
            raise TypeError("execution rejection requires an Action")
        object_ids = set(self._rejections.object_ids)
        actions = list(self._rejections.actions)
        if reject_object:
            if action.object_id < 0:
                raise ValueError("object-wide rejection requires a resolved object ID")
            object_ids.add(action.object_id)
        else:
            actions.append(action.copy())
        self._rejections = ExecutionRejections(
            actions=tuple(actions),
            object_ids=frozenset(object_ids),
            xy_radius=self._rejections.xy_radius,
            z_radius=self._rejections.z_radius,
            same_object_only=self._rejections.same_object_only,
            use_initial_pose=self._rejections.use_initial_pose,
        )
        reason = self._rejections.reason(action)
        for child in self.root.children:
            if child.action is None:
                continue
            child_reason = self._rejections.reason(child.action)
            if child_reason is not None:
                child.clear_children()
                child.reject(child_reason, info=child.info)
        if reason is None:
            raise RuntimeError("execution rejection policy did not retain the action")
        self.root.expansion_exhausted = False

    def add_seed(self, action: Action) -> None:
        """Queue one externally proposed root action for the next search only."""

        if not isinstance(action, Action):
            raise TypeError("MCTS external seed must be an Action")
        self._seed_actions.append(action.copy())

    def clear_execution_rejections(self) -> None:
        """Clear feedback after the platform establishes a new execution context."""

        for child in self.root.children:
            if child.rejection_reason not in {
                "execution_rejected_action",
                "execution_rejected_object",
            }:
                continue
            child.rejection_reason = None
            child.failed = bool(child.state is not None and child.state.failed)
        self._rejections = ExecutionRejections(
            xy_radius=self._rejections.xy_radius,
            z_radius=self._rejections.z_radius,
            same_object_only=self._rejections.same_object_only,
            use_initial_pose=self._rejections.use_initial_pose,
        )
        self.root.expansion_exhausted = False
