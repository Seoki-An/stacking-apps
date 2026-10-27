"""Multi-step lifecycle for a reusable MCTS tree."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Generic, TypeVar

import numpy as np

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
        self._search_context = None

    @property
    def execution_rejections(self) -> ExecutionRejections:
        return self._rejections

    def search(self, request: MCTSSearchRequest) -> MCTSSearchResult[ObservationT]:
        """Search from the current root with accumulated runtime feedback."""

        context = (getattr(self.search_engine, 'config', None),
                   id(getattr(self.search_engine, 'transition_evaluator', self.search_engine)),
                   request.evaluation_fidelity)
        if self._search_context is not None and context != self._search_context:
            self.root.clear_children()
            self.root.reset_statistics(recursive=True)
        self._search_context = context
        effective_request = replace(
            request,
            execution_rejections=self._rejections.merged(
                request.execution_rejections
            ),
            seed_actions=(*self._seed_actions, *request.seed_actions),
        )
        result = self.search_engine.search(self.root, effective_request)
        self._seed_actions.clear()
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
        changed = observed_state is not None and not _same_scene(selected.state, observed_state)
        changed |= observed_state is None and observed_observation is not None
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
        if not preserve_subtree or changed:
            selected.clear_children()
            selected.reset_statistics(recursive=True)
        else:
            # Root generation starts a new pool (possibly CEM), so a deeper
            # node's attempt cursor and exhausted flag cannot carry over.
            # Keep its evaluated descendants and search evidence intact.
            selected.expansion_attempts = 0
            selected.expansion_exhausted = False
            selected.rejected_expansions.clear()
            selected.info.pop("external_seed_attempts", None)
        self.root = selected
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


def _same_scene(first: State, second: State) -> bool:
    """Reuse only identical physical/reward inputs; no pose-drift tolerance."""
    return (
        first.object_ids == second.object_ids
        and first.placed_object_indices == second.placed_object_indices
        and first.object_poses.keys() == second.object_poses.keys()
        and all(np.array_equal(first.object_poses[key], second.object_poses[key])
                for key in first.object_poses)
        and (first.terminated, first.failed, first.simulation_settled) ==
            (second.terminated, second.failed, second.simulation_settled)
        and first.pose_identified_object_ids == second.pose_identified_object_ids
        and first.latest_velocity_integrals() == second.latest_velocity_integrals()
        and len(first.contact_points) == len(second.contact_points)
        and all(np.array_equal(getattr(a, field.name), getattr(b, field.name))
                for a, b in zip(first.contact_points, second.contact_points)
                for field in fields(a))
        and len(first.action_history) == len(second.action_history)
        and all(a.object_id == b.object_id and a.object_index == b.object_index
                and np.array_equal(a.pose, b.pose)
                for a, b in zip(first.action_history, second.action_history))
    )
