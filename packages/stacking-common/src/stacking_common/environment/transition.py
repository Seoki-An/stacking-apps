"""State bookkeeping around an injected placement simulation backend."""

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

import numpy as np

from stacking_common.domain import (
    Action,
    ContactPoint,
    StackObjectTrajectory,
    State,
)


@dataclass(frozen=True)
class PlacementResult:
    object_poses: Mapping[int, np.ndarray]
    trajectories: Mapping[int, StackObjectTrajectory] = field(default_factory=dict)
    contact_points: Sequence[ContactPoint] = field(default_factory=tuple)
    terminated: bool = False
    failed: bool = False
    settled: bool = True
    pose_identified_object_ids: frozenset[int] = field(default_factory=frozenset)
    c_feq: float | None = None
    c_gap: float | None = None
    place_robustness_displacement: float | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)


class PlacementBackend(Protocol):
    """Physics boundary implemented by stacking-core bindings or an adapter."""

    def reset(self, object_ids: Sequence[int]) -> None:
        """Reset backend state for one inventory."""

    def place(self, state: State, action: Action) -> PlacementResult:
        """Settle and evaluate one candidate placement."""

    def restore(self, state: State) -> None:
        """Restore backend state from a domain snapshot."""

    def close(self) -> None:
        """Release backend resources."""


class SimulationTransition:
    """Concrete domain transition independent of a particular physics API."""

    def __init__(
        self, object_ids: Sequence[int], backend: PlacementBackend
    ) -> None:
        self.object_ids = [int(object_id) for object_id in object_ids]
        if any(object_id < 0 for object_id in self.object_ids):
            raise ValueError("transition object IDs must be non-negative")
        if len(self.object_ids) != len(set(self.object_ids)):
            raise ValueError("transition object IDs must be unique")
        self.backend = backend
        self._state = self._empty_state()

    def reset(self) -> State:
        self.backend.reset(self.object_ids)
        self._state = self._empty_state()
        return self.snapshot()

    def step(self, action: Action) -> State:
        self._validate(action, self._state)
        candidate = action.copy()
        candidate.object_id = self.object_ids[candidate.object_index]
        result = self.backend.place(self.snapshot(), candidate)
        return self._apply_result(candidate, result)

    def step_batch(self, states, actions):
        if len(states) != len(actions):
            raise ValueError("transition batch states and actions must align")
        # A batch backend reads the supplied branch states directly. Validate
        # against those parents without restoring/copying their full histories.
        prepared = []
        for state, action in zip(states, actions):
            if [int(value) for value in state.object_ids] != self.object_ids:
                raise ValueError("state inventory does not match this transition")
            self._validate(action, state)
            candidate = action.copy()
            candidate.object_id = self.object_ids[candidate.object_index]
            prepared.append(candidate)
        results = self.backend.place_batch(states, prepared)
        if len(results) != len(prepared):
            raise RuntimeError("placement batch returned the wrong number of results")
        outputs = []
        for state, candidate, result in zip(states, prepared, results):
            outputs.append(self._apply_result(candidate, result, state=state))
        return tuple(outputs)

    def _apply_result(self, candidate, result, *, state=None):
        state = self._state if state is None else state
        self._validate_result(result)
        poses = {
            int(object_id): np.asarray(pose, dtype=float).copy()
            for object_id, pose in state.object_poses.items()
        }
        poses.update(
            {
                int(object_id): np.asarray(pose, dtype=float).copy()
                for object_id, pose in result.object_poses.items()
            }
        )
        if candidate.object_id not in poses:
            poses[candidate.object_id] = candidate.pose.copy()
        candidate.pose = poses[candidate.object_id].copy()
        self._apply_action_result(candidate, result)

        sequence = [
            *state.placed_object_indices,
            candidate.object_index,
        ]
        identified = set(state.pose_identified_object_ids)
        identified.update(
            int(value) for value in result.pose_identified_object_ids
        )
        identified.intersection_update(poses)
        self._state = State(
            object_ids=self.object_ids.copy(),
            placed_object_indices=sequence,
            object_poses=poses,
            trajectories=[
                *state.trajectories,
                dict(result.trajectories),
            ],
            action_history=[
                *state.action_history,
                candidate.copy(),
            ],
            terminated=result.terminated or len(sequence) == len(self.object_ids),
            failed=result.failed,
            simulation_settled=result.settled,
            contact_points=list(result.contact_points),
            pose_identified_object_ids=identified,
        )
        return self.snapshot()

    def restore(self, state: State) -> None:
        if [int(value) for value in state.object_ids] != self.object_ids:
            raise ValueError("state inventory does not match this transition")
        self._state = state.copy()
        self.backend.restore(self.snapshot())

    def snapshot(self) -> State:
        return self._state.copy()

    def close(self) -> None:
        self.backend.close()

    def _empty_state(self) -> State:
        return State(
            self.object_ids.copy(),
            [],
            {},
            [],
            [],
            terminated=not self.object_ids,
        )

    def _validate(self, action: Action, state: State) -> None:
        if not 0 <= action.object_index < len(self.object_ids):
            raise IndexError("action object index is outside the inventory")
        expected_id = self.object_ids[action.object_index]
        if action.object_id not in (-1, expected_id):
            raise ValueError("action object ID does not match its inventory index")
        if action.object_index in state.placed_object_indices:
            raise ValueError("action object has already been placed")
        if state.terminated or state.failed:
            raise RuntimeError("cannot step a completed environment")

    def _validate_result(self, result: PlacementResult) -> None:
        known_ids = set(self.object_ids)
        result_ids = {
            int(object_id)
            for values in (
                result.object_poses,
                result.trajectories,
                result.pose_identified_object_ids,
            )
            for object_id in values
        }
        if not result_ids.issubset(known_ids):
            raise ValueError("placement result contains an unknown object")

    @staticmethod
    def _apply_action_result(
        action: Action, result: PlacementResult
    ) -> None:
        if result.c_feq is not None:
            action.c_feq = float(result.c_feq)
        if result.c_gap is not None:
            action.c_gap = float(result.c_gap)
        if result.place_robustness_displacement is not None:
            action.place_robustness_displacement = float(
                result.place_robustness_displacement
            )
        action.diagnostics.update(result.diagnostics)
