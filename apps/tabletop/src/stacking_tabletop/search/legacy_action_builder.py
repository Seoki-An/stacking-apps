"""Compatibility boundary around the legacy tabletop action builder."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from stacking_common.domain import Action, State
from stacking_common.environment import Observation
from stacking_common.search import ActionGenerationRequest


class LegacyInventorySurface(Protocol):
    """Inventory operation consumed by the legacy `ActionBuilder`."""

    def update_from_state(self, state: object) -> None:
        """Synchronize legacy object order and poses from a state view."""


class LegacyActionBuilderSurface(Protocol):
    """Sampling operation exposed by the legacy `ActionBuilder`."""

    def get_action_samples(
        self,
        inventory: LegacyInventorySurface,
        state: object,
        *,
        scene_height_map: np.ndarray | None,
        n_action_samples: int,
        sampling_priority: str | None,
    ) -> tuple[object, np.ndarray]:
        """Return a legacy iterable action list and aligned numeric mask."""


@dataclass
class _LegacyStateView:
    """Legacy field names needed by inventory, pose solving, and sampling."""

    stone_set: list[int]
    stone_seq: list[int]
    stone_poses: dict[int, np.ndarray]

    @classmethod
    def from_common(cls, state: State) -> _LegacyStateView:
        return cls(
            stone_set=list(state.object_ids),
            stone_seq=list(state.placed_object_indices),
            stone_poses={
                int(object_id): np.asarray(pose, dtype=float).copy()
                for object_id, pose in state.object_poses.items()
            },
        )


class LegacyActionBuilderSampler:
    """Adapt legacy stone-named action sampling to the common masked shape."""

    def __init__(
        self,
        builder: LegacyActionBuilderSurface,
        inventory: LegacyInventorySurface,
    ) -> None:
        self.builder = builder
        self.inventory = inventory

    def __call__(
        self,
        state: State,
        observation: Observation,
        request: ActionGenerationRequest,
    ) -> tuple[list[Action | None], np.ndarray]:
        legacy_state = _LegacyStateView.from_common(state)
        self.inventory.update_from_state(legacy_state)
        legacy_actions, mask = self.builder.get_action_samples(
            self.inventory,
            legacy_state,
            scene_height_map=observation.scene_height_map,
            n_action_samples=request.max_candidates,
            sampling_priority=request.sampling_priority,
        )
        actions = [
            _common_action(action, state)
            for action in legacy_actions
        ]
        return actions, np.asarray(mask, dtype=float)


def _common_action(legacy_action: object, state: State) -> Action | None:
    object_index = int(getattr(legacy_action, "stone_idx"))
    object_id = int(getattr(legacy_action, "stone_id", -1))
    if object_index < 0:
        return None
    if object_index >= len(state.object_ids):
        raise ValueError("legacy action object index is outside the inventory")
    expected_id = int(state.object_ids[object_index])
    if object_id != expected_id:
        raise ValueError("legacy action object ID does not match its index")

    pose = np.asarray(getattr(legacy_action, "pose"), dtype=float)
    solved_pose = getattr(legacy_action, "solved_pose", None)
    return Action(
        object_index=object_index,
        object_id=object_id,
        pose=pose,
        init_pose=np.asarray(getattr(legacy_action, "init_pose"), dtype=float),
        solved_pose=(
            pose
            if solved_pose is None
            else np.asarray(solved_pose, dtype=float)
        ),
        c_feq=float(getattr(legacy_action, "c_feq", 0.0)),
        c_gap=float(getattr(legacy_action, "c_gap", 0.0)),
        place_robustness_displacement=float(
            getattr(legacy_action, "place_robustness_displacement", 0.0)
        ),
        diagnostics=copy.deepcopy(
            getattr(legacy_action, "diagnostics", {}) or {}
        ),
    )
