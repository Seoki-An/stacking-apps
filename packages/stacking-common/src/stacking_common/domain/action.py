"""Candidate placements and collections used by stacking search."""

import copy
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

from ._validation import pose_array


@dataclass
class Action:
    """One candidate placement in the canonical application frame."""

    object_index: int
    pose: np.ndarray
    init_pose: np.ndarray
    solved_pose: np.ndarray | None = None
    object_id: int = -1
    c_feq: float = 0.0
    c_gap: float = 0.0
    place_robustness_displacement: float = 0.0
    diagnostics: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.object_index = int(self.object_index)
        self.object_id = int(self.object_id)
        if self.object_index < 0:
            raise ValueError("action object index must be non-negative")
        if self.object_id < -1:
            raise ValueError("action object ID must be -1 or non-negative")
        self.pose = pose_array(self.pose)
        self.init_pose = pose_array(self.init_pose, "initial pose")
        self.solved_pose = pose_array(
            self.pose if self.solved_pose is None else self.solved_pose,
            "solved pose",
        )
        self.c_feq = float(self.c_feq)
        self.c_gap = float(self.c_gap)
        self.place_robustness_displacement = float(
            self.place_robustness_displacement
        )
        self.diagnostics = copy.deepcopy(self.diagnostics)

    def copy(self) -> "Action":
        """Copy owned values without revalidating an accepted action.

        Construction validates incoming poses; a copy keeps the same values and
        owns new pose arrays and one deep copy of the diagnostics.
        """
        copied = copy.copy(self)
        copied.pose = self.pose.copy()
        copied.init_pose = self.init_pose.copy()
        copied.solved_pose = (
            np.asarray(self.solved_pose, dtype=float).copy()
            if self.solved_pose is not None
            else self.pose.copy()
        )
        copied.diagnostics = copy.deepcopy(self.diagnostics)
        return copied


class ActionList:
    """A candidate collection with the array conversion used by MCTS."""

    def __init__(self, actions: Sequence[Action]):
        self.actions = list(actions)

    def mask(self, action_mask: np.ndarray) -> list[Action]:
        action_mask = np.asarray(action_mask, dtype=float)
        if action_mask.shape != (len(self.actions),):
            raise ValueError("action mask must have one value per action")
        return [
            action
            for action, mask in zip(self.actions, action_mask)
            if mask > -np.inf
        ]

    def get_dict_array(self) -> dict[str, np.ndarray]:
        if not self.actions:
            return {
                "object_index": np.array([], dtype=int),
                "object_id": np.array([], dtype=int),
                "pose": np.zeros((0, 7)),
                "solved_pose": np.zeros((0, 7)),
                "init_pose": np.zeros((0, 7)),
                "c_feq": np.array([]),
                "c_gap": np.array([]),
                "place_robustness_displacement": np.array([]),
            }
        return {
            "object_index": np.array(
                [action.object_index for action in self.actions]
            ),
            "object_id": np.array([action.object_id for action in self.actions]),
            "pose": np.vstack([action.pose for action in self.actions]),
            "solved_pose": np.vstack([action.solved_pose for action in self.actions]),
            "init_pose": np.array([action.init_pose for action in self.actions]),
            "c_feq": np.array([action.c_feq for action in self.actions]),
            "c_gap": np.array([action.c_gap for action in self.actions]),
            "place_robustness_displacement": np.array(
                [action.place_robustness_displacement for action in self.actions]
            ),
        }

    def __iter__(self) -> Iterator[Action]:
        return iter(self.actions)

    def __len__(self) -> int:
        return len(self.actions)
