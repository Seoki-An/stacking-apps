"""State of one stacking search node."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._validation import pose_array
from .action import Action
from .contact import ContactPoint
from .trajectory import StackObjectTrajectory


@dataclass
class State:
    object_ids: list[int]
    placed_object_indices: list[int]
    object_poses: dict[int, np.ndarray]
    trajectories: list[dict[int, StackObjectTrajectory]]
    action_history: list[Action]
    terminated: bool = False
    failed: bool = False
    simulation_settled: bool = True
    contact_points: list[ContactPoint] = field(default_factory=list)
    pose_identified_object_ids: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.object_ids = [int(object_id) for object_id in self.object_ids]
        if any(object_id < 0 for object_id in self.object_ids):
            raise ValueError("state object IDs must be non-negative")
        if len(self.object_ids) != len(set(self.object_ids)):
            raise ValueError("state object IDs must be unique")

        self.placed_object_indices = [
            int(index) for index in self.placed_object_indices
        ]
        if len(self.placed_object_indices) != len(
            set(self.placed_object_indices)
        ):
            raise ValueError("placed object indices must be unique")
        if any(
            index < 0 or index >= len(self.object_ids)
            for index in self.placed_object_indices
        ):
            raise ValueError("placed object index is outside the inventory")

        object_id_set = set(self.object_ids)
        self.object_poses = {
            int(object_id): pose_array(pose, "object pose")
            for object_id, pose in self.object_poses.items()
        }
        if not set(self.object_poses).issubset(object_id_set):
            raise ValueError("state contains a pose for an unknown object")

        owned_trajectories = []
        for stage in self.trajectories:
            owned_stage = {}
            for object_id, trajectory in stage.items():
                object_id = int(object_id)
                if object_id not in object_id_set:
                    raise ValueError("trajectory belongs to an unknown object")
                if trajectory.object_id != object_id:
                    raise ValueError("trajectory key and object ID do not match")
                owned_stage[object_id] = trajectory.copy()
            owned_trajectories.append(owned_stage)
        self.trajectories = owned_trajectories

        if not all(isinstance(action, Action) for action in self.action_history):
            raise TypeError("state action history must contain Action values")
        self.action_history = [action.copy() for action in self.action_history]

        if not all(
            isinstance(contact, ContactPoint) for contact in self.contact_points
        ):
            raise TypeError("state contacts must be ContactPoint values")
        self.contact_points = list(self.contact_points)
        self.pose_identified_object_ids = {
            int(object_id) for object_id in self.pose_identified_object_ids
        }
        if not self.pose_identified_object_ids.issubset(object_id_set):
            raise ValueError("pose-identified IDs must belong to the inventory")

    def copy(self) -> "State":
        """Return a branch-independent snapshot of this state."""

        return State(
            object_ids=self.object_ids,
            placed_object_indices=self.placed_object_indices,
            object_poses=self.object_poses,
            trajectories=self.trajectories,
            action_history=self.action_history,
            terminated=self.terminated,
            failed=self.failed,
            simulation_settled=self.simulation_settled,
            contact_points=self.contact_points,
            pose_identified_object_ids=self.pose_identified_object_ids,
        )

    def latest_velocity_integrals(self) -> dict[int, float]:
        """Return the velocity integral for the latest simulation stage."""

        if not self.trajectories:
            return {}
        return {
            int(object_id): float(trajectory.vel_integral)
            for object_id, trajectory in self.trajectories[-1].items()
            if hasattr(trajectory, "vel_integral")
        }
