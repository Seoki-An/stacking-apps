"""Stack-object motion recorded during one simulation stage."""

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from ._validation import pose_array


@dataclass
class StackObjectTrajectory:
    object_id: int
    poses: list[np.ndarray] = field(default_factory=list)
    energy: float = 0.0
    vel_integral: float = 0.0
    settle_pose_count: int = 0
    settle_position_delta: float = 0.0
    settle_path_length: float = 0.0
    times: list[float] | None = None

    def __post_init__(self) -> None:
        self.object_id = int(self.object_id)
        if self.object_id < 0:
            raise ValueError("trajectory object ID must be non-negative")
        self.poses = [pose_array(pose) for pose in self.poses]
        self.energy = float(self.energy)
        self.vel_integral = float(self.vel_integral)
        self.settle_pose_count = int(self.settle_pose_count)
        if self.settle_pose_count < 0:
            raise ValueError("settle pose count must be non-negative")
        self.settle_position_delta = float(self.settle_position_delta)
        self.settle_path_length = float(self.settle_path_length)
        if self.times is not None:
            self.times = [float(value) for value in self.times]
            if (len(self.times) != len(self.poses)
                    or not np.all(np.isfinite(self.times))
                    or any(value < 0 for value in self.times)
                    or np.any(np.diff(self.times) <= 0)):
                raise ValueError("trajectory times must be increasing seconds, one per pose")

    def add_pose(self, pose: np.ndarray, *, time: float | None = None) -> None:
        pose = pose_array(pose)
        if time is not None:
            time = float(time)
            if (not np.isfinite(time) or time < 0
                    or (self.times and time <= self.times[-1])):
                raise ValueError("trajectory time must be finite and increasing")
            if self.times is None:
                if self.poses:
                    raise ValueError("cannot append a timed pose to an untimed trajectory")
                self.times = []
            self.times.append(time)
        elif self.times is not None:
            raise ValueError("timed trajectories require a time for every pose")
        self.poses.append(pose)

    def copy(self) -> "StackObjectTrajectory":
        return StackObjectTrajectory(
            object_id=self.object_id,
            poses=[pose.copy() for pose in self.poses],
            energy=self.energy,
            vel_integral=self.vel_integral,
            settle_pose_count=self.settle_pose_count,
            settle_position_delta=self.settle_position_delta,
            settle_path_length=self.settle_path_length,
            times=None if self.times is None else self.times.copy(),
        )

    def __len__(self) -> int:
        return len(self.poses)

    def __getitem__(self, index):
        return self.poses[index]

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self.poses)
