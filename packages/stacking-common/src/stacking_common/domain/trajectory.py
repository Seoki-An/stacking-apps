"""Stack-object motion recorded during one simulation stage."""

from copy import copy as shallow_copy
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

    @classmethod
    def from_samples(cls, object_id, poses, times):
        """Own a dense pose history; expose the usual mutable list on demand."""
        values = np.array(poses, dtype=float, order="C", copy=True)
        seconds = np.asarray(times, dtype=float)
        if values.ndim != 2 or values.shape[1] != 7:
            raise ValueError("trajectory poses must have shape (N, 7)")
        if not np.all(np.isfinite(values)) or np.any(np.linalg.norm(values[:, 3:], axis=1) <= 1e-12):
            raise ValueError("trajectory poses must be finite with non-zero quaternions")
        if (seconds.shape != (len(values),) or not np.all(np.isfinite(seconds))
                or np.any(seconds < 0) or np.any(np.diff(seconds) <= 0)):
            raise ValueError("trajectory times must be increasing seconds, one per pose")
        result = _PackedTrajectory(object_id)
        result.times = seconds.tolist()
        result._packed_poses = values
        return result

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
        """Copy owned arrays without revalidating previously accepted history.

        Construction and add_pose validate incoming poses. A branch copy keeps
        the same values and gives every mutable pose/time container a new owner.
        It owns them as one packed array, so copies of the copy are cheap.
        """
        copied = _PackedTrajectory.__new__(_PackedTrajectory)
        copied.__dict__.update(self.__dict__)
        del copied.__dict__["poses"]
        copied._packed_poses = np.array(self.poses, dtype=float).reshape(-1, 7)
        copied.times = None if self.times is None else self.times.copy()
        return copied

    def __len__(self) -> int:
        return len(self.poses)

    def __getitem__(self, index):
        return self.poses[index]

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self.poses)


class _PackedTrajectory(StackObjectTrajectory):
    """Preserve the public dataclass/list contract while copying history in bulk.

    Search uses final states and scalar metrics. Access to poses (including
    dataclasses.asdict for checkpoints) materializes ordinary mutable row views.
    Each trajectory owns its buffer; no CUDA scratch or sibling state is shared.
    """

    def __getattribute__(self, name):
        if name == "poses":
            values = object.__getattribute__(self, "__dict__")
            packed = values.pop("_packed_poses", None)
            if packed is not None:
                values["poses"] = list(packed)
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name == "poses":
            self.__dict__.pop("_packed_poses", None)
        super().__setattr__(name, value)

    def __len__(self):
        packed = self.__dict__.get("_packed_poses")
        return len(packed) if packed is not None else super().__len__()

    def copy(self):
        packed = self.__dict__.get("_packed_poses")
        if packed is None:
            return super().copy()
        copied = shallow_copy(self)
        copied._packed_poses = packed.copy()
        copied.times = None if self.times is None else self.times.copy()
        return copied
