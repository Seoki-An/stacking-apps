"""Ordered stack-object inventory with stable identity and initial poses."""

from collections.abc import Iterator, Mapping, Sequence

import numpy as np

from ._validation import pose_array
from .object import StackObject
from .target import TargetRegion


class ObjectInventory(Sequence[StackObject]):
    """One ordered candidate set and its initial scene poses."""

    def __init__(
        self,
        objects: Sequence[StackObject],
        initial_poses: Mapping[int, np.ndarray],
        target: TargetRegion,
    ) -> None:
        self.objects = tuple(objects)
        self.target = target
        object_ids = [stack_object.object_id for stack_object in self.objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("inventory object IDs must be unique")
        if set(initial_poses) != set(object_ids):
            raise ValueError("initial poses must match the inventory object IDs")

        self._index_by_id = {
            object_id: index for index, object_id in enumerate(object_ids)
        }
        self._initial_poses = {
            int(object_id): pose_array(pose, "initial pose")
            for object_id, pose in initial_poses.items()
        }

    @property
    def object_ids(self) -> tuple[int, ...]:
        return tuple(stack_object.object_id for stack_object in self.objects)

    @property
    def models(self) -> dict[int, object]:
        return {
            stack_object.object_id: stack_object.body_model
            for stack_object in self.objects
        }

    def initial_pose(self, object_id: int) -> np.ndarray:
        try:
            return self._initial_poses[int(object_id)].copy()
        except KeyError as error:
            raise KeyError(f"object {object_id} is not in the inventory") from error

    def pose(
        self,
        object_id: int,
        object_poses: Mapping[int, np.ndarray] | None = None,
    ) -> np.ndarray:
        if object_poses is not None and int(object_id) in object_poses:
            return pose_array(object_poses[int(object_id)], "object pose")
        return self.initial_pose(object_id)

    def object_index(self, object_id: int) -> int:
        try:
            return self._index_by_id[int(object_id)]
        except KeyError as error:
            raise ValueError(f"object {object_id} is not in the inventory") from error

    def __getitem__(self, index: int) -> StackObject:
        return self.objects[index]

    def __iter__(self) -> Iterator[StackObject]:
        return iter(self.objects)

    def __len__(self) -> int:
        return len(self.objects)
