"""Measured pickup-scene poses supplied to planning workflows."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from stacking_common.domain._validation import pose_array


@dataclass(frozen=True)
class PickSceneSnapshot:
    """Immutable world-frame object poses and their acquisition provenance."""

    frame_id: str
    object_poses: Mapping[int, np.ndarray]
    source: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frame_id = str(self.frame_id).strip()
        source = str(self.source).strip()
        if not frame_id:
            raise ValueError("pick-scene frame ID must not be empty")
        if not source:
            raise ValueError("pick-scene source must not be empty")
        poses: dict[int, np.ndarray] = {}
        for raw_id, raw_pose in self.object_poses.items():
            object_id = int(raw_id)
            if object_id < 0:
                raise ValueError("pick-scene object IDs must be non-negative")
            if object_id in poses:
                raise ValueError(f"duplicate pick-scene object ID: {object_id}")
            pose = pose_array(raw_pose, f"pick pose for object {object_id}")
            norm = np.linalg.norm(pose[3:])
            # Repeated checkpoint copies must preserve already-unit quaternions
            # exactly: a further division can change the provenance hash by an ULP.
            if abs(norm - 1.0) > 8 * np.finfo(float).eps:
                pose[3:] /= norm
            pose.setflags(write=False)
            poses[object_id] = pose
        if not poses:
            raise ValueError("pick scene must contain at least one object pose")
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "object_poses", MappingProxyType(poses))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(copy.deepcopy(dict(self.metadata))),
        )

    @property
    def object_ids(self) -> tuple[int, ...]:
        return tuple(self.object_poses)

    def pose(self, object_id: int) -> np.ndarray:
        try:
            return self.object_poses[int(object_id)].copy()
        except KeyError as error:
            raise KeyError(f"object {object_id} is not in the pick scene") from error

    def select(self, object_ids: Sequence[int]) -> "PickSceneSnapshot":
        selected_ids = tuple(int(value) for value in object_ids)
        if not selected_ids or len(selected_ids) != len(set(selected_ids)):
            raise ValueError("selected pick-scene object IDs must be unique")
        missing = [value for value in selected_ids if value not in self.object_poses]
        if missing:
            raise ValueError(
                f"pick scene does not contain requested object IDs: {missing}"
            )
        return PickSceneSnapshot(
            self.frame_id,
            {object_id: self.object_poses[object_id] for object_id in selected_ids},
            self.source,
            self.metadata,
        )

    def artifact_metadata(self) -> dict[str, object]:
        """Return a JSON-compatible description for run provenance."""

        return {
            **dict(self.metadata),
            "frame_id": self.frame_id,
            "source": self.source,
            "object_poses": {
                str(object_id): pose.tolist()
                for object_id, pose in self.object_poses.items()
            },
        }
