"""Typed conversion from application objects to stacking-core scenes."""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np


class SceneMobility(str, Enum):
    """Physical role of an object in one particular scene."""

    STATIC = "static"
    KINEMATIC = "kinematic"
    DYNAMIC = "dynamic"


@dataclass(frozen=True)
class SceneObjectInstance:
    """One stack object with pose and mobility resolved for a scene."""

    object_index: int
    object_id: int
    pose: np.ndarray
    mobility: SceneMobility

    def __post_init__(self) -> None:
        if self.object_index < 0:
            raise ValueError("object index must be non-negative")
        if not isinstance(self.mobility, SceneMobility):
            raise TypeError("mobility must be a SceneMobility")
        pose = np.asarray(self.pose, dtype=float)
        if pose.shape != (7,):
            raise ValueError("pose must have shape (7,)")
        if not np.all(np.isfinite(pose)):
            raise ValueError("pose must contain only finite values")
        if np.linalg.norm(pose[3:]) <= 1e-12:
            raise ValueError("pose quaternion must have non-zero norm")
        object.__setattr__(self, "pose", pose.copy())


class CoreSceneBuilder:
    """Build stacking-core snapshots without exposing native types upstream."""

    def __init__(
        self,
        models: Mapping[int, object],
        frame_id: int,
        core_module=None,
    ) -> None:
        if core_module is None:
            try:
                import stacking_core as core_module
            except ImportError as error:
                raise RuntimeError(
                    "stacking-core Python bindings are required to build scenes"
                ) from error
        self.core = core_module
        self.models = {int(key): value for key, value in models.items()}
        self.frame_id = int(frame_id)

    def build(
        self,
        objects: Sequence[SceneObjectInstance],
        fixed_bodies: Sequence[object] = (),
    ):
        """Return a canonical snapshot for the supplied per-solve roles."""

        self._validate_unique_objects(objects)
        bodies = list(fixed_bodies)
        occupied_entity_ids = {
            int(body.id) for body in bodies if hasattr(body, "id")
        }
        for stack_object in objects:
            entity_id = self.entity_id(stack_object.object_index)
            if entity_id in occupied_entity_ids:
                raise ValueError(f"duplicate scene entity ID: {entity_id}")
            try:
                model = self.models[stack_object.object_id]
            except KeyError as error:
                raise ValueError(
                    f"missing body model for object {stack_object.object_id}"
                ) from error
            bodies.append(
                self.core.BodyInstance(
                    entity_id,
                    model,
                    stack_object.pose,
                    mobility=self._native_mobility(stack_object.mobility),
                )
            )
            occupied_entity_ids.add(entity_id)
        return self.core.SceneSnapshot(self.frame_id, bodies)

    @staticmethod
    def entity_id(object_index: int) -> int:
        """Map a zero-based inventory index to its stable scene entity ID."""

        if object_index < 0:
            raise ValueError("object index must be non-negative")
        return int(object_index) + 1

    def _native_mobility(self, mobility: SceneMobility):
        return {
            SceneMobility.STATIC: self.core.Mobility.STATIC,
            SceneMobility.KINEMATIC: self.core.Mobility.KINEMATIC,
            SceneMobility.DYNAMIC: self.core.Mobility.DYNAMIC,
        }[mobility]

    @staticmethod
    def _validate_unique_objects(
        objects: Sequence[SceneObjectInstance],
    ) -> None:
        indices = [stack_object.object_index for stack_object in objects]
        if len(indices) != len(set(indices)):
            raise ValueError("scene object indices must be unique")
        object_ids = [stack_object.object_id for stack_object in objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("scene object IDs must be unique")
