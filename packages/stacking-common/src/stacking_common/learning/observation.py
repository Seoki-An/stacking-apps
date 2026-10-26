"""Conversion from runtime observations to learning-model inputs."""

import numpy as np

from stacking_common.environment import Observation


def encode_heightmap_observation(
    observation: Observation,
) -> dict[str, np.ndarray]:
    """Build the compact height-map record used by learning pipelines."""

    scene_height_map = observation.scene_height_map
    target_height_map = observation.target_height_map
    if scene_height_map is None:
        scene_height_map = np.zeros_like(target_height_map, dtype=np.float32)
    if target_height_map is None:
        target_height_map = np.zeros_like(scene_height_map, dtype=np.float32)

    values = {
        "scene_height_map": _array(scene_height_map, np.float16),
        "target_height_map": _array(target_height_map, np.float16),
        "height_map_xlim": _array(observation.height_map_xlim, np.float32),
        "height_map_ylim": _array(observation.height_map_ylim, np.float32),
    }
    if observation.target_kind == "composite":
        # Preserve the geometry that scalar envelope dimensions cannot encode.
        values.update({
            "target_is_composite": np.array(True),
            "target_bounds": _array(observation.target_bounds, np.float32),
            "target_points": _array(observation.target_points, np.float32),
            "target_faces": _array(observation.target_faces, np.int32),
            "target_component_ids": np.asarray(observation.target_component_ids),
            "target_component_point_offsets": _array(observation.target_component_point_offsets, np.int32),
            "target_component_face_offsets": _array(observation.target_component_face_offsets, np.int32),
            "target_footprint_mask": _array(observation.target_footprint_mask, bool),
        })
    if observation.object_ids is None:
        return values

    companions = {
        "object_poses": observation.object_poses,
        "object_physical_features": observation.object_physical_features,
    }
    missing = [name for name, value in companions.items() if value is None]
    if missing:
        raise ValueError(
            "observation object IDs require fields: " + ", ".join(missing)
        )
    values.update(
        {
            "object_ids": _array(observation.object_ids, np.int32),
            "object_poses": _array(observation.object_poses, np.float32),
            "scene_mask": _array(observation.scene_mask, bool),
            "object_physical_features": _array(
                observation.object_physical_features, np.float32
            ),
            "target_width": _array(observation.target_width, np.float32),
            "target_length": _array(observation.target_length, np.float32),
            "target_height": _array(observation.target_height, np.float32),
            "target_origin": _array(observation.target_origin, np.float32),
        }
    )
    return values


def _array(value, dtype) -> np.ndarray:
    return np.array(value, dtype=dtype, copy=True)
