"""Explicit import boundary for recorded JSON or trusted legacy pickup pickles."""

import hashlib
import json
import pickle
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from stacking_common.planning import PickSceneSnapshot


def load_pick_scene(path, *, frame_id="world", object_ids=None):
    """Read object-ID → XYZW pose or (position, quaternion) values.

    Pickle inputs must be trusted local planner artifacts. New portable inputs
    can use JSON without requiring the legacy repository or Python classes.
    """
    source = Path(path).expanduser().resolve()
    data = source.read_bytes()
    if source.suffix.lower() == ".json":
        values = json.loads(data)
    elif source.suffix.lower() in (".pkl", ".pickle"):
        values = pickle.loads(data)
    else:
        raise ValueError("pickup scene must be JSON or a trusted .pkl/.pickle file")
    if not isinstance(values, Mapping):
        raise TypeError("pickup scene must contain an object mapping")
    poses = {}
    for key, value in values.items():
        object_id = int(key)
        if object_id in poses:
            raise ValueError(f"duplicate pickup object ID: {object_id}")
        if isinstance(value, (tuple, list)) and len(value) == 2:
            position, quaternion = (np.asarray(item, dtype=float) for item in value)
            if position.shape != (3,) or quaternion.shape != (4,):
                raise ValueError("pickup position/quaternion must have shapes (3,) and (4,)")
            poses[object_id] = np.r_[position, quaternion]
        else:
            poses[object_id] = np.asarray(value, dtype=float)
    scene = PickSceneSnapshot(frame_id, poses, str(source),
                              {"format": "recorded_pick_scene", "sha256": hashlib.sha256(data).hexdigest()})
    return scene if object_ids is None else scene.select(object_ids)
