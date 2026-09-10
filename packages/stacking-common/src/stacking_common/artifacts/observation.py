"""Serialization boundary for array observations."""

from collections.abc import Mapping
from dataclasses import fields

import numpy as np

from stacking_common.environment import Observation


_OBSERVATION_FIELDS = frozenset(item.name for item in fields(Observation))


def decode_observation(values: Mapping[str, object]) -> Observation:
    """Build an observation from known artifact fields."""

    return Observation(
        **{
            key: _owned(value)
            for key, value in values.items()
            if key in _OBSERVATION_FIELDS
        }
    )


def encode_observation(observation: Observation) -> dict[str, object]:
    """Return an independently owned full observation record."""

    values = {
        item.name: _owned(getattr(observation, item.name))
        for item in fields(observation)
    }
    if values["scene_height_map"] is None:
        values["scene_height_map"] = np.zeros_like(
            values["target_height_map"], dtype=np.float32
        )
    return values


def _owned(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    return value
