"""Serialization boundary for action records."""

import copy
from collections.abc import Mapping
from numbers import Integral

from stacking_common.domain import Action


_ACTION_FIELDS = frozenset(
    {
        "object_index",
        "object_id",
        "pose",
        "solved_pose",
        "init_pose",
        "c_feq",
        "c_gap",
        "place_robustness_displacement",
        "diagnostics",
    }
)


def decode_action(values: Mapping[str, object]) -> Action:
    """Build a validated domain action while ignoring unrelated metadata."""

    return Action(
        **{key: value for key, value in values.items() if key in _ACTION_FIELDS}
    )


def encode_action(action: Action) -> dict[str, object]:
    """Return an independently owned mapping suitable for artifact encoders."""

    return {
        "object_index": action.object_index,
        "object_id": action.object_id,
        "pose": action.pose.copy(),
        "solved_pose": action.solved_pose.copy(),
        "init_pose": action.init_pose.copy(),
        "c_feq": action.c_feq,
        "c_gap": action.c_gap,
        "place_robustness_displacement": action.place_robustness_displacement,
        "diagnostics": _encode_diagnostics(action.diagnostics),
    }


def _encode_diagnostics(value):
    """Normalize entity-keyed maps at the artifact boundary, not in the domain.

    JSON object keys are strings. Native posegen contact maps use integer
    entity IDs, including NumPy integers. Reject ambiguous conversions rather
    than silently losing a value (for example keys 5 and "5" together).
    """
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, (str, Integral)):
                raise TypeError("action diagnostic keys must be strings or integers")
            name = str(key)
            if name in result:
                raise ValueError(f"duplicate serialized action diagnostic key: {name}")
            result[name] = _encode_diagnostics(item)
        return result
    if isinstance(value, list):
        return [_encode_diagnostics(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_encode_diagnostics(item) for item in value)
    return copy.deepcopy(value)
