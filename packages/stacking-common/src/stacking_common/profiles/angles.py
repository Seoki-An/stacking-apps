"""Explicit degree-valued YAML fields resolved to canonical radians."""

import numpy as np


def resolve_degree_fields(values, fields):
    """Accept either a canonical field or its ``_degrees`` alias, never both."""
    resolved = dict(values)
    for name in fields:
        alias = f"{name}_degrees"
        if alias not in resolved:
            continue
        if name in resolved:
            raise ValueError(f"specify only one of {name} and {alias}")
        degrees = np.asarray(resolved.pop(alias), dtype=float)
        if not np.all(np.isfinite(degrees)):
            raise ValueError(f"{alias} must contain finite angles")
        resolved[name] = np.deg2rad(degrees).tolist()
    return resolved
