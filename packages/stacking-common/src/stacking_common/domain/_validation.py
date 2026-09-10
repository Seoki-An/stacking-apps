"""Shared validation for small domain arrays."""

import numpy as np


def pose_array(value, name: str = "pose") -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (7,):
        raise ValueError(f"{name} must have shape (7,)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if np.linalg.norm(result[3:]) <= 1e-12:
        raise ValueError(f"{name} quaternion must have non-zero norm")
    return result.copy()


def vector3_array(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result = result.copy()
    result.setflags(write=False)
    return result
