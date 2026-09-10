"""Small geometry operations needed by environment features."""

import numpy as np

from stacking_common.domain.geometry import quaternion_matrix, transform_points


__all__ = ["exposed_surface_normal", "quaternion_matrix", "transform_points"]

def exposed_surface_normal(
    normals: np.ndarray,
    areas: np.ndarray,
    min_z: float = 0.0,
) -> np.ndarray | None:
    normals = np.asarray(normals, dtype=float)
    areas = np.asarray(areas, dtype=float)
    candidates = np.flatnonzero(normals[:, 2] >= min_z)
    if len(candidates) == 0:
        return None

    weights = areas[candidates]
    if len(weights) != len(candidates) or float(np.sum(weights)) <= 1e-12:
        weights = np.ones(len(candidates), dtype=float)
    normal = np.average(normals[candidates], axis=0, weights=weights)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return None
    return normal / norm
