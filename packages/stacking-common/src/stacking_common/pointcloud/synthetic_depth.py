"""Analytic optical ray intersections for deterministic replay fixtures."""

import numpy as np

from .types import validate_rigid_transform


def raycast_ellipsoids(rays, frame_T_camera, centres, radii, table_z, table_bounds):
    """Return axial depth and nearest-object labels (0 miss, 1 table, 2+ solids)."""
    pose = validate_rigid_transform(frame_T_camera)
    rays = np.asarray(rays, dtype=float)
    centres, radii = np.asarray(centres, dtype=float), np.asarray(radii, dtype=float)
    if (rays.ndim != 3 or rays.shape[-1] != 3 or not np.isfinite(rays).all()
            or not np.all(rays[..., 2] == 1) or centres.shape != radii.shape
            or centres.ndim != 2 or centres.shape[1] != 3 or not np.isfinite(centres).all()
            or not np.isfinite(radii).all() or np.any(radii <= 0)):
        raise ValueError('invalid synthetic optical scene')
    origin = pose[:3, 3]
    directions = rays @ pose[:3, :3].T
    distance = np.full(rays.shape[:2], np.inf)
    labels = np.zeros(distance.shape, dtype=np.uint8)
    dz = directions[..., 2]
    t = np.divide(table_z-origin[2], dz, out=np.full_like(dz, np.inf), where=np.abs(dz) > 1e-12)
    hits = origin[:2] + np.where(np.isfinite(t), t, 0)[..., None] * directions[..., :2]
    bounds = np.asarray(table_bounds)
    keep = (t > 0) & np.isfinite(t) & np.all((hits >= bounds[0]) & (hits <= bounds[1]), axis=-1)
    distance[keep], labels[keep] = t[keep], 1
    for index, (centre, radius) in enumerate(zip(centres, radii)):
        offset, direction = (origin-centre)/radius, directions/radius
        a = np.sum(direction*direction, axis=-1)
        b = 2*np.sum(offset*direction, axis=-1)
        c = np.sum(offset*offset)-1
        discriminant = b*b - 4*a*c
        root = np.sqrt(np.maximum(discriminant, 0))
        near, far = (-b-root)/(2*a), (-b+root)/(2*a)
        t = np.where(near > 0, near, far)
        keep = (discriminant >= 0) & (t > 0) & (t < distance)
        distance[keep], labels[keep] = t[keep], index+2
    distance[~np.isfinite(distance)] = 0
    return distance, labels
