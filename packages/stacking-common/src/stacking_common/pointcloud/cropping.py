"""Spatial membership tests without changing retained sensor samples."""

import numpy as np

from .types import validate_rigid_transform


def box_mask(points, box_T_sensor, half_extents, *, batch_size=65536):
    """Test sensor XYZ against an oriented box with bounded temporary memory.

    The caller resolves the box transform at capture time. Returned membership
    can index every sensor channel without rotating or resampling those values.
    """
    xyz = np.asarray(points)
    transform = validate_rigid_transform(box_T_sensor, "box_T_sensor")
    extent = np.asarray(half_extents, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.dtype.kind not in "fiu":
        raise ValueError("sensor points must be numeric Nx3")
    if extent.shape != (3,) or not np.all(np.isfinite(extent)) or np.any(extent <= 0):
        raise ValueError("crop half extents must be finite positive XYZ")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("crop batch size must be a positive integer")
    mask = np.zeros(len(xyz), dtype=bool)
    for start in range(0, len(xyz), batch_size):
        values = xyz[start:start + batch_size]
        if not np.all(np.isfinite(values)):
            raise ValueError("sensor points must be finite")
        # Plane coefficients avoid materializing a full transformed cloud.
        selected = np.ones(len(values), dtype=bool)
        for axis in range(3):
            coordinate = values @ transform[axis, :3] + transform[axis, 3]
            selected &= np.abs(coordinate) <= extent[axis]
        mask[start:start + len(values)] = selected
    return mask
