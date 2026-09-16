"""Array-only axial-depth projection in an optical frame (+Z forward)."""

import numpy as np

from .types import PointCloudData


def optical_rays(width, height, matrix, distortion=(), *, model='none'):
    """Return HxWx3 rays with z=1; Brown coefficients are k1,k2,p1,p2,k3.

    Pixel coordinates refer to integer pixel centres. Brown inversion must
    converge; unsupported models are rejected even when coefficients are zero.
    """
    matrix = np.asarray(matrix, dtype=float)
    coefficients = np.asarray(distortion, dtype=float)
    if (not isinstance(width, (int, np.integer)) or not isinstance(height, (int, np.integer))
            or min(width, height) < 1 or matrix.shape != (3, 3)
            or not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1], rtol=0, atol=1e-12)
            or matrix[0, 1] != 0 or matrix[1, 0] != 0):
        raise ValueError('invalid pinhole image geometry')
    if (coefficients.ndim != 1 or coefficients.size not in (0, 5)
            or not np.isfinite(coefficients).all()):
        raise ValueError('expected zero or five finite Brown distortion coefficients')
    if model not in ('none', 'brown_conrady'):
        raise ValueError(f'unsupported depth distortion model: {model}')
    if model == 'none' and np.any(coefficients):
        raise ValueError('nonzero coefficients contradict distortion model none')
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    xd, yd = (u - matrix[0, 2]) / matrix[0, 0], (v - matrix[1, 2]) / matrix[1, 1]
    x, y = xd.copy(), yd.copy()
    if np.any(coefficients):
        k1, k2, p1, p2, k3 = coefficients
        with np.errstate(over='raise', divide='raise', invalid='raise'):
            try:
                for _ in range(100):
                    radius = x*x + y*y
                    radial = 1 + radius * (k1 + radius * (k2 + radius*k3))
                    dx = 2*p1*x*y + p2*(radius + 2*x*x)
                    dy = 2*p2*x*y + p1*(radius + 2*y*y)
                    next_x, next_y = (xd-dx)/radial, (yd-dy)/radial
                    error = max(np.max(np.abs(next_x-x)), np.max(np.abs(next_y-y)))
                    x, y = next_x, next_y
                    if error < 1e-12:
                        break
                else:
                    raise ValueError('Brown depth undistortion did not converge')
            except FloatingPointError as error:
                raise ValueError('Brown depth undistortion diverged') from error
    return np.stack((x, y, np.ones_like(x)), axis=-1)


def backproject_depth(depth, rays, metres_per_unit):
    """Return finite XYZ and matching flattened native-pixel indices.

    Depth is axial Z, not Euclidean range. Zero/nonfinite pixels are invalid;
    negative depth is rejected. No range crop, filtering or depth offset is used.
    """
    depth, rays = np.asarray(depth), np.asarray(rays, dtype=float)
    if (depth.ndim != 2 or depth.dtype.kind not in 'uif' or rays.shape != depth.shape + (3,)
            or not np.isfinite(rays).all() or not np.all(rays[..., 2] == 1)
            or not np.isfinite(metres_per_unit) or metres_per_unit <= 0 or np.any(depth < 0)):
        raise ValueError('invalid axial depth, rays or depth units')
    valid = np.isfinite(depth) & (depth > 0)
    indices = np.flatnonzero(valid).astype(np.int64)
    return PointCloudData(rays[valid] * (depth[valid].astype(float) * metres_per_unit)[:, None]), indices
