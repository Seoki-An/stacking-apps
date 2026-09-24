"""Uncorrected height-field diagnostics for an already selected planar region."""

import numpy as np
from scipy.stats import binned_statistic_2d

from .types import PointCloudData


def height_grid(points, minimum_xy, maximum_xy, cell_size, *, minimum_samples=5):
    """Median Z per XY cell; empty/undersampled cells stay NaN, never interpolated."""
    points = PointCloudData(points).points
    lower, upper = np.asarray(minimum_xy, float), np.asarray(maximum_xy, float)
    if (lower.shape != (2,) or upper.shape != (2,) or
            not np.all(np.isfinite([lower, upper])) or np.any(upper <= lower) or
            not np.isfinite(cell_size) or cell_size <= 0 or
            isinstance(minimum_samples, bool) or not isinstance(minimum_samples, int) or
            minimum_samples < 1):
        raise ValueError('invalid height-grid bounds, cell size or sample count')
    shape = np.ceil((upper-lower)/cell_size).astype(int)
    if np.prod(shape.astype(float)) > 1_000_000:
        raise ValueError('height grid is too large')
    edges = [np.minimum(lower[i]+np.arange(shape[i]+1)*cell_size, upper[i]) for i in range(2)]
    centres = [(edge[1:]+edge[:-1])/2 for edge in edges]
    x, y = np.meshgrid(*centres, indexing='ij')
    xy = np.stack((x, y), axis=-1)
    if not len(points):
        return xy, np.full(tuple(shape), np.nan), np.zeros(tuple(shape), dtype=int)
    heights = binned_statistic_2d(*points.T, statistic='median', bins=edges).statistic
    counts = binned_statistic_2d(*points.T, statistic='count', bins=edges).statistic.astype(int)
    heights[counts < minimum_samples] = np.nan
    return xy, heights, counts


def fit_height_plane(xy, z):
    """Robust z=ax+by+c fit with equal cell weights; never corrects observations.

    Five 3-MAD trimming steps with a 1 mm vertical floor. This estimates a
    near-horizontal support plane, not a general vertical-plane detector.
    """
    xy, z = np.asarray(xy, float), np.asarray(z, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or z.shape != (len(xy),):
        raise ValueError('plane fit requires Nx2 XY and N heights')
    if len(z) < 6 or not np.all(np.isfinite(xy)) or not np.all(np.isfinite(z)):
        raise ValueError('plane fit requires at least six finite cells')
    centre = xy.mean(axis=0)
    design = np.column_stack((xy-centre, np.ones(len(z))))
    keep = np.ones(len(z), dtype=bool)

    def solve(mask):
        if mask.sum() < 6 or np.linalg.matrix_rank(design[mask]) < 3:
            raise ValueError('plane cells have insufficient two-dimensional coverage')
        return np.linalg.lstsq(design[mask], z[mask], rcond=None)[0]

    for _ in range(5):
        parameters = solve(keep)
        residual = z-design@parameters
        median = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep]-median))
        keep = np.abs(residual-median) <= max(.001, 3*1.4826*mad)
    parameters = solve(keep)
    normal = np.r_[-parameters[:2], 1.]
    normal /= np.linalg.norm(normal)
    residual = (z-design@parameters)*normal[2]
    parameters[2] -= centre@parameters[:2]
    return dict(parameters=parameters.tolist(), normal=normal.tolist(),
                inlier_cells=int(keep.sum()), total_cells=len(z),
                inlier_rms_mm=float(np.sqrt(np.mean(residual[keep]**2))*1000),
                all_cell_rms_mm=float(np.sqrt(np.mean(residual**2))*1000))


def difference_statistics(values_m):
    """Signed offset and absolute disagreement, with every supplied cell retained."""
    values = np.asarray(values_m, float).ravel()*1000
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError('difference statistics require finite observations')
    return dict(cell_count=len(values), signed_median_mm=float(np.median(values)),
                rms_mm=float(np.sqrt(np.mean(values**2))),
                median_absolute_mm=float(np.median(np.abs(values))),
                p95_absolute_mm=float(np.percentile(np.abs(values), 95)),
                max_absolute_mm=float(np.max(np.abs(values))),
                fraction_within_1mm=float(np.mean(np.abs(values) <= 1)))
