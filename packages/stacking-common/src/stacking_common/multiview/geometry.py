"""Fixed-pose sparse triangulation with separate unused-view reprojection checks.

Camera poses map optical camera coordinates to a shared metric world frame.
Observations are undistorted normalized image coordinates, not raw pixels.
"""

import numpy as np

from stacking_common.pointcloud.types import validate_rigid_transform


def triangulate_track(observations, world_T_cameras):
    """Fit the largest-baseline pair only; return residuals in every view.

    No reprojection or reference-cloud gate is applied here. Callers retain
    failures and decide which candidates qualify for their experiment.
    """
    xy = np.asarray(observations, dtype=float)
    poses = np.asarray(world_T_cameras, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError('track requires at least two finite normalized observations')
    if poses.shape != (len(xy), 4, 4):
        raise ValueError('one world_T_camera is required per observation')
    for pose in poses:
        validate_rigid_transform(pose)
    centres = poses[:, :3, 3]
    baselines = np.linalg.norm(centres[:, None] - centres[None, :], axis=-1)
    i, j = np.unravel_index(np.argmax(baselines), baselines.shape)
    if baselines[i, j] < 1e-6:
        raise ValueError('triangulation requires a nonzero translation baseline')
    projections = np.linalg.inv(poses)[:, :3]
    equations = np.array([xy[k, a] * projections[k, 2] - projections[k, a]
                          for k in (i, j) for a in (0, 1)])
    _, _, vh = np.linalg.svd(equations)
    h = vh[-1]
    if abs(h[3]) < 1e-12:
        raise ValueError('triangulation point is at infinity')
    point = h[:3] / h[3]
    camera_points = np.einsum('nij,j->ni', projections, np.r_[point, 1.])
    if np.any(np.abs(camera_points[:, 2]) < 1e-12):
        raise ValueError('triangulation point lies on a camera plane')
    residuals = camera_points[:, :2] / camera_points[:, 2, None] - xy
    rays = point - centres[[i, j]]
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    angle = np.degrees(np.arccos(np.clip(rays[0] @ rays[1], -1., 1.)))
    return dict(point=point, fit_pair=np.array([i, j]), residuals=residuals,
                depths=camera_points[:, 2], parallax_deg=float(angle),
                baseline_m=float(baselines[i, j]))


def epipolar_distances(xy_a, xy_b, world_T_a, world_T_b, matrix):
    """Symmetric epipolar distance in undistorted pixels, from fixed poses."""
    a, b = np.asarray(xy_a, float), np.asarray(xy_b, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
        raise ValueError('paired normalized observations must be Nx2')
    relative = np.linalg.inv(validate_rigid_transform(world_T_b)) @ validate_rigid_transform(world_T_a)
    x, y, z = relative[:3, 3]
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    k_inv = np.linalg.inv(matrix)
    fundamental = k_inv.T @ skew @ relative[:3, :3] @ k_inv
    ah = np.column_stack((a, np.ones(len(a)))) @ np.asarray(matrix).T
    bh = np.column_stack((b, np.ones(len(b)))) @ np.asarray(matrix).T
    line_b, line_a = ah @ fundamental.T, bh @ fundamental
    numerator = np.abs(np.sum(bh * line_b, axis=1))
    denominators = np.column_stack((np.linalg.norm(line_a[:, :2], axis=1),
                                    np.linalg.norm(line_b[:, :2], axis=1)))
    if np.any(denominators < 1e-15):
        raise ValueError('epipolar geometry has a degenerate baseline/ray')
    return (numerator[:, None] / denominators).mean(axis=1)
