"""Camera-aligned metric inputs for object pose estimators."""
import numpy as np
from stacking_common.pointcloud import validate_rigid_transform


def stereo_pose_input(disparity, valid, matrix, world_T_camera, baseline_m, box_xy, support_plane,
                      height_margin=.008):
    """Recover depth and a visible stone-region mask without a catalog pose prior.

    The mask uses observed stereo XYZ, a coarse region and the measured table.
    Pixels outside stereo validity stay unknown (zero depth), not filled surfaces.
    """
    disparity = np.asarray(disparity)
    valid = np.asarray(valid, dtype=bool)
    matrix = np.asarray(matrix, dtype=float)
    pose = validate_rigid_transform(world_T_camera)
    if (disparity.ndim != 2 or valid.shape != disparity.shape or matrix.shape != (3, 3)
            or not np.isfinite(matrix).all() or matrix[0, 0] <= 0
            or not np.isfinite(baseline_m) or baseline_m <= 0):
        raise ValueError('invalid stereo depth geometry')
    valid = valid & np.isfinite(disparity) & (disparity > 0)
    depth = np.zeros(disparity.shape, dtype=np.float32)
    depth[valid] = matrix[0, 0] * baseline_m / disparity[valid]
    v, u = np.nonzero(valid)
    rays = np.column_stack((u, v, np.ones(len(u)))) @ np.linalg.inv(matrix).T
    points = (rays * depth[valid, None]) @ pose[:3, :3].T + pose[:3, 3]
    lo, hi = np.asarray(box_xy, dtype=float)
    plane = np.asarray(support_plane, dtype=float)
    inside = np.all((points[:, :2] >= lo) & (points[:, :2] <= hi), axis=1)
    inside &= points[:, 2] > points[:, :2] @ plane[:2] + plane[2] + height_margin
    mask = np.zeros(disparity.shape, dtype=bool)
    mask[v[inside], u[inside]] = True
    return depth, mask
