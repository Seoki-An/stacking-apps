"""Sparse bundle adjustment with fixed intrinsics and explicit metric pose priors.

Only image observations and recorded camera poses enter this fit. No sensor
reference cloud, surface model, or world scale variable is optimized.
"""

from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from stacking_common.pointcloud.types import validate_rigid_transform


@dataclass(frozen=True)
class PosePrior:
    translation_sigma_m: float = .001
    rotation_sigma_deg: float = .2
    bound_sigmas: float = 3.
    point_bound_m: float = .05
    max_evaluations: int = 200

    def __post_init__(self):
        values = [self.translation_sigma_m, self.rotation_sigma_deg,
                  self.bound_sigmas, self.point_bound_m]
        if not np.isfinite(values).all() or min(values) <= 0 or self.max_evaluations < 1:
            raise ValueError('pose prior scales, bounds and evaluation count must be positive')


def project_normalized(points, poses, view_indices, point_indices):
    """Project world points through world_T_camera poses; also return depth."""
    v, p = np.asarray(view_indices, int), np.asarray(point_indices, int)
    camera = np.einsum('nji,nj->ni', poses[v, :3, :3], points[p]-poses[v, :3, 3])
    depth = camera[:, 2]
    denominator = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
    return camera[:, :2]/denominator[:, None], depth


def refine_cameras(poses, points, view_indices, point_indices, observations, focal, *, prior=PosePrior()):
    """Anchor camera zero, regularize other cameras, jointly fit training points.

    Translation increments use world axes. Rotation increments are camera-local
    rotation vectors. Component bounds are explicit; vector norms can reach
    sqrt(3) times a component bound. Prior sigmas are experiment assumptions,
    not measured calibration uncertainties. Image terms use a smooth robust
    cost; pose priors remain quadratic.
    """
    poses, points = np.asarray(poses, float), np.asarray(points, float)
    views, ids = np.asarray(view_indices, int), np.asarray(point_indices, int)
    xy, focal = np.asarray(observations, float), np.asarray(focal, float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 3:
        raise ValueError('at least three world_T_camera poses are required')
    for pose in poses:
        validate_rigid_transform(pose)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 6 or not np.isfinite(points).all():
        raise ValueError('at least six finite training points are required')
    if (views.ndim != 1 or ids.shape != views.shape or xy.shape != (len(views), 2)
            or not len(views) or not np.isfinite(xy).all()
            or np.any(views < 0) or np.any(views >= len(poses))
            or np.any(ids < 0) or np.any(ids >= len(points))
            or focal.shape != (2,) or not np.isfinite(focal).all() or np.any(focal <= 0)):
        raise ValueError('invalid image observation arrays')
    if len(set(zip(views.tolist(), ids.tolist()))) != len(views):
        raise ValueError('duplicate point observation in one view')
    if np.any(np.bincount(ids, minlength=len(points)) < 3) or np.any(np.bincount(views, minlength=len(poses)) < 6):
        raise ValueError('require three observations per point and six per camera')
    _, depths = project_normalized(points, poses, views, ids)
    if np.any(depths <= 0):
        raise ValueError('initial training points must be in front of every observing camera')
    camera_variables = 6*(len(poses)-1)
    scales = np.r_[np.full(3, np.radians(prior.rotation_sigma_deg)),
                   np.full(3, prior.translation_sigma_m)]
    def unpack(parameters):
        delta = parameters[:camera_variables].reshape(-1, 6)*scales
        updated = poses.copy()
        updated[1:, :3, :3] = poses[1:, :3, :3] @ Rotation.from_rotvec(delta[:, :3]).as_matrix()
        updated[1:, :3, 3] += delta[:, 3:]
        return updated, parameters[camera_variables:].reshape(-1, 3)
    def residual(parameters):
        updated, xyz = unpack(parameters)
        projected, _ = project_normalized(xyz, updated, views, ids)
        pixel = ((projected-xy)*focal).ravel()
        # Equivalent to soft-L1 for image components, linear least-squares for priors.
        robust = np.sqrt(2.)*pixel/np.sqrt(np.sqrt(1.+pixel*pixel)+1.)
        return np.r_[robust, parameters[:camera_variables]]
    initial = np.r_[np.zeros(camera_variables), points.ravel()]
    lower = np.r_[np.full(camera_variables, -prior.bound_sigmas), (points-prior.point_bound_m).ravel()]
    upper = np.r_[np.full(camera_variables, prior.bound_sigmas), (points+prior.point_bound_m).ravel()]
    sparsity = lil_matrix((len(views)*2+camera_variables, len(initial)), dtype=int)
    for index, (v, p) in enumerate(zip(views, ids)):
        if v > 0:
            sparsity[2*index:2*index+2, 6*(v-1):6*v] = 1
        sparsity[2*index:2*index+2, camera_variables+3*p:camera_variables+3*p+3] = 1
    for index in range(camera_variables):
        sparsity[2*len(views)+index, index] = 1
    result = least_squares(residual, initial, jac_sparsity=sparsity.tocsr(),
                           bounds=(lower, upper), x_scale='jac', max_nfev=prior.max_evaluations,
                           ftol=1e-7, xtol=1e-7, gtol=1e-7)
    updated, xyz = unpack(result.x)
    projected, depths = project_normalized(xyz, updated, views, ids)
    original, _ = project_normalized(points, poses, views, ids)
    corrections = result.x[:camera_variables].reshape(-1, 6)*scales
    return dict(poses=updated, points=xyz, success=bool(result.success and np.all(depths > 0)),
                message=result.message, evaluations=int(result.nfev), optimality=float(result.optimality),
                initial_pixel_errors=np.linalg.norm((original-xy)*focal, axis=1),
                final_pixel_errors=np.linalg.norm((projected-xy)*focal, axis=1),
                corrections=np.vstack((np.zeros(6), corrections)),
                camera_bound_hits=int(np.count_nonzero(result.active_mask[:camera_variables])),
                point_bound_hits=int(np.count_nonzero(result.active_mask[camera_variables:])),
                all_points_in_front=bool(np.all(depths > 0)), prior=asdict(prior))


def align_support_plane(points, target_parameters):
    """Diagnostic rigid plane alignment; tangent translation/yaw are unmeasured.

    Rotate by the minimum angle between normals about the source-plane centroid,
    then translate along the target normal. This is a separate world-frame
    reference adjustment, not a per-camera calibration update.
    """
    from stacking_common.pointcloud.plane_diagnostics import fit_height_plane
    points, target = np.asarray(points, float), np.asarray(target_parameters, float)
    if points.ndim != 2 or points.shape[1] != 3 or target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError('support-plane alignment requires Nx3 points and z=ax+by+c parameters')
    fit = fit_height_plane(points[:, :2], points[:, 2])
    source_normal = np.array(fit['normal'])
    target_normal = np.r_[-target[:2], 1.]
    target_normal /= np.linalg.norm(target_normal)
    rotation, _ = Rotation.align_vectors(target_normal[None], source_normal[None])
    centre = points.mean(axis=0)
    centre[2] = np.r_[centre[:2], 1.] @ fit['parameters']
    distance = (centre[2]-np.r_[centre[:2], 1.] @ target)*target_normal[2]
    transform = np.eye(4)
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = centre-distance*target_normal-transform[:3, :3]@centre
    return dict(transform=transform, source_plane=fit, anchor_xyz=centre.tolist(),
                normal_displacement_mm=float(-distance*1000), rotation_deg=float(np.degrees(rotation.magnitude())))
