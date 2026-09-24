"""Rigid diagnostic alignment and matched-cell comparisons of upper surfaces."""
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .plane_diagnostics import height_grid, difference_statistics
from .types import validate_rigid_transform


def transform_points(points, transform):
    t = validate_rigid_transform(transform)
    return np.asarray(points) @ t[:3, :3].T + t[:3, 3]


def surface_grid(points, bounds, cell_size=.003):
    xy, z, count = height_grid(points, *bounds, cell_size, minimum_samples=2)
    valid = np.isfinite(z)
    interior = binary_erosion(valid, structure=np.ones((3, 3)), border_value=0)
    return dict(xy=xy, z=z, count=count, interior=interior)


def fit_shared_rigid(reference_grids, rgb_grids):
    """One bounded, robust point-to-plane ICP fit; equal total weight per region.

    Grid-centre median heights approximate surfaces; this is a diagnostic fit,
    not a camera calibration. Both input lists contain only training regions.
    """
    sources, targets, normals, groups = [], [], [], []
    for group, (ref, rgb) in enumerate(zip(reference_grids, rgb_grids)):
        valid = ref['interior'] & rgb['interior']
        if valid.sum() < 12:
            raise ValueError('insufficient co-observed interior cells for rigid fit')
        dx = ref['xy'][1, 0, 0]-ref['xy'][0, 0, 0]
        dy = ref['xy'][0, 1, 1]-ref['xy'][0, 0, 1]
        gx, gy = np.gradient(ref['z'], dx, dy)
        n = np.stack((-gx, -gy, np.ones_like(gx)), axis=-1)[ref['interior']]
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        targets.append(np.column_stack((ref['xy'][ref['interior']], ref['z'][ref['interior']])))
        normals.append(n)
        sources.append(np.column_stack((rgb['xy'][valid], rgb['z'][valid])))
        groups.append(np.full(valid.sum(), group))
    source = np.concatenate(sources)
    group_ids = np.concatenate(groups)
    pivot = source.mean(axis=0)
    trees = [cKDTree(p) for p in targets]
    parameters = np.zeros(6)
    bounds = np.r_[np.full(3, np.deg2rad(3)), np.full(3, .015)]
    converged = False
    for iteration in range(30):
        moved = (source-pivot) @ Rotation.from_rotvec(parameters[:3]).as_matrix().T + pivot + parameters[3:]
        matches = np.zeros_like(source); normal = np.zeros_like(source); valid = np.zeros(len(source), bool)
        weights = np.zeros(len(source))
        for group, tree in enumerate(trees):
            ids = np.flatnonzero(group_ids == group)
            distances, nearest = tree.query(moved[ids])
            valid[ids] = distances <= .015
            matches[ids] = targets[group][nearest]
            normal[ids] = normals[group][nearest]
            count = np.count_nonzero(valid[ids])
            if count < 12:
                raise ValueError('rigid fit lost support in a training region')
            weights[ids] = np.sqrt(len(source)/len(trees)/count)
        def residual(p):
            q = (source[valid]-pivot) @ Rotation.from_rotvec(p[:3]).as_matrix().T + pivot + p[3:]
            return np.sum((q-matches[valid])*normal[valid], axis=1)*weights[valid]
        solved = least_squares(residual, parameters, bounds=(-bounds, bounds), loss='soft_l1', f_scale=.001,
                               xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=150)
        change = np.linalg.norm(solved.x-parameters)
        parameters = solved.x
        if change < 1e-7:
            converged = bool(solved.success)
            break
    rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = pivot+parameters[3:]-rotation@pivot
    # Scale rotation columns by scene radius to compare sensitivity with translation.
    jacobian = solved.jac.copy()
    radius = np.sqrt(np.mean(np.sum((source-pivot)**2, axis=1)))
    jacobian[:, :3] /= max(radius, .001)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    return dict(transform=transform.tolist(), pivot_m=pivot.tolist(),
                displacement_at_pivot_mm=(parameters[3:]*1000).tolist(),
                rotation_deg=float(np.rad2deg(np.linalg.norm(parameters[:3]))),
                rotation_vector_deg=np.rad2deg(parameters[:3]).tolist(),
                converged=converged, iterations=iteration+1,
                hit_parameter_bound=bool(np.any(np.abs(parameters) >= .999*bounds)),
                training_cells=len(source), matched_training_cells=int(valid.sum()),
                scaled_jacobian_singular_values=singular.tolist(),
                scaled_jacobian_condition=float(singular[0]/max(singular[-1], 1e-15)))


def compare_region(reference, rgb, reference_after, bounds, transform):
    grids = [surface_grid(p, bounds) for p in (reference, rgb, transform_points(rgb, transform), reference_after)]
    common = np.logical_and.reduce([g['interior'] for g in grids])
    eligible = grids[0]['interior'] & grids[1]['interior']
    if common.sum() < 6:
        raise ValueError('insufficient common interior cells for surface comparison')
    z = [g['z'] for g in grids]
    metrics = dict(common_cells=int(common.sum()), initial_rgb_reference_interior_cells=int(eligible.sum()),
                   excluded_from_common=int(np.count_nonzero(eligible & ~common)),
                   before=difference_statistics((z[1]-z[0])[common]),
                   after=difference_statistics((z[2]-z[0])[common]),
                   femto_bracket=difference_statistics((z[3]-z[0])[common]))
    # Use fixed central, adequately supported one-cell-wide strips. Keep NaN gaps.
    slices = []
    for varying_axis in (0, 1):
        fixed_axis = 1-varying_axis
        count = common.sum(axis=varying_axis)
        supported = np.flatnonzero(count >= 4)
        fixed = int(supported[np.argmin(np.abs(supported-(common.shape[fixed_axis]-1)/2))]) if len(supported) else -1
        slices.append(fixed)
    arrays = dict(xy=grids[0]['xy'], common=common, reference=z[0], rgb_before=z[1], rgb_after=z[2],
                  reference_after=z[3], counts=np.stack([g['count'] for g in grids]), slice_indices=np.array(slices))
    return metrics, arrays
