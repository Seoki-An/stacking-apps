"""Partial-observation to fixed-scale catalog comparison; diagnostic poses only."""
from itertools import permutations
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .types import PointCloudData, validate_rigid_transform


def distance_summary(distances):
    d = np.asarray(distances, float)*1000
    if not len(d) or not np.isfinite(d).all():
        raise ValueError('finite nonempty distances required')
    return dict(count=len(d), median_mm=float(np.median(d)), p90_mm=float(np.percentile(d, 90)),
                p95_mm=float(np.percentile(d, 95)), rms_mm=float(np.sqrt(np.mean(d*d))),
                within_1mm=float(np.mean(d <= 1)), within_2mm=float(np.mean(d <= 2)),
                within_3mm=float(np.mean(d <= 3)))


def observed_to_model(points, model, base_T_model):
    t = validate_rigid_transform(base_T_model)
    local = (np.asarray(points)-t[:3, 3]) @ t[:3, :3]
    return cKDTree(PointCloudData(model).points).query(local)[0]


def spatial_split(points):
    """Fixed world-grid 4mm checkerboard; holdout pixels never enter pose fitting."""
    points = PointCloudData(points).points
    cell = np.floor(points[:, :2]/.004).astype(np.int64)
    train = (cell.sum(axis=1) % 2) == 0
    if min(np.count_nonzero(train), np.count_nonzero(~train)) < 30:
        raise ValueError('insufficient support for spatial train/holdout split')
    return train


def assignment_ranking(costs, model_ids):
    costs = np.asarray(costs, float)
    if costs.shape != (len(model_ids), len(model_ids)) or not np.isfinite(costs).all():
        raise ValueError('assignment requires a finite square cost matrix')
    rankings = [dict(stone_ids=[int(model_ids[j]) for j in order],
                     total_score_mm=float(sum(costs[i, j] for i, j in enumerate(order))))
                for order in permutations(range(len(model_ids)))]
    return sorted(rankings, key=lambda r: r['total_score_mm'])


def fit_catalog_candidate(observed, model):
    """144 orientation starts; fit/train rank only, then report untouched patches.

    Observed points are the ICP source so unseen model surfaces are not penalized.
    Rigid only. Partial coverage can still admit incorrect poses/identities.
    """
    import open3d as o3d
    reg = o3d.pipelines.registration
    observed = PointCloudData(observed).points
    model = PointCloudData(model).points
    train = spatial_split(observed)
    def cloud(points, voxel):
        c = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(points, copy=True)))
        c = c.voxel_down_sample(voxel)
        c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.006, max_nn=30))
        return c
    source = cloud(observed[train], .0025)
    target = cloud(model, .0025)
    model_tree = cKDTree(model)
    evaluation = np.asarray(source.points)
    def score(transform):
        local = evaluation @ transform[:3, :3].T + transform[:3, 3]
        d = model_tree.query(local)[0]
        return float((np.median(d)+np.percentile(d, 90))*.5*1000)
    bases = [Rotation.identity()] + [Rotation.from_euler('x', deg, degrees=True) for deg in (90, 180, 270)] + [Rotation.from_euler('y', deg, degrees=True) for deg in (90, -90)]
    seeds = []
    # Initial position is estimated using training observations only.
    centre = (observed[train].min(axis=0)+observed[train].max(axis=0))/2
    top = np.percentile(observed[train, 2], 95)
    for base in bases:
        for yaw in range(0, 360, 15):
            rotation = (Rotation.from_euler('z', yaw, degrees=True)*base).as_matrix()
            rotated = model @ rotation.T
            shift = centre-(rotated.min(axis=0)+rotated.max(axis=0))/2
            shift[2] = top-np.percentile(rotated[:, 2], 95)
            t = np.eye(4); t[:3, :3] = rotation; t[:3, 3] = shift
            result = reg.registration_icp(source, target, .015, np.linalg.inv(t),
                reg.TransformationEstimationPointToPoint(), reg.ICPConvergenceCriteria(max_iteration=35))
            seeds.append((score(result.transformation), np.asarray(result.transformation)))
    seeds.sort(key=lambda item: item[0])
    fine_source = cloud(observed[train], .001)
    fine_target = cloud(model, .001)
    candidates = []
    # Refine distinct basins, rather than spending every retry on duplicate poses.
    distinct = []
    for _, transform in seeds:
        if any(Rotation.from_matrix(transform[:3, :3] @ previous[:3, :3].T).magnitude() < np.deg2rad(12)
               and np.linalg.norm(transform[:3, 3]-previous[:3, 3]) < .005 for previous in distinct):
            continue
        distinct.append(transform)
        if len(distinct) == 8:
            break
    for transform in distinct:
        result = reg.registration_icp(fine_source, fine_target, .008, transform,
            reg.TransformationEstimationPointToPlane(reg.HuberLoss(.002)),
            reg.ICPConvergenceCriteria(max_iteration=80, relative_fitness=1e-8, relative_rmse=1e-8))
        t = np.linalg.inv(result.transformation)
        distances = observed_to_model(observed[train], model, t)
        stats = distance_summary(distances)
        candidates.append(dict(base_T_model=t.tolist(), train=stats,
                               score_mm=(stats['median_mm']+stats['p90_mm'])/2))
    candidates.sort(key=lambda c: c['score_mm'])
    best = candidates[0]
    distances = observed_to_model(observed, model, best['base_T_model'])
    best.update(holdout=distance_summary(distances[~train]), all_points=distance_summary(distances),
                seed_count=len(seeds), refined_basins=len(candidates),
                alternate_training_scores_mm=[c['score_mm'] for c in candidates[1:]],
                alternate_poses=[c['base_T_model'] for c in candidates[1:]])
    return best, train, distances
