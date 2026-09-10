"""Headless point-cloud cleanup at explicitly configured metric scales."""

from collections.abc import Callable

import numpy as np

from stacking_common.pointcloud import PointCloudData

from .config import ReconstructionCleanupProfile
from .types import ReconstructionOperatorOptions


class PointCloudCleaner:
    def __init__(self, profile: ReconstructionCleanupProfile) -> None:
        self.profile = profile

    def clean(
        self,
        cloud: PointCloudData,
        options: ReconstructionOperatorOptions,
        *,
        manual_remover: Callable[[PointCloudData, bool], PointCloudData] | None = None,
    ) -> PointCloudData:
        o3d = _open3d()
        value = _cloud(o3d, cloud)
        if options.remove_statistical_outliers:
            value, _ = value.remove_statistical_outlier(
                nb_neighbors=self.profile.statistical_neighbors,
                std_ratio=self.profile.statistical_std_ratio,
            )
        if options.remove_detached_clusters:
            value = _remove_detached_clusters(value, self.profile)
        if options.remove_radial_fringe:
            value = _remove_radial_fringe(value, self.profile)
        if options.keep_largest_cluster:
            value = value.voxel_down_sample(self.profile.cluster_voxel_size)
            labels = np.asarray(
                value.cluster_dbscan(
                    eps=self.profile.cluster_eps,
                    min_points=self.profile.cluster_min_points,
                    print_progress=False,
                )
            )
            valid = labels[labels >= 0]
            if not len(valid):
                raise ValueError("largest-cluster cleanup found no valid cluster")
            largest = int(np.bincount(valid).argmax())
            value = value.select_by_index(np.flatnonzero(labels == largest))
        result = PointCloudData(np.asarray(value.points))
        if options.manual_remove:
            if manual_remover is None:
                raise RuntimeError(
                    "manual cleanup was requested without an interaction adapter"
                )
            result = manual_remover(result, options.manual_left_click)
        if len(result) < 4:
            raise ValueError("cleanup retained fewer than four object points")
        return result


def _remove_radial_fringe(value, profile: ReconstructionCleanupProfile):
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:
        raise RuntimeError("SciPy is required for radial fringe cleanup") from error
    points = np.asarray(value.points)
    if not len(points):
        return value
    center = np.quantile(points, 0.5, axis=0)
    vectors = points - center
    radii = np.linalg.norm(vectors, axis=1)
    valid = radii > 1e-9
    indices = np.flatnonzero(valid)
    if len(indices) < 3:
        return value
    directions = vectors[valid] / radii[valid, None]
    count = min(profile.fringe_neighbors, len(indices))
    neighbors = cKDTree(directions).query(directions, k=count)[1]
    if count == 1:
        neighbors = neighbors[:, None]
    reference = np.quantile(
        radii[indices][neighbors],
        profile.fringe_reference_quantile,
        axis=1,
    )
    remove = radii[indices] > reference + profile.fringe_margin
    keep = np.ones(len(points), dtype=bool)
    keep[indices[remove]] = False
    return value.select_by_index(np.flatnonzero(keep))


def _remove_detached_clusters(value, profile: ReconstructionCleanupProfile):
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:
        raise RuntimeError("SciPy is required for detached-cluster cleanup") from error
    if not len(value.points):
        return value
    down = value.voxel_down_sample(profile.detached_voxel_size)
    labels = np.asarray(
        down.cluster_dbscan(
            eps=profile.detached_eps,
            min_points=profile.detached_minimum_points,
            print_progress=False,
        )
    )
    valid = labels[labels >= 0]
    if not len(valid):
        return value
    points = np.asarray(down.points)
    counts = np.bincount(valid)
    largest = int(np.argmax(counts))
    largest_points = points[labels == largest]
    tree = cKDTree(largest_points)
    kept = [largest]
    for label, count in enumerate(counts):
        if label == largest:
            continue
        component = points[labels == label]
        distance = float(np.min(tree.query(component, k=1)[0]))
        attached = (
            distance <= profile.detached_attach_distance
            and count >= profile.detached_minimum_attached_size
        )
        large = count >= counts[largest] * profile.detached_keep_ratio
        if attached or large:
            kept.append(label)
    retained = points[np.isin(labels, kept)]
    if not len(retained):
        return value
    distance = cKDTree(retained).query(np.asarray(value.points), k=1)[0]
    threshold = max(
        profile.detached_voxel_size * 1.75,
        profile.detached_eps * 0.25,
    )
    return value.select_by_index(np.flatnonzero(distance <= threshold))


def _cloud(o3d, cloud: PointCloudData):
    value = o3d.geometry.PointCloud()
    value.points = o3d.utility.Vector3dVector(np.array(cloud.points, copy=True))
    return value


def _open3d():
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D is required for reconstruction cleanup") from error
    return o3d
