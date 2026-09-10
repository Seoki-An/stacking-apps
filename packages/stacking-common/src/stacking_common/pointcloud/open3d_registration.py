"""Open3D implementation of the common registration contract."""

import numpy as np

from .registration import (
    MultiscaleRegistrationConfig,
    RegistrationResult,
    RegistrationScaleResult,
)
from .types import PointCloudData, validate_rigid_transform


class Open3dRegistrationBackend:
    """Run multiscale point-to-point ICP or generalized ICP."""

    def __init__(
        self,
        config: MultiscaleRegistrationConfig,
        *,
        generalized: bool = False,
        robust_kernel_scale: float | None = None,
    ) -> None:
        self.config = config
        self.generalized = generalized
        self.robust_kernel_scale = robust_kernel_scale

    def register(
        self,
        source: PointCloudData,
        target: PointCloudData,
        initial_transform: np.ndarray,
    ) -> RegistrationResult:
        o3d = _open3d()
        config = self.config
        if len(source) < config.minimum_points:
            raise ValueError("source point cloud is too small for registration")
        if len(target) < config.minimum_points:
            raise ValueError("target point cloud is too small for registration")
        source_cloud = _cloud(o3d, source.points)
        target_cloud = _cloud(o3d, target.points)
        registration = o3d.pipelines.registration
        transform = validate_rigid_transform(
            initial_transform,
            "initial registration transform",
        )
        scales = []
        correspondences = 0
        for voxel_size, max_iterations in zip(
            config.voxel_sizes,
            config.max_iterations,
        ):
            source_down = source_cloud.voxel_down_sample(voxel_size)
            target_down = target_cloud.voxel_down_sample(voxel_size)
            if len(source_down.points) < 3 or len(target_down.points) < 3:
                raise ValueError(
                    "registration scale contains fewer than three points"
                )
            criteria = registration.ICPConvergenceCriteria(
                max_iteration=max_iterations
            )
            distance = voxel_size * config.correspondence_distance_scale
            if self.generalized:
                kernel = (
                    None
                    if self.robust_kernel_scale is None
                    else registration.TukeyLoss(
                        voxel_size * self.robust_kernel_scale
                    )
                )
                estimation = (
                    registration.TransformationEstimationForGeneralizedICP()
                    if kernel is None
                    else registration.TransformationEstimationForGeneralizedICP(
                        1e-3,
                        kernel,
                    )
                )
                result = registration.registration_generalized_icp(
                    source_down,
                    target_down,
                    distance,
                    transform,
                    estimation,
                    criteria,
                )
            else:
                result = registration.registration_icp(
                    source_down,
                    target_down,
                    distance,
                    transform,
                    registration.TransformationEstimationPointToPoint(),
                    criteria,
                )
            transform = np.asarray(result.transformation, dtype=np.float64)
            correspondences = len(result.correspondence_set)
            scales.append(
                RegistrationScaleResult(
                    voxel_size=voxel_size,
                    fitness=float(result.fitness),
                    rmse=float(result.inlier_rmse),
                    correspondences=correspondences,
                )
            )
        target_coverage, source_coverage, rmse = _alignment_metrics(
            o3d,
            source_cloud,
            target_cloud,
            transform,
            config.evaluation_distance,
        )
        return RegistrationResult(
            transform=transform,
            target_coverage=target_coverage,
            source_coverage=source_coverage,
            rmse=rmse,
            correspondences=correspondences,
            scales=tuple(scales),
        )

    def evaluate(
        self,
        source: PointCloudData,
        target: PointCloudData,
        transform: np.ndarray,
        *,
        scales: tuple[RegistrationScaleResult, ...] = (),
        correspondences: int = 0,
    ) -> RegistrationResult:
        """Evaluate an existing transform with the configured metric distance."""

        o3d = _open3d()
        value = validate_rigid_transform(transform)
        target_coverage, source_coverage, rmse = _alignment_metrics(
            o3d,
            _cloud(o3d, source.points),
            _cloud(o3d, target.points),
            value,
            self.config.evaluation_distance,
        )
        return RegistrationResult(
            transform=value,
            target_coverage=target_coverage,
            source_coverage=source_coverage,
            rmse=rmse,
            correspondences=correspondences,
            scales=scales,
        )


def _cloud(o3d, points: np.ndarray):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.array(points, dtype=np.float64, copy=True)
    )
    return cloud


def _alignment_metrics(
    o3d,
    source,
    target,
    transform: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    aligned = _cloud(
        o3d,
        np.asarray(source.points) @ transform[:3, :3].T + transform[:3, 3],
    )
    source_distances = np.asarray(aligned.compute_point_cloud_distance(target))
    target_distances = np.asarray(target.compute_point_cloud_distance(aligned))
    source_coverage = float(np.mean(source_distances <= threshold))
    target_coverage = float(np.mean(target_distances <= threshold))
    inliers = np.concatenate(
        (
            source_distances[source_distances <= threshold],
            target_distances[target_distances <= threshold],
        )
    )
    rmse = float(np.sqrt(np.mean(inliers**2))) if len(inliers) else float("inf")
    return target_coverage, source_coverage, rmse


def _open3d():
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D is required for point-cloud registration") from error
    return o3d
