"""Support-surface fitting and object clustering shared by platforms."""

from dataclasses import dataclass
import numpy as np

from .types import PointCloudData


@dataclass(frozen=True)
class SupportSurfaceProfile:
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    voxel_size: float
    plane_distance: float
    ransac_iterations: int
    maximum_plane_candidates: int
    maximum_tilt_degrees: float
    designated_height: float
    maximum_object_height: float
    minimum_cluster_points: int
    cluster_distance: float
    random_seed: int

    def __post_init__(self) -> None:
        lower = np.asarray(self.workspace_min, dtype=np.float64)
        upper = np.asarray(self.workspace_max, dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
            raise ValueError("support-surface workspace bounds are invalid")
        metric = (
            self.voxel_size,
            self.plane_distance,
            self.maximum_object_height,
            self.cluster_distance,
        )
        if not np.all(np.isfinite(metric)) or any(value <= 0.0 for value in metric):
            raise ValueError("support-surface metric values must be positive")
        if not np.isfinite(self.designated_height):
            raise ValueError("designated support height must be finite")
        if not 0.0 <= self.maximum_tilt_degrees < 90.0:
            raise ValueError("maximum support tilt must be in [0, 90)")
        if (
            self.ransac_iterations <= 0
            or self.maximum_plane_candidates <= 0
            or self.minimum_cluster_points <= 0
        ):
            raise ValueError("support-surface counts must be positive")
        lower.setflags(write=False)
        upper.setflags(write=False)
        object.__setattr__(self, "workspace_min", lower)
        object.__setattr__(self, "workspace_max", upper)


@dataclass(frozen=True)
class SupportPlaneFit:
    cloud: PointCloudData
    plane: np.ndarray
    height_at_origin: float
    tilt_degrees: float
    inlier_ratio: float
    rmse: float


@dataclass(frozen=True)
class SupportSurfaceSegmentation:
    cropped: PointCloudData
    object_points: PointCloudData
    clusters: tuple[PointCloudData, ...]
    support_plane: np.ndarray
    fitted_plane: np.ndarray
    measured_support_height: float
    designated_support_height: float
    height_correction: float

    @property
    def measured_table_height(self) -> float:
        """Compatibility spelling for tabletop adapters."""
        return self.measured_support_height

    @property
    def designated_table_height(self) -> float:
        """Compatibility spelling for tabletop adapters."""
        return self.designated_support_height


class SupportSurfaceSegmenter:
    """Crop a scene, fit its support plane, and cluster raised objects."""

    def __init__(self, profile: SupportSurfaceProfile) -> None:
        self.profile = profile

    def fit_support_plane(self, cloud: PointCloudData) -> SupportPlaneFit:
        o3d = _open3d()
        profile = self.profile
        cropped = _crop_cloud(o3d, cloud, profile)
        points = np.asarray(cropped.points)
        o3d.utility.random.seed(profile.random_seed)
        remaining = np.arange(len(points), dtype=np.int64)
        selected_indices = None
        rejected_tilts = []
        for _ in range(profile.maximum_plane_candidates):
            if len(remaining) < 100:
                break
            candidate_cloud = _cloud(o3d, points[remaining])
            plane, local_indices = candidate_cloud.segment_plane(
                distance_threshold=profile.plane_distance,
                ransac_n=3,
                num_iterations=profile.ransac_iterations,
            )
            plane = np.asarray(plane, dtype=np.float64)
            norm = float(np.linalg.norm(plane[:3]))
            if norm <= 1e-12 or not local_indices:
                break
            plane /= norm
            if plane[2] < 0.0:
                plane *= -1.0
            tilt = float(np.rad2deg(np.arccos(np.clip(plane[2], -1.0, 1.0))))
            local_indices = np.asarray(local_indices, dtype=np.int64)
            if tilt <= profile.maximum_tilt_degrees:
                selected_indices = remaining[local_indices]
                break
            rejected_tilts.append(tilt)
            keep = np.ones(len(remaining), dtype=bool)
            keep[local_indices] = False
            remaining = remaining[keep]
        if selected_indices is None:
            detail = ", ".join(f"{value:.1f}" for value in rejected_tilts)
            raise ValueError(
                "no support-orientation plane found"
                + (f"; rejected tilts=[{detail}] degrees" if detail else "")
            )
        plane, selected_indices = _refine_plane(
            points,
            selected_indices,
            profile.plane_distance,
        )
        distances = points[selected_indices] @ plane[:3] + plane[3]
        if abs(plane[2]) <= 1e-12:
            raise ValueError("support plane has no finite Z intercept")
        height = float(-plane[3] / plane[2])
        tilt = float(np.rad2deg(np.arccos(np.clip(plane[2], -1.0, 1.0))))
        return SupportPlaneFit(
            cloud=PointCloudData(points),
            plane=_readonly(plane),
            height_at_origin=height,
            tilt_degrees=tilt,
            inlier_ratio=float(len(selected_indices) / len(points)),
            rmse=float(np.sqrt(np.mean(distances**2))),
        )

    def segment(self, cloud: PointCloudData) -> SupportSurfaceSegmentation:
        o3d = _open3d()
        profile = self.profile
        fit = self.fit_support_plane(cloud)
        correction = profile.designated_height - fit.height_at_origin
        corrected = PointCloudData(cloud.points + np.array([0.0, 0.0, correction]))
        cropped_cloud = _crop_cloud(o3d, corrected, profile)
        cropped = np.asarray(cropped_cloud.points)
        fitted_plane = fit.plane.copy()
        fitted_plane[3] -= fitted_plane[2] * correction
        heights = cropped @ fitted_plane[:3] + fitted_plane[3]
        keep = (heights > profile.plane_distance) & (
            heights < profile.maximum_object_height
        )
        object_cloud = _cloud(o3d, cropped[keep])
        if len(object_cloud.points) < profile.minimum_cluster_points:
            raise ValueError("no object points remain above the support surface")
        labels = np.asarray(
            object_cloud.cluster_dbscan(
                eps=profile.cluster_distance,
                min_points=profile.minimum_cluster_points,
                print_progress=False,
            )
        )
        clusters = []
        if labels.size:
            for label in range(int(labels.max()) + 1):
                indices = np.flatnonzero(labels == label)
                if len(indices) >= profile.minimum_cluster_points:
                    clusters.append(PointCloudData(cropped[keep][indices]))
        clusters.sort(key=len, reverse=True)
        if not clusters:
            raise ValueError("DBSCAN found no object clusters")
        support_plane = np.array(
            [0.0, 0.0, 1.0, -profile.designated_height]
        )
        return SupportSurfaceSegmentation(
            cropped=PointCloudData(cropped),
            object_points=PointCloudData(cropped[keep]),
            clusters=tuple(clusters),
            support_plane=_readonly(support_plane),
            fitted_plane=_readonly(fitted_plane),
            measured_support_height=fit.height_at_origin,
            designated_support_height=profile.designated_height,
            height_correction=correction,
        )


    def fit_table_plane(self, cloud: PointCloudData) -> SupportPlaneFit:
        """Compatibility entry point retained for tabletop callers."""
        return self.fit_support_plane(cloud)


def _crop_cloud(o3d, cloud: PointCloudData, profile: SupportSurfaceProfile):
    points = cloud.points
    keep = np.all(
        (points >= profile.workspace_min) & (points <= profile.workspace_max),
        axis=1,
    )
    cropped = _cloud(o3d, points[keep]).voxel_down_sample(profile.voxel_size)
    if len(cropped.points) < 100:
        raise ValueError(f"only {len(cropped.points)} points remain after workspace crop")
    cropped, _ = cropped.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return cropped


def _refine_plane(
    points: np.ndarray,
    indices: np.ndarray,
    distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(3):
        selected = points[indices]
        centroid = np.mean(selected, axis=0)
        _, _, axes = np.linalg.svd(selected - centroid, full_matrices=False)
        normal = axes[-1]
        if normal[2] < 0.0:
            normal *= -1.0
        plane = np.r_[normal, -float(normal @ centroid)]
        refined = np.flatnonzero(np.abs(points @ plane[:3] + plane[3]) <= distance)
        if np.array_equal(refined, indices):
            return plane, indices
        indices = refined
    selected = points[indices]
    centroid = np.mean(selected, axis=0)
    _, _, axes = np.linalg.svd(selected - centroid, full_matrices=False)
    normal = axes[-1]
    if normal[2] < 0.0:
        normal *= -1.0
    return np.r_[normal, -float(normal @ centroid)], indices


def _cloud(o3d, points: np.ndarray):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64, copy=True))
    return cloud


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _open3d():
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D is required for support-surface segmentation") from error
    return o3d
