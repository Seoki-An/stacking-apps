"""Bounded dual-Femto registration and PCD fusion."""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from stacking_common.pointcloud import PointCloudData, RegistrationResult
from stacking_tabletop.config import (
    DualFemtoFusionProfile,
    FieldSegmentationProfile,
)

from .registration import Open3dRegistrationBackend
from .segmentation import TabletopFieldSegmenter


@dataclass(frozen=True)
class RecordedFemtoScan:
    cloud: PointCloudData
    camera_serial: str
    frame_id: str
    metadata: dict
    directory: Path


@dataclass(frozen=True)
class DualFemtoFusionResult:
    cloud: PointCloudData
    secondary_to_primary: np.ndarray
    registration: RegistrationResult
    table_plane: np.ndarray
    primary_registration_points: int
    secondary_registration_points: int


class DualFemtoFusion:
    """Fuse two depth-corrected clouds already expressed in one base frame."""

    def __init__(
        self,
        profile: DualFemtoFusionProfile,
        segmentation_profile: FieldSegmentationProfile,
    ) -> None:
        self.profile = profile
        self.segmentation_profile = segmentation_profile
        self.backend = Open3dRegistrationBackend(
            profile.registration,
            generalized=True,
            robust_kernel_scale=1.5,
        )

    def fuse(
        self,
        primary: PointCloudData,
        secondary: PointCloudData,
    ) -> DualFemtoFusionResult:
        lower = np.maximum(
            np.min(primary.points, axis=0),
            np.min(secondary.points, axis=0),
        )
        upper = np.minimum(
            np.max(primary.points, axis=0),
            np.max(secondary.points, axis=0),
        )
        if np.any(lower >= upper):
            raise ValueError("dual-Femto clouds have no shared volume")
        primary_shared = _crop(primary, lower, upper)
        secondary_shared = _crop(secondary, lower, upper)
        profile = self.profile
        if len(primary_shared) < profile.registration.minimum_points:
            raise ValueError("too few primary points in shared camera volume")
        if len(secondary_shared) < profile.registration.minimum_points:
            raise ValueError("too few secondary points in shared camera volume")
        plane_profile = replace(
            self.segmentation_profile,
            workspace_min=lower,
            workspace_max=upper,
            voxel_size=profile.table_voxel_size,
            table_distance=profile.table_distance,
        )
        fit = TabletopFieldSegmenter(plane_profile).fit_table_plane(
            primary_shared
        )
        primary_objects = _above_table(
            primary_shared,
            fit.plane,
            profile.table_distance,
            profile.maximum_object_height,
        )
        secondary_objects = _above_table(
            secondary_shared,
            fit.plane,
            profile.table_distance,
            profile.maximum_object_height,
        )
        minimum = profile.registration.minimum_points
        if len(primary_objects) < minimum or len(secondary_objects) < minimum:
            raise ValueError("too few above-table points for dual-Femto ICP")
        registration = self.backend.register(
            secondary_objects,
            primary_objects,
            np.eye(4),
        )
        translation = float(np.linalg.norm(registration.transform[:3, 3]))
        rotation = _rotation_degrees(registration.transform[:3, :3])
        overlap = min(
            registration.target_coverage,
            registration.source_coverage,
        )
        if (
            translation > profile.maximum_translation
            or rotation > profile.maximum_rotation_degrees
            or overlap < profile.minimum_symmetric_overlap
            or not np.isfinite(registration.rmse)
            or registration.rmse > profile.maximum_symmetric_rmse
        ):
            raise ValueError(
                "dual-Femto ICP failed bounded quality gates: "
                f"translation={translation:.4f}, rotation={rotation:.3f}, "
                f"overlap={overlap:.3f}, rmse={registration.rmse:.4f}"
            )
        aligned_secondary = secondary.transformed(registration.transform)
        merged = _voxel_downsample(
            PointCloudData(
                np.vstack([primary.points, aligned_secondary.points])
            ),
            profile.merge_voxel_size,
        )
        return DualFemtoFusionResult(
            cloud=merged,
            secondary_to_primary=registration.transform,
            registration=registration,
            table_plane=fit.plane,
            primary_registration_points=len(primary_objects),
            secondary_registration_points=len(secondary_objects),
        )

    def fuse_recorded(
        self,
        primary_directory: str | Path,
        secondary_directory: str | Path,
    ) -> DualFemtoFusionResult:
        primary = load_recorded_femto_scan(primary_directory)
        secondary = load_recorded_femto_scan(secondary_directory)
        if primary.frame_id != secondary.frame_id:
            raise ValueError("recorded Femto scans use different output frames")
        if primary.camera_serial == secondary.camera_serial:
            raise ValueError("dual-Femto fusion requires different camera serials")
        return self.fuse(primary.cloud, secondary.cloud)


def load_recorded_femto_scan(directory: str | Path) -> RecordedFemtoScan:
    """Load one depth-corrected recorded scan and validate its provenance."""

    scan_directory = Path(directory).expanduser().resolve()
    status = scan_directory / "capture_status.json"
    if status.is_file() and json.loads(status.read_text()).get("status") != "complete":
        raise ValueError("field capture is incomplete")
    metadata_path = scan_directory / "field_scan.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing field scan metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    frame_id = str(metadata.get("output_frame", "")).strip()
    serial = str(metadata.get("camera_serial", "")).strip()
    if not serial and metadata.get("fusion_method") == "dual_femto_bounded_multiscale_gicp":
        serials = metadata.get("camera_serials", [])
        if len(serials) == 2 and all(isinstance(value, str) and value.strip() for value in serials) and serials[0] != serials[1]:
            serial = "+".join(serials)
    if not frame_id or not serial:
        raise ValueError("field scan frame and camera serial are required")
    if not bool(metadata.get("fused_depth_corrected", False)):
        raise ValueError("field scan must be depth corrected before fusion")
    path = scan_directory / str(
        metadata.get("fused_file", "field_fused_base_link.pcd")
    )
    o3d = _open3d()
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError(f"recorded field cloud is empty: {path}")
    return RecordedFemtoScan(
        PointCloudData(points),
        serial,
        frame_id,
        metadata,
        scan_directory,
    )


def write_dual_femto_fusion(
    directory: str | Path,
    result: DualFemtoFusionResult,
    camera_serials: tuple[str, str],
    frame_id: str = "base_link",
) -> Path:
    """Persist one fused PCD and the bounded-registration provenance."""

    output = Path(directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "field_fused_base_link.pcd"
    o3d = _open3d()
    if not o3d.io.write_point_cloud(str(path), _cloud(o3d, result.cloud.points)):
        raise RuntimeError(f"failed to write fused point cloud: {path}")
    metadata = {
        "version": 1,
        "fusion_method": "dual_femto_bounded_multiscale_gicp",
        "output_frame": frame_id,
        "fused_file": path.name,
        "fused_depth_corrected": True,
        "camera_serials": list(camera_serials),
        "secondary_to_primary": result.secondary_to_primary.tolist(),
        "point_count": len(result.cloud),
        "registration": {
            "target_coverage": result.registration.target_coverage,
            "source_coverage": result.registration.source_coverage,
            "rmse": result.registration.rmse,
        },
    }
    (output / "field_scan.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return path


def _crop(cloud: PointCloudData, lower, upper) -> PointCloudData:
    keep = np.all(
        (cloud.points >= lower) & (cloud.points <= upper),
        axis=1,
    )
    return PointCloudData(cloud.points[keep])


def _above_table(
    cloud: PointCloudData,
    plane: np.ndarray,
    minimum_height: float,
    maximum_height: float,
) -> PointCloudData:
    heights = cloud.points @ plane[:3] + plane[3]
    keep = (heights > minimum_height) & (heights < maximum_height)
    return PointCloudData(cloud.points[keep])


def _voxel_downsample(cloud: PointCloudData, voxel_size: float) -> PointCloudData:
    o3d = _open3d()
    down = _cloud(o3d, cloud.points).voxel_down_sample(voxel_size)
    return PointCloudData(np.asarray(down.points))


def _rotation_degrees(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def _cloud(o3d, points: np.ndarray):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.array(points, dtype=np.float64, copy=True)
    )
    return cloud


def _open3d():
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D is required for dual-Femto fusion") from error
    return o3d
