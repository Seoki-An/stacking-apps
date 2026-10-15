"""Known-object pose refinement from a completed execution-time hand scan."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from stacking_common.pointcloud import MultiViewVoxelMerge, PointCloudData, VoxelMergeConfig
from stacking_tabletop.acquisition.storage import write_xyz_pcd

from .capture import TabletopRawCapture
from .organized_depth import DepthFilterConfig


MODEL_NEIGHBORHOOD_M = 0.025


@dataclass(frozen=True)
class InHandIdentification:
    """Accepted or rejected tool-frame pose refinement and its evidence."""

    object_id: int
    initial_tool_T_object: np.ndarray
    identified_tool_T_object: np.ndarray
    accepted: bool
    reasons: tuple[str, ...]
    correction_translation_m: float
    correction_rotation_degrees: float
    target_coverage: float
    source_coverage: float
    rmse_m: float
    correspondences: int
    observation_points: int
    capture_directory: Path
    report_path: Path

    def report(self) -> dict:
        return {
            "status": "accepted" if self.accepted else "rejected",
            "object_id": self.object_id,
            "capture_directory": str(self.capture_directory),
            "report_path": str(self.report_path),
            "initial_tool_T_object": self.initial_tool_T_object.tolist(),
            "identified_tool_T_object": self.identified_tool_T_object.tolist(),
            "correction_translation_m": self.correction_translation_m,
            "correction_rotation_degrees": self.correction_rotation_degrees,
            "target_coverage": self.target_coverage,
            "source_coverage": self.source_coverage,
            "rmse_m": self.rmse_m,
            "correspondences": self.correspondences,
            "observation_points": self.observation_points,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class FailedInHandIdentification:
    """Recoverable processing failure after the physical scan was saved."""

    object_id: int
    capture_directory: Path
    report_path: Path
    error: str
    accepted: bool = False

    def report(self) -> dict:
        return {
            "status": "failed",
            "object_id": self.object_id,
            "capture_directory": str(self.capture_directory),
            "report_path": str(self.report_path),
            "error": self.error,
        }


def failed_inhand_identification(capture_directory, object_id, error):
    """Persist a recoverable identification failure beside the raw scan."""

    root = Path(capture_directory).expanduser().resolve()
    report_path = root / "inhand_identification.json"
    result = FailedInHandIdentification(
        object_id=int(object_id),
        capture_directory=root,
        report_path=report_path,
        error=str(error) or type(error).__name__,
    )
    report_path.write_text(json.dumps(result.report(), indent=2) + "\n")
    return result


def identify_inhand_capture(
    application,
    capture_directory,
    object_id,
    initial_tool_T_object,
    *,
    check=lambda: None,
):
    """Register the known held object and persist a reviewable result."""

    initial = np.asarray(initial_tool_T_object, dtype=np.float64)
    check()
    capture = TabletopRawCapture.load(capture_directory)
    profile = application.perception_profile
    capture_profile = profile.in_hand_capture
    views = capture.load_tool_views(
        DepthFilterConfig(
            minimum_temporal_support=capture_profile.minimum_temporal_support,
            temporal_tolerance=capture_profile.temporal_tolerance,
            enable_boundary_filter=capture_profile.enable_boundary_filter,
            discontinuity_threshold=capture_profile.discontinuity_threshold,
            discontinuity_radius_pixels=capture_profile.discontinuity_radius_pixels,
            boundary_dilation_pixels=capture_profile.boundary_dilation_pixels,
        ),
        capture_profile.crop_min,
        capture_profile.crop_max,
        progress=lambda _completed, _total: check(),
    )
    check()
    model = application.build_pcd_catalog().load(object_id)
    predicted = model.transformed(initial)
    tree = cKDTree(predicted.points)
    cropped = []
    for view in views:
        check()
        distance = tree.query(view.cloud.points, workers=-1)[0]
        cropped.append(
            PointCloudData(view.cloud.points[distance <= MODEL_NEIGHBORHOOD_M])
        )
    cropped = tuple(cropped)
    cropped = tuple(view for view in cropped if len(view))
    if not cropped:
        raise ValueError("in-hand scan contains no points near the planned held object")
    merge = MultiViewVoxelMerge(
        VoxelMergeConfig(
            capture_profile.reference_voxel_size,
            min(profile.reconstruction.minimum_view_support, len(cropped)),
        )
    ).merge(cropped, progress=lambda _completed, _total: check())
    if len(merge.cloud) < profile.in_hand_registration.minimum_points:
        raise ValueError(
            "in-hand object observation is too small: "
            f"{len(merge.cloud)} < {profile.in_hand_registration.minimum_points} points"
        )
    registration, decision = application.build_in_hand_registrar().register_with_decision(
        object_id,
        merge.cloud,
        initial,
    )
    check()
    root = Path(capture_directory).expanduser().resolve()
    cloud_path = root / "inhand_observation_tool0.pcd"
    report_path = root / "inhand_identification.json"
    write_xyz_pcd(cloud_path, merge.cloud.points)
    result = InHandIdentification(
        object_id=int(object_id),
        initial_tool_T_object=initial.copy(),
        identified_tool_T_object=registration.transform.copy(),
        accepted=decision.accepted,
        reasons=decision.reasons,
        correction_translation_m=decision.correction_translation,
        correction_rotation_degrees=decision.correction_rotation_degrees,
        target_coverage=registration.target_coverage,
        source_coverage=registration.source_coverage,
        rmse_m=registration.rmse,
        correspondences=registration.correspondences,
        observation_points=len(merge.cloud),
        capture_directory=root,
        report_path=report_path,
    )
    report_path.write_text(json.dumps({**result.report(), "merge": asdict(merge.stats)}, indent=2) + "\n")
    return result
