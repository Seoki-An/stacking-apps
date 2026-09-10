"""Femto axial-depth calibration independent of ROS acquisition."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stacking_common.pointcloud import PointCloudData


CALIBRATION_METHOD = "femto_charuco_plane_robust_offset_v2"
LEGACY_CALIBRATION_METHODS = {"femto_charuco_plane_affine_depth_v1"}


@dataclass(frozen=True)
class FemtoDepthCalibration:
    path: Path
    scale: float
    offset: float
    depth_frame: str
    depth_mode: str
    camera_serial: str

    def correct(self, cloud: PointCloudData) -> PointCloudData:
        return PointCloudData(
            correct_points_along_rays(cloud.points, self.scale, self.offset)
        )


def load_femto_depth_calibration(
    path: str | Path,
    *,
    expected_camera_serial: str = "",
    expected_depth_mode: str = "",
) -> FemtoDepthCalibration:
    """Load and validate one recorded Femto calibration artifact."""

    calibration_path = Path(path).expanduser().resolve()
    if not calibration_path.is_file():
        raise FileNotFoundError(
            f"missing depth calibration file: {calibration_path}"
        )
    with np.load(calibration_path, allow_pickle=False) as data:
        method = str(_scalar(data, "method")).strip()
        scale = float(_scalar(data, "depth_scale"))
        offset = float(_scalar(data, "depth_offset_m"))
        depth_frame = str(_scalar(data, "depth_frame")).strip()
        depth_mode = str(_scalar(data, "depth_mode")).strip()
        serial = (
            str(_scalar(data, "camera_serial")).strip()
            if "camera_serial" in data
            else ""
        )
    if method != CALIBRATION_METHOD and method not in LEGACY_CALIBRATION_METHODS:
        raise ValueError(f"unsupported depth calibration method {method!r}")
    if method == CALIBRATION_METHOD and not np.isclose(scale, 1.0, atol=1e-12):
        raise ValueError("robust offset calibration must use scale=1")
    if not np.isfinite(scale) or not 0.8 <= scale <= 1.2:
        raise ValueError("depth calibration scale is invalid")
    if not np.isfinite(offset) or not -0.10 <= offset <= 0.10:
        raise ValueError("depth calibration offset is invalid")
    if not depth_frame or not depth_mode:
        raise ValueError("depth calibration frame and mode are required")
    if expected_camera_serial and serial != expected_camera_serial:
        raise ValueError("depth calibration camera serial mismatch")
    if expected_depth_mode and depth_mode != expected_depth_mode:
        raise ValueError("depth calibration mode mismatch")
    return FemtoDepthCalibration(
        calibration_path,
        scale,
        offset,
        depth_frame,
        depth_mode,
        serial,
    )


def correct_points_along_rays(
    points: np.ndarray,
    scale: float,
    offset: float,
) -> np.ndarray:
    """Correct optical-axis depth while preserving each camera ray."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("depth points must have shape (N, 3)")
    corrected = values.copy()
    valid = np.all(np.isfinite(values), axis=1) & (values[:, 2] > 0.0)
    corrected_depth = scale * values[valid, 2] + offset
    if np.any(corrected_depth <= 0.0):
        raise ValueError("depth correction produced non-positive depth")
    corrected[valid] *= (corrected_depth / values[valid, 2])[:, None]
    return corrected


def _scalar(data, key: str):
    if key not in data:
        raise ValueError(f"depth calibration is missing {key!r}")
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"depth calibration {key!r} must be scalar")
    return value.item()
