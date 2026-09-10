"""Load a frame-aware fixed-camera calibration for runtime ROS TF use."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


LEGACY_POSEID_METHOD = "d435_femto_simultaneous_charuco"
BASE_LINK_T_UR_BASE = np.diag([-1.0, -1.0, 1.0, 1.0])


@dataclass(frozen=True)
class CameraCalibration:
    parent_frame: str
    child_frame: str
    transform: np.ndarray
    camera: str
    method: str
    correction: str
    camera_serial: str = ""
    observation_frame: str = ""
    depth_frame: str = ""


@dataclass(frozen=True)
class TransformDifference:
    translation_m: float
    rotation_deg: float


def _string_value(value) -> str:
    array = np.asarray(value)
    return str(array.item() if array.shape == () else value)


def validate_transform(transform, name: str = "camera transform") -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation is not right-handed")
    return transform


def camera_cloud_transform_difference(
    calibration_parent_T_camera,
    camera_T_cloud,
    published_parent_T_cloud,
) -> TransformDifference:
    """Compare a camera artifact, composed to the cloud, with active ROS TF."""
    calibration_parent_T_camera = validate_transform(
        calibration_parent_T_camera,
        "calibration parent_T_camera",
    )
    camera_T_cloud = validate_transform(camera_T_cloud, "camera_T_cloud")
    published_parent_T_cloud = validate_transform(
        published_parent_T_cloud,
        "published parent_T_cloud",
    )
    expected_parent_T_cloud = calibration_parent_T_camera @ camera_T_cloud
    translation_m = float(
        np.linalg.norm(
            expected_parent_T_cloud[:3, 3]
            - published_parent_T_cloud[:3, 3]
        )
    )
    relative_rotation = (
        expected_parent_T_cloud[:3, :3].T
        @ published_parent_T_cloud[:3, :3]
    )
    cosine = np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
    rotation_deg = float(np.rad2deg(np.arccos(cosine)))
    return TransformDifference(translation_m, rotation_deg)


def _load_depth_T_color(
    path: Path | str,
) -> tuple[np.ndarray, str, str, str]:
    """Load the fixed depth/color extrinsic captured by depth calibration."""
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        required = ("capture_depth_T_color", "depth_frame", "color_frame")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                f"{path} cannot locate the depth camera: missing {missing}"
            )
        transforms = np.asarray(
            data["capture_depth_T_color"], dtype=np.float64
        )
        depth_frame = _string_value(data["depth_frame"]).strip()
        color_frame = _string_value(data["color_frame"]).strip()
        camera_serial = (
            _string_value(data["camera_serial"]).strip()
            if "camera_serial" in data
            else ""
        )

    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(
            "capture_depth_T_color must have shape (N, 4, 4)"
        )
    if len(transforms) == 0:
        raise ValueError("capture_depth_T_color cannot be empty")
    if not depth_frame or not color_frame:
        raise ValueError("depth/color calibration frame names cannot be empty")

    validated = np.asarray(
        [
            validate_transform(transform, "capture depth_T_color")
            for transform in transforms
        ]
    )
    reference = validated[0]
    if not np.allclose(validated, reference[None, :, :], atol=1e-7):
        raise ValueError(
            "capture_depth_T_color changed during calibration; the factory "
            "color/depth extrinsic must be static"
        )
    return reference, depth_frame, color_frame, camera_serial


def load_camera_calibration(
    path: Path | str,
    depth_calibration_path: Path | str | None = None,
    expected_camera_serial: str = "",
) -> CameraCalibration:
    """Load PoseID and optionally relocate its mount to the depth origin.

    PoseID observes the Femto color optical frame.  In the color-only
    calibration profile the Orbbec driver places ``camera_link`` at that color
    origin, while the depth/point-cloud profile places the same named frame at
    the depth origin.  The depth calibration snapshots the factory
    ``depth_T_color`` transform, which lets us relocate the published mount
    origin without changing its axes.
    """
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        if "T_base_to_camera" in data:
            transform = data["T_base_to_camera"]
        elif "T_base_to_cam2" in data:
            transform = data["T_base_to_cam2"]
        else:
            raise ValueError(f"{path} has no supported camera transform")

        if "parent_frame" not in data or "child_frame" not in data:
            raise ValueError(
                f"{path} has no ROS frame metadata. Legacy poseid calibration "
                "must be re-exported with the exact calibrated camera frame."
            )
        parent_frame = _string_value(data["parent_frame"]).strip()
        child_frame = _string_value(data["child_frame"]).strip()
        camera = (
            _string_value(data["camera"]).strip()
            if "camera" in data
            else "unknown"
        )
        camera_serial = (
            _string_value(data["camera_serial"]).strip()
            if "camera_serial" in data
            else ""
        )
        method = (
            _string_value(data["method"]).strip()
            if "method" in data
            else "unknown"
        )
        ee_pose_frame = (
            _string_value(data["ee_pose_frame"]).strip()
            if "ee_pose_frame" in data
            else ""
        )
        observation_frame = (
            _string_value(data["observation_frame"]).strip()
            if "observation_frame" in data
            else ""
        )
        base_T_observation = (
            np.asarray(data["T_base_to_observation"], dtype=np.float64)
            if "T_base_to_observation" in data
            else None
        )
    if not parent_frame or not child_frame:
        raise ValueError("camera calibration frame names cannot be empty")

    correction = ""
    if (
        parent_frame == "base_link"
        and method == LEGACY_POSEID_METHOD
        and ee_pose_frame in ("", "base")
    ):
        transform = BASE_LINK_T_UR_BASE @ transform
        if base_T_observation is not None:
            base_T_observation = BASE_LINK_T_UR_BASE @ base_T_observation
        correction = "legacy_ur_base_to_base_link"

    depth_frame = ""
    if depth_calibration_path is not None and str(depth_calibration_path).strip():
        if base_T_observation is None or not observation_frame:
            raise ValueError(
                f"{path} must contain T_base_to_observation and "
                "observation_frame to compose the Femto depth origin"
            )
        if child_frame != "camera_link":
            raise ValueError(
                "Femto depth-origin relocation requires child_frame "
                f"'camera_link', got {child_frame!r}"
            )
        (
            depth_T_color,
            depth_frame,
            color_frame,
            depth_camera_serial,
        ) = _load_depth_T_color(depth_calibration_path)
        if camera_serial != depth_camera_serial:
            raise ValueError(
                "Pose and depth calibration camera serials differ: "
                f"{camera_serial!r} != {depth_camera_serial!r}"
            )
        if observation_frame != color_frame:
            raise ValueError(
                "PoseID observation frame and depth-calibration color frame "
                f"differ: {observation_frame!r} != {color_frame!r}"
            )

        base_T_observation = validate_transform(
            base_T_observation,
            "base_T_observation",
        )
        base_T_depth = base_T_observation @ np.linalg.inv(depth_T_color)
        transform = np.asarray(transform, dtype=np.float64).copy()
        transform[:3, 3] = base_T_depth[:3, 3]
        correction = ",".join(
            item
            for item in (
                correction,
                "femto_color_to_depth_origin",
            )
            if item
        )

    expected_camera_serial = expected_camera_serial.strip()
    if expected_camera_serial and camera_serial != expected_camera_serial:
        raise ValueError(
            "Camera calibration serial mismatch: "
            f"expected {expected_camera_serial!r}, got {camera_serial!r}"
        )

    return CameraCalibration(
        parent_frame=parent_frame,
        child_frame=child_frame,
        transform=validate_transform(transform),
        camera=camera,
        method=method,
        correction=correction,
        camera_serial=camera_serial,
        observation_frame=observation_frame,
        depth_frame=depth_frame,
    )
