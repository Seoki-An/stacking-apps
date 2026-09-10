"""Validated loading of raw in-hand tabletop captures."""

import csv
import json
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stacking_common.pointcloud import PointCloudData, validate_rigid_transform

from .depth import correct_points_along_rays, load_femto_depth_calibration
from .organized_depth import DepthFilterConfig, DepthFilterStats, OrganizedDepthDataset


UR_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


@dataclass(frozen=True)
class TabletopCaptureView:
    filename: str
    capture_index: int
    joints: np.ndarray
    gripper_position: float
    base_T_camera: np.ndarray
    base_T_tool: np.ndarray

    def __post_init__(self) -> None:
        joints = np.asarray(self.joints, dtype=np.float64)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise ValueError("capture joints must contain six finite values")
        if self.capture_index < 0 or not np.isfinite(self.gripper_position):
            raise ValueError("capture index and gripper position are invalid")
        camera = validate_rigid_transform(self.base_T_camera, "base_T_camera")
        tool = validate_rigid_transform(self.base_T_tool, "base_T_tool")
        for value in (joints, camera, tool):
            value.setflags(write=False)
        object.__setattr__(self, "joints", joints)
        object.__setattr__(self, "base_T_camera", camera)
        object.__setattr__(self, "base_T_tool", tool)

    @property
    def tool_T_camera(self) -> np.ndarray:
        value = np.linalg.inv(self.base_T_tool) @ self.base_T_camera
        value.setflags(write=False)
        return value


@dataclass(frozen=True)
class LoadedCaptureView:
    metadata: TabletopCaptureView
    cloud: PointCloudData
    depth_filter: DepthFilterStats


class TabletopRawCapture:
    """Raw Femto capture with frame-explicit robot and camera transforms."""

    def __init__(
        self,
        directory: Path,
        camera: dict,
        views: tuple[TabletopCaptureView, ...],
        missing_transform_count: int,
    ) -> None:
        self.directory = directory
        self.camera = camera
        self.views = views
        self.missing_transform_count = missing_transform_count
        self.depth = OrganizedDepthDataset(
            directory,
            str(camera["cloud_frame"]),
            str(camera.get("depth_mode", "")),
        )

    @classmethod
    def load(cls, directory: str | Path) -> "TabletopRawCapture":
        root = Path(directory).expanduser().resolve()
        status = root / "capture_status.json"
        if status.is_file() and json.loads(status.read_text()).get("status") != "complete":
            raise ValueError("capture is incomplete; finish or explicitly recover it before processing")
        camera_path = root / "camera_frame.json"
        frames_path = root / "frames.csv"
        if not camera_path.is_file():
            raise FileNotFoundError(f"missing camera frame record: {camera_path}")
        if not frames_path.is_file():
            raise FileNotFoundError(f"missing capture frame table: {frames_path}")
        camera = json.loads(camera_path.read_text(encoding="utf-8"))
        expected_frame = str(camera.get("cloud_frame", "")).strip()
        if not expected_frame:
            raise ValueError("camera metadata has no cloud_frame")
        views = []
        missing = 0
        with frames_path.open(newline="", encoding="utf-8") as stream:
            for row_index, row in enumerate(csv.DictReader(stream)):
                filename = str(row.get("filename", "")).strip()
                if str(row.get("cloud_frame", "")).strip() != expected_frame:
                    raise ValueError(f"{filename}: cloud frame differs from camera metadata")
                if not (root / "raw_pcd" / filename).is_file():
                    raise FileNotFoundError(f"missing raw PCD: {root / 'raw_pcd' / filename}")
                if str(row.get("transform_status", "")).strip() != "exact":
                    missing += 1
                    continue
                base_T_camera = _matrix(row, "base_T_cloud")
                base_T_tool = _matrix(row, "base_T_tool")
                if base_T_camera is None or base_T_tool is None:
                    missing += 1
                    continue
                raw_index = str(row.get("view_index", "")).strip()
                raw_gripper = str(row.get("gripper_position", "")).strip()
                if not raw_gripper:
                    raise ValueError(f"{filename}: missing gripper position")
                views.append(
                    TabletopCaptureView(
                        filename=filename,
                        capture_index=int(raw_index) if raw_index else row_index,
                        joints=np.asarray([float(row[name]) for name in UR_JOINT_NAMES]),
                        gripper_position=float(raw_gripper),
                        base_T_camera=base_T_camera,
                        base_T_tool=base_T_tool,
                    )
                )
        if not views:
            raise ValueError("capture has no views with exact camera and tool transforms")
        indices = [view.capture_index for view in views]
        if any(second <= first for first, second in zip(indices, indices[1:])):
            raise ValueError("capture view indices must be strictly increasing")
        return cls(root, camera, tuple(views), missing)

    def load_tool_views(
        self,
        depth_filter: DepthFilterConfig,
        crop_min: np.ndarray,
        crop_max: np.ndarray,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[LoadedCaptureView, ...]:
        lower = np.asarray(crop_min, dtype=np.float64)
        upper = np.asarray(crop_max, dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
            raise ValueError("tool crop bounds are invalid")
        scale, offset = self._depth_correction()
        loaded = []
        total = len(self.views)
        for index, view in enumerate(self.views):
            points, stats = self.depth.points_for_view(view.filename, depth_filter)
            if scale != 1.0 or offset != 0.0:
                points = correct_points_along_rays(points, scale, offset)
            transform = view.tool_T_camera
            points = points @ transform[:3, :3].T + transform[:3, 3]
            inside = np.all((points >= lower) & (points <= upper), axis=1)
            loaded.append(LoadedCaptureView(view, PointCloudData(points[inside]), stats))
            if progress is not None:
                progress(index + 1, total)
        return tuple(loaded)

    def _depth_correction(self) -> tuple[float, float]:
        path = Path(str(self.camera.get("depth_calibration_file", ""))).expanduser()
        if not path.is_absolute():
            path = self.directory / path
        if path.is_file():
            expected = self.camera.get("depth_calibration_sha256")
            if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("capture depth calibration snapshot hash mismatch")
            value = load_femto_depth_calibration(
                path,
                expected_camera_serial=str(self.camera.get("camera_serial", "")),
                expected_depth_mode=str(self.camera.get("depth_mode", "")),
            )
            if value.depth_frame != str(self.camera["cloud_frame"]):
                raise ValueError("depth calibration frame differs from capture frame")
            return value.scale, value.offset
        if self.camera.get("depth_calibration_sha256"):
            raise FileNotFoundError(f"missing recorded depth calibration snapshot: {path}")
        scale = float(self.camera.get("depth_scale", 1.0))
        offset = float(self.camera.get("depth_offset_m", 0.0))
        if not np.isfinite(scale) or not 0.8 <= scale <= 1.2:
            raise ValueError("recorded depth scale is invalid")
        if not np.isfinite(offset) or not -0.10 <= offset <= 0.10:
            raise ValueError("recorded depth offset is invalid")
        return scale, offset


def _matrix(row: dict[str, str], prefix: str) -> np.ndarray | None:
    names = [f"{prefix}_{i}{j}" for i in range(4) for j in range(4)]
    values = [str(row.get(name, "")).strip() for name in names]
    if not all(values):
        return None
    return np.asarray([float(value) for value in values], dtype=np.float64).reshape(4, 4)
