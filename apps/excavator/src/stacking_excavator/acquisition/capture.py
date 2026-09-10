"""Capture-time geometry and purpose-specific stone_grab transfer crops."""

from dataclasses import dataclass, field
from types import MappingProxyType
import hashlib
import json
import re

import numpy as np

from stacking_common.pointcloud.cropping import box_mask
from stacking_common.pointcloud.types import validate_rigid_transform


KINDS = {"field", "inhand", "scene", "diagnostic"}
GRIPPER_FRAME = "stone_grab_geometry"


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("identifier must be a bounded filename-safe name")
    return value


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def owned_matrix(value, name):
    result = validate_rigid_transform(value, name)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CaptureFrame:
    frame_id: str
    sensor_frame: str
    timestamp_ns: int | None
    joint_timestamp_ns: int | None
    world_T_sensor: np.ndarray
    world_T_gripper: np.ndarray
    calibration_sha256: str
    points: np.ndarray
    joint_positions: tuple[float, ...]
    gripper_opening: float | None
    attributes: dict = field(default_factory=dict)
    stationary: bool = True
    timestamp_source: str = "emulated"

    def __post_init__(self):
        identifier(self.frame_id)
        identifier(self.sensor_frame)
        if self.timestamp_source == "recorded_unknown":
            if self.timestamp_ns is not None or self.joint_timestamp_ns is not None:
                raise ValueError("unknown recorded timestamps must be null, never invented")
        else:
            integer(self.timestamp_ns, "sensor timestamp", 1)
            integer(self.joint_timestamp_ns, "joint timestamp", 1)
        if self.timestamp_source not in {"emulated", "sensor", "receipt", "recorded_unknown"}:
            raise ValueError("unknown timestamp provenance")
        if not isinstance(self.stationary, bool):
            raise ValueError("stationary must be a bool")
        joints = np.asarray(self.joint_positions, dtype=float)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise ValueError("capture requires six finite measured joints in canonical order")
        object.__setattr__(self, "joint_positions", tuple(joints.tolist()))
        if self.gripper_opening is not None:
            if isinstance(self.gripper_opening, bool) or not np.isfinite(self.gripper_opening):
                raise ValueError("measured stone_grab opening must be finite when available")
            object.__setattr__(self, "gripper_opening", float(self.gripper_opening))
        if not re.fullmatch(r"[a-f0-9]{64}", self.calibration_sha256):
            raise ValueError("capture requires a calibration SHA-256")
        for name in ("world_T_sensor", "world_T_gripper"):
            object.__setattr__(self, name, owned_matrix(getattr(self, name), name))
        points = np.asarray(self.points)
        if (points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in "fiu"
                or not np.all(np.isfinite(points))):
            raise ValueError("capture XYZ must be finite numeric Nx3")
        points = points.copy()
        points.setflags(write=False)
        object.__setattr__(self, "points", points)
        attributes = {}
        for name, values in self.attributes.items():
            identifier(name)
            values = np.asarray(values)
            if (values.ndim < 1 or len(values) != len(points) or values.dtype.kind not in "fiub"
                    or not np.all(np.isfinite(values))):
                raise ValueError("capture channels must be finite numeric arrays aligned with XYZ")
            values = values.copy()
            values.setflags(write=False)
            attributes[name] = values
        object.__setattr__(self, "attributes", MappingProxyType(attributes))

    def metadata(self):
        return {"frame_id": self.frame_id, "sensor_frame": self.sensor_frame,
                "timestamp_ns": self.timestamp_ns, "joint_timestamp_ns": self.joint_timestamp_ns,
                "world_T_sensor": self.world_T_sensor.tolist(),
                "world_T_gripper": self.world_T_gripper.tolist(),
                "calibration_sha256": self.calibration_sha256,
                "joint_positions": self.joint_positions,
                "gripper_opening": self.gripper_opening,
                "stationary": self.stationary, "timestamp_source": self.timestamp_source}


@dataclass(frozen=True)
class CropRequest:
    kind: str
    reference_frame: str
    reference_T_box: np.ndarray
    half_extents: tuple[float, float, float]
    margins: tuple[float, float, float]
    calibration_sha256: str
    scene_revision: int
    object_ids: tuple[int, ...] = ()

    def __post_init__(self):
        if self.kind not in KINDS or self.reference_frame not in {"world", GRIPPER_FRAME}:
            raise ValueError("unsupported crop purpose or coordinate frame")
        if self.kind == "inhand" and self.reference_frame != GRIPPER_FRAME:
            raise ValueError("in-hand crop requires commissioned stone_grab geometry frame")
        integer(self.scene_revision, "scene revision")
        ids = tuple(self.object_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate crop object IDs")
        for value in ids:
            integer(value, "object ID")
        if self.kind in {"inhand", "field"} and len(ids) != 1:
            raise ValueError("field/in-hand identification requires one target ID")
        if not re.fullmatch(r"[a-f0-9]{64}", self.calibration_sha256):
            raise ValueError("crop requires a calibration SHA-256")
        object.__setattr__(self, "object_ids", ids)
        object.__setattr__(self, "reference_T_box", owned_matrix(self.reference_T_box, "reference_T_box"))
        for name in ("half_extents", "margins"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("crop extents/margins must be nonnegative finite XYZ")
            if name == "half_extents" and np.any(values == 0):
                raise ValueError("crop half extents must be positive")
            object.__setattr__(self, name, tuple(values.tolist()))

    def values(self):
        return {"kind": self.kind, "reference_frame": self.reference_frame,
                "reference_T_box": self.reference_T_box.tolist(),
                "half_extents": self.half_extents, "margins": self.margins,
                "calibration_sha256": self.calibration_sha256,
                "scene_revision": self.scene_revision, "object_ids": self.object_ids}

    def select(self, frame, *, maximum_timestamp_skew_ns):
        integer(maximum_timestamp_skew_ns, "maximum timestamp skew")
        if frame.calibration_sha256 != self.calibration_sha256:
            raise ValueError("crop/capture calibration mismatch")
        if not frame.stationary:
            raise ValueError("moving capture requires a commissioned synchronization adapter")
        if (frame.timestamp_source != "recorded_unknown" and
                abs(frame.timestamp_ns - frame.joint_timestamp_ns) > maximum_timestamp_skew_ns):
            raise ValueError("stale capture-time joint feedback")
        world_T_reference = frame.world_T_gripper if self.reference_frame == GRIPPER_FRAME else np.eye(4)
        box_T_sensor = np.linalg.inv(world_T_reference @ self.reference_T_box) @ frame.world_T_sensor
        return box_mask(frame.points, box_T_sensor,
                        np.asarray(self.half_extents) + self.margins)
