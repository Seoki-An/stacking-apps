"""Backend-neutral values consumed by visualization frontends."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stacking_common.domain import CompositeTargetGeometry, DirectionalTaper, TargetComponent, TargetGeometry


@dataclass(frozen=True)
class CameraView:
    """A portable perspective-camera pose."""

    eye: np.ndarray
    look_at: np.ndarray
    up: np.ndarray
    vertical_fov_degrees: float = 45.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "eye", _vector3(self.eye, "camera eye"))
        object.__setattr__(
            self, "look_at", _vector3(self.look_at, "camera look-at point")
        )
        up = _vector3(self.up, "camera up vector")
        if np.linalg.norm(up) <= 1e-12:
            raise ValueError("camera up vector must have non-zero norm")
        if np.linalg.norm(self.eye - self.look_at) <= 1e-12:
            raise ValueError("camera eye and look-at point must differ")
        fov = float(self.vertical_fov_degrees)
        if not np.isfinite(fov) or not 0.0 < fov < 180.0:
            raise ValueError("camera vertical field of view must be between 0 and 180")
        object.__setattr__(self, "up", up)
        object.__setattr__(self, "vertical_fov_degrees", fov)


@dataclass(frozen=True)
class TargetVolume:
    """Minimal target geometry shared by planning and debug viewers."""

    origin: np.ndarray
    width: float
    length: float
    height: float
    structure_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    taper: DirectionalTaper = field(default_factory=DirectionalTaper)

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", _vector3(self.origin, "target origin"))
        object.__setattr__(
            self,
            "structure_offset",
            _vector3(self.structure_offset, "target structure offset"),
        )
        for name in ("width", "length", "height"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"target {name} must be positive and finite")
            object.__setattr__(self, name, value)
        if not isinstance(self.taper, DirectionalTaper):
            raise TypeError("target taper must be a DirectionalTaper")

    @property
    def points(self) -> np.ndarray:
        """Eight shell corners, starting at the floor rather than below it."""
        return TargetGeometry(
            width=self.width,
            length=self.length,
            height=self.height,
            origin=self.origin + self.structure_offset,
            taper=self.taper,
            support_overlap=0.0,
        ).points


@dataclass(frozen=True)
class CompositeTargetVolume:
    """Explicit local component meshes/poses, never a hull across components."""

    components: tuple[TargetComponent, ...]
    structure_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        geometry = CompositeTargetGeometry(tuple(self.components))
        object.__setattr__(self, "components", geometry.components)
        object.__setattr__(self, "structure_offset", _vector3(self.structure_offset, "target structure offset"))

    @property
    def points(self):
        return CompositeTargetGeometry(self.components).points + self.structure_offset

    @property
    def faces(self):
        return CompositeTargetGeometry(self.components).faces


TargetDisplay = TargetVolume | CompositeTargetVolume


def target_display(target, *, structure_offset=(0., 0., 0.)) -> TargetDisplay:
    """Project physical target geometry without changing the legacy shell."""
    if isinstance(target, CompositeTargetGeometry):
        return CompositeTargetVolume(target.components, structure_offset)
    return TargetVolume(target.origin, target.width, target.length, target.height,
                        structure_offset=structure_offset, taper=target.taper)


@dataclass(frozen=True)
class SceneObjectPose:
    """One named object transform in a canonical visualization scene."""

    object_id: int
    pose: np.ndarray
    object_index: int | None = None

    def __post_init__(self) -> None:
        object_id = int(self.object_id)
        object_index = (
            None if self.object_index is None else int(self.object_index)
        )
        if object_id < 0:
            raise ValueError("scene object ID must be non-negative")
        if object_index is not None and object_index < 0:
            raise ValueError("scene object index must be non-negative")
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "object_index", object_index)
        object.__setattr__(self, "pose", _pose(self.pose))


@dataclass(frozen=True)
class SceneSnapshot:
    """Static object layout at one visualization state."""

    frame_id: str
    objects: tuple[SceneObjectPose, ...]

    def __post_init__(self) -> None:
        frame_id = str(self.frame_id).strip()
        objects = tuple(self.objects)
        if not frame_id:
            raise ValueError("scene frame ID is required")
        if not all(isinstance(value, SceneObjectPose) for value in objects):
            raise TypeError("scene objects must be SceneObjectPose values")
        object_ids = [value.object_id for value in objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("scene object IDs must be unique")
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "objects", objects)

    def object_pose(self, object_id: int) -> np.ndarray:
        """Return an owned pose for one object."""

        for value in self.objects:
            if value.object_id == int(object_id):
                return value.pose.copy()
        raise KeyError(object_id)


@dataclass(frozen=True)
class PoseTrack:
    """Object motion with a hold-last-pose replay rule."""

    object_id: int
    poses: np.ndarray
    times: np.ndarray | None = None

    def __post_init__(self) -> None:
        object_id = int(self.object_id)
        poses = np.asarray(self.poses, dtype=float)
        if object_id < 0:
            raise ValueError("pose-track object ID must be non-negative")
        if poses.ndim != 2 or poses.shape[1:] != (7,) or len(poses) == 0:
            raise ValueError("pose track must have shape (N, 7), N >= 1")
        if not np.all(np.isfinite(poses)):
            raise ValueError("pose must contain only finite values")
        if np.any(np.linalg.norm(poses[:, 3:], axis=1) <= 1e-12):
            raise ValueError("pose quaternion must have non-zero norm")
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "poses", _readonly(poses))
        if self.times is not None:
            times = np.asarray(self.times, dtype=float)
            if (times.shape != (len(poses),) or not np.all(np.isfinite(times))
                    or np.any(times < 0) or np.any(np.diff(times) <= 0)):
                raise ValueError("pose-track times must be increasing seconds, one per pose")
            object.__setattr__(self, "times", _readonly(times))

    def pose_at_time(self, time: float) -> np.ndarray:
        """Interpolate translation and shortest-arc quaternion rotation."""
        if self.times is None:
            raise ValueError("pose track has no timestamps")
        if not np.isfinite(time):
            raise ValueError("replay time must be finite")
        right = int(np.searchsorted(self.times, time, side="right"))
        if right == 0 or right == len(self.times):
            return self.poses[0 if right == 0 else -1].copy()
        left = right - 1
        fraction = (time - self.times[left]) / (self.times[right] - self.times[left])
        start, end = self.poses[left], self.poses[right]
        pose = (1 - fraction) * start + fraction * end
        q0, q1 = start[3:].copy(), end[3:].copy()
        q0 /= np.linalg.norm(q0)
        q1 /= np.linalg.norm(q1)
        dot = float(np.dot(q0, q1))
        if dot < 0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            quaternion = (1 - fraction) * q0 + fraction * q1
        else:
            angle = np.arccos(np.clip(dot, -1, 1))
            quaternion = (np.sin((1 - fraction) * angle) * q0
                          + np.sin(fraction * angle) * q1) / np.sin(angle)
        pose[3:] = quaternion / np.linalg.norm(quaternion)
        return pose

    def pose_at(self, frame: int) -> np.ndarray:
        """Return an owned pose, holding the last sample for shorter tracks."""

        frame = int(frame)
        if frame < 0:
            raise IndexError("timeline frame must be non-negative")
        return self.poses[min(frame, len(self.poses) - 1)].copy()


@dataclass(frozen=True)
class SceneTimeline:
    """Synchronized object tracks for simulation or execution replay."""

    tracks: tuple[PoseTrack, ...]

    def __post_init__(self) -> None:
        tracks = tuple(self.tracks)
        if not all(isinstance(value, PoseTrack) for value in tracks):
            raise TypeError("timeline tracks must be PoseTrack values")
        object_ids = [value.object_id for value in tracks]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("timeline object IDs must be unique")
        object.__setattr__(self, "tracks", tracks)

    @property
    def frame_count(self) -> int:
        return max((len(value.poses) for value in self.tracks), default=0)

    def poses_at(self, frame: int) -> tuple[SceneObjectPose, ...]:
        if frame < 0 or frame >= self.frame_count:
            raise IndexError("timeline frame is outside the replay")
        return tuple(
            SceneObjectPose(value.object_id, value.pose_at(frame))
            for value in self.tracks
        )

    def video_times(self, fps: float) -> np.ndarray | None:
        """Output times including a final endpoint frame; None for untimed data."""
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("video FPS must be finite and positive")
        if not self.tracks or any(track.times is None for track in self.tracks):
            return None
        duration = max(float(track.times[-1]) for track in self.tracks)
        count = int(np.ceil(duration * fps - 1e-9)) + 1
        return np.minimum(np.arange(max(count, 1), dtype=float) / fps, duration)

    def poses_at_time(self, time: float) -> tuple[SceneObjectPose, ...]:
        return tuple(
            SceneObjectPose(track.object_id, track.pose_at_time(time))
            for track in self.tracks
        )

    def track(self, object_id: int) -> PoseTrack:
        for value in self.tracks:
            if value.object_id == int(object_id):
                return value
        raise KeyError(object_id)


@dataclass(frozen=True)
class ContactMarker:
    """Post-simulation contact geometry."""

    object_id_first: int | None
    object_id_second: int | None
    point: np.ndarray
    normal: np.ndarray
    gap: float

    def __post_init__(self) -> None:
        for name in ("object_id_first", "object_id_second"):
            value = getattr(self, name)
            value = None if value is None else int(value)
            if value is not None and value < 0:
                raise ValueError("contact object IDs must be non-negative")
            object.__setattr__(self, name, value)
        normal = _vector3(self.normal, "contact normal")
        if np.linalg.norm(normal) <= 1e-12:
            raise ValueError("contact normal must have non-zero norm")
        gap = float(self.gap)
        if not np.isfinite(gap):
            raise ValueError("contact gap must be finite")
        object.__setattr__(self, "point", _vector3(self.point, "contact point"))
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "gap", gap)


@dataclass(frozen=True)
class ForceMarker:
    """Pose-solve force, intentionally separate from simulation contacts."""

    object_id: int
    role: str
    point: np.ndarray
    normal: np.ndarray
    force: np.ndarray

    def __post_init__(self) -> None:
        object_id = int(self.object_id)
        role = str(self.role).strip()
        if object_id < 0:
            raise ValueError("force-marker object ID must be non-negative")
        if not role:
            raise ValueError("force-marker role is required")
        normal = _vector3(self.normal, "force normal")
        if np.linalg.norm(normal) <= 1e-12:
            raise ValueError("force normal must have non-zero norm")
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "point", _vector3(self.point, "force point"))
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "force", _vector3(self.force, "force"))


@dataclass(frozen=True)
class ScoreField:
    """A state-scoped planar score field and its selected locations."""

    field_id: str
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    values: np.ndarray
    valid: np.ndarray
    selected_xy: np.ndarray
    scope: str = "state"

    def __post_init__(self) -> None:
        field_id = str(self.field_id).strip()
        scope = str(self.scope).strip()
        x_coordinates = _axis(self.x_coordinates, "score-field x coordinates")
        y_coordinates = _axis(self.y_coordinates, "score-field y coordinates")
        values = np.asarray(self.values, dtype=float)
        valid = np.asarray(self.valid, dtype=bool)
        selected_xy = np.asarray(self.selected_xy, dtype=float)
        expected = (len(y_coordinates), len(x_coordinates))
        if not field_id:
            raise ValueError("score-field ID is required")
        if scope != "state":
            raise ValueError("score fields must be scoped to one search state")
        if values.shape != expected or valid.shape != expected:
            raise ValueError("score values and validity must match their axes")
        if not np.all(np.isfinite(values)):
            raise ValueError("score values must be finite")
        if selected_xy.size == 0:
            selected_xy = np.empty((0, 2), dtype=float)
        if selected_xy.ndim != 2 or selected_xy.shape[1] != 2:
            raise ValueError("selected score locations must have shape (N, 2)")
        if not np.all(np.isfinite(selected_xy)):
            raise ValueError("selected score locations must be finite")
        object.__setattr__(self, "field_id", field_id)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "x_coordinates", x_coordinates)
        object.__setattr__(self, "y_coordinates", y_coordinates)
        object.__setattr__(self, "values", _readonly(values))
        object.__setattr__(self, "valid", _readonly(valid))
        object.__setattr__(self, "selected_xy", _readonly(selected_xy))


def _axis(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(result)) or np.any(np.diff(result) <= 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return _readonly(result)


def _pose(value) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    _validate_pose(result)
    return _readonly(result)


def _validate_pose(value: np.ndarray) -> None:
    if value.shape != (7,):
        raise ValueError("pose must have shape (7,)")
    if not np.all(np.isfinite(value)):
        raise ValueError("pose must contain only finite values")
    if np.linalg.norm(value[3:]) <= 1e-12:
        raise ValueError("pose quaternion must have non-zero norm")


def _vector3(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    return _readonly(result)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result
