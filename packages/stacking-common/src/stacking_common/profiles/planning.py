"""Typed platform scene and model configuration for motion planning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from stacking_common.planning import FrameTransform
from .angles import resolve_degree_fields


@dataclass(frozen=True)
class PlanningObstacle:
    """One finite world-frame collision box."""

    obstacle_id: str
    minimum: np.ndarray
    maximum: np.ndarray

    def __post_init__(self) -> None:
        obstacle_id = str(self.obstacle_id).strip()
        minimum = np.asarray(self.minimum, dtype=float)
        maximum = np.asarray(self.maximum, dtype=float)
        if not obstacle_id:
            raise ValueError("planning obstacle ID must not be empty")
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("planning obstacle bounds must be three-vectors")
        if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
            raise ValueError("planning obstacle bounds must be finite")
        if np.any(maximum <= minimum):
            raise ValueError("planning obstacle maximum must exceed its minimum")
        object.__setattr__(self, "obstacle_id", obstacle_id)
        object.__setattr__(self, "minimum", _readonly(minimum))
        object.__setattr__(self, "maximum", _readonly(maximum))

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.minimum + self.maximum)

    @property
    def size(self) -> np.ndarray:
        return self.maximum - self.minimum


@dataclass(frozen=True)
class RobotPlanningProfile:
    urdf: Path
    tool_link: str
    home_positions: np.ndarray
    link_from_tool: np.ndarray
    gripper_opening: float

    def __post_init__(self) -> None:
        tool_link = str(self.tool_link).strip()
        positions = np.asarray(self.home_positions, dtype=float)
        opening = float(self.gripper_opening)
        if not tool_link:
            raise ValueError("planning robot tool link must not be empty")
        if positions.ndim != 1 or positions.size == 0:
            raise ValueError("planning robot home positions must be a vector")
        if not np.all(np.isfinite(positions)) or not np.isfinite(opening):
            raise ValueError("planning robot values must be finite")
        object.__setattr__(self, "urdf", Path(self.urdf))
        object.__setattr__(self, "tool_link", tool_link)
        object.__setattr__(self, "home_positions", _readonly(positions))
        object.__setattr__(
            self,
            "link_from_tool",
            _readonly(_pose(self.link_from_tool, "robot link_from_tool")),
        )
        object.__setattr__(self, "gripper_opening", opening)


@dataclass(frozen=True)
class GripperPlanningProfile:
    urdf: Path
    root_link: str
    grasp_from_root: np.ndarray
    opening_offset: np.ndarray
    opening_direction: np.ndarray
    opening_lower: float
    opening_upper: float

    def __post_init__(self) -> None:
        root_link = str(self.root_link).strip()
        offset = np.asarray(self.opening_offset, dtype=float)
        direction = np.asarray(self.opening_direction, dtype=float)
        lower = float(self.opening_lower)
        upper = float(self.opening_upper)
        if not root_link:
            raise ValueError("planning gripper root link must not be empty")
        if offset.ndim != 1 or offset.shape != direction.shape:
            raise ValueError("gripper opening vectors must have matching shapes")
        if offset.size == 0 or not np.all(np.isfinite(offset)):
            raise ValueError("gripper opening offset must be a finite vector")
        if not np.all(np.isfinite(direction)) or not np.any(direction):
            raise ValueError("gripper opening direction must be finite and nonzero")
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ValueError("gripper opening limits are invalid")
        object.__setattr__(self, "urdf", Path(self.urdf))
        object.__setattr__(self, "root_link", root_link)
        object.__setattr__(
            self,
            "grasp_from_root",
            _readonly(_pose(self.grasp_from_root, "grasp_from_root")),
        )
        object.__setattr__(self, "opening_offset", _readonly(offset))
        object.__setattr__(self, "opening_direction", _readonly(direction))
        object.__setattr__(self, "opening_lower", lower)
        object.__setattr__(self, "opening_upper", upper)


@dataclass(frozen=True)
class PlanningProfile:
    """Resolved world, robot, gripper, and collision-scene boundary."""

    world_T_stack: FrameTransform
    frame_id: int
    robot: RobotPlanningProfile
    gripper: GripperPlanningProfile
    obstacle_friction: float
    handoff_position: np.ndarray
    obstacles: tuple[PlanningObstacle, ...]

    def __post_init__(self) -> None:
        frame_id = int(self.frame_id)
        friction = float(self.obstacle_friction)
        handoff = np.asarray(self.handoff_position, dtype=float)
        obstacles = tuple(self.obstacles)
        if frame_id < 0:
            raise ValueError("planning frame ID must be non-negative")
        if not np.isfinite(friction):
            raise ValueError("planning obstacle friction must be finite")
        if friction < 0.0:
            raise ValueError("planning obstacle friction must be non-negative")
        if handoff.shape != (3,) or not np.all(np.isfinite(handoff)):
            raise ValueError("planning handoff position must be a finite XYZ vector")
        if not all(isinstance(value, PlanningObstacle) for value in obstacles):
            raise TypeError("planning obstacles have the wrong type")
        obstacle_ids = [value.obstacle_id for value in obstacles]
        if len(obstacle_ids) != len(set(obstacle_ids)):
            raise ValueError("planning obstacle IDs must be unique")
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "obstacle_friction", friction)
        object.__setattr__(self, "handoff_position", _readonly(handoff))
        object.__setattr__(self, "obstacles", obstacles)


def load_planning_profile(path: str | Path) -> PlanningProfile:
    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(f"missing planning profile: {profile_path}")
    values = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise TypeError("planning profile root must be a mapping")
    if values.get("units") != "si":
        raise ValueError("planning profile units must be 'si'")
    frames = _mapping(values, "frames")
    robot = resolve_degree_fields(_mapping(values, "robot"), ("home_positions", "gripper_opening"))
    gripper = resolve_degree_fields(_mapping(values, "gripper"), ("opening_offset", "opening_limits"))
    scene = _mapping(values, "scene")
    raw_obstacles = _required(scene, "obstacles")
    if not isinstance(raw_obstacles, Sequence) or isinstance(raw_obstacles, str):
        raise TypeError("planning obstacles must be a sequence")
    limits = _sequence(_required(gripper, "opening_limits"), "opening_limits", 2)
    return PlanningProfile(
        world_T_stack=FrameTransform(
            str(_required(frames, "world")),
            str(_required(frames, "stack")),
            _required(frames, "world_T_stack"),
        ),
        frame_id=int(_required(scene, "frame_id")),
        robot=RobotPlanningProfile(
            urdf=Path(str(_required(robot, "urdf"))),
            tool_link=str(_required(robot, "tool_link")),
            home_positions=np.asarray(_required(robot, "home_positions"), dtype=float),
            link_from_tool=np.asarray(_required(robot, "link_from_tool"), dtype=float),
            gripper_opening=float(_required(robot, "gripper_opening")),
        ),
        gripper=GripperPlanningProfile(
            urdf=Path(str(_required(gripper, "urdf"))),
            root_link=str(_required(gripper, "root_link")),
            grasp_from_root=np.asarray(
                _required(gripper, "grasp_from_root"), dtype=float
            ),
            opening_offset=np.asarray(
                _required(gripper, "opening_offset"), dtype=float
            ),
            opening_direction=np.asarray(
                _required(gripper, "opening_direction"), dtype=float
            ),
            opening_lower=float(limits[0]),
            opening_upper=float(limits[1]),
        ),
        obstacle_friction=float(_required(scene, "obstacle_friction")),
        handoff_position=np.asarray(_required(scene, "handoff_position"), dtype=float),
        obstacles=tuple(
            _obstacle(_mapping_value(value, "planning obstacle"))
            for value in raw_obstacles
        ),
    )


def planning_profile_with_pick_height(
    profile: PlanningProfile,
    height: float,
) -> PlanningProfile:
    """Return a runtime profile with the finite pickup table at ``height``."""

    value = float(height)
    if not np.isfinite(value):
        raise ValueError("pickup table height must be finite")
    found = False
    obstacles = []
    for obstacle in profile.obstacles:
        if obstacle.obstacle_id != "pick_table":
            obstacles.append(obstacle)
            continue
        maximum = obstacle.maximum.copy()
        maximum[2] = value
        obstacles.append(replace(obstacle, maximum=maximum))
        found = True
    if not found:
        raise ValueError("planning profile has no pick_table obstacle")
    return replace(profile, obstacles=tuple(obstacles))


def _obstacle(values: Mapping[str, object]) -> PlanningObstacle:
    return PlanningObstacle(
        obstacle_id=str(_required(values, "id")),
        minimum=np.asarray(_required(values, "minimum"), dtype=float),
        maximum=np.asarray(_required(values, "maximum"), dtype=float),
    )


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=float).copy()
    result.setflags(write=False)
    return result


def _pose(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite scalar-last pose")
    if np.linalg.norm(result[3:]) <= 1e-12:
        raise ValueError(f"{name} quaternion must be nonzero")
    return result


def _mapping(values: Mapping[str, object], key: str) -> Mapping[str, object]:
    return _mapping_value(_required(values, key), key)


def _mapping_value(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"planning profile '{name}' must be a mapping")
    return value


def _sequence(value: object, name: str, length: int) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f"planning profile '{name}' must be a sequence")
    if len(value) != length:
        raise ValueError(f"planning profile '{name}' must contain {length} values")
    return value


def _required(values: Mapping[str, object], key: str):
    if key not in values:
        raise ValueError(f"planning profile is missing '{key}'")
    return values[key]
