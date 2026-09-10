"""Platform-neutral contracts for planning one selected placement."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Protocol

import numpy as np

from stacking_common.domain import Action, State
from stacking_common.domain._validation import pose_array

from .frames import FrameTransform
from .scene import SceneMobility, SceneObjectInstance


@dataclass(frozen=True)
class PlacementPlanningProblem:
    """World-frame motion-planning input derived from one search action."""

    frame_id: str
    action: Action
    pick_pose: np.ndarray
    place_pose: np.ndarray
    scene_objects: tuple[SceneObjectInstance, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)
    pick_scene_objects: tuple[SceneObjectInstance, ...] | None = None
    place_scene_objects: tuple[SceneObjectInstance, ...] | None = None

    def __post_init__(self) -> None:
        frame_id = str(self.frame_id).strip()
        if not frame_id:
            raise ValueError("planning frame ID must not be empty")
        if not isinstance(self.action, Action):
            raise TypeError("planning action must be an Action")
        objects = _scene_objects(self.scene_objects, "planning")
        pick_objects = _scene_objects(
            objects if self.pick_scene_objects is None else self.pick_scene_objects,
            "pick-scene",
        )
        place_objects = _scene_objects(
            objects if self.place_scene_objects is None else self.place_scene_objects,
            "place-scene",
        )
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "action", self.action.copy())
        object.__setattr__(self, "pick_pose", pose_array(self.pick_pose, "pick pose"))
        object.__setattr__(
            self,
            "place_pose",
            pose_array(self.place_pose, "place pose"),
        )
        object.__setattr__(self, "scene_objects", objects)
        object.__setattr__(self, "pick_scene_objects", pick_objects)
        object.__setattr__(self, "place_scene_objects", place_objects)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(copy.deepcopy(dict(self.metadata))),
        )

    @classmethod
    def from_search(
        cls,
        state: State,
        action: Action,
        *,
        pick_pose: np.ndarray,
        world_T_stack: FrameTransform,
        pick_scene_poses: Mapping[int, np.ndarray] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> "PlacementPlanningProblem":
        """Convert a stack-local search state/action into a world-frame problem."""

        if action.object_index >= len(state.object_ids):
            raise ValueError(
                "planning action object index is outside the state inventory"
            )
        expected_id = state.object_ids[action.object_index]
        if action.object_id not in {-1, expected_id}:
            raise ValueError("planning action object ID does not match its index")
        target_id = expected_id if action.object_id == -1 else action.object_id
        place_objects: list[SceneObjectInstance] = []
        for index in state.placed_object_indices:
            object_id = state.object_ids[index]
            try:
                local_pose = state.object_poses[object_id]
            except KeyError as error:
                raise ValueError(
                    f"placed object {object_id} has no search-state pose"
                ) from error
            place_objects.append(
                SceneObjectInstance(
                    index,
                    object_id,
                    world_T_stack.child_pose_to_parent(local_pose),
                    SceneMobility.STATIC,
                )
            )
        if action.object_index in state.placed_object_indices:
            raise ValueError("planning target is already placed in the source state")
        world_pick_pose = pose_array(pick_pose, "pick pose")
        target = SceneObjectInstance(
            action.object_index,
            target_id,
            world_pick_pose,
            SceneMobility.KINEMATIC,
        )
        place_objects.append(target)
        if pick_scene_poses is None:
            pick_objects = tuple(place_objects)
        else:
            normalized_pick_poses = {
                int(object_id): pose_array(
                    pose,
                    f"pick-scene pose for object {object_id}",
                )
                for object_id, pose in pick_scene_poses.items()
            }
            unplaced_indices = [
                index
                for index in range(len(state.object_ids))
                if index not in state.placed_object_indices
            ]
            expected_ids = {state.object_ids[index] for index in unplaced_indices}
            if set(normalized_pick_poses) != expected_ids:
                missing = sorted(expected_ids - set(normalized_pick_poses))
                extra = sorted(set(normalized_pick_poses) - expected_ids)
                raise ValueError(
                    "pick-scene poses must match the unplaced objects: "
                    f"missing={missing}, extra={extra}"
                )
            pick_objects = tuple(
                SceneObjectInstance(
                    index,
                    state.object_ids[index],
                    (
                        world_pick_pose
                        if index == action.object_index
                        else normalized_pick_poses[state.object_ids[index]]
                    ),
                    (
                        SceneMobility.KINEMATIC
                        if index == action.object_index
                        else SceneMobility.STATIC
                    ),
                )
                for index in unplaced_indices
            )
        return cls(
            frame_id=world_T_stack.parent_frame,
            action=action,
            pick_pose=world_pick_pose,
            # ``pose`` is the authoritative transition result. ``solved_pose``
            # is the pre-simulation pose-generator result and may differ after
            # settling.
            place_pose=world_T_stack.child_pose_to_parent(action.pose),
            scene_objects=tuple(place_objects),
            metadata=metadata or {},
            pick_scene_objects=pick_objects,
            place_scene_objects=tuple(place_objects),
        )


def _scene_objects(values, name: str) -> tuple[SceneObjectInstance, ...]:
    objects = tuple(values)
    if not all(isinstance(value, SceneObjectInstance) for value in objects):
        raise TypeError(f"{name} objects must be SceneObjectInstance values")
    indices = [value.object_index for value in objects]
    object_ids = [value.object_id for value in objects]
    if len(indices) != len(set(indices)):
        raise ValueError(f"{name} object indices must be unique")
    if len(object_ids) != len(set(object_ids)):
        raise ValueError(f"{name} object IDs must be unique")
    return objects


@dataclass(frozen=True)
class MotionPlanSegment:
    """One joint-space path with optional timing and gripper opening."""

    name: str
    positions: np.ndarray
    times: np.ndarray | None = None
    openings: np.ndarray | None = None
    mode: str | None = None
    target_poses: tuple[np.ndarray | None, ...] | None = None
    scene_phase: str | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        positions = np.asarray(self.positions, dtype=float)
        times = (
            None if self.times is None else np.asarray(self.times, dtype=float)
        )
        openings = (
            None
            if self.openings is None
            else np.asarray(self.openings, dtype=float)
        )
        if not name:
            raise ValueError("motion segment name must not be empty")
        if self.mode not in {None, "free", "attached"}:
            raise ValueError("motion mode must be free or attached")
        if self.scene_phase not in {None, "pick", "handoff", "place", "transfer"}:
            raise ValueError("unknown motion scene phase")
        if positions.ndim != 2 or positions.shape[0] == 0:
            raise ValueError("motion segment positions must have shape (steps, joints)")
        if times is not None and times.shape != (positions.shape[0],):
            raise ValueError("motion segment times must align with positions")
        if openings is not None and openings.shape != (positions.shape[0],):
            raise ValueError("motion segment openings must align with positions")
        if not np.all(np.isfinite(positions)) or (
            times is not None and not np.all(np.isfinite(times))
        ) or (openings is not None and not np.all(np.isfinite(openings))):
            raise ValueError("motion segment values must be finite")
        if times is not None and (
            times[0] < 0.0 or np.any(np.diff(times) < 0.0)
        ):
            raise ValueError("motion segment times must be non-negative and ordered")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "positions", positions.copy())
        if self.target_poses is not None:
            if len(self.target_poses) != len(positions):
                raise ValueError("target poses must align with motion samples")
            poses = tuple(None if pose is None else pose_array(pose, "target pose")
                          for pose in self.target_poses)
            for pose in poses:
                if pose is not None:
                    pose.setflags(write=False)
            object.__setattr__(self, "target_poses", poses)
        object.__setattr__(
            self,
            "times",
            None if times is None else times.copy(),
        )
        object.__setattr__(
            self,
            "openings",
            None if openings is None else openings.copy(),
        )


@dataclass(frozen=True)
class GraspEvent:
    """Acquire/release at a segment sample, in the planning frame.

    Opening is the gripper model's scalar coordinate (possibly signed), not a
    hardware command. frame_from_grasp identifies the planner's grasp frame.
    """

    event: str
    segment_index: int
    sample_index: int
    frame_from_grasp: np.ndarray
    opening: float

    def __post_init__(self):
        if self.event not in {"acquire", "release"}:
            raise ValueError("grasp event must acquire or release")
        for value in (self.segment_index, self.sample_index):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("grasp event indices must be non-negative integers")
        pose = pose_array(self.frame_from_grasp, "grasp event pose")
        pose.setflags(write=False)
        if not math.isfinite(self.opening):
            raise ValueError("grasp opening must be finite")
        object.__setattr__(self, "frame_from_grasp", pose)
        object.__setattr__(self, "opening", float(self.opening))


def validate_grasp_events(events, segments):
    events = tuple(events)
    previous = (-1, -1)
    for event in events:
        if not isinstance(event, GraspEvent):
            raise TypeError("grasp events must be GraspEvent values")
        location = (event.segment_index, event.sample_index)
        if location <= previous or event.segment_index >= len(segments):
            raise ValueError("grasp events must be ordered within segments")
        if event.sample_index >= len(segments[event.segment_index].positions):
            raise ValueError("grasp event sample is outside its segment")
        previous = location
    return events


class PlanningStatus(str, Enum):
    SUCCESS = "success"
    INFEASIBLE = "infeasible"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class PlacementPlanningResult:
    """Backend-independent result returned to orchestration and the viewer."""

    status: PlanningStatus
    segments: tuple[MotionPlanSegment, ...] = ()
    reason: str | None = None
    score: float | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    grasp_events: tuple[GraspEvent, ...] = ()

    def __post_init__(self) -> None:
        status = PlanningStatus(self.status)
        segments = tuple(self.segments)
        if not all(isinstance(value, MotionPlanSegment) for value in segments):
            raise TypeError("planning segments must be MotionPlanSegment values")
        reason = None if self.reason is None else str(self.reason).strip()
        if status is PlanningStatus.SUCCESS:
            if not segments:
                raise ValueError(
                    "a successful planning result requires motion segments"
                )
            if reason:
                raise ValueError(
                    "a successful planning result must not have a failure reason"
                )
        elif not reason:
            raise ValueError("an unsuccessful planning result requires a reason")
        score = None if self.score is None else float(self.score)
        if score is not None and not math.isfinite(score):
            raise ValueError("planning result score must be finite")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "grasp_events", validate_grasp_events(self.grasp_events, segments))
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "score", score)
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(copy.deepcopy(dict(self.diagnostics))),
        )

    @property
    def succeeded(self) -> bool:
        return self.status is PlanningStatus.SUCCESS


class PlacementPlanner(Protocol):
    """Motion-planning backend consumed by platform orchestration."""

    def plan(self, problem: PlacementPlanningProblem) -> PlacementPlanningResult:
        """Plan one selected pick-and-place action."""
