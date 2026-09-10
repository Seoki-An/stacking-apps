"""Serialization of placement-planning attempts stored in stacking runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from stacking_common.execution.values import sha256_text

from stacking_common.planning import (
    GraspEvent,
    MotionPlanSegment,
    PlacementPlanningProblem,
    PlacementPlanningResult,
    PlanningStatus,
    SceneMobility,
    SceneObjectInstance,
)

from .action import decode_action, encode_action


@dataclass(frozen=True)
class PlacementPlanRecord:
    """One motion-planning attempt associated with an MCTS step."""

    step: int
    attempt: int
    problem: PlacementPlanningProblem
    result: PlacementPlanningResult
    candidate_id: str | None = None
    context_sha256: str | None = None

    def __post_init__(self) -> None:
        step = int(self.step)
        attempt = int(self.attempt)
        candidate_id = (
            None
            if self.candidate_id is None
            else str(self.candidate_id).strip()
        )
        if step <= 0 or attempt <= 0:
            raise ValueError("placement plan step and attempt must be positive")
        if not isinstance(self.problem, PlacementPlanningProblem):
            raise TypeError("placement plan record requires a planning problem")
        if not isinstance(self.result, PlacementPlanningResult):
            raise TypeError("placement plan record requires a planning result")
        if self.problem.action.object_id < 0:
            raise ValueError("stored planning action must have a resolved object ID")
        if candidate_id == "":
            raise ValueError("placement plan candidate ID must not be empty")
        if self.context_sha256 is not None:
            sha256_text(self.context_sha256, "planning context hash")
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "attempt", attempt)
        object.__setattr__(self, "candidate_id", candidate_id)


def encode_placement_plan(record: PlacementPlanRecord) -> dict[str, object]:
    return {
        "schema_version": 1,
        "step": record.step,
        "attempt": record.attempt,
        "candidate_id": record.candidate_id,
        "context_sha256": record.context_sha256,
        "problem": _encode_problem(record.problem),
        "result": _encode_result(record.result),
    }


def decode_placement_plan(values: Mapping[str, object]) -> PlacementPlanRecord:
    if int(values.get("schema_version", 0)) != 1:
        raise ValueError("unsupported placement plan schema version")
    return PlacementPlanRecord(
        step=int(values["step"]),
        attempt=int(values["attempt"]),
        candidate_id=values.get("candidate_id"),
        context_sha256=values.get("context_sha256"),
        problem=_decode_problem(_mapping(values["problem"], "planning problem")),
        result=_decode_result(_mapping(values["result"], "planning result")),
    )


def _encode_problem(value: PlacementPlanningProblem) -> dict[str, object]:
    return {
        "frame_id": value.frame_id,
        "action": _plain(encode_action(value.action)),
        "pick_pose": value.pick_pose.tolist(),
        "place_pose": value.place_pose.tolist(),
        "scene_objects": _encode_scene_objects(value.scene_objects),
        "pick_scene_objects": _encode_scene_objects(value.pick_scene_objects),
        "place_scene_objects": _encode_scene_objects(value.place_scene_objects),
        "metadata": _plain(dict(value.metadata)),
    }


def _decode_problem(values: Mapping[str, object]) -> PlacementPlanningProblem:
    objects = values.get("scene_objects")
    if not isinstance(objects, (list, tuple)):
        raise TypeError("planning scene objects must be a sequence")
    pick_objects = values.get("pick_scene_objects")
    place_objects = values.get("place_scene_objects")
    return PlacementPlanningProblem(
        frame_id=str(values["frame_id"]),
        action=decode_action(_mapping(values["action"], "planning action")),
        pick_pose=np.asarray(values["pick_pose"], dtype=float),
        place_pose=np.asarray(values["place_pose"], dtype=float),
        scene_objects=_decode_scene_objects(objects),
        metadata=_mapping(values.get("metadata", {}), "planning metadata"),
        pick_scene_objects=(
            None
            if pick_objects is None
            else _decode_scene_objects(pick_objects)
        ),
        place_scene_objects=(
            None
            if place_objects is None
            else _decode_scene_objects(place_objects)
        ),
    )


def _encode_scene_objects(values) -> list[dict[str, object]]:
    return [
        {
            "object_index": item.object_index,
            "object_id": item.object_id,
            "pose": item.pose.tolist(),
            "mobility": item.mobility.value,
        }
        for item in values
    ]


def _decode_scene_objects(values) -> tuple[SceneObjectInstance, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError("planning scene objects must be a sequence")
    return tuple(
        SceneObjectInstance(
            int(_mapping(item, "planning scene object")["object_index"]),
            int(_mapping(item, "planning scene object")["object_id"]),
            np.asarray(_mapping(item, "planning scene object")["pose"]),
            SceneMobility(_mapping(item, "planning scene object")["mobility"]),
        )
        for item in values
    )


def _encode_result(value: PlacementPlanningResult) -> dict[str, object]:
    result = {
        "status": value.status.value,
        "reason": value.reason,
        "score": value.score,
        "segments": [
            {
                "name": segment.name,
                **({"mode": segment.mode} if segment.mode is not None else {}),
                **({"scene_phase": segment.scene_phase} if segment.scene_phase is not None else {}),
                **({"target_poses": [None if pose is None else pose.tolist()
                                     for pose in segment.target_poses]}
                   if segment.target_poses is not None else {}),
                "positions": segment.positions.tolist(),
                "times": (
                    None
                    if segment.times is None
                    else segment.times.tolist()
                ),
                "openings": (
                    None
                    if segment.openings is None
                    else segment.openings.tolist()
                ),
            }
            for segment in value.segments
        ],
        "diagnostics": _plain(dict(value.diagnostics)),
    }
    # Optional v1 extension: absent semantics retain historical plan digests.
    if value.grasp_events:
        result["grasp_events"] = [encode_grasp_event(event) for event in value.grasp_events]
    return result


def encode_grasp_event(event):
    return {"event": event.event, "segment_index": event.segment_index,
            "sample_index": event.sample_index,
            "frame_from_grasp": event.frame_from_grasp.tolist(), "opening": event.opening}


def decode_grasp_event(value):
    return GraspEvent(**dict(value))


def _decode_result(values: Mapping[str, object]) -> PlacementPlanningResult:
    segments = values.get("segments")
    if not isinstance(segments, (list, tuple)):
        raise TypeError("planning result segments must be a sequence")
    return PlacementPlanningResult(
        status=PlanningStatus(values["status"]),
        segments=tuple(
            MotionPlanSegment(
                name=str(_mapping(item, "motion segment")["name"]),
                mode=item.get("mode"),
                scene_phase=item.get("scene_phase"),
                target_poses=item.get("target_poses"),
                positions=np.asarray(
                    _mapping(item, "motion segment")["positions"],
                    dtype=float,
                ),
                times=(
                    None
                    if _mapping(item, "motion segment").get("times") is None
                    else np.asarray(
                        _mapping(item, "motion segment")["times"],
                        dtype=float,
                    )
                ),
                openings=(
                    None
                    if _mapping(item, "motion segment").get("openings") is None
                    else np.asarray(
                        _mapping(item, "motion segment")["openings"],
                        dtype=float,
                    )
                ),
            )
            for item in segments
        ),
        reason=values.get("reason"),
        grasp_events=tuple(decode_grasp_event(event) for event in values.get("grasp_events", ())),
        score=values.get("score"),
        diagnostics=_mapping(
            values.get("diagnostics", {}),
            "planning diagnostics",
        ),
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _plain(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value
