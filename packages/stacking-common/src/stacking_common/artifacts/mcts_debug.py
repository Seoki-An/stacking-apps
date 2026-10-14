"""JSON boundary for backend-neutral MCTS visualization artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from stacking_common.domain import DirectionalTaper
from stacking_common.visualization.contracts import (
    CameraView,
    ContactMarker,
    ForceMarker,
    PoseTrack,
    SceneObjectPose,
    SceneSnapshot,
    SceneTimeline,
    ScoreField,
    TargetVolume,
)
from stacking_common.visualization.mcts import (
    CandidateOutcome,
    MCTSCandidateDiagnostic,
    MCTSDebugArtifact,
    MCTSDebugStep,
    SceneMotionDiagnostic,
)


class MCTSDebugArtifactStore:
    """Atomically persist portable MCTS viewer inputs as JSON."""

    def save(self, artifact: MCTSDebugArtifact, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(encode_mcts_debug(artifact), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    def load(self, path: str | Path) -> MCTSDebugArtifact:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return decode_mcts_debug(_mapping(values, "MCTS debug artifact"))


def encode_mcts_debug(artifact: MCTSDebugArtifact) -> dict[str, object]:
    """Return a JSON-compatible, independently owned artifact mapping."""

    return {
        "schema_version": artifact.schema_version,
        "frame_id": artifact.frame_id,
        "target": None if artifact.target is None else _encode_target(artifact.target),
        "camera": None if artifact.camera is None else _encode_camera(artifact.camera),
        "steps": [_encode_step(value) for value in artifact.steps],
    }


def decode_mcts_debug(values: Mapping[str, object]) -> MCTSDebugArtifact:
    """Decode the versioned common artifact schema."""

    version = int(values.get("schema_version", 0))
    if version != 1:
        raise ValueError("unsupported MCTS debug artifact schema version")
    frame_id = str(values["frame_id"])
    raw_target = values.get("target")
    raw_camera = values.get("camera")
    return MCTSDebugArtifact(
        schema_version=version,
        frame_id=frame_id,
        target=(
            None
            if raw_target is None
            else _decode_target(_mapping(raw_target, "target"))
        ),
        camera=(
            None
            if raw_camera is None
            else _decode_camera(_mapping(raw_camera, "camera"))
        ),
        steps=tuple(
            _decode_step(_mapping(value, "MCTS debug step"), frame_id)
            for value in _sequence(values.get("steps", []), "steps")
        ),
    )


def decode_legacy_mcts_debug(values: Mapping[str, object]) -> MCTSDebugArtifact:
    """Adapt the step-1 legacy characterization fixture to common values."""

    if int(values.get("fixture_version", 0)) != 1:
        raise ValueError("unsupported legacy visualization fixture version")
    frame_id = str(values["frame_id"])
    target = _decode_target(_mapping(values["target"], "legacy target"))
    camera_behavior = _mapping(values["camera_behavior"], "camera behavior")
    camera = _decode_camera(_mapping(camera_behavior["camera"], "legacy camera"))
    steps = tuple(
        _decode_legacy_step(_mapping(value, "legacy step"), frame_id)
        for value in _sequence(values.get("steps", []), "legacy steps")
    )
    return MCTSDebugArtifact(
        frame_id=frame_id,
        target=target,
        camera=camera,
        steps=steps,
    )


def encode_mcts_step(value: MCTSDebugStep) -> dict[str, object]:
    """Encode one step for incremental container persistence."""

    if not isinstance(value, MCTSDebugStep):
        raise TypeError("MCTS debug step must be an MCTSDebugStep")
    return _encode_step(value)


def decode_mcts_step(
    values: Mapping[str, object], *, frame_id: str
) -> MCTSDebugStep:
    """Decode one independently stored MCTS step."""

    return _decode_step(values, str(frame_id))


def _encode_target(value: TargetVolume) -> dict[str, object]:
    return {
        "origin": value.origin.tolist(),
        "width": value.width,
        "length": value.length,
        "height": value.height,
        "structure_offset": value.structure_offset.tolist(),
        "taper": asdict(value.taper),
    }


def _decode_target(values: Mapping[str, object]) -> TargetVolume:
    return TargetVolume(
        origin=values["origin"],
        width=values["width"],
        length=values["length"],
        height=values["height"],
        structure_offset=values.get("structure_offset", [0.0, 0.0, 0.0]),
        taper=DirectionalTaper(**_mapping(values.get("taper", {}), "target taper")),
    )


def _encode_camera(value: CameraView) -> dict[str, object]:
    return {
        "eye": value.eye.tolist(),
        "look_at": value.look_at.tolist(),
        "up": value.up.tolist(),
        "vertical_fov_degrees": value.vertical_fov_degrees,
    }


def _decode_camera(values: Mapping[str, object]) -> CameraView:
    return CameraView(
        eye=values["eye"],
        look_at=values["look_at"],
        up=values["up"],
        vertical_fov_degrees=values.get("vertical_fov_degrees", 45.0),
    )


def _encode_step(value: MCTSDebugStep) -> dict[str, object]:
    return {
        "step": value.step,
        "succeeded": value.succeeded,
        "state_only": value.state_only,
        "failure_reason": value.failure_reason,
        "scene": {
            "objects": [_encode_object(item) for item in value.scene.objects],
        },
        "score_field": (
            None
            if value.score_field is None
            else _encode_score_field(value.score_field)
        ),
        "selected_candidate_id": value.selected_candidate_id,
        "rejected_expansions": dict(value.rejected_expansions),
        "candidates": [_encode_candidate(item) for item in value.candidates],
    }


def _decode_step(values: Mapping[str, object], frame_id: str) -> MCTSDebugStep:
    raw_scene = _mapping(values["scene"], "scene")
    raw_score = values.get("score_field")
    return MCTSDebugStep(
        step=values["step"],
        succeeded=values["succeeded"],
        state_only=values.get("state_only", False),
        failure_reason=values.get("failure_reason"),
        scene=SceneSnapshot(
            frame_id,
            tuple(
                _decode_object(_mapping(value, "scene object"))
                for value in _sequence(raw_scene.get("objects", []), "scene objects")
            ),
        ),
        score_field=(
            None
            if raw_score is None
            else _decode_score_field(_mapping(raw_score, "score field"))
        ),
        selected_candidate_id=values.get("selected_candidate_id"),
        rejected_expansions=dict(
            _mapping(values.get("rejected_expansions", {}), "rejected expansions")
        ),
        candidates=tuple(
            _decode_candidate(_mapping(value, "candidate"))
            for value in _sequence(values.get("candidates", []), "candidates")
        ),
    )


def _encode_candidate(value: MCTSCandidateDiagnostic) -> dict[str, object]:
    return {
        **({"construction_diagnostics": dict(value.construction_diagnostics)}
           if value.construction_diagnostics else {}),
        "candidate_id": value.candidate_id,
        "rank": value.rank,
        "object_index": value.object_index,
        "object_id": value.object_id,
        "outcome": value.outcome.value,
        "selected": value.selected,
        "rejection_reason": value.rejection_reason,
        "score": value.score,
        "prior": value.prior,
        "visits": value.visits,
        "mean_value": value.mean_value,
        "reward": value.reward,
        "score_field_id": value.score_field_id,
        "init_pose": value.init_pose.tolist(),
        "solved_pose": value.solved_pose.tolist(),
        "action_pose": value.action_pose.tolist(),
        "final_pose": None if value.final_pose is None else value.final_pose.tolist(),
        "timeline": _encode_timeline(value.timeline),
        "simulation_contacts": [
            _encode_contact(item) for item in value.simulation_contacts
        ],
        "pose_solve_forces": [
            _encode_force(item) for item in value.pose_solve_forces
        ],
        "best_sequence": [_encode_object(item) for item in value.best_sequence],
        "scene_motion": (
            None
            if value.scene_motion is None
            else _encode_scene_motion(value.scene_motion)
        ),
    }


def _decode_candidate(values: Mapping[str, object]) -> MCTSCandidateDiagnostic:
    raw_final = values.get("final_pose")
    raw_motion = values.get("scene_motion")
    return MCTSCandidateDiagnostic(
        construction_diagnostics=values.get("construction_diagnostics", {}),
        candidate_id=values["candidate_id"],
        rank=values["rank"],
        object_index=values["object_index"],
        object_id=values["object_id"],
        outcome=CandidateOutcome(values["outcome"]),
        selected=values["selected"],
        rejection_reason=values.get("rejection_reason"),
        score=values.get("score"),
        prior=values.get("prior", 0.0),
        visits=values.get("visits", 0),
        mean_value=values.get("mean_value", 0.0),
        reward=values.get("reward"),
        score_field_id=values.get("score_field_id"),
        init_pose=values["init_pose"],
        solved_pose=values["solved_pose"],
        action_pose=values["action_pose"],
        final_pose=None if raw_final is None else raw_final,
        timeline=_decode_timeline(
            _mapping(values.get("timeline", {}), "candidate timeline")
        ),
        simulation_contacts=tuple(
            _decode_contact(_mapping(value, "simulation contact"))
            for value in _sequence(
                values.get("simulation_contacts", []), "simulation contacts"
            )
        ),
        pose_solve_forces=tuple(
            _decode_force(_mapping(value, "pose-solve force"))
            for value in _sequence(
                values.get("pose_solve_forces", []), "pose-solve forces"
            )
        ),
        best_sequence=tuple(
            _decode_object(_mapping(value, "best-sequence pose"))
            for value in _sequence(values.get("best_sequence", []), "best sequence")
        ),
        scene_motion=(
            None
            if raw_motion is None
            else _decode_scene_motion(_mapping(raw_motion, "scene motion"))
        ),
    )


def _decode_legacy_step(
    values: Mapping[str, object], frame_id: str
) -> MCTSDebugStep:
    raw_scene = _mapping(values["scene"], "legacy scene")
    poses = _mapping(raw_scene.get("object_poses", {}), "legacy object poses")
    score_field = _decode_legacy_score_field(
        _mapping(values["score_map"], "legacy score map")
    )
    return MCTSDebugStep(
        step=values["step"],
        succeeded=values["succeeded"],
        state_only=values.get("state_only", False),
        failure_reason=values.get("failure_reason"),
        scene=SceneSnapshot(
            frame_id,
            tuple(
                SceneObjectPose(object_id, poses[str(object_id)])
                for object_id in _sequence(
                    raw_scene.get("object_ids", []), "legacy scene object IDs"
                )
            ),
        ),
        score_field=score_field,
        selected_candidate_id=values.get("selected_candidate_id"),
        candidates=tuple(
            _decode_legacy_candidate(
                _mapping(value, "legacy candidate"), score_field.field_id
            )
            for value in _sequence(values.get("candidates", []), "legacy candidates")
        ),
    )


def _decode_legacy_candidate(
    values: Mapping[str, object], score_field_id: str
) -> MCTSCandidateDiagnostic:
    if bool(values.get("failed")):
        outcome = CandidateOutcome.FAILED
    elif bool(values.get("validated")):
        outcome = CandidateOutcome.SUCCEEDED
    else:
        outcome = CandidateOutcome.PENDING
    raw_final = values.get("final_pose")
    raw_motion = values.get("scene_motion")
    return MCTSCandidateDiagnostic(
        candidate_id=values["candidate_id"],
        rank=values["rank"],
        object_index=values["object_index"],
        object_id=values["object_id"],
        outcome=outcome,
        selected=values["selected"],
        rejection_reason=values.get("rejection_reason"),
        score=values.get("score"),
        prior=values.get("prior", 0.0),
        visits=values.get("visits", 0),
        mean_value=values.get("mean_value", 0.0),
        reward=values.get("reward"),
        score_field_id=score_field_id,
        init_pose=values["init_pose"],
        solved_pose=values["solved_pose"],
        action_pose=values["action_pose"],
        final_pose=None if raw_final is None else raw_final,
        timeline=SceneTimeline(
            tuple(
                PoseTrack(int(object_id), poses)
                for object_id, poses in _mapping(
                    values.get("simulation_tracks", {}),
                    "legacy simulation tracks",
                ).items()
            )
        ),
        simulation_contacts=tuple(
            ContactMarker(
                value.get("object_id_first"),
                value.get("object_id_second"),
                value["point"],
                value["normal"],
                value["gap"],
            )
            for raw_value in _sequence(
                values.get("simulation_contacts", []),
                "legacy simulation contacts",
            )
            for value in (_mapping(raw_value, "legacy simulation contact"),)
        ),
        pose_solve_forces=tuple(
            ForceMarker(
                value["object_id"],
                value["role"],
                value["point"],
                value["normal"],
                value["force"],
            )
            for raw_value in _sequence(
                values.get("pose_solve_contacts", []),
                "legacy pose-solve contacts",
            )
            for value in (_mapping(raw_value, "legacy pose-solve contact"),)
        ),
        best_sequence=tuple(
            SceneObjectPose(value["object_id"], value["pose"])
            for raw_value in _sequence(
                values.get("best_sequence", []), "legacy best sequence"
            )
            for value in (_mapping(raw_value, "legacy best-sequence pose"),)
        ),
        scene_motion=(
            None
            if raw_motion is None
            else _decode_legacy_scene_motion(
                _mapping(raw_motion, "legacy scene motion")
            )
        ),
    )


def _encode_object(value: SceneObjectPose) -> dict[str, object]:
    return {
        "object_id": value.object_id,
        "object_index": value.object_index,
        "pose": value.pose.tolist(),
    }


def _decode_object(values: Mapping[str, object]) -> SceneObjectPose:
    return SceneObjectPose(
        values["object_id"], values["pose"], values.get("object_index")
    )


def _encode_timeline(value: SceneTimeline) -> dict[str, object]:
    return {
        "tracks": [
            {"object_id": track.object_id, "poses": track.poses.tolist(),
             **({"times": track.times.tolist()} if track.times is not None else {})}
            for track in value.tracks
        ]
    }


def _decode_timeline(values: Mapping[str, object]) -> SceneTimeline:
    return SceneTimeline(
        tuple(
            PoseTrack(item["object_id"], item["poses"], item.get("times"))
            for raw_item in _sequence(values.get("tracks", []), "timeline tracks")
            for item in (_mapping(raw_item, "timeline track"),)
        )
    )


def _encode_contact(value: ContactMarker) -> dict[str, object]:
    return {
        "object_id_first": value.object_id_first,
        "object_id_second": value.object_id_second,
        "point": value.point.tolist(),
        "normal": value.normal.tolist(),
        "gap": value.gap,
    }


def _decode_contact(values: Mapping[str, object]) -> ContactMarker:
    return ContactMarker(
        values.get("object_id_first"),
        values.get("object_id_second"),
        values["point"],
        values["normal"],
        values["gap"],
    )


def _encode_force(value: ForceMarker) -> dict[str, object]:
    return {
        "object_id": value.object_id,
        "role": value.role,
        "point": value.point.tolist(),
        "normal": value.normal.tolist(),
        "force": value.force.tolist(),
    }


def _decode_force(values: Mapping[str, object]) -> ForceMarker:
    return ForceMarker(
        values["object_id"],
        values["role"],
        values["point"],
        values["normal"],
        values["force"],
    )


def _encode_score_field(value: ScoreField) -> dict[str, object]:
    return {
        "field_id": value.field_id,
        "scope": value.scope,
        "x_coordinates": value.x_coordinates.tolist(),
        "y_coordinates": value.y_coordinates.tolist(),
        "values": value.values.tolist(),
        "valid": value.valid.tolist(),
        "selected_xy": value.selected_xy.tolist(),
    }


def _decode_score_field(values: Mapping[str, object]) -> ScoreField:
    return ScoreField(
        field_id=values["field_id"],
        scope=values.get("scope", "state"),
        x_coordinates=values["x_coordinates"],
        y_coordinates=values["y_coordinates"],
        values=values["values"],
        valid=values["valid"],
        selected_xy=values.get("selected_xy", []),
    )


def _decode_legacy_score_field(values: Mapping[str, object]) -> ScoreField:
    return ScoreField(
        field_id=values["id"],
        scope=values.get("scope", "state"),
        x_coordinates=values["x_coords"],
        y_coordinates=values["y_coords"],
        values=values["scores"],
        valid=values["valid"],
        selected_xy=values.get("selected_xy", []),
    )


def _encode_scene_motion(value: SceneMotionDiagnostic) -> dict[str, object]:
    return {
        "object_id": value.object_id,
        "parent_pose": value.parent_pose.tolist(),
        "failed_final_pose": value.failed_final_pose.tolist(),
        "trajectory": value.trajectory.poses.tolist(),
        **({"times": value.trajectory.times.tolist()}
           if value.trajectory.times is not None else {}),
    }


def _decode_scene_motion(values: Mapping[str, object]) -> SceneMotionDiagnostic:
    object_id = int(values["object_id"])
    return SceneMotionDiagnostic(
        object_id,
        values["parent_pose"],
        values["failed_final_pose"],
        PoseTrack(object_id, values["trajectory"], values.get("times")),
    )


def _decode_legacy_scene_motion(
    values: Mapping[str, object]
) -> SceneMotionDiagnostic:
    return _decode_scene_motion(values)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _sequence(value: object, name: str) -> list | tuple:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a sequence")
    return value
