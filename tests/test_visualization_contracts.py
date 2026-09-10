import json
from pathlib import Path

import numpy as np
import pytest

from stacking_common.artifacts import (
    MCTSDebugArtifactStore,
    StackRunStore,
    decode_legacy_mcts_debug,
    decode_mcts_debug,
    encode_mcts_debug,
)
from stacking_common.visualization import (
    CandidatePoseMode,
    CandidateRenderMode,
    PoseTrack,
    SceneTimeline,
    ScoreField,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "legacy_visualization"
    / "mcts_candidate_viewer.json"
)


def _legacy_values() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_scene_timeline_owns_arrays_and_holds_shorter_tracks() -> None:
    source = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    timeline = SceneTimeline(
        (
            PoseTrack(10, source[:1]),
            PoseTrack(20, source),
        )
    )
    source[0, 0] = 99.0

    assert timeline.frame_count == 2
    assert np.isclose(timeline.track(10).pose_at(1)[0], 0.0)
    assert np.isclose(timeline.track(20).pose_at(1)[0], 0.1)
    assert [value.object_id for value in timeline.poses_at(1)] == [10, 20]
    with pytest.raises(ValueError):
        timeline.track(10).poses[0, 0] = 1.0
    with pytest.raises(IndexError):
        timeline.poses_at(2)


def test_score_field_requires_state_scope_and_axis_aligned_shapes() -> None:
    with pytest.raises(ValueError, match="scoped to one search state"):
        ScoreField("map", [0.0], [0.0], [[1.0]], [[True]], [], "candidate")
    with pytest.raises(ValueError, match="match their axes"):
        ScoreField("map", [0.0, 1.0], [0.0], [[1.0]], [[True]], [])


def test_timed_replay_interpolates_and_resamples_in_seconds():
    poses = np.array([[0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0]], dtype=float)
    track = PoseTrack(10, poses, [0, 1])
    halfway = track.pose_at_time(0.5)
    np.testing.assert_allclose(halfway, [0.5, 0, 0, 0, 0, 2**-0.5, 2**-0.5])
    np.testing.assert_allclose(track.pose_at_time(2), poses[-1])
    flipped = poses.copy()
    flipped[1, 3:] *= -1
    np.testing.assert_allclose(PoseTrack(10, flipped, [0, 1]).pose_at_time(0.25)[:3], [0.25, 0, 0])
    timeline = SceneTimeline((track, PoseTrack(20, poses[:1], [0])))
    for fps in (10, 30, 60):
        times = timeline.video_times(fps)
        assert len(times) == fps + 1  # Includes final endpoint frame.
        assert times[0] == 0 and times[-1] == 1
        np.testing.assert_allclose(timeline.poses_at_time(0.5)[1].pose, poses[0])
    assert SceneTimeline((PoseTrack(10, poses),)).video_times(30) is None
    for times in ([0], [0, 0], [1, 0], [0, np.nan], [-1, 0]):
        with pytest.raises(ValueError):
            PoseTrack(10, poses, times)
    with pytest.raises(ValueError):
        timeline.video_times(0)


def test_timed_trajectory_survives_projection_and_stack_roundtrip(tmp_path):
    from dataclasses import replace
    from stacking_common.domain import StackObjectTrajectory, State
    from stacking_common.visualization.mcts import _timeline

    pose = np.array([0, 0, 0, 0, 0, 0, 1.0])
    trajectory = StackObjectTrajectory(22)
    trajectory.add_pose(pose, time=0)
    trajectory.add_pose(pose, time=0.1)
    trajectory.add_pose(pose, time=0.3)
    copied = trajectory.copy()
    trajectory.times[-1] = 0.4
    assert copied.times == [0, 0.1, 0.3]
    state = State([22], [0], {22: pose}, [{22: copied}], [])
    timeline = _timeline(state)
    artifact = decode_legacy_mcts_debug(_legacy_values())
    step = artifact.steps[0]
    candidate = replace(step.candidates[0], timeline=timeline)
    artifact = replace(artifact, steps=(replace(step, candidates=(candidate,)),))
    decoded = decode_mcts_debug(encode_mcts_debug(artifact))
    np.testing.assert_allclose(decoded.steps[0].candidates[0].timeline.tracks[0].times, [0, 0.1, 0.3])
    path = tmp_path / "timed.stack"
    StackRunStore.save_mcts(artifact, path)
    loaded = StackRunStore.load_mcts(path)
    np.testing.assert_allclose(loaded.steps[0].candidates[0].timeline.tracks[0].times, [0, 0.1, 0.3])


def test_legacy_fixture_projects_to_explicit_common_pose_and_render_modes() -> None:
    values = _legacy_values()
    artifact = decode_legacy_mcts_debug(values)
    step = artifact.steps[0]
    candidate = step.render_candidates(CandidateRenderMode.SELECTED)[0]

    mode_names = {
        CandidateRenderMode.ALL: "all",
        CandidateRenderMode.ALL_OPAQUE: "all_opaque",
        CandidateRenderMode.SELECTED: "selected",
        CandidateRenderMode.STATE_ONLY: "no_select",
    }
    expected = values["steps"][0]["expected_render_modes"]
    for mode, legacy_name in mode_names.items():
        actual_ids = [
            value.candidate_id for value in step.render_candidates(mode)
        ]
        assert actual_ids == expected[legacy_name]["candidate_draw_order"]

    assert not np.allclose(
        candidate.pose(CandidatePoseMode.ACTION),
        candidate.pose(CandidatePoseMode.SIMULATION_START),
    )
    assert np.allclose(
        candidate.pose(CandidatePoseMode.FINAL),
        candidate.timeline.track(candidate.object_id).poses[-1],
    )
    assert candidate.score_field_id == step.score_field.field_id
    assert len(candidate.simulation_contacts) == 1
    assert len(candidate.pose_solve_forces) == 1
    assert artifact.steps[1].candidates == ()
    assert artifact.steps[1].failure_reason == "no_candidates"


def test_mcts_debug_artifact_json_round_trip_is_lossless(tmp_path: Path) -> None:
    artifact = decode_legacy_mcts_debug(_legacy_values())
    encoded = encode_mcts_debug(artifact)
    decoded = decode_mcts_debug(encoded)
    path = MCTSDebugArtifactStore().save(artifact, tmp_path / "debug.json")
    loaded = MCTSDebugArtifactStore().load(path)

    assert decoded.frame_id == artifact.frame_id
    assert loaded.schema_version == 1
    assert loaded.target.width == artifact.target.width
    assert loaded.steps[0].selected_candidate_id == "step-1-candidate-1"
    assert np.allclose(
        loaded.steps[0].score_field.values,
        artifact.steps[0].score_field.values,
    )
    assert loaded.steps[0].candidates[1].scene_motion.object_id == 11
    assert json.loads(path.read_text(encoding="utf-8")) == encoded


def test_stack_run_is_single_file_with_incremental_lossless_steps(
    tmp_path: Path,
) -> None:
    artifact = decode_legacy_mcts_debug(_legacy_values())
    header = type(artifact)(
        frame_id=artifact.frame_id,
        target=artifact.target,
        camera=artifact.camera,
        steps=(),
    )
    path = tmp_path / "planning.stack"
    writer = StackRunStore.create_mcts(
        path,
        header,
        metadata={"platform": "tabletop", "seed": 7},
    )

    writer.append_mcts_step(
        artifact.steps[0],
        statistics={"step": 1, "iterations": 12, "timed_out": False},
    )
    partial = StackRunStore.load_mcts(path)
    assert [step.step for step in partial.steps] == [1]

    writer.append_mcts_step(artifact.steps[1])
    video = b"\x00\x00\x00\x18ftypmp42-test-video"
    writer.add_attachment(
        "exports/search-process.mp4",
        video,
        role="derived_video",
        media_type="video/mp4",
        metadata={"fps": 30},
    )
    writer.finish()
    loaded = StackRunStore.load_mcts(path)
    info = StackRunStore.info(path)

    assert path.read_bytes()[:16] == b"SQLite format 3\x00"
    assert sorted(value.name for value in tmp_path.iterdir()) == ["planning.stack"]
    assert info.status == "complete"
    assert info.content_types == ("mcts_debug",)
    assert info.metadata["platform"] == "tabletop"
    assert StackRunStore.load_mcts_statistics(path) == (
        {"iterations": 12, "step": 1, "timed_out": False},
    )
    assert StackRunStore.attachments(path)[0].media_type == "video/mp4"
    assert StackRunStore.attachments(path)[0].metadata == {"fps": 30}
    assert StackRunStore.read_attachment(
        path, "exports/search-process.mp4"
    ) == video
    assert [step.step for step in loaded.steps] == [1, 2]
    assert np.allclose(
        loaded.steps[0].candidates[0].timeline.track(11).poses,
        artifact.steps[0].candidates[0].timeline.track(11).poses,
    )


def test_stack_run_requires_public_extension(tmp_path: Path) -> None:
    artifact = decode_legacy_mcts_debug(_legacy_values())

    with pytest.raises(ValueError, match=r"must end with \.stack"):
        StackRunStore.save_mcts(artifact, tmp_path / "planning.sqlite")
