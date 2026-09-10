import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "legacy_visualization"
    / "mcts_candidate_viewer.json"
)


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _candidate(step: dict, candidate_id: str) -> dict:
    return next(
        candidate
        for candidate in step["candidates"]
        if candidate["candidate_id"] == candidate_id
    )


def _track_pose_at(candidate: dict, object_id: int, frame: int) -> np.ndarray:
    """Apply the legacy replay rule: shorter tracks hold their last pose."""

    poses = np.asarray(candidate["simulation_tracks"][str(object_id)], dtype=float)
    return poses[min(frame, len(poses) - 1)]


def test_fixture_covers_success_failure_pending_and_zero_candidate_steps() -> None:
    data = _fixture()
    first, second = data["steps"]

    assert data["fixture_version"] == 1
    assert data["units"] == "si"
    assert data["frame_id"] == "world"
    assert first["succeeded"]
    assert {candidate["validated"] for candidate in first["candidates"]} == {
        True,
        False,
    }
    assert any(candidate["failed"] for candidate in first["candidates"])
    assert any(
        not candidate["validated"] and not candidate["failed"]
        for candidate in first["candidates"]
    )
    assert second["failure_reason"] == "no_candidates"
    assert second["candidates"] == []
    assert second["score_map"]["scope"] == "state"


def test_selected_candidate_identity_is_explicit_and_not_inferred_from_rank() -> None:
    first, second = _fixture()["steps"]
    selected = [
        candidate
        for candidate in first["candidates"]
        if candidate["selected"]
    ]

    assert len(selected) == 1
    assert selected[0]["candidate_id"] == first["selected_candidate_id"]
    assert selected[0]["rank"] == 1
    assert second["selected_candidate_id"] is None


def test_legacy_candidate_render_modes_preserve_selection_semantics() -> None:
    first = _fixture()["steps"][0]
    modes = first["expected_render_modes"]

    assert modes["all"]["candidate_draw_order"][-1] == first[
        "selected_candidate_id"
    ]
    assert modes["all"]["selected_overlay"]
    assert not modes["all_opaque"]["selected_overlay"]
    assert modes["all_opaque"]["opaque"]
    assert modes["selected"]["candidate_draw_order"] == [
        first["selected_candidate_id"]
    ]
    assert modes["no_select"]["candidate_draw_order"] == []


def test_pose_modes_distinguish_designed_and_physical_poses() -> None:
    data = _fixture()
    first = data["steps"][0]
    selected = _candidate(first, first["selected_candidate_id"])
    target_track = np.asarray(selected["simulation_tracks"]["22"], dtype=float)

    assert data["pose_semantics"] == {
        "initial": "init_pose",
        "solved": "solved_pose",
        "legacy_settled": "action_pose",
        "simulation_start": "first target-object simulation track pose",
        "final": "final_pose",
        "note": (
            "The legacy settled_pose name is a compatibility alias for the "
            "designed action pose; it is not the simulation-start pose."
        ),
    }
    assert not np.allclose(selected["init_pose"], selected["solved_pose"])
    assert not np.allclose(selected["action_pose"], target_track[0])
    assert not np.allclose(target_track[0], selected["final_pose"])
    assert np.allclose(target_track[-1], selected["final_pose"])


def test_score_maps_are_step_scoped_and_independent_of_candidate_selection() -> None:
    for step in _fixture()["steps"]:
        score_map = step["score_map"]
        scores = np.asarray(score_map["scores"], dtype=float)
        valid = np.asarray(score_map["valid"], dtype=bool)

        assert score_map["scope"] == "state"
        assert scores.shape == valid.shape
        assert scores.shape == (
            len(score_map["y_coords"]),
            len(score_map["x_coords"]),
        )
        assert all(
            candidate["score_map_id"] == score_map["id"]
            for candidate in step["candidates"]
        )


def test_simulation_and_pose_solve_contacts_remain_distinct() -> None:
    first = _fixture()["steps"][0]
    selected = _candidate(first, first["selected_candidate_id"])
    simulation = selected["simulation_contacts"][0]
    pose_solve = selected["pose_solve_contacts"][0]

    assert "gap" in simulation
    assert "force" not in simulation
    assert "force" in pose_solve
    assert pose_solve["role"] == "target"
    assert np.linalg.norm(pose_solve["force"]) > 0.0


def test_replay_holds_shorter_scene_tracks_at_their_last_pose() -> None:
    first = _fixture()["steps"][0]
    selected = _candidate(first, first["selected_candidate_id"])
    scene_track = np.asarray(selected["simulation_tracks"]["11"], dtype=float)
    target_track = np.asarray(selected["simulation_tracks"]["22"], dtype=float)

    assert len(scene_track) < len(target_track)
    assert np.allclose(_track_pose_at(selected, 11, 2), scene_track[-1])
    assert np.allclose(_track_pose_at(selected, 22, 2), target_track[2])


def test_scene_motion_failure_preserves_parent_final_and_path() -> None:
    first = _fixture()["steps"][0]
    failed = _candidate(first, "step-1-candidate-2")
    motion = failed["scene_motion"]
    trajectory = np.asarray(motion["trajectory"], dtype=float)

    assert failed["rejection_reason"] == "scene_motion_limit"
    assert motion["object_id"] == 11
    assert np.allclose(trajectory[0], motion["parent_pose"])
    assert np.allclose(trajectory[-1], motion["failed_final_pose"])
    assert not np.allclose(motion["parent_pose"], motion["failed_final_pose"])


def test_camera_is_preserved_for_local_changes_and_refit_only_explicitly() -> None:
    camera = _fixture()["camera_behavior"]

    assert set(camera["preserve_on"]) == {
        "candidate_change",
        "mesh_source_change",
        "candidate_render_change",
        "candidate_pose_change",
        "layer_toggle",
        "refresh",
    }
    assert camera["refit_on"] == ["step_change", "explicit_reset"]
    assert np.isclose(np.linalg.norm(camera["camera"]["up"]), 1.0)

