import json
from dataclasses import replace
from pathlib import Path
import tomllib

import numpy as np
import pytest

from stacking_common.artifacts import StackRunStore, decode_legacy_mcts_debug
from stacking_common.visualization import CandidatePoseMode, CandidateRenderMode
from stacking_tabletop.config import load_visualization_profile
from stacking_tabletop.gui.operations import OperationsController
from stacking_tabletop.gui.operations.application import build_parser
from stacking_tabletop.gui.theme import DARK_STYLESHEET


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "legacy_visualization"
    / "mcts_candidate_viewer.json"
)


def _artifact():
    return decode_legacy_mcts_debug(
        json.loads(FIXTURE.read_text(encoding="utf-8"))
    )


def test_operations_console_has_distinct_canonical_command_names() -> None:
    values = tomllib.loads(
        (ROOT / "apps/tabletop/pyproject.toml").read_text(encoding="utf-8")
    )
    scripts = values["project"]["scripts"]

    assert scripts["stacking-tabletop-reconstruction-workbench"].endswith(
        "gui.reconstruction.application:main"
    )
    assert scripts["stacking-tabletop-operations"].endswith(
        "gui.operations.application:main"
    )
    assert scripts["stacking-tabletop-workbench"] == scripts[
        "stacking-tabletop-reconstruction-workbench"
    ]


def test_operations_parser_is_headless_and_accepts_initial_artifact() -> None:
    args = build_parser().parse_args(
        [str(ROOT), "--run", str(FIXTURE)]
    )

    assert args.repository == ROOT
    assert args.run == FIXTURE


def test_operations_controller_loads_native_stack_run(tmp_path: Path) -> None:
    path = StackRunStore.save_mcts(_artifact(), tmp_path / "search.stack")

    controller = OperationsController()
    loaded = controller.load(path)

    assert len(loaded.steps) == 2
    assert controller.current_step.selected_candidate_id == "step-1-candidate-1"


def test_refresh_reads_committed_steps_without_resetting_selection(tmp_path):
    artifact = _artifact()
    path = tmp_path / "growing.stack"
    with StackRunStore.create_mcts(path, replace(artifact, steps=())) as writer:
        writer.append_mcts_step(artifact.steps[0])
        controller = OperationsController()
        controller.load(path)
        controller.consume_camera_reset_request()
        candidate_id = controller.active_candidate_id
        controller.set_pose_mode(CandidatePoseMode.TRAJECTORY)
        controller.set_timeline_frame(1)
        writer.append_mcts_step(artifact.steps[1])
        controller.reload(path)
        assert len(controller.artifact.steps) == 2
        assert controller.current_step.step == artifact.steps[0].step
        assert controller.active_candidate_id == candidate_id
        assert controller.timeline_frame == 1
        assert controller.pose_mode is CandidatePoseMode.TRAJECTORY
        assert not controller.consume_camera_reset_request()
        loaded = controller.artifact
        with pytest.raises((OSError, ValueError)):
            controller.reload(tmp_path / "missing.stack")
        assert controller.artifact is loaded
        assert controller.timeline_frame == 1


def test_refresh_of_initially_empty_run_shows_first_committed_step(tmp_path):
    artifact = _artifact()
    path = tmp_path / "empty.stack"
    with StackRunStore.create_mcts(path, replace(artifact, steps=())) as writer:
        controller = OperationsController()
        controller.load(path)
        assert controller.current_step is None
        controller.consume_camera_reset_request()
        writer.append_mcts_step(artifact.steps[0])
        controller.reload(path)
        assert controller.current_step.step == artifact.steps[0].step
        assert controller.consume_camera_reset_request()


def test_tabletop_visualization_profile_preserves_environment_layers() -> None:
    profile = load_visualization_profile(
        ROOT / "apps/tabletop/config/visualization.yml"
    )
    boxes = {value.component_id: value for value in profile.boxes}

    assert profile.layer_visibility == {
        "ground": True,
        "support": True,
        "obstacle": False,
    }
    assert set(boxes) == {
        "ground",
        "pick_table",
        "place_table",
        "third_view_camera",
    }
    assert boxes["ground"].opacity == 1.0
    assert boxes["pick_table"].layer == "support"
    assert boxes["pick_table"].bounds == (
        -0.5,
        0.5,
        -1.0,
        -0.4,
        0.0,
        0.1,
    )
    assert boxes["third_view_camera"].layer == "obstacle"
    assert boxes["third_view_camera"].opacity == 0.2
    assert "background-color: #171b21" in DARK_STYLESHEET


def test_controller_preserves_camera_for_local_search_changes() -> None:
    controller = OperationsController(_artifact())

    assert controller.consume_camera_reset_request()
    controller.select_candidate("step-1-candidate-2")
    controller.set_render_mode(CandidateRenderMode.SELECTED)
    controller.set_pose_mode(CandidatePoseMode.TRAJECTORY)
    controller.set_timeline_frame(1)
    controller.set_diagnostic_layers(contacts=True, forces=True)
    assert not controller.consume_camera_reset_request()

    controller.select_step(1)
    assert not controller.consume_camera_reset_request()
    controller.request_camera_reset()
    assert controller.consume_camera_reset_request()


def test_search_inspector_distinguishes_active_and_planned_candidates() -> None:
    controller = OperationsController(_artifact())
    controller.consume_camera_reset_request()
    controller.select_candidate("step-1-candidate-2")
    controller.set_render_mode(CandidateRenderMode.SELECTED)

    frame = controller.frame()

    assert controller.current_step.selected_candidate_id == "step-1-candidate-1"
    assert frame.active_candidate_id == "step-1-candidate-2"
    assert [value.object_id for value in frame.overlays] == [33]
    assert not frame.overlays[0].selected
    assert frame.overlays[0].color == pytest.approx((1.0, 0.8, 0.05))


def test_candidate_overlays_use_legacy_score_palette() -> None:
    controller = OperationsController(_artifact())

    frame = controller.frame()

    assert len({value.color for value in frame.overlays}) == len(frame.overlays)
    assert frame.overlays[-1].color == pytest.approx((1.0, 0.8, 0.05))
    assert frame.overlays[-1].opacity == 1.0
    assert frame.overlays[1].color == pytest.approx((0.20, 0.32, 0.85))
    assert 0.42 <= frame.overlays[0].opacity <= 0.65


def test_operations_frame_supports_composite_and_timeline_pose_modes() -> None:
    controller = OperationsController(_artifact())
    controller.set_render_mode(CandidateRenderMode.SELECTED)
    controller.set_pose_mode(CandidatePoseMode.INITIAL_AND_SOLVED)

    composite = controller.frame()

    assert [value.layer for value in composite.overlays] == [
        "candidate_initial",
        "candidate_solved",
    ]
    assert not np.allclose(
        composite.overlays[0].pose,
        composite.overlays[1].pose,
    )

    controller.set_pose_mode(CandidatePoseMode.TRAJECTORY)
    controller.set_timeline_frame(2)
    trajectory = controller.frame()
    by_id = {value.object_id: value.pose for value in trajectory.overlays}
    candidate = controller.active_candidate

    assert trajectory.timeline_frame_count == 3
    assert len(trajectory.objects) == len({value.object_id for value in trajectory.objects})
    assert np.allclose(by_id[11], candidate.timeline.track(11).poses[-1])
    assert np.allclose(by_id[22], candidate.timeline.track(22).poses[2])


def test_operations_frame_exposes_selected_diagnostic_layers() -> None:
    controller = OperationsController(_artifact())
    controller.set_diagnostic_layers(contacts=True, forces=True)

    frame = controller.frame()

    assert len(frame.contacts) == 1
    assert len(frame.forces) == 1
    assert "force" not in frame.contacts[0].__dataclass_fields__
    assert np.linalg.norm(frame.forces[0].force) > 0.0


def test_replay_replaces_tracked_stones_but_keeps_untracked_scene_objects() -> None:
    artifact = _artifact()
    step = artifact.steps[0]
    untracked = replace(step.scene.objects[0], object_id=99)
    step = replace(step, scene=replace(step.scene, objects=step.scene.objects + (untracked,)))
    controller = OperationsController(replace(artifact, steps=(step,)))
    controller.set_pose_mode(CandidatePoseMode.TRAJECTORY)

    for index in range(controller.timeline_frame_count):
        controller.set_timeline_frame(index)
        frame = controller.frame()
        assert [value.object_id for value in frame.placed_objects] == [99]
        assert sorted(value.object_id for value in frame.objects) == [11, 22, 99]
        assert all(value.opacity == 1.0 for value in frame.objects)

    controller.set_pose_mode(CandidatePoseMode.ACTION)
    assert [value.object_id for value in controller.frame().placed_objects] == [11, 99]


def test_final_view_uses_active_candidates_simulated_neighbours():
    artifact = _artifact()
    step = artifact.steps[0]
    untracked = replace(step.scene.objects[0], object_id=99)
    step = replace(step, scene=replace(step.scene, objects=step.scene.objects + (untracked,)))
    controller = OperationsController(replace(artifact, steps=(step,)))
    controller.set_render_mode(CandidateRenderMode.SELECTED)
    controller.set_pose_mode(CandidatePoseMode.FINAL)
    candidate = controller.active_candidate
    frame = controller.frame()
    by_id = {value.object_id: value.pose for value in frame.objects}
    assert len(by_id) == len(frame.objects)
    assert [value.object_id for value in frame.placed_objects] == [99]
    for track in candidate.timeline.tracks:
        if track.object_id != candidate.object_id:
            np.testing.assert_allclose(by_id[track.object_id], track.poses[-1])
    np.testing.assert_allclose(by_id[candidate.object_id], candidate.final_pose)
    np.testing.assert_allclose(by_id[99], untracked.pose)
    controller.set_pose_mode(CandidatePoseMode.SOLVED)
    assert [value.object_id for value in controller.frame().placed_objects] == [11, 99]


def test_final_view_without_timeline_preserves_untracked_scene():
    from stacking_common.visualization import SceneTimeline
    artifact = _artifact()
    step = artifact.steps[0]
    candidate = replace(step.candidates[0], timeline=SceneTimeline(()))
    step = replace(step, candidates=(candidate,))
    controller = OperationsController(replace(artifact, steps=(step,)))
    controller.set_pose_mode(CandidatePoseMode.FINAL)
    frame = controller.frame()
    assert len(frame.overlays) == 1
    np.testing.assert_allclose(frame.overlays[0].pose, candidate.final_pose)
    assert len(frame.placed_objects) == len(step.scene.objects)


@pytest.mark.parametrize('mode', list(CandidateRenderMode))
@pytest.mark.parametrize('compare', [False, True])
def test_final_candidate_visibility_colors_and_simulation_start_comparison(mode, compare):
    controller = OperationsController(_artifact())
    controller.consume_camera_reset_request()
    controller.set_render_mode(mode)
    controller.set_pose_mode(CandidatePoseMode.SIMULATION_START_AND_FINAL if compare else CandidatePoseMode.FINAL)
    frame = controller.frame()
    visible = (() if mode is CandidateRenderMode.STATE_ONLY else
               (controller.active_candidate,) if mode is CandidateRenderMode.SELECTED else
               controller.current_step.candidates)
    finals = {item.instance_id: item for item in frame.overlays if item.layer == 'candidate_final'}
    starts = {item.instance_id: item for item in frame.overlays if item.layer == 'candidate_simulation_start'}
    assert len(finals) == sum(candidate.final_pose is not None for candidate in visible)
    assert len(starts) == (sum(candidate.pose(CandidatePoseMode.SIMULATION_START) is not None
                              for candidate in visible) if compare else 0)
    for candidate in visible:
        if candidate.final_pose is not None:
            final = finals[f'candidate/{candidate.candidate_id}/final']
            np.testing.assert_array_equal(final.pose, candidate.final_pose)
            assert final.color == pytest.approx((1., .8, .05))
            assert final.opacity == (1. if mode is not CandidateRenderMode.ALL
                                     or candidate is controller.active_candidate else .45)
        if compare and candidate.pose(CandidatePoseMode.SIMULATION_START) is not None:
            start = starts[f'candidate/{candidate.candidate_id}/simulation_start']
            np.testing.assert_array_equal(start.pose, candidate.pose(CandidatePoseMode.SIMULATION_START))
            assert start.color == pytest.approx((.15, .65, 1.))
            assert start.opacity <= .45
    if mode is CandidateRenderMode.STATE_ONLY:
        assert frame.overlays == ()
        assert len(frame.placed_objects) == len(controller.current_step.scene.objects)
    assert len({item.instance_id for item in frame.objects}) == len(frame.objects)
    assert not controller.consume_camera_reset_request()


def test_comparison_ghosts_only_moved_neighbours_of_the_active_candidate():
    controller = OperationsController(_artifact())
    controller.set_render_mode(CandidateRenderMode.SELECTED)
    controller.set_pose_mode(CandidatePoseMode.SIMULATION_START_AND_FINAL)
    active = controller.active_candidate
    frame = controller.frame()
    for track in active.timeline.tracks:
        if track.object_id == active.object_id:
            continue
        final = next(item for item in frame.overlays
                     if item.layer == 'final' and item.object_id == track.object_id)
        np.testing.assert_array_equal(final.pose, track.poses[-1])
        ghosts = [item for item in frame.overlays if item.layer == 'simulation_start_scene'
                  and item.object_id == track.object_id]
        moved = not np.allclose(track.poses[0], track.poses[-1], rtol=0., atol=1e-9)
        assert len(ghosts) == int(moved)
        if moved:
            np.testing.assert_array_equal(ghosts[0].pose, track.poses[0])


def test_zero_candidate_step_renders_state_without_timeline() -> None:
    controller = OperationsController(_artifact())
    controller.select_step(1)

    frame = controller.frame()

    assert frame.active_candidate_id is None
    assert frame.overlays == ()
    assert frame.timeline_frame_count == 0
    with pytest.raises(IndexError):
        controller.set_timeline_frame(0)


def test_multi_object_pose_modes_are_projected_by_console_controller() -> None:
    candidate = OperationsController(_artifact()).active_candidate

    with pytest.raises(ValueError, match="multi-object display mode"):
        candidate.pose(CandidatePoseMode.BEST_SEQUENCE)
