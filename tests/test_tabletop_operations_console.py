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


def test_tabletop_visualization_profile_preserves_environment_layers() -> None:
    profile = load_visualization_profile(
        ROOT / "apps/tabletop/config/visualization.yml"
    )
    boxes = {value.component_id: value for value in profile.boxes}

    assert profile.layer_visibility == {
        "ground": True,
        "support": False,
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
