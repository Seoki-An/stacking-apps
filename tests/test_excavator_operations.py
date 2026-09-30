"""Recorded excavator replay, UI navigation, and isolated planning lifecycle."""

from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stacking_common.artifacts import PlacementPlanRecord, StackRunStore
from stacking_common.domain import Action, State
from stacking_common.planning import MotionPlanSegment, PlacementPlanningResult
from stacking_common.profiles.native import load_native_solver_profile
from stacking_common.planning import PickSceneSnapshot
from stacking_common.visualization import MCTSDebugArtifact, TargetVolume
from stacking_excavator import ExcavatorApplication
from stacking_excavator.planning.cli import planning_context
from stacking_excavator.gui.operations.controller import OperationsController
from stacking_excavator.gui.operations.application import build_parser

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/legacy_planner"


def matrix_pose(matrix):
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()]


def replay_run(path, case_index=1):
    application = ExcavatorApplication.from_repository(ROOT)
    reference = json.loads((FIXTURE / "reference.json").read_text())
    case = reference["cases"][case_index]
    object_id = case["action"]["stone_id"]
    pick = np.concatenate(reference["pickup_poses"][str(object_id)])
    inventory = application.load_inventory([object_id], {object_id: pick})
    solver = load_native_solver_profile(application.solver_profiles.grasp, application.solver_profiles.motion)
    context = planning_context(application, inventory, PickSceneSnapshot("world", {object_id: pick}, "recorded fixture"), solver, {})
    action = Action(0, case["action"]["pose"], case["action"]["pose"], object_id=object_id)
    problem = application.build_placement_planning_problem(State([object_id], [], {}, [], []), action, pick_pose=pick)
    names = (["pick_approach", "pick_retreat", "place_approach", "place_retreat"] if case_index == 0 else
             ["pick_approach", "pick_retreat", "handoff_place", "handoff_retreat",
              "handoff_approach", "handoff_pick", "place_approach", "place_retreat"])
    with np.load(FIXTURE / "motion.npz", allow_pickle=False) as arrays:
        segments = tuple(MotionPlanSegment(name,
            arrays[f"step_{case['step']}_joints_{index}"],
            target_poses=tuple(map(matrix_pose, arrays[f"step_{case['step']}_target_{index}"])),
            # Openings are synthetic UI characterization samples, not a legacy oracle.
            openings=np.linspace(-.4, -.2, 25), times=np.arange(25) * .01,
            scene_phase=name.split('_')[0]) for index, name in enumerate(names))
    record = PlacementPlanRecord(case["step"], 1, problem,
        PlacementPlanningResult("success", segments), "candidate-1", context.sha256)
    target = inventory.target
    with StackRunStore.create_mcts(path, MCTSDebugArtifact("stack", (),
        TargetVolume(target.origin, target.width, target.length, target.height)),
        metadata={"platform": "excavator"}) as writer:
        writer.append_run_context(context)
        writer.append_placement_plan(record)
    return record, context


@pytest.mark.parametrize("case_index", [0, 1])
def test_recorded_direct_regrasp_frames_and_partial_step(tmp_path, case_index):
    path = tmp_path / "run.stack"
    record, context = replay_run(path, case_index)
    controller = OperationsController(ROOT)
    controller.load(path)
    assert controller.current_step.state_only  # attempt exists before search-step commit
    assert controller.current_plan.step == record.step
    for index, segment in enumerate(record.result.segments):
        controller.select_plan_segment(index)
        controller.set_plan_sample(7)
        frame = controller.planning_frame()
        current = next(obj for obj in frame.objects if obj.instance_id == "plan/target")
        np.testing.assert_allclose(current.pose, segment.target_poses[7])
        assert len(frame.robot_parts) == 12
        assert len({part.part_id for part in frame.robot_parts}) == 12
        assert frame.environment_boxes[0].component_id == f"{segment.scene_phase}_ground"
        controller.set_plan_interpolation(.5)
        middle = next(obj for obj in controller.planning_frame().objects if obj.instance_id == "plan/target")
        np.testing.assert_allclose(middle.pose[:3], (segment.target_poses[7][:3] + segment.target_poses[8][:3])/2)
        np.testing.assert_allclose(frame.target_transform[:3], [-6., 3., 0.])
        # Old artifact headers omit taper, but their recorded physical context
        # is authoritative for this platform's replay and search inspector.
        assert frame.target.taper.width_positive == pytest.approx(np.deg2rad(5.))
    # Preserve signed, opposed-axis jaw convention through recorded profile.
    projector = controller.trajectory_projector
    opening = .5 * (segment.openings[7] + segment.openings[8])
    np.testing.assert_allclose(projector.gripper_state.positions, [opening, opening])
    assert not any(part.part_id.startswith("robot/grip_body/") for part in frame.robot_parts)
    assert len(StackRunStore.load_run_contexts(path)) == 1
    assert controller.frame().target.taper.width_positive == pytest.approx(np.deg2rad(5.))
    assert controller.artifact.target.taper.width_positive == 0.
    assert StackRunStore.load_mcts(path).target.taper.width_positive == 0.


def test_unknown_target_track_is_not_invented_and_assets_are_verified(tmp_path):
    path = tmp_path / "run.stack"
    record, context = replay_run(path)
    controller = OperationsController(ROOT)
    controller.load(path)
    controller.planning_records = (replace(record, result=replace(record.result,
        segments=(replace(record.result.segments[0], target_poses=None),))),)
    assert not any(obj.instance_id == "plan/target" for obj in controller.planning_frame().objects)
    from stacking_excavator.gui.operations.controller import recorded_planning_profile, verify_replay_assets
    bad = replace(context, asset_hashes={**context.asset_hashes, "robot/urdf": "0" * 64})
    with pytest.raises(ValueError, match="Recorded replay asset differs"):
        verify_replay_assets(ROOT, bad, recorded_planning_profile(context))


def test_search_only_and_wrong_platform_runs(tmp_path):
    path = tmp_path / "run.stack"
    _, context = replay_run(path)
    search = tmp_path / "search.stack"
    with StackRunStore.create_mcts(search, MCTSDebugArtifact("stack", ()), metadata={"platform": "excavator"}) as writer:
        writer.append_run_context(context)
    controller = OperationsController(ROOT)
    controller.load(search)
    assert controller.current_plan is None
    assert controller.contexts
    wrong = tmp_path / "tabletop.stack"
    with StackRunStore.create_mcts(wrong, MCTSDebugArtifact("stack", ())):
        pass
    with pytest.raises(ValueError, match="excavator"):
        controller.load(wrong)
    assert controller.contexts  # failed loads preserve the previous selection


def test_console_parser_has_explicit_excavator_entrypoint():
    args = build_parser().parse_args([str(ROOT), "--run", "plan.stack"])
    assert args.repository == ROOT
    assert args.run == Path("plan.stack")


def test_console_navigation_and_playback(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QWidget
    from PySide6.QtTest import QTest
    import stacking_excavator.gui.operations.window as windows
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "run.stack"
    replay_run(path)

    class View(QWidget):
        def __init__(self, parent, *args):
            super().__init__(parent)
            self.frames = []
        def render_frame(self, frame, **kwargs):
            self.frames.append(frame)
        def show_empty(self):
            pass

    monkeypatch.setattr(windows, "OperationsSceneView", View)
    window = windows.ExcavatorOperationsWindow(ROOT, path)
    try:
        assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
            "Plan", "Search Inspector", "Perception", "Execution", "History"]
        assert window.plan_view.segments.count() == 8
        view = window.plan_view
        view.sample_slider.setValue(view.sample_slider.maximum() - 1)
        view.play_button.click()
        for _ in range(50):
            QTest.qWait(10)
            if not view.playback_timer.isActive():
                break
        assert window.controller.plan_segment_index == 7
        assert window.controller.plan_sample_index == 24
        assert not view.playback_timer.isActive()
        assert any(obj.instance_id == "plan/target" for obj in window.scene_view.frames[-1].objects)
        window._new_planning()
        assert window._planning_dialog is not None
        assert not window._planning_dialog.running
    finally:
        window.close()


def test_refresh_preserves_excavator_plan_and_search_tab(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QWidget
    import stacking_excavator.gui.operations.window as windows
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "refresh.stack"
    record, _ = replay_run(path)

    class View(QWidget):
        def __init__(self, parent, *args):
            super().__init__(parent)
            self.resets = []
        def render_frame(self, frame, **kwargs):
            self.resets.append(kwargs["reset_camera"])
        def show_empty(self):
            pass

    monkeypatch.setattr(windows, "OperationsSceneView", View)
    window = windows.ExcavatorOperationsWindow(ROOT, path)
    try:
        window.tabs.setCurrentWidget(window.search_inspector)
        window.controller.select_plan_segment(1)
        window.controller.set_plan_sample(7)
        window.controller.set_plan_interpolation(.4)
        with StackRunStore.open_append(path) as writer:
            writer.append_placement_plan(replace(record, attempt=record.attempt + 1))
            window.search_inspector.refresh_button.click()
            assert len(window.controller.current_plans) == 2
            assert window.controller.plan_attempt == record.attempt
            assert window.controller.plan_segment_index == 1
            assert window.controller.plan_sample_index == 7
            assert window.controller.plan_sample_fraction == .4
            assert window.tabs.currentWidget() is window.search_inspector
            assert window.scene_view.resets[-1] is False
            assert "Refreshed" in window.statusBar().currentMessage()
            writer.append_placement_plan(replace(record, step=record.step + 1))
            window.refresh_action.trigger()
            assert window.search_inspector.step_selector.count() == 2
            assert window.controller.current_step.step == record.step
    finally:
        window.close()
        app.processEvents()


def test_plan_step_selector_replays_all_steps_and_syncs_tabs(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QWidget
    from PySide6.QtTest import QTest
    import stacking_excavator.gui.operations.window as windows
    app = QApplication.instance() or QApplication([])
    fixture = tmp_path / "fixture.stack"
    record, context = replay_run(fixture)
    path = tmp_path / "five-steps.stack"
    with StackRunStore.create_mcts(path, StackRunStore.load_mcts(fixture),
                                  metadata={"platform": "excavator"}) as writer:
        writer.append_run_context(context)
        for step in range(1, 6):
            segments = []
            for segment in record.result.segments:
                positions = segment.positions.copy()
                positions[:, 0] += step * .01
                segments.append(replace(segment, positions=positions))
            writer.append_placement_plan(replace(record, step=step,
                result=replace(record.result, segments=tuple(segments))))

    class View(QWidget):
        def __init__(self, parent, *args):
            super().__init__(parent)
            self.frames = []
        def render_frame(self, frame, **kwargs):
            self.frames.append(frame)
        def show_empty(self):
            pass

    monkeypatch.setattr(windows, "OperationsSceneView", View)
    window = windows.ExcavatorOperationsWindow(ROOT, path)
    try:
        view = window.plan_view
        assert view.step_selector.count() == 5
        for step in range(1, 6):
            view.step_selector.setCurrentIndex(view.step_selector.findData(step))
            assert window.controller.current_plan.step == step
            assert view.segments.count() == len(record.result.segments)
            assert view.play_button.isEnabled()
            np.testing.assert_allclose(window.controller.current_plan_segment.positions[:, 0],
                                       record.result.segments[0].positions[:, 0] + step * .01)
            view.sample_slider.setValue(view.sample_slider.maximum() - 1)
            view.play_button.click()
            for _ in range(50):
                QTest.qWait(10)
                if not view.playback_timer.isActive():
                    break
            assert not view.playback_timer.isActive()
            assert view.sample_slider.value() == view.sample_slider.maximum()
            assert window.scene_view.frames[-1].step == step
            assert window.scene_view.frames[-1].robot_parts
            window.tabs.setCurrentWidget(window.search_inspector)
            assert window.search_inspector.step_selector.currentData() == step
            window.tabs.setCurrentWidget(view)
        window.tabs.setCurrentWidget(window.search_inspector)
        window.search_inspector.step_selector.setCurrentIndex(1)
        window.tabs.setCurrentWidget(view)
        assert view.step_selector.currentData() == 2
        assert view.sample_slider.value() == 0
        view.play_button.click()
        view.step_selector.setCurrentIndex(3)
        assert not view.playback_timer.isActive()
        assert window.controller.current_plan.step == 4
        assert view.sample_slider.value() == 0
    finally:
        window.close()


def test_planning_dialog_launch_and_stop_preserves_run(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest
    from stacking_excavator.gui.operations.planning_run import PlanningRunDialog
    app = QApplication.instance() or QApplication([])
    pickup = tmp_path / "pickup.json"
    pickup.write_text(json.dumps({55: [2., -4., .4, 0., 0., 0., 1.]}))
    path = tmp_path / "planning.stack"
    dialog = PlanningRunDialog(ROOT)
    dialog.pick_path.setText(str(pickup))
    dialog.output_path.setText(str(path))
    dialog.search_only.setChecked(True)
    assert "--search-only" in dialog.arguments()
    assert dialog.pose_from_scan.currentData() is None
    assert [dialog.pose_from_scan.itemText(i) for i in range(dialog.pose_from_scan.count())] == [
        "Use configuration (no override)", "true", "false"]
    assert "--pose-from-scan" not in dialog.arguments()
    assert "--no-pose-from-scan" not in dialog.arguments()
    configured = replace(ExcavatorApplication.from_repository(ROOT), pose_from_scan=True)
    with monkeypatch.context() as patch:
        patch.setattr(ExcavatorApplication, "from_repository", lambda repository: configured)
        dialog.pick_path.clear()
        with pytest.raises(ValueError, match="Choose recorded pickup poses"):
            dialog.arguments()
    dialog.pick_path.setText(str(pickup))
    dialog.pose_from_scan.setCurrentIndex(dialog.pose_from_scan.findData(True))
    assert "--pose-from-scan" in dialog.arguments()
    dialog.pick_path.clear()
    with pytest.raises(ValueError, match="Choose recorded pickup poses"):
        dialog.arguments()
    dialog.pose_from_scan.setCurrentIndex(dialog.pose_from_scan.findData(False))
    assert "--no-pose-from-scan" in dialog.arguments()
    assert "--pose-from-scan" not in dialog.arguments()
    dialog.pick_path.clear()
    assert "--pick-pose-data" not in dialog.arguments()
    dialog.search_only.setChecked(False)
    with pytest.raises(ValueError, match="Choose recorded pickup poses"):
        dialog.arguments()
    dialog.search_only.setChecked(True)
    dialog.pick_path.setText(str(pickup))
    dialog.start()
    try:
        deadline = time.monotonic() + 8
        # File creation and context append are separate commits. Stop only
        # after the context exists so this tests preservation after startup.
        while dialog.running and time.monotonic() < deadline:
            if path.exists() and StackRunStore.load_run_contexts(path):
                break
            QTest.qWait(20)
        assert path.exists(), dialog.log.toPlainText()
        assert StackRunStore.load_run_contexts(path), dialog.log.toPlainText()
        dialog.reject()
        deadline = time.monotonic() + 8
        while dialog.running and time.monotonic() < deadline:
            QTest.qWait(20)
        assert not dialog.running
        assert StackRunStore.info(path).status == "cancelled"
        assert dialog.open_button.isEnabled()
        assert StackRunStore.load_run_contexts(path)
        assert StackRunStore.load_run_contexts(path)[0].profiles["pose_from_scan"] is False
    finally:
        if dialog.running:
            dialog.reject()
            dialog.process.waitForFinished(10000)
        dialog.close()
