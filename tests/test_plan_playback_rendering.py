"""Exercise real Qt playback signals and URDF visual/collision rendering."""

import json
import os
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pv = pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from stacking_common.artifacts import PlacementPlanRecord, decode_legacy_mcts_debug
from stacking_common.domain import Action
from stacking_common.planning import MotionPlanSegment, PlacementPlanningProblem, PlacementPlanningResult
from stacking_tabletop import TabletopApplication
from stacking_tabletop.gui.operations.controller import OperationsController
from stacking_tabletop.gui.operations.plan_view import PlanView
from stacking_tabletop.gui.operations.scene_view import OperationsSceneView, _prepare_robot_mesh
from stacking_tabletop.gui.operations.window import StackingOperationsWindow
from stacking_tabletop.planning import TabletopRobotTrajectoryProjector


ROOT = Path(__file__).resolve().parents[1]


def _controller(lengths=(3, 2)):
    artifact = decode_legacy_mcts_debug(json.loads(
        (ROOT / "tests/fixtures/legacy_visualization/mcts_candidate_viewer.json").read_text()
    ))
    pose = np.array([0, 0, 0.1, 0, 0, 0, 1.0])
    plan = PlacementPlanRecord(
        1, 1, PlacementPlanningProblem("world", Action(1, pose, pose, object_id=22), pose, pose, ()),
        PlacementPlanningResult("success", tuple(
            MotionPlanSegment(f"segment-{index}", np.zeros((count, 6)))
            for index, count in enumerate(lengths)
        )), "step-1-candidate-1",
    )
    controller = OperationsController()
    controller.set_artifact(artifact, planning_records=(plan,))
    return controller


class PlaybackWindow(StackingOperationsWindow):
    def __init__(self, controller):
        QMainWindow.__init__(self)
        self._video_exporting = False
        self.controller = controller
        self.plan_view = PlanView(controller)
        self.tabs = QTabWidget(self)
        self.tabs.addTab(self.plan_view, "Plan")
        self.scene_view = Mock()
        self.search_inspector = Mock()
        self.search_inspector.environment_visibility.return_value = {}
        self.plan_view.state_changed.connect(self._refresh_scene)
        self._refresh_scene()


@pytest.mark.parametrize("lengths", [(3, 2), (1, 1, 1)])
def test_playback_survives_window_refresh_and_crosses_segments(lengths):
    app = QApplication.instance() or QApplication([])
    window = PlaybackWindow(_controller(lengths))
    view = window.plan_view
    seen = []
    view.state_changed.connect(lambda: seen.append((window.controller.plan_segment_index, window.controller.plan_sample_index)))
    view.playback_timer.setInterval(1)
    try:
        view.play_button.click()
        deadline = time.monotonic() + 2
        while view.playback_timer.isActive() and time.monotonic() < deadline:
            QTest.qWait(10)
        assert window.controller.plan_segment_index == len(lengths) - 1
        assert window.controller.plan_sample_index == lengths[-1] - 1
        assert not view.playback_timer.isActive()
        assert len(seen) > sum(lengths) - 1  # intermediate visual frames
        assert view.play_button.text() == "Play"
        view.play_button.click()
        assert (window.controller.plan_segment_index, window.controller.plan_sample_index) == (0, 0)
        assert view.playback_timer.isActive()
        view.play_button.click()
        assert not view.playback_timer.isActive()
    finally:
        view._stop_playback()
        window.close()
        app.processEvents()


def test_changed_step_stops_playback_but_geometry_selection_does_not():
    app = QApplication.instance() or QApplication([])
    window = PlaybackWindow(_controller())
    view = window.plan_view
    try:
        view.play_button.click()
        view.robot_geometry_selector.setCurrentIndex(2)
        assert window.controller.robot_geometry_mode == "visual_collision"
        assert view.playback_timer.isActive()
        window.controller.select_step(1)
        window._refresh_scene()
        assert not view.playback_timer.isActive()
    finally:
        window.close()
        app.processEvents()


def test_whole_step_slider_scrubs_across_segments_and_segment_mode_stops_locally():
    app = QApplication.instance() or QApplication([])
    window = PlaybackWindow(_controller((3, 4, 2)))
    view = window.plan_view
    try:
        assert view.sample_slider.maximum() == 8
        view.sample_slider.setValue(5)
        assert (window.controller.plan_segment_index, window.controller.plan_sample_index) == (1, 2)
        assert view.segments.currentRow() == 1
        assert view.sample_label.text() == "6/9"
        view.sample_slider.setValue(8)
        assert (window.controller.plan_segment_index, window.controller.plan_sample_index) == (2, 1)
        view.segments.setCurrentRow(1)
        assert view.sample_slider.value() == 3
        view.playback_scope_selector.setCurrentIndex(1)
        assert view.sample_slider.maximum() == 3
        assert view.sample_slider.value() == 0
        view.playback_timer.setInterval(1)
        view.play_button.click()
        deadline = time.monotonic() + 2
        while view.playback_timer.isActive() and time.monotonic() < deadline:
            QTest.qWait(10)
        assert (window.controller.plan_segment_index, window.controller.plan_sample_index) == (1, 3)
        assert not view.playback_timer.isActive()
        view.playback_scope_selector.setCurrentIndex(0)
        assert view.sample_slider.value() == 6
        assert view.sample_slider.maximum() == 8
    finally:
        view._stop_playback()
        window.close()
        app.processEvents()


def test_projector_uses_visual_origins_and_keeps_collision_geometry_separate():
    application = TabletopApplication.from_repository(ROOT)
    projector = TabletopRobotTrajectoryProjector(application.planning_profile, ROOT)
    q = application.planning_profile.robot.home_positions
    visual = projector.project(q, 0.3)
    collision = projector.project(q, 0.3, geometry_mode="collision")
    both = projector.project(q, 0.3, geometry_mode="visual_collision")
    assert len(visual) == 12
    assert all(part.color == "#161616" for part in visual if part.part_id.startswith("gripper/"))
    assert len(collision) == 14
    assert len({part.part_id for part in both}) == 26
    assert all("visual" in part.mesh_path.parts for part in visual)
    assert all("collision" in part.mesh_path.parts for part in collision)
    assert any("base_link_inertia" in part.part_id for part in visual)
    # The fingertip visual STL has a different URDF origin from its fitted DSF.
    tip_visual = next(part for part in visual if "left_finger_tip" in part.part_id)
    tip_collision = next(part for part in collision if "left_finger_tip" in part.part_id)
    assert not np.allclose(tip_visual.pose[3:], tip_collision.pose[3:])
    np.testing.assert_allclose(tip_visual.pose[:3], tip_collision.pose[:3])


def test_playback_interpolates_timed_joints_and_opening_without_changing_record(monkeypatch):
    from dataclasses import replace
    app = QApplication.instance() or QApplication([])
    controller = _controller((2,))
    record = controller.current_plan
    segment = MotionPlanSegment("timed", np.array([[0.] * 6, [1.] * 6]),
        times=np.array([0., 2.]), openings=np.array([0.2, 0.8]))
    controller.set_artifact(controller.artifact, planning_records=(replace(record,
        result=PlacementPlanningResult("success", (segment,))),))
    projector = Mock()
    projector.project.return_value = ()
    controller.trajectory_projector = projector
    view = PlanView(controller)
    try:
        view._last_tick = 10.0
        monkeypatch.setattr("stacking_tabletop.gui.operations.plan_view.time.monotonic", lambda: 11.0)
        view._advance_playback()
        controller.planning_frame()
        np.testing.assert_allclose(projector.project.call_args.args[0], np.full(6, 0.5))
        assert projector.project.call_args.args[1] == pytest.approx(0.5)
        np.testing.assert_array_equal(segment.positions, [[0.] * 6, [1.] * 6])
        controller.select_plan_segment(0)
        assert controller.plan_sample_fraction == 0.0
    finally:
        view.close()
        app.processEvents()


def test_visual_stl_welds_facets_for_smooth_point_normals():
    source = pv.read(ROOT / "apps/tabletop/assets/sr_gripper/meshes/visual/body.stl")
    visual = _prepare_robot_mesh(source, "visual")
    assert "Normals" in visual.point_data
    assert np.all(np.isfinite(visual.point_data["Normals"]))
    # Unlike collision hulls the visual material has a soft specular response.
    from stacking_tabletop.gui.operations.scene_view import _ROBOT_VISUAL_MATERIAL
    assert _ROBOT_VISUAL_MATERIAL["specular"] > 0


def test_stationary_segment_boundary_interpolates_opening_but_not_a_joint_gap():
    from dataclasses import replace
    controller = _controller((1, 1))
    projector = Mock()
    projector.project.return_value = ()
    controller.trajectory_projector = projector
    for next_position, expected_opening in [(0.0, 0.5), (1.0, 0.2)]:
        record = controller.current_plan
        segments = (
            MotionPlanSegment("before", np.zeros((1, 6)), openings=np.array([0.2])),
            MotionPlanSegment("after", np.full((1, 6), next_position), openings=np.array([0.8])),
        )
        controller.set_artifact(controller.artifact, planning_records=(replace(record,
            result=PlacementPlanningResult("success", segments)),))
        controller.set_plan_interpolation(0.5)
        controller.planning_frame()
        np.testing.assert_array_equal(projector.project.call_args.args[0], np.zeros(6))
        assert projector.project.call_args.args[1] == pytest.approx(expected_opening)


def test_playback_updates_actor_transforms_without_rebuilding_static_scene():
    from dataclasses import replace
    from stacking_tabletop.gui.operations.scene_view import _robot_scene_key
    application = TabletopApplication.from_repository(ROOT)
    projector = TabletopRobotTrajectoryProjector(application.planning_profile, ROOT)
    parts = projector.project(application.planning_profile.robot.home_positions, 0.3)
    frame = replace(_controller().planning_frame(), robot_parts=parts)
    view = Mock()
    view._last_robot_scene_key = _robot_scene_key(frame, {})
    view.actors = {f"plan/{part.part_id}": Mock() for part in parts}
    view.actors.update({value.instance_id: Mock() for value in frame.objects})
    moved = tuple(replace(p, pose=p.pose + np.array([0.1, 0, 0, 0, 0, 0, 0])) for p in parts)
    OperationsSceneView.render_frame(view, replace(frame, robot_parts=moved),
        reset_camera=False, environment_visibility={})
    view._populate_frame.assert_not_called()
    view.render.assert_called_once()
    np.testing.assert_allclose(view.actors[f"plan/{moved[0].part_id}"].user_matrix[:3, 3], moved[0].pose[:3])
    assert _robot_scene_key(frame, {"target": False}) != view._last_robot_scene_key


def test_stone_replay_updates_transforms_without_rebuilding_scene():
    from dataclasses import replace
    from stacking_common.visualization import CandidatePoseMode
    from stacking_tabletop.gui.operations.scene_view import _robot_scene_key
    controller = _controller()
    controller.set_pose_mode(CandidatePoseMode.TRAJECTORY)
    frame = controller.frame()
    view = Mock()
    view._last_robot_scene_key = _robot_scene_key(frame, {})
    view.actors = {value.instance_id: Mock() for value in frame.objects}
    moved = tuple(replace(value, pose=value.pose + np.array([0.1, 0, 0, 0, 0, 0, 0]))
                  for value in frame.overlays)
    OperationsSceneView.render_frame(view, replace(frame, overlays=moved),
        reset_camera=False, environment_visibility={})
    view._populate_frame.assert_not_called()
    view.render.assert_called_once()
    for value in moved:
        np.testing.assert_allclose(view.actors[value.instance_id].user_matrix[:3, 3], value.pose[:3])
    OperationsSceneView.render_frame(view, frame, reset_camera=False,
                                    environment_visibility={"ground": False})
    view._populate_frame.assert_called_once()


def test_stone_actor_keeps_mesh_in_local_coordinates():
    from stacking_common.visualization import CandidatePoseMode
    controller = _controller()
    controller.set_pose_mode(CandidatePoseMode.TRAJECTORY)
    value = controller.frame().overlays[0]
    view = Mock()
    mesh = pv.Sphere()
    points = mesh.points.copy()
    view._object_mesh.return_value = mesh
    OperationsSceneView._add_object(view, value)
    assert view.add_mesh.call_args.args[0] is mesh
    np.testing.assert_allclose(mesh.points, points)
    np.testing.assert_allclose(view.add_mesh.return_value.user_matrix[:3, 3], value.pose[:3])
    assert view.add_mesh.call_args.kwargs["color"] == "#808080"


def test_collision_normals_are_outward_and_visual_vertex_colors_survive():
    paths = list((ROOT / "apps/tabletop/assets/ur5e/collision").glob("*.obj"))
    paths += list((ROOT / "apps/tabletop/assets/sr_gripper/meshes/collision").glob("*.obj"))
    for path in paths:
        mesh = _prepare_robot_mesh(pv.read(path), "collision")
        vectors = mesh.cell_centers().points - mesh.points.mean(axis=0)
        assert np.all(np.einsum("ij,ij->i", mesh.cell_data["Normals"], vectors) > 0), path
        assert "Normals" not in mesh.point_data
    source = pv.read(ROOT / "apps/tabletop/assets/ur5e/visual/shoulder.ply")
    visual = _prepare_robot_mesh(source, "visual")
    assert set(map(tuple, visual.point_data["RGBA"])).issubset(set(map(tuple, source.point_data["RGBA"])))
    assert "Normals" in visual.point_data
    assert np.all(np.isfinite(visual.point_data["Normals"]))


def test_normal_repair_corrects_reversed_faces_without_modifying_source():
    source = pv.Box().triangulate()
    faces = source.faces.reshape(-1, 4).copy()
    faces[::2, 1:] = faces[::2, 1:][:, ::-1]
    source.faces = faces.ravel()
    repaired = _prepare_robot_mesh(source, "collision")
    np.testing.assert_array_equal(source.faces, faces.ravel())
    centers = repaired.cell_centers().points - repaired.points.mean(axis=0)
    assert np.all(np.einsum("ij,ij->i", repaired.cell_data["Normals"], centers) > 0)


def test_render_material_uses_rgb_for_visual_and_flat_collision_wireframe():
    application = TabletopApplication.from_repository(ROOT)
    projector = TabletopRobotTrajectoryProjector(application.planning_profile, ROOT)
    parts = projector.project(application.planning_profile.robot.home_positions, 0.3, geometry_mode="visual_collision")
    visual = next(part for part in parts if "shoulder_link/visual" in part.part_id)
    collision = next(part for part in parts if "shoulder_link/collision" in part.part_id)
    view = Mock()
    view._robot_mesh.side_effect = lambda path, kind: _prepare_robot_mesh(pv.read(path), kind)
    OperationsSceneView._add_robot_part(view, visual)
    source = _prepare_robot_mesh(pv.read(visual.mesh_path), "visual")
    np.testing.assert_allclose(view.add_mesh.call_args.args[0].points, source.points * visual.mesh_scale)
    assert view.add_mesh.call_args.kwargs["scalars"] == "RGBA"
    assert view.add_mesh.call_args.kwargs["rgb"] is True
    OperationsSceneView._add_robot_part(view, collision, collision_overlay=True)
    assert view.add_mesh.call_args.kwargs["smooth_shading"] is False
    assert view.add_mesh.call_args.kwargs["style"] == "wireframe"
    OperationsSceneView._add_robot_part(view, collision)
    assert view.add_mesh.call_args.kwargs["style"] == "surface"
