"""Environment overlays must use the same rigid frame as the stone poses."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("pyvistaqt")
pv = pytest.importorskip("pyvista")

from stacking_common.planning.frames import FrameTransform
from stacking_tabletop.config import load_planning_profile, load_visualization_profile
from stacking_tabletop.gui.operations.environment import planning_scene_boxes, search_scene_boxes
from stacking_tabletop.gui.operations.scene_view import OperationsSceneView

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("angle", [0., .63])
def test_search_boxes_transform_all_vertices_and_preserve_ground(angle):
    planning = load_planning_profile(ROOT / "apps/tabletop/config/planning.yml")
    visualization = load_visualization_profile(ROOT / "apps/tabletop/config/visualization.yml")
    transform = FrameTransform("world", "stack", np.array([
        -.55, .2, .115, 0., 0., np.sin(angle / 2), np.cos(angle / 2)]))
    planning = replace(planning, world_T_stack=transform)
    boxes = search_scene_boxes(planning, visualization)
    assert boxes[0] is next(box for box in visualization.boxes if box.layer == "ground")
    assert boxes[0].maximum[2] == 0
    renderer = SimpleNamespace(add_mesh=Mock())
    for world, search in zip(planning_scene_boxes(planning), boxes[1:]):
        original = pv.Box(bounds=world.bounds).points.copy()
        expected = np.array([transform.parent_pose_to_child(
            np.r_[point, 0., 0., 0., 1.])[:3] for point in original])
        OperationsSceneView._add_scene_box(renderer, search)
        mesh = renderer.add_mesh.call_args.args[0]
        np.testing.assert_allclose(mesh.points, expected, atol=1e-7)
        np.testing.assert_allclose(pv.Box(bounds=world.bounds).points, original)
        assert search.layer == world.layer
        assert search.opacity == world.opacity
