"""Ground framing follows complete targets, independent of layer visibility."""

from dataclasses import replace
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stacking_common.domain.geometry import transform_points
from stacking_common.profiles.inventory import load_inventory_profile
from stacking_common.profiles.visualization import SceneBox
from stacking_common.visualization import TargetVolume, target_display
from stacking_common.visualization.operations import OperationsFrame
from stacking_common.visualization.rendering import fit_ground_to_target

ROOT = Path(__file__).resolve().parents[1]


def ground():
    return SceneBox("search_ground", "ground", [-2.6, -3.6, -.03],
                    [3.4, 2.4, 0.], (.353, .314, .275), 1.)


@pytest.mark.parametrize("name", [
    "default_long_wall", "tower_and_wall", "l_shaped_wall_nonsymmetric", "pyramid_large",
])
@pytest.mark.parametrize("world_frame", [False, True])
def test_all_components_have_equal_margins_and_center(name, world_frame):
    profile = load_inventory_profile(ROOT / "apps/excavator/config/inventory.yml", target_shape=name)
    target = target_display(profile.target.geometry(), structure_offset=[.2, -.3, 0.])
    pose = np.r_[[-6., 3., .5], Rotation.from_euler("z", 35, degrees=True).as_quat()] if world_frame else None
    points = target.points if pose is None else transform_points(target.points, pose)
    original = ground()
    fitted = fit_ground_to_target(original, target, margin=.5, transform=pose)
    lo, hi = points[:, :2].min(0), points[:, :2].max(0)
    np.testing.assert_allclose(lo - fitted.minimum[:2], [.5, .5])
    np.testing.assert_allclose(fitted.maximum[:2] - hi, [.5, .5])
    np.testing.assert_allclose((fitted.minimum[:2] + fitted.maximum[:2]) / 2, (lo + hi) / 2)
    assert fitted.minimum[2] == original.minimum[2] and fitted.maximum[2] == original.maximum[2]
    assert fitted.color == original.color and fitted.opacity == original.opacity
    np.testing.assert_array_equal(original.minimum, [-2.6, -3.6, -.03])


@pytest.mark.parametrize("platform,margin", [("excavator", .5), ("tabletop", .05)])
@pytest.mark.parametrize("recorded_boxes", [False, True])
def test_viewer_fits_ground_even_when_target_is_hidden(platform, margin, recorded_boxes):
    pytest.importorskip("pyvistaqt")
    module = importlib.import_module(f"stacking_{platform}.gui.operations.scene_view")
    target = TargetVolume([0., 0., 0.], 2., 7.2, 2.5)
    obstacle = replace(ground(), component_id="obstacle", layer="obstacle")
    pick = replace(ground(), component_id="pick_ground")
    boxes = (ground(), obstacle, pick) if platform == "excavator" else (ground(), obstacle)
    frame = OperationsFrame(1, (), (), (), (), target, None, None, 0, 0,
                            environment_boxes=boxes if recorded_boxes else None)
    view = Mock()
    view.visualization = SimpleNamespace(boxes=boxes)
    view._transparent_stones = False
    module.OperationsSceneView._populate_frame(
        view, frame, None, {"ground": True, "obstacle": True, "target": False})
    drawn = {call.args[0].component_id: call.args[0] for call in view._add_scene_box.call_args_list}
    np.testing.assert_allclose(drawn["search_ground"].minimum[:2], [-1.-margin, -3.6-margin])
    np.testing.assert_allclose(drawn["search_ground"].maximum[:2], [1.+margin, 3.6+margin])
    assert drawn["obstacle"] is obstacle
    if platform == "excavator":
        assert drawn["pick_ground"] is pick
    view._add_target.assert_not_called()


def test_missing_target_keeps_fallback_ground():
    box = ground()
    assert fit_ground_to_target(box, None, margin=.5) is box
