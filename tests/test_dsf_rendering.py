"""Curved DSF geometry and matching operations-console surface controls."""

import importlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stacking_common.visualization.dsf import (
    dsf_support_points, object_surface_path, read_dsf_components,
)
from stacking_common.rendering.pyvista import read_dsf_surface

ROOT = Path(__file__).resolve().parents[1]
CUBE = np.array(list(itertools.product((-1., 1.), repeat=3)))


def test_support_map_is_curved_and_preserves_component_origin():
    directions = np.random.default_rng(8).normal(size=(100, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    center = np.array([4., -3., 2.])
    # For symmetric cube nodes at p=2, sum positive projections squared = 4:
    # the DSF is a sphere of radius 2, not the node hull or a smoothed cube.
    surface = dsf_support_points(CUBE + center, directions, 2)
    np.testing.assert_allclose(surface, center + 2 * directions, atol=1e-12)
    p16 = dsf_support_points(CUBE, [[1., 0., 0.]], 16)[0]
    np.testing.assert_allclose(p16, [4 ** (1/16), 0, 0], atol=1e-12)
    assert p16[0] > CUBE[:, 0].max()


def _two_components(path):
    text = "# sharpness: 2\no first\n"
    text += "".join(f"v {x} {y} {z}\n" for x, y, z in CUBE)
    text += "g second\n# sharpness: 16\n"
    text += "".join(f"v {x+10} {y} {z}\n" for x, y, z in CUBE)
    text += "f -1/-1 -2/-2 -3/-3\n"
    path.write_text(text)


def test_component_metadata_and_surface_do_not_bridge_concavities(tmp_path):
    pytest.importorskip("pyvista")
    path = tmp_path / "dsf.obj"
    _two_components(path)
    first, second = read_dsf_components(path)
    assert first[1] == 2 and second[1] == 16
    np.testing.assert_array_equal(first[0], CUBE)
    np.testing.assert_array_equal(second[0], CUBE + [10, 0, 0])
    mesh = read_dsf_surface(path)
    assert mesh.n_points > 1000
    regions = mesh.connectivity().cell_data["RegionId"]
    assert len(np.unique(regions)) == 2
    assert mesh.n_open_edges == 0
    assert np.all(np.isfinite(mesh.point_data["Normals"]))


@pytest.mark.parametrize("platform", ["tabletop", "excavator"])
def test_surface_toggle_caches_actual_dsf_and_ply_and_invalidates_actors(platform):
    pv = pytest.importorskip("pyvista")
    module = importlib.import_module(f"stacking_{platform}.gui.operations.scene_view")
    view = SimpleNamespace(_mesh_cache={}, _object_geometry_mode="dsf", _last_robot_scene_key="old")
    path = ROOT / f"apps/{platform}/assets/objects/model_10.obj"
    dsf = module.OperationsSceneView._recorded_object_mesh(view, path)
    assert "dsf_surface" in dsf.field_data
    assert dsf.n_points > pv.read(path).n_points
    module.OperationsSceneView.set_object_geometry_mode(view, "ply")
    assert view._last_robot_scene_key is None
    ply = module.OperationsSceneView._recorded_object_mesh(view, path)
    np.testing.assert_array_equal(ply.points, pv.read(path.with_name("model_10_mesh.ply")).points)
    module.OperationsSceneView.set_object_geometry_mode(view, "dsf")
    assert module.OperationsSceneView._recorded_object_mesh(view, path) is dsf


def test_gripper_collision_uses_dsf_surface():
    pytest.importorskip("pyvista")
    from stacking_excavator.gui.operations.scene_view import OperationsSceneView
    path = ROOT / "apps/excavator/assets/stone_grab_3/StoneGrab_Body_dsf.obj"
    mesh = OperationsSceneView._robot_mesh(SimpleNamespace(_robot_mesh_cache={}), path, "collision")
    assert "dsf_surface" in mesh.field_data
    assert mesh.n_points > 1000


def test_missing_ply_does_not_silently_display_node_hull(tmp_path):
    path = tmp_path / "model_1.obj"
    _two_components(path)
    assert object_surface_path(path, "dsf") == path
    with pytest.raises(FileNotFoundError, match="Missing PLY"):
        object_surface_path(path, "ply")


@pytest.mark.parametrize("platform", ["tabletop", "excavator"])
def test_console_surface_controls_match_and_keep_camera(platform, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QWidget
    from stacking_common.artifacts import decode_legacy_mcts_debug
    module = importlib.import_module(f"stacking_{platform}.gui.operations.window")
    application = QApplication.instance() or QApplication([])

    class View(QWidget):
        def __init__(self, parent, *args):
            super().__init__(parent)
            self.modes, self.resets = [], []

        def set_object_geometry_mode(self, mode):
            self.modes.append(mode)

        def render_frame(self, frame, *, reset_camera, **kwargs):
            self.resets.append(reset_camera)

        def show_empty(self):
            pass

    monkeypatch.setattr(module, "OperationsSceneView", View)
    window_type = module.StackingOperationsWindow if platform == "tabletop" else module.ExcavatorOperationsWindow
    window = window_type(ROOT)
    try:
        artifact = decode_legacy_mcts_debug(json.loads(
            (ROOT / "tests/fixtures/legacy_visualization/mcts_candidate_viewer.json").read_text()))
        window.controller.set_artifact(artifact)
        window._refresh_scene()
        selector = window.object_geometry_selector
        assert [(selector.itemText(i), selector.itemData(i)) for i in range(selector.count())] == [
            ("DSF surface", "dsf"), ("PLY mesh", "ply")]
        selector.setCurrentIndex(1)
        selector.setCurrentIndex(0)
        assert window.scene_view.modes == ["ply", "dsf"]
        assert window.scene_view.resets[-2:] == [False, False]
        robots = window.plan_view.robot_geometry_selector
        assert [robots.itemText(i) for i in range(robots.count())] == [
            "Visual meshes", "Collision meshes", "Visual + collision wireframe"]
    finally:
        window.close()


def test_all_excavator_objects_have_provenance_for_ply_surfaces():
    import hashlib
    directory = ROOT / "apps/excavator/assets"
    provenance = json.loads((directory / "provenance.json").read_text())
    objects = list((directory / "objects").glob("model_*.obj"))
    assert len(objects) == 49
    for obj in objects:
        ply = object_surface_path(obj, "ply")
        record = provenance["assets"]["assets/objects/" + ply.name]
        assert hashlib.sha256(ply.read_bytes()).hexdigest() == record["source_sha256"]
