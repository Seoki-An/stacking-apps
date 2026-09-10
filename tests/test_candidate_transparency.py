"""Outward normals and consistent transparency in both operations viewers."""

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from stacking_common.rendering.pyvista import read_dsf_surface
from stacking_common.visualization.operations import RenderObject
from stacking_common.visualization.rendering import (
    configure_scene_passes, scene_is_translucent,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("platform", ["tabletop", "excavator"])
def test_dsf_normals_and_winding_point_out_of_each_convex_component(platform):
    # Both assets contain multiple convex components. A whole-body center is
    # not an interior point of each component and gives false inward reports.
    mesh = read_dsf_surface(ROOT / f"apps/{platform}/assets/objects/model_8.obj")
    regions = mesh.connectivity()
    for region in np.unique(regions.point_data["RegionId"]):
        selected = regions.point_data["RegionId"] == region
        points = regions.points[selected]
        normals = regions.point_data["Normals"][selected]
        center = points.mean(axis=0)
        assert np.all(np.einsum("ij,ij->i", normals, points-center) > 0)
        face_indices = regions.faces.reshape(-1, 4)[:, 1:]
        faces = face_indices[regions.cell_data["RegionId"] == region]
        triangles = regions.points[faces]
        geometric_normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        assert np.all(np.einsum("ij,ij->i", geometric_normals, triangles.mean(axis=1)-center) > 0)


@pytest.mark.parametrize("platform", ["tabletop", "excavator"])
@pytest.mark.parametrize("opacity", [.4, 1.])
def test_stone_material_culls_only_translucent_backfaces(platform, opacity):
    module = importlib.import_module(f"stacking_{platform}.gui.operations.scene_view")
    mesh = pv.Sphere()
    original_points = mesh.points.copy()
    view = Mock()
    view._object_mesh.return_value = mesh
    value = RenderObject("candidate/8", 8, [0, 0, 0, 0, 0, 0, 1], "candidate", opacity=opacity)
    module.OperationsSceneView._add_object(view, value)
    kwargs = view.add_mesh.call_args.kwargs
    assert kwargs["culling"] == ("back" if opacity < 1. else False)
    assert kwargs["smooth_shading"] is True
    assert kwargs["opacity"] == opacity
    assert kwargs["ambient"] == .42 and kwargs["diffuse"] == .58 and kwargs["specular"] == 0
    np.testing.assert_array_equal(mesh.points, original_points)


def test_translucent_layers_select_the_correct_pass():
    frame = SimpleNamespace(target=None, objects=(), robot_parts=())
    assert not scene_is_translucent(frame, (), {})
    frame.target = object()
    assert scene_is_translucent(frame, (), {})
    assert not scene_is_translucent(frame, (), {"target": False})
    frame.target = None
    box = SimpleNamespace(layer="obstacle", opacity=.3)
    assert not scene_is_translucent(frame, (box,), {"obstacle": False})
    assert scene_is_translucent(frame, (box,), {"obstacle": True})
    frame.objects = (SimpleNamespace(opacity=.4),)
    assert scene_is_translucent(frame, (), {})
    frame.objects = ()
    frame.robot_parts = (SimpleNamespace(part_id="scan_preview/1"),)
    assert scene_is_translucent(frame, (), {})


def test_actual_depth_peeling_survives_switch_from_ssao():
    view = pv.Plotter(off_screen=True, window_size=(160, 160))
    try:
        if not view.enable_depth_peeling():
            pytest.skip("OpenGL backend does not support depth peeling")
        configure_scene_passes(view, translucent=False, occlusion_radius=.3, occlusion_bias=.015)
        assert view.renderer.GetPass() is not None
        configure_scene_passes(view, translucent=True, occlusion_radius=.3, occlusion_bias=.015)
        assert view.renderer.GetPass() is None
        assert view.render_window.GetMultiSamples() == 0
        assert view.renderer.GetMaximumNumberOfPeels() == 0
        view.set_background("black")
        view.add_mesh(pv.Sphere(), opacity=.4, color="red", culling="back", lighting=False)
        view.show(auto_close=False)
        assert view.renderer.GetLastRenderingUsedDepthPeeling() == 1
        pixels = view.screenshot(return_img=True)
        assert np.count_nonzero(pixels[:, :, 0] > 30) > 100
        configure_scene_passes(view, translucent=False, occlusion_radius=.3, occlusion_bias=.015)
        assert view.renderer.GetPass() is not None
    finally:
        view.close()
