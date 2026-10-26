"""Composite target profiles preserve the legacy shorthand and component frames."""

from dataclasses import asdict, fields
from itertools import product

import numpy as np
import pytest
import yaml

from stacking_common.domain import CompositeTargetGeometry, TargetGeometry
from stacking_common.profiles import (
    CompositeTargetProfile,
    TargetComponentProfile,
    TargetProfile,
    load_inventory_profile,
)


def shape_values():
    return {
        "width": 2.0,
        "length": 2.0,
        "height": 1.0,
        "origin": [0.0, 0.0, 0.0],
        "support_overlap": 0.0,
        "taper_degrees": {
            "width_positive": 0.0,
            "width_negative": 0.0,
            "length_positive": 0.0,
            "length_negative": 0.0,
        },
    }


def cube_nodes():
    return [list(point) for point in product([0.0, 1.0], repeat=3)]


def load_target(tmp_path, target):
    path = tmp_path / "inventory.yml"
    path.write_text(yaml.safe_dump({
        "units": "si",
        "objects": {
            "model_directory": "objects",
            "ids": [1, 2],
            "density": 1000.0,
        },
        "target": target,
    }))
    return load_inventory_profile(path).target


def test_single_target_profile_layout_and_geometry_are_unchanged(tmp_path):
    target = load_target(tmp_path, shape_values())
    assert type(target) is TargetProfile
    assert [field.name for field in fields(target)] == [
        "width", "length", "height", "origin", "support_overlap", "taper",
    ]
    assert set(asdict(target)) == {
        "width", "length", "height", "origin", "support_overlap", "taper",
    }
    geometry = target.geometry()
    assert type(geometry) is TargetGeometry
    assert geometry.volume == pytest.approx(4.0)
    np.testing.assert_array_equal(geometry.points.min(axis=0), [-1., -1., 0.])
    np.testing.assert_array_equal(geometry.points.max(axis=0), [1., 1., 1.])


def test_composite_loads_shapes_and_nodes_as_separate_posed_components(tmp_path):
    shape = shape_values()
    shape["origin"] = [1.0, 0.0, 0.0]
    profile = load_target(tmp_path, {"components": [
        {"id": "left", "shape": shape, "pose": [-4., 0., 0., 0., 0., 0., 1.]},
        {"id": "right", "nodes": cube_nodes(), "pose": [3., 0., 0., 0., 0., 0., 1.]},
    ]})
    assert type(profile) is CompositeTargetProfile
    assert all(isinstance(item, TargetComponentProfile) for item in profile.components)
    geometry = profile.geometry()
    assert type(geometry) is CompositeTargetGeometry
    assert [item.component_id for item in geometry.components] == ["left", "right"]
    np.testing.assert_array_equal(geometry.contains(np.array([
        [-3., 0., .5], [3.5, .5, .5], [0., 0., .5],
    ])), [True, True, False])
    assert not hasattr(profile, "width")
    assert not hasattr(profile, "origin")


def test_component_default_pose_and_outward_node_faces(tmp_path):
    profile = load_target(tmp_path, {"components": [
        {"id": "cube", "nodes": cube_nodes()},
    ]})
    component = profile.geometry().components[0]
    np.testing.assert_array_equal(component.pose, [0., 0., 0., 0., 0., 0., 1.])
    triangles = component.geometry.points[component.geometry.faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert np.all(np.einsum("ij,ij->i", normals, triangles.mean(axis=1) - .5) > 0)


def test_component_rotates_and_translates_local_nodes(tmp_path):
    profile = load_target(tmp_path, {"components": [
        {"id": "rotated", "nodes": cube_nodes(),
         "pose": [2., 0., 0., 0., 0., float(np.sqrt(.5)), float(np.sqrt(.5))]},
    ]})
    np.testing.assert_array_equal(profile.geometry().contains(np.array([
        [1.5, .5, .5], [2.5, .5, .5],
    ])), [True, False])


@pytest.mark.parametrize("target", [
    {"components": []},
    {"components": [{"id": "same", "nodes": cube_nodes()},
                    {"id": "same", "nodes": cube_nodes()}]},
    {"components": [{"nodes": cube_nodes()}]},
    {"components": [{"id": "", "nodes": cube_nodes()}]},
    {"components": [{"id": "   ", "nodes": cube_nodes()}]},
    {"components": [{"id": "missing-geometry"}]},
    {"components": [{"id": "both", "shape": shape_values(), "nodes": cube_nodes()}]},
    {"components": [{"id": "cube", "nodes": cube_nodes(), "position": [0., 0., 0.]}]},
    {"components": [{"id": "cube", "nodes": cube_nodes()}], "width": 1.0},
])
def test_invalid_composite_structure_is_rejected(tmp_path, target):
    with pytest.raises(ValueError):
        load_target(tmp_path, target)


@pytest.mark.parametrize("target", [
    {"components": "not-a-list"},
    {"components": {}},
    {"components": ["not-a-component"]},
    {"components": [{"id": "shape", "shape": []}]},
])
def test_invalid_composite_types_are_rejected(tmp_path, target):
    with pytest.raises(TypeError):
        load_target(tmp_path, target)


@pytest.mark.parametrize("nodes", [
    [],
    [[0., 0., 0.]],
    [[0., 0.], [1., 0.], [0., 1.], [1., 1.]],
    [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]],
    [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., float("nan")]],
    [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., float("inf")]],
])
def test_invalid_node_hulls_are_rejected(tmp_path, nodes):
    with pytest.raises(ValueError):
        load_target(tmp_path, {"components": [{"id": "invalid", "nodes": nodes}]})


@pytest.mark.parametrize("pose", [
    [0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0.],
    [float("inf"), 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., float("nan"), 1.],
])
def test_invalid_component_poses_are_rejected(tmp_path, pose):
    with pytest.raises(ValueError):
        load_target(tmp_path, {"components": [
            {"id": "invalid", "nodes": cube_nodes(), "pose": pose},
        ]})


def test_component_profile_copies_node_input():
    nodes = np.array(cube_nodes())
    profile = TargetComponentProfile("cube", nodes=nodes)
    nodes[:] = 0.0
    assert profile.geometry().geometry.volume == pytest.approx(1.0)
    assert not profile.nodes.flags.writeable
