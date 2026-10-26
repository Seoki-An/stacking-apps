"""Composite target queries preserve gaps, poses, and the legacy single box."""

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from stacking_common.domain import (
    CompositeTargetGeometry, ConvexGeometry, DirectionalTaper,
    TargetComponent, TargetGeometry,
)


def box(width=2., length=2., height=2.):
    return TargetGeometry(width, length, height, np.zeros(3), DirectionalTaper(), 0.)


def component(name, center=(0., 0., 0.), *, shape=None, quaternion=(0., 0., 0., 1.)):
    shape = box() if shape is None else shape
    return TargetComponent(name, ConvexGeometry(shape.points, shape.faces),
                           np.r_[center, quaternion])


def test_single_target_exposes_component_contract_without_changing_geometry():
    target = TargetGeometry(2., 4., 3., np.array([1., 2., 5.]),
                            DirectionalTaper.from_degrees(3., 6., 2., 4.), .02)
    comp, = target.components
    np.testing.assert_array_equal(comp.world_geometry.points, target.points)
    np.testing.assert_array_equal(comp.world_geometry.faces, target.faces)
    assert target.component_ids == ("target",)
    assert comp.world_geometry.volume == target.volume
    np.testing.assert_array_equal(target.bounds, [target.points.min(0), target.points.max(0)])
    points = np.array([[1., 2., 6.], [10., 2., 6.]])
    np.testing.assert_array_equal(target.component_membership(points), [[True], [False]])
    np.testing.assert_array_equal(target.horizontal_mask(points[:, :2], 6.), [True, False])


def test_disconnected_target_is_a_union_not_its_bounding_box_or_hull():
    target = CompositeTargetGeometry((component("left", (-3., 0., 0.)),
                                      component("right", (3., 0., 0.))))
    points = np.array([[-3., 0., 1.], [0., 0., 1.], [3., 0., 1.]])
    np.testing.assert_array_equal(target.contains(points), [True, False, True])
    np.testing.assert_array_equal(target.bounds, [[-4., -1., 0.], [4., 1., 2.]])
    np.testing.assert_array_equal(target.component_membership(points),
                                  [[True, False], [False, False], [False, True]])
    np.testing.assert_allclose(target.footprint_distance(points[:, :2]), [0., 2., 0.])
    np.testing.assert_allclose(target.footprint_distance([[0., 2.]]), [np.sqrt(5.)])
    assert target.component_ids == ("left", "right")
    assert len(target.points) == 16 and len(target.faces) == 24
    assert target.faces[:12].max() < 8 and target.faces[12:].min() >= 8


def test_l_shaped_union_preserves_notch_and_overlapping_membership():
    horizontal = component("horizontal", (1., .5, 0.), shape=box(4., 1.))
    vertical = component("vertical", (-.5, 2., 0.), shape=box(1., 4.))
    target = CompositeTargetGeometry((horizontal, vertical))
    xy = np.array([[2., .5], [-.5, 3.], [2., 3.], [-.5, .5], [3., 4.]])
    np.testing.assert_array_equal(target.horizontal_mask(xy, 1.),
                                  [True, True, False, True, False])
    assert not target.horizontal_mask(xy, 3.).any()
    assert target.component_membership([[-.5, .5, 1.]]).tolist() == [[True, True]]
    reverse = CompositeTargetGeometry((vertical, horizontal))
    np.testing.assert_array_equal(reverse.contains(np.c_[xy, np.ones(len(xy))]),
                                  target.horizontal_mask(xy, 1.))


def test_adjacent_component_seam_has_union_membership():
    target = CompositeTargetGeometry((component("left", (-1., 0., 0.)),
                                      component("right", (1., 0., 0.))))
    # Point membership is a true union. Native whole-stone containment remains
    # conservative and is tested separately; these are not the same query.
    points = np.array([[-.5, 0., 1.], [0., 0., 1.], [.5, 0., 1.]])
    assert target.contains(points).all()
    assert not target.component_membership(points).all(axis=0).any()


def test_rigid_transform_applied_once_and_projection_is_not_component_aabb():
    q = Rotation.from_euler("z", 45., degrees=True).as_quat()
    item = component("rotated", (2., 3., 5.), quaternion=q * 2.)
    np.testing.assert_allclose(np.linalg.norm(item.pose[3:]), 1.)
    target = CompositeTargetGeometry((item,))
    d = np.sqrt(2.)
    np.testing.assert_allclose(target.bounds, [[2.-d, 3.-d, 5.], [2.+d, 3.+d, 7.]])
    np.testing.assert_array_equal(target.contains([[2., 3., 6.], [3.3, 4.3, 6.]]),
                                  [True, False])
    np.testing.assert_allclose(target.footprint_distance([[2., 3.], [3.3, 4.3]]),
                              [0., 1.3 * d - 1.])
    # Also exercise a rotation not restricted to yaw.
    tilted = component("tilted", quaternion=Rotation.from_euler("y", 90., degrees=True).as_quat())
    np.testing.assert_allclose(tilted.bounds, [[0., -1., -1.], [2., 1., 1.]], atol=1e-15)


def test_arbitrary_convex_component_uses_its_shape_not_a_rectangle():
    points = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])
    item = TargetComponent("tetrahedron", ConvexGeometry(points, ConvexHull(points).simplices))
    target = CompositeTargetGeometry((item,))
    np.testing.assert_array_equal(target.horizontal_mask([[.2, .2], [1., 1.]], 1.), [True, False])
    np.testing.assert_allclose(target.footprint_distance([[.2, .2], [2., 2.]]), [0., np.sqrt(2.)])


def test_arrays_are_owned_and_component_world_geometry_is_immutable():
    geometry = ConvexGeometry(box().points, box().faces)
    pose = np.array([1., 2., 3., 0., 0., 0., 1.])
    item = TargetComponent("piece", geometry, pose)
    target = CompositeTargetGeometry([item])
    pose[0] = 100.
    assert item.pose[0] == 1.
    with pytest.raises(ValueError, match="read-only"):
        item.pose[0] = 2.
    with pytest.raises(ValueError, match="read-only"):
        item.world_geometry.points[0, 0] = 2.
    points, faces = target.meshes[0]
    points[:] = 0.
    faces[:] = 0
    assert target.points.any() and target.faces.any()
    bounds = target.bounds
    bounds[:] = 0.
    assert target.bounds.any()


@pytest.mark.parametrize("identifier", ["", " ", None, 2])
def test_invalid_component_ids(identifier):
    with pytest.raises(ValueError, match="ID"):
        component(identifier)


def test_empty_duplicate_and_wrong_component_types_rejected():
    with pytest.raises(ValueError, match="at least one"):
        CompositeTargetGeometry(())
    with pytest.raises(ValueError, match="unique"):
        CompositeTargetGeometry((component("piece"), component(" piece ")))
    with pytest.raises(TypeError, match="TargetComponent"):
        CompositeTargetGeometry((box(),))
    with pytest.raises(TypeError, match="ConvexGeometry"):
        TargetComponent("wrong", box())


@pytest.mark.parametrize("pose", [np.zeros(6), np.zeros(7), [np.nan, 0., 0., 0., 0., 0., 1.]])
def test_invalid_component_poses(pose):
    with pytest.raises(ValueError):
        TargetComponent("piece", ConvexGeometry(box().points, box().faces), pose)


def test_query_shapes_and_empty_queries():
    target = CompositeTargetGeometry((component("piece"),))
    assert target.contains(np.empty((0, 3))).shape == (0,)
    assert target.component_membership(np.empty((0, 3))).shape == (0, 1)
    assert target.footprint_distance(np.empty((0, 2))).shape == (0,)
    for xy in ([0., 0.], [[np.nan, 0.]], [[0., 0., 0.]]):
        with pytest.raises(ValueError, match="XY points"):
            target.horizontal_mask(xy, 0.)
    with pytest.raises(ValueError, match="height"):
        target.horizontal_mask([[0., 0.]], np.inf)
