"""Metric raycasting and occlusion checks independent of camera hardware."""

import numpy as np
import open3d as o3d

from stacking_common.multiview.synthetic import TriangleScene, sample_surface, supported_samples


def box_scene():
    box = o3d.geometry.TriangleMesh.create_box(2., 2., 1.).translate([-1., -1., 1.])
    return TriangleScene([(np.asarray(box.vertices), np.asarray(box.triangles))])


def test_raycast_returns_axial_depth_for_off_axis_rays():
    scene = box_scene()
    k = np.array([[4., 0, 2], [0, 4., 2], [0, 0, 1.]])
    hit = scene.image(k, np.eye(4), (5, 5))
    np.testing.assert_allclose(hit['depth'], 1., atol=1e-6)
    np.testing.assert_allclose(hit['points'][0, 0], [-.5, -.5, 1.], atol=1e-6)
    assert np.linalg.norm(hit['points'][0, 0]) > hit['depth'][0, 0]


def test_visibility_rejects_occlusion_behind_camera_and_outside_image():
    k = np.array([[4., 0, 2], [0, 4., 2], [0, 0, 1.]])
    points = np.array([[0, 0, 1.], [0, 0, 2.], [0, 0, -1.], [4, 0, 1.]])
    np.testing.assert_array_equal(box_scene().visible(points, k, np.eye(4), (5, 5)),
                                  [True, False, False, False])


def test_visibility_is_invariant_under_world_frame_change():
    mesh = o3d.geometry.TriangleMesh.create_box(2., 2., 1.).translate([-1., -1., 1.])
    change = np.diag([-1., -1., 1., 1.]); change[:3, 3] = [.3, -.7, .2]
    moved = np.asarray(mesh.vertices) @ change[:3, :3].T + change[:3, 3]
    scene = TriangleScene([(moved, np.asarray(mesh.triangles))])
    k = np.array([[4., 0, 2], [0, 4., 2], [0, 0, 1.]])
    np.testing.assert_allclose(scene.image(k, change, (5, 5))['depth'], 1., atol=1e-6)


def test_multiview_support_needs_visibility_and_translation():
    points = np.array([[0., 0., 1.], [.1, 0., 1.]])
    visible = np.array([[True, True], [True, False]])
    assert not supported_samples(visible, points, np.zeros((2, 3))).any()
    np.testing.assert_array_equal(supported_samples(visible, points, [[0, 0, 0], [.1, 0, 0]]), [True, False])


def test_surface_samples_preserve_area_weights_and_normals():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                         [3., 0., 0.], [5., 0., 0.], [3., 2., 0.]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    points, normals = sample_surface(vertices, faces, 10000)
    np.testing.assert_allclose(normals, np.tile([0, 0, 1.], (10000, 1)))
    assert abs(np.mean(points[:, 0] >= 3.) - .8) < .02
