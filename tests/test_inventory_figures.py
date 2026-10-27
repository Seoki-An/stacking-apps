"""Inventory layouts must preserve physical scale and rigid mesh geometry."""

from itertools import product
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from stacking_common.visualization.inventory import inventory_grid, principal_axis_transform


def box_points(size):
    return np.array(list(product((-1., 1.), repeat=3))) * np.asarray(size)/2


def test_principal_axes_preserve_geometry_and_align_long_side_to_x():
    points = Rotation.from_euler("xyz", [.4, .8, -.3]).apply(box_points([3., 2., 1.])) + [5., -4., 7.]
    original = points.copy()
    transform = principal_axis_transform(points)
    rotation = transform[:3, :3]
    aligned = points @ rotation.T + transform[:3, 3]
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(rotation), 1., atol=1e-14)
    np.testing.assert_allclose(np.ptp(aligned, axis=0), [3., 2., 1.], atol=1e-14)
    np.testing.assert_allclose((aligned.min(axis=0)+aligned.max(axis=0))[:2], [0., 0.], atol=1e-14)
    np.testing.assert_allclose(aligned[:, 2].min(), 0., atol=1e-14)
    np.testing.assert_array_equal(points, original)


def test_grid_preserves_sizes_avoids_overlap_and_keeps_scale_on_partial_pages():
    meshes = {i: SimpleNamespace(points=box_points([i, .5*i, .25*i])) for i in range(17, 0, -1)}
    records, cell = inventory_grid(meshes)
    assert [r["object_id"] for r in records] == list(range(1, 18))
    assert [(r["page"], r["row"], r["column"]) for r in (records[0], records[3], records[4], records[16])] == [
        (1, 1, 1), (1, 1, 4), (1, 2, 1), (2, 1, 1)]
    np.testing.assert_allclose(cell, 1.5*17)
    for entry in records:
        object_id = entry["object_id"]
        points = meshes[object_id].points
        transform = entry["transform"]
        placed = points @ transform[:3, :3].T + transform[:3, 3]
        np.testing.assert_allclose(np.ptp(placed, axis=0), [object_id, .5*object_id, .25*object_id])
        center = (placed.min(axis=0)+placed.max(axis=0))/2
        expected = np.array([(entry["column"]-2.5)*cell, (2.5-entry["row"])*cell])
        np.testing.assert_allclose(center[:2], expected)
        assert np.all(np.abs(placed[:, :2]-expected) < cell/2)
