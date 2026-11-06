"""Exact Minkowski distances agree with the original vertex/edge calculation."""

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from stacking_common.actions.connectivity import footprint_distances


def vertex_edge_reference(a, b, translations):
    values = []
    edges = np.concatenate((np.roll(a, -1, axis=0) - a, np.roll(b, -1, axis=0) - b))
    axes = np.column_stack((-edges[:, 1], edges[:, 0]))
    for offset in translations:
        moved = b + offset
        pa, pb = a @ axes.T, b @ axes.T + offset @ axes.T
        if not np.any((pb.min(0) > pa.max(0) + 1e-12)
                      | (pb.max(0) < pa.min(0) - 1e-12)):
            values.append(0.)
            continue
        closest = float('inf')
        for vertices, polygon in ((a, moved), (moved, a)):
            for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
                edge = end - start
                delta = vertices - start
                t = np.clip(delta @ edge / (edge @ edge), 0., 1.)
                closest = min(closest, np.linalg.norm(delta - t[:, None] * edge, axis=1).min())
        values.append(closest)
    return np.array(values)


@pytest.mark.parametrize('scale', [.001, 1., 100.])
def test_random_convex_polygons_match_vertex_edge_reference(scale):
    rng = np.random.default_rng(20261106)
    for _ in range(10):
        points = [rng.normal(size=(25, 2)) * scale for _ in range(2)]
        a, b = [p[ConvexHull(p).vertices] for p in points]
        shifts = rng.uniform(-8, 8, (80, 2)) * scale
        expected = vertex_edge_reference(a, b, shifts)
        np.testing.assert_allclose(footprint_distances(a, b, shifts), expected,
                                   rtol=2e-12, atol=2e-12 * scale)
        np.testing.assert_allclose(footprint_distances(a[::-1], b[::-1], shifts), expected,
                                   rtol=2e-12, atol=2e-12 * scale)


def test_touching_containment_and_empty_distance_queries():
    square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    shifts = np.array([[0., 0.], [1., 0.], [1., 1.], [2., 2.], [1. + 1e-13, 0.]])
    np.testing.assert_allclose(footprint_distances(square, square, shifts),
                               [0., 0., 0., np.sqrt(2), 0.], atol=1e-12)
    assert footprint_distances(square, square * .2, [[.4, .4]])[0] == 0.
    assert footprint_distances(square, square, []).shape == (0,)
