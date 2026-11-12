"""Synthetic stones must be new closed geometry with reproducible deformations."""

from itertools import product

import numpy as np
import pytest

from stacking_common.reconstruction.augmentation import deform_surface
from stacking_common.reconstruction.geometry import convex_faces, estimate_mesh_inertial, validate_triangle_mesh
from stacking_common.reconstruction.types import TriangleMeshData


def source_mesh():
    vertices = np.array(list(product((-1., 1.), repeat=3))) * [1., .7, .4]
    return TriangleMeshData(vertices, convex_faces(vertices))


def test_deformation_is_reproducible_preserves_topology_and_produces_valid_new_geometry():
    source = source_mesh()
    original = source.vertices.copy()
    mesh, record = deform_surface(source, 123)
    repeat, repeated_record = deform_surface(source, 123)
    other, _ = deform_surface(source, 124)
    np.testing.assert_array_equal(source.vertices, original)
    np.testing.assert_array_equal(mesh.triangles, source.triangles)
    np.testing.assert_array_equal(mesh.vertices, repeat.vertices)
    assert record == repeated_record
    assert not np.allclose(mesh.vertices, source.vertices)
    assert not np.allclose(mesh.vertices, other.vertices)
    assert validate_triangle_mesh(mesh).valid
    assert estimate_mesh_inertial(mesh).mass > 0
    assert 0 < record["warp_jacobian_norm_bound"] <= .30


def test_deformation_does_not_depend_on_input_translation():
    source = source_mesh()
    shifted = TriangleMeshData(source.vertices + [3., -5., 7.], source.triangles)
    mesh, _ = deform_surface(source, 123)
    translated, _ = deform_surface(shifted, 123)
    np.testing.assert_allclose(mesh.vertices, translated.vertices, atol=1e-12)


def test_flat_surface_is_not_accepted_as_a_synthetic_stone():
    mesh = TriangleMeshData(np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                            np.array([[0, 1, 2]]))
    with pytest.raises(ValueError, match="positive extent"):
        deform_surface(mesh, 123)
