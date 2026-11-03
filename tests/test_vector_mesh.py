"""Vector projection must preserve visible geometry and world-frame ordering."""

import numpy as np
import pytest

from stacking_common.visualization.vector_mesh import orthographic_basis, vector_mesh_faces


def test_camera_basis_is_orthonormal_right_handed_and_world_up():
    basis = orthographic_basis([7., -4., 3.8])
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-14)
    assert np.linalg.det(basis) == pytest.approx(1.)
    assert basis[0, 2] == pytest.approx(0.)
    assert basis[1, 2] > 0


def test_vector_faces_cull_back_faces_and_draw_far_before_near_without_rescaling():
    basis = orthographic_basis([0., -1., 0.])
    near = np.array([[0., -2., 0.], [2., -2., 0.], [0., -2., 1.]])
    far = near + [0., 3., 0.]
    hidden = near[::-1] + [0., -1., 0.]
    triangles = np.array([near, hidden, far])
    original = triangles.copy()
    polygons, colors = vector_mesh_faces(triangles, basis)
    np.testing.assert_array_equal(polygons, np.array([far, near]) @ basis[:2].T)
    np.testing.assert_array_equal(triangles, original)
    assert colors.shape == (2, 3)
    assert np.isfinite(colors).all() and ((0 <= colors) & (colors <= 1)).all()


def test_zero_or_vertical_camera_is_rejected():
    for direction in ([0., 0., 0.], [0., 0., 1.], [np.nan, 0., 1.]):
        with pytest.raises(ValueError, match="camera direction"):
            orthographic_basis(direction)
