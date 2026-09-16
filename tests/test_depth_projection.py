"""Independent projection and analytic-intersection checks, without device SDKs."""

import numpy as np
import pytest

from stacking_common.pointcloud.depth import optical_rays, backproject_depth
from stacking_common.pointcloud.synthetic_depth import raycast_ellipsoids


def test_axial_depth_units_pixel_indices_and_noncommuting_transform():
    rays = optical_rays(3, 2, [[2., 0., 1.], [0., 2., 0.], [0., 0., 1.]])
    cloud, pixels = backproject_depth(np.array([[0, 4000, 4000], [8000, 0, 0]], np.uint16), rays, .00025)
    np.testing.assert_array_equal(pixels, [1, 2, 3])
    np.testing.assert_allclose(cloud.points, [[0, 0, 1], [.5, 0, 1], [-1, 1, 2]])
    pose = np.array([[0, -1, 0, .3], [1, 0, 0, -.2], [0, 0, 1, .5], [0, 0, 0, 1]])
    np.testing.assert_allclose(cloud.transformed(pose).points, [[.3, -.2, 1.5], [.3, .3, 1.5], [-.7, -1.2, 2.5]])


def test_brown_inverse_matches_independent_forward_equations():
    matrix = np.array([[80., 0., 31.], [0., 81., 24.], [0., 0., 1.]])
    coeff = [.12, -.04, .001, -.002, .005]
    rays = optical_rays(64, 48, matrix, coeff, model='brown_conrady')
    x, y = rays[..., 0], rays[..., 1]
    r2 = x*x+y*y
    radial = 1 + .12*r2 - .04*r2**2 + .005*r2**3
    u = 80*(x*radial + .002*x*y - .002*(r2+2*x*x))+31
    v = 81*(y*radial - .004*x*y + .001*(r2+2*y*y))+24
    expected_u, expected_v = np.meshgrid(np.arange(64), np.arange(48))
    np.testing.assert_allclose(u, expected_u, atol=1e-9)
    np.testing.assert_allclose(v, expected_v, atol=1e-9)


@pytest.mark.parametrize('model', ['inverse_brown_conrady', 'equidistant', 'unknown'])
def test_unsupported_projection_never_silently_uses_pinhole(model):
    with pytest.raises(ValueError, match='unsupported'):
        optical_rays(2, 2, np.eye(3), np.zeros(5), model=model)


def test_invalid_geometry_units_and_distortion():
    with pytest.raises(ValueError, match='geometry'):
        optical_rays(2, 2, np.diag([0, 1, 1]))
    with pytest.raises(ValueError, match='contradict'):
        optical_rays(2, 2, np.eye(3), [1, 0, 0, 0, 0])
    rays = optical_rays(2, 2, np.eye(3))
    with pytest.raises(ValueError):
        backproject_depth(np.ones((2, 2)), rays, 0)
    cloud, pixels = backproject_depth(np.array([[np.nan, np.inf], [0, 1.]]), rays, 1.)
    assert len(cloud) == 1 and pixels.tolist() == [3]


def test_analytic_nearest_surface_occlusion_and_axial_not_range():
    rays = optical_rays(2, 1, np.eye(3))
    depth, labels = raycast_ellipsoids(rays, np.eye(4), [[0, 0, 2]], [[.5, .5, .5]],
                                      3., [[-4, -4], [4, 4]])
    np.testing.assert_allclose(depth, [[1.5, 3.]])
    np.testing.assert_array_equal(labels, [[2, 1]])
