"""Reference-plane footprint sizes and frame invariance."""
import numpy as np
from scipy.spatial.transform import Rotation

from stacking_common.multiview.coverage import plane_coverage


K = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])


def test_footprint_scales_with_distance_and_bounds_translated_views():
    center = np.eye(4)
    right = np.eye(4); right[0, 3] = .1
    coverage = plane_coverage({'center': center, 'right': right}, K, np.zeros(8),
                              (640, 480), [0, 0, .4], np.eye(3))
    np.testing.assert_allclose(coverage['footprints'][0]['span_m'], [.5112, .3832])
    np.testing.assert_allclose(coverage['combined_bounding_span_m'], [.6112, .3832])
    closer = plane_coverage({'center': center}, K, np.zeros(8), (640, 480), [0, 0, .2], np.eye(3))
    np.testing.assert_allclose(closer['footprints'][0]['span_m'], [.2556, .1916])


def test_coverage_is_frame_invariant_and_uses_distortion():
    dist = np.array([.1, -.02, .001, -.002, 0.])
    reference = plane_coverage({'center': np.eye(4)}, K, dist, (640, 480), [0, 0, .4], np.eye(3))
    rotation = Rotation.from_euler('xyz', [20, 35, -60], degrees=True).as_matrix()
    translation = np.array([.1, -.5, .2])
    camera = np.eye(4); camera[:3, :3] = rotation; camera[:3, 3] = translation
    transformed = plane_coverage({'center': camera}, K, dist, (640, 480),
                                 rotation @ [0, 0, .4] + translation, rotation)
    np.testing.assert_allclose(transformed['footprints'][0]['plane_xy_m'],
                               reference['footprints'][0]['plane_xy_m'], atol=1e-12)
    assert reference['footprints'][0]['span_m'][0] < .5112


def test_plane_horizon_does_not_produce_misleading_finite_bounds():
    camera = np.eye(4)
    camera[:3, :3] = Rotation.from_euler('y', 70, degrees=True).as_matrix()
    coverage = plane_coverage({'center': np.eye(4), 'side': camera}, K, np.zeros(8),
                              (640, 480), [0, 0, .4], np.eye(3))
    assert coverage['footprints'][0]['bounded']
    assert not coverage['footprints'][1]['bounded']
    assert coverage['combined_bounding_span_m'] is None
