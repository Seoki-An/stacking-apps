"""Geometry checks independent of model weights and CUDA."""
import numpy as np
import pytest
from stacking_common.multiview.learned_stereo import rectify_pair, disparity_points, select_pairs


@pytest.mark.parametrize('direction', [(1, 0, 0), (0, 1, 0), (-1, 0, 0)])
def test_rectification_is_horizontal_metric_for_arbitrary_baseline(direction):
    a = np.eye(4); b = np.eye(4); b[:3, 3] = np.array(direction)*.1
    k = np.array([[100., 0, 79.5], [0, 100., 59.5], [0, 0, 1]])
    images = [np.zeros((120, 160, 3), np.uint8)]*2
    _, _, kr, pose, baseline = rectify_pair(images, [k]*2, [np.zeros(5)]*2, [a, b], 80)
    centers = (b[:3, 3]-a[:3, 3]) @ pose[:3, :3]
    np.testing.assert_allclose(centers, [.1, 0, 0], atol=1e-10)
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.)
    assert kr[0, 0] == pytest.approx(50.)
    disparity = np.full((60, 80), 5., np.float32)
    points, valid = disparity_points(disparity, disparity, [np.ones_like(disparity, bool)]*2,
                                     kr, pose, baseline)
    camera = (points-pose[:3, 3]) @ pose[:3, :3]
    np.testing.assert_allclose(camera[:, 2], 1.)
    assert not valid[:, :5].any()


def test_occlusion_invalid_depth_and_sensor_border_rejected():
    d = np.full((4, 20), 2., np.float32)
    reverse = d.copy(); reverse[:, 5:8] = 8
    d[0, 15] = np.nan; d[1, 15] = 0
    masks = [np.ones_like(d, bool), np.ones_like(d, bool)]
    masks[1][2, 10] = False
    points, valid = disparity_points(d, reverse, masks, np.eye(3), np.eye(4), .1)
    assert not valid[:, 7:10].any()
    assert not valid[0, 15] and not valid[1, 15] and not valid[2, 12]
    assert np.isfinite(points).all()


def test_pair_selection_covers_every_view_and_rejects_zero_baseline():
    poses = np.tile(np.eye(4), (4, 1, 1)); poses[:, 0, 3] = [0, .1, .4, .5]
    assert select_pairs(poses) == [(0, 1), (2, 3)]
    with pytest.raises(ValueError, match='baseline'):
        select_pairs(np.tile(np.eye(4), (2, 1, 1)))
