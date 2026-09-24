import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stacking_common.multiview.geometry import triangulate_track, epipolar_distances
from stacking_common.multiview.features import build_tracks


def scene():
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[0, :3, 3] = [-.04, 0, 0]
    poses[1, :3, 3] = [.04, 0, 0]
    poses[2, :3, 3] = [0, .03, 0]
    poses[2, :3, :3] = Rotation.from_euler('x', 3, degrees=True).as_matrix()
    p = np.array([.01, .02, .4])
    camera = np.einsum('nij,j->ni', np.linalg.inv(poses), np.r_[p, 1.])
    return p, poses, camera[:, :2] / camera[:, 2, None]


def test_metric_point_and_frame_convention():
    p, poses, xy = scene()
    result = triangulate_track(xy, poses)
    np.testing.assert_allclose(result['point'], p, atol=1e-12)
    np.testing.assert_allclose(result['residuals'], 0, atol=1e-12)
    assert result['baseline_m'] == pytest.approx(.08)
    assert result['parallax_deg'] > 10
    assert np.all(result['depths'] > 0)
    k = np.diag([900., 900., 1.])
    np.testing.assert_allclose(epipolar_distances(xy[:1], xy[2:], poses[0], poses[2], k), 0, atol=1e-10)


def test_unused_view_error_does_not_move_fitted_point():
    p, poses, xy = scene()
    xy[2, 0] += .01
    result = triangulate_track(xy, poses)
    np.testing.assert_allclose(result['point'], p, atol=1e-12)
    assert np.linalg.norm(result['residuals'][2]) == pytest.approx(.01)
    np.testing.assert_allclose(result['residuals'][:2], 0, atol=1e-12)


def test_rotation_only_cannot_establish_metric_depth():
    _, poses, xy = scene()
    poses[:, :3, 3] = 0
    with pytest.raises(ValueError, match='baseline'):
        triangulate_track(xy, poses)


def test_behind_camera_is_reported_not_silently_accepted():
    _, poses, _ = scene()
    p = np.array([.01, .02, -.4])
    camera = np.einsum('nij,j->ni', np.linalg.inv(poses), np.r_[p, 1.])
    result = triangulate_track(camera[:, :2] / camera[:, 2, None], poses)
    assert np.all(result['depths'] < 0)


def test_conflicting_track_components_are_rejected():
    edges = [((0, 0), (1, 2)), ((1, 2), (2, 3)),
             ((0, 4), (1, 5)), ((1, 5), (0, 6))]
    tracks, conflicts = build_tracks(edges)
    assert tracks == [[(0, 0), (1, 2), (2, 3)]]
    assert conflicts == 1


def test_common_camera_bias_is_invisible_to_reprojection():
    p, poses, xy = scene()
    # A shared calibration offset moves the cloud while keeping every pixel fit.
    poses[:, :3, 3] += [.003, -.002, .004]
    result = triangulate_track(xy, poses)
    np.testing.assert_allclose(result['point'], p + [.003, -.002, .004], atol=1e-12)
    np.testing.assert_allclose(result['residuals'], 0, atol=1e-12)


def test_real_feature_matching_under_known_image_translation():
    cv2 = pytest.importorskip('cv2')
    from stacking_common.multiview.features import extract_features, match_features
    rng = np.random.default_rng(17)
    image = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), .7)
    shifted = cv2.warpAffine(image, np.array([[1., 0, 12], [0, 1., 7]]), (400, 300))
    k = np.array([[500., 0, 200], [0, 500., 150], [0, 0, 1]])
    a = extract_features(image, k, np.zeros(5)); b = extract_features(shifted, k, np.zeros(5))
    matches = np.array(match_features(a['descriptors'], b['descriptors']))
    assert len(matches) > 30
    delta = b['pixels'][matches[:, 1]] - a['pixels'][matches[:, 0]]
    assert np.median(np.linalg.norm(delta-[12, 7], axis=1)) < .1
    assert match_features(None, b['descriptors']) == []


def test_fitted_distortion_round_trip_for_rgb_rays():
    cv2 = pytest.importorskip('cv2')
    k = np.array([[900., 0, 640], [0, 910., 360], [0, 0, 1]])
    distortion = np.array([-.08, .03, .001, -.001, 0.])
    p = np.array([[.1, -.1, .4], [-.1, .06, .5]])
    pixels, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), k, distortion)
    rays = cv2.undistortPoints(pixels, k, distortion).reshape(-1, 2)
    np.testing.assert_allclose(rays, p[:, :2]/p[:, 2, None], atol=1e-8)
