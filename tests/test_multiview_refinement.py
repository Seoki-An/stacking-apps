import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from stacking_common.multiview.refinement import refine_cameras, project_normalized, PosePrior


def problem():
    rng = np.random.default_rng(19)
    points = rng.uniform([-.1, -.08, .35], [.1, .08, .5], (24, 3))
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[:, :3, 3] = [[0, 0, 0], [-.04, 0, 0], [.04, 0, 0], [0, .04, 0]]
    views, ids = np.meshgrid(np.arange(4), np.arange(len(points)))
    views, ids = views.ravel(), ids.ravel()
    xy, _ = project_normalized(points, poses, views, ids)
    return points, poses, views, ids, xy


def test_exact_geometry_is_preserved_and_anchor_fixed():
    points, poses, views, ids, xy = problem()
    r = refine_cameras(poses, points, views, ids, xy, [900, 900])
    assert r['success']
    np.testing.assert_allclose(r['points'], points, atol=1e-9)
    np.testing.assert_allclose(r['poses'], poses, atol=1e-9)


def test_small_rotation_error_improves_untrained_points():
    points, truth, views, ids, xy = problem()
    poses = truth.copy()
    poses[1, :3, :3] = Rotation.from_euler('y', .12, degrees=True).as_matrix()
    training = ids < 18
    r = refine_cameras(poses, points[:18], views[training], ids[training], xy[training], [900, 900])
    assert r['success']
    assert np.median(r['final_pixel_errors']) < .15
    np.testing.assert_array_equal(r['poses'][0], poses[0])
    val = ~training
    old, _ = project_normalized(points, poses, views[val], ids[val])
    new, _ = project_normalized(points, r['poses'], views[val], ids[val])
    assert np.mean(np.linalg.norm(new-xy[val], axis=1)) < np.mean(np.linalg.norm(old-xy[val], axis=1))
    assert np.max(np.abs(r['corrections'][:, 3:])) <= .003
    assert np.max(np.abs(r['corrections'][:, :3])) <= np.radians(.6)


def test_common_metric_bias_cannot_be_recovered_from_images():
    points, poses, views, ids, xy = problem()
    offset = np.array([.003, .002, -.007])
    poses[:, :3, 3] += offset
    r = refine_cameras(poses, points+offset, views, ids, xy, [900, 900])
    np.testing.assert_allclose(r['points'], points+offset, atol=1e-9)
    np.testing.assert_allclose(r['final_pixel_errors'], 0, atol=1e-9)


def test_bad_observation_support_fails():
    points, poses, views, ids, xy = problem()
    with pytest.raises(ValueError, match='three observations'):
        keep = views < 2
        refine_cameras(poses, points, views[keep], ids[keep], xy[keep], [900, 900])
    with pytest.raises(ValueError, match='positive'):
        PosePrior(translation_sigma_m=0)


def test_plane_alignment_is_rigid_and_maps_support_without_changing_shape():
    from stacking_common.multiview.refinement import align_support_plane
    x, y = np.meshgrid(np.linspace(-.2, .2, 8), np.linspace(.4, .8, 8))
    p = np.column_stack((x.ravel(), y.ravel(), .11+.01*x.ravel()))
    target = np.array([-.02, .01, .12])
    result = align_support_plane(p, target)
    t = result['transform']; aligned = p@t[:3, :3].T+t[:3, 3]
    np.testing.assert_allclose(aligned[:, 2], aligned[:, :2]@target[:2]+target[2], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(aligned[1:]-aligned[0], axis=1), np.linalg.norm(p[1:]-p[0], axis=1), atol=1e-12)
