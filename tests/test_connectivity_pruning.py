"""Conservative score-map pruning preserves exact polygon admission."""
import numpy as np
import pytest
from scipy.spatial import ConvexHull

from stacking_common.actions import connectivity, composite_sampling
from test_previous_action_connectivity import setup
from test_composite_target_sampling import sampler


@pytest.mark.parametrize('scale', [.001, .1, 1., 100.])
def test_random_rotated_polygons_match_exact_admission(scale):
    rng = np.random.default_rng(947)
    for _ in range(12):
        a, b = [rng.normal(size=(16, 2)) * scale for _ in range(2)]
        a, b = a[ConvexHull(a).vertices], b[ConvexHull(b).vertices]
        translations = rng.uniform(-8., 8., (200, 2)) * scale
        distances = connectivity.footprint_distances(a, b, translations)
        for gap in (1e-9, .1 * scale, scale):
            np.testing.assert_array_equal(
                connectivity.footprints_within_gap(a, b, translations, gap),
                distances <= gap)


def test_threshold_touching_and_empty_batches():
    square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    gap = .1 + 1e-9
    shifts = [[0., 0.], [1., 0.], [1. + gap - 1e-10, 0.],
              [1. + gap, 0.], [1. + gap + 1e-10, 0.],
              [1. + gap / np.sqrt(2), 1. + gap / np.sqrt(2)]]
    np.testing.assert_array_equal(
        connectivity.footprints_within_gap(square, square, shifts, gap),
        connectivity.footprint_distances(square, square, shifts) <= gap)
    result = connectivity.footprints_within_gap(square, square, [], gap)
    assert result.shape == (0,) and result.dtype == bool


def test_aabb_only_prunes_and_never_accepts_overlapping_boxes(monkeypatch):
    diamond = np.array([[0., -1.], [1., 0.], [0., 1.], [-1., 0.]])
    original = connectivity.footprint_distances
    checked = []
    def record(a, b, translations):
        checked.extend(translations.tolist())
        return original(a, b, translations)
    monkeypatch.setattr(connectivity, 'footprint_distances', record)
    result = connectivity.footprints_within_gap(
        diamond, diamond, [[0., 0.], [1.5, 1.5], [20., 20.]], .1)
    np.testing.assert_array_equal(result, [True, False, False])
    assert checked == [[0., 0.], [1.5, 1.5]]


@pytest.mark.parametrize('composite', [False, True])
def test_sampled_poses_and_scores_match_unpruned_policy(composite, monkeypatch):
    inv, state, obs = setup(composite)
    optimized = sampler(inv)
    optimized.settings['connectivity'] = 'previous_action'
    actual = optimized.sample(inv, state, obs, 2, 24, 'lower_floor_first')
    def exact(a, b, translations, gap):
        return connectivity.footprint_distances(a, b, translations) <= gap
    monkeypatch.setattr(connectivity, 'footprints_within_gap', exact)
    monkeypatch.setattr(composite_sampling, 'footprints_within_gap', exact)
    baseline = sampler(inv)
    baseline.settings['connectivity'] = 'previous_action'
    expected = baseline.sample(inv, state, obs, 2, 24, 'lower_floor_first')
    assert actual and len(actual) == len(expected)
    np.testing.assert_array_equal([p.pose for p in actual], [p.pose for p in expected])
    np.testing.assert_array_equal([p.score for p in actual], [p.score for p in expected])
