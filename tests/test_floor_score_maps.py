"""Array boundary scoring must preserve the scalar geometric policy."""

from types import SimpleNamespace

import numpy as np
import pytest

from stacking_common.actions import _scored_floor as floor


@pytest.mark.parametrize("taper", [
    (0., 0., 0., 0.),
    (.15, -.2, .3, .05),
    (1.1, .8, 1.2, .9),  # Bounds collapse at high elevations.
])
def test_batched_target_bounds_and_scores_match_scalar_policy(taper):
    wall = SimpleNamespace(
        origin=np.array([.3, -.2, .4]), height=2., width=3., length=4.,
        taper=dict(zip(("width_negative", "width_positive",
                        "length_negative", "length_positive"), taper)),
    )
    static = (wall, wall.origin[:2], np.array([1.5, 2.]), .02, np.array([.1, .2]))
    heights = np.array([[-1., .4, .7], [1.4, 2.4, 5.]])
    actual_lower, actual_upper = floor.target_anchor_bounds(None, None, heights, _static=static)
    expected_lower, expected_upper = [], []
    for z in heights.ravel():
        relative = np.clip(float(z) - wall.origin[2], 0., wall.height)
        lower = wall.origin[:2] - static[2]
        upper = wall.origin[:2] + static[2]
        lower += relative * np.tan([max(taper[0], 0.), max(taper[2], 0.)])
        upper -= relative * np.tan([max(taper[1], 0.), max(taper[3], 0.)])
        lower += static[3] + static[4]
        upper -= static[3] + static[4]
        expected_lower.append(lower)
        expected_upper.append(upper)
    expected_lower = np.array(expected_lower).reshape(2, 3, 2)
    expected_upper = np.array(expected_upper).reshape(2, 3, 2)
    np.testing.assert_array_equal(actual_lower, expected_lower)
    np.testing.assert_array_equal(actual_upper, expected_upper)

    # Centers, edges, outside points, and invalid bounds.
    for xy in (actual_lower, actual_upper, (actual_lower + actual_upper) / 2,
               actual_lower - .01, actual_upper + .01):
        expected = []
        for point, lower, upper in zip(xy.reshape(-1, 2),
                                     expected_lower.reshape(-1, 2),
                                     expected_upper.reshape(-1, 2)):
            if np.any(upper <= lower) or np.any(point < lower) or np.any(point > upper):
                score = 0.
            else:
                center = .5 * (lower + upper)
                half = .5 * (upper - lower)
                score = float(np.clip(np.max(np.abs((point - center) / half)), 0., 1.))
            expected.append(score)
            assert floor.target_boundary_proximity(point, lower, upper) == score
        with np.errstate(divide="raise", invalid="raise"):
            actual = floor.target_boundary_proximity(xy, actual_lower, actual_upper)
        np.testing.assert_array_equal(actual, np.array(expected).reshape(2, 3))


def test_boundary_scores_support_empty_batches_and_shared_bounds():
    lower, upper = np.array([-1., -2.]), np.array([1., 2.])
    assert floor.target_boundary_proximity(np.empty((0, 2)), lower, upper).shape == (0,)
    np.testing.assert_array_equal(
        floor.target_boundary_proximity(np.array([[0., 0.], [1., 0.], [0., -1.],
                                                  [2., 0.]]), lower, upper),
        [0., 1., .5, 0.],
    )


@pytest.mark.parametrize("all_invalid", [False, True])
def test_score_map_keeps_candidate_order_height_fallback_and_orientation_ties(all_invalid, monkeypatch):
    from stacking_common.actions._support_maps import KernelSupportMaps
    wall = SimpleNamespace(origin=np.zeros(3), width=2., length=2., height=2.,
        taper=dict.fromkeys(("width_negative", "width_positive",
                             "length_negative", "length_positive"), 0.))
    inventory = SimpleNamespace(
        stones=[], target_wall=wall, xlim=(-.5, .5), ylim=(-.5, .5),
        cfg=SimpleNamespace(action=SimpleNamespace(planar={
            "score_map_size": 2,
            "floor_fill": {"boundary_inset_radius_scale": 0.},
            "score": {"height": 0., "support_likelihood": 1., "open_area": 0.,
                      "fill_area": 0., "frontier": 0., "target_boundary": 0.,
                      "excavator_distance": 0.},
        })),
        support_settings=SimpleNamespace(ground_z=0., contact_gap_tolerance=.01),
    )
    support = np.array([[[np.nan, 1.], [np.inf, 1.]], [[1., 1.], [1., 1.]]])
    if all_invalid:
        support[:] = np.nan
    contact = np.zeros_like(support)
    contact[0, 0, 1] = np.nan
    contact[1, 1, 0] = np.inf
    original_contact = contact.copy()
    monkeypatch.setattr(floor, "support_likelihood_maps_by_kernel",
        lambda *args: KernelSupportMaps(support, contact, np.ones_like(support)))
    result = floor.score_xy_debug_map(
        inventory, SimpleNamespace(stone_seq=[]), None,
        scene_height_map=np.zeros((2, 2)), support_kernels=[],
        active_floor={"occupied": np.zeros((2, 2), dtype=bool)},
    )
    np.testing.assert_array_equal(contact, original_contact)
    if all_invalid:
        assert result["candidates"] == []
        assert not result["candidate_mask"].any()
        return
    candidates = result["candidates"]
    assert [item["orientation_index"] for item in candidates] == [0, 0, 1, 1, 1, 1]
    np.testing.assert_array_equal([item["xy"] for item in candidates],
        [[.5, -.5], [.5, .5], [-.5, -.5], [.5, -.5], [-.5, .5], [.5, .5]])
    assert all(item["height"] == 0. for item in candidates)
    assert all(item["normalized_score"] == 0. for item in candidates)
    np.testing.assert_array_equal(result["orientation_index"], [[1, 0], [1, 0]])


def test_shifted_samples_match_per_offset_slices():
    from stacking_common.actions._support_maps import _shifted_samples
    values = np.arange(12, dtype=float).reshape(3, 4)
    values[1, 2] = np.nan
    offsets = [(0, 0), (1, -2), (-1, 3), (2, 2), (-5, 0), (0, 7)]  # Last two leave the grid.
    actual = _shifted_samples(values, [y for y, _ in offsets], [x for _, x in offsets], -np.inf)
    for (dy, dx), grid in zip(offsets, actual):
        expected = np.full(values.shape, -np.inf)
        for y in range(3):
            for x in range(4):
                if 0 <= y + dy < 3 and 0 <= x + dx < 4:
                    expected[y, x] = values[y + dy, x + dx]
        np.testing.assert_array_equal(grid, expected)
