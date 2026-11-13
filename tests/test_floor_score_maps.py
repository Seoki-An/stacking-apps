"""Array boundary scoring must preserve the scalar geometric policy."""

from types import SimpleNamespace

import numpy as np
import pytest

from stacking_common.actions import _scored_floor as floor


@pytest.mark.parametrize("count", [1, 4, 8, 20])
def test_selection_uses_scores_without_gap_or_spatial_groups(count):
    inventory = SimpleNamespace(cfg=SimpleNamespace(action=SimpleNamespace(planar={
        "boltzmann_temperature": .3, "score": {"elite_fraction": .25},
    })))
    plain = [dict(score=float(10 - i)) for i in range(10)]
    annotated = [dict(item, height=float(i), xy=np.array([(-1.)**i, i - 5.]),
                      gap_region=(i, 0, 0), gap_component_size=100 * i)
                 for i, item in enumerate(plain)]
    selections = [floor.sample_scored_candidates(
        inventory, pool, count, True, rng=np.random.default_rng(17))
        for pool in (plain, annotated)]
    assert [p["score"] for p in selections[0]] == [p["score"] for p in selections[1]]
    chosen = selections[0]
    assert len(chosen) == min(count, len(plain))
    assert len({id(p) for p in chosen}) == len(chosen)
    elite_count = min(int(np.ceil(count * .25)), len(plain))
    assert [p["score"] for p in chosen[:elite_count]] == [p["score"] for p in plain[:elite_count]]


@pytest.mark.parametrize("elite_fraction", [0., .25])
def test_small_draw_follows_scores_without_high_y_priority(elite_fraction):
    inventory = SimpleNamespace(
        target_wall=SimpleNamespace(origin=np.zeros(3)),
        cfg=SimpleNamespace(action=SimpleNamespace(planar={
            "boltzmann_temperature": .3,
            "score": {"elite_fraction": elite_fraction},
        })),
    )
    candidates = [dict(xy=np.array([-1., y]), height=0., score=score)
                  for y, score in [(-3., 1000.), (0., 0.), (3., -1000.)]]
    selected = floor.sample_scored_candidates(
        inventory, candidates, 1, True, rng=np.random.default_rng(0))
    assert len(selected) == 1
    assert selected[0] is candidates[0]


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
    static = (wall, wall.origin[:2], np.array([1.5, 2.]), .02, np.array([[-.1, -.2], [.1, .2]]))
    heights = np.array([[-1., .4, .7], [1.4, 2.4, 5.]])
    actual_lower, actual_upper = floor.target_anchor_bounds(None, None, heights, _static=static)
    expected_lower, expected_upper = [], []
    for z in heights.ravel():
        relative = np.clip(float(z) - wall.origin[2], 0., wall.height)
        lower = wall.origin[:2] - static[2]
        upper = wall.origin[:2] + static[2]
        lower += relative * np.tan([max(taper[0], 0.), max(taper[2], 0.)])
        upper -= relative * np.tan([max(taper[1], 0.), max(taper[3], 0.)])
        lower += static[3] - static[4][0]
        upper -= static[3] + static[4][1]
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
            "floor": {"boundary_inset_radius_scale": 0.},
            "score": {"height": 0., "support_likelihood": 1.,
                      "fill_area": 0., "target_boundary": 0.,
                      "planar_bias": 0.},
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
        lambda surface, kernels, tolerance, *, cells: KernelSupportMaps(*(values[kernels, :, :][:, cells[0], cells[1]] for values in (
            support, contact, np.ones_like(support)))))
    result = floor.score_xy_debug_map(
        inventory, SimpleNamespace(stone_seq=[]), None,
        scene_height_map=np.zeros((2, 2)), support_kernels=[0, 1],
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


def test_covered_cells_are_admitted_but_empty_connected_cells_score_higher():
    from stacking_common.actions._support_maps import BottomHeightKernel
    wall = SimpleNamespace(origin=np.zeros(3), width=2., length=2., height=2.,
        taper=dict.fromkeys(("width_negative", "width_positive",
                             "length_negative", "length_positive"), 0.))
    inventory = SimpleNamespace(
        stones=[], target_wall=wall, xlim=(-.5, .5), ylim=(-.5, .5),
        cfg=SimpleNamespace(action=SimpleNamespace(planar={
            "score_map_size": 3, "floor": {"footprint_scale": 1.},
            "score": {"height": 0., "support_likelihood": 0.,
                      "fill_area": 2., "target_boundary": 0., "planar_bias": 0.},
        })),
        support_settings=SimpleNamespace(ground_z=0., contact_gap_tolerance=.01),
    )
    xx, yy = np.meshgrid([-.5, 0., .5], [-.5, 0., .5])
    occupied = np.zeros((3, 3), dtype=bool)
    occupied[:, 0] = True
    kwargs = dict(scene_height_map=np.zeros((3, 3)),
                  active_floor=dict(occupied=occupied, xx=xx, yy=yy),
                  support_kernels=[BottomHeightKernel(np.array([0]), np.array([0]), np.array([0.]))],
                  footprint_bounds=np.array([[[-.1, -.1], [.1, .1]]]))
    result = floor.score_xy_debug_map(inventory, None, None, **kwargs)
    assert result['candidate_mask'][:, 0].all()  # Covered cells remain eligible.
    assert result['candidate_mask'][:, 1].all()
    assert not result['candidate_mask'][:, 2].any()  # Disconnected cells do not.
    np.testing.assert_array_equal(result['fill_area'][:, 0], 0.)
    np.testing.assert_array_equal(result['fill_area'][:, 1], 1.)
    assert np.all(result['scores'][:, 1] > result['scores'][:, 0])


@pytest.mark.parametrize("shape", [(3, 4), (5, 7), (16, 2)])
def test_resized_height_maps_take_finite_block_maxima(shape):
    from stacking_common.actions._support_maps import resize_height_map_max
    source = np.random.default_rng(1).normal(size=(11, 9))
    source[0, :] = np.nan
    source[4, 3] = np.inf
    source[:, -1] = -np.inf
    dst_h, dst_w = shape
    y_edges = np.linspace(0., 11., dst_h + 1)
    x_edges = np.linspace(0., 9., dst_w + 1)
    expected = np.zeros(shape)
    for row in range(dst_h):
        y0 = min(int(np.floor(y_edges[row])), 10)
        y1 = max(min(int(np.ceil(y_edges[row + 1])), 11), y0 + 1)
        for column in range(dst_w):
            x0 = min(int(np.floor(x_edges[column])), 8)
            x1 = max(min(int(np.ceil(x_edges[column + 1])), 9), x0 + 1)
            block = source[y0:y1, x0:x1]
            finite = block[np.isfinite(block)]
            expected[row, column] = finite.max() if finite.size else 0.
    np.testing.assert_array_equal(resize_height_map_max(source, shape), expected)


def test_support_at_cells_matches_full_maps():
    from stacking_common.actions._support_maps import (
        BottomHeightKernel, support_likelihood_maps_by_kernel)
    surface = np.random.default_rng(2).uniform(0., .3, size=(9, 8))
    surface[2, 3] = np.nan
    kernels = [
        BottomHeightKernel(np.array([0, 1, -1, 2]), np.array([0, -1, 1, 3]), np.array([0., .01, .02, 0.])),
        BottomHeightKernel(np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0)),
    ]
    rows, columns = np.nonzero(np.ones(surface.shape, dtype=bool))
    full = support_likelihood_maps_by_kernel(surface, kernels, .02)
    cells = support_likelihood_maps_by_kernel(surface, kernels, .02, cells=(rows[::3], columns[::3]))
    for name in ("support", "contact_height", "placement_height"):
        np.testing.assert_array_equal(getattr(cells, name), getattr(full, name)[:, rows[::3], columns[::3]])
