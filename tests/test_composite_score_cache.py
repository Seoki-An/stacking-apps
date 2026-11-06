"""Revisited MCTS states reuse maps without changing proposals or CEM scoring."""

from dataclasses import replace

import numpy as np
import pytest

from test_composite_target_sampling import inventory, pose, sampler, state, target


@pytest.mark.parametrize("separation", [3., 100.])
def test_revisit_matches_uncached_draws_and_mutation_scores(separation):
    inv = inventory(target(separation))
    cached, uncached = sampler(inv), sampler(inv)
    uncached._composite._cache_max_states = 0
    states = [state(inv), state(inv, [pose(-separation)]),
              state(inv, [pose(separation, z=.1)])]
    first = cached.score_map(states[0], None, 1)
    original_surface = next(iter(cached._composite._surfaces.values())).copy()
    for index in (0, 1, 2, 0, 2, 1, 0):
        current = states[index]
        a, b = (item.sample(inv, current, None, 1, 5, "lower_floor_first")
                for item in (cached, uncached))
        assert len(a) == len(b)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left.pose, right.pose)
            assert left.score == right.score
            assert left.diagnostics == right.diagnostics
        for probe in (pose(-separation, yaw=.1), pose(separation), pose(0.)):
            assert cached.score_at_object_pose(None, probe, 1) == uncached.score_at_object_pose(None, probe, 1)
        assert cached._composite._state is current
    assert cached.score_map(states[0], None, 1) is first
    np.testing.assert_array_equal(next(iter(cached._composite._surfaces.values())), original_surface)
    np.testing.assert_array_equal(cached.rng.get_state()[1], uncached.rng.get_state()[1])


def test_revisit_does_not_restore_unrequested_latest_maps_or_mutable_aliases():
    inv = inventory()
    sample = sampler(inv)
    a, b = state(inv), state(inv, [pose(-3.)])
    old = sample.score_map(a, None, 0)
    sample.score_map(a, None, 1)
    sample.score_map(b, None, 2)
    assert sample.score_map(a, None, 0) is old
    assert set(sample._composite._latest_maps) == {0}
    assert sample.score_at_object_pose(None, pose(-3.), 1) == (0., 0.)
    with pytest.raises(TypeError):
        old["normalization"]["score_min"] = 999.
    region = next(iter(old["regions"].values()))
    with pytest.raises(ValueError):
        region["scores"][:] = 999.
    with pytest.raises(TypeError):
        old["candidates"].batches[0][2]["score"] = np.zeros(1)


@pytest.mark.parametrize("change", ["pose", "order", "score", "config", "seed",
                                     "connectivity", "length", "floor", "rotation"])
def test_context_and_rotation_changes_do_not_return_stale_maps(change):
    inv = inventory()
    sample = sampler(inv)
    current = state(inv, [pose(-3.), pose(3.)])
    old = sample.score_map(current, None, 2)
    sample.score_map(state(inv), None, 2)
    if change == "pose":
        current.object_poses[inv[0].object_id][0] += .2
    elif change == "order":
        current.placed_object_indices.reverse()
    elif change == "score":
        sample.settings["score"]["height"] = -6.
    elif change == "config":
        sample.config = replace(sample.config, grid_shape=(16, 20))
    elif change == "seed":
        sample.relax_corner_seed = True
    elif change == "connectivity":
        sample.connectivity_enabled = False
    elif change == "length":
        sample.connectivity_length *= 2.
    elif change == "floor":
        sample.floor_policy.fill_ratio = .5
    else:
        sample.quaternions = lambda *args: (pose(yaw=.2)[3:],)
    assert sample.score_map(current, None, 2) is not old


def test_lru_eviction_byte_budget_and_oversized_state():
    inv = inventory()
    sample = sampler(inv)
    cache = sample._composite
    cache._cache_max_states = 1
    a, b, c = (state(inv, [pose(x)]) for x in (-3., -3.1, -3.2))
    first = sample.score_map(a, None, 1)
    sample.score_map(b, None, 1)
    assert len(cache._states) == 1
    assert 0 < cache._states_bytes <= cache._cache_max_bytes
    # Budget too small for any complete state: current results still work.
    cache._cache_max_bytes = 1
    sample.score_map(c, None, 1)
    assert len(cache._states) == cache._states_bytes == 0
    assert sample.score_map(a, None, 1) is not first
    assert len(cache._states) == cache._states_bytes == 0


def test_recently_restored_state_survives_lru_eviction():
    inv = inventory()
    sample = sampler(inv)
    sample._composite._cache_max_states = 1
    a, b, c = (state(inv, [pose(x)]) for x in (-3., -3.1, -3.2))
    first = sample.score_map(a, None, 1)
    second = sample.score_map(b, None, 1)
    assert sample.score_map(a, None, 1) is first
    sample.score_map(c, None, 1)
    assert sample.score_map(a, None, 1) is first
    assert sample.score_map(b, None, 1) is not second


def test_byte_limit_evicts_before_state_count_limit():
    inv = inventory()
    sample = sampler(inv)
    cache = sample._composite
    states = [state(inv, [pose(-3. + i * .01)]) for i in range(8)]
    sample.score_map(states[0], None, 1)
    sample.score_map(states[1], None, 1)
    # Enough for each individual state, but not all eight.
    cache._cache_max_bytes = 2 * cache._states_bytes
    for current in states[2:]:
        sample.score_map(current, None, 1)
        assert cache._states_bytes <= cache._cache_max_bytes
    assert 0 < len(cache._states) < len(states) - 1
