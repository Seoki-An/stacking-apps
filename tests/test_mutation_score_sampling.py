"""Mutation rotation fidelity and regional elite/random budget regressions."""

from types import SimpleNamespace

import numpy as np
import pytest

from stacking_common.domain import DirectionalTaper, TargetGeometry
from stacking_common.visualization.heightmap import HeightMapRenderer
from test_composite_target_sampling import box, inventory, pose, sampler, state, target


def scene(composite):
    geometry = target() if composite else TargetGeometry(2., 2., 3., np.zeros(3), DirectionalTaper(), 0.)
    inv = inventory(geometry, [box(.4, .4, .5), box(.8, .2, .25)])
    current = state(inv, [pose(-3. if composite else 0.)])
    renderer = HeightMapRenderer(inv, (64, 64), .1)
    observation = SimpleNamespace(scene_height_map=renderer.render(current),
        height_map_xlim=renderer.xlim, height_map_ylim=renderer.ylim)
    return inv, current, observation


@pytest.mark.parametrize("composite", [False, True])
def test_unsampled_rotation_scores_match_fresh_bank_including_normalization(composite):
    inv, current, observation = scene(composite)
    sample = sampler(inv)
    bank = sample.score_map(current, observation, 1)
    candidates = list(bank["candidates"])
    # Exercise both rotations at common XY cells, not the best-rotation display.
    by_xy = {}
    for item in candidates:
        by_xy.setdefault(tuple(item["xy"]), []).append(item)
    pairs = [items for items in by_xy.values()
             if len(items) > 1 and np.ptp([item["score"] for item in items]) > .01]
    assert pairs
    for items in pairs[::max(1, len(pairs) // 12)]:
        for item in items:
            probe = pose(*item["xy"], yaw=sample.config.yaw_angles[item["orientation_index"]])
            expected = (item["score"], item["normalized_score"])
            np.testing.assert_allclose(sample.score_at_object_pose(observation, probe, 1), expected, atol=1e-10)
            probe[3:] *= -1.  # Quaternion sign cannot change the physical score.
            np.testing.assert_allclose(sample.score_at_object_pose(observation, probe, 1), expected, atol=1e-10)
    assert sample.score_map(current, observation, 1) is bank


@pytest.mark.parametrize("composite", [False, True])
def test_off_bank_yaw_uses_actual_rotation_and_original_score_range(composite):
    inv, current, observation = scene(composite)
    original, reference = sampler(inv), sampler(inv, yaw_angles=(.37,))
    for sample in (original, reference):
        # Remove bank-dependent height/fill normalization for an independent
        # one-rotation oracle; support and boundary still depend on rotation.
        sample.settings["score"].update(height=0., fill_area=0.)
    bank = original.score_map(current, observation, 1)
    fresh = reference.score_map(current, observation, 1)
    low = min(item["score"] for item in bank["candidates"])
    span = max(max(item["score"] for item in bank["candidates"]) - low, 1e-12)
    candidates = list(fresh["candidates"])
    assert candidates
    for item in candidates[::max(1, len(candidates) // 20)]:
        expected = item["score"], np.clip((item["score"] - low) / span, 0., 1.)
        np.testing.assert_allclose(original.score_at_object_pose(
            observation, pose(*item["xy"], yaw=.37), 1), expected, atol=1e-10)
    assert original.score_map(current, observation, 1) is bank


@pytest.mark.parametrize("elite,expected_random", [(0., 8), (.25, 6), (1., 0)])
def test_composite_allocations_preserve_random_draws_and_region_balance(elite, expected_random):
    class CountingRandom(np.random.RandomState):
        draws = 0

        def choice(self, *args, **kwargs):
            self.draws += 1
            return super().choice(*args, **kwargs)

    inv = inventory()
    sample = sampler(inv)
    sample.relax_corner_seed = True
    sample.settings["score"]["elite_fraction"] = elite
    sample.rng = CountingRandom(12)
    proposals = sample.sample(inv, state(inv), None, 0, 8, "lower_floor_first")
    assert sample.rng.draws == expected_random
    regions = [p.diagnostics["target_region_id"] for p in proposals]
    assert regions[::2] == [regions[0]] * 4
    assert regions[1::2] == [regions[1]] * 4
    assert regions[0] != regions[1]
    assert len({tuple(p.pose) for p in proposals}) == 8


def test_composite_sampling_preserves_priority_and_exhausts_without_duplicates():
    inv = inventory()
    sample = sampler(inv)
    current = state(inv)
    bank = sample.score_map(current, None, 0)
    # Request beyond capacity so every seed and ordinary candidate is drawn.
    proposals = sample.sample(inv, current, None, 0, len(bank["candidates"]) + 10, "lower_floor_first")
    assert len(proposals) == len(bank["candidates"])
    assert len({tuple(p.pose) for p in proposals}) == len(proposals)
    for region_id, region_map in bank["regions"].items():
        regional = [p for p in proposals if p.diagnostics["target_region_id"] == region_id]
        seeds = int(region_map["columns"]["seed"].sum())
        assert seeds > 0
        assert all(p.diagnostics["sampling_policy"] == "corner_seed" for p in regional[:seeds])
        assert all(p.diagnostics["sampling_policy"] == "scored" for p in regional[seeds:])
