"""Array proposals preserve the pre-optimization scorer and sampling policy."""

import hashlib
import json
from collections.abc import Mapping

import numpy as np
import pytest

from stacking_common.actions.composite_sampling import _CandidateRows
from stacking_common.actions.target_regions import CachedAnchorContainment, TargetRegions, anchor_components
from stacking_common.domain import CompositeTargetGeometry, TargetComponent
from test_composite_target_sampling import box, inventory, pose, sampler, state, target


def proposal_snapshot(kind, sampler_class=None, decimals=12):
    """Exercise complete candidate banks, repeated draws, CEM scores and RNG.

    Digests below were captured from commit 2053973 before array conversion.
    Round only serialization noise; direct old/new comparisons also use None.
    """
    geometry = {
        "separated": lambda: target(),
        "overlap": lambda: CompositeTargetGeometry((
            TargetComponent("north", box(2., 4.), pose(-1.)),
            TargetComponent("east", box(2., 4.), pose(0., -1., yaw=np.pi / 2)))),
        "fallback": lambda: target(100.),
    }[kind]()
    inv = inventory(geometry, [box(.4, .3, .2)] * 3)
    results = []
    for elite in (0., .25, 1.):
        sample = sampler(inv)
        sample.settings["score"]["elite_fraction"] = elite
        sample.settings["boltzmann_temperature"] = .3
        sample.settings["boltzmann_min_probability"] = .01
        if sampler_class is not None:
            sample._composite = sampler_class(sample)
        first = geometry.components[0].pose[:2]
        for poses, relaxed in (([], False), ([pose(*first, z=.1)], False),
                               ([pose(*first, z=.1)], True)):
            current = state(inv, poses)
            sample.relax_corner_seed = relaxed
            sample.relax_floor_fill = relaxed
            score_map = sample.score_map(current, None, 1)
            regions = {key: {"scores": value["scores"], "mask": value["candidate_mask"],
                             "source": value["grid_source"]}
                       for key, value in score_map["regions"].items()}
            draws = []
            for priority in (None, "lower_floor_first"):
                proposals = sample.sample(inv, current, None, 1, 7, priority)
                draws.append([(p.pose, p.score, p.diagnostics) for p in proposals])
            probes = [pose(*first), pose(0., 0.), pose(200., 200.)]
            results.append((list(score_map["candidates"]), regions, draws,
                            [sample.score_at_object_pose(None, p, 1) for p in probes],
                            sample.rng.get_state()))

    def serial(value):
        if isinstance(value, np.ndarray):
            return serial(value.tolist())
        if isinstance(value, (tuple, list)):
            return [serial(item) for item in value]
        if isinstance(value, Mapping):
            return {str(key): serial(item) for key, item in value.items()}
        if isinstance(value, (float, np.floating)) and decimals is not None:
            return round(float(value), decimals)
        return value

    return json.dumps(serial(results), sort_keys=True).encode()


@pytest.mark.parametrize("kind,expected", [
    ("separated", "04ce804b49998d820a539a933ecc95c40e78b8c93f6ce78fff71d21cb3f06dec"),
    ("overlap", "bb97db27f5d3f215a3d622fb7aa96bc24dc688f26b5d65b67cb893ba41c306ba"),
    ("fallback", "2e7fe74ef8cb30359d2c1d0e65394ffc3d83db856e19e62b341ec1f22344700a"),
])
def test_proposal_banks_and_draws_match_pre_array_baseline(kind, expected):
    assert hashlib.sha256(proposal_snapshot(kind)).hexdigest() == expected


@pytest.mark.parametrize("scale,margin", [(1., 0.), (.9, .01), (.6, -.01)])
def test_cached_containment_matches_uncached_with_changed_heights_and_vertices(scale, margin):
    region = TargetRegions(target()).regions[0]
    cached = CachedAnchorContainment(region)
    rng = np.random.RandomState(12)
    points = box(.4, .3, .2).points.copy()
    xy = rng.uniform([-5., -2.], [-1., 2.], (100, 2))
    for z in (0., .1, 2.9):
        expected = anchor_components(region, xy, points, z, inset_scale=scale, margin=margin)
        np.testing.assert_array_equal(cached(xy, points, z, inset_scale=scale, margin=margin), expected)
    assert cached.offsets.cache_info().misses == 1
    assert cached.offsets.cache_info().hits == 2
    points[:, 0] *= 2.
    np.testing.assert_array_equal(
        cached(xy, points, .1, inset_scale=scale, margin=margin),
        anchor_components(region, xy, points, .1, inset_scale=scale, margin=margin))
    assert cached.offsets.cache_info().misses == 2
    for changed_scale, changed_margin in ((scale + .1, margin), (scale, margin + .1)):
        np.testing.assert_array_equal(
            cached(xy, points, .1, inset_scale=changed_scale, margin=changed_margin),
            anchor_components(region, xy, points, .1,
                              inset_scale=changed_scale, margin=changed_margin))
    assert cached.offsets.cache_info().misses == 4
    assert cached.offsets.cache_info().maxsize == 1024
    assert not any(array.flags.writeable for planes in cached.planes for array in planes)


def test_containment_cache_evicts_old_offsets_and_handles_empty_queries():
    region = TargetRegions(target()).regions[0]
    cached = CachedAnchorContainment(region)
    points = box(.4, .3, .2).points
    for index in range(1030):
        result = cached([], points, 0., margin=index / 10000.)
        assert result.shape == (0, 1)
    assert cached.offsets.cache_info().currsize == 1024
    np.testing.assert_array_equal(cached([[-3., 0.]], points, .1),
                                  anchor_components(region, [[-3., 0.]], points, .1))


def test_sampling_materializes_only_selected_rows_and_keeps_columns_immutable(monkeypatch):
    inv = inventory()
    sample = sampler(inv)
    current = state(inv)
    rows = []
    original = _CandidateRows.__getitem__

    def counted(self, index):
        rows.append(index)
        return original(self, index)

    monkeypatch.setattr(_CandidateRows, "__getitem__", counted)
    score_map = sample.score_map(current, None, 0)
    assert rows == []
    assert len(score_map["candidates"]) > 100
    proposals = sample.sample(inv, current, None, 0, 3, None)
    assert len(proposals) == len(rows) == 3
    for region_map in score_map["regions"].values():
        assert all(not column.flags.writeable for column in region_map["columns"].values())
    assert sample.score_map(current, None, 0) is score_map
    # Diagnostic copies cannot mutate cached coordinates.
    first = score_map["candidates"][0]
    first["xy"][:] = 1e6
    assert not np.any(score_map["candidates"][0]["xy"] == 1e6)
