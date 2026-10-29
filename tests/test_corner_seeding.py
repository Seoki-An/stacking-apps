"""Geometry-based first-stone seeds, state scope, and ordinary sampling fallback."""
from dataclasses import replace
from types import SimpleNamespace as NS
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stacking_common.actions import _scored_floor as floor
from stacking_common.search import ActionGenerationRequest
from stacking_excavator import ExcavatorApplication
from stacking_tabletop import TabletopApplication
from test_action_generation import _inventory, _state, _Initializer, _PoseSolver

ROOT = Path(__file__).resolve().parents[1]


def grid_candidates(bounds, orientation=0):
    xs = np.linspace(bounds[0][0], bounds[1][0], 11)
    ys = np.linspace(bounds[0][1], bounds[1][1], 11)
    return [dict(xy=np.array([x, y]), score=0., orientation_index=orientation)
            for y in ys for x in xs], np.array([xs[1] - xs[0], ys[1] - ys[0]])


def test_four_corner_neighborhoods_exclude_edges_and_center():
    bounds = np.array([[[-1., -2.], [1., 2.]]])
    candidates, spacing = grid_candidates(bounds[0])
    pool = floor.corner_seed_candidates(candidates, bounds, spacing)
    assert len(pool) == 16
    assert {item['seed_corner'] for item in pool} == {0, 1, 2, 3}
    assert all(abs(item['xy'][0]) >= .8 - 1e-9 and abs(item['xy'][1]) >= 1.6 - 1e-9 for item in pool)
    assert all('seed_corner' not in item for item in candidates)
    view = NS(cfg=NS(action=NS(planar={'score': {'elite_fraction': .25}})))
    selected = floor.sample_corner_candidates(view, pool, 4, rng=np.random.RandomState(7))
    assert len(selected) == len({id(item) for item in selected}) == 4
    assert {item['seed_corner'] for item in selected} == {0, 1, 2, 3}
    assert len(floor.sample_corner_candidates(view, pool, 32, rng=np.random.RandomState(7))) == 16


def test_small_requests_do_not_always_pick_same_corner():
    bounds = np.array([[[-1., -1.], [1., 1.]]])
    candidates, spacing = grid_candidates(bounds[0])
    pool = floor.corner_seed_candidates(candidates, bounds, spacing)
    view = NS(cfg=NS(action=NS(planar={'score': {'elite_fraction': .25}})))
    rng = np.random.RandomState(0)
    assert {floor.sample_corner_candidates(view, pool, 1, rng=rng)[0]['seed_corner']
            for _ in range(32)} == {0, 1, 2, 3}


def test_narrow_or_invalid_bounds_do_not_duplicate_or_invent_candidates():
    item = dict(xy=np.array([0., 0.]), score=0., orientation_index=0)
    assert len(floor.corner_seed_candidates([item], [[[-.01, -.01], [.01, .01]]], [.1, .1])) == 1
    assert floor.corner_seed_candidates([item], [[[1., 1.], [-1., -1.]]], [.1, .1]) == []
    assert floor.corner_seed_candidates([], [], [.1, .1]) == []


@pytest.fixture(params=[ExcavatorApplication, TabletopApplication])
def sampler_setup(request):
    app = request.param.from_repository(ROOT)
    inventory = _inventory()
    orientations = NS(quaternion=lambda *args: Rotation.from_euler('z', 30, degrees=True).as_quat())
    generator = app.build_action_generator(inventory, _PoseSolver(), orientation_provider=orientations)
    observation = NS(scene_height_map=np.zeros((32, 32)),
        height_map_xlim=np.array([-.15, .15]), height_map_ylim=np.array([-.15, .15]))
    try:
        yield generator, inventory, observation
    finally:
        generator.close()


def test_empty_scene_uses_full_rotated_footprint_corners(sampler_setup):
    generator, inventory, observation = sampler_setup
    sampler = generator.planar_sampler
    state = _state(inventory)
    proposals = sampler.sample(inventory, state, observation, 0, 16, None)
    assert len(proposals) >= 4
    assert {p.diagnostics['seed_corner'] for p in proposals[:4]} == {0, 1, 2, 3}
    points, _ = inventory[0].mesh_arrays()
    spacing = .3 / 31
    for p in proposals:
        assert p.diagnostics['sampling_policy'] == 'corner_seed'
        footprint = Rotation.from_quat(p.pose[3:]).apply(points)[:, :2] + p.pose[:2]
        assert np.all(footprint.min(axis=0) >= -.15 - 1e-10)
        assert np.all(footprint.max(axis=0) <= .15 + 1e-10)
        gaps = np.minimum(footprint.min(axis=0) + .15, .15 - footprint.max(axis=0))
        assert np.all(gaps <= spacing + 1e-10)
    again = sampler.score_map(state, observation, 0)
    assert again['corner_candidates']


def test_corner_fallback_is_scoped_and_leaves_floor_filters_unchanged(sampler_setup):
    generator, inventory, observation = sampler_setup
    sampler = generator.planar_sampler
    state = _state(inventory)
    filters = generator.filters
    with pytest.raises(RuntimeError):
        with generator.corner_seed_fallback() as enabled:
            assert enabled
            assert generator.filters is filters
            proposals = sampler.sample(inventory, state, observation, 0, 32, None)
            assert proposals and all(p.diagnostics['sampling_policy'] == 'scored' for p in proposals)
            assert all(p.diagnostics['corner_seed_fallback'] for p in proposals)
            raise RuntimeError('exercise restoration')
    assert not sampler.relax_corner_seed
    assert all(p.diagnostics['sampling_policy'] == 'corner_seed' for p in
               sampler.sample(inventory, state, observation, 0, 4, None))


def test_placed_state_never_uses_corner_seeds_even_with_empty_height_map(sampler_setup):
    generator, inventory, observation = sampler_setup
    state = _state(inventory, placed=(0,))
    proposals = generator.planar_sampler.sample(inventory, state, observation, 1, 8, None)
    assert proposals
    assert all(p.diagnostics['sampling_policy'] == 'scored' for p in proposals)
    assert all('seed_corner' not in p.diagnostics and 'corner_seed_fallback' not in p.diagnostics for p in proposals)


def test_no_feasible_corner_cells_falls_back_immediately(sampler_setup, monkeypatch):
    generator, inventory, observation = sampler_setup
    sampler = generator.planar_sampler
    state = _state(inventory)
    scores = dict(sampler.score_map(state, observation, 0), corner_candidates=[])
    monkeypatch.setattr(sampler, 'score_map', lambda *a: scores)
    proposals = sampler.sample(inventory, state, observation, 0, 4, None)
    assert proposals and all(p.diagnostics['sampling_policy'] == 'scored' for p in proposals)
    assert all(p.diagnostics['corner_seed_fallback'] for p in proposals)


def test_generator_preserves_four_corner_draws_under_object_budget(sampler_setup, monkeypatch):
    generator, inventory, observation = sampler_setup
    # Isolate budget allocation from physics, which is checked in the MCTS tests.
    generator.initializer = _Initializer()
    generator.filters = ()
    generator.config = replace(generator.config, banned_object_ids=(20,))
    draws = []
    def build(jobs):
        draws.extend(jobs)
        return [(None, 'test_rejection') for _ in jobs]
    monkeypatch.setattr(generator, '_build_jobs', build)
    generator.generate(_state(inventory), observation, ActionGenerationRequest(max_candidates=4))
    assert {job[-1].diagnostics['seed_corner'] for job in draws} == {0, 1, 2, 3}
