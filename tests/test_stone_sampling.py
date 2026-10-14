"""Volume-biased proposal budgets, independent of XY scores and MCTS rewards."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

from stacking_common.actions.config import PlacementActionGeneratorConfig
from stacking_common.actions.scored import ScoredPlacementActionGenerator
from stacking_common.profiles.action import load_action_generation_profile
from stacking_common.search import ActionGenerationRequest
from stacking_excavator import ExcavatorApplication
from stacking_tabletop import TabletopApplication
from test_action_generation import _inventory, _state, _PoseSolver

ROOT = Path(__file__).resolve().parents[1]


def generator(volumes, exponent=1., cap=32):
    return ScoredPlacementActionGenerator(
        inventory=[NS(volume=v) for v in volumes],
        config=PlacementActionGeneratorConfig(cap, 0., 0., None,
                                             stone_volume_exponent=exponent),
        planar_sampler=NS(seed=17), initializer=None, pose_solver=None,
    )


def test_larger_stones_receive_more_slots_without_excluding_small_stones():
    sampler = generator([1., 3.])
    counts = np.zeros(2, dtype=int)
    for _ in range(4000):
        allocated = sampler._proposal_counts([0, 1], 1, True)
        counts += list(allocated.values())
    assert counts.sum() == 4000
    assert .72 < counts[1] / counts.sum() < .78
    assert counts[0] > 0


def test_allocation_is_seeded_and_independent_of_volume_units():
    a, b = generator([1., 2., 4.]), generator([.001, .002, .004])
    for _ in range(20):
        assert a._proposal_counts([0, 1, 2], 12, True) == b._proposal_counts([0, 1, 2], 12, True)


def test_caps_empty_pools_and_available_subset():
    sampler = generator([1., 2., 100.], cap=2)
    assert sampler._proposal_counts([], 10, True) == {}
    assert sampler._proposal_counts([0, 1], 100, True) == {0: 2, 1: 2}
    assert sampler._proposal_counts([1], 1, True) == {1: 1}
    assert sampler._proposal_counts([0, 2], 0, True) == {0: 0, 2: 0}


def test_zero_exponent_preserves_equal_allocation_and_rotating_small_budgets():
    sampler = generator([1., 100., 10000.], exponent=0.)
    assert sampler._proposal_counts([0, 1, 2], 6, True) == {0: 2, 1: 2, 2: 2}
    for call in (1, 2, 3):
        sampler._sampling_calls = call
        counts = sampler._proposal_counts([0, 1, 2], 1, True)
        assert counts[call - 1] == 1 and sum(counts.values()) == 1


@pytest.mark.parametrize('value', [-1., float('nan'), float('inf')])
def test_invalid_volume_exponent_is_rejected(value):
    with pytest.raises(ValueError, match='volume exponent'):
        generator([1.], exponent=value)


@pytest.mark.parametrize('platform', ['excavator', 'tabletop'])
def test_profile_records_bias_and_missing_setting_preserves_old_runs(platform, tmp_path):
    source = ROOT / f'apps/{platform}/config/action_generation.yml'
    profile = load_action_generation_profile(source, max_poses_per_object=32, banned_object_ids=())
    assert profile.generator.stone_volume_exponent == 1.
    values = yaml.safe_load(source.read_text())
    values.pop('stone_sampling')
    path = tmp_path / 'old.yml'
    path.write_text(yaml.safe_dump(values))
    old = load_action_generation_profile(path, max_poses_per_object=32, banned_object_ids=())
    assert old.generator.stone_volume_exponent == 0.


@pytest.mark.parametrize('application', [ExcavatorApplication, TabletopApplication])
def test_generation_excludes_placed_and_banned_stones(application, monkeypatch):
    app = application.from_repository(ROOT)
    inventory = _inventory(3)
    gen = app.build_action_generator(inventory, _PoseSolver())
    gen.config = replace(gen.config, banned_object_ids=frozenset({20}))
    sampler = gen.planar_sampler
    from stacking_common.actions.types import PlanarPlacement
    monkeypatch.setattr(sampler, 'sample', lambda *args: tuple(
        PlanarPlacement(np.array([i * .01, 0., 0., 0., 0., 0., 1.]), 0.)
        for i in range(32)))
    seen = []
    def build(jobs):
        seen.extend(job[2] for job in jobs)
        return [(None, 'test_rejection') for _ in jobs]
    monkeypatch.setattr(gen, '_build_jobs', build)
    try:
        result = gen.generate(_state(inventory, placed=(0,)), None,
                              ActionGenerationRequest(max_candidates=8, sampling_priority='lower_floor_first'))
        assert result.attempts == 8
        assert seen == [2] * 8
    finally:
        gen.close()
