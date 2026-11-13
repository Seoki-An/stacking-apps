"""Separate refinement damping must not bleed into structural validation."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from stacking_common.environment import CoreSimulationProfile
from stacking_common.artifacts.planning_context import _values
from stacking_common.profiles.simulation import load_simulation_profile

ROOT = Path(__file__).resolve().parents[1]


def values():
    return yaml.safe_load((ROOT/'apps/tabletop/config/solvers/simulation.yml').read_text())


def test_settling_and_structural_rates_are_independent():
    data = values()
    data['integration']['damping'] = .3
    data['physics']['settling']['damping'] = {'linear': .5, 'angular': 1.}
    data['physics']['structural']['damping'] = {'linear': 0., 'angular': 0.}
    profile = CoreSimulationProfile.from_mapping(data)
    settling, structural = profile.settling_native_config(), profile.native_config()
    assert (settling.linear_damping, settling.angular_damping) == (.5, 1.)
    assert (structural.linear_damping, structural.angular_damping) == (0., 0.)
    assert settling.damping == structural.damping == .3  # explicit zero overrides fallback
    recorded = _values(profile)
    assert recorded['settling_physics']['angular_damping'] == 1.
    assert recorded['linear_damping'] == 0.


def test_legacy_scalar_context_and_native_behavior_remain_available():
    data = values()
    for phase in data['physics'].values():
        phase.pop('damping', None)
    data['integration']['damping'] = .3
    profile = CoreSimulationProfile.from_mapping(data)
    for native in (profile.native_config(), profile.settling_native_config()):
        assert native.damping == .3
        assert native.linear_damping is native.angular_damping is None
    assert 'linear_damping' not in _values(profile)
    assert 'angular_damping' not in _values(profile.settling_physics)


@pytest.mark.parametrize('phase', ['settling', 'structural'])
@pytest.mark.parametrize('bad', [{'linear': .5}, {'angular': 1.}, {'linear': 0., 'angular': 0., 'typo': 2.}, .5])
def test_phase_damping_requires_both_named_rates(phase, bad):
    data = values()
    data['physics'][phase]['damping'] = bad
    with pytest.raises(ValueError, match='linear and angular'):
        CoreSimulationProfile.from_mapping(data)


@pytest.mark.parametrize('name', ['linear_damping', 'angular_damping'])
@pytest.mark.parametrize('bad', [-1., np.nan, np.inf])
def test_invalid_rates_fail_before_native_allocation(name, bad):
    profile = CoreSimulationProfile.from_mapping(values())
    with pytest.raises(ValueError, match=name):
        replace(profile, **{name:bad})


def test_requested_nonzero_damping_never_silently_uses_an_old_binding():
    from types import SimpleNamespace
    profile = CoreSimulationProfile.from_mapping(values())
    profile = replace(profile, linear_damping=.5, angular_damping=1.)
    with pytest.raises(RuntimeError, match='separate linear/angular damping'):
        profile.native_config(SimpleNamespace(SimulationConfig=lambda: SimpleNamespace(damping=0.)))


def test_excavator_damps_only_pose_refinement():
    profile = load_simulation_profile(ROOT/'apps/excavator/config/solvers/simulation.yml')
    assert (profile.settling_physics.linear_damping, profile.settling_physics.angular_damping) == (.5,1.)
    assert profile.linear_damping == profile.angular_damping == 0.
    assert profile.placement.settle_dt == .01
    assert profile.placement.settle_steps == 500
