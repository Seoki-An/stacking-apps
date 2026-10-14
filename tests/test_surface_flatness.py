from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from stacking_common.domain import Action, DirectionalTaper, ObjectInventory, StackObject, State, TargetGeometry
from stacking_common.environment import EnvironmentProfile, RewardConfig, RewardWeights, StackingReward
from stacking_common.environment.surface import SurfaceFlatness
from test_action_generation import _BodyModel, _cube, _pose


def inputs(scale=1., floor_z=0.):
    objects = [StackObject(i, _BodyModel(), [_cube(scale)]) for i in (10, 20)]
    inventory = ObjectInventory(objects, {i: _pose() for i in (10, 20)},
        TargetGeometry(2 * scale, scale, 4 * scale,
                       np.array([0., 0., floor_z]), DirectionalTaper(), 0.))
    return inventory


def state(left=.5, right=None, scale=1., floor_z=0.):
    poses = {10: _pose(x=-.5 * scale, z=left * scale + floor_z)}
    if right is not None:
        poses[20] = _pose(x=.5 * scale, z=right * scale + floor_z)
    indices = list(range(len(poses)))
    index = indices[-1]
    oid = (10, 20)[index]
    return State([10, 20], indices, poses, [],
                 [Action(index, poses[oid], poses[oid], object_id=oid)])


def test_empty_floor_and_global_level_difference():
    surface = SurfaceFlatness(inputs())
    assert surface.measure(State([10, 20], [], {}, [], [])) == (0., 0.)
    assert surface.measure(state()) == pytest.approx((1., .5))
    assert surface.measure(state(right=.5)) == pytest.approx((1., 1.))
    assert surface.measure(state(right=2.5)) == pytest.approx((3., 2.))


def test_incremental_overlap_siblings_and_motion_match_fresh_computation(monkeypatch):
    inventory = inputs()
    surface = SurfaceFlatness(inventory)
    parent = state()
    surface.measure(parent)
    # A completely hidden stone contributes no added height to the envelope.
    overlap = state(right=.5)
    overlap.object_poses[20] = overlap.object_poses[10].copy()
    assert surface.measure(overlap) == pytest.approx((1., .5))
    for child in (state(right=.5), state(right=2.5), state(left=1.5, right=.5)):
        assert surface.measure(child) == pytest.approx(SurfaceFlatness(inventory).measure(child))
    # Lowering a formerly high stone must lower the max, not retain stale peaks.
    moved = state(left=.1, right=.1)
    assert surface.measure(moved) == pytest.approx((.6, .6))
    assert surface.measure(parent) == pytest.approx((1., .5))
    monkeypatch.setattr(surface, '_object_surface', lambda *_: pytest.fail('cached scene rerasterized'))
    assert surface.measure(moved) == pytest.approx((.6, .6))


@pytest.mark.parametrize('scale,floor_z', [(1., 0.), (.01, 2.), (10., -5.)])
def test_reward_sign_normalization_and_disabled_path(scale, floor_z, monkeypatch):
    inventory = inputs(scale, floor_z)
    overlap = SimpleNamespace(target_coverage=lambda *_: 0., object_containment=lambda *_: 1.)
    config = RewardConfig(1., .5, RewardWeights(0., 0., 0., 0., 0., 0., surface_flatness=2.))
    reward = StackingReward(inventory, overlap, config, support_evaluator=lambda *_: (1, True))
    value, info = reward.compute(state(right=2.5, scale=scale, floor_z=floor_z))
    assert value == pytest.approx(-2.)
    assert info['surface_flatness'] == pytest.approx(-1.)
    assert info['surface_height_relief'] == pytest.approx(scale)
    assert info['surface_flatness_reward'] == pytest.approx(value)
    assert reward.compute(state(right=.5, scale=scale, floor_z=floor_z))[0] == pytest.approx(0.)
    monkeypatch.setattr(SurfaceFlatness, '__init__', lambda *_: pytest.fail('disabled reward created surface'))
    disabled = StackingReward(inventory, overlap,
        replace(config, weights=replace(config.weights, surface_flatness=0.)),
        support_evaluator=lambda *_: (1, True))
    assert disabled.compute(state(right=2.5, scale=scale, floor_z=floor_z))[0] == 0.


@pytest.mark.parametrize('platform', ['excavator', 'tabletop'])
def test_profile_weight_and_legacy_default(platform):
    from pathlib import Path
    values = yaml.safe_load((Path(__file__).resolve().parents[1] /
                            f'apps/{platform}/config/environment.yml').read_text())
    values['reward']['weights']['surface_flatness'] = .75
    assert EnvironmentProfile.from_mapping(values).reward.weights.surface_flatness == .75
    del values['reward']['weights']['surface_flatness']
    assert EnvironmentProfile.from_mapping(values).reward.weights.surface_flatness == 0.
    values['reward']['weights']['surface_flatness'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        EnvironmentProfile.from_mapping(values)
