"""Noisy-trial excursion and bounded-rocking acceptance boundaries."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stacking_common.environment.robustness import PlaceStabilityConfig, pose_window_extent, trial_outcome


BASE = np.array([0., 0., 0., 0., 0., 0., 1.])
CONFIG = PlaceStabilityConfig(True, 1, rotation_weight=.1, fail_displacement=.2,
    reject_failed_trials=True, settling_position_tolerance=.002, settling_rotation_tolerance_deg=.5)
PHASE = NS(dt=.02, extra_steps=150, min_time=3.)


def trace():
    return np.tile(BASE, (751, 1))


def test_energy_settled_has_excursion_penalty_without_rocking_label():
    poses = trace()
    poses[:, 0] = .03
    peak, reason = trial_outcome(BASE, poses, CONFIG, PHASE, energy_settled=True)
    assert peak == pytest.approx(.03) and reason is None


def test_small_rocking_is_accepted_with_nonzero_penalty():
    poses = trace()
    angles = .15 * np.sin(np.arange(len(poses)) * .25)
    poses[:, 0] = .0005 * np.sin(np.arange(len(poses)) * .25)
    poses[:, 3:] = Rotation.from_euler('x', angles[:, None], degrees=True).as_quat()
    peak, reason = trial_outcome(BASE, poses, CONFIG, PHASE, energy_settled=False)
    assert reason == 'bounded_rocking' and peak > 0
    assert not CONFIG.failed(peak, [reason])
    assert trial_outcome(BASE, poses, replace(CONFIG, settling_position_tolerance=0), PHASE,
                         energy_settled=False)[1] == 'not_settled'


@pytest.mark.parametrize('kind', ['drift', 'large_rocking', 'rotation'])
def test_persistent_motion_outside_window_limits_rejects(kind):
    poses = trace()
    if kind == 'drift':
        poses[:, 0] = np.linspace(0, .05, len(poses))
    elif kind == 'large_rocking':
        poses[:, 0] = .005 * np.sin(np.arange(len(poses)) * .25)
    else:
        poses[:, 3:] = Rotation.from_euler('x', np.sin(np.arange(len(poses)) * .25)[:, None], degrees=True).as_quat()
    peak, reason = trial_outcome(BASE, poses, CONFIG, PHASE, energy_settled=False)
    assert reason == 'not_settled' and CONFIG.failed(peak, [reason])


@pytest.mark.parametrize('kind', ['slide', 'tip'])
def test_large_excursion_cannot_be_hidden_by_returning_to_initial_pose(kind):
    poses = trace()
    if kind == 'slide':
        poses[100, 0] = .3
    else:
        poses[100, 3:] = Rotation.from_euler('x', 150, degrees=True).as_quat()
    peak, reason = trial_outcome(BASE, poses, CONFIG, PHASE, energy_settled=True)
    assert reason == 'excursion_limit' and peak > .2
    assert CONFIG.failed(peak, [reason])
    assert replace(CONFIG, reject_failed_trials=False).failed(peak, [reason])


def test_short_history_cannot_count_as_bounded_settling():
    assert trial_outcome(BASE, trace()[:150], CONFIG, PHASE, energy_settled=False)[1] == 'not_settled'


def test_window_is_independent_of_quaternion_sign_and_handles_rotation():
    poses = trace()[:20]
    poses[:, 3:] = Rotation.from_euler('x', np.linspace(-.2, .2, 20)[:, None], degrees=True).as_quat()
    reference = pose_window_extent(poses)
    poses[::2, 3:] *= -1
    assert pose_window_extent(poses) == pytest.approx(reference)
    assert np.rad2deg(reference[1]) == pytest.approx(.4, abs=1e-5)


@pytest.mark.parametrize('bad', ['empty', 'nan', 'zero_quaternion'])
def test_invalid_history_is_a_hard_failure(bad):
    poses = trace()
    if bad == 'empty': poses = poses[:0]
    elif bad == 'nan': poses[2, 0] = np.nan
    else: poses[2, 3:] = 0
    peak, reason = trial_outcome(BASE, poses, CONFIG, PHASE, energy_settled=True)
    assert reason == 'nonfinite_simulation'
    assert replace(CONFIG, reject_failed_trials=False, fail_displacement=None).failed(peak, [reason])


@pytest.mark.parametrize('values', [dict(settling_position_tolerance=-1), dict(settling_rotation_tolerance_deg=np.nan)])
def test_invalid_window_configuration_is_rejected(values):
    with pytest.raises(ValueError):
        replace(CONFIG, **values)


def test_support_excursion_rejects_even_when_new_stone_recovers():
    from stacking_common.environment.robustness import scene_trial_outcome
    support = trace()
    support[100, 0] = .3
    peak, reason = scene_trial_outcome({1: BASE, 2: BASE}, {1: trace(), 2: support},
        CONFIG, PHASE, energy_settled=True)
    assert peak == pytest.approx(.3) and reason == 'excursion_limit'


def test_every_dynamic_body_must_meet_bounded_rocking_limits():
    from stacking_common.environment.robustness import scene_trial_outcome
    support = trace()
    support[:, 0] = np.linspace(0, .05, len(support))
    _, reason = scene_trial_outcome({1: BASE, 2: BASE}, {1: trace(), 2: support},
        CONFIG, PHASE, energy_settled=False)
    assert reason == 'not_settled'
