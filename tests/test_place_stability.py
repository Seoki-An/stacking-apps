"""Upward noisy-placement trials, isolation, and reward propagation."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stacking_common.domain import Action, State
from stacking_common.environment import CorePlacementBackend, CorePlacementConfig, SimulationPhaseConfig, SimulationTransition, StackingReward, RewardConfig, RewardWeights
from stacking_common.environment.robustness import PlaceStabilityConfig, evaluate_place_stability, pose_displacement
from stacking_common.profiles import load_environment_profile

ROOT = Path(__file__).resolve().parents[1]


def pose(z=.2):
    return np.array([0., 0., z, 0., 0., 0., 1.])


@pytest.mark.parametrize('platform,count,std,weight', [
    ('excavator', 4, [.02, .02, .02], .1),
    ('tabletop', 8, [.005, .005, .005], .01),
])
def test_platform_profiles_reproduce_seeded_upward_noise(platform, count, std, weight):
    profile = load_environment_profile(ROOT / f'apps/{platform}/config/environment.yml')
    config = profile.place_stability
    assert config.active and config.n_noise == count
    assert config.rotation_weight == weight
    np.testing.assert_array_equal(config.position_std, std)
    initial = pose()
    initial[3:] = Rotation.from_euler('xyz', [25, 12, 47], degrees=True).as_quat()
    # Seed and left-multiplied Gaussian rotvec convention from both legacy simulators.
    rng = np.random.default_rng(1009 * 2 + 6361 * 1 + 9176 * 20)
    actual = list(config.noisy_poses(initial, step=2, object_index=1, object_id=20))
    for noisy in actual:
        expected = initial.copy()
        position_noise = rng.normal(0., std, size=3)
        expected[:3] += [position_noise[0], position_noise[1], abs(position_noise[2])]
        noise = rng.normal(0., np.deg2rad(5.), size=3)
        expected[3:] = (Rotation.from_rotvec(noise) * Rotation.from_quat(initial[3:])).as_quat()
        np.testing.assert_allclose(noisy, expected, atol=1e-14)
    np.testing.assert_array_equal(actual, list(config.noisy_poses(initial, step=2, object_index=1, object_id=20)))


@pytest.mark.parametrize('changes', [dict(reject_failed_trials='true'), dict(enabled='true'), dict(n_noise=-1), dict(n_noise=1.2), dict(seed=-1), dict(position_std=[1, 2]), dict(position_std=[0, -1, 0]), dict(rotation_std_deg=np.inf), dict(rotation_weight=-1), dict(fail_displacement=-1)])
def test_invalid_settings_are_rejected(changes):
    with pytest.raises(ValueError):
        PlaceStabilityConfig(**changes)


def test_displacement_uses_translation_plus_shortest_rotation():
    initial = pose()
    final = initial.copy()
    final[:3] += [.03, .04, 0]
    final[3:] = Rotation.from_euler('z', 90, degrees=True).as_quat()
    assert pose_displacement(initial, final, .1) == pytest.approx(.05 + .1 * np.pi / 2)
    final[3:] *= -1
    assert pose_displacement(initial, final, .1) == pytest.approx(.05 + .1 * np.pi / 2)
    assert pose_displacement(initial, np.full(7, np.nan), .1) == np.inf


@pytest.mark.parametrize('failed_trial', [False, True])
def test_worst_trial_uses_settled_pose_without_mutating_state(monkeypatch, failed_trial):
    import stacking_common.environment.robustness as module
    config = PlaceStabilityConfig(True, 3, (.001, .001, .001), 5., .1, .02)
    backend = NS(place_stability=config)
    action = Action(0, pose(.3), pose(.4), object_id=10)
    state = State([10], [], {}, [], [])
    starts = []
    def simulate(b, s, a, noisy):
        starts.append(noisy.copy())
        final = noisy.copy()
        final[0] += .01 * len(starts)
        if failed_trial and len(starts) == 2:
            final[:] = np.nan
        return final, 'nonfinite_simulation' if not np.all(np.isfinite(final)) else None
    monkeypatch.setattr(module, '_simulate_noisy_place', simulate)
    displacement, trials, reasons = evaluate_place_stability(backend, state, action, pose(.2))
    assert len(trials) == 3
    assert displacement == pytest.approx(np.inf if failed_trial else .03, abs=1e-8)
    assert config.failed(displacement)
    assert all(abs(p[2] - .2) < .01 for p in starts)
    np.testing.assert_array_equal(action.pose, pose(.3))
    assert state.placed_object_indices == []
    backend.place_stability = replace(config, enabled=False)
    assert evaluate_place_stability(backend, state, action, pose()) == (0., [], [])
    assert len(starts) == 3


def make_backend(config):
    core = pytest.importorskip('stacking_core')
    points = np.array([[x, y, z] for x in [-.05, .05] for y in [-.05, .05] for z in [-.05, .05]])
    model = core.BodyModel.dsf(10, points, mass=1., inertia=np.eye(3), friction=.6, sharpness=100)
    return CorePlacementBackend({10: model}, CorePlacementConfig(
        structural=SimulationPhaseConfig(.01, 1., 1., 0., .1, 1),
        settle_dt=.01, settle_steps=2, contact_gap_tolerance=.01, minimum_support_count=1,
    ), place_stability=config)


def test_native_trial_result_reaches_reward_and_does_not_move_main_scene():
    config = PlaceStabilityConfig(True, 2, (.001, .001, .001), 5., .01)
    backend = make_backend(config)
    transition = SimulationTransition([10], backend)
    transition.reset()
    state = transition.step(Action(0, pose(), pose(.3)))
    displacement = state.action_history[-1].place_robustness_displacement
    assert displacement > 0
    assert len(state.action_history[-1].diagnostics['place_robustness_trial_displacements']) == 2
    baseline = SimulationTransition([10], make_backend(replace(config, enabled=False)))
    baseline.reset()
    without_trials = baseline.step(Action(0, pose(), pose(.3)))
    np.testing.assert_allclose(state.object_poses[10], without_trials.object_poses[10])
    assert without_trials.action_history[-1].place_robustness_displacement == 0.
    stack_object = NS(object_id=10, volume=1., representative_face_normals=lambda: ([], []))
    inventory = NS(objects=[stack_object], initial_pose=lambda _: pose(), target=NS(origin=np.zeros(3), height=1., volume=1.))
    overlap = NS(object_containment=lambda *a: 1.)
    reward = StackingReward(inventory, overlap, RewardConfig(1., .5, RewardWeights(0., 0., 0., 2., 0., 0.)), support_evaluator=lambda *a: (1, True))
    value, info = reward.compute(state)
    assert info['place_stability'] == -displacement
    assert value == pytest.approx(-2 * displacement)


def test_native_trials_keep_previous_bodies_static_and_start_fresh():
    backend = make_backend(PlaceStabilityConfig(True, 2))
    backend.models[20] = backend.models[10]
    backend.scene_builder.models[20] = backend.models[10]
    backend.reset([20, 10])
    previous = pose(.05); previous[0] = 2.
    state = State([20, 10], [0], {20: previous.copy()}, [], [])
    action = Action(1, pose(), pose(.3), object_id=10)
    scenes = []
    original = backend._new_simulator
    def simulator():
        native = original()
        def step(scene, dt):
            result = native.step(scene, dt)
            np.testing.assert_array_equal(result.snapshot.body_pose(1), previous)
            np.testing.assert_array_equal(result.snapshot.body_linear_velocity(1), np.zeros(3))
            return result
        scenes.append(native)
        return NS(step=step)
    backend._new_simulator = simulator
    displacement, trials, reasons = evaluate_place_stability(backend, state, action, pose())
    assert displacement > 0 and len(trials) == len(scenes) == 2
    assert scenes[0] is not scenes[1]
    np.testing.assert_array_equal(state.object_poses[20], previous)


def test_hard_reject_reaches_transition_even_when_reward_would_clip(monkeypatch):
    import stacking_common.environment.core_backend as module
    backend = make_backend(PlaceStabilityConfig(True, 2, fail_displacement=.01))
    monkeypatch.setattr(module, 'evaluate_place_stability', lambda *a: (np.inf, [np.inf], ['nonfinite_simulation']))
    transition = SimulationTransition([10], backend)
    transition.reset()
    state = transition.step(Action(0, pose(), pose(.3)))
    assert state.failed
    assert state.action_history[-1].diagnostics['rejection_reason'] == 'place_robustness'
    assert np.isinf(state.action_history[-1].place_robustness_displacement)


@pytest.mark.parametrize('platform', ['excavator', 'tabletop'])
def test_search_backend_receives_robustness_for_both_fidelities(monkeypatch, platform):
    from stacking_excavator import ExcavatorApplication
    from stacking_tabletop import TabletopApplication
    from stacking_common.search import EvaluationFidelity
    cls = ExcavatorApplication if platform == 'excavator' else TabletopApplication
    application = cls.from_repository(ROOT)
    calls = []
    def build(models, profile, **kwargs):
        calls.append((profile, kwargs['place_stability']))
        return 'backend'
    monkeypatch.setattr(CorePlacementBackend, 'from_profile', staticmethod(build))
    for fidelity in (EvaluationFidelity.SHORT, EvaluationFidelity.LONG):
        assert application.build_core_backend(NS(models={}), fidelity) == 'backend'
    assert calls[0][0] is application.short_simulation_profile
    assert calls[1][0] is application.simulation_profile
    assert all(config is application.environment_profile.place_stability for _, config in calls)


def test_older_environment_profiles_keep_trials_disabled():
    import yaml
    from stacking_common.environment import EnvironmentProfile
    values = yaml.safe_load((ROOT / 'apps/tabletop/config/environment.yml').read_text())
    del values['reward']['place_stability']
    assert not EnvironmentProfile.from_mapping(values).place_stability.active


@pytest.mark.parametrize('reason', ['target_motion_limit', 'not_settled', 'nonfinite_simulation'])
@pytest.mark.parametrize('reject', [False, True])
def test_any_failed_trial_rejects_candidate_only_when_enabled(monkeypatch, reason, reject):
    import stacking_common.environment.core_backend as module
    backend = make_backend(PlaceStabilityConfig(
        True, 3, reject_failed_trials=reject,
    ))
    # A finite, small score must not hide a failure in the middle trial.
    monkeypatch.setattr(module, 'evaluate_place_stability', lambda *a: (
        .001, [.001, .001, .001], [None, reason, None],
    ))
    transition = SimulationTransition([10], backend)
    transition.reset()
    state = transition.step(Action(0, pose(), pose(.3)))
    assert state.failed is reject
    diagnostics = state.action_history[-1].diagnostics
    assert diagnostics['place_robustness_trial_failure_reasons'] == [None, reason, None]
    assert diagnostics['place_robustness_failed'] is reject
    if reject:
        assert diagnostics['rejection_reason'] == 'place_robustness'


def test_trial_filter_and_displacement_limit_are_independent():
    config = PlaceStabilityConfig(True, 2, reject_failed_trials=True)
    assert not config.failed(.4, [None, None])
    assert config.failed(np.inf, [None, None])
    assert replace(config, fail_displacement=.1).failed(.4, [None, None])
    assert not replace(config, enabled=False).failed(np.inf, ['not_settled'])
    assert not replace(config, n_noise=0).failed(np.inf, ['not_settled'])


@pytest.mark.parametrize('scenario,expected', [
    ('rest', None),
    ('motion', 'target_motion_limit'),
    ('timeout', 'not_settled'),
    ('shortcut', 'not_settled'),
    ('nonfinite_pose', 'nonfinite_simulation'),
    ('nonfinite_velocity', 'nonfinite_simulation'),
])
def test_trial_simulation_reports_actual_failure(scenario, expected):
    from stacking_common.environment.robustness import _simulate_noisy_place
    phase = SimulationPhaseConfig(
        dt=.01, energy_threshold=1e-8,
        velocity_integral_threshold=.05 if scenario == 'shortcut' else .5,
        min_time=0., max_time=.1, extra_steps=2, final_extra_steps=2,
    )
    calls = []
    def step(scene, dt):
        calls.append(dt)
        speed = {'rest': 0., 'motion': 100., 'timeout': 1.,
                 'shortcut': 2.3 if len(calls) == 1 else .001,
                 'nonfinite_pose': 0., 'nonfinite_velocity': np.nan}[scenario]
        final = np.full(7, np.nan) if scenario == 'nonfinite_pose' else pose()
        return NS(snapshot=NS(
            body_pose=lambda _: final,
            body_linear_velocity=lambda _: np.array([speed, 0., 0.]),
            body_angular_velocity=lambda _: np.zeros(3),
        ))
    backend = NS(
        config=NS(structural=phase), object_ids=[10],
        scene_builder=NS(entity_id=lambda _: 1),
        models={10: NS(inertia=np.eye(3), mass=1.)},
        _scene=lambda *a, **kw: None, _new_simulator=lambda: NS(step=step),
    )
    state = State([10], [], {}, [], [])
    action = Action(0, pose(), pose(.3), object_id=10)
    _, reason = _simulate_noisy_place(backend, state, action, pose())
    assert reason == expected
    assert calls and all(dt == .02 for dt in calls)


def test_excavator_profile_rejects_failed_or_excessively_displaced_noise_trials():
    config = load_environment_profile(
        ROOT / 'apps/excavator/config/environment.yml').place_stability
    assert not config.failed(.1, [None] * config.n_noise)
    assert config.failed(.1, ['not_settled'] + [None] * (config.n_noise - 1))
    assert config.failed(.1, ['target_motion_limit'] + [None] * (config.n_noise - 1))
    assert config.failed(.16, [None] * config.n_noise)
    assert config.failed(float('inf'), [None] * config.n_noise)


@pytest.mark.parametrize('z_std', [0., .02, .05])
@pytest.mark.parametrize('base_z', [-.5, .2])
def test_vertical_noise_never_lowers_pose_and_horizontal_noise_keeps_both_signs(z_std, base_z):
    config = PlaceStabilityConfig(True, 128, (.02, .02, z_std), 5.)
    initial = pose(base_z)
    before = initial.copy()
    trials = np.array(list(config.noisy_poses(initial, step=7, object_index=1, object_id=2)))
    offsets = trials[:, :3] - initial[:3]
    assert np.all(offsets[:, 2] >= 0.)
    if z_std == 0.:
        np.testing.assert_array_equal(offsets[:, 2], 0.)
    else:
        assert np.any(offsets[:, 2] > 0.)
    assert np.all(np.any(offsets[:, :2] < 0., axis=0))
    assert np.all(np.any(offsets[:, :2] > 0., axis=0))
    np.testing.assert_array_equal(initial, before)


def test_step7_stone2_trial3_reflects_downward_noise_upward():
    # This seed previously translated the third trial 12.57 cm into the ground.
    config = PlaceStabilityConfig(True, 4, (.02, .02, .05), 5.)
    initial = pose()
    trials = list(config.noisy_poses(initial, step=7, object_index=1, object_id=2))
    np.testing.assert_allclose(trials[2][:3] - initial[:3], [.01022, .01973, .12571], atol=1e-5)
