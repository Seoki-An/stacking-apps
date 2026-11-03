"""Velocity-disturbance trials, isolation, and reward propagation."""
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


@pytest.mark.parametrize('platform,magnitude,weight', [
    ('tabletop', [.02, .02, .02], .01),
])
def test_platform_profiles_cover_signed_xy_with_downward_velocity(platform, magnitude, weight):
    config = load_environment_profile(ROOT / f'apps/{platform}/config/environment.yml').place_stability
    assert config.active and config.n_noise == 4
    assert config.contact_hops == 2
    assert config.rotation_weight == weight
    np.testing.assert_array_equal(config.linear_velocity_magnitude, magnitude)
    actual = np.array(list(config.velocity_disturbances(step=2, object_index=1, object_id=20)))
    vx, vy, vz = magnitude
    np.testing.assert_allclose(actual[:, 0], [[vx, 0, -vz], [0, vy, -vz],
                                            [-vx, 0, -vz], [0, -vy, -vz]], atol=1e-16)
    np.testing.assert_array_equal(actual[:, 1], 0.)
    np.testing.assert_array_equal(actual, list(config.velocity_disturbances(step=30, object_index=45, object_id=55)))


@pytest.mark.parametrize('count', [1, 2, 3, 4, 8, 16])
def test_direction_count_changes_resolution_without_changing_kick_speed(count):
    config = PlaceStabilityConfig(True, count, linear_velocity_magnitude=(.1, .1, .05))
    trials = np.array(list(config.velocity_disturbances(step=1, object_index=0, object_id=1)))
    assert len(trials) == count
    np.testing.assert_allclose(np.linalg.norm(trials[:, 0, :2], axis=1), .1)
    np.testing.assert_array_equal(trials[:, 0, 2], -.05)
    np.testing.assert_array_equal(trials[:, 1], 0.)
    if count > 1:
        np.testing.assert_allclose(trials[:, 0, :2].sum(axis=0), 0., atol=1e-15)
    if count % 2 == 0:
        np.testing.assert_allclose(trials[count // 2:, 0, :2], -trials[:count // 2, 0, :2], atol=1e-15)
    assert not list(replace(config, enabled=False).velocity_disturbances(step=1, object_index=0, object_id=1))
    assert not list(replace(config, n_noise=0).velocity_disturbances(step=1, object_index=0, object_id=1))


def test_saved_gaussian_profile_retains_original_draws():
    config = PlaceStabilityConfig.from_mapping(dict(enabled=True, n_noise=4,
        linear_velocity_std=[.1, .1, .05], angular_velocity_std_deg=5., seed=0))
    rng = np.random.default_rng(1009 * 2 + 6361 + 9176 * 20)
    for linear, angular in config.velocity_disturbances(step=2, object_index=1, object_id=20):
        np.testing.assert_array_equal(linear, rng.normal(0., [.1, .1, .05], size=3))
        np.testing.assert_array_equal(angular, rng.normal(0., np.deg2rad(5.), size=3))


def test_directional_profile_roundtrip_preserves_sampling():
    from dataclasses import asdict
    config = PlaceStabilityConfig(True, 8, linear_velocity_magnitude=(.1, .1, .05))
    restored = PlaceStabilityConfig.from_mapping(asdict(config))
    assert restored == config
    np.testing.assert_array_equal(list(config.velocity_disturbances(step=1, object_index=0, object_id=1)),
                                  list(restored.velocity_disturbances(step=2, object_index=1, object_id=2)))


@pytest.mark.parametrize('changes', [dict(reject_failed_trials='true'), dict(enabled='true'),
    dict(n_noise=-1), dict(n_noise=1.2), dict(seed=-1), dict(linear_velocity_std=[1, 2]),
    dict(linear_velocity_std=[0, -1, 0]), dict(angular_velocity_std_deg=np.inf),
    dict(rotation_weight=-1), dict(fail_displacement=-1),
    dict(linear_velocity_magnitude=[1, 2]), dict(linear_velocity_magnitude=[1, -1, 0]),
    dict(linear_velocity_magnitude=[1, np.inf, 0]),
    dict(linear_velocity_magnitude=[1, 1, 0], linear_velocity_std=[1, 1, 1]),
    dict(linear_velocity_magnitude=[1, 1, 0], angular_velocity_std_deg=5.),
    dict(force_ratio=-1), dict(force_ratio=np.inf), dict(force_ratio=True),
    dict(force_duration=0), dict(force_duration=np.nan),
    dict(force_ratio=.05, linear_velocity_magnitude=[0, 0, 0]),
    dict(force_ratio=.05, linear_velocity_std=[.1, 0, 0]),
    dict(force_ratio=.05, angular_velocity_std_deg=1), dict(contact_hops=-1), dict(contact_hops=True), dict(contact_hops=1.5), dict(contact_hops="2")])
def test_invalid_settings_are_rejected(changes):
    with pytest.raises(ValueError):
        PlaceStabilityConfig(**changes)


def test_pose_noise_config_cannot_be_silently_reinterpreted_as_velocity():
    with pytest.raises(ValueError, match='pose-noise'):
        PlaceStabilityConfig.from_mapping(dict(position_std=[.02] * 3))


def test_signed_disturbances_include_every_direction():
    config = PlaceStabilityConfig(True, 128, (.05, .05, .05), 5.)
    trials = np.array(list(config.velocity_disturbances(step=7, object_index=1, object_id=2)))
    assert np.all(np.any(trials < 0., axis=0))
    assert np.all(np.any(trials > 0., axis=0))
    zero = PlaceStabilityConfig(True, 2)
    np.testing.assert_array_equal(list(zero.velocity_disturbances(step=1, object_index=0, object_id=1)), 0.)


@pytest.mark.parametrize('failed_trial', [False, True])
def test_worst_trial_preserves_nominal_snapshot_and_state(monkeypatch, failed_trial):
    import stacking_common.environment.robustness as module
    backend = NS(place_stability=PlaceStabilityConfig(True, 3, (.05, .05, .05), 5.))
    state = State([10], [], {}, [], [])
    action = Action(0, pose(.3), pose(.4), object_id=10)
    snapshot = NS(bodies=[])
    backend.core = NS(Mobility=NS(DYNAMIC="dynamic"))
    calls = []
    def simulate(b, s, a, nominal, linear, angular, dynamic, trial):
        assert dynamic == set()
        assert b is backend and s is state and a is action and nominal is snapshot
        calls.append((linear, angular))
        if failed_trial and len(calls) == 2:
            return np.inf, 'nonfinite_simulation'
        return .01 * len(calls), None
    monkeypatch.setattr(module, '_simulate_disturbance', simulate)
    maximum, values, reasons = evaluate_place_stability(backend, state, action, snapshot)
    assert len(values) == len(reasons) == len(calls) == 3
    assert maximum == pytest.approx(np.inf if failed_trial else .03)
    np.testing.assert_array_equal(action.pose, pose(.3))
    assert state.object_poses == {}
    backend.place_stability = replace(backend.place_stability, enabled=False)
    assert evaluate_place_stability(backend, state, action, snapshot) == (0., [], [])


def make_backend(config):
    core = pytest.importorskip('stacking_core')
    points = np.array([[x, y, z] for x in [-.05, .05] for y in [-.05, .05] for z in [-.05, .05]])
    model = core.BodyModel.dsf(10, points, mass=1., inertia=np.eye(3), friction=.6, sharpness=100)
    return CorePlacementBackend({10: model}, CorePlacementConfig(
        structural=SimulationPhaseConfig(.01, 1., 1., 0., .1, 1),
        settle_dt=.01, settle_steps=2, contact_gap_tolerance=.01, minimum_support_count=1,
    ), place_stability=config)


def test_native_trial_result_reaches_reward_and_does_not_move_main_scene():
    config = PlaceStabilityConfig(True, 4, rotation_weight=.01,
                                  linear_velocity_magnitude=(.001, .001, .001))
    backend = make_backend(config)
    transition = SimulationTransition([10], backend)
    transition.reset()
    state = transition.step(Action(0, pose(), pose(.3)))
    displacement = state.action_history[-1].place_robustness_displacement
    assert displacement > 0
    assert len(state.action_history[-1].diagnostics['place_robustness_trial_displacements']) == 4
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


def test_trial_scene_preserves_poses_roles_and_resets_residual_motion():
    from stacking_common.environment.robustness import disturbance_scene
    backend = make_backend(PlaceStabilityConfig(True, 2))
    core = backend.core
    model = backend.models[10]
    backend.reset([10])
    q = pose(.5)
    q[3:] = Rotation.from_euler('z', 90, degrees=True).as_quat()
    bodies = [core.BodyInstance(i, model, q + np.array([i, 0, 0, 0, 0, 0, 0]),
        linear=np.ones(3), angular=np.ones(3), mobility=mobility)
        for i, mobility in [(1, core.Mobility.DYNAMIC), (2, core.Mobility.DYNAMIC),
                            (3, core.Mobility.STATIC)]]
    snapshot = core.SceneSnapshot(0, bodies)
    action = Action(0, pose(), pose(), object_id=10)
    v, w = np.array([.1, -.2, -.1]), np.array([.2, -.1, .3])
    scene = disturbance_scene(backend, snapshot, action, v, w)
    for before, after in zip(snapshot.bodies, scene.bodies):
        np.testing.assert_array_equal(after.pose, before.pose)
        assert after.mobility == before.mobility
        np.testing.assert_array_equal(after.linear, v if before.id == 1 else np.zeros(3))
        np.testing.assert_array_equal(after.angular, w if before.id == 1 else np.zeros(3))
        np.testing.assert_array_equal(before.linear, np.ones(3))


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
    assert calls[0][0] is application.simulation_profile
    assert calls[1][0] is application.simulation_profile
    assert all(config is application.environment_profile.place_stability for _, config in calls)


def test_older_environment_profiles_keep_trials_disabled():
    import yaml
    from stacking_common.environment import EnvironmentProfile
    values = yaml.safe_load((ROOT / 'apps/tabletop/config/environment.yml').read_text())
    del values['reward']['place_stability']
    assert not EnvironmentProfile.from_mapping(values).place_stability.active


@pytest.mark.parametrize('reason', ['target_motion_limit', 'not_settled', 'nonfinite_simulation', 'bounded_rocking'])
@pytest.mark.parametrize('reject', [False, True])
def test_trial_outcome_rejection_and_diagnostics(monkeypatch, reason, reject):
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
    expected = reason == 'nonfinite_simulation' or (reject and reason != 'bounded_rocking')
    assert state.failed is expected
    diagnostics = state.action_history[-1].diagnostics
    assert diagnostics['place_robustness_trial_failure_reasons'] == [None, None if reason == 'bounded_rocking' else reason, None]
    assert diagnostics['place_robustness_trial_outcomes'] == ['settled', reason, 'settled']
    assert diagnostics['place_robustness_failed'] is expected
    if expected:
        assert diagnostics['rejection_reason'] == 'place_robustness'


def test_trial_filter_and_displacement_limit_are_independent():
    config = PlaceStabilityConfig(True, 2, reject_failed_trials=True)
    assert not config.failed(.4, [None, None])
    assert config.failed(np.inf, [None, None])
    assert replace(config, fail_displacement=.1).failed(.4, [None, None])
    assert not replace(config, enabled=False).failed(np.inf, ['not_settled'])
    assert not replace(config, n_noise=0).failed(np.inf, ['not_settled'])


@pytest.mark.parametrize('physical_reason,settled', [('target_motion_limit', True), (None, False)])
def test_nominal_failure_skips_disturbance_trials(monkeypatch, physical_reason, settled):
    import stacking_common.environment.core_backend as module
    backend = make_backend(PlaceStabilityConfig(True, 2))
    backend.reset([10])
    original = backend._simulate_structural
    def structural(*args):
        return replace(original(*args), settled=settled)
    monkeypatch.setattr(backend, '_simulate_structural', structural)
    monkeypatch.setattr(backend, '_physical_rejection_reason', lambda *a: physical_reason)
    def unexpected(*a):
        pytest.fail('must skip disturbances after nominal failure')
    monkeypatch.setattr(module, 'evaluate_place_stability', unexpected)
    result = backend.place(State([10], [], {}, [], []), Action(0, pose(), pose(), object_id=10))
    assert not result.diagnostics['place_robustness_evaluated']
    assert result.diagnostics['place_robustness_trial_displacements'] == []


def test_excavator_profile_preserves_strict_recovery_and_hard_guards():
    config = load_environment_profile(ROOT / 'apps/excavator/config/environment.yml').place_stability
    assert config.reject_failed_trials
    assert not config.failed(.1, [None, 'bounded_rocking'])
    assert config.failed(.1, ['not_settled'])
    for reason in ('nonfinite_simulation', 'simulation_cuda_geometry_failure',
                   'incomplete_robustness_history', 'excursion_limit'):
        assert replace(config, reject_failed_trials=False).failed(.1, [reason])
    assert config.failed(config.fail_displacement + .01, [None])


def test_robustness_receives_final_structural_snapshot(monkeypatch):
    import stacking_common.environment.core_backend as module
    backend = make_backend(PlaceStabilityConfig(True, 2))
    backend.reset([10])
    original = backend._simulate_structural
    snapshots = []
    def structural(*args):
        result = original(*args)
        assert result.settled
        snapshots.append((result.snapshot, result.contacts))
        return result
    def evaluate(b, state, action, snapshot, contacts):
        assert snapshot is snapshots[0][0]
        assert contacts is snapshots[0][1]
        return .01, [.01], [None]
    monkeypatch.setattr(backend, '_simulate_structural', structural)
    monkeypatch.setattr(module, 'evaluate_place_stability', evaluate)
    result = backend.place(State([10], [], {}, [], []), Action(0, pose(), pose(), object_id=10))
    assert result.diagnostics['place_robustness_evaluated']
    assert result.diagnostics['place_robustness_model'] == 'velocity_disturbance'


@pytest.mark.parametrize('hops,expected', [
    (None, {1, 2, 3, 4, 5}), (0, {1}), (1, {1, 2}),
    (2, {1, 2, 3}), (3, {1, 2, 3, 4}), (100, {1, 2, 3, 4}),
])
def test_contact_neighborhood_excludes_ground_and_disconnected_stones(hops, expected):
    from stacking_common.environment.robustness import robustness_dynamic_entities
    backend = NS(place_stability=PlaceStabilityConfig(contact_hops=hops),
                 core=NS(Mobility=NS(DYNAMIC='dynamic')),
                 scene_builder=NS(entity_id=lambda index: index + 1))
    scene = NS(bodies=[NS(id=i, mobility='dynamic' if i else 'static') for i in range(6)])
    # Reversed edges and a cycle; the floor must not join disconnected stone 5.
    contacts = [dict(entity_first=a, entity_second=b)
                for a, b in [(2, 1), (2, 3), (4, 3), (3, 2), (0, 1), (0, 5)]]
    assert robustness_dynamic_entities(backend, scene, NS(object_index=0), contacts) == expected
    if hops is not None:
        assert robustness_dynamic_entities(backend, scene, NS(object_index=0), []) == {1}


def test_old_robustness_profiles_preserve_full_scene_mobility():
    assert PlaceStabilityConfig.from_mapping({'enabled': True, 'n_noise': 4}).contact_hops is None
    assert PlaceStabilityConfig.from_mapping({'contact_hops': None}).contact_hops is None


def test_local_trial_freezes_only_distant_stones_without_changing_nominal_scene():
    from stacking_common.environment.robustness import disturbance_scene, robustness_dynamic_entities
    backend = make_backend(PlaceStabilityConfig(True, 2, contact_hops=2))
    backend.reset([10])
    core = backend.core
    snapshot = core.SceneSnapshot(0, [core.BodyInstance(i, backend.models[10], pose(i * .2),
        linear=np.ones(3), angular=np.ones(3),
        mobility=core.Mobility.DYNAMIC if i else core.Mobility.STATIC) for i in range(5)])
    contacts = [dict(entity_first=i, entity_second=i + 1) for i in range(4)]
    action = Action(0, pose(), pose(), object_id=10)
    dynamic = robustness_dynamic_entities(backend, snapshot, action, contacts)
    assert dynamic == {1, 2, 3}
    v, w = np.array([.1, -.2, .3]), np.array([-.3, .2, .1])
    scene = disturbance_scene(backend, snapshot, action, v, w, dynamic)
    for before, after in zip(snapshot.bodies, scene.bodies):
        assert after.mobility == (core.Mobility.DYNAMIC if before.id in dynamic else core.Mobility.STATIC)
        assert before.mobility == (core.Mobility.DYNAMIC if before.id else core.Mobility.STATIC)
        np.testing.assert_array_equal(before.pose, after.pose)
        np.testing.assert_array_equal(before.linear, np.ones(3))
        np.testing.assert_array_equal(after.linear, v if before.id == 1 else np.zeros(3))
        np.testing.assert_array_equal(after.angular, w if before.id == 1 else np.zeros(3))


def test_excavator_profile_uses_eight_weight_normalized_force_probes():
    from stacking_common.environment.robustness import disturbance_forces
    config = load_environment_profile(ROOT / 'apps/excavator/config/environment.yml').place_stability
    assert config.active and config.n_noise == 8 and config.contact_hops is None
    assert config.force_ratio == .05 and config.force_duration == 1.
    backend = make_backend(config)
    backend.reset([10])
    source = State([10], [], {}, [], [])
    action = Action(0, pose(), pose(), object_id=10)
    snapshot = backend._scene(source, action, [0], {0})
    forces = np.array([disturbance_forces(backend, snapshot, action, n)[1] for n in range(8)])
    np.testing.assert_allclose(np.linalg.norm(forces, axis=1), .05 * 9.81)
    np.testing.assert_array_equal(forces[:, 2], 0.)
    np.testing.assert_allclose(forces[4:], -forces[:4], atol=1e-15)
    np.testing.assert_array_equal(list(config.velocity_disturbances(step=1, object_index=0, object_id=10)), 0.)
    from dataclasses import asdict
    assert PlaceStabilityConfig.from_mapping(asdict(config)) == config
    backend.close()


def test_force_pulse_rounds_up_and_requires_recovery_time():
    from stacking_common.environment.robustness import disturbance_phase, disturbance_force_steps
    backend = make_backend(PlaceStabilityConfig(True, 1, force_ratio=.05, force_duration=.011))
    backend.reset([10])
    phase = disturbance_phase(backend, State([10], [], {}, [], []))
    assert phase.dt == .02
    assert disturbance_force_steps(backend.place_stability, phase.dt) == 1
    assert phase.min_time == .02
    backend.place_stability = replace(backend.place_stability, force_duration=.1)
    with pytest.raises(ValueError, match='leave time for recovery'):
        disturbance_phase(backend, State([10], [], {}, [], []))
    backend.close()
