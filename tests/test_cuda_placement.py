"""GPU placement phases and batched transition bookkeeping."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import stacking_core as core

from stacking_common.domain import Action, State
from stacking_common.environment import SimulationTransition, PlacementResult
from stacking_common.environment.core_backend import (
    CorePlacementBackend, CorePlacementConfig, SimulationPhaseConfig,
)
from stacking_common.environment.cuda_backend import CudaPlacementBackend, _steps_after
from stacking_common.environment.robustness import PlaceStabilityConfig


def pose(x=0., z=1.):
    return np.array([x, 0., z, 0., 0., 0., 1.])


def state(placed=False):
    return State([10, 20], [0] if placed else [], {10: pose()} if placed else {}, [], [])


@pytest.fixture
def backends(request):
    if not core.cuda_capabilities().get('simulation_solve') or not core.cuda_capabilities().get('devices'):
        pytest.skip('requires core GPU simulation and device')
    models = {i: core.BodyModel.dsf(i, np.concatenate([np.eye(3), -np.eye(3)]) * .05,
                                  mass=1., inertia=np.eye(3), sharpness=2) for i in (10, 20)}
    short = SimulationPhaseConfig(.01, 1e-5, .2, .025, .15, 2, final_extra_steps=3)
    long = replace(short, min_time=.055, max_time=.2, extra_steps=4, final_extra_steps=6)
    config = CorePlacementConfig(long, .01, 4, .02, 1,
                                 short_gate=short if getattr(request, "param", True) else None)
    physics = core.SimulationConfig()
    physics.gravity = np.zeros(3)
    gpu = CudaPlacementBackend(models, config, physics, settling_simulation_config=physics)
    cpu = CorePlacementBackend(models, config, physics, settling_simulation_config=physics)
    for backend in (cpu, gpu):
        backend.reset([10, 20])
    try:
        yield cpu, gpu
    finally:
        cpu.close()
        gpu.close()


def test_phase_time_bounds_are_strict():
    assert _steps_after(.5, .01) == 51
    assert _steps_after(.505, .01) == 51
    assert _steps_after(0., .01) == 1


def test_compact_nominal_histories_preserve_all_stone_trajectories(backends, monkeypatch):
    _, gpu = backends
    source = state(True)
    action = Action(1, pose(x=.5), pose(x=.5), object_id=20)
    settled, = gpu._settling((source,), (action,))
    assert {body.id for body in settled.samples[0].snapshot.bodies} == {2}
    assert len(settled.snapshot.bodies) == 3  # Both stones and ground remain in physics.
    phase = gpu._phase(gpu.config.structural, [0, 1], final_object=True)
    assert phase.sampled_entities == [1, 2]
    actual = gpu.place(source, action)
    run = gpu._run

    def full_histories(problems):
        for problem in problems:
            phases = problem.phases
            for phase in phases:
                phase.sampled_entities = None
            problem.phases = phases
        return run(problems)

    monkeypatch.setattr(gpu, '_run', full_histories)
    expected = gpu.place(source, action)
    assert (actual.failed, actual.settled) == (expected.failed, expected.settled)
    for object_id in actual.object_poses:
        np.testing.assert_array_equal(actual.object_poses[object_id], expected.object_poses[object_id])
        np.testing.assert_array_equal(actual.trajectories[object_id].poses,
                                      expected.trajectories[object_id].poses)
        np.testing.assert_array_equal(actual.trajectories[object_id].times,
                                      expected.trajectories[object_id].times)


@pytest.mark.parametrize("backends", [False, True], indirect=True,
                         ids=["unified", "legacy_staged"])
def test_native_batch_matches_scalar_cpu_and_preserves_history(backends):
    cpu, gpu = backends
    states = (state(), state(True))
    actions = (Action(0, pose(), pose(), object_id=10), Action(1, pose(x=.5), pose(x=.5), object_id=20))
    actual = gpu.place_batch(states, actions)
    expected = [cpu.place(s, a) for s, a in zip(states, actions)]
    for result, reference in zip(actual, expected):
        assert result.failed == reference.failed
        assert result.settled == reference.settled
        assert result.diagnostics['simulation_backend'] == 'cuda'
        assert result.diagnostics['simulation_stage_times'] == pytest.approx(reference.diagnostics['simulation_stage_times'])
        phase = 'long_extension' if gpu.config.short_gate else 'structural'
        assert result.diagnostics['simulation_stage_times'][phase] > 0
        for object_id in result.object_poses:
            np.testing.assert_allclose(result.object_poses[object_id], reference.object_poses[object_id], atol=1e-6)
            np.testing.assert_allclose(result.trajectories[object_id].times, reference.trajectories[object_id].times, atol=1e-6)
    assert states[0].placed_object_indices == []
    assert states[1].placed_object_indices == [0]
    reordered = gpu.place_batch(states[::-1], actions[::-1])
    for a, b in zip(actual, reordered[::-1]):
        for object_id in a.object_poses:
            np.testing.assert_array_equal(a.object_poses[object_id], b.object_poses[object_id])
    assert gpu.place_batch((), ()) == ()


def test_capacity_failure_isolated_without_cpu_simulation(backends, monkeypatch):
    cpu, gpu = backends
    cap = core.CudaCapacity.small(2)
    cap.bodies = 2  # One stone plus ground fits; two stones do not.
    gpu.cuda_simulator = core.CudaSimulator(cap)
    monkeypatch.setattr(CorePlacementBackend, 'place', lambda *args: pytest.fail('unexpected CPU retry'))
    outputs = gpu.place_batch((state(), state(True)), (
        Action(0, pose(), pose(), object_id=10), Action(1, pose(.5), pose(.5), object_id=20)))
    assert not outputs[0].failed
    assert outputs[1].failed and not outputs[1].settled
    assert outputs[1].diagnostics['rejection_reason'] == 'simulation_cuda_capacity_overflow'
    assert outputs[1].diagnostics['simulation_cuda_stage'] == 'settling'


def test_fixed_settling_preserves_placed_poses(backends):
    _, gpu = backends
    source = state(True)
    source.pose_identified_object_ids.add(10)
    result = gpu.settle_fixed(source, Action(1, pose(.5), pose(.5), object_id=20))
    assert not result.failed
    np.testing.assert_array_equal(result.object_poses[10], source.object_poses[10])
    assert result.diagnostics['simulation_backend'] == 'cuda'
    assert result.diagnostics['fixed_scene_resettle_steps'] == 4


@pytest.mark.parametrize('directional', [False, True])
def test_velocity_disturbance_cpu_cuda_parity(backends, directional):
    cpu, gpu = backends
    velocities = ({'linear_velocity_magnitude': (.001, .001, .001)} if directional
                  else {'linear_velocity_std': (.001, .001, .001), 'seed': 4})
    config = PlaceStabilityConfig(enabled=True, n_noise=4, **velocities)
    cpu.place_stability = gpu.place_stability = config
    action = Action(0, pose(), pose(), object_id=10)
    actual, reference = gpu.place(state(), action), cpu.place(state(), action)
    assert actual.diagnostics['place_robustness_enabled']
    assert len(actual.diagnostics['place_robustness_trial_displacements']) == 4
    assert actual.diagnostics['place_robustness_trial_failure_reasons'] == reference.diagnostics['place_robustness_trial_failure_reasons']
    assert actual.place_robustness_displacement == pytest.approx(reference.place_robustness_displacement, abs=2e-7)


def test_falling_disturbance_trial_exceeds_excursion_limit_on_cpu_and_cuda(backends):
    from stacking_common.environment.robustness import _simulate_disturbance
    cpu, gpu = backends
    policy = PlaceStabilityConfig(True, 1, fail_displacement=.02, reject_failed_trials=True,
        settling_position_tolerance=.002, settling_rotation_tolerance_deg=.5)
    action = Action(0, pose(z=.4), pose(z=.4), object_id=10)
    physics = core.SimulationConfig()
    physics.gravity = np.array([0., 0., -9.81])
    for backend in (cpu, gpu):
        backend.place_stability = policy
        backend.simulation_config = physics
    snapshot = cpu._scene(state(), action, [0], dynamic_indices={0})
    peak, reason = _simulate_disturbance(cpu, state(), action, snapshot, np.zeros(3), np.zeros(3), {1})
    assert reason == 'excursion_limit' and peak > .02
    # Robustness normally starts from the final structural snapshot. This
    # deliberately raised snapshot exercises a known falling release instead.
    snapshot = gpu._scene(state(), action, [0], dynamic_indices={0})
    maximum, displacements, reasons = gpu._robustness([(state(), action, snapshot, ())])[0]
    assert reasons == ['excursion_limit'] and maximum > .02
    assert policy.failed(maximum, reasons)


def test_missing_capability_fails_at_construction(monkeypatch):
    monkeypatch.setattr(CorePlacementBackend, '__init__', lambda self, *args, **kwargs: setattr(
        self, 'core', SimpleNamespace(cuda_capabilities=lambda: {'simulation_solve': False})))
    with pytest.raises(RuntimeError, match='GPU simulation'):
        CudaPlacementBackend()


def test_transition_batch_keeps_branches_independent_and_rejections_visible():
    class Backend:
        def reset(self, ids):
            pass
        def restore(self, source):
            pass
        def place_batch(self, states, actions):
            assert [a.object_id for a in actions] == [10, 20]
            return [PlacementResult({10: pose()}, diagnostics={'simulation_backend': 'cuda'}),
                    PlacementResult({20: pose(.5)}, failed=True, settled=False,
                                    diagnostics={'rejection_reason': 'simulation_cuda_geometry_failure'})]
    transition = SimulationTransition([10, 20], Backend())
    transition.reset()
    sources = (state(), state(True))
    actions = (Action(0, pose(), pose()), Action(1, pose(.5), pose(.5)))
    outputs = transition.step_batch(sources, actions)
    assert [s.placed_object_indices for s in outputs] == [[0], [0, 1]]
    assert not outputs[0].failed and outputs[1].failed
    assert outputs[1].action_history[-1].diagnostics['rejection_reason'] == 'simulation_cuda_geometry_failure'
    assert actions[0].object_id == -1
    assert sources[0].placed_object_indices == []
    assert sources[1].placed_object_indices == [0]


def test_bad_gpu_batch_ids_raise(backends, monkeypatch):
    _, gpu = backends
    gpu.cuda_simulator = SimpleNamespace(run=lambda items: [])
    with pytest.raises(RuntimeError, match='request IDs'):
        gpu.place(state(), Action(0, pose(), pose(), object_id=10))


def test_motion_limit_rejects_without_running_long_extension(backends):
    _, gpu = backends
    physics = core.SimulationConfig()
    gpu.simulation_config = physics
    short = replace(gpu.config.short_gate, velocity_integral_threshold=.0001)
    gpu.config = replace(gpu.config, short_gate=short)
    result = gpu.place(state(), Action(0, pose(), pose(), object_id=10))
    assert result.failed
    assert result.diagnostics['rejection_reason'] == 'target_motion_limit'
    assert result.diagnostics['simulation_stage_times']['long_extension'] == 0.
    assert result.diagnostics['simulation_cuda_phase_statuses'] == ('motion_limit',)


def test_identified_stones_remain_static_under_gravity(backends):
    _, gpu = backends
    gpu.simulation_config = core.SimulationConfig()
    source = state(True)
    source.pose_identified_object_ids.add(10)
    result = gpu.place(source, Action(1, pose(.5), pose(.5), object_id=20))
    np.testing.assert_array_equal(result.object_poses[10], source.object_poses[10])
    assert result.trajectories[10].vel_integral == 0.


def test_evaluator_uses_one_gpu_batch_for_multiple_search_branches(backends):
    from stacking_common.environment import StackingEnvironment
    from stacking_common.search import StackingEnvironmentEvaluator, EvaluationFidelity
    _, gpu = backends
    environment = StackingEnvironment(SimulationTransition([10, 20], gpu),
        SimpleNamespace(build=lambda s: len(s.placed_object_indices)),
        SimpleNamespace(compute=lambda s: (float(len(s.placed_object_indices)), {})))
    evaluator = StackingEnvironmentEvaluator({EvaluationFidelity.LONG: environment})
    calls = []
    original = gpu.place_batch
    def record(states, actions):
        calls.append(len(states))
        return original(states, actions)
    gpu.place_batch = record
    results = evaluator.evaluate_batch((state(), state()), (
        Action(0, pose(), pose()), Action(1, pose(.5), pose(.5))), EvaluationFidelity.LONG)
    assert calls == [2]
    assert evaluator.worker_count == gpu.batch_size
    assert [r.state.placed_object_indices for r in results] == [[0], [1]]
    assert [r.reward for r in results] == [1., 1.]
    assert all(r.diagnostics['simulation_backend'] == 'cuda' for r in results)



@pytest.mark.parametrize('placement_size', [1, 2, 8])
def test_streamed_gpu_results_match_serial_results(backends, placement_size):
    from stacking_common.environment import StackingEnvironment

    _, gpu = backends
    gpu.placement_batch_size = placement_size

    def evaluators():
        return (SimpleNamespace(build=lambda s: tuple(s.placed_object_indices)),
                SimpleNamespace(compute=lambda s: (float(len(s.placed_object_indices)), {})))

    environment = StackingEnvironment(SimulationTransition([10, 20], gpu),
                                     *evaluators(), worker_evaluators=(evaluators(),))
    environment.result_batch_size = 1
    sources = (state(), state(True), state())
    actions = (Action(0, pose(), pose()), Action(1, pose(.5), pose(.5)),
               Action(1, pose(.5), pose(.5)))
    try:
        expected = environment._step_results(environment.transition.step_batch(sources, actions))
        actual = environment.step_batch(sources, actions)
        assert (environment._pipeline_executor is not None) == (placement_size < len(sources))
        for result, reference in zip(actual, expected):
            assert result.reward == reference.reward
            assert result.observation == reference.observation
            assert result.state.failed == reference.state.failed
            assert result.state.terminated == reference.state.terminated
            assert result.state.simulation_settled == reference.state.simulation_settled
            for object_id, value in reference.state.object_poses.items():
                np.testing.assert_array_equal(result.state.object_poses[object_id], value)
            for trajectories, originals in zip(result.state.trajectories, reference.state.trajectories):
                for object_id, trajectory in originals.items():
                    np.testing.assert_array_equal(trajectories[object_id].poses, trajectory.poses)
                    assert trajectories[object_id].times == trajectory.times
    finally:
        environment.close()



def test_robustness_monitors_all_dynamic_entities_without_cumulative_cutoff(backends):
    _, gpu = backends
    phase = gpu._robustness_phase(gpu.config.structural, [1, 2])
    assert phase.monitored == [1, 2]
    assert phase.motion_limit == 0 and phase.sample_stride == 1
    assert phase.robustness is None


def test_bulk_placement_samples_match_snapshot_path(backends, monkeypatch):
    _, gpu = backends
    if not hasattr(core.CudaSimulationResult, 'sample_poses'):
        pytest.skip('requires bulk sample bindings')
    gpu.place_stability = PlaceStabilityConfig(enabled=True, n_noise=2,
        linear_velocity_std=(.001, .001, .001), seed=4)
    action = Action(1, pose(.5), pose(.5), object_id=20)
    actual = gpu.place(state(True), action)
    for trajectory in actual.trajectories.values():
        assert '_packed_poses' in vars(trajectory)
    original = gpu._run

    class SnapshotOnly:
        def __init__(self, value):
            self.value = value
        def __getattr__(self, name):
            if name in ('sample_poses', 'sample_times'):
                raise AttributeError(name)
            return getattr(self.value, name)

    monkeypatch.setattr(gpu, '_run', lambda problems: [SnapshotOnly(out) for out in original(problems)])
    expected = gpu.place(state(True), action)
    assert actual.failed == expected.failed
    assert actual.settled == expected.settled
    assert actual.place_robustness_displacement == expected.place_robustness_displacement
    for object_id, trajectory in actual.trajectories.items():
        reference = expected.trajectories[object_id]
        np.testing.assert_array_equal(actual.object_poses[object_id], expected.object_poses[object_id])
        np.testing.assert_array_equal(trajectory.poses, reference.poses)
        assert trajectory.times == reference.times
        for name in ('energy', 'vel_integral', 'settle_pose_count', 'settle_position_delta', 'settle_path_length'):
            assert getattr(trajectory, name) == getattr(reference, name)


def test_support_motion_is_checked_on_cpu_and_cuda(backends):
    from stacking_common.environment.robustness import evaluate_place_stability
    cpu, gpu = backends
    # Target is fixed in this synthetic scene to isolate a falling support's
    # excursion. A real trial keeps the new stone dynamic too.
    action = Action(1, pose(x=2., z=.4), pose(x=2., z=.4), object_id=20)
    source = state(True)
    physics = core.SimulationConfig()
    physics.gravity = np.array([0., 0., -9.81])
    for backend in (cpu, gpu):
        backend.simulation_config = physics
        backend.place_stability = PlaceStabilityConfig(True, 1, fail_displacement=.02)
    snapshot = core.SceneSnapshot(0, [
        core.BodyInstance(1, cpu.models[10], pose(z=.4)),
        core.BodyInstance(2, cpu.models[20], action.pose, mobility=core.Mobility.STATIC),
    ])
    expected = evaluate_place_stability(cpu, source, action, snapshot)
    actual = gpu._robustness([(source, action, snapshot, ())])[0]
    for maximum, values, reasons in (actual, expected):
        assert maximum > .02 and reasons == ['excursion_limit']


def test_cuda_nominal_failure_skips_disturbances(backends, monkeypatch):
    _, gpu = backends
    gpu.place_stability = PlaceStabilityConfig(True, 2)
    gpu.simulation_config = core.SimulationConfig()
    gpu.config = replace(gpu.config,
        short_gate=replace(gpu.config.short_gate, velocity_integral_threshold=.0001))
    original = gpu._robustness
    def capture(jobs):
        assert jobs == []
        return original(jobs)
    monkeypatch.setattr(gpu, '_robustness', capture)
    result = gpu.place(state(), Action(0, pose(), pose(), object_id=10))
    assert result.diagnostics['rejection_reason'] == 'target_motion_limit'
    assert not result.diagnostics['place_robustness_evaluated']


@pytest.mark.parametrize('hops,expected', [(None, {1, 2, 3, 4, 5}), (0, {1}), (2, {1, 2, 3})])
def test_cpu_cuda_trials_use_same_contact_neighborhood(backends, monkeypatch, hops, expected):
    import stacking_common.environment.robustness as robustness
    import stacking_common.environment.cuda_backend as cuda
    cpu, gpu = backends
    source = state()
    action = Action(0, pose(), pose(), object_id=10)
    snapshot = core.SceneSnapshot(0, [
        core.BodyInstance(i, cpu.models[10], pose(x=float(i))) for i in range(1, 6)])
    contacts = [dict(entity_first=i, entity_second=i + 1) for i in range(1, 4)]
    observed = []
    original = robustness.disturbance_scene
    def capture(*args):
        scene = original(*args)
        observed.append({b.id for b in scene.bodies if b.mobility == core.Mobility.DYNAMIC})
        return scene
    monkeypatch.setattr(robustness, 'disturbance_scene', capture)
    monkeypatch.setattr(cuda, 'disturbance_scene', capture)
    for backend in (cpu, gpu):
        backend.place_stability = PlaceStabilityConfig(True, 2, contact_hops=hops)
    cpu_result = robustness.evaluate_place_stability(cpu, source, action, snapshot, contacts)
    gpu_result = gpu._robustness([(source, action, snapshot, contacts)])[0]
    assert observed == [expected] * 4
    assert cpu_result[2] == gpu_result[2] == [None, None]
    assert cpu_result[0] == pytest.approx(gpu_result[0], abs=2e-7)
    assert all(b.mobility == core.Mobility.DYNAMIC for b in snapshot.bodies)


def test_cuda_robustness_receives_final_nominal_contacts(backends, monkeypatch):
    _, gpu = backends
    gpu.place_stability = PlaceStabilityConfig(True, 1, contact_hops=2)
    structural, evaluate = gpu._structural_result, gpu._robustness
    results, received = [], []
    def capture_structural(*args):
        result = structural(*args)
        results.append(result)
        return result
    def capture_robustness(jobs):
        assert len(jobs) == 1
        assert jobs[0][2] is results[0].snapshot
        assert jobs[0][3] is results[0].contacts
        received.extend(jobs)
        return evaluate(jobs)
    monkeypatch.setattr(gpu, '_structural_result', capture_structural)
    monkeypatch.setattr(gpu, '_robustness', capture_robustness)
    result = gpu.place(state(), Action(0, pose(), pose(), object_id=10))
    assert len(received) == 1
    assert result.diagnostics['place_robustness_contact_hops'] == 2


def test_nominal_and_robustness_request_their_consumed_histories(backends):
    _, gpu = backends
    if not hasattr(core.CudaSimulationPhase, 'sampled_entities'):
        pytest.skip('requires selective native histories')
    nominal = gpu._phase(gpu.config.structural, [0, 1], final_object=False)
    assert nominal.sampled_entities == nominal.monitored == [1, 2]
    trial = gpu._robustness_phase(gpu.config.structural, {1: pose()})
    assert trial.sampled_entities == trial.monitored == [1]
    gpu._compact_robustness_history = False
    assert gpu._robustness_phase(gpu.config.structural, {1: pose()}).sampled_entities is None


def test_compact_and_full_history_paths_have_identical_trial_results(backends):
    _, gpu = backends
    if not hasattr(core.CudaSimulationPhase, 'sampled_entities'):
        pytest.skip('requires selective native histories')
    gpu.place_stability = PlaceStabilityConfig(True, 8, (.001, .001, .001),
                                               contact_hops=0)
    action = Action(0, pose(), pose(), object_id=10)
    compact = gpu.place(state(), action)
    gpu._compact_robustness_history = False
    full = gpu.place(state(), action)
    assert compact.failed == full.failed
    for name in ('place_robustness_trial_displacements', 'place_robustness_trial_outcomes'):
        assert compact.diagnostics[name] == full.diagnostics[name]
    for entity in full.object_poses:
        np.testing.assert_array_equal(compact.object_poses[entity], full.object_poses[entity])


@pytest.mark.parametrize('backends', [False, True], indirect=True,
                         ids=['unified', 'legacy_staged'])
def test_prepared_cuda_placements_match_serial_physics_and_all_samples(backends):
    _, gpu = backends
    gpu.place_stability = PlaceStabilityConfig(True, 2, (.001, .001, .001), contact_hops=2)
    parents = (state(), state(True))
    actions = (Action(0, pose(), pose(), object_id=10),
               Action(1, pose(x=.5), pose(x=.5), object_id=20))
    expected = gpu.place_batch(parents, actions)
    nominal = []
    actual = gpu.place_batch_with_preparation(parents, actions,
                                             lambda start, poses: nominal.extend(poses))
    assert len(nominal) == len(actual)
    for reference, result, poses in zip(expected, actual, nominal):
        assert (result.failed, result.settled) == (reference.failed, reference.settled)
        assert result.place_robustness_displacement == reference.place_robustness_displacement
        assert {k: v for k, v in result.diagnostics.items() if 'wall_seconds' not in k} == \
               {k: v for k, v in reference.diagnostics.items() if 'wall_seconds' not in k}
        for oid, value in result.object_poses.items():
            np.testing.assert_array_equal(value, reference.object_poses[oid])
            np.testing.assert_array_equal(value, poses[oid])
        for oid, trajectory in result.trajectories.items():
            other = reference.trajectories[oid]
            np.testing.assert_array_equal(trajectory.poses, other.poses)
            np.testing.assert_array_equal(trajectory.times, other.times)
            assert trajectory.energy == other.energy
            assert trajectory.vel_integral == other.vel_integral
            assert trajectory.settle_path_length == other.settle_path_length


def test_cuda_preparation_failure_drains_native_worker(backends, monkeypatch):
    from threading import Event
    from concurrent.futures import ThreadPoolExecutor

    _, gpu = backends
    entered, release, finished = Event(), Event(), Event()
    original = gpu._robustness

    def blocked(jobs):
        entered.set()
        assert release.wait(3)
        try:
            return original(jobs)
        finally:
            finished.set()

    def broken(*_):
        assert entered.wait(3)
        raise ValueError('preparation failed')

    monkeypatch.setattr(gpu, '_robustness', blocked)
    with ThreadPoolExecutor(max_workers=1) as caller:
        task = caller.submit(gpu.place_batch_with_preparation, (state(),),
                             (Action(0, pose(), pose(), object_id=10),), broken)
        try:
            assert entered.wait(3)
            assert not task.done()
            release.set()
            with pytest.raises(ValueError, match='preparation failed'):
                task.result(timeout=5)
            assert finished.is_set()
        finally:
            release.set()


def test_force_robustness_matches_cpu_and_preserves_nominal_scene(backends, monkeypatch):
    cpu, gpu = backends
    from stacking_common.environment.robustness import evaluate_place_stability
    cfg = PlaceStabilityConfig(True, 4, force_ratio=.05, force_duration=.04,
        rotation_weight=.01, fail_displacement=1., reject_failed_trials=True)
    physics = core.SimulationConfig()
    physics.gravity = np.array([0., 0., -9.81])
    source = state()
    action = Action(0, pose(z=10), pose(z=10), object_id=10)
    for backend in (cpu, gpu):
        backend.place_stability = cfg
        backend.simulation_config = physics
    snapshot = gpu._scene(source, action, [0], {0})
    original = np.array(snapshot.body_pose(1))
    expected = evaluate_place_stability(cpu, source, action, snapshot, ())
    import stacking_common.environment.cuda_backend as cuda_backend
    build_scene = cuda_backend.disturbance_scene
    scene_calls = []
    def record_scene(*args, **kwargs):
        scene_calls.append(1)
        return build_scene(*args, **kwargs)
    monkeypatch.setattr(cuda_backend, 'disturbance_scene', record_scene)
    actual, = gpu._robustness([(source, action, snapshot, ())])
    assert len(scene_calls) == 1  # Force directions share a resting scene, not evolving state.
    np.testing.assert_allclose(actual[1], expected[1], atol=2e-6)
    assert actual[2] == expected[2]
    np.testing.assert_array_equal(snapshot.body_pose(1), original)
    assert source.placed_object_indices == []
