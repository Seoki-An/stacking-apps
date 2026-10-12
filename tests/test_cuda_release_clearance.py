"""CUDA release correction matches CPU geometry and preserves trial bookkeeping."""

import numpy as np
import pytest

from stacking_common.domain import Action, State
from stacking_common.environment.core_backend import (
    CorePlacementBackend, CorePlacementConfig, SimulationPhaseConfig,
)
from stacking_common.environment.cuda_backend import CudaPlacementBackend
from stacking_common.environment.release_clearance import ReleaseTrial
from stacking_common.environment.robustness import PlaceStabilityConfig


IDS = [10, 20, 30]


def pose(x=0., z=.1):
    return np.array([x, 0., z, 0., 0., 0., 1.])


class SmallCudaBackend(CudaPlacementBackend):
    batch_size = 8


@pytest.fixture
def backends():
    core = pytest.importorskip('stacking_core')
    info = core.cuda_capabilities()
    if not info.get('simulation_solve') or not info.get('devices'):
        pytest.skip('requires native CUDA geometry and simulation with a visible GPU')
    nodes = np.r_[np.eye(3), -np.eye(3)] * .1
    models = {i: core.BodyModel.dsf(i, nodes, mass=1., inertia=np.eye(3), sharpness=2)
              for i in IDS[:2]}
    models[30] = core.BodyModel.dsf_components(
        30, [nodes + [.2, 0., .04], nodes - [.2, 0., .04]],
        mass=1., inertia=np.eye(3), frictions=[1., 1.], sharpnesses=[2, 2])
    phase = SimulationPhaseConfig(.01, 1e-5, .2, .025, .15, 2)
    config = CorePlacementConfig(phase, .01, 2, .02, 1)
    physics = core.SimulationConfig()
    physics.gravity = np.zeros(3)
    stability = PlaceStabilityConfig(enabled=True, n_noise=3)
    cpu = CorePlacementBackend(models, config, physics, place_stability=stability)
    gpu = SmallCudaBackend(models, config, physics, place_stability=stability)
    for backend in (cpu, gpu):
        backend.reset(IDS)
    try:
        yield cpu, gpu
    finally:
        cpu.close()
        gpu.close()


def test_native_ground_neighbor_lateral_and_component_corrections_match_cpu(backends):
    cpu, gpu = backends
    empty = State(IDS, [], {}, [], [])
    occupied = State(IDS, [0], {10: pose()}, [], [])
    cases = [
        (empty, 0, pose(z=.08), pose(), None),
        (occupied, 1, pose(z=.28), pose(z=.3), None),
        (occupied, 1, pose(x=.108, z=.244), pose(x=.12, z=.26), None),
        (occupied, 1, pose(x=.18), pose(x=.2), 'initial_clearance_lateral_contact'),
        (empty, 2, pose(z=.12), pose(z=.14), None),
    ]
    rows = [(state, Action(index, noisy.copy(), base.copy(), object_id=IDS[index]))
            for state, index, noisy, base, _ in cases]
    noisy = [case[2] for case in cases]
    bases = [case[3] for case in cases]
    original = [p.copy() for p in noisy]
    expected = cpu._release_clearance_checker().correct(rows, noisy, bases)
    actual = gpu._release_clearance_checker().correct(rows, noisy, bases)
    assert [trial.failure_reason for trial in expected] == [case[4] for case in cases]
    assert [trial.failure_reason for trial in actual] == [case[4] for case in cases]
    for result, reference, start, (_, action) in zip(actual, expected, original, rows):
        np.testing.assert_allclose(result.pose, reference.pose, atol=2e-6)
        assert result.lift == pytest.approx(reference.lift, abs=2e-6)
        np.testing.assert_array_equal(result.pose[[0, 1, 3, 4, 5, 6]],
                                      start[[0, 1, 3, 4, 5, 6]])
        np.testing.assert_array_equal(action.pose, start)
    np.testing.assert_array_equal(noisy, original)
    np.testing.assert_array_equal(occupied.object_poses[10], pose())
    assert actual[0].pose[2] == pytest.approx(.1001, abs=2e-6)
    assert actual[1].pose[2] == pytest.approx(.3001, abs=2e-6)
    assert actual[4].pose[2] == pytest.approx(.1401, abs=2e-6)


@pytest.mark.parametrize('all_invalid', [False, True])
def test_cuda_robustness_preserves_trial_slots_and_corrected_reference(backends, monkeypatch, all_invalid):
    import stacking_common.environment.release_clearance as clearance

    _, gpu = backends
    state = State(IDS, [], {}, [], [])
    action = Action(0, pose(z=.1), pose(z=.15), object_id=10)
    snapshot = gpu._scene(state, action, [0], {0})
    groups = []
    for job in range(2):
        groups.append([
            ReleaseTrial(pose(x=.1 * job, z=.2 + .1 * number),
                         'initial_clearance_lateral_contact'
                         if all_invalid or (job + number) % 2 else None,
                         lift=.05)
            for number in range(3)])
    monkeypatch.setattr(clearance, 'prepare_release_trials', lambda backend, jobs: groups)
    native_run = gpu._run
    starts = []

    def run(problems):
        assert not all_invalid, 'invalid initial poses must not reach GPU simulation'
        entity = gpu.scene_builder.entity_id(action.object_index)
        starts.extend(problem.scene.body_pose(entity) for problem in problems)
        return native_run(problems)

    monkeypatch.setattr(gpu, '_run', run)
    results = gpu._robustness([(state, action, snapshot)] * 2)
    expected_starts = [trial.pose for group in groups for trial in group
                       if trial.failure_reason is None]
    np.testing.assert_array_equal(starts, expected_starts)
    for (_, displacements, reasons), group in zip(results, groups):
        assert len(displacements) == len(reasons) == 3
        for displacement, reason, release in zip(displacements, reasons, group):
            assert reason == release.failure_reason
            if release.failure_reason:
                assert displacement == np.inf
            else:
                # No gravity: the repaired release pose remains stationary.
                # Measuring from the old uncorrected pose would include lift.
                assert displacement == pytest.approx(0., abs=1e-7)
    np.testing.assert_array_equal(action.pose, pose())
    np.testing.assert_array_equal(snapshot.body_pose(1), pose())
