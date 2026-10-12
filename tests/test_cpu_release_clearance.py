"""Native CPU release queries use the same DSF shapes as placement physics."""

from types import SimpleNamespace

import numpy as np
import pytest

from stacking_common.domain import Action, State
from stacking_common.environment.core_backend import (
    CorePlacementBackend, CorePlacementConfig, SimulationPhaseConfig,
)
from stacking_common.environment._cpu_release_clearance import CpuReleaseQueries


def pose(x=0., z=.1):
    return np.array([x, 0., z, 0., 0., 0., 1.])


@pytest.fixture
def backend():
    core = pytest.importorskip('stacking_core')
    # The positive-power DSF of the six axis nodes, with p=2, is a sphere.
    models = {i: core.BodyModel.dsf(i, np.r_[np.eye(3), -np.eye(3)] * .1,
                                  mass=1., inertia=np.eye(3), sharpness=2)
              for i in (20, 10)}
    backend = CorePlacementBackend(models, CorePlacementConfig(
        structural=SimulationPhaseConfig(.01, 1., 1., 0., .1, 1),
        settle_dt=.01, settle_steps=2, contact_gap_tolerance=.01,
        minimum_support_count=1,
    ))
    backend.reset([20, 10])
    yield backend
    backend.close()


def test_native_ground_gaps_and_input_isolation(backend):
    state = State([20, 10], [], {}, [], [])
    action = Action(1, pose(z=.3), pose(z=.4), object_id=10)
    noisy = pose(z=.08)
    query = CpuReleaseQueries(backend)
    assert query.simulator is not backend.simulator
    contacts, errors = query.query([(state, action)] * 2, [noisy, pose(z=.2)])
    assert errors == {}
    np.testing.assert_allclose(contacts, [(0, -.02, 1.)], atol=1e-12)
    np.testing.assert_array_equal(action.pose, pose(z=.3))
    np.testing.assert_array_equal(action.init_pose, pose(z=.4))
    np.testing.assert_array_equal(noisy, pose(z=.08))
    assert state.placed_object_indices == [] and state.object_poses == {}


@pytest.mark.parametrize('x,z,derivative', [(.18, .1, 0.), (.108, .244, .8)])
def test_native_neighbor_gap_and_vertical_derivative(backend, x, z, derivative):
    previous = pose()
    state = State([20, 10], [0], {20: previous.copy()}, [], [])
    action = Action(1, pose(x, z), pose(x, z), object_id=10)
    contacts, errors = CpuReleaseQueries(backend).query([(state, action)], [action.pose])
    assert errors == {}
    neighbor = [row for row in contacts if row[1] < -.001]
    np.testing.assert_allclose(neighbor, [(0, -.02, derivative)], atol=1e-12)
    np.testing.assert_array_equal(state.object_poses[20], previous)


@pytest.mark.parametrize('bad', ['nonfinite', 'nonstationary', 'exception'])
def test_invalid_native_query_is_isolated_to_its_trial(backend, bad):
    state = State([20, 10], [], {}, [], [])
    action = Action(1, pose(z=.08), pose(), object_id=10)
    query = CpuReleaseQueries(backend)
    native = query.simulator
    calls = 0

    def step(scene, dt):
        nonlocal calls
        calls += 1
        if calls == 1 and bad == 'exception':
            raise RuntimeError('native direction optimizer failed')
        contacts = native.step(scene, dt).contacts
        if calls == 1:
            if bad == 'nonfinite':
                contacts[0]['gap'] = np.nan
            else:
                contacts[0]['point_second'][0] += .01
        return SimpleNamespace(contacts=contacts)

    query.simulator = SimpleNamespace(step=step, clear_warm_start=native.clear_warm_start)
    contacts, errors = query.query([(state, action)] * 2, [action.pose] * 2)
    assert errors == {0: 'initial_clearance_query_failure'}
    np.testing.assert_allclose(contacts, [(1, -.02, 1.)], atol=1e-12)
