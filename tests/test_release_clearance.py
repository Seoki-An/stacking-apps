"""Bounded noisy-release correction and ordered trial preparation."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stacking_common.domain import Action, State
from stacking_common.domain.geometry import quaternion_matrix
from stacking_common.environment.release_clearance import (
    LIFT_CLEARANCE, MAX_ROUNDS, ReleaseClearance, ReleaseTrial, prepare_release_trials,
)
from stacking_common.environment.robustness import PlaceStabilityConfig


def pose(z=.05):
    return np.array([0., 0., z, 0., 0., 0., 1.])


class Queries:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def query(self, rows, poses):
        self.calls.append((list(rows), [p.copy() for p in poses]))
        return self.answer(rows, poses, len(self.calls))


def setup_case(queries, *, extent=.05, tolerance=.002):
    model = NS(geometries=[dict(
        bounds_min=np.full(3, -extent), bounds_max=np.full(3, extent),
        body_from_geometry=np.array([0., 0., 0., 0., 0., 0., 1.]),
    )])
    backend = NS(models={10: model}, config=NS(contact_gap_tolerance=tolerance))
    state = State([10], [], {}, [], [])
    action = Action(0, pose(), pose(.2), object_id=10)
    return ReleaseClearance(backend, queries), state, action


def test_rotated_ground_penetration_is_repaired_and_rechecked_without_mutating_inputs():
    points = np.array([[x, y, z] for x in (-.05, .05)
                       for y in (-.05, .05) for z in (-.05, .05)])
    def ground(rows, poses, call):
        return [(i, float((points @ quaternion_matrix(p[3:]).T)[:, 2].min() + p[2]), 1.)
                for i, p in enumerate(poses)], {}
    queries = Queries(ground)
    checker, state, action = setup_case(queries)
    base = pose()
    noisy = base.copy()
    noisy[3:] = Rotation.from_euler('x', 45, degrees=True).as_quat()
    before = noisy.copy()

    trial, = checker.correct([(state, action)], [noisy], [base])

    assert trial.failure_reason is None
    assert len(queries.calls) == 2
    assert trial.lift == pytest.approx(.05 * np.sqrt(2) - .05 + LIFT_CLEARANCE)
    assert checker.last['final_valid_min_gap_m'] == pytest.approx(LIFT_CLEARANCE)
    np.testing.assert_array_equal(trial.pose[:2], noisy[:2])
    np.testing.assert_array_equal(trial.pose[3:], noisy[3:])
    np.testing.assert_array_equal(noisy, before)
    np.testing.assert_array_equal(base, pose())
    np.testing.assert_array_equal(action.pose, pose())
    assert state.object_poses == {} and state.placed_object_indices == []


def test_lift_is_not_accepted_before_rechecking_new_contacts():
    queries = Queries(lambda rows, poses, call: (
        [(0, -.002, 1.)] if call == 1 else [(0, -.01, 0.)], {}))
    checker, state, action = setup_case(queries)
    trial, = checker.correct([(state, action)], [pose(.06)], [pose()])
    assert len(queries.calls) == 2
    assert trial.lift == pytest.approx(.002 + LIFT_CLEARANCE)
    assert trial.failure_reason == 'initial_clearance_lateral_contact'


@pytest.mark.parametrize('derivative,reason', [
    (0., 'initial_clearance_lateral_contact'),
    (-1., 'initial_clearance_lateral_contact'),
    (.0001, 'initial_clearance_lift_limit'),
])
def test_lateral_or_excessive_vertical_correction_is_rejected(derivative, reason):
    queries = Queries(lambda *args: ([(0, -.002, derivative)], {}))
    checker, state, action = setup_case(queries)
    trial, = checker.correct([(state, action)], [pose()], [pose()])
    assert trial.failure_reason == reason and trial.lift == 0.
    assert len(queries.calls) == 1


def test_repeated_small_lifts_cannot_bypass_iteration_limit():
    queries = Queries(lambda *args: ([(0, -.00002, 1.)], {}))
    checker, state, action = setup_case(queries)
    trial, = checker.correct([(state, action)], [pose(.15)], [pose()])
    assert len(queries.calls) == MAX_ROUNDS
    assert trial.failure_reason == 'initial_clearance_iteration_limit'
    assert trial.lift == pytest.approx((MAX_ROUNDS - 1) * (.00002 + LIFT_CLEARANCE))


@pytest.mark.parametrize('gap,dz', [(np.nan, 1.), (np.inf, 1.), (0., np.nan), (0., np.inf)])
def test_nonfinite_query_rejects_without_lifting(gap, dz):
    queries = Queries(lambda *args: ([(0, gap, dz)], {}))
    checker, state, action = setup_case(queries)
    trial, = checker.correct([(state, action)], [pose()], [pose()])
    assert trial.failure_reason == 'initial_clearance_query_failure'
    assert trial.lift == 0.


def test_provider_error_is_preserved_and_does_not_become_lateral_failure():
    queries = Queries(lambda *args: ([(0, -.01, 0.)], {0: 'initial_clearance_query_failure'}))
    checker, state, action = setup_case(queries)
    trial, = checker.correct([(state, action)], [pose()], [pose()])
    assert trial.failure_reason == 'initial_clearance_query_failure'
    assert trial.lift == 0.


def test_no_nearby_cpu_contacts_is_a_clear_release():
    queries = Queries(lambda *args: ([], {}))
    checker, state, action = setup_case(queries)
    trial, = checker.correct([(state, action)], [pose()], [pose()])
    assert trial.failure_reason is None and trial.lift == 0.
    assert checker.last['final_valid_min_gap_m'] is None


def test_invalid_poses_and_empty_batches_do_not_query_and_keep_finite_metrics():
    queries = Queries(lambda *args: pytest.fail('invalid trial reached geometry queries'))
    checker, state, action = setup_case(queries)
    assert checker.correct([], [], []) == []
    assert checker.last['rounds'] == checker.last['queries'] == 0
    nonfinite = pose()
    nonfinite[2] = np.nan
    zero_quaternion = pose()
    zero_quaternion[3:] = 0.
    trials = checker.correct([(state, action)] * 2, [nonfinite, zero_quaternion], [pose()] * 2)
    assert [trial.failure_reason for trial in trials] == ['initial_clearance_nonfinite'] * 2
    assert [trial.lift for trial in trials] == [0., 0.]
    assert checker.last['max_lift_m'] == 0.
    assert queries.calls == []


@pytest.mark.parametrize('extent,noise,tolerance,allowed', [
    (.5, 60., .002, True),
    (.003, 60., .002, False),
    (.5, 0., .002, False),
    (.005, 0., .02, True),
])
def test_lift_budget_uses_geometry_rotation_and_platform_contact_tolerance(extent, noise, tolerance, allowed):
    queries = Queries(lambda rows, poses, call: ([(0, -.01 if call == 1 else .0001, 1.)], {}))
    checker, state, action = setup_case(queries, extent=extent, tolerance=tolerance)
    noisy = pose()
    noisy[3:] = Rotation.from_euler('x', noise, degrees=True).as_quat()
    trial, = checker.correct([(state, action)], [noisy], [pose()])
    assert (trial.failure_reason is None) is allowed
    if not allowed:
        assert trial.failure_reason == 'initial_clearance_lift_limit'


def test_translation_noise_adds_to_lift_budget():
    queries = Queries(lambda rows, poses, call: ([(0, -.02 if call == 1 else .0001, 1.)], {}))
    checker, state, action = setup_case(queries)
    noisy = pose()
    noisy[0] = .03
    trial, = checker.correct([(state, action)], [noisy], [pose()])
    assert trial.failure_reason is None
    assert trial.lift == pytest.approx(.02 + LIFT_CLEARANCE)


def test_preparation_preserves_seeded_noise_order_across_jobs_and_does_not_mutate_bases():
    config = PlaceStabilityConfig(True, 3, (.005, .005, .005), 5.)
    empty = State([10, 20], [], {}, [], [])
    occupied = State([10, 20], [0], {10: pose()}, [], [])
    jobs = [(empty, Action(0, pose(), pose(), object_id=10), pose()),
            (occupied, Action(1, pose(.2), pose(.2), object_id=20), pose(.2))]
    calls = []
    def correct(rows, poses, bases):
        calls.append((rows, poses, bases))
        return [ReleaseTrial(p.copy(), 'initial_clearance_lateral_contact' if i == 1 else None)
                for i, p in enumerate(poses)]
    backend = NS(place_stability=config, _release_clearance_checker=lambda: NS(correct=correct))

    groups = prepare_release_trials(backend, jobs)

    assert len(calls) == 1 and [len(group) for group in groups] == [3, 3]
    for group, (state, action, base) in zip(groups, jobs):
        expected = list(config.noisy_poses(base, step=len(state.placed_object_indices) + 1,
                                          object_index=action.object_index, object_id=action.object_id))
        np.testing.assert_array_equal([trial.pose for trial in group], expected)
    assert groups[0][1].failure_reason == 'initial_clearance_lateral_contact'
    assert all(trial.failure_reason is None for trial in groups[1])
    np.testing.assert_array_equal(jobs[0][2], pose())
    np.testing.assert_array_equal(jobs[1][2], pose(.2))


def test_preparation_skips_disabled_trials_even_with_nonfinite_base():
    backend = NS(place_stability=PlaceStabilityConfig(False, 4))
    assert prepare_release_trials(backend, []) == []
    assert prepare_release_trials(backend, [(None, None, np.full(7, np.nan))]) == [[]]
