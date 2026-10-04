"""Compact histories retain public mutation, copy and checkpoint semantics."""
from dataclasses import asdict
import pickle

import numpy as np
import pytest

from stacking_common.domain import StackObjectTrajectory, State


def samples():
    return np.array([[0., 0., 1., 0., 0., 0., 1.],
                     [.1, .2, .8, .1, .2, .3, .9]])


def test_compact_copies_own_buffers_until_public_list_access():
    values = samples()
    trajectory = StackObjectTrajectory.from_samples(10, values, [0., .1])
    trajectory.energy, trajectory.vel_integral = 3., 2.
    source = State([10], [0], {10: values[-1]}, [{10: trajectory}], [])
    sibling = source.copy()
    assert len(trajectory) == 2
    for t in (trajectory, source.trajectories[0][10], sibling.trajectories[0][10]):
        assert '_packed_poses' in vars(t)
    values[0, 0] = 99.
    assert source.trajectories[0][10].poses[0][0] == 0.
    source.trajectories[0][10].poses[0][0] = 42.
    assert sibling.trajectories[0][10].poses[0][0] == 0.
    assert trajectory.poses[0][0] == 0.
    assert isinstance(trajectory.poses, list)
    assert sibling.latest_velocity_integrals() == {10: 2.}


def test_compact_history_pickle_dataclass_and_append_match_regular_history():
    expected = StackObjectTrajectory(10, samples(), times=[0., .1])
    packed = StackObjectTrajectory.from_samples(10, samples(), [0., .1])
    restored = pickle.loads(pickle.dumps(packed))
    assert '_packed_poses' in vars(restored)
    actual = asdict(restored)
    np.testing.assert_array_equal(actual['poses'], expected.poses)
    assert actual['times'] == expected.times
    copied = packed.copy()
    copied.add_pose(samples()[0], time=.2)
    assert len(copied) == 3 and len(packed) == 2
    assert packed.times == [0., .1]
    copied.poses[-1][0] = 6.
    assert packed.poses[-1][0] == .1


def test_assigning_pose_list_replaces_compact_storage():
    packed = StackObjectTrajectory.from_samples(10, samples(), [0., .1])
    packed.poses = [samples()[1]]
    packed.times = [0.]
    assert len(packed) == 1
    np.testing.assert_array_equal(packed.poses, samples()[1:])


@pytest.mark.parametrize('case', ['shape', 'nan', 'quaternion', 'times', 'negative', 'infinite'])
def test_compact_external_values_are_validated(case):
    values, times = samples(), [0., .1]
    if case == 'shape':
        values = values[:, :6]
    elif case == 'nan':
        values[0, 0] = np.nan
    elif case == 'quaternion':
        values[0, 3:] = 0.
    elif case == 'times':
        times = [.1, .1]
    elif case == 'negative':
        times = [-1., .1]
    elif case == 'infinite':
        times = [0., np.inf]
    with pytest.raises(ValueError):
        StackObjectTrajectory.from_samples(10, values, times)


def test_compact_history_can_be_made_untimed_before_copying():
    packed = StackObjectTrajectory.from_samples(10, samples(), [0., .1])
    packed.times = None
    copied = packed.copy()
    assert copied.times is None
    copied.add_pose(samples()[0])
    assert len(copied) == 3 and len(packed) == 2
