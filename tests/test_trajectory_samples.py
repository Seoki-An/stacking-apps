"""Compact histories retain public mutation, copy and checkpoint semantics."""
from dataclasses import asdict
import pickle

import numpy as np
import pytest

from stacking_common.domain import StackObjectTrajectory, State


def samples():
    return np.array([[0., 0., 1., 0., 0., 0., 1.],
                     [.1, .2, .8, .1, .2, .3, .9]])


def test_compact_copies_share_readonly_buffers_until_public_list_access():
    values = samples()
    trajectory = StackObjectTrajectory.from_samples(10, values, [0., .1])
    trajectory.energy, trajectory.vel_integral = 3., 2.
    source = State([10], [0], {10: values[-1]}, [{10: trajectory}], [])
    sibling = source.copy()
    assert len(trajectory) == 2
    for t in (trajectory, source.trajectories[0][10], sibling.trajectories[0][10]):
        assert '_packed_poses' in vars(t)
        for name in ('_packed_poses', '_packed_times'):
            assert vars(t)[name] is vars(trajectory)[name]
            assert not vars(t)[name].flags.writeable
    values[0, 0] = 99.
    assert source.trajectories[0][10].poses[0][0] == 0.
    source.trajectories[0][10].poses[0][0] = 42.
    assert sibling.trajectories[0][10].poses[0][0] == 0.
    assert trajectory.poses[0][0] == 0.
    assert isinstance(trajectory.poses, list)
    assert sibling.latest_velocity_integrals() == {10: 2.}


def test_shared_times_detach_and_metadata_is_branch_local():
    values, times = samples(), np.array([0., .1])
    source = StackObjectTrajectory.from_samples(10, values, times)
    left, right = source.copy(), source.copy()
    times[0] = 99.
    assert vars(left)['_packed_times'][0] == 0.
    left.times[0] = .01
    left.energy = 4.
    assert right.times == source.times == [0., .1]
    assert right.energy == source.energy == 0.
    assert vars(left)['_packed_poses'] is vars(source)['_packed_poses']
    left.add_pose(samples()[0], time=.2)
    assert len(left) == 3 and len(right) == len(source) == 2


def test_copy_snapshots_lists_that_have_external_mutable_references():
    source = StackObjectTrajectory.from_samples(10, samples(), [0., .1])
    poses, times = source.poses, source.times
    first = source.copy()
    poses[0][0] = 3.
    times[0] = .01
    second = source.copy()
    assert first.poses[0][0] == 0. and first.times == [0., .1]
    assert second.poses[0][0] == 3. and second.times == [.01, .1]
    second.poses[0][0] = 5.
    assert poses[0][0] == 3.


def test_state_history_sharing_stays_constant_across_many_branches():
    stages = [{10: StackObjectTrajectory.from_samples(10, samples(), [0., .1])}
              for _ in range(20)]
    root = State([10], [0], {10: samples()[-1]}, stages, [])
    branches = [root.copy() for _ in range(50)]
    arrays = {id(vars(stage[10])[key]) for state in [root, *branches]
              for stage in state.trajectories for key in ('_packed_poses', '_packed_times')}
    assert len(arrays) == 2 * len(stages)
    branches[0].trajectories[0][10].poses[0][0] = 9.
    branches[0].trajectories[-1] = {}
    assert len(root.trajectories[-1]) == len(branches[1].trajectories[-1]) == 1
    assert root.trajectories[0][10].poses[0][0] == branches[1].trajectories[0][10].poses[0][0] == 0.


def test_pickle_restores_shared_samples_as_readonly():
    source = StackObjectTrajectory.from_samples(10, samples(), [0., .1])
    first, second = pickle.loads(pickle.dumps([source, source.copy()]))
    for key in ('_packed_poses', '_packed_times'):
        assert vars(first)[key] is vars(second)[key]
        assert not vars(first)[key].flags.writeable
    first.poses[0][0] = 7.
    first.times[0] = .01
    assert second.poses[0][0] == 0. and second.times == [0., .1]


def test_checkpoint_roundtrip_preserves_all_shared_history():
    from stacking_common.artifacts.planning_checkpoint import (
        PlanningCheckpoint, encode_planning_checkpoint, decode_planning_checkpoint)
    from stacking_common.planning import PickSceneSnapshot, FrameTransform

    values = samples()
    root = State([10], [0], {10: values[-1]}, [
        {10: StackObjectTrajectory.from_samples(10, values, [0., .1])},
        {10: StackObjectTrajectory.from_samples(10, values[::-1], [0., .2])},
    ], [])
    root.trajectories[-1][10].vel_integral = 2.
    pose = np.array([0., 0., 0., 0., 0., 0., 1.])
    checkpoint = PlanningCheckpoint(2, '0' * 64, root,
        PickSceneSnapshot('world', {10: pose}, 'test'), FrameTransform('world', 'stack', pose))
    encoded = encode_planning_checkpoint(checkpoint)
    restored = decode_planning_checkpoint(encoded)
    for stage, expected, times in zip(restored.state.trajectories,
                                     (values, values[::-1]), ([0., .1], [0., .2])):
        np.testing.assert_array_equal(stage[10].poses, expected)
        assert stage[10].times == times
    assert restored.state.latest_velocity_integrals() == {10: 2.}
    for stage in root.trajectories:
        assert '_packed_poses' in vars(stage[10])
        assert '_packed_times' in vars(stage[10])
    restored.state.trajectories[0][10].poses[0][0] = 8.
    assert root.trajectories[0][10].poses[0][0] == 0.


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
