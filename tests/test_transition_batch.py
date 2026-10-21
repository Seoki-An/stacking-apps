"""Batch transitions preserve parent ownership without restoring each branch."""
import numpy as np
import pytest

from stacking_common.domain import Action, StackObjectTrajectory, State
from stacking_common.environment import PlacementResult, SimulationTransition


def pose(x=0.):
    return np.array([x, 0., 1., 0., 0., 0., 1.])


def parent():
    return State(
        [10, 20, 30], [0], {10: pose()},
        [{10: StackObjectTrajectory(10, poses=[pose()], times=[0.])}],
        [Action(0, pose(), pose(), object_id=10, diagnostics={'values': [1.]})],
    )


class BatchBackend:
    def __init__(self):
        self.calls = 0

    def reset(self, ids):
        pass

    def restore(self, state):
        raise AssertionError('batch states must not require backend restoration')

    def place_batch(self, states, actions):
        self.calls += 1
        return tuple(PlacementResult(
            {a.object_id: a.pose},
            {a.object_id: StackObjectTrajectory(a.object_id, poses=[a.pose], times=[0.])},
        ) for a in actions)


@pytest.mark.parametrize('chunk_size', [None, 1])
def test_batch_outputs_own_histories_and_leave_inputs_and_snapshot_independent(chunk_size):
    backend = BatchBackend()
    if chunk_size is not None:
        backend.placement_batch_size = chunk_size
    transition = SimulationTransition([10, 20, 30], backend)
    transition.reset()
    source = parent()
    actions = [Action(i, pose(i), pose(i)) for i in (1, 2)]
    first, second = transition.step_batch([source, source], actions)
    assert first.placed_object_indices == [0, 1]
    assert second.placed_object_indices == [0, 2]
    assert source.placed_object_indices == [0]
    assert [a.object_id for a in actions] == [-1, -1]
    assert backend.calls == (2 if chunk_size else 1)

    first.trajectories[0][10].poses[0][0] = 9.
    first.trajectories[0][10].times[0] = 2.
    with pytest.raises(TypeError, match='in place'):
        first.action_history[0].diagnostics['values'][0] = 3.
    first.action_history[0].diagnostics['values'] = [3.]
    first.object_poses[10][0] = 4.
    for untouched in (source, second, transition.snapshot()):
        assert untouched.trajectories[0][10].poses[0][0] == 0.
        assert untouched.trajectories[0][10].times == [0.]
        assert untouched.action_history[0].diagnostics['values'] == [1.]
        assert untouched.object_poses[10][0] == 0.
    second.trajectories[-1][30].poses[0][0] = 99.
    second.action_history[-1].pose[0] = 99.
    snapshot = transition.snapshot()
    assert snapshot.trajectories[-1][30].poses[0][0] == 2.
    assert snapshot.action_history[-1].pose[0] == 2.
    assert transition.step_batch([], []) == ()
    assert transition.snapshot().placed_object_indices == [0, 2]


@pytest.mark.parametrize('case,exception,message', [
    ('inventory', ValueError, 'inventory'),
    ('placed', ValueError, 'already been placed'),
    ('terminated', RuntimeError, 'completed'),
    ('failed', RuntimeError, 'completed'),
    ('index', IndexError, 'outside'),
    ('object_id', ValueError, 'does not match'),
])
@pytest.mark.parametrize('chunk_size', [None, 1])
def test_batch_validates_every_parent_before_dispatch(case, exception, message, chunk_size):
    backend = BatchBackend()
    if chunk_size is not None:
        backend.placement_batch_size = chunk_size
    transition = SimulationTransition([10, 20, 30], backend)
    transition.reset()
    source = parent()
    action = Action(1, pose(1.), pose(1.))
    if case == 'inventory':
        source.object_ids = [10, 30, 20]
    elif case == 'placed':
        action.object_index = 0
    elif case in ('terminated', 'failed'):
        setattr(source, case, True)
    elif case == 'index':
        action.object_index = 3
    elif case == 'object_id':
        action.object_id = 30
    with pytest.raises(exception, match=message):
        transition.step_batch([parent(), source], [Action(1, pose(), pose()), action])
    assert backend.calls == 0
    assert transition.snapshot().placed_object_indices == []


def test_chunk_iterator_yields_before_starting_next_physics_chunk():
    backend = BatchBackend()
    backend.placement_batch_size = 1
    transition = SimulationTransition([10, 20, 30], backend)
    transition.reset()
    batches = transition.iter_step_batches([parent()] * 2,
        [Action(i, pose(i), pose(i)) for i in (1, 2)])
    assert backend.calls == 0
    first, = next(batches)
    assert backend.calls == 1
    assert first.placed_object_indices == [0, 1]
    second, = next(batches)
    assert backend.calls == 2
    assert second.placed_object_indices == [0, 2]
    assert first.placed_object_indices == [0, 1]
    with pytest.raises(StopIteration):
        next(batches)


def test_streamed_chunk_validates_backend_result_count():
    backend = BatchBackend()
    backend.placement_batch_size = 1
    backend.place_batch = lambda states, actions: ()
    transition = SimulationTransition([10, 20, 30], backend)
    with pytest.raises(RuntimeError, match='wrong number'):
        next(transition.iter_step_batches([parent()], [Action(1, pose(), pose())]))
