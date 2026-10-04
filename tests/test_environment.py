from dataclasses import dataclass, field

import numpy as np

from stacking_common.domain import Action, State
from stacking_common.environment import StackingEnvironment, StepResult


def state(*, terminated: bool = False, failed: bool = False) -> State:
    return State([], [], {}, [], [], terminated=terminated, failed=failed)


def action() -> Action:
    pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    return Action(0, pose, pose.copy())


@dataclass
class FakeTransition:
    current: State = field(default_factory=state)
    applied_actions: list[Action] = field(default_factory=list)
    closed: bool = False

    def reset(self) -> State:
        self.current = state()
        return self.current

    def step(self, candidate: Action) -> State:
        self.applied_actions.append(candidate)
        self.current = state(terminated=True)
        return self.current

    def restore(self, restored: State) -> None:
        self.current = restored

    def snapshot(self) -> State:
        return self.current

    def close(self) -> None:
        self.closed = True


class FakeObservationBuilder:
    def build(self, current: State) -> tuple[bool, bool]:
        return current.terminated, current.failed


class FakeRewardEvaluator:
    def compute(self, current: State) -> tuple[float, dict[str, object]]:
        return 2.5, {"terminated": current.terminated}


def environment() -> tuple[
    StackingEnvironment[tuple[bool, bool]], FakeTransition
]:
    transition = FakeTransition()
    return (
        StackingEnvironment(
            transition,
            FakeObservationBuilder(),
            FakeRewardEvaluator(),
        ),
        transition,
    )


def test_reset_returns_state_and_observation() -> None:
    env, _ = environment()

    result = env.reset()

    assert result.state == state()
    assert result.observation == (False, False)


def test_step_returns_explicit_transition_result() -> None:
    env, transition = environment()
    candidate = action()

    result = env.step(candidate)

    assert transition.applied_actions == [candidate]
    assert result.observation == (True, False)
    assert result.reward == 2.5
    assert result.info == {"terminated": True}
    assert result.done is True


def test_failed_state_is_done() -> None:
    env, transition = environment()
    transition.current = state(failed=True)

    result = StepResult(
        state=transition.snapshot(),
        observation=env.observation(),
        reward=0.0,
    )

    assert result.observation == (False, True)
    assert result.done is True


def test_restore_snapshot_and_close_are_delegated() -> None:
    env, transition = environment()
    restored = state(failed=True)

    env.restore(restored)

    assert env.snapshot() is restored
    env.close()
    assert transition.closed is True


def test_parallel_batch_results_preserve_order_and_isolate_evaluators():
    import threading
    from types import SimpleNamespace

    barrier = threading.Barrier(3)
    caller = threading.get_ident()
    pairs = []
    states = [state() for _ in range(9)]
    for index, current in enumerate(states):
        current.tag = index

    class Evaluator:
        def __init__(self):
            self.calls = []
            self.thread = None
            self.current = None

        def compute(self, current):
            if not self.calls:
                barrier.wait(timeout=5)
            self.thread = threading.get_ident()
            self.current = current
            self.calls.append(current.tag)
            return float(current.tag), {'tag': current.tag}

        def build(self, current):
            assert self.current is current
            return current.tag, current.failed

    class BatchTransition(FakeTransition):
        backend = SimpleNamespace(place_batch=lambda: None)

        def step_batch(self, inputs, actions):
            assert threading.get_ident() == caller  # Physics stays on its caller.
            return inputs

    for _ in range(3):
        evaluator = Evaluator()
        pairs.append((evaluator, evaluator))
    transition = BatchTransition()
    env = StackingEnvironment(transition, *pairs[0], worker_evaluators=pairs[1:])
    assert env.step_batch([], []) == ()
    assert env._result_executor is None
    try:
        results = env.step_batch(states, [action()] * len(states))
        assert [r.state for r in results] == states
        assert [r.reward for r in results] == list(range(9))
        assert [r.observation for r in results] == [(i, False) for i in range(9)]
        assert [r.info for r in results] == [{'tag': i} for i in range(9)]
        assert len({pair[0].thread for pair in pairs}) == 3
        assert [pair[0].calls for pair in pairs] == [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
        executor = env._result_executor
        env.step_batch(states, [action()] * len(states))
        assert env._result_executor is executor
    finally:
        env.close()
    assert transition.closed
    assert all(not thread.is_alive() for thread in executor._threads)


def test_parallel_batch_drains_workers_after_failure_and_can_be_reused():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from types import SimpleNamespace
    import pytest

    started, release = Event(), Event()

    class BrokenReward(FakeRewardEvaluator):
        broken = True

        def compute(self, current):
            if self.broken:
                raise ValueError('failed reward')
            return super().compute(current)

    class WaitingReward(FakeRewardEvaluator):
        def compute(self, current):
            started.set()
            assert release.wait(timeout=5)
            return super().compute(current)

    transition = FakeTransition()
    transition.backend = SimpleNamespace(place_batch=lambda: None)
    transition.step_batch = lambda states, actions: states
    broken = BrokenReward()
    env = StackingEnvironment(transition, FakeObservationBuilder(), broken,
        worker_evaluators=[(FakeObservationBuilder(), WaitingReward())])
    states, actions = [state(), state()], [action(), action()]
    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            future = caller.submit(env.step_batch, states, actions)
            try:
                assert started.wait(timeout=5)
                assert not future.done()
            finally:
                release.set()
            with pytest.raises(ValueError, match='failed reward'):
                future.result(timeout=5)
        broken.broken = False
        assert len(env.step_batch(states, actions)) == 2
    finally:
        release.set()
        env.close()


def test_pipeline_overlaps_chunks_preserving_worker_assignment_and_order():
    from threading import Event, Lock, get_ident
    from types import SimpleNamespace

    cpu_started, next_physics = Event(), Event()
    caller = get_ident()
    inputs = [state() for _ in range(11)]
    for tag, item in enumerate(inputs):
        item.tag = tag

    class Evaluator:
        def __init__(self):
            self.lock = Lock()
            self.calls = []
            self.current = None

        def compute(self, item):
            assert self.lock.acquire(blocking=False), 'concurrent evaluator reuse'
            self.current = item
            self.calls.append(item.tag)
            if item.tag == 0:
                cpu_started.set()
                assert next_physics.wait(5), 'CPU work did not overlap next physics chunk'
            return float(item.tag), {'tag': item.tag}

        def build(self, item):
            assert self.current is item
            self.lock.release()
            return item.tag

    class ChunkedTransition(FakeTransition):
        backend = SimpleNamespace(place_batch=lambda: None, placement_batch_size=4)

        def iter_step_batches(self, states, actions):
            for start in range(0, len(states), 4):
                assert get_ident() == caller
                if start == 4:
                    assert cpu_started.wait(5)
                    next_physics.set()
                yield states[start:start + 4]

    evaluators = [Evaluator() for _ in range(3)]
    env = StackingEnvironment(ChunkedTransition(), evaluators[0], evaluators[0],
        worker_evaluators=[(e, e) for e in evaluators[1:]])
    try:
        results = env.step_batch(inputs, [action()] * len(inputs))
        assert [r.state for r in results] == inputs
        assert [r.reward for r in results] == list(range(11))
        assert [r.observation for r in results] == list(range(11))
        assert [e.calls for e in evaluators] == [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8]]
        executor = env._pipeline_executor
        env.step_batch(inputs, [action()] * len(inputs))
        assert env._pipeline_executor is executor
    finally:
        next_physics.set()
        env.close()
    assert all(not thread.is_alive() for thread in executor._threads)
    assert all(not thread.is_alive() for thread in env._result_executor._threads)


def test_pipeline_single_state_tail_uses_its_assigned_evaluator():
    from types import SimpleNamespace

    class ChunkedTransition(FakeTransition):
        backend = SimpleNamespace(place_batch=lambda: None, placement_batch_size=2)

        def iter_step_batches(self, states, actions):
            yield states[:2]
            yield states[2:]

    evaluators = [SimpleNamespace(compute=lambda s, i=i: (float(i), {})) for i in range(3)]
    env = StackingEnvironment(ChunkedTransition(), FakeObservationBuilder(), evaluators[0],
        worker_evaluators=[(FakeObservationBuilder(), e) for e in evaluators[1:]])
    try:
        results = env.step_batch([state()] * 3, [action()] * 3)
        assert [r.reward for r in results] == [0., 1., 2.]
    finally:
        env.close()


def test_pipeline_errors_drain_workers_and_allow_reuse():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from types import SimpleNamespace
    import pytest

    for failure in ('physics', 'reward'):
        started, release, finished = Event(), Event(), Event()

        class ChunkedTransition(FakeTransition):
            backend = SimpleNamespace(place_batch=lambda: None, placement_batch_size=2)
            broken = True
            chunks = 0

            def iter_step_batches(self, states, actions):
                for start in range(0, len(states), 2):
                    self.chunks += 1
                    if start == 2 and self.broken:
                        assert started.wait(5)
                        if failure == 'physics':
                            raise RuntimeError('failed physics')
                    yield states[start:start + 2]

        class Reward(FakeRewardEvaluator):
            broken = True

            def compute(self, current):
                if self.broken and failure == 'reward':
                    raise RuntimeError('failed reward')
                return super().compute(current)

        class WaitingReward(FakeRewardEvaluator):
            def compute(self, current):
                started.set()
                assert release.wait(5)
                finished.set()
                return super().compute(current)

        transition, reward = ChunkedTransition(), Reward()
        env = StackingEnvironment(transition, FakeObservationBuilder(), reward,
            worker_evaluators=[(FakeObservationBuilder(), WaitingReward())])
        try:
            with ThreadPoolExecutor(max_workers=1) as caller:
                pending = caller.submit(env.step_batch, [state()] * 6, [action()] * 6)
                try:
                    assert started.wait(5)
                    assert not pending.done()
                finally:
                    release.set()
                with pytest.raises(RuntimeError, match='failed ' + failure):
                    pending.result(timeout=5)
            assert finished.is_set()
            assert 2 <= transition.chunks <= (2 if failure == 'physics' else 3)
            transition.broken = reward.broken = False
            assert len(env.step_batch([state()] * 6, [action()] * 6)) == 6
        finally:
            release.set()
            env.close()


def test_cpu_chunks_are_independent_and_gpu_lookahead_is_bounded():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock, get_ident
    from types import SimpleNamespace

    third_physics, release = Event(), Event()
    inputs = [state() for _ in range(13)]
    for index, item in enumerate(inputs):
        item.tag = index

    class Evaluator:
        def __init__(self):
            self.lock = Lock()
            self.calls = []

        def compute(self, item):
            assert self.lock.acquire(blocking=False), 'concurrent evaluator reuse'
            self.calls.append(item.tag)
            if item.tag == 0:
                assert release.wait(5)
            return float(item.tag), {}

        def build(self, item):
            self.lock.release()
            return item.tag

    class Transition(FakeTransition):
        backend = SimpleNamespace(place_batch=lambda: None, placement_batch_size=4)
        chunks = 0
        threads = set()

        def iter_step_batches(self, states, actions):
            for start in range(0, len(states), 4):
                self.threads.add(get_ident())
                self.chunks += 1
                if self.chunks == 3:
                    third_physics.set()
                yield states[start:start + 4]

    evaluators = [Evaluator() for _ in range(3)]
    transition = Transition()
    env = StackingEnvironment(transition, evaluators[0], evaluators[0],
        worker_evaluators=[(e, e) for e in evaluators[1:]])
    env.result_batch_size = 2
    chunks = []
    original = env._step_results

    def record(states, *, start_index=0):
        chunks.append((len(states), start_index))
        return original(states, start_index=start_index)

    env._step_results = record
    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            pending = caller.submit(env.step_batch, inputs, [action()] * len(inputs))
            try:
                assert third_physics.wait(5), 'GPU waited for the first CPU group'
                assert transition.chunks == 3, 'GPU work exceeded bounded lookahead'
                assert not pending.done()
            finally:
                release.set()
            results = pending.result(timeout=5)
        assert len(transition.threads) == 1
        assert [r.state for r in results] == inputs
        assert [r.reward for r in results] == list(range(13))
        assert [r.observation for r in results] == list(range(13))
        assert chunks == [(2, i) for i in range(0, 12, 2)] + [(1, 12)]
        assert [e.calls for e in evaluators] == [list(range(i, 13, 3)) for i in range(3)]
    finally:
        release.set()
        env.close()
