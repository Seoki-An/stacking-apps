"""Nominal geometry may run early; validation and reward publication may not."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from threading import Event

import numpy as np
import pytest

from stacking_common.domain import Action, State, TargetGeometry, DirectionalTaper
from stacking_common.environment import PlacementResult, SimulationTransition, StackingEnvironment
from test_composite_target_evaluation import composite, reward, placed
from test_composite_target_sampling import box, inventory, pose


@pytest.mark.parametrize('composite_target', [False, True])
@pytest.mark.parametrize('displacement', [0., .3, float('inf')])
def test_prepared_terms_survive_cache_eviction_and_use_final_robustness(
        monkeypatch, composite_target, displacement):
    target = composite((box(2., 2., 2.), pose()), (box(2., 2., 2.), pose(.8))) \
        if composite_target else TargetGeometry(2., 2., 2., np.zeros(3), DirectionalTaper())
    inv = inventory(target, [box(.4, .5, .6)] * 2)
    current = placed(inv, [pose(-.2, z=.3), pose(.3, z=.7)])
    evaluator = reward(inv, surface_flatness=2., place_stability=1., object_containment=1.)
    geometry = evaluator.prepare_geometry(current.placed_object_indices, current.object_poses)
    with pytest.raises(FrozenInstanceError):
        geometry.containment = 0.
    current.action_history[-1].place_robustness_displacement = displacement
    current.failed = True  # Final rejection must not reuse provisional acceptance.
    expected = evaluator.compute(current)
    evaluator.overlap_evaluator._intersections.clear()
    evaluator.overlap_evaluator._occupancies.clear()
    evaluator._surface._objects.clear()
    evaluator._surface._scenes.clear()
    monkeypatch.setattr(evaluator, 'prepare_geometry', lambda *_: pytest.fail('prepared geometry recomputed'))
    assert evaluator.compute_prepared(current, geometry) == expected


class PreparationBackend:
    placement_batch_size = 3

    def __init__(self, started, release):
        self.started, self.release = started, release
        self.validated = Event()
        self.fail = False
        self.preparation_calls = 0

    def place_batch_with_preparation(self, states, actions, prepare):
        self.validated.clear()
        poses = [{10: a.pose.copy()} for a in actions]
        # Exercise multiple native chunks inside one transition chunk.
        for start in range(0, len(actions), 2):
            self.preparation_calls += 1
            prepare(start, poses[start:start + 2])
        assert self.started.wait(3), 'CPU preparation did not overlap physics'
        assert self.release.wait(3)
        if self.fail:
            raise RuntimeError('GPU failure')
        self.validated.set()
        return [PlacementResult(p, place_robustness_displacement=7., failed=True) for p in poses]

    def close(self):
        pass


class PreparedEvaluator:
    def __init__(self, index, backend, fail=False, hold=None):
        self.index, self.backend, self.fail, self.hold = index, backend, fail, hold
        self.prepared, self.finalized = [], []

    def prepare_geometry(self, indices, poses):
        assert indices == [0]
        self.backend.started.set()
        if self.hold is not None:
            assert self.hold.wait(3)
        if self.fail:
            raise ValueError('CPU preparation failure')
        value = float(poses[10][0])
        self.prepared.append(value)
        return self.index, value

    def compute_prepared(self, state, geometry):
        assert self.backend.validated.is_set(), 'provisional result was published'
        assert geometry == (self.index, state.object_poses[10][0])
        self.finalized.append(geometry[1])
        assert state.failed
        return state.action_history[-1].place_robustness_displacement + geometry[1], {}

    def compute(self, state):
        pytest.fail('geometry was not prepared')

    def build(self, state):
        return state.failed


def setup(workers=2, *, cpu_fail=False, hold=None):
    started, release = Event(), Event()
    backend = PreparationBackend(started, release)
    evaluators = [PreparedEvaluator(i, backend, cpu_fail and i == 0, hold) for i in range(workers)]
    env = StackingEnvironment(SimulationTransition([10], backend), evaluators[0], evaluators[0],
                             worker_evaluators=[(e, e) for e in evaluators[1:]])
    parents = [State([10], [], {}, [], []) for _ in range(7)]
    actions = [Action(0, pose(x), pose(x), object_id=10) for x in range(7)]
    return env, backend, evaluators, parents, actions


@pytest.mark.parametrize('workers', [2, 4])
def test_preparation_overlaps_and_preserves_worker_ownership_order_and_final_status(workers):
    env, backend, evaluators, parents, actions = setup(workers)
    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(env.step_batch, parents, actions)
        assert backend.started.wait(3)
        assert not future.done()
        backend.release.set()
        try:
            results = future.result(timeout=5)
            assert [r.reward for r in results] == list(range(7, 14))
            assert all(r.state.failed and r.observation for r in results)
            for e in evaluators:
                assert e.prepared == e.finalized == list(range(e.index, 7, workers))
            assert all(not p.placed_object_indices for p in parents)
            assert all(a.place_robustness_displacement == 0. for a in actions)
        finally:
            env.close()


def test_cpu_failure_waits_for_physics_and_allows_evaluator_reuse():
    env, backend, evaluators, parents, actions = setup(cpu_fail=True)
    with ThreadPoolExecutor(max_workers=1) as caller:
        task = caller.submit(env.step_batch, parents, actions)
        assert backend.started.wait(3)
        assert not task.done()
        backend.release.set()
        with pytest.raises(ValueError, match='CPU preparation failure'):
            task.result(timeout=5)
        for evaluator in evaluators:
            evaluator.fail = False
        assert len(env.step_batch(parents, actions)) == 7
    env.close()


def test_gpu_failure_drains_cpu_before_returning():
    hold = Event()
    env, backend, _, parents, actions = setup(hold=hold)
    backend.fail = True
    backend.release.set()
    with ThreadPoolExecutor(max_workers=1) as caller:
        task = caller.submit(env.step_batch, parents, actions)
        try:
            assert backend.started.wait(3)
            assert not task.done()
            hold.set()
            with pytest.raises(RuntimeError, match='GPU failure'):
                task.result(timeout=5)
        finally:
            hold.set()
            env.close()


def test_one_worker_retains_the_serial_path(monkeypatch):
    env, backend, evaluators, parents, actions = setup(workers=1)
    def serial(states, candidates):
        return [PlacementResult({10: candidate.pose}, place_robustness_displacement=3.)
                for candidate in candidates]
    monkeypatch.setattr(backend, 'place_batch', serial, raising=False)
    monkeypatch.setattr(evaluators[0], 'compute', lambda current: (3., {}))
    try:
        assert [result.reward for result in env.step_batch(parents, actions)] == [3.] * 7
        assert backend.preparation_calls == 0
    finally:
        env.close()


def test_missing_nominal_snapshot_uses_final_failed_state(monkeypatch):
    env, backend, evaluators, parents, actions = setup()
    original = backend.place_batch_with_preparation
    def missing_snapshot(states, candidates, prepare):
        def notify(start, poses):
            prepare(start, [None, *poses[1:]])
        return original(states, candidates, notify)
    monkeypatch.setattr(backend, 'place_batch_with_preparation', missing_snapshot)
    def compute(current):
        assert backend.validated.is_set() and current.failed
        return 7. + current.object_poses[10][0], {}
    for evaluator in evaluators:
        monkeypatch.setattr(evaluator, 'compute', compute)
    backend.release.set()
    try:
        results = env.step_batch(parents, actions)
        assert [result.reward for result in results] == list(range(7, 14))
        assert all(result.state.failed for result in results)
    finally:
        env.close()
