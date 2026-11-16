"""A locally frozen scene must never be committed as a root."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from stacking_common.environment.core_backend import CorePlacementConfig, SimulationPhaseConfig
from stacking_common.search import (
    MCTSNode, MCTSSearchRequest, MonteCarloTreeSearch, TransitionEvaluation,
    EvaluationFidelity, StackingEnvironmentEvaluator, ParallelTransitionEvaluator,
)
from test_mcts_search import _config, _RemainingObjectGenerator, _ObjectPriorityScorer, _ZeroEstimator, _state, _action


class Evaluator:
    worker_count = 8

    def __init__(self, reject_all=False):
        self.reject_all = reject_all
        self.calls = []

    def evaluate(self, state, action, fidelity):
        result = state.copy()
        result.placed_object_indices.append(action.object_index)
        result.object_poses[action.object_id] = action.pose.copy()
        result.action_history.append(action.copy())
        return TransitionEvaluation(result, None, 10. - action.object_index, fidelity,
                                    diagnostics={'simulation_frozen_object_count': 2})

    def evaluate_full_scene_batch(self, states, actions, fidelity):
        results = []
        for state, action in zip(states, actions):
            self.calls.append(action.object_index)
            local = self.evaluate(state, action, fidelity)
            # Existing stones, too, must retain their full-simulation positions.
            local.state.object_poses[10] = _action(0).pose + np.array([.01, 0, 0, 0, 0, 0, 0])
            failed = self.reject_all or action.object_index == 1
            results.append(replace(local, failed=failed,
                rejection_reason='scene_motion_limit' if failed else None,
                diagnostics={'simulation_frozen_object_count': 0}))
        return results


def setup(reject_all=False):
    state = _state((0,))
    state.object_ids = [10, 20, 30]
    state.object_poses = {10: _action(0).pose.copy()}
    evaluator = Evaluator(reject_all)
    search = MonteCarloTreeSearch(replace(_config(), max_depth=1, rejected_root_retry_rounds=0),
        _RemainingObjectGenerator(), _ObjectPriorityScorer(), evaluator, _ZeroEstimator())
    return search, MCTSNode(state=state), evaluator


@pytest.mark.parametrize('reject_all', [False, True])
def test_roots_are_simulated_once_in_the_full_scene(reject_all):
    search, root, evaluator = setup(reject_all)
    result = search.search(root, MCTSSearchRequest(iterations=1))
    assert evaluator.calls == [1, 2]
    assert root.children[0].rejection_reason == 'scene_motion_limit'
    assert (result.selected is None) == reject_all
    if not reject_all:
        assert result.selected.action.object_index == 2
        assert result.selected.info['simulation_frozen_object_count'] == 0
        assert result.selected.state.object_poses[10][0] == pytest.approx(.01)
        assert root.state.object_poses[10][0] == 0.
    assert result.root_validation.rejected == (2 if reject_all else 1)


def test_frozen_roots_without_full_scene_evaluation_are_refused():
    search, root, evaluator = setup()
    evaluator.evaluate_full_scene_batch = None
    with pytest.raises(ValueError, match='require full-scene evaluation'):
        search.search(root, MCTSSearchRequest(iterations=1))


@pytest.mark.parametrize('crash', [False, True])
def test_full_scene_adapter_restores_freeze_even_on_error(crash):
    phase = SimulationPhaseConfig(.01, 1e-5, .4, 1., 10., 200, freeze_radius=3.)
    backend = SimpleNamespace(config=CorePlacementConfig(phase, .01, 500, .02, 1))
    original = backend.config
    env = SimpleNamespace(transition=SimpleNamespace(backend=backend))
    evaluator = StackingEnvironmentEvaluator({EvaluationFidelity.LONG: env})
    def batch(*args):
        assert backend.config.structural.freeze_radius is None
        if crash:
            raise RuntimeError('native failure')
        return ('validated',)
    evaluator.evaluate_batch = batch
    if crash:
        with pytest.raises(RuntimeError, match='native failure'):
            evaluator.evaluate_full_scene_batch([None], [None], EvaluationFidelity.LONG)
    else:
        assert evaluator.evaluate_full_scene_batch([None], [None], EvaluationFidelity.LONG) == ('validated',)
    assert backend.config is original


def test_parallel_full_scene_validation_preserves_input_order():
    workers = [SimpleNamespace(evaluate_full_scene_batch=lambda states, actions, fidelity: tuple(actions))
               for _ in range(2)]
    evaluator = ParallelTransitionEvaluator(workers)
    try:
        assert evaluator.evaluate_full_scene_batch([None]*5, list(range(5)), EvaluationFidelity.LONG) == tuple(range(5))
    finally:
        evaluator.close()
