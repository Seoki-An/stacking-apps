"""Final selection uses existing full-simulation evidence without path repair."""

import numpy as np
import pytest
from dataclasses import replace

from stacking_common.domain import Action, State
from stacking_common.search import (
    ActionCandidate, ActionGenerationResult, EvaluationFidelity, FinalSelectionConfig,
    MCTSConfig, MCTSNode, MCTSSearchRequest, MonteCarloTreeSearch,
    ProgressiveWideningConfig, TransitionEvaluation,
)
from stacking_common.search.mcts.search import _SearchCounters


def action(index, x=0.0):
    pose = np.array([x, 0, 0, 0, 0, 0, 1.0])
    return Action(object_index=index, object_id=index, pose=pose, init_pose=pose)


def child(parent, index, *, x=0.0, authoritative=True, reward=1.0):
    node = parent.add_candidate(action(index, x))
    state = parent.state.copy()
    state.placed_object_indices.append(index)
    state.action_history.append(node.action)
    state.object_poses[index] = node.action.pose.copy()
    node.set_result(state, (), reward, info=(
        {"authoritative_fidelity": "long"} if authoritative else {}
    ))
    return node


class Generator:
    def __init__(self):
        self.calls = []

    def generate(self, state, observation, request):
        self.calls.append((tuple(state.placed_object_indices), request.attempt_offset,
                           request.deadline))
        index = next(i for i in state.object_ids if i not in state.placed_object_indices)
        return ActionGenerationResult(
            (ActionCandidate(action(index, 1.0 + request.attempt_offset)),), attempts=1,
            exhausted=False,
        )

    def generate_batch(self, states, observations, requests):
        return tuple(self.generate(s, o, r) for s, o, r in zip(states, observations, requests))


class Evaluator:
    worker_count = 4

    def __init__(self, *, fail_all=False, clock=None, drift=False):
        self.calls = []
        self.fail_all = fail_all
        self.clock = clock
        self.drift = drift

    def evaluate(self, state, action, fidelity):
        return self.evaluate_batch([state], [action], fidelity)[0]

    def evaluate_batch(self, states, actions, fidelity):
        # Repair screens new children at short fidelity; both use one rule.
        self.calls.append((fidelity, tuple((tuple(s.placed_object_indices), a.pose[0])
                                           for s, a in zip(states, actions))))
        results = []
        for state, a in zip(states, actions):
            new = state.copy()
            new.placed_object_indices.append(a.object_index)
            new.action_history.append(a)
            new.object_poses[a.object_id] = a.pose.copy()
            if self.drift:
                new.object_poses[a.object_id][2] += 0.5
            failed = self.fail_all or a.pose[0] < 0
            new.failed = failed
            results.append(TransitionEvaluation(
                state=new, observation=(), reward=-1.0 if failed else 1.0,
                fidelity=fidelity, failed=failed,
                rejection_reason="physical_failure" if failed else None,
            ))
        if self.clock is not None:
            self.clock[0] += 1.0
        return tuple(results)


class Scorer:
    def score(self, state, observation, candidates):
        return [1.0] * len(candidates)


class Estimator:
    def estimate(self, state, observation):
        return 0.0


def setup(*, evaluator=None, depth=2, rounds=3, batch_size=4):
    root = MCTSNode(state=State(list(range(8)), [], {}, [], []), observation=())
    generator = Generator()
    evaluator = evaluator or Evaluator()
    search = MonteCarloTreeSearch(
        MCTSConfig(
            max_depth=3, max_children=2, root_max_children=4,
            exploration_constant=2.0, discount=1.0,
            evaluation_batch_size=batch_size, expansion_retry_batches=rounds,
            expansion_attempt_multiplier=10.0,
            widening=ProgressiveWideningConfig(enabled=False),
            final_selection=FinalSelectionConfig(
                max_weight=1.0, descendant_depth=depth,
            ),
        ), generator, Scorer(), evaluator, Estimator(),
    )
    request = MCTSSearchRequest(iterations=1)
    return root, generator, evaluator, search, request


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_depth_filter_is_identical_for_all_weights(weight):
    root, generator, evaluator, search, request = setup()
    search.config = replace(search.config, final_selection=FinalSelectionConfig(
        max_weight=weight, descendant_depth=2))
    complete = child(root, 0)
    child(child(complete, 1), 2)
    shallow = child(root, 3, reward=100)
    shallow.record_visit(100)
    pending = shallow.add_candidate(action(4))
    pending.add_candidate(action(5))  # Unevaluated paths never satisfy the filter.
    selected, ranked = search._final_select(root)
    assert selected is complete and ranked == (complete,)
    assert shallow.selectable  # A depth filter is not a physical rejection.
    assert not generator.calls and not evaluator.calls


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_exhaustion_falls_back_to_mean_of_feasible_roots_without_repair(weight):
    root, generator, evaluator, search, request = setup()
    search.config = replace(search.config, final_selection=FinalSelectionConfig(
        max_weight=weight, descendant_depth=2))
    low = child(root, 0)
    low.record_visit(2)
    best = child(root, 1)
    best.record_visit(5)
    failed = child(root, 2)
    failed.record_visit(100)
    failed.reject("simulation_unsettled")
    selected, ranked = search._final_select(root)
    assert selected is best and ranked == (best, low)
    assert selected.info["descendant_depth_fallback"]
    assert not generator.calls and not evaluator.calls
    assert not failed.selectable


def test_path_score_is_maximum_of_evaluated_paths_not_first_passing_path():
    root, generator, evaluator, search, request = setup(depth=1)
    candidate = child(root, 0, reward=2)
    child(candidate, 1, reward=1)
    child(candidate, 2, reward=10)
    failed = child(candidate, 3, reward=100)
    failed.reject("physical_failure")
    selected, _ = search._final_select(root)
    assert selected.info["descendant_path_return"] == 12
    assert selected.info["final_selection_score"] == 12
    assert not evaluator.calls


def test_terminal_success_satisfies_depth_but_failed_terminal_does_not():
    root, _, _, search, _ = setup()
    complete = child(root, 0)
    complete.terminal = True
    failed = child(root, 1)
    failed.terminal = True
    failed.reject("physical_failure")
    selected, ranked = search._final_select(root)
    assert selected is complete and ranked == (complete,)
    assert selected.info["descendant_depth_satisfied"]


def test_depth_is_capped_at_search_horizon():
    root, _, _, search, _ = setup(depth=9)
    search.config = replace(search.config, max_depth=1)
    candidate = child(root, 0, reward=3)
    selected, _ = search._final_select(root)
    assert selected is candidate
    assert selected.info["descendant_depth_satisfied"]


def test_all_transitions_use_one_fidelity_once_and_final_selection_is_pure():
    root, generator, evaluator, search, request = setup()
    result = search.search(root, replace(request, iterations=12))
    calls = [(parents, x) for _, batch in evaluator.calls for parents, x in batch]
    def count(node):
        return sum(int(c.evaluated) + count(c) for c in node.children)
    assert len(calls) == count(root) == result.evaluated_candidates
    assert {fidelity for fidelity, _ in evaluator.calls} == {EvaluationFidelity.LONG}
    assert result.final_validation_attempts == result.descendant_validation_attempts == 0
    before = len(evaluator.calls), len(generator.calls)
    search._final_select(root)
    assert (len(evaluator.calls), len(generator.calls)) == before


def test_root_physics_failure_cannot_be_selected_by_depth_fallback():
    root, _, evaluator, search, request = setup(evaluator=Evaluator(fail_all=True))
    result = search.search(root, request)
    assert result.selected is None
    assert result.failed_evaluations > 0


def test_depth_filter_rejects_negative_configuration():
    with pytest.raises(ValueError, match="descendant depth"):
        FinalSelectionConfig(descendant_depth=-1)
