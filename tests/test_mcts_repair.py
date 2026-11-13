"""Pure final ranking without continuation repair or final revalidation."""

import numpy as np
import pytest
from dataclasses import replace

from stacking_common.domain import Action, State
from stacking_common.search import (
    ActionCandidate, ActionGenerationResult, EvaluationFidelity, FinalSelectionConfig,
    MCTSConfig, MCTSNode, MCTSSearchRequest, MonteCarloTreeSearch,
    ProgressiveWideningConfig, TransitionEvaluation,
)


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
        # Every new transition uses the same physical acceptance rule.
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


def setup(*, evaluator=None, rounds=3, batch_size=4):
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
                max_weight=1.0,
            ),
        ), generator, Scorer(), evaluator, Estimator(),
    )
    request = MCTSSearchRequest(iterations=1)
    return root, generator, evaluator, search, request


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_all_feasible_roots_rank_by_the_same_mix_without_depth_gate(weight):
    root, generator, evaluator, search, _ = setup()
    search.config = replace(search.config, final_selection=FinalSelectionConfig(max_weight=weight))
    complete = child(root, 0)
    child(child(complete, 1), 2)
    complete.record_visit(4)
    shallow = child(root, 3, reward=10)
    shallow.record_visit(20)  # Includes an optimistic bootstrap, unlike the path return.
    pending = shallow.add_candidate(action(4))
    pending.add_candidate(action(5))
    selected, ranked = search._final_select(root)
    assert selected is shallow and set(ranked) == {shallow, complete}
    assert shallow.info['deepest_feasible_depth'] == 1
    assert complete.info['deepest_feasible_depth'] == 3
    assert shallow.info['descendant_path_return'] == 10
    assert shallow.info['final_selection_score'] == (1-weight)*20 + weight*10
    assert not generator.calls and not evaluator.calls


def test_path_score_prefers_deepest_then_best_return_and_never_uses_bootstrap():
    root, _, evaluator, search, _ = setup()
    search.config = replace(search.config, discount=.9)
    candidate = child(root, 0, reward=2)
    shallow = child(candidate, 1, reward=100)
    shallow.record_visit(1000)
    first = child(candidate, 2, reward=1)
    child(first, 3, reward=-10)
    second = child(candidate, 4, reward=2)
    child(second, 5, reward=-5)
    failed = child(candidate, 6, reward=999)
    failed.reject("physical_failure")
    selected, _ = search._final_select(root)
    expected = 2 + .9*2 + .9**2*(-5)
    assert selected.info['deepest_feasible_depth'] == 3
    assert selected.info['descendant_path_return'] == pytest.approx(expected)
    assert selected.info['final_selection_score'] == pytest.approx(expected)
    assert not evaluator.calls
    from stacking_common.visualization.mcts import _best_sequence
    assert [pose.object_id for pose in _best_sequence(candidate)] == [0, 4, 5]


def test_projection_after_ranked_subtree_is_cleared_and_rejected():
    from stacking_common.visualization import project_mcts_step
    root, _, _, search, _ = setup()
    candidate = child(root, 0)
    child(candidate, 1)
    survivor = child(root, 2)
    search._final_select(root)
    assert candidate.info['deepest_path_child_index'] == 0
    # Full-scene revalidation discards descendants, then may reject this root.
    candidate.clear_children()
    candidate.reject('scene_motion_limit', info=candidate.info)
    selected, _ = search._final_select(root)
    assert selected is survivor
    step = project_mcts_step(root, selected, step=32)
    assert [p.object_id for p in step.candidates[0].best_sequence] == [0]


@pytest.mark.parametrize('index', [-1, 5])
def test_projection_does_not_follow_a_stale_path_index(index):
    from stacking_common.visualization.mcts import _best_sequence
    root, _, _, _, _ = setup()
    candidate = child(root, 0)
    child(candidate, 1)
    candidate.info['deepest_path_child_index'] = index
    assert [p.object_id for p in _best_sequence(candidate)] == [0]


@pytest.mark.parametrize('mutation', ['clear', 'detach', 'retire'])
def test_removing_children_invalidates_the_ranked_path(mutation):
    root, _, _, search, _ = setup()
    candidate = child(root, 0)
    first = child(candidate, 1, reward=1.)
    child(candidate, 2, reward=2.)
    search._final_select(root)
    assert candidate.info['deepest_path_child_index'] == 1
    if mutation == 'clear':
        candidate.clear_children()
    elif mutation == 'detach':
        first.detach()
    else:
        first.reject('scene_motion_limit')
        candidate.retire_failed_children()
    assert 'deepest_path_child_index' not in candidate.info


@pytest.mark.parametrize("minimum", [2, 3])
def test_minimum_feasible_depth_excludes_shallow_optimistic_roots(minimum):
    root, generator, evaluator, search, _ = setup()
    search.config = replace(search.config, final_selection=FinalSelectionConfig(
        max_weight=.5, min_feasible_depth=minimum))
    shallow = child(root, 0, reward=10)
    shallow.record_visit(20)
    complete = child(root, 1)
    tail = complete
    for index in range(2, minimum+1):
        tail = child(tail, index)
    # Pending and rejected children cannot satisfy the depth requirement.
    pending = shallow.add_candidate(action(5))
    pending.add_candidate(action(6))
    failed = child(shallow, 7, reward=100)
    failed.reject("physical_failure")
    selected, ranked = search._final_select(root)
    assert selected is complete and ranked == (complete,)
    assert root.info["final_selection_depth_gate"] == dict(
        min_feasible_depth=minimum, feasible_candidates=2, eligible_candidates=1, fallback=False)
    assert not generator.calls and not evaluator.calls


def test_minimum_feasible_depth_falls_back_and_refreshes_after_tree_changes():
    root, _, _, search, request = setup(evaluator=Evaluator(fail_all=True))
    search.config = replace(search.config, final_selection=FinalSelectionConfig(
        max_weight=.5, min_feasible_depth=3))
    first = child(root, 0, reward=2)
    second = child(root, 1, reward=1)
    failed = child(root, 2, reward=100)
    failed.reject("physical_failure")
    selected, ranked = search._final_select(root)
    assert selected is first and set(ranked) == {first, second}
    assert root.info["final_selection_depth_gate"]["fallback"] is True
    child(child(second, 3), 4)
    selected, ranked = search._final_select(root)
    assert selected is second and ranked == (second,)
    assert root.info["final_selection_depth_gate"]["fallback"] is False
    # The gate is relative to the current root, after tree promotion.
    second.detach()
    selected, ranked = search._final_select(second)
    assert root.info["final_selection_depth_gate"]["eligible_candidates"] == 1
    assert second.info["final_selection_depth_gate"]["fallback"] is True


def test_minimum_feasible_depth_never_falls_back_to_a_failed_root():
    root, _, _, search, request = setup(evaluator=Evaluator(fail_all=True))
    search.config = replace(search.config, final_selection=FinalSelectionConfig(
        min_feasible_depth=3))
    result = search.search(root, request)
    assert result.selected is None
    assert root.info["final_selection_depth_gate"]["feasible_candidates"] == 0
    assert root.info["final_selection_depth_gate"]["fallback"] is False


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5])
def test_observed_return_supports_variable_depth_and_search_horizon(length):
    root, _, _, search, _ = setup()
    search.config = replace(search.config, max_depth=5, discount=.9)
    candidate = child(root, 0, reward=1)
    tail = candidate
    for i in range(1, length):
        tail = child(tail, i, reward=i+1)
    selected, _ = search._final_select(root)
    assert selected.info['deepest_feasible_depth'] == length
    assert selected.info['descendant_path_return'] == pytest.approx(
        sum(.9**i * (i+1) for i in range(length)))
    # Retained deeper nodes do not bypass a shorter search horizon.
    search.config = replace(search.config, max_depth=1)
    search._final_select(root)
    assert candidate.info['deepest_feasible_depth'] == 1
    assert candidate.info['descendant_path_return'] == 1


def test_terminal_success_ends_path_and_failed_root_is_excluded():
    root, _, _, search, _ = setup()
    complete = child(root, 0, reward=3)
    child(complete, 2, reward=100)  # A terminal node must not traverse stale children.
    complete.terminal = True
    failed = child(root, 1, reward=100)
    failed.terminal = True
    failed.reject("physical_failure")
    selected, ranked = search._final_select(root)
    assert selected is complete and ranked == (complete,)
    assert selected.info['deepest_feasible_depth'] == 1
    assert selected.info['descendant_path_return'] == 3


def test_descendant_failures_remain_in_mean_but_not_observed_path_return():
    from stacking_common.search.mcts.backup import backpropagate
    root, _, _, search, _ = setup()
    search.config = replace(search.config, final_selection=FinalSelectionConfig(max_weight=.5))
    candidate = child(root, 0, reward=1)
    failed = child(candidate, 1)
    failed.reject("physical_failure")
    backpropagate(failed, search.config)  # One -1 failure cost, plus candidate reward 1.
    assert candidate.mean_value == 0
    selected, _ = search._final_select(root)
    assert selected.info['descendant_path_return'] == 1
    assert selected.info['final_selection_score'] == .5
    good = child(candidate, 2, reward=2)
    backpropagate(good, search.config)
    search._final_select(root)
    assert candidate.mean_value == 1.5
    assert candidate.info['descendant_path_return'] == 3
    assert candidate.info['final_selection_score'] == 2.25


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



def test_selection_values_are_recorded_in_stack_statistics():
    from stacking_common.workflows.mcts_reporting import search_statistics
    root, _, _, search, request = setup()
    result = search.search(root, replace(request, iterations=12))
    statistics = search_statistics(result, 1., search, step=1)
    selected = result.selected
    assert statistics['selected_value'] == {
        'visits': selected.visits, 'reward': selected.reward,
        'mean_return': selected.mean_value,
        'deepest_feasible_depth': selected.info['deepest_feasible_depth'],
        'observed_path_return': selected.info['descendant_path_return'],
        'score': selected.info['final_selection_score'],
    }
