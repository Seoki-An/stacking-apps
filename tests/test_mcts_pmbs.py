"""PMBS batch allocation, accounting, and recovery on deterministic trees."""

from dataclasses import replace
import random

import numpy as np
import pytest

from stacking_common.domain import Action, State
from stacking_common.search import (
    ActionCandidate, ActionGenerationResult, EvaluationFidelity, MCTSConfig,
    MCTSNode, MCTSSearchRequest, MonteCarloTreeSearch, ProgressiveWideningConfig,
    TransitionEvaluation,
)
from stacking_common.search.mcts.tree_policy import child_limit, select_child


def state(placed=()):
    return State(
        object_ids=[10, 20, 30, 40], placed_object_indices=list(placed),
        object_poses={}, trajectories=[], action_history=[],
        terminated=len(placed) == 4,
    )


def action(index):
    pose = np.array([index * 0.01, 0, 0.1, 0, 0, 0, 1.0])
    return Action(object_index=index, object_id=10 * (index + 1),
                  pose=pose, init_pose=pose)


class Generator:
    def __init__(self):
        self.batches = []

    def generate(self, state, observation, request):
        remaining = [i for i in range(4) if i not in state.placed_object_indices]
        chosen = remaining[request.attempt_offset:][:request.max_candidates]
        return ActionGenerationResult(
            candidates=tuple(ActionCandidate(action(i)) for i in chosen),
            attempts=len(chosen),
            exhausted=request.attempt_offset + len(chosen) >= len(remaining),
        )

    def generate_batch(self, states, observations, requests):
        self.batches.append(tuple(tuple(s.placed_object_indices) for s in states))
        return tuple(self.generate(s, o, r) for s, o, r in zip(states, observations, requests))


class Scorer:
    def score(self, state, observation, candidates):
        return [float(4 - c.action.object_index) for c in candidates]


class Evaluator:
    def __init__(self):
        self.calls = []
        self.batches = []

    def evaluate(self, previous, action, fidelity):
        self.calls.append((tuple(previous.placed_object_indices), action.object_index, fidelity))
        next_state = state([*previous.placed_object_indices, action.object_index])
        return TransitionEvaluation(
            state=next_state, observation=tuple(next_state.placed_object_indices),
            reward=2.0, fidelity=fidelity, terminal=next_state.terminated,
        )

    def evaluate_batch(self, states, actions, fidelity):
        self.batches.append((len(actions), fidelity))
        return tuple(self.evaluate(s, a, fidelity) for s, a in zip(states, actions))


class ZeroEstimator:
    def estimate(self, state, observation):
        return 0.0


def config(**kwargs):
    return replace(MCTSConfig(
        max_depth=3, max_children=3, root_max_children=2,
        exploration_constant=0.0, discount=0.9,
        generation_batch_size=3, evaluation_batch_size=3,
        root_lookahead=False, widening=ProgressiveWideningConfig(enabled=False),
    ), **kwargs)


def tree(*, pending=True, visits=10, means=(10.0, 1.0)):
    root = MCTSNode(state=state(), observation=(), visits=2 * visits)
    root.expansion_exhausted = True
    for i, mean in enumerate(means):
        branch = root.add_candidate(action(i), prior=float(i))
        branch.set_result(state([i]), (i,), 1.0)
        branch.visits = visits
        branch.value_sum = mean * visits
        if pending:
            for index in (2, 3):
                branch.add_candidate(action(index), prior=float(4 - index))
    return root


def search(root, *, cfg=None, generator=None, evaluator=None, iterations=3):
    engine = MonteCarloTreeSearch(
        cfg or config(), generator or Generator(), Scorer(),
        evaluator or Evaluator(), ZeroEstimator(), rng=random.Random(0),
    )
    return engine.search(root, MCTSSearchRequest(iterations=iterations, reset_statistics=False))


def snapshot(root):
    return (root.visits, root.value_sum, tuple(snapshot(c) for c in root.children))


def test_global_puct_assigns_multiple_distinct_actions_to_stronger_branch():
    root = tree()
    evaluator = Evaluator()
    result = search(root, evaluator=evaluator)
    assert [call[0] for call in evaluator.calls] == [(0,), (0,), (1,)]
    assert len(set(evaluator.calls)) == 3
    assert [c.visits for c in root.children] == [12, 11]
    assert root.visits == 23
    assert result.completed_iterations == result.descendant_frontier_iterations == 3
    assert result.root_lookahead.attempted == 0


def test_virtual_visits_diversify_close_branches_without_changing_real_statistics():
    root = tree(visits=1, means=(10.0, 9.0))
    before = snapshot(root)
    a, b = root.children
    assert select_child(root, config()) is a
    assert select_child(root, config(), virtual_visits={root: 1, a: 1}) is b
    assert snapshot(root) == before
    evaluator = Evaluator()
    search(root, evaluator=evaluator, cfg=config(evaluation_batch_size=2), iterations=2)
    assert [call[0] for call in evaluator.calls] == [(0,), (1,)]
    assert [c.visits for c in root.children] == [2, 2]


def test_expansion_can_fill_batch_from_one_strong_branch():
    root = tree(pending=False)
    evaluator = Evaluator()
    result = search(root, evaluator=evaluator)
    assert [c.visits for c in root.children] == [13, 10]
    assert len(root.children[0].children) == 3
    assert not root.children[1].children
    assert {call[0] for call in evaluator.calls} == {(0,)}
    assert result.completed_iterations == 3


def test_widening_uses_completed_visits_and_batches_different_states():
    root = tree(pending=False, visits=1, means=(10.0, 9.0))
    generator = Generator()
    cfg = config(widening=ProgressiveWideningConfig(coefficient=1, exponent=0.5))
    assert child_limit(root.children[0], cfg) == 1
    result = search(root, cfg=cfg, generator=generator, iterations=2)
    assert generator.batches == [((0,), (1,))]
    assert [len(c.children) for c in root.children] == [1, 1]
    assert result.completed_iterations == 2


@pytest.mark.parametrize('phase', ['generation', 'evaluation'])
def test_exception_leaves_no_virtual_visits_or_false_backups(phase):
    root = tree(pending=phase == 'evaluation')
    before = snapshot(root)

    class BrokenGenerator(Generator):
        def generate(self, *args):
            raise RuntimeError('generation failed')

    class BrokenEvaluator(Evaluator):
        def evaluate_batch(self, *args):
            raise RuntimeError('evaluation failed')

    with pytest.raises(RuntimeError, match='failed'):
        search(root, generator=BrokenGenerator() if phase == 'generation' else None,
               evaluator=BrokenEvaluator() if phase == 'evaluation' else None)
    assert snapshot(root) == before
    # Reusing the same tree must behave exactly like a fresh search.
    fresh = tree(pending=phase == 'evaluation')
    search(root)
    search(fresh)
    assert snapshot(root) == snapshot(fresh)


def test_single_slot_matches_serial_coordinator():
    class SerialSearch(MonteCarloTreeSearch):
        def _run_descendant_frontier_batch(self, *args):
            return 0

    roots = [tree(pending=False), tree(pending=False)]
    results = []
    for engine_type, root in zip((MonteCarloTreeSearch, SerialSearch), roots):
        engine = engine_type(config(evaluation_batch_size=1), Generator(), Scorer(),
                             Evaluator(), ZeroEstimator(), rng=random.Random(0))
        results.append(engine.search(root, MCTSSearchRequest(iterations=9, reset_statistics=False)))
    assert snapshot(roots[0]) == snapshot(roots[1])
    assert results[0].selected.action.object_index == results[1].selected.action.object_index


def test_rejected_generation_does_not_backup_every_reserved_slot():
    class EmptyGenerator(Generator):
        def generate(self, state, observation, request):
            return ActionGenerationResult(candidates=(), attempts=request.max_candidates,
                                          exhausted=True, rejection_counts={'infeasible': request.max_candidates})

    root = tree(pending=False)
    result = search(root, generator=EmptyGenerator(), iterations=3)
    assert result.completed_iterations == 2
    assert root.visits == 20
    assert [branch.visits for branch in root.children] == [10, 10]
    # Exhausted generation adds no new value evidence to previously visited leaves.
    assert result.descendant_frontier_batches >= 2
    assert all(not branch.children for branch in root.children)


def test_partial_generation_backs_up_only_real_candidates():
    class SparseGenerator(Generator):
        def generate(self, state, observation, request):
            candidate = next(i for i in range(4) if i not in state.placed_object_indices)
            return ActionGenerationResult(candidates=(ActionCandidate(action(candidate)),),
                                          attempts=request.max_candidates, exhausted=True)

    root = tree(pending=False)
    evaluator = Evaluator()
    result = search(root, generator=SparseGenerator(), evaluator=evaluator, iterations=3)
    assert result.completed_iterations == 3
    assert root.visits == 23
    assert len(evaluator.calls) == len(set(evaluator.calls))
    assert all(len(branch.children) <= 1 for branch in root.children)


def test_mixed_depth_batch_uses_one_simulation_fidelity():
    root = tree(visits=1, means=(10.0, 9.0))
    first = root.children[0]
    first.clear_children()
    middle = first.add_candidate(action(2))
    middle.set_result(state((0, 2)), (0, 2), 2.0)
    middle.visits = 1
    middle.value_sum = 10.0
    middle.add_candidate(action(3))
    first.expansion_exhausted = True
    evaluator = Evaluator()
    engine = MonteCarloTreeSearch(config(evaluation_batch_size=2), Generator(), Scorer(), evaluator, ZeroEstimator())
    result = engine.search(root, MCTSSearchRequest(
        iterations=2, reset_statistics=False, ))
    assert [call[2] for call in evaluator.calls] == [EvaluationFidelity.LONG, EvaluationFidelity.LONG]
    assert result.completed_iterations == 2


def test_expired_generation_budget_does_not_start_another_batch(monkeypatch):
    now = [0.0]
    monkeypatch.setattr('stacking_common.search.mcts.search.time.monotonic', lambda: now[0])

    class SlowEmptyGenerator(Generator):
        def generate(self, state, observation, request):
            now[0] = 2.0
            return ActionGenerationResult(candidates=(), attempts=0, exhausted=True)

    root = tree(pending=False)
    engine = MonteCarloTreeSearch(config(), SlowEmptyGenerator(), Scorer(), Evaluator(), ZeroEstimator())
    result = engine.search(root, MCTSSearchRequest(iterations=20, reset_statistics=False, max_time_seconds=1))
    assert result.completed_iterations == 0
    assert result.timed_out
    assert root.visits == 20  # No backup after generation exhausts the deadline.


def test_cached_sibling_is_evaluated_once_without_an_extra_backup():
    root = tree()
    extra = root.children[0].add_candidate(action(1), prior=-1.0)
    evaluator = Evaluator()
    result = search(root, evaluator=evaluator, iterations=2)
    assert result.completed_iterations == 2
    assert root.visits == 22
    assert root.children[0].visits == 12
    assert extra.evaluated and extra.visits == 0
    assert len(evaluator.calls) == len(set(evaluator.calls)) == 3


def test_failed_evaluation_uses_existing_failure_backup_convention():
    class RejectingEvaluator(Evaluator):
        def evaluate(self, previous, action, fidelity):
            result = super().evaluate(previous, action, fidelity)
            if action.object_index == 2:
                return replace(result, failed=True, reward=-1.0, rejection_reason='unstable')
            return result

    root = tree()
    result = search(root, evaluator=RejectingEvaluator())
    assert result.completed_iterations == 3
    assert result.failed_evaluations == 2
    assert root.visits == 23
    for branch in root.children:
        for child in branch.children:
            if child.failed:
                assert child.visits == 1
                assert child.value_sum == 0.0
                assert not child.selectable


def test_serial_only_generator_remains_supported_for_multi_state_expansion():
    class SerialGenerator:
        def generate(self, *args):
            return Generator().generate(*args)

    root = tree(pending=False, visits=1, means=(10.0, 9.0))
    result = search(root, generator=SerialGenerator(), iterations=2)
    assert result.completed_iterations == 2
    assert all(branch.children for branch in root.children)
    assert [branch.visits for branch in root.children] == [2, 2]


class PoseBudgetGenerator(Generator):
    """Supply distinct poses for one remaining object, with optional rejection."""

    def __init__(self, reject_first=False):
        super().__init__()
        self.requests = []
        self.reject_first = reject_first

    def generate(self, previous, observation, request):
        self.requests.append((tuple(previous.placed_object_indices),
                              request.max_candidates, request.attempt_offset))
        index = next(i for i in range(4) if i not in previous.placed_object_indices)
        rejected = int(self.reject_first and request.attempt_offset == 0)
        candidates = []
        for offset in range(rejected, request.max_candidates):
            candidate = action(index)
            candidate.pose[0] += (request.attempt_offset + offset) * 0.001
            candidates.append(ActionCandidate(candidate))
        return ActionGenerationResult(
            candidates=tuple(candidates), attempts=request.max_candidates,
            rejection_counts={'infeasible': rejected} if rejected else {},
        )


def test_frontier_divides_generation_by_reserved_evaluation_slots():
    root = tree(pending=False, visits=1, means=(10.0, 9.0))
    generator = PoseBudgetGenerator()
    result = search(root, generator=generator, iterations=2,
                    cfg=config(max_children=16, generation_batch_size=16,
                               evaluation_batch_size=2))
    assert [size for _, size, _ in generator.requests] == [1, 1]
    assert result.generated_candidates == result.evaluated_candidates == 2
    assert all(len(branch.children) == 1 for branch in root.children)
    assert [branch.visits for branch in root.children] == [2, 2]


def test_root_lookahead_shares_spare_capacity_without_overgeneration():
    root = tree(pending=False, visits=1)
    generator = PoseBudgetGenerator()
    result = search(root, generator=generator, iterations=2,
                    cfg=config(max_children=16, generation_batch_size=16,
                               evaluation_batch_size=5, root_lookahead=True))
    assert sorted(size for _, size, _ in generator.requests) == [2, 3]
    assert result.root_lookahead.attempted == 2
    assert result.generated_candidates == result.evaluated_candidates == 5
    assert sum(child.visits for branch in root.children for child in branch.children) == 2


def test_existing_pending_siblings_use_capacity_before_extra_generation():
    root = tree(pending=False, visits=1, means=(10.0, 9.0))
    for index in (2, 3):
        root.children[1].add_candidate(action(index), prior=1.0)
    generator = PoseBudgetGenerator()
    result = search(root, generator=generator, iterations=2,
                    cfg=config(max_children=16, generation_batch_size=16,
                               evaluation_batch_size=3))
    assert [size for _, size, _ in generator.requests] == [1]
    assert result.generated_candidates == 1
    assert result.evaluated_candidates == 3
    assert all(child.evaluated for branch in root.children for child in branch.children)


def test_demand_retries_fill_new_slots_even_when_old_children_are_feasible():
    from stacking_common.search.mcts.search import _SearchCounters

    root = tree(pending=False)
    branch = root.children[0]
    old = branch.add_candidate(action(2))
    old.set_result(state((0, 2)), (0, 2), 2.0)
    # Reject the first new proposal, despite the existing feasible child.
    class RejectOnce(PoseBudgetGenerator):
        def generate(self, previous, observation, request):
            if not self.requests:
                self.requests.append((tuple(previous.placed_object_indices),
                                      request.max_candidates, request.attempt_offset))
                return ActionGenerationResult(
                    candidates=(), attempts=request.max_candidates,
                    rejection_counts={'infeasible': request.max_candidates},
                )
            return super().generate(previous, observation, request)

    generator = RejectOnce()
    engine = MonteCarloTreeSearch(
        config(max_children=16, generation_batch_size=16,
               expansion_retry_batches=3, expansion_attempt_multiplier=3),
        generator, Scorer(), Evaluator(), ZeroEstimator(),
    )
    counters = _SearchCounters()
    engine._expand_many([branch], counters, None, MCTSSearchRequest(iterations=2),
                        new_candidate_targets={branch: 2})
    assert [size for _, size, _ in generator.requests] == [2, 2]
    assert [offset for _, _, offset in generator.requests] == [1, 3]
    assert len(branch.children) == 3
    assert counters.generated_candidates == 2


def test_partial_generation_retries_only_the_unfilled_demand():
    from stacking_common.search.mcts.search import _SearchCounters

    branch = tree(pending=False).children[0]
    generator = PoseBudgetGenerator(reject_first=True)
    engine = MonteCarloTreeSearch(
        config(max_children=16, expansion_retry_batches=3),
        generator, Scorer(), Evaluator(), ZeroEstimator(),
    )
    counters = _SearchCounters()
    engine._expand_many([branch], counters, None, MCTSSearchRequest(iterations=3),
                        new_candidate_targets={branch: 3})
    assert [size for _, size, _ in generator.requests] == [3, 1]
    assert len(branch.children) == counters.generated_candidates == 3
    assert branch.rejected_expansions == {'infeasible': 1}


@pytest.mark.parametrize('stop', ['deadline', 'rounds', 'attempts', 'exhausted'])
def test_demand_refill_obeys_all_generation_limits(monkeypatch, stop):
    from stacking_common.search.mcts.search import _SearchCounters

    now = [0.0]
    monkeypatch.setattr('stacking_common.search.mcts.search.time.monotonic', lambda: now[0])

    class RejectingGenerator(PoseBudgetGenerator):
        def generate(self, previous, observation, request):
            self.requests.append(request.max_candidates)
            now[0] += 2.0
            return ActionGenerationResult(
                candidates=(), attempts=request.max_candidates,
                exhausted=stop == 'exhausted',
                rejection_counts={'infeasible': request.max_candidates},
            )

    branch = tree(pending=False).children[0]
    generator = RejectingGenerator()
    cfg = config(max_children=16, expansion_retry_batches=2,
                 expansion_attempt_multiplier=1)
    if stop == 'attempts':
        branch.expansion_attempts = 15
    engine = MonteCarloTreeSearch(cfg, generator, Scorer(), Evaluator(), ZeroEstimator())
    counters = _SearchCounters(generation_deadline=1.0 if stop == 'deadline' else None)
    engine._expand_many([branch], counters, None, MCTSSearchRequest(iterations=3),
                        new_candidate_targets={branch: 3})
    assert generator.requests == ([3, 3] if stop == 'rounds' else [1] if stop == 'attempts' else [3])
    assert not branch.children
    assert counters.generated_candidates == 0


@pytest.mark.parametrize('widening', [False, True])
def test_exhausted_state_slots_refill_other_states_within_widening(widening):
    from stacking_common.search.mcts.search import _SearchCounters

    class OneExhaustedBranch(PoseBudgetGenerator):
        def generate(self, previous, observation, request):
            if previous.placed_object_indices == [0]:
                self.requests.append(((0,), request.max_candidates, request.attempt_offset))
                return ActionGenerationResult(
                    candidates=(), attempts=request.max_candidates, exhausted=True,
                    rejection_counts={'infeasible': request.max_candidates},
                )
            return super().generate(previous, observation, request)

    root = tree(pending=False)
    generator = OneExhaustedBranch()
    engine = MonteCarloTreeSearch(
        config(max_children=16, expansion_retry_batches=3,
               widening=ProgressiveWideningConfig(
                   enabled=widening, coefficient=2, exponent=0)),
        generator, Scorer(), Evaluator(), ZeroEstimator(),
    )
    counters = _SearchCounters()
    engine._expand_many(root.children, counters, None, MCTSSearchRequest(iterations=4),
                        new_candidate_targets={node: 2 for node in root.children})
    assert not root.children[0].children
    assert len(root.children[1].children) == (2 if widening else 4)
    assert [size for _, size, _ in generator.requests] == ([2, 2] if widening else [2, 2, 2])
    assert counters.generated_candidates <= 4
