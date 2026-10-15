"""Final selection repairs missing continuations without discarding valid states."""

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
                mode="max_descendant", validate_descendant_depth=depth,
                validate_descendant_path=True, require_validated_descendant=True,
            ),
        ), generator, Scorer(), evaluator, Estimator(),
    )
    request = MCTSSearchRequest(iterations=1, final_validation_fidelity=EvaluationFidelity.LONG)
    return root, generator, evaluator, search, request


def test_repairs_roots_even_when_another_candidate_already_has_a_valid_path():
    root, generator, evaluator, search, request = setup()
    ready = child(root, 0)
    ready_second = child(ready, 1)
    child(ready_second, 2)
    missing = [child(root, index, reward=4.0) for index in (3, 4)]
    retained = [child(node, 5) for node in missing]
    failed = [child(node, 6, x=-1, authoritative=False) for node in retained]
    counters = _SearchCounters(generation_deadline=12.0)

    selected, ranked = search._final_select(root, request, counters)

    assert selected in missing
    assert set(ranked) == {ready, *missing}
    assert counters.generation_deadline == 12.0
    assert {call[0] for call in generator.calls} == {(3, 5), (4, 5)}
    long = [batch for fidelity, batch in evaluator.calls if fidelity is EvaluationFidelity.LONG]
    short = [batch for fidelity, batch in evaluator.calls if fidelity is EvaluationFidelity.SHORT]
    assert len(long) == 2  # One shared rejection batch, one repair batch.
    assert len(short) == 1  # New repair children are screened together first.
    assert all(len(batch) == 2 for batch in long + short)
    for node, prefix, rejected in zip(missing, retained, failed):
        assert node.children[0] is prefix
        assert rejected in prefix.children and rejected.failed
        assert node.info["validated_descendant_path_return"] == 6.0
    assert all(offset == 1 for _, offset, _ in generator.calls)


def test_repair_builds_two_levels_when_no_continuation_exists():
    root, generator, _, search, request = setup()
    candidate = child(root, 0)
    selected, _ = search._final_select(root, request, _SearchCounters())
    assert selected is candidate
    assert selected.info["validated_descendant_path_return"] == 3.0
    assert any(len(placed) == 2 for placed, _, _ in generator.calls)


def test_failed_repair_is_bounded_without_a_deadline():
    root, generator, evaluator, search, request = setup(evaluator=Evaluator(fail_all=True), rounds=2)
    candidate = child(root, 0)
    selected, _ = search._final_select(root, request, _SearchCounters())
    assert selected is None
    assert candidate.rejection_reason == "missing_validated_descendant"
    assert len(generator.calls) == 4  # Two rounds per required descendant level.
    assert [offset for _, offset, _ in generator.calls] == [0, 1, 2, 3]
    # Every generated child fails short screening, so none reaches long validation.
    assert [fidelity for fidelity, _ in evaluator.calls] == [EvaluationFidelity.SHORT] * 4


@pytest.mark.parametrize("blocked", ["physical_failure", "exhausted", "attempt_limit"])
def test_repair_respects_physical_failures_and_generation_limits(blocked):
    root, generator, _, search, request = setup()
    candidate = child(root, 0)
    if blocked == "physical_failure":
        candidate.reject("physical_failure")
    elif blocked == "exhausted":
        candidate.expansion_exhausted = True
    else:
        candidate.expansion_attempts = 20
    selected, _ = search._final_select(root, request, _SearchCounters())
    assert selected is None
    assert not generator.calls
    if blocked == "physical_failure":
        assert candidate.rejection_reason == "physical_failure"


def test_repair_deadline_keeps_an_already_valid_candidate(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("stacking_common.search.mcts.search.time.monotonic", lambda: clock[0])
    root, generator, _, search, request = setup(evaluator=Evaluator(clock=clock), batch_size=1)
    ready = child(root, 0)
    child(child(ready, 1), 2)
    missing = child(root, 3)
    counters = _SearchCounters()
    selected, ranked = search._final_select(root, request, counters, deadline=0.5)
    assert selected is ready and ranked == (ready,)
    assert counters.final_validation_timed_out
    assert len(generator.calls) == 1
    assert missing.rejection_reason == "missing_validated_descendant"


def test_repair_regenerates_below_a_state_changed_by_validation():
    root, generator, _, search, request = setup(evaluator=Evaluator(drift=True))
    candidate = child(root, 0)
    prefix = child(candidate, 1, authoritative=False)
    stale = child(prefix, 2)
    selected, _ = search._final_select(root, request, _SearchCounters())
    assert selected is candidate
    assert candidate.children[0] is prefix
    assert stale.parent is None
    assert generator.calls[0][0] == (0, 1)


@pytest.mark.parametrize("mode", ["mean", "optional", "no_validation"])
def test_repair_does_not_change_selection_without_required_validation(mode):
    from dataclasses import replace

    root, generator, _, search, request = setup()
    selection = search.config.final_selection
    selection = replace(selection, **{
        "mean": {"mode": "mean"},
        "optional": {"require_validated_descendant": False},
        "no_validation": {"validate_descendant_path": False},
    }[mode])
    search.config = replace(search.config, final_selection=selection)
    child(root, 0)
    search._final_select(root, request, _SearchCounters())
    assert not generator.calls


@pytest.mark.parametrize("retry_rounds", [0, 2])
def test_exhausted_search_selects_best_feasible_root_across_retired_pools(retry_rounds):
    root, generator, _, search, request = setup()
    generator.generate = lambda *args: ActionGenerationResult((), attempts=0, exhausted=True)
    search.config = replace(search.config, rejected_root_retry_rounds=retry_rounds)
    roots = [child(root, index, reward=score) for index, score in enumerate((2., 7., 4.))]
    for node in roots:
        node.expansion_exhausted = True
        node.record_visit(node.reward)
    root.expansion_exhausted = True
    result = search.search(root, replace(request, reset_statistics=False))
    assert result.selected is roots[1]
    assert result.ranked_candidates == (roots[1], roots[2], roots[0])
    for node in result.ranked_candidates:
        assert node.selectable and node.parent is root
        assert not node.state.failed
        assert node.info["validated_descendant_fallback"]
        assert not node.info["validated_descendant_path_available"]
        assert node.info["final_selection_used_mean_fallback"]
        assert node.info["final_selection_score"] == node.mean_value
    assert result.root_retry.rounds == (1 if retry_rounds else 0)
    # Restored roots remain usable by normal session promotion.
    from stacking_common.search import MCTSSession
    session = MCTSSession(search, root)
    assert session.promote(result.selected).state.placed_object_indices == [1]


def test_exhausted_search_prefers_valid_descendant_over_higher_mean_fallback():
    root, _, _, search, request = setup()
    root.expansion_exhausted = True
    missing = child(root, 0, reward=100.)
    missing.expansion_exhausted = True
    missing.record_visit(100.)
    ready = child(root, 1, reward=1.)
    child(child(ready, 2), 3)
    result = search.search(root, replace(request, reset_statistics=False))
    assert result.selected is ready
    assert not ready.info.get("validated_descendant_fallback", False)
    assert missing.rejection_reason == "missing_validated_descendant"


def test_descendant_fallback_never_revives_physical_or_execution_failures():
    from stacking_common.search import ExecutionRejections
    root, _, _, search, request = setup()
    candidates = [child(root, index) for index in range(5)]
    for node in candidates:
        node.reject("missing_validated_descendant")
    candidates[0].reject("simulation_unsettled")
    candidates[1].state.failed = True
    candidates[2].info["authoritative_fidelity"] = "short"
    candidates[4].info.clear()
    request = replace(request, execution_rejections=ExecutionRejections(actions=(candidates[3].action,)))
    selected, ranked = search._select_descendant_fallback(root, request, candidates)
    assert selected is None and ranked == ()
    assert all(not node.selectable for node in candidates)


def test_descendant_fallback_uses_validated_root_after_deadline_without_new_work(monkeypatch):
    root, generator, evaluator, search, request = setup()
    candidate = child(root, 0, reward=3.)
    candidate.record_visit(3.)
    ticks = iter([0., 2.])
    monkeypatch.setattr("stacking_common.search.mcts.search.time.monotonic", lambda: next(ticks, 2.))
    result = search.search(root, replace(request, max_time_seconds=1., reset_statistics=False))
    assert result.selected is candidate
    assert result.timed_out
    assert candidate.info["validated_descendant_fallback"]
    assert not generator.calls and not evaluator.calls
