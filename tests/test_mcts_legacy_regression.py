"""Fixed and live comparisons with the legacy tabletop MCTS policies."""

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stacking_common.domain import Action, State
from stacking_common.search import (
    ActionCandidate,
    ActionGenerationResult,
    EvaluationFidelity,
    MCTSConfig,
    MCTSNode,
    MCTSSearchRequest,
    MCTSSession,
    MonteCarloTreeSearch,
    ParallelTransitionEvaluator,
    ProgressiveWideningConfig,
    TransitionEvaluation,
    backpropagate,
    select_child,
)


ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = ROOT.parent / "stacking-tabletop"
POLICY_FIXTURE = (
    ROOT / "tests/fixtures/legacy_mcts/policy_scenarios.json"
)

LEGACY_SELECTED_OBJECT_ID = 102
LEGACY_REJECTED_OBJECT_ID = 101
LEGACY_FINAL_REJECTION = "structural_failure"


def _state(placed=(), actions=()) -> State:
    return State(
        object_ids=[101, 102, 103],
        placed_object_indices=list(placed),
        object_poses={},
        trajectories=[],
        action_history=list(actions),
        terminated=len(placed) == 3,
    )


def _action(index: int) -> Action:
    pose = np.array([0.02 * index, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0])
    return Action(
        object_index=index,
        object_id=101 + index,
        pose=pose,
        init_pose=pose,
    )


class _RemainingGenerator:
    def generate(self, state, observation, request):
        del observation
        actions = [
            _action(index)
            for index in range(3)
            if index not in state.placed_object_indices
        ]
        start = min(request.attempt_offset, len(actions))
        selected = actions[start : start + request.max_candidates]
        return ActionGenerationResult(
            candidates=tuple(ActionCandidate(action) for action in selected),
            attempts=len(selected),
            exhausted=start + len(selected) >= len(actions),
        )


class _LegacyPrior:
    def score(self, state, observation, candidates):
        del state, observation
        return [
            {101: 3.0, 102: 2.0, 103: 1.0}[candidate.action.object_id]
            for candidate in candidates
        ]


class _LegacyScenarioEvaluator:
    worker_count = 1

    def evaluate(self, state, action, fidelity):
        placed = [*state.placed_object_indices, action.object_index]
        next_state = _state(placed, [*state.action_history, action])
        if (
            fidelity is EvaluationFidelity.LONG
            and action.object_id == LEGACY_REJECTED_OBJECT_ID
        ):
            return TransitionEvaluation(
                state=next_state,
                observation=tuple(placed),
                reward=-1.0,
                fidelity=fidelity,
                failed=True,
                rejection_reason=LEGACY_FINAL_REJECTION,
            )
        rewards = (
            {101: 6.0, 102: 4.0, 103: 2.0}
            if fidelity is EvaluationFidelity.SHORT
            else {101: -1.0, 102: 9.0, 103: 5.0}
        )
        return TransitionEvaluation(
            state=next_state,
            observation=tuple(placed),
            reward=rewards[action.object_id],
            fidelity=fidelity,
            terminal=next_state.terminated,
        )


class _ZeroEstimator:
    def estimate(self, state, observation):
        del state, observation
        return 0.0


def _build_search(evaluator) -> MonteCarloTreeSearch:
    return MonteCarloTreeSearch(
        MCTSConfig(
            max_depth=2,
            max_children=3,
            root_max_children=3,
            exploration_constant=0.5,
            discount=0.9,
            generation_batch_size=3,
            evaluation_batch_size=3,
            final_validation_candidates=3,
            widening=ProgressiveWideningConfig(enabled=False),
        ),
        _RemainingGenerator(),
        _LegacyPrior(),
        evaluator,
        _ZeroEstimator(),
        rng=random.Random(17),
    )


def _run(evaluator):
    search = _build_search(evaluator)
    result = search.search(
        MCTSNode(state=_state(), observation=()),
        MCTSSearchRequest(
            iterations=9,
            final_validation_fidelity=EvaluationFidelity.LONG,
        ),
    )
    return search, result


def _tree_decisions(root):
    return tuple(
        (
            child.action.object_id,
            child.visits,
            round(child.value_sum, 8),
            child.failed,
            child.rejection_reason,
        )
        for child in root.children
    )


def test_common_mcts_preserves_fixed_legacy_staged_validation_decision() -> None:
    search, result = _run(_LegacyScenarioEvaluator())

    assert result.selected.action.object_id == LEGACY_SELECTED_OBJECT_ID
    assert result.final_validation_attempts == 3
    assert result.final_validation_failures == 1
    rejected = next(
        child
        for child in result.root.children
        if child.action.object_id == LEGACY_REJECTED_OBJECT_ID
    )
    assert rejected.failed
    assert rejected.rejection_reason == LEGACY_FINAL_REJECTION
    search.close()


def test_parallel_mcts_matches_serial_fixed_legacy_scenario() -> None:
    serial_search, serial = _run(_LegacyScenarioEvaluator())
    parallel_evaluator = ParallelTransitionEvaluator(
        [_LegacyScenarioEvaluator(), _LegacyScenarioEvaluator()],
        max_workers=2,
    )
    parallel_search, parallel = _run(parallel_evaluator)

    assert parallel.selected.action.object_id == LEGACY_SELECTED_OBJECT_ID
    assert _tree_decisions(parallel.root) == _tree_decisions(serial.root)
    assert parallel.completed_iterations == serial.completed_iterations
    assert parallel.generated_candidates == serial.generated_candidates
    assert parallel.evaluated_candidates == serial.evaluated_candidates
    assert parallel.failed_evaluations == serial.failed_evaluations
    assert parallel.final_validation_attempts == serial.final_validation_attempts
    assert parallel.final_validation_failures == serial.final_validation_failures
    serial_search.close()
    parallel_search.close()


def test_multistep_session_preserves_legacy_staged_validation_sequence() -> None:
    """Exercise search, promotion, revalidation, and exhausted-root behavior."""

    search = _build_search(_LegacyScenarioEvaluator())
    session = MCTSSession(
        search,
        MCTSNode(state=_state(), observation=()),
    )
    selected_ids = []
    results = []

    for _ in range(3):
        result = session.search(
            MCTSSearchRequest(
                iterations=9,
                final_validation_fidelity=EvaluationFidelity.LONG,
            )
        )
        results.append(result)
        if result.selected is None:
            break
        selected_ids.append(result.selected.action.object_id)
        session.promote(result.selected, preserve_subtree=True)

    assert selected_ids == [102, 103]
    assert results[-1].selected is None
    # Batched search may preserve this descendant and reject it during subtree
    # revalidation, before the final-selection validation stage is reached.
    assert (
        results[-1].final_validation_failures + results[-1].revalidation.rejected
    ) == 1
    assert not session.root.terminal
    search.close()


def _common_policy_outcomes() -> dict[str, object]:
    config = MCTSConfig(
        max_depth=3,
        max_children=3,
        root_max_children=3,
        exploration_constant=4.0,
        discount=0.5,
        initial_depth1_steps=3,
        widening=ProgressiveWideningConfig(enabled=False),
    )
    effective_depth = {
        str(placed): config.effective_max_depth(placed)
        for placed in (0, 2, 3)
    }

    selection_root = MCTSNode(state=_state(), observation=())
    selection_root.visits = 16
    for index, prior, value in ((0, -2.0, 3.0), (1, 2.0, 0.0)):
        action = _action(index)
        child = selection_root.add_candidate(action, prior=prior)
        child.set_result(_state([index], [action]), (index,), 0.0)
        child.record_visit(value)
    selected = select_child(selection_root, config)

    root = MCTSNode(state=_state(), observation=())
    first_action = _action(0)
    child = root.add_candidate(first_action)
    child.set_result(_state([0], [first_action]), (0,), 2.0)
    second_action = _action(1)
    leaf = child.add_candidate(second_action)
    leaf.set_result(_state([0, 1], [first_action, second_action]), (0, 1), 3.0)
    backpropagate(
        leaf,
        config,
        bootstrap_value=config.discount * 4.0,
    )

    return {
        "effective_depth": effective_depth,
        "puct_selected_id": selected.action.object_id,
        "backup": {
            "root": [root.visits, root.value_sum],
            "child": [child.visits, child.value_sum],
            "leaf": [leaf.visits, leaf.value_sum],
        },
    }


def test_common_policies_match_captured_legacy_scenarios() -> None:
    fixture = json.loads(POLICY_FIXTURE.read_text(encoding="utf-8"))

    assert _common_policy_outcomes() == fixture["outcomes"]


def test_live_legacy_policies_match_common_regression_fixture() -> None:
    legacy_mcts = LEGACY_ROOT / "agent/mcts/mcts.py"
    legacy_node = LEGACY_ROOT / "agent/mcts/node.py"
    if not legacy_mcts.is_file() or not legacy_node.is_file():
        pytest.skip("sibling stacking-tabletop checkout is unavailable")
    fixture = json.loads(POLICY_FIXTURE.read_text(encoding="utf-8"))
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tests/legacy_mcts_harness.py"),
            str(LEGACY_ROOT),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "/usr/lib/python3/dist-packages"},
    )

    assert json.loads(completed.stdout) == fixture["outcomes"]
