import random
from dataclasses import replace

import numpy as np
import pytest

from stacking_common.actions.cem import CEMRootActionGenerator, CEMRootConfig
from stacking_common.domain import Action, State
from stacking_common.search import (
    ActionCandidate,
    ActionGenerationResult,
    EvaluationFidelity,
    ExecutionRejections,
    MCTSConfig,
    MCTSNode,
    MCTSSearchRequest,
    MCTSSession,
    MonteCarloTreeSearch,
    ProgressiveWideningConfig,
    TransitionEvaluation,
)


def _action(index: int, x: float | None = None) -> Action:
    x = 0.1 * index if x is None else x
    pose = np.array([x, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0])
    return Action(index, pose, pose, object_id=10 + index)


def _state(placed=()) -> State:
    poses = {10 + index: _action(index).pose for index in placed}
    return State(
        object_ids=[10, 11, 12],
        placed_object_indices=list(placed),
        object_poses=poses,
        trajectories=[],
        action_history=[],
        terminated=len(placed) == 3,
    )


class _Generator:
    def generate(self, state, observation, request):
        del observation
        actions = [
            _action(index)
            for index in range(3)
            if index not in state.placed_object_indices
        ]
        values = actions[
            request.attempt_offset : request.attempt_offset + request.max_candidates
        ]
        return ActionGenerationResult(
            tuple(ActionCandidate(value) for value in values),
            attempts=len(values),
            exhausted=request.attempt_offset + len(values) >= len(actions),
        )


class _Prior:
    def score(self, state, observation, candidates):
        del state, observation
        return [float(candidate.action.object_id) for candidate in candidates]


class _Evaluator:
    def __init__(self):
        self.calls = []

    def evaluate(self, state, action, fidelity):
        self.calls.append(
            (tuple(state.placed_object_indices), action.object_id, fidelity)
        )
        placed = [*state.placed_object_indices, action.object_index]
        result = _state(placed)
        return TransitionEvaluation(
            result,
            tuple(placed),
            reward=float(action.object_id),
            fidelity=fidelity,
            terminal=result.terminated,
        )


class _Value:
    def estimate(self, state, observation):
        del state, observation
        return 0.0


def _search(evaluator):
    return MonteCarloTreeSearch(
        MCTSConfig(
            max_depth=2,
            max_children=3,
            root_max_children=3,
            exploration_constant=0.0,
            discount=1.0,
            generation_batch_size=3,
            evaluation_batch_size=3,
            widening=ProgressiveWideningConfig(enabled=False),
        ),
        _Generator(),
        _Prior(),
        evaluator,
        _Value(),
        rng=random.Random(0),
    )


def test_search_accepts_external_seed_and_filters_execution_feedback() -> None:
    evaluator = _Evaluator()
    search = _search(evaluator)
    root = MCTSNode(state=_state(), observation=())
    seed = _action(1, x=0.35)

    result = search.search(
        root,
        MCTSSearchRequest(
            iterations=2,
            seed_actions=(seed,),
            execution_rejections=ExecutionRejections(object_ids={10}),
        ),
    )

    assert result.seeded_candidates == 1
    assert result.execution_rejected_candidates == 1
    assert any(child.info.get("external_seed") for child in root.children)
    assert all(child.action.object_id != 10 for child in root.children)


def test_execution_rejection_matches_nearby_actions_without_aliasing() -> None:
    rejected = _action(1, x=0.20)
    policy = ExecutionRejections(actions=(rejected,), xy_radius=0.05)
    rejected.pose[0] = 9.0

    assert policy.reason(_action(1, x=0.24)) == "execution_rejected_action"
    assert policy.reason(_action(1, x=0.30)) is None
    assert policy.reason(_action(2, x=0.20)) is None


def test_session_promotes_revalidates_and_accumulates_rejections() -> None:
    evaluator = _Evaluator()
    search = _search(evaluator)
    root = MCTSNode(state=_state(), observation=())
    selected_action = _action(0)
    selected = root.add_candidate(selected_action, prior=1.0)
    selected.set_result(_state([0]), (0,), 10.0)
    continuation_action = _action(1)
    continuation = selected.add_candidate(continuation_action, prior=1.0)
    continuation.set_result(_state([0, 1]), (0, 1), 11.0)
    session = MCTSSession(search, root)

    session.promote(selected, preserve_subtree=True)
    result = session.search(MCTSSearchRequest(iterations=1))

    assert result.revalidation.attempted == 1
    assert any(call[2] is EvaluationFidelity.LONG for call in evaluator.calls)

    session.reject_execution(continuation_action, reject_object=True)
    assert 11 in session.execution_rejections.object_ids
    assert not continuation.selectable

    seed = _action(2, x=0.42)
    session.add_seed(seed)
    next_result = session.search(MCTSSearchRequest(iterations=1))
    assert next_result.seeded_candidates == 1


@pytest.mark.parametrize("previously_exhausted", [False, True])
@pytest.mark.parametrize("root_keep", [12, 32], ids=["cpu_pool", "cuda_pool"])
def test_promotion_starts_new_cem_pool_without_discarding_subtree(previously_exhausted, root_keep):
    class FreshPool:
        def generate(self, state, observation, request):
            candidates = tuple(ActionCandidate(_action(1, x=float(i))) for i in range(32))
            return ActionGenerationResult(candidates, attempts=32, exhausted=True)

    class RecordingCEM(CEMRootActionGenerator):
        def generate(self, state, observation, request):
            offsets.append(request.attempt_offset)
            return super().generate(state, observation, request)

    offsets = []
    evaluator = _Evaluator()
    search = _search(evaluator)
    search.config = replace(search.config, max_depth=1, root_max_children=root_keep,
                            generation_batch_size=32, expansion_attempt_multiplier=3.0,
                            validate_root_before_search=True)
    search.root_action_generator = RecordingCEM(FreshPool(), _Prior(), CEMRootConfig(
        population=32, elite=4, iterations=1, keep=root_keep, mutation_xy_std=0.0,
        mutation_yaw_std=0.0, fresh_fraction=1.0, duplicate_xy_threshold=0.01,
        duplicate_yaw_threshold=0.0,
    ))
    root = MCTSNode(state=_state(), observation=())
    selected = root.add_candidate(_action(0), prior=1.0)
    selected.set_result(_state([0]), (0,), 10.0)
    continuation = selected.add_candidate(_action(2), prior=1.0)
    continuation.set_result(_state([0, 2]), (0, 2), 12.0)
    descendant = continuation.add_candidate(_action(1), prior=1.0)
    descendant.set_result(_state([0, 2, 1]), (0, 2, 1), 11.0, terminal=True)
    selected.expansion_attempts = 48
    selected.expansion_exhausted = previously_exhausted
    selected.rejected_expansions["old_generation_failure"] = 4
    selected.info["external_seed_attempts"] = 3
    selected.record_visit(10.0)
    continuation.expansion_attempts = 7
    continuation.expansion_exhausted = True
    continuation.record_visit(12.0)
    session = MCTSSession(search, root)

    session.promote(selected, preserve_subtree=True)

    assert selected.children == [continuation]
    assert continuation.children == [descendant]
    assert continuation.parent is selected and descendant.parent is continuation
    assert selected.visits == continuation.visits == 1
    assert continuation.expansion_attempts == 7 and continuation.expansion_exhausted
    result = session.search(MCTSSearchRequest(iterations=1, reset_statistics=False,
                                            final_validation_fidelity=EvaluationFidelity.LONG))

    assert offsets and offsets[0] == 0
    assert search.root_action_generator.last_statistics.final_candidates == root_keep
    assert result.generated_candidates == root_keep - 1
    assert result.revalidation.attempted == result.revalidation.accepted == 1
    assert continuation in selected.children and continuation.children == [descendant]
    assert "old_generation_failure" not in selected.rejected_expansions
    assert "external_seed_attempts" not in selected.info
