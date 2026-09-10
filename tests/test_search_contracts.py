from collections.abc import Sequence

import numpy as np
import pytest

from stacking_common.domain import Action, State
from stacking_common.search import (
    ActionCandidate,
    ActionGenerationRequest,
    ActionGenerationResult,
    ActionGenerator,
    EvaluationFidelity,
    PriorScorer,
    DiagnosticPriorScorer,
    HorizonValueEstimator,
    ParallelTransitionEvaluator,
    PlacementPriorScorer,
    TransitionEvaluation,
    TransitionEvaluator,
    ValueEstimator,
)


def _state() -> State:
    return State(
        object_ids=[10, 20],
        placed_object_indices=[],
        object_poses={},
        trajectories=[],
        action_history=[],
    )


def _action(index: int = 0) -> Action:
    pose = np.array([float(index), 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])
    return Action(
        object_index=index,
        object_id=(index + 1) * 10,
        pose=pose,
        init_pose=pose,
    )


def test_action_generation_request_validates_bound_and_priority() -> None:
    request = ActionGenerationRequest(
        max_candidates=4,
        sampling_priority=" lower_floor_first ",
    )

    assert request.max_candidates == 4
    assert request.sampling_priority == "lower_floor_first"
    assert request.attempt_offset == 0
    with pytest.raises(ValueError, match="limit"):
        ActionGenerationRequest(max_candidates=0)
    with pytest.raises(ValueError, match="priority"):
        ActionGenerationRequest(max_candidates=1, sampling_priority=" ")
    with pytest.raises(ValueError, match="offset"):
        ActionGenerationRequest(max_candidates=1, attempt_offset=-1)


def test_action_candidate_owns_action_and_diagnostics() -> None:
    action = _action()
    diagnostics = {"source": "planar", "scores": [1.0, 2.0]}
    candidate = ActionCandidate(action=action, diagnostics=diagnostics)

    action.pose[0] = 99.0
    diagnostics["scores"].append(3.0)

    assert candidate.action.pose[0] == 0.0
    assert candidate.diagnostics["scores"] == [1.0, 2.0]
    with pytest.raises(TypeError):
        candidate.diagnostics["new"] = True


def test_generation_result_accounts_for_accepted_and_rejected_attempts() -> None:
    result = ActionGenerationResult(
        candidates=(ActionCandidate(_action()),),
        attempts=4,
        exhausted=True,
        rejection_counts={"duplicate": 2, "pose_solver": 1},
    )

    assert len(result.candidates) == 1
    assert result.attempts == 4
    assert result.exhausted
    assert result.rejection_counts == {"duplicate": 2, "pose_solver": 1}
    with pytest.raises(TypeError):
        result.rejection_counts["other"] = 1

    with pytest.raises(ValueError, match="exceed"):
        ActionGenerationResult(
            candidates=(ActionCandidate(_action()),),
            attempts=1,
            rejection_counts={"duplicate": 1},
        )


def test_empty_generation_result_distinguishes_exhaustion() -> None:
    retryable = ActionGenerationResult.empty(exhausted=False)
    exhausted = ActionGenerationResult.empty(exhausted=True)

    assert retryable.candidates == exhausted.candidates == ()
    assert not retryable.exhausted
    assert exhausted.exhausted


def test_transition_evaluation_owns_state_and_reports_fidelity() -> None:
    state = _state()
    evaluation = TransitionEvaluation(
        state=state,
        observation="observation",
        reward=2.5,
        fidelity="short",
        terminal=True,
        diagnostics={"settled": True},
    )

    state.object_ids[0] = 999

    assert evaluation.state.object_ids == [10, 20]
    assert evaluation.fidelity is EvaluationFidelity.SHORT
    assert evaluation.reward == 2.5
    assert evaluation.done
    with pytest.raises(TypeError):
        evaluation.diagnostics["new"] = True


def test_rejected_transition_must_be_failed() -> None:
    with pytest.raises(ValueError, match="marked failed"):
        TransitionEvaluation(
            state=_state(),
            observation=None,
            reward=-1.0,
            fidelity=EvaluationFidelity.LONG,
            failed=False,
            rejection_reason="outside_target",
        )

    evaluation = TransitionEvaluation(
        state=_state(),
        observation=None,
        reward=-1.0,
        fidelity=EvaluationFidelity.LONG,
        failed=True,
        rejection_reason="outside_target",
    )
    assert evaluation.done
    assert evaluation.rejection_reason == "outside_target"


def test_transition_cannot_hide_terminal_or_failed_state_flags() -> None:
    state = _state()
    state.terminated = True
    state.failed = True

    evaluation = TransitionEvaluation(
        state=state,
        observation=None,
        reward=-1.0,
        fidelity=EvaluationFidelity.SHORT,
    )

    assert evaluation.terminal
    assert evaluation.failed
    assert evaluation.done


class _Generator:
    def generate(
        self,
        state: State,
        observation: str,
        request: ActionGenerationRequest,
    ) -> ActionGenerationResult:
        del state, observation
        candidates = tuple(
            ActionCandidate(_action(index))
            for index in range(request.max_candidates)
        )
        return ActionGenerationResult(
            candidates=candidates,
            attempts=len(candidates),
        )


class _Scorer:
    def score(
        self,
        state: State,
        observation: str,
        candidates: Sequence[ActionCandidate],
    ) -> Sequence[float]:
        del state, observation
        return [float(candidate.action.object_index) for candidate in candidates]


class _Evaluator:
    def evaluate(
        self,
        state: State,
        action: Action,
        fidelity: EvaluationFidelity,
    ) -> TransitionEvaluation[str]:
        return TransitionEvaluation(
            state=state,
            observation=f"object-{action.object_id}",
            reward=1.0,
            fidelity=fidelity,
        )


class _Estimator:
    def estimate(self, state: State, observation: str) -> float:
        del state, observation
        return 3.0


def _exercise_contracts(
    generator: ActionGenerator[str],
    scorer: PriorScorer[str],
    evaluator: TransitionEvaluator[str],
    estimator: ValueEstimator[str],
) -> None:
    state = _state()
    generated = generator.generate(
        state,
        "root",
        ActionGenerationRequest(max_candidates=2),
    )
    scores = scorer.score(state, "root", generated.candidates)
    evaluation = evaluator.evaluate(
        state,
        generated.candidates[0].action,
        EvaluationFidelity.SHORT,
    )

    assert list(scores) == [0.0, 1.0]
    assert evaluation.observation == "object-10"
    assert estimator.estimate(evaluation.state, evaluation.observation) == 3.0


def test_protocols_support_framework_independent_collaborators() -> None:
    _exercise_contracts(_Generator(), _Scorer(), _Evaluator(), _Estimator())


def test_concrete_prior_and_horizon_value_policies() -> None:
    state = _state()
    candidates = (
        ActionCandidate(_action(0), diagnostics={"planar_score": 2.0}),
        ActionCandidate(_action(1), diagnostics={}),
    )

    assert DiagnosticPriorScorer().score(state, None, candidates) == [2.0, 0.0]
    assert HorizonValueEstimator(2.5, horizon=4, discount=0.5).estimate(
        state, None
    ) == 3.75


class _RecordingEvaluator(_Evaluator):
    def __init__(self, worker):
        self.worker = worker
        self.actions = []
        self.closed = False

    def evaluate(self, state, action, fidelity):
        self.actions.append(action.object_index)
        result = super().evaluate(state, action, fidelity)
        return TransitionEvaluation(
            state=result.state,
            observation=self.worker,
            reward=float(action.object_index),
            fidelity=fidelity,
        )

    def close(self):
        self.closed = True


def test_parallel_transition_evaluator_preserves_batch_order() -> None:
    workers = [_RecordingEvaluator(0), _RecordingEvaluator(1)]
    evaluator = ParallelTransitionEvaluator(workers)

    results = evaluator.evaluate_batch(
        [_state(), _state()],
        [_action(0), _action(1)],
        EvaluationFidelity.SHORT,
    )

    assert [result.reward for result in results] == [0.0, 1.0]
    assert workers[0].actions == [0]
    assert workers[1].actions == [1]
    evaluator.close()
    assert all(worker.closed for worker in workers)


def test_placement_prior_combines_stability_and_containment() -> None:
    inventory = type(
        "Inventory",
        (),
        {
            "object_ids": (10, 20),
            "__getitem__": lambda self, index: f"object-{index}",
        },
    )()
    overlap = type(
        "Overlap",
        (),
        {"object_containment": lambda self, stack_object, pose: 0.75},
    )()
    action = _action(0)
    action.c_feq = 0.5
    scorer = PlacementPriorScorer(
        inventory,
        overlap,
        posegen_threshold=1.0,
        stability_weight=0.5,
        containment_weight=2.0,
    )

    assert scorer.score(_state(), None, [ActionCandidate(action)]) == [1.25]
