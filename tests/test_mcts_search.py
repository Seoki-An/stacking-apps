from collections.abc import Sequence

import numpy as np
import pytest

from stacking_common.domain import Action, State
from stacking_common.environment import StackingEnvironment
from stacking_common.search import (
    ActionCandidate,
    ActionGenerationRequest,
    ActionGenerationResult,
    EvaluationFidelity,
    FinalSelectionConfig,
    MCTSConfig,
    MCTSNode,
    MCTSSearchRequest,
    MaskedActionGeneratorAdapter,
    MonteCarloTreeSearch,
    ProgressiveWideningConfig,
    StackingEnvironmentEvaluator,
    TransitionEvaluation,
)


def _state(
    placed: Sequence[int] = (),
    *,
    failed: bool = False,
    terminated: bool = False,
    actions: Sequence[Action] = (),
) -> State:
    return State(
        object_ids=[10, 20],
        placed_object_indices=list(placed),
        object_poses={},
        trajectories=[],
        action_history=list(actions),
        failed=failed,
        terminated=terminated,
    )


def _action(index: int, *, diagnostics=None) -> Action:
    pose = np.array([float(index), 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])
    return Action(
        object_index=index,
        object_id=(index + 1) * 10,
        pose=pose,
        init_pose=pose,
        diagnostics=dict(diagnostics or {}),
    )


def _config() -> MCTSConfig:
    return MCTSConfig(
        max_depth=2,
        max_children=2,
        root_max_children=2,
        exploration_constant=0.0,
        discount=1.0,
        widening=ProgressiveWideningConfig(enabled=False),
    )


class _RemainingObjectGenerator:
    def generate(self, state, observation, request):
        del observation
        remaining = [
            index
            for index in range(len(state.object_ids))
            if index not in state.placed_object_indices
        ][: request.max_candidates]
        candidates = tuple(ActionCandidate(_action(index)) for index in remaining)
        return ActionGenerationResult(
            candidates=candidates,
            attempts=len(candidates),
            exhausted=len(remaining) == len(state.object_ids) - len(
                state.placed_object_indices
            ),
        )


class _BatchRemainingObjectGenerator(_RemainingObjectGenerator):
    def __init__(self):
        self.batch_depths = []

    def generate_batch(self, states, observations, requests):
        self.batch_depths.append(
            tuple(len(state.placed_object_indices) for state in states)
        )
        return tuple(
            self.generate(state, observation, request)
            for state, observation, request in zip(
                states, observations, requests
            )
        )


class _ObjectPriorityScorer:
    def score(self, state, observation, candidates):
        del state, observation
        return [
            2.0 if candidate.action.object_index == 0 else 1.0
            for candidate in candidates
        ]


class _DeterministicEvaluator:
    def evaluate(self, state, action, fidelity):
        placed = [*state.placed_object_indices, action.object_index]
        next_state = _state(
            placed,
            terminated=len(placed) == len(state.object_ids),
            actions=[*state.action_history, action],
        )
        return TransitionEvaluation(
            state=next_state,
            observation=tuple(placed),
            reward=5.0 if action.object_index == 0 else 1.0,
            fidelity=fidelity,
            terminal=next_state.terminated,
        )


class _ZeroEstimator:
    def estimate(self, state, observation):
        del state, observation
        return 0.0


def test_coordinator_runs_complete_deterministic_search_cycles() -> None:
    search = MonteCarloTreeSearch(
        _config(),
        _RemainingObjectGenerator(),
        _ObjectPriorityScorer(),
        _DeterministicEvaluator(),
        _ZeroEstimator(),
    )
    root = MCTSNode(state=_state(), observation=())

    result = search.search(root, MCTSSearchRequest(iterations=6))

    assert result.root is root
    assert result.completed_iterations == 6
    assert root.visits == 6
    assert len(root.children) == 2
    assert all(child.evaluated for child in root.children)
    assert result.selected is root.children[0]
    assert result.selected.action is not None
    assert result.selected.action.object_index == 0
    assert len(result.selected.children) == 1
    assert result.selected.children[0].terminal
    assert result.expansion_calls >= 2
    assert result.generated_candidates >= 3
    assert result.evaluated_candidates >= 3
    assert result.failed_evaluations == 0


def test_empty_root_can_retry_unconsumed_fresh_proposals_and_receives_deadline():
    from dataclasses import replace
    from types import SimpleNamespace

    class EmptyThenFresh:
        def __init__(self):
            self.retry = False
            self.deadlines = []
            self.last_statistics = SimpleNamespace(fresh_exhausted=False)

        def generate(self, state, observation, request):
            self.deadlines.append(request.deadline)
            if not self.retry:
                return ActionGenerationResult((), attempts=0, exhausted=True)
            self.last_statistics.fresh_exhausted = True
            return ActionGenerationResult((ActionCandidate(_action(0)),), attempts=1, exhausted=True)

        def begin_retry(self, excluded):
            assert not excluded
            self.retry = True

    generator = EmptyThenFresh()
    search = MonteCarloTreeSearch(
        replace(_config(), max_depth=1, validate_root_before_search=True,
                rejected_root_retry_rounds=1),
        _RemainingObjectGenerator(), _ObjectPriorityScorer(),
        _DeterministicEvaluator(), _ZeroEstimator(), root_action_generator=generator,
    )
    result = search.search(MCTSNode(state=_state(), observation=()),
        MCTSSearchRequest(iterations=3, max_time_seconds=10.0, final_validation_reserve_seconds=1.0))
    assert result.selected is not None
    assert generator.retry
    assert len(generator.deadlines) == 2
    assert generator.deadlines[1] - generator.deadlines[0] == pytest.approx(1.0)


def test_initial_depth1_schedule_prevents_early_descendant_expansion() -> None:
    search = MonteCarloTreeSearch(
        MCTSConfig(
            max_depth=2,
            max_children=2,
            root_max_children=2,
            exploration_constant=0.0,
            discount=1.0,
            initial_depth1_steps=3,
            widening=ProgressiveWideningConfig(enabled=False),
        ),
        _RemainingObjectGenerator(),
        _ObjectPriorityScorer(),
        _DeterministicEvaluator(),
        _ZeroEstimator(),
    )

    result = search.search(
        MCTSNode(state=_state(), observation=()),
        MCTSSearchRequest(iterations=6),
    )

    assert result.effective_max_depth == 1
    assert result.root.children
    assert all(not child.children for child in result.root.children)


class _DepthAwareEstimator:
    def __init__(self):
        self.remaining_depths = []

    def estimate(self, state, observation):
        del state, observation
        raise AssertionError("depth-aware estimate should be used")

    def estimate_at_depth(
        self,
        state,
        observation,
        remaining_depth,
    ):
        del state, observation
        self.remaining_depths.append(remaining_depth)
        return 4.0


def test_coordinator_matches_legacy_depth_bounded_bootstrap_discount() -> None:
    estimator = _DepthAwareEstimator()
    search = MonteCarloTreeSearch(
        MCTSConfig(
            max_depth=2,
            max_children=1,
            root_max_children=1,
            exploration_constant=0.0,
            discount=0.5,
            widening=ProgressiveWideningConfig(enabled=False),
        ),
        _RemainingObjectGenerator(),
        _ObjectPriorityScorer(),
        _DeterministicEvaluator(),
        estimator,
    )

    result = search.search(
        MCTSNode(state=_state(), observation=()),
        MCTSSearchRequest(iterations=1),
    )

    child = result.root.children[0]
    assert estimator.remaining_depths == [2]
    assert child.mean_value == pytest.approx(6.0)
    assert result.root.mean_value == pytest.approx(3.0)


class _OffsetGenerator:
    def __init__(self):
        self.offsets = []
        self.priorities = []

    def generate(self, state, observation, request):
        del state, observation
        self.offsets.append(request.attempt_offset)
        self.priorities.append(request.sampling_priority)
        if request.attempt_offset == 0:
            return ActionGenerationResult(
                candidates=(),
                attempts=1,
                rejection_counts={"posegen_gap": 1},
            )
        return ActionGenerationResult(
            candidates=(ActionCandidate(_action(0)),),
            attempts=1,
            exhausted=True,
        )


def test_coordinator_resumes_generation_after_recorded_attempts() -> None:
    generator = _OffsetGenerator()
    config = MCTSConfig(
        max_depth=1,
        max_children=1,
        exploration_constant=0.0,
        discount=1.0,
        expansion_attempt_multiplier=2.0,
        widening=ProgressiveWideningConfig(enabled=False),
    )
    search = MonteCarloTreeSearch(
        config,
        generator,
        _ObjectPriorityScorer(),
        _DeterministicEvaluator(),
        _ZeroEstimator(),
    )

    root = MCTSNode(state=_state(), observation=())
    search.search(
        root,
        MCTSSearchRequest(
            iterations=2,
            root_sampling_priority=" lower_floor_first ",
        ),
    )

    assert generator.offsets == [0, 1]
    assert generator.priorities == ["lower_floor_first", "lower_floor_first"]
    assert root.expansion_attempts == 2
    assert root.rejected_expansions == {"posegen_gap": 1}
    assert len(root.children) == 1


def test_search_request_rejects_empty_sampling_priority() -> None:
    with pytest.raises(ValueError, match="root action sampling priority"):
        MCTSSearchRequest(iterations=1, root_sampling_priority=" ")


def test_search_request_requires_wall_budget_for_final_validation_reserve() -> None:
    with pytest.raises(ValueError, match="requires a maximum search time"):
        MCTSSearchRequest(
            iterations=1,
            final_validation_reserve_seconds=1.0,
        )
    with pytest.raises(ValueError, match="smaller than"):
        MCTSSearchRequest(
            iterations=1,
            max_time_seconds=1.0,
            final_validation_reserve_seconds=1.0,
        )


class _WrongLengthScorer:
    def score(self, state, observation, candidates):
        del state, observation, candidates
        return []


def test_coordinator_rejects_misaligned_prior_scores_before_tree_mutation() -> None:
    search = MonteCarloTreeSearch(
        _config(),
        _RemainingObjectGenerator(),
        _WrongLengthScorer(),
        _DeterministicEvaluator(),
        _ZeroEstimator(),
    )
    root = MCTSNode(state=_state(), observation=())

    with pytest.raises(ValueError, match="one value per candidate"):
        search.search(root, MCTSSearchRequest(iterations=1))

    assert root.children == []
    assert root.expansion_attempts == 0


class _WrongFidelityEvaluator(_DeterministicEvaluator):
    def evaluate(self, state, action, fidelity):
        result = super().evaluate(state, action, fidelity)
        return TransitionEvaluation(
            state=result.state,
            observation=result.observation,
            reward=result.reward,
            fidelity=EvaluationFidelity.LONG,
            terminal=result.terminal,
        )


def test_coordinator_rejects_wrong_evaluator_fidelity() -> None:
    search = MonteCarloTreeSearch(
        _config(),
        _RemainingObjectGenerator(),
        _ObjectPriorityScorer(),
        _WrongFidelityEvaluator(),
        _ZeroEstimator(),
    )

    with pytest.raises(ValueError, match="wrong fidelity"):
        search.search(
            MCTSNode(state=_state(), observation=()),
            MCTSSearchRequest(
                iterations=1,
                evaluation_fidelity=EvaluationFidelity.SHORT,
            ),
        )


def test_masked_action_adapter_preserves_valid_actions_and_reasons() -> None:
    actions = [
        _action(0),
        _action(1, diagnostics={"action_mask_reason": "outside_target"}),
    ]

    def sampler(state, observation, request):
        del state, observation, request
        return actions, np.array([0.0, -np.inf])

    generator = MaskedActionGeneratorAdapter(sampler)
    result = generator.generate(
        _state(),
        (),
        ActionGenerationRequest(max_candidates=2),
    )

    assert [candidate.action.object_index for candidate in result.candidates] == [0]
    assert result.attempts == 2
    assert result.rejection_counts == {"outside_target": 1}
    assert not result.exhausted


class _Transition:
    def __init__(self) -> None:
        self.state = _state()
        self.restored = None
        self.closed = False
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        self.state = _state()
        return self.state.copy()

    def restore(self, state):
        self.restored = state.copy()
        self.state = state.copy()

    def snapshot(self):
        return self.state.copy()

    def step(self, action):
        self.state = _state(
            [*self.state.placed_object_indices, action.object_index],
            actions=[*self.state.action_history, action],
        )
        return self.state.copy()

    def close(self):
        self.closed = True


class _ObservationBuilder:
    def build(self, state):
        return tuple(state.placed_object_indices)


class _RewardEvaluator:
    def compute(self, state):
        return 2.5, {"placed": len(state.placed_object_indices)}


def test_stacking_environment_evaluator_restores_and_steps_selected_profile() -> None:
    transition = _Transition()
    environment = StackingEnvironment(
        transition,
        _ObservationBuilder(),
        _RewardEvaluator(),
    )
    evaluator = StackingEnvironmentEvaluator(
        {EvaluationFidelity.SHORT: environment}
    )
    parent = _state()

    result = evaluator.evaluate(
        parent,
        _action(0),
        EvaluationFidelity.SHORT,
    )

    assert transition.restored is not None
    assert transition.reset_count == 1
    assert transition.restored.placed_object_indices == []
    assert result.state.placed_object_indices == [0]
    assert result.observation == (0,)
    assert result.reward == 2.5
    assert result.fidelity is EvaluationFidelity.SHORT
    assert result.diagnostics == {"placed": 1}

    evaluator.close()
    assert transition.closed


def test_environment_evaluator_rejects_outside_target_without_more_simulation() -> None:
    class OutsideReward:
        def compute(self, state):
            del state
            return 1.0, {
                "target_containment_fraction": 0.25,
                "target_containment_threshold": 0.5,
            }

    environment = StackingEnvironment(
        _Transition(),
        _ObservationBuilder(),
        OutsideReward(),
    )
    evaluator = StackingEnvironmentEvaluator(
        {EvaluationFidelity.SHORT: environment}
    )

    result = evaluator.evaluate(
        _state(), _action(0), EvaluationFidelity.SHORT
    )

    assert result.failed
    assert result.rejection_reason == "stone_outside_target"


def test_environment_evaluator_rejects_unsettled_simulation() -> None:
    class UnsettledTransition(_Transition):
        def step(self, action):
            state = super().step(action)
            state.simulation_settled = False
            self.state.simulation_settled = False
            return state

    environment = StackingEnvironment(
        UnsettledTransition(),
        _ObservationBuilder(),
        _RewardEvaluator(),
    )
    evaluator = StackingEnvironmentEvaluator(
        {EvaluationFidelity.SHORT: environment}
    )

    result = evaluator.evaluate(
        _state(), _action(0), EvaluationFidelity.SHORT
    )

    assert result.failed
    assert result.rejection_reason == "simulation_unsettled"


class _BatchEvaluator(_DeterministicEvaluator):
    def __init__(self):
        self.batch_sizes = []

    def evaluate_batch(self, states, actions, fidelity):
        self.batch_sizes.append(len(actions))
        return tuple(
            self.evaluate(state, action, fidelity)
            for state, action in zip(states, actions)
        )


def test_coordinator_batches_sibling_evaluation() -> None:
    evaluator = _BatchEvaluator()
    config = MCTSConfig(
        max_depth=1,
        max_children=2,
        exploration_constant=0.0,
        discount=1.0,
        generation_batch_size=2,
        evaluation_batch_size=2,
        widening=ProgressiveWideningConfig(enabled=False),
    )
    search = MonteCarloTreeSearch(
        config,
        _RemainingObjectGenerator(),
        _ObjectPriorityScorer(),
        evaluator,
        _ZeroEstimator(),
    )

    result = search.search(
        MCTSNode(state=_state(), observation=()),
        MCTSSearchRequest(iterations=1),
    )

    assert evaluator.batch_sizes == [2]
    assert result.evaluated_candidates == 2
    assert len(result.root.evaluated_children()) == 2
    assert result.short_simulation_batches == 1
    assert result.short_simulation_jobs == 2
    assert result.short_max_batch_size == 2
    assert result.long_simulation_jobs == 0


class _FinalEvaluator(_DeterministicEvaluator):
    def evaluate(self, state, action, fidelity):
        result = super().evaluate(state, action, fidelity)
        if fidelity is EvaluationFidelity.LONG and action.object_index == 0:
            result.state.failed = True
            return TransitionEvaluation(
                state=result.state,
                observation=result.observation,
                reward=-1.0,
                fidelity=fidelity,
                failed=True,
                rejection_reason="structural_failure",
            )
        reward = 9.0 if fidelity is EvaluationFidelity.LONG else result.reward
        return TransitionEvaluation(
            state=result.state,
            observation=result.observation,
            reward=reward,
            fidelity=fidelity,
            terminal=result.terminal,
        )


def test_final_long_validation_rejects_failure_and_selects_by_long_reward() -> None:
    config = MCTSConfig(
        max_depth=1,
        max_children=2,
        exploration_constant=0.0,
        discount=1.0,
        generation_batch_size=2,
        evaluation_batch_size=2,
        final_validation_candidates=2,
        widening=ProgressiveWideningConfig(enabled=False),
    )
    search = MonteCarloTreeSearch(
        config,
        _RemainingObjectGenerator(),
        _ObjectPriorityScorer(),
        _FinalEvaluator(),
        _ZeroEstimator(),
    )

    result = search.search(
        MCTSNode(state=_state(), observation=()),
        MCTSSearchRequest(
            iterations=2,
            final_validation_fidelity=EvaluationFidelity.LONG,
        ),
    )

    assert result.final_validation_attempts == 2
    assert result.final_validation_failures == 1
    assert result.long_simulation_batches == 2
    assert result.long_simulation_jobs == 2
    assert result.selected.action.object_index == 1
    assert result.selected.info["final_validation_reward"] == 9.0


def test_final_validation_stops_scheduling_after_wall_deadline(
    monkeypatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(
        "stacking_common.search.mcts.search.time.monotonic",
        lambda: clock[0],
    )

    class TimedFinalEvaluator(_FinalEvaluator):
        def evaluate(self, state, action, fidelity):
            result = super().evaluate(state, action, fidelity)
            if fidelity is EvaluationFidelity.LONG:
                clock[0] = 1.1
            return result

    config = MCTSConfig(
        max_depth=1,
        max_children=2,
        exploration_constant=0.0,
        discount=1.0,
        generation_batch_size=2,
        evaluation_batch_size=2,
        final_validation_candidates=2,
        widening=ProgressiveWideningConfig(enabled=False),
    )
    search = MonteCarloTreeSearch(
        config,
        _RemainingObjectGenerator(),
        _ObjectPriorityScorer(),
        TimedFinalEvaluator(),
        _ZeroEstimator(),
    )

    result = search.search(
        MCTSNode(state=_state(), observation=()),
        MCTSSearchRequest(
            iterations=1,
            max_time_seconds=1.0,
            final_validation_fidelity=EvaluationFidelity.LONG,
        ),
    )

    assert result.final_validation_attempts == 1
    assert result.final_validation_timed_out
    assert result.timed_out


class _RevalidationEvaluator:
    def evaluate(self, state, action, fidelity):
        pose = action.pose.copy()
        pose[0] += 0.2
        next_state = _state([action.object_index], actions=[action])
        next_state.object_poses[action.object_id] = pose
        return TransitionEvaluation(
            state=next_state,
            observation="revalidated",
            reward=3.0,
            fidelity=fidelity,
        )


def test_preserved_subtree_revalidation_clears_drifted_descendants() -> None:
    root = MCTSNode(state=_state(), observation=())
    action = _action(0)
    child = root.add_candidate(action)
    old_state = _state([0], actions=[action])
    old_state.object_poses[10] = action.pose.copy()
    child.set_result(old_state, "old", 1.0)
    grandchild = child.add_candidate(_action(1))
    grandchild.set_result(_state([0, 1]), "grandchild", 1.0)
    search = MonteCarloTreeSearch(
        _config(),
        _RemainingObjectGenerator(),
        _ObjectPriorityScorer(),
        _RevalidationEvaluator(),
        _ZeroEstimator(),
    )

    result = search.revalidate_children(root, max_drift=0.05)

    assert result.attempted == result.accepted == 1
    assert result.cleared_subtrees == 1
    assert child.children == []
    assert child.info["authoritative_fidelity"] == "long"


class _RootGateEvaluator(_DeterministicEvaluator):
    def __init__(self, *, reject_initial=()):
        self.reject_initial = set(reject_initial)
        self.calls = []

    def evaluate(self, state, action, fidelity):
        self.calls.append(
            (fidelity, tuple(state.placed_object_indices), action.object_index)
        )
        result = super().evaluate(state, action, fidelity)
        if (
            fidelity is EvaluationFidelity.LONG
            and not state.placed_object_indices
            and action.object_index in self.reject_initial
        ):
            result.state.failed = True
            return TransitionEvaluation(
                state=result.state,
                observation=result.observation,
                reward=-1.0,
                fidelity=fidelity,
                failed=True,
                rejection_reason="structural_failure",
            )
        return result


class _BatchRootGateEvaluator(_RootGateEvaluator):
    def __init__(self):
        super().__init__()
        self.batch_calls = []

    def evaluate(self, state, action, fidelity):
        result = super().evaluate(state, action, fidelity)
        return TransitionEvaluation(
            state=result.state,
            observation=result.observation,
            reward=result.reward,
            fidelity=fidelity,
            terminal=result.terminal,
            failed=result.failed,
            rejection_reason=result.rejection_reason,
            diagnostics={
                "simulation_settling_wall_seconds": 0.1,
                "simulation_structural_wall_seconds": 0.2,
            },
        )

    def evaluate_batch(self, states, actions, fidelity):
        self.batch_calls.append(
            (
                fidelity,
                tuple(tuple(state.placed_object_indices) for state in states),
            )
        )
        return tuple(
            self.evaluate(state, action, fidelity)
            for state, action in zip(states, actions)
        )


def _root_policy_config(**overrides) -> MCTSConfig:
    values = {
        "max_depth": 2,
        "max_children": 2,
        "root_max_children": 2,
        "exploration_constant": 0.0,
        "discount": 1.0,
        "generation_batch_size": 2,
        "evaluation_batch_size": 2,
        "validate_root_before_search": True,
        "root_lookahead": True,
        "final_validation_candidates": 2,
        "widening": ProgressiveWideningConfig(enabled=False),
    }
    values.update(overrides)
    return MCTSConfig(**values)


def test_root_gate_long_validates_before_descendant_search() -> None:
    evaluator = _RootGateEvaluator(reject_initial=(0,))
    search = MonteCarloTreeSearch(
        _root_policy_config(),
        _RemainingObjectGenerator(),
        _ObjectPriorityScorer(),
        evaluator,
        _ZeroEstimator(),
    )

    result = search.search(
        MCTSNode(state=_state(), observation=()),
        MCTSSearchRequest(
            iterations=1,
            final_validation_fidelity=EvaluationFidelity.LONG,
        ),
    )

    assert evaluator.calls[:2] == [
        (EvaluationFidelity.LONG, (), 0),
        (EvaluationFidelity.LONG, (), 1),
    ]
    assert evaluator.calls[2] == (EvaluationFidelity.SHORT, (1,), 0)
    assert result.root_validation.attempted == 2
    assert result.root_validation.accepted == 1
    assert result.root_validation.rejected == 1
    assert result.final_validation_attempts == 0
    assert result.selected is not None
    assert result.selected.action.object_index == 1


def test_root_lookahead_covers_every_authoritative_root_first() -> None:
    evaluator = _RootGateEvaluator()
    search = MonteCarloTreeSearch(
        _root_policy_config(),
        _RemainingObjectGenerator(),
        _ObjectPriorityScorer(),
        evaluator,
        _ZeroEstimator(),
    )

    result = search.search(
        MCTSNode(state=_state(), observation=()),
        MCTSSearchRequest(
            iterations=2,
            final_validation_fidelity=EvaluationFidelity.LONG,
        ),
    )

    assert result.root_lookahead.eligible == 2
    assert result.root_lookahead.attempted == 2
    assert result.root_lookahead.generated == 2
    assert result.root_lookahead.evaluated == 2
    assert all(child.evaluated_children() for child in result.root.children)


def test_root_lookahead_batches_independent_root_continuations() -> None:
    evaluator = _BatchRootGateEvaluator()
    search = MonteCarloTreeSearch(
        _root_policy_config(),
        _RemainingObjectGenerator(),
        _ObjectPriorityScorer(),
        evaluator,
        _ZeroEstimator(),
    )

    result = search.search(
        MCTSNode(state=_state(), observation=()),
        MCTSSearchRequest(
            iterations=2,
            final_validation_fidelity=EvaluationFidelity.LONG,
        ),
    )

    short_batches = [
        states
        for fidelity, states in evaluator.batch_calls
        if fidelity is EvaluationFidelity.SHORT
    ]
    assert short_batches == [((0,), (1,))]
    assert result.completed_iterations == 2
    assert result.short_simulation_batches == 1
    assert result.short_max_batch_size == 2
    assert np.isclose(result.short_settling_job_seconds, 0.2)
    assert np.isclose(result.short_structural_job_seconds, 0.4)
    assert np.isclose(result.long_settling_job_seconds, 0.2)
    assert np.isclose(result.long_structural_job_seconds, 0.4)


def test_nonroot_frontiers_batch_action_generation_across_root_branches() -> None:
    class InventoryPreservingBatchEvaluator(_BatchEvaluator):
        def evaluate(self, state, action, fidelity):
            placed = [*state.placed_object_indices, action.object_index]
            next_state = State(
                object_ids=list(state.object_ids),
                placed_object_indices=placed,
                object_poses={},
                trajectories=[],
                action_history=[*state.action_history, action],
                terminated=len(placed) == len(state.object_ids),
            )
            return TransitionEvaluation(
                state=next_state,
                observation=tuple(placed),
                reward=5.0 if action.object_index == 0 else 1.0,
                fidelity=fidelity,
                terminal=next_state.terminated,
            )

    generator = _BatchRemainingObjectGenerator()
    evaluator = InventoryPreservingBatchEvaluator()
    search = MonteCarloTreeSearch(
        MCTSConfig(
            max_depth=3,
            max_children=2,
            root_max_children=3,
            exploration_constant=0.0,
            discount=1.0,
            generation_batch_size=2,
            evaluation_batch_size=3,
            validate_root_before_search=True,
            root_lookahead=True,
            widening=ProgressiveWideningConfig(enabled=False),
        ),
        generator,
        _ObjectPriorityScorer(),
        evaluator,
        _ZeroEstimator(),
    )
    root_state = State(
        object_ids=[10, 20, 30],
        placed_object_indices=[],
        object_poses={},
        trajectories=[],
        action_history=[],
    )

    result = search.search(
        MCTSNode(state=root_state, observation=()),
        MCTSSearchRequest(iterations=9),
    )

    assert (1, 1, 1) in generator.batch_depths
    # Ordinary batches follow global PUCT; they need not cover every branch.
    assert any(
        len(depths) > 1 and set(depths) == {2}
        for depths in generator.batch_depths
    )
    assert result.pose_generation_max_node_batch == 3
    assert result.descendant_frontier_batches == 2
    assert result.descendant_frontier_iterations == 6
    assert result.completed_iterations == 9


class _DescendantEvaluator(_DeterministicEvaluator):
    def evaluate(self, state, action, fidelity):
        result = super().evaluate(state, action, fidelity)
        first = state.placed_object_indices[0]
        reward = -5.0 if first == 0 else 20.0
        return TransitionEvaluation(
            state=result.state,
            observation=result.observation,
            reward=reward,
            fidelity=fidelity,
            terminal=result.terminal,
        )


def _final_selection_root() -> MCTSNode:
    root = MCTSNode(state=_state(terminated=True), observation=())
    for root_index, mean, tail_reward in ((0, 10.0, -5.0), (1, 5.0, 20.0)):
        root_action = _action(root_index)
        child = root.add_candidate(root_action)
        child.set_result(
            _state([root_index], actions=[root_action]),
            (root_index,),
            0.0,
            info={"authoritative_fidelity": "long"},
        )
        child.record_visit(mean)
        descendant_index = 1 - root_index
        descendant_action = _action(descendant_index)
        descendant = child.add_candidate(descendant_action)
        descendant.set_result(
            _state(
                [root_index, descendant_index],
                terminated=True,
                actions=[root_action, descendant_action],
            ),
            (root_index, descendant_index),
            tail_reward,
            terminal=True,
        )
    root.expansion_exhausted = True
    return root


@pytest.mark.parametrize(
    ("selection", "expected_index", "descendant_attempts"),
    [
        (FinalSelectionConfig(mode="mean"), 0, 0),
        (
            FinalSelectionConfig(
                mode="max_descendant",
                validate_descendant_path=True,
            ),
            1,
            2,
        ),
        (
            FinalSelectionConfig(
                mode="mixed",
                max_weight=0.75,
                validate_descendant_path=True,
            ),
            1,
            2,
        ),
    ],
)
def test_final_selection_supports_mean_max_and_mixed_modes(
    selection,
    expected_index,
    descendant_attempts,
) -> None:
    search = MonteCarloTreeSearch(
        _root_policy_config(
            validate_root_before_search=False,
            root_lookahead=False,
            final_selection=selection,
        ),
        _RemainingObjectGenerator(),
        _ObjectPriorityScorer(),
        _DescendantEvaluator(),
        _ZeroEstimator(),
    )

    result = search.search(
        _final_selection_root(),
        MCTSSearchRequest(
            iterations=1,
            reset_statistics=False,
            final_validation_fidelity=EvaluationFidelity.LONG,
        ),
    )

    assert result.selected is not None
    assert result.selected.action.object_index == expected_index
    assert result.ranked_candidates[0] is result.selected
    assert {node.action.object_index for node in result.ranked_candidates} == {0, 1}
    assert result.descendant_validation_attempts == descendant_attempts
    assert result.final_selection_mode == selection.mode


class _RetryingRootGenerator:
    def __init__(self):
        self.retry_round = 0
        self.excluded = ()

    def generate(self, state, observation, request):
        del state, observation
        if self.retry_round == 0:
            candidates = tuple(
                ActionCandidate(_action(index))
                for index in range(2)
            )
        else:
            action = _action(0)
            action.pose[0] = 2.0
            action.init_pose[0] = 2.0
            candidates = (ActionCandidate(action),)
        selected = candidates[: request.max_candidates]
        return ActionGenerationResult(
            candidates=selected,
            attempts=len(selected),
            exhausted=True,
        )

    def begin_retry(self, excluded_actions):
        self.retry_round += 1
        self.excluded = tuple(excluded_actions)


class _RetryEvaluator(_DeterministicEvaluator):
    def evaluate(self, state, action, fidelity):
        result = super().evaluate(state, action, fidelity)
        if fidelity is EvaluationFidelity.LONG and action.pose[0] < 2.0:
            result.state.failed = True
            return TransitionEvaluation(
                state=result.state,
                observation=result.observation,
                reward=-1.0,
                fidelity=fidelity,
                failed=True,
                rejection_reason="structural_failure",
            )
        return result


def test_all_rejected_roots_trigger_diversified_retry() -> None:
    generator = _RetryingRootGenerator()
    search = MonteCarloTreeSearch(
        _root_policy_config(
            max_depth=1,
            root_lookahead=False,
            rejected_root_retry_rounds=1,
        ),
        generator,
        _ObjectPriorityScorer(),
        _RetryEvaluator(),
        _ZeroEstimator(),
        root_action_generator=generator,
    )

    result = search.search(
        MCTSNode(state=_state(), observation=()),
        MCTSSearchRequest(
            iterations=1,
            final_validation_fidelity=EvaluationFidelity.LONG,
        ),
    )

    assert len(generator.excluded) == 2
    assert result.root_retry.rounds == 1
    assert result.root_retry.retired == 2
    assert result.root_retry.generated == 1
    assert result.root_retry.accepted == 1
    assert result.selected is not None
    assert result.selected.action.pose[0] == 2.0


def test_early_lookahead_prefers_a_placeable_continuation_over_greedy_dead_end():
    from dataclasses import replace

    class PackingGenerator(_RemainingObjectGenerator):
        def generate(self, state, observation, request):
            if state.placed_object_indices == [0]:
                return ActionGenerationResult((), attempts=0, exhausted=True)
            remaining = [i for i in range(2) if i not in state.placed_object_indices]
            selected = remaining[request.attempt_offset:request.attempt_offset + request.max_candidates]
            return ActionGenerationResult(tuple(ActionCandidate(_action(i)) for i in selected),
                attempts=len(selected), exhausted=request.attempt_offset + len(selected) >= len(remaining))

    def selected_object(warmup):
        search = MonteCarloTreeSearch(
            replace(_config(), initial_depth1_steps=warmup, root_lookahead=True,
                    validate_root_before_search=True, exploration_constant=2.0),
            PackingGenerator(), _ObjectPriorityScorer(),
            _DeterministicEvaluator(), _ZeroEstimator(),
        )
        result = search.search(MCTSNode(state=_state(), observation=()),
            MCTSSearchRequest(iterations=200))
        return result.selected.action.object_index

    assert selected_object(3) == 0  # Larger immediate reward, but blocks the next stone.
    assert selected_object(0) == 1
