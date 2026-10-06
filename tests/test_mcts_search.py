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


class _RetryPathEvaluator:
    worker_count = 4

    def __init__(self, *, fail_descendants=False, clock=None):
        self.fail_descendants = fail_descendants
        self.clock = clock
        self.batches = []

    def evaluate(self, state, action, fidelity):
        next_state = state.copy()
        next_state.placed_object_indices.append(action.object_index)
        next_state.action_history.append(action)
        depth = len(next_state.placed_object_indices)
        failed = (depth == 1 and action.pose[0] < 2.0) or (
            depth > 1 and self.fail_descendants and fidelity is EvaluationFidelity.LONG
        )
        next_state.failed = failed
        return TransitionEvaluation(
            state=next_state,
            observation=tuple(next_state.placed_object_indices),
            reward=-1.0 if failed else 1.0,
            fidelity=fidelity,
            failed=failed,
            rejection_reason="structural_failure" if failed else None,
        )

    def evaluate_batch(self, states, actions, fidelity):
        depths = tuple(len(state.placed_object_indices) + 1 for state in states)
        self.batches.append((fidelity, depths))
        results = tuple(self.evaluate(s, a, fidelity) for s, a in zip(states, actions))
        if self.clock is not None and 2 in depths:
            self.clock[0] = 11.0
        return results


def _retry_path_search(depth, evaluator):
    state = _state()
    state.object_ids = [10, 20, 30, 40, 50]
    search = MonteCarloTreeSearch(
        _root_policy_config(
            max_depth=4,
            evaluation_batch_size=4,
            rejected_root_retry_rounds=1,
            final_selection=FinalSelectionConfig(
                mode="mixed", max_weight=0.5,
                validate_descendant_path=True,
                validate_descendant_depth=depth,
                require_validated_descendant=True,
            ),
        ),
        _RemainingObjectGenerator(),
        _ObjectPriorityScorer(),
        evaluator,
        _ZeroEstimator(),
        root_action_generator=_RetryingRootGenerator(),
    )
    return search.search(
        MCTSNode(state=state, observation=()),
        MCTSSearchRequest(
            iterations=1, max_time_seconds=10.0,
            final_validation_fidelity=EvaluationFidelity.LONG,
        ),
    )


@pytest.mark.parametrize("depth", [1, 2, 3, 5])
def test_retry_builds_required_descendant_path(depth):
    evaluator = _RetryPathEvaluator()
    result = _retry_path_search(depth, evaluator)

    assert result.root_retry.rounds == 1
    assert result.selected is not None
    assert result.selected.action.pose[0] == 2.0
    assert result.selected.info["validated_descendant_path_available"]
    required = min(depth, 3)
    assert result.selected.info["validated_descendant_path_return"] == 1 + required
    assert max(d for _, batch in evaluator.batches for d in batch) == 1 + required
    if depth > 1:
        assert any(f is EvaluationFidelity.SHORT and batch == (3, 3, 3, 3)
                   for f, batch in evaluator.batches)


def test_retry_depth_extension_preserves_physical_validation():
    result = _retry_path_search(2, _RetryPathEvaluator(fail_descendants=True))

    assert result.selected is None
    assert result.descendant_validation_failures > 0


def test_retry_depth_extension_stops_at_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        "stacking_common.search.mcts.search.time.monotonic", lambda: clock[0]
    )
    evaluator = _RetryPathEvaluator(clock=clock)
    result = _retry_path_search(2, evaluator)

    assert result.selected is None
    assert result.timed_out
    assert max(d for _, batch in evaluator.batches for d in batch) == 2


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


class _BatchedLongEvaluator:
    """Long validation in recorded batches; the path 0 -> 1 fails."""

    worker_count = 8
    rewards = {(0, 1): 9.0, (0, 2): 3.0, (1, 0): 8.0, (1, 2): 2.0}

    def __init__(self):
        self.batches = []

    def evaluate(self, state, action, fidelity):
        self.batches.append(1)  # The search calls evaluate for single jobs.
        return self._result(state, action, fidelity)

    def _result(self, state, action, fidelity):
        placed = [*state.placed_object_indices, action.object_index]
        failed = placed == [0, 1]
        next_state = State(
            object_ids=[10, 20, 30], placed_object_indices=placed, object_poses={},
            trajectories=[], action_history=[*state.action_history, action], failed=failed,
        )
        return TransitionEvaluation(
            state=next_state, observation=tuple(placed), reward=self.rewards[tuple(placed)],
            fidelity=fidelity, failed=failed,
        )

    def evaluate_batch(self, states, actions, fidelity):
        self.batches.append(len(actions))
        return tuple(self._result(s, a, fidelity) for s, a in zip(states, actions))


def test_descendant_validation_batches_roots_and_keeps_serial_order() -> None:
    def state(placed=(), actions=()):
        return State(object_ids=[10, 20, 30], placed_object_indices=list(placed),
                     object_poses={}, trajectories=[], action_history=list(actions),
                     terminated=not placed)

    root = MCTSNode(state=state(), observation=())
    for root_index in (0, 1):
        root_action = _action(root_index)
        child = root.add_candidate(root_action)
        child.set_result(state([root_index], [root_action]), (root_index,), 0.0,
                         info={"authoritative_fidelity": "long"})
        child.record_visit(0.0)
        for index in (0, 1, 2):
            if index == root_index:
                continue
            # Short-fidelity results: each root's best-known descendant is tried first.
            action = _action(index)
            descendant = child.add_candidate(action)
            descendant.set_result(state([root_index, index], [root_action, action]),
                                  (root_index, index),
                                  _BatchedLongEvaluator.rewards[(root_index, index)])
    root.expansion_exhausted = True
    evaluator = _BatchedLongEvaluator()
    search = MonteCarloTreeSearch(
        _root_policy_config(
            validate_root_before_search=False, root_lookahead=False, evaluation_batch_size=4,
            final_selection=FinalSelectionConfig(
                mode="max_descendant", validate_descendant_path=True,
                require_validated_descendant=True),
        ),
        _RemainingObjectGenerator(), _ObjectPriorityScorer(), evaluator, _ZeroEstimator(),
    )

    result = search.search(root, MCTSSearchRequest(
        iterations=1, reset_statistics=False,
        final_validation_fidelity=EvaluationFidelity.LONG))

    # Both roots validate all their children in one shared round. Root 0
    # still falls back to its second child after 0 -> 1 fails, as a
    # one-child-at-a-time walk would.
    assert evaluator.batches == [4]
    assert result.descendant_validation_attempts == 4
    assert result.descendant_validation_failures == 1
    assert result.selected.action.object_index == 1
    scores = {node.action.object_index: node.info["validated_descendant_path_return"]
              for node in result.ranked_candidates}
    assert scores == {0: 3.0, 1: 8.0}


@pytest.mark.parametrize(("long_failures", "path_return"), [
    ((), 9.0),        # The known child passes and forms the path.
    (((0, 1),), None),  # It fails; the short failure is not retried, so no path remains.
])
def test_descendant_validation_skips_children_that_failed_short(long_failures, path_return) -> None:
    class Evaluator(_BatchedLongEvaluator):
        def _result(self, state, action, fidelity):
            result = super()._result(state, action, fidelity)
            placed = tuple(result.state.placed_object_indices)
            failed = placed in long_failures
            result.state.failed = failed
            return TransitionEvaluation(state=result.state, observation=result.observation,
                                        reward=result.reward, fidelity=fidelity, failed=failed)

    def state(placed=(), actions=(), failed=False):
        return State(object_ids=[10, 20, 30], placed_object_indices=list(placed),
                     object_poses={}, trajectories=[], action_history=list(actions),
                     terminated=not placed, failed=failed)

    root = MCTSNode(state=state(), observation=())
    root_action = _action(0)
    child = root.add_candidate(root_action)
    child.set_result(state([0], [root_action]), (0,), 0.0, info={"authoritative_fidelity": "long"})
    child.record_visit(0.0)
    for index, failed in ((1, False), (2, True)):
        action = _action(index)
        descendant = child.add_candidate(action)
        descendant.set_result(state([0, index], [root_action, action], failed=failed),
                              (0, index), _BatchedLongEvaluator.rewards[(0, index)])
    root.expansion_exhausted = True
    evaluator = Evaluator()
    search = MonteCarloTreeSearch(
        _root_policy_config(
            validate_root_before_search=False, root_lookahead=False, evaluation_batch_size=4,
            final_selection=FinalSelectionConfig(
                mode="max_descendant", validate_descendant_path=True,
                require_validated_descendant=True),
        ),
        _RemainingObjectGenerator(), _ObjectPriorityScorer(), evaluator, _ZeroEstimator(),
    )

    result = search.search(root, MCTSSearchRequest(
        iterations=1, reset_statistics=False,
        final_validation_fidelity=EvaluationFidelity.LONG))

    assert evaluator.batches == [1]  # The short-failed child is never long-validated.
    if path_return is None:
        assert result.selected is None
        assert child.rejection_reason == "missing_validated_descendant"
    else:
        assert result.selected.info["validated_descendant_path_return"] == path_return


class _FloorFallbackGenerator(_RemainingObjectGenerator):
    def __init__(self, *, reject_floor=False):
        self.relaxed = False
        self.reject_floor = reject_floor
        self.calls = []
        self.retries = 0

    def begin_retry(self, excluded_actions):
        self.retries += 1

    def floor_fill_fallback(self):
        from contextlib import contextmanager

        @contextmanager
        def relax():
            self.relaxed = True
            try:
                yield True
            finally:
                self.relaxed = False
        return relax()

    def generate(self, state, observation, request):
        self.calls.append((self.relaxed, request.sampling_priority))
        if self.reject_floor and not self.relaxed:
            return ActionGenerationResult((), attempts=1, exhausted=True,
                rejection_counts={"active_floor_above_active_layer": 1})
        result = super().generate(state, observation, request)
        for candidate in result.candidates:
            if self.relaxed:
                candidate.action.diagnostics['floor_fill_fallback'] = True
        return result


class _FloorFallbackEvaluator:
    def __init__(self, *, strict_succeeds=False, fail_relaxed=False, fail_descendants=False, clock=None):
        self.strict_succeeds = strict_succeeds
        self.fail_relaxed = fail_relaxed
        self.fail_descendants = fail_descendants
        self.clock = clock
        self.calls = []

    def evaluate(self, state, action, fidelity):
        relaxed = bool(action.diagnostics.get('floor_fill_fallback'))
        self.calls.append((relaxed, fidelity))
        failed = (self.fail_relaxed if relaxed else not self.strict_succeeds)
        failed |= self.fail_descendants and bool(state.placed_object_indices)
        next_state = state.copy()
        next_state.placed_object_indices.append(action.object_index)
        next_state.object_poses[action.object_id] = action.pose.copy()
        next_state.action_history.append(action.copy())
        next_state.failed = failed
        if self.clock is not None:
            self.clock[0] = 11.
        return TransitionEvaluation(next_state, (), -1. if failed else 1., fidelity,
            failed=failed, rejection_reason='target_motion_limit' if failed else None)


def _floor_fallback_search(generator, evaluator, *, depth=1):
    return MonteCarloTreeSearch(
        _root_policy_config(max_depth=depth, root_lookahead=True, rejected_root_retry_rounds=2,
            final_selection=FinalSelectionConfig(mode='mixed', max_weight=.5,
                validate_descendant_path=True, validate_descendant_depth=2,
                require_validated_descendant=True)),
        generator, _ObjectPriorityScorer(), evaluator, _ZeroEstimator(),
        root_action_generator=generator,
    )


@pytest.mark.parametrize('reject_floor', [False, True])
def test_floor_fallback_finds_valid_placement_after_strict_exhaustion(reject_floor):
    from stacking_common.workflows.mcts_reporting import search_statistics
    generator = _FloorFallbackGenerator(reject_floor=reject_floor)
    evaluator = _FloorFallbackEvaluator()
    search = _floor_fallback_search(generator, evaluator)
    result = search.search(
        MCTSNode(state=_state(), observation=()), MCTSSearchRequest(iterations=1,
            root_sampling_priority='lower_floor_first', deeper_sampling_priority='lower_floor_first',
            final_validation_fidelity=EvaluationFidelity.LONG))
    assert result.selected is not None
    assert result.floor_fill_fallback.attempted and result.floor_fill_fallback.selected
    assert result.floor_fill_fallback.accepted == 2
    assert result.selected.action.diagnostics['floor_fill_fallback']
    assert generator.calls[0] == (False, 'lower_floor_first')
    assert all(priority is None for relaxed, priority in generator.calls if relaxed)
    assert not generator.relaxed
    assert all(f is EvaluationFidelity.LONG for _, f in evaluator.calls)
    saved = search_statistics(result, 0., search, step=1)
    assert saved['floor_fill_fallback'] == {
        'attempted': True, 'pools': 1, 'generated': 2, 'accepted': 2, 'selected': True,
    }
    if not reject_floor:
        assert result.root_retry.rounds == 2


def test_floor_fallback_does_not_run_when_strict_search_succeeds():
    generator = _FloorFallbackGenerator()
    result = _floor_fallback_search(generator, _FloorFallbackEvaluator(strict_succeeds=True)).search(
        MCTSNode(state=_state(), observation=()), MCTSSearchRequest(iterations=1,
            final_validation_fidelity=EvaluationFidelity.LONG))
    assert result.selected is not None
    assert not result.floor_fill_fallback.attempted
    assert not any(relaxed for relaxed, _ in generator.calls)


@pytest.mark.parametrize('fail_descendants', [False, True])
def test_floor_fallback_preserves_physics_and_required_continuations(fail_descendants):
    generator = _FloorFallbackGenerator()
    evaluator = _FloorFallbackEvaluator(fail_relaxed=not fail_descendants,
                                       fail_descendants=fail_descendants)
    result = _floor_fallback_search(generator, evaluator, depth=3 if fail_descendants else 1).search(
        MCTSNode(state=_state(), observation=()), MCTSSearchRequest(iterations=1,
            final_validation_fidelity=EvaluationFidelity.LONG))
    assert result.selected is None
    assert result.floor_fill_fallback.attempted
    assert not result.floor_fill_fallback.selected
    assert result.floor_fill_fallback.pools == 3  # First pool plus two retry pools.
    assert not generator.relaxed
    if fail_descendants:
        # Relaxed-floor descendants fail at short search fidelity, so none is
        # revalidated long.
        assert (True, EvaluationFidelity.SHORT) in evaluator.calls
        assert result.descendant_validation_failures == 0
    else:
        assert result.floor_fill_fallback.accepted == 0


def test_floor_fallback_respects_search_deadline(monkeypatch):
    clock = [0.]
    monkeypatch.setattr('stacking_common.search.mcts.search.time.monotonic', lambda: clock[0])
    generator = _FloorFallbackGenerator()
    result = _floor_fallback_search(generator, _FloorFallbackEvaluator(clock=clock)).search(
        MCTSNode(state=_state(), observation=()), MCTSSearchRequest(iterations=1,
            max_time_seconds=10., final_validation_fidelity=EvaluationFidelity.LONG))
    assert result.selected is None and result.timed_out
    assert not result.floor_fill_fallback.attempted
    assert not any(relaxed for relaxed, _ in generator.calls)


def test_next_placement_rebuilds_relaxed_continuations_under_strict_policy():
    generator = _FloorFallbackGenerator()
    evaluator = _FloorFallbackEvaluator()
    search = _floor_fallback_search(generator, evaluator, depth=3)
    state = State([10, 20, 30, 40], [], {}, [], [])
    request = MCTSSearchRequest(iterations=1, final_validation_fidelity=EvaluationFidelity.LONG)
    first = search.search(MCTSNode(state=state, observation=()), request)
    assert first.selected is not None and first.selected.children
    retained = tuple(first.selected.children)
    calls_before = len(generator.calls)
    second = search.search(first.selected, request)
    assert generator.calls[calls_before][0] is False
    assert all(child.parent is None for child in retained)
    assert second.floor_fill_fallback.attempted


def test_floor_fallback_retries_fresh_pools_until_one_qualifies():
    class Generator(_FloorFallbackGenerator):
        relaxed_pools = 0
        excluded = None

        def begin_retry(self, excluded_actions):
            super().begin_retry(excluded_actions)
            if self.relaxed:
                self.relaxed_pools += 1
                self.excluded = tuple(excluded_actions)

        def generate(self, state, observation, request):
            result = super().generate(state, observation, request)
            for candidate in result.candidates:
                candidate.action.diagnostics['relaxed_pool'] = self.relaxed_pools
            return result

    class Evaluator(_FloorFallbackEvaluator):
        def evaluate(self, state, action, fidelity):
            result = super().evaluate(state, action, fidelity)
            if action.diagnostics.get('relaxed_pool') != 1:
                return result
            # Every candidate of the first relaxed pool fails physically.
            result.state.failed = True
            return TransitionEvaluation(result.state, (), -1., fidelity, failed=True,
                                        rejection_reason='target_motion_limit')

    generator = Generator()
    result = _floor_fallback_search(generator, Evaluator()).search(
        MCTSNode(state=_state(), observation=()), MCTSSearchRequest(iterations=1,
            final_validation_fidelity=EvaluationFidelity.LONG))
    assert result.selected is not None
    assert result.selected.action.diagnostics['relaxed_pool'] == 2
    assert result.floor_fill_fallback.pools == 2 and result.floor_fill_fallback.selected
    # The second pool excludes every candidate of the first.
    assert generator.excluded and all(
        action.diagnostics['relaxed_pool'] == 1 for action in generator.excluded)
