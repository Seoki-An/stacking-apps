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
        ][request.attempt_offset : request.attempt_offset + request.max_candidates]
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
    assert root.visits == 9  # Pseudocount, two root seeds, six iterations.
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
        replace(_config(), max_depth=1, rejected_root_retry_rounds=1),
        _RemainingObjectGenerator(), _ObjectPriorityScorer(),
        _DeterministicEvaluator(), _ZeroEstimator(), root_action_generator=generator,
    )
    result = search.search(MCTSNode(state=_state(), observation=()),
        MCTSSearchRequest(iterations=3, max_time_seconds=10.0, ))
    assert result.selected is not None
    assert generator.retry
    assert len(generator.deadlines) == 2
    assert generator.deadlines[1] == generator.deadlines[0]


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


def test_coordinator_discounts_depth_bounded_bootstrap_once() -> None:
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
    assert estimator.remaining_depths == [1]
    assert child.mean_value == pytest.approx(6.25)  # Includes root admission.
    assert result.root.value_sum == pytest.approx(6.25)
    assert result.root.mean_value == pytest.approx(6.25 / 3)


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


def test_environment_evaluator_accepts_settled_pose_without_floor_reclassification():
    class SettledTransition(_Transition):
        def step(self, action):
            state = super().step(action)
            state.object_poses[action.object_id] = action.pose.copy()
            state.object_poses[action.object_id][2] += 0.2
            self.state = state.copy()
            return state
    environment = StackingEnvironment(SettledTransition(), _ObservationBuilder(), _RewardEvaluator())
    evaluator = StackingEnvironmentEvaluator({EvaluationFidelity.LONG: environment})
    result = evaluator.evaluate(_state(), _action(0), EvaluationFidelity.LONG)
    assert not result.failed
    assert result.rejection_reason is None
    assert 'settled_active_floor_checked' not in result.diagnostics



class _BatchEvaluator(_DeterministicEvaluator):
    worker_count = 2

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
    assert result.long_simulation_batches == 1
    assert result.long_simulation_jobs == 2
    assert result.max_evaluation_batch_size == 2
    assert result.short_simulation_jobs == 0


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
        "root_lookahead": True,
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
        ),
    )

    assert evaluator.calls[:2] == [
        (EvaluationFidelity.LONG, (), 0),
        (EvaluationFidelity.LONG, (), 1),
    ]
    assert evaluator.calls[2] == (EvaluationFidelity.LONG, (1,), 0)
    assert result.root_validation.attempted == 2
    assert result.root_validation.accepted == 1
    assert result.root_validation.rejected == 1
    assert result.final_validation_attempts == 0
    assert result.selected is not None
    assert result.selected.action.object_index == 1


def test_reset_reseeds_retained_authoritative_roots_with_their_rewards() -> None:
    search = MonteCarloTreeSearch(
        _root_policy_config(max_depth=1),
        _RemainingObjectGenerator(),
        _ObjectPriorityScorer(),
        _RootGateEvaluator(),
        _ZeroEstimator(),
    )
    root = MCTSNode(state=_state(), observation=())
    request = MCTSSearchRequest(iterations=1)
    search.search(root, request)
    retained = tuple(root.evaluated_children())
    assert retained

    # Repeat the reset to ensure existing seed markers do not suppress backup.
    for _ in range(2):
        search.search(root, request)
        assert tuple(root.evaluated_children()) == retained
        for child in retained:
            assert child.visits >= 2  # Reset pseudocount and reward backup.
            assert child.value_sum == pytest.approx((child.visits - 1) * child.reward)
        assert sum(child.visits for child in retained) == 2 * len(retained) + 1
        assert root.visits == 1 + len(retained) + 1


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
        ),
    )

    batches = [states for fidelity, states in evaluator.batch_calls
               if fidelity is EvaluationFidelity.LONG]
    assert batches == [((0,), (1,))]
    assert result.completed_iterations == 2
    assert result.long_simulation_batches == 3
    assert result.max_evaluation_batch_size == 2
    assert result.short_simulation_jobs == 0
    assert np.isclose(result.long_settling_job_seconds, 0.4)
    assert np.isclose(result.long_structural_job_seconds, 0.8)


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
        len(depths) > 1 and 2 in depths
        for depths in generator.batch_depths
    )
    assert result.pose_generation_max_node_batch == 3
    assert result.descendant_frontier_batches == 2
    assert result.descendant_frontier_iterations == 5  # Duplicate proposals consume no visit.
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
    ("selection", "expected_index"),
    [
        (FinalSelectionConfig(max_weight=0.0), 0),
        (FinalSelectionConfig(max_weight=1.0), 1),
        (FinalSelectionConfig(max_weight=0.75), 1),
    ],
)
def test_final_selection_blends_mean_and_path_with_weight(
    selection,
    expected_index,
) -> None:
    search = MonteCarloTreeSearch(
        _root_policy_config(
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
            ),
    )

    assert result.selected is not None
    assert result.selected.action.object_index == expected_index
    assert result.ranked_candidates[0] is result.selected
    assert {node.action.object_index for node in result.ranked_candidates} == {0, 1}
    assert result.descendant_validation_attempts == 0
    assert result.final_selection_max_weight == selection.max_weight
    assert result.selected.info['final_selection_score'] == pytest.approx(
        {0.0: 10.0, 1.0: 20.0, 0.75: 16.25}[selection.max_weight])
    assert result.selected.info['final_selection_max_weight'] == selection.max_weight


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
                max_weight=0.5,
                descendant_depth=depth,
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
            ),
    )


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
                    exploration_constant=2.0),
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


class _CornerFallbackGenerator(_RemainingObjectGenerator):
    def __init__(self, *, reject_floor=False):
        self.relaxed = False
        self.reject_floor = reject_floor
        self.calls = []
        self.retries = 0

    def begin_retry(self, excluded_actions):
        self.retries += 1

    def corner_seed_fallback(self):
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


class _CornerFallbackEvaluator:
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


def _corner_fallback_search(generator, evaluator, *, depth=1):
    return MonteCarloTreeSearch(
        _root_policy_config(max_depth=depth, root_lookahead=True, rejected_root_retry_rounds=2,
            final_selection=FinalSelectionConfig(max_weight=.5,
                descendant_depth=2,
                )),
        generator, _ObjectPriorityScorer(), evaluator, _ZeroEstimator(),
        root_action_generator=generator,
    )


@pytest.mark.parametrize('reject_floor', [False, True])
def test_corner_fallback_finds_valid_placement_after_seed_exhaustion(reject_floor):
    from stacking_common.workflows.mcts_reporting import search_statistics
    generator = _CornerFallbackGenerator(reject_floor=reject_floor)
    evaluator = _CornerFallbackEvaluator()
    search = _corner_fallback_search(generator, evaluator)
    result = search.search(
        MCTSNode(state=_state(), observation=()), MCTSSearchRequest(iterations=1,
            root_sampling_priority='lower_floor_first', deeper_sampling_priority='lower_floor_first',
            ))
    assert result.selected is not None
    assert result.floor_fill_fallback.attempted and result.floor_fill_fallback.selected
    assert result.floor_fill_fallback.accepted == 2
    assert result.selected.action.diagnostics['floor_fill_fallback']
    assert generator.calls[0] == (False, 'lower_floor_first')
    assert all(priority == 'lower_floor_first' for relaxed, priority in generator.calls if relaxed)
    assert not generator.relaxed
    assert all(f is EvaluationFidelity.LONG for _, f in evaluator.calls)
    saved = search_statistics(result, 0., search, step=1)
    assert saved['floor_fill_fallback'] == {
        'attempted': True, 'pools': 1, 'generated': 2, 'accepted': 2, 'selected': True,
        'policy': 'corner_seed',
    }
    if not reject_floor:
        assert result.root_retry.rounds == 2


@pytest.mark.parametrize("recovery", ["root_retry", "floor_fallback"])
def test_recovered_root_resumes_deep_search_with_remaining_budget(recovery):
    class Generator(_CornerFallbackGenerator):
        def begin_retry(self, excluded_actions):
            super().begin_retry(excluded_actions)
            if recovery == "root_retry":
                self.relaxed = True

    generator = Generator(reject_floor=recovery == "floor_fallback")
    evaluator = _CornerFallbackEvaluator()
    state = _state()
    state.object_ids = [10 * (i + 1) for i in range(8)]
    search = MonteCarloTreeSearch(
        _root_policy_config(max_depth=5, max_children=4, root_max_children=4,
            generation_batch_size=16, evaluation_batch_size=16,
            expansion_attempt_multiplier=3., rejected_root_retry_rounds=3,
            exploration_constant=1.,
            widening=ProgressiveWideningConfig(coefficient=2.)),
        generator, _ObjectPriorityScorer(), evaluator, _ZeroEstimator(),
        root_action_generator=generator,
    )
    result = search.search(MCTSNode(state=state, observation=()),
                           MCTSSearchRequest(iterations=128))
    from stacking_common.workflows.mcts_reporting import tree_statistics
    assert tree_statistics(result.root)["max_depth"] == 5
    assert any(len(child.children) > 2 for child in result.root.evaluated_children())
    assert result.completed_iterations == 128
    assert result.floor_fill_fallback.attempted is (recovery == "floor_fallback")
    assert not result.timed_out


def test_empty_search_does_not_spend_iterations_or_claim_a_timeout():
    result = _corner_fallback_search(
        _CornerFallbackGenerator(reject_floor=True),
        _CornerFallbackEvaluator(fail_relaxed=True), depth=3,
    ).search(MCTSNode(state=_state(), observation=()), MCTSSearchRequest(iterations=128))
    assert result.selected is None
    assert result.completed_iterations == 0
    assert not result.timed_out


def test_corner_fallback_does_not_run_when_strict_search_succeeds():
    generator = _CornerFallbackGenerator()
    result = _corner_fallback_search(generator, _CornerFallbackEvaluator(strict_succeeds=True)).search(
        MCTSNode(state=_state(), observation=()), MCTSSearchRequest(iterations=1,
            ))
    assert result.selected is not None
    assert not result.floor_fill_fallback.attempted
    assert not any(relaxed for relaxed, _ in generator.calls)


@pytest.mark.parametrize('fail_descendants', [False, True])
def test_root_survives_missing_continuations_without_sampling_relaxation(fail_descendants):
    class Generator(_CornerFallbackGenerator):
        roots = []

        def generate(self, state, observation, request):
            if not state.placed_object_indices:
                self.roots.append(self.relaxed)
            elif not self.relaxed:
                return ActionGenerationResult((), attempts=1, exhausted=True,
                    rejection_counts={'active_floor_above_active_layer': 1})
            return super().generate(state, observation, request)

    generator = Generator()
    evaluator = _CornerFallbackEvaluator(strict_succeeds=True,
                                       fail_descendants=fail_descendants)
    search = _corner_fallback_search(generator, evaluator, depth=3)
    root = MCTSNode(state=State([10, 20, 30, 40], [], {}, [], []), observation=())
    result = search.search(root, MCTSSearchRequest(iterations=1,
        root_sampling_priority='lower_floor_first', deeper_sampling_priority='lower_floor_first',
        ))
    assert not generator.relaxed
    assert result.selected is not None
    assert result.selected.info["descendant_depth_fallback"]
    assert not result.floor_fill_fallback.attempted
    assert not any(generator.roots)
    assert not result.selected.children
    assert result.selected.rejected_expansions["active_floor_above_active_layer"] > 0
    assert all(priority == 'lower_floor_first' for _, priority in generator.calls)


@pytest.mark.parametrize('fail_descendants', [False, True])
def test_corner_fallback_preserves_physics_and_descendant_depth_fallback(fail_descendants):
    generator = _CornerFallbackGenerator()
    evaluator = _CornerFallbackEvaluator(fail_relaxed=not fail_descendants,
                                       fail_descendants=fail_descendants)
    result = _corner_fallback_search(generator, evaluator, depth=3 if fail_descendants else 1).search(
        MCTSNode(state=_state(), observation=()), MCTSSearchRequest(iterations=1,
            ))
    assert result.floor_fill_fallback.attempted
    assert result.floor_fill_fallback.selected is fail_descendants
    assert result.floor_fill_fallback.pools == (1 if fail_descendants else 3)
    assert not generator.relaxed
    if fail_descendants:
        assert result.selected is not None
        assert result.selected.info["descendant_depth_fallback"]
        assert not result.selected.state.failed
        # Descendants use the same physics; no extra validation/repair follows.
        assert (True, EvaluationFidelity.LONG) in evaluator.calls
        assert result.descendant_validation_failures == 0
    else:
        assert result.selected is None
        assert result.floor_fill_fallback.accepted == 0


def test_corner_fallback_respects_search_deadline(monkeypatch):
    clock = [0.]
    monkeypatch.setattr('stacking_common.search.mcts.search.time.monotonic', lambda: clock[0])
    generator = _CornerFallbackGenerator()
    result = _corner_fallback_search(generator, _CornerFallbackEvaluator(clock=clock)).search(
        MCTSNode(state=_state(), observation=()), MCTSSearchRequest(iterations=1,
            max_time_seconds=10., ))
    assert result.selected is None and result.timed_out
    assert not result.floor_fill_fallback.attempted
    assert not any(relaxed for relaxed, _ in generator.calls)


def test_promoted_corner_fallback_keeps_evaluated_descendants():
    generator = _CornerFallbackGenerator()
    evaluator = _CornerFallbackEvaluator()
    search = _corner_fallback_search(generator, evaluator, depth=3)
    state = State([10, 20, 30, 40], [], {}, [], [])
    first = search.search(MCTSNode(state=state, observation=()), MCTSSearchRequest(iterations=1))
    assert first.selected is not None and first.selected.children
    retained = tuple(first.selected.children)
    second = search.search(first.selected, MCTSSearchRequest(iterations=1))
    assert all(child.parent is second.root for child in retained)
    assert not second.floor_fill_fallback.attempted


def test_nonempty_root_does_not_invoke_floor_or_corner_relaxation():
    class Generator(_RemainingObjectGenerator):
        def corner_seed_fallback(self):
            pytest.fail('corner fallback only applies to the empty scene')
        def floor_fill_fallback(self):
            pytest.fail('floor relaxation has been removed')
    state = _state()
    state.placed_object_indices = [0]
    state.object_poses[10] = np.array([0., 0., 0., 0., 0., 0., 1.])
    result = _corner_fallback_search(Generator(), _CornerFallbackEvaluator()).search(
        MCTSNode(state=state, observation=()), MCTSSearchRequest(iterations=1))
    assert result.selected is None
    assert not result.floor_fill_fallback.attempted



def test_corner_fallback_retries_fresh_pools_until_one_qualifies():
    class Generator(_CornerFallbackGenerator):
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

    class Evaluator(_CornerFallbackEvaluator):
        def evaluate(self, state, action, fidelity):
            result = super().evaluate(state, action, fidelity)
            if action.diagnostics.get('relaxed_pool') != 1:
                return result
            # Every candidate of the first relaxed pool fails physically.
            result.state.failed = True
            return TransitionEvaluation(result.state, (), -1., fidelity, failed=True,
                                        rejection_reason='target_motion_limit')

    generator = Generator()
    result = _corner_fallback_search(generator, Evaluator()).search(
        MCTSNode(state=_state(), observation=()), MCTSSearchRequest(iterations=1,
            ))
    assert result.selected is not None
    assert result.selected.action.diagnostics['relaxed_pool'] == 2
    assert result.floor_fill_fallback.pools == 2 and result.floor_fill_fallback.selected
    # The second pool excludes every candidate of the first.
    assert generator.excluded and all(
        action.diagnostics['relaxed_pool'] == 1 for action in generator.excluded)


@pytest.mark.parametrize('failure_depth', [0, 1])
def test_backend_capacity_failure_stops_search_without_retry_or_floor_fallback(failure_depth):
    from stacking_common.errors import BackendCapacityError
    from stacking_common.workflows.mcts_reporting import search_statistics

    class Generator(_RemainingObjectGenerator):
        failures = 0
        def generate(self, state, observation, request):
            if len(state.placed_object_indices) == failure_depth:
                self.failures += 1
                raise BackendCapacityError('posegen', 'bodies', 33, 32, request_id=4, stage=0)
            return super().generate(state, observation, request)
        def corner_seed_fallback(self):
            pytest.fail('capacity exhaustion must not retry with relaxed floor rules')

    generator = Generator()
    search = MonteCarloTreeSearch(_config(), generator, _ObjectPriorityScorer(),
                                 _DeterministicEvaluator(), _ZeroEstimator())
    root = MCTSNode(state=_state(), observation=())
    result = search.search(root, MCTSSearchRequest(iterations=100))
    assert result.root is root
    assert generator.failures == 1
    assert result.selected is None and result.ranked_candidates == ()
    assert not result.timed_out
    assert not result.floor_fill_fallback.attempted
    assert result.root_retry.rounds == 0
    assert result.failure_reason == 'posegen_capacity_overflow'
    assert result.backend_failure['required'] == 33
    assert result.backend_failure['capacity'] == 32
    if failure_depth:
        assert root.children  # Retain the partial tree for inspection.
    saved = search_statistics(result, 0., search, step=30)
    assert saved['backend_failure'] == result.backend_failure


@pytest.mark.parametrize('generation_failure', [False, True])
def test_empty_root_retries_ordinary_sampling_after_corner_validation_failure(generation_failure):
    generator = _CornerFallbackGenerator(reject_floor=generation_failure)
    result = _corner_fallback_search(generator, _CornerFallbackEvaluator()).search(
        MCTSNode(state=_state(), observation=()), MCTSSearchRequest(iterations=1,
            root_sampling_priority='lower_floor_first', deeper_sampling_priority='lower_floor_first',
            ))
    assert result.selected is not None
    assert result.floor_fill_fallback.policy == 'corner_seed'
    assert result.floor_fill_fallback.attempted and result.floor_fill_fallback.selected
    assert all(priority == 'lower_floor_first' for _, priority in generator.calls)
    assert not generator.relaxed


@pytest.mark.parametrize("lookahead,batch_size", [(True, 2), (False, 2), (False, 1)])
def test_generation_past_deadline_does_not_start_descendant_simulation(monkeypatch, lookahead, batch_size):
    from dataclasses import replace
    clock = [0.]
    monkeypatch.setattr('stacking_common.search.mcts.search.time.monotonic', lambda: clock[0])

    class SlowDescendants(_BatchRemainingObjectGenerator):
        def generate(self, state, observation, request):
            result = super().generate(state, observation, request)
            if state.placed_object_indices:
                clock[0] = 10.
            return result

    search = MonteCarloTreeSearch(
        replace(_config(), root_lookahead=lookahead, evaluation_batch_size=batch_size),
        SlowDescendants(), _ObjectPriorityScorer(), _DeterministicEvaluator(), _ZeroEstimator())
    result = search.search(MCTSNode(state=_state(), observation=()),
        MCTSSearchRequest(iterations=8, max_time_seconds=10.))
    assert result.timed_out
    assert result.completed_iterations == 0
    assert result.evaluated_candidates == 2  # Only the admitted root pool.
    assert result.selected is not None and result.selected.evaluated
    pending = [node for child in result.root.children for node in child.children]
    assert pending
    assert all(not node.evaluated and node.visits == 0 for node in pending)
