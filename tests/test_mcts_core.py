import random

import numpy as np
import pytest

from stacking_common.domain import Action, State
from stacking_common.search.mcts import (
    FinalSelectionConfig,
    MCTSConfig,
    MCTSNode,
    ProgressiveWideningConfig,
    attempt_limit,
    backpropagate,
    best_child,
    can_expand,
    child_limit,
    expansion_room,
    select_child,
    selection_score,
)


def _state(*, terminated: bool = False, failed: bool = False) -> State:
    return State(
        object_ids=[10, 20, 30],
        placed_object_indices=[],
        object_poses={},
        trajectories=[],
        action_history=[],
        terminated=terminated,
        failed=failed,
    )


def _action(index: int) -> Action:
    pose = np.array([float(index), 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])
    return Action(
        object_index=index,
        object_id=(index + 1) * 10,
        pose=pose,
        init_pose=pose,
    )


def _config(**overrides) -> MCTSConfig:
    values = {
        "max_depth": 3,
        "max_children": 8,
        "root_max_children": 4,
        "exploration_constant": 2.0,
        "discount": 0.9,
        "expansion_attempt_multiplier": 2.0,
        "widening": ProgressiveWideningConfig(
            enabled=True,
            coefficient=2.0,
            exponent=0.5,
        ),
    }
    values.update(overrides)
    return MCTSConfig(**values)


def _evaluated_root() -> MCTSNode[str]:
    return MCTSNode(state=_state(), observation="root")


def test_config_rejects_invalid_search_values() -> None:
    with pytest.raises(ValueError, match="maximum depth"):
        _config(max_depth=-1)
    with pytest.raises(ValueError, match="discount"):
        _config(discount=1.1)
    with pytest.raises(ValueError, match="attempt multiplier"):
        _config(expansion_attempt_multiplier=0.5)
    with pytest.raises(ValueError, match="exponent"):
        ProgressiveWideningConfig(exponent=1.1)
    with pytest.raises(ValueError, match="final-selection mode"):
        FinalSelectionConfig(mode="unknown")
    with pytest.raises(ValueError, match="maximum weight"):
        FinalSelectionConfig(max_weight=1.1)
    with pytest.raises(ValueError, match="descendant-validation depth"):
        FinalSelectionConfig(validate_descendant_depth=-1)
    with pytest.raises(ValueError, match="retry rounds"):
        _config(rejected_root_retry_rounds=-1)
    with pytest.raises(ValueError, match="depth-1 step count"):
        _config(initial_depth1_steps=-1)


def test_initial_depth1_schedule_matches_legacy_placement_boundary() -> None:
    config = _config(max_depth=3, initial_depth1_steps=3)

    assert config.effective_max_depth(0) == 1
    assert config.effective_max_depth(2) == 1
    assert config.effective_max_depth(3) == 3
    assert _config(max_depth=0, initial_depth1_steps=3).effective_max_depth(0) == 0


def test_root_limit_is_fixed_and_descendants_progressively_widen() -> None:
    config = _config()
    root = _evaluated_root()
    child = root.add_candidate(_action(0))
    child.set_result(_state(), "child", 1.0)

    assert child_limit(root, config) == 4
    assert child_limit(child, config) == 2

    child.visits = 9
    assert child_limit(child, config) == 6

    child.visits = 100
    assert child_limit(child, config) == 8


def test_expansion_room_counts_rejections_without_using_child_slots() -> None:
    root = _evaluated_root()
    config = _config(root_max_children=2, expansion_attempt_multiplier=2.0)

    root.reject_expansion("duplicate")
    root.reject_expansion("pose_solver_failed")
    assert attempt_limit(root, config) == 4
    assert expansion_room(root, config) == 2

    root.add_candidate(_action(0))
    assert expansion_room(root, config) == 1

    root.reject_expansion("outside_target")
    assert expansion_room(root, config) == 0
    assert not can_expand(root, config)
    assert root.rejected_expansions == {
        "duplicate": 1,
        "pose_solver_failed": 1,
        "outside_target": 1,
    }


def test_terminal_failed_exhausted_and_max_depth_nodes_cannot_expand() -> None:
    config = _config(max_depth=1)
    root = _evaluated_root()
    child = root.add_candidate(_action(0))
    child.set_result(_state(), "child", 1.0)
    assert child.depth == 1
    assert not can_expand(child, config)

    root.terminal = True
    assert not can_expand(root, config)
    root.terminal = False
    root.failed = True
    assert not can_expand(root, config)
    root.failed = False
    root.expansion_exhausted = True
    assert not can_expand(root, config)


def test_selection_evaluates_pending_children_in_prior_order() -> None:
    root = _evaluated_root()
    low = root.add_candidate(_action(0), prior=-1.0)
    high = root.add_candidate(_action(1), prior=2.0)

    assert select_child(root, _config()) is high
    high.set_result(_state(), "high", 0.0)
    assert select_child(root, _config()) is low


def test_pending_selection_uses_an_injected_rng_for_epsilon() -> None:
    root = _evaluated_root()
    first = root.add_candidate(_action(0), prior=10.0)
    second = root.add_candidate(_action(1), prior=-10.0)

    selected = select_child(
        root,
        _config(),
        epsilon=1.0,
        rng=random.Random(0),
    )

    assert selected is second
    assert selected is not first


def test_puct_balances_backed_up_value_and_softmax_prior() -> None:
    root = _evaluated_root()
    root.visits = 16
    high_value = root.add_candidate(_action(0), prior=-2.0)
    high_prior = root.add_candidate(_action(1), prior=2.0)
    high_value.set_result(_state(), "value", 0.0)
    high_prior.set_result(_state(), "prior", 0.0)
    high_value.record_visit(3.0)
    high_prior.record_visit(0.0)

    assert select_child(root, _config(exploration_constant=0.0)) is high_value
    assert select_child(root, _config(exploration_constant=4.0)) is high_prior

    high_prior.reject("validation_failed")
    assert select_child(root, _config(exploration_constant=4.0)) is high_value


def test_puct_requires_positive_real_or_virtual_parent_visits() -> None:
    root = _evaluated_root()
    child = root.add_candidate(_action(0))
    child.set_result(_state(), "child", 0.0)

    with pytest.raises(ValueError, match="positive parent visits"):
        selection_score(root, child, 1.0, _config())

    assert selection_score(
        root, child, 1.0, _config(), virtual_visits={root: 1, child: 1}
    ) == pytest.approx(1.0)
    assert root.visits == child.visits == 0


def test_backpropagation_matches_legacy_discounted_reward_order() -> None:
    root = _evaluated_root()
    child = root.add_candidate(_action(0))
    child.set_result(_state(), "child", 2.0)
    leaf = child.add_candidate(_action(1))
    leaf.set_result(_state(terminated=True), "leaf", 3.0)

    value = backpropagate(leaf, _config(discount=0.5), bootstrap_value=4.0)

    # Legacy order includes each node's own reward, including the root's zero.
    assert leaf.mean_value == pytest.approx(5.0)
    assert child.mean_value == pytest.approx(4.5)
    assert root.mean_value == pytest.approx(2.25)
    assert value == pytest.approx(2.25)
    assert (root.visits, child.visits, leaf.visits) == (1, 1, 1)


def test_failed_node_uses_failure_reward_but_records_zero_for_itself() -> None:
    root = _evaluated_root()
    child = root.add_candidate(_action(0))
    child.set_result(
        _state(failed=True),
        "failed",
        100.0,
        rejection_reason="simulation_failed",
    )

    value = backpropagate(child, _config(discount=0.5), bootstrap_value=0.0)

    assert child.mean_value == 0.0
    assert root.mean_value == pytest.approx(-0.5)
    assert value == pytest.approx(-0.5)


def test_preserved_subtree_detaches_and_resets_search_statistics() -> None:
    root = _evaluated_root()
    child = root.add_candidate(_action(0))
    child.set_result(_state(), "child", 1.0)
    leaf = child.add_candidate(_action(1))
    leaf.set_result(_state(), "leaf", 2.0)
    pending = leaf.add_candidate(_action(2))
    backpropagate(leaf, _config(discount=1.0))

    child.detach().reset_statistics()

    assert child.parent is None
    assert child.depth == 0
    assert leaf.depth == 1
    assert root.children == []
    assert child.visits == leaf.visits == 1
    assert child.value_sum == leaf.value_sum == 0.0
    assert pending.visits == 0
    assert select_child(child, _config()) is leaf

    backpropagate(leaf, _config(discount=1.0))
    assert leaf.visits == child.visits == 2
    assert leaf.mean_value == pytest.approx(1.0)
    assert child.mean_value == pytest.approx(1.5)


def test_best_child_ignores_pending_and_failed_candidates() -> None:
    root = _evaluated_root()
    root.add_candidate(_action(0), prior=10.0)
    first = root.add_candidate(_action(1))
    second = root.add_candidate(_action(2))
    first.set_result(_state(), "first", 0.0)
    second.set_result(_state(), "second", 0.0)
    first.record_visit(1.0)
    second.record_visit(4.0)
    second.reject("final_validation_failed")

    assert best_child(root) is first


def test_node_owns_action_and_state_snapshots() -> None:
    root_state = _state()
    root = MCTSNode(state=root_state, observation="root")
    action = _action(0)
    child = root.add_candidate(action)

    action.pose[0] = 99.0
    root_state.object_ids[0] = 999

    assert child.action is not None
    assert child.action.pose[0] == 0.0
    assert root.state is not None
    assert root.state.object_ids == [10, 20, 30]
