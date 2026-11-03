"""Failure has a fixed terminal cost, independent of horizon and inventory."""

from types import SimpleNamespace

import numpy as np
import pytest

from stacking_common.domain import Action, State
from stacking_common.search.mcts import MCTSConfig, MCTSNode, MonteCarloTreeSearch, backpropagate


def config(depth=4, discount=.9, penalty=-1.):
    return MCTSConfig(max_depth=depth, max_children=2, exploration_constant=1.,
                      discount=discount, failure_reward=penalty)


def chain(depth, *, stones=8, already_placed=0, append_failed=True):
    ids = list(range(10, 10 + stones))

    def state(count, failed=False):
        return State(ids, list(range(count)), {}, [], [], failed=failed, terminated=failed)

    root = MCTSNode(state=state(already_placed))
    node = root
    for level in range(1, depth + 1):
        index = already_placed + level - 1
        pose = np.array([0., 0., .1, 0., 0., 0., 1.])
        node = node.add_candidate(Action(index, pose, pose, object_id=ids[index]))
        failed = level == depth
        count = index if failed and not append_failed else index + 1
        node.set_result(state(count, failed), None, 100. if failed else 0.)
    return root, node


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
@pytest.mark.parametrize("search_depth", [4, 6, 8])
def test_failed_return_is_discounted_once_per_ancestor_edge(depth, search_depth):
    failed_return = -1.
    root, leaf = chain(depth)
    result = backpropagate(leaf, config(depth=search_depth), bootstrap_value=100.)
    assert result == pytest.approx(.9**depth * failed_return)
    assert root.mean_value == pytest.approx(result)
    assert leaf.parent.mean_value == pytest.approx(.9 * failed_return)
    assert leaf.mean_value == 0.  # Preserve failed-node visit accounting.
    assert leaf.reward == 100.  # Preserve the recorded transition reward.
    assert not leaf.selectable


@pytest.mark.parametrize("discount", [0., .5, 1.])
@pytest.mark.parametrize("penalty", [-1., -2.])
def test_failure_cost_is_applied_once(discount, penalty):
    _, leaf = chain(1)
    expected = penalty
    assert backpropagate(leaf, config(discount=discount, penalty=penalty)) == pytest.approx(discount * expected)


@pytest.mark.parametrize("append_failed", [False, True])
@pytest.mark.parametrize("stones,already_placed,depth,future", [
    (5, 4, 1, 0),  # Last remaining stone.
    (5, 3, 1, 1),
    (5, 2, 2, 1),
])
def test_inventory_and_failed_state_format_do_not_change_failure_cost(
        append_failed, stones, already_placed, depth, future):
    _, leaf = chain(depth, stones=stones, already_placed=already_placed, append_failed=append_failed)
    expected = -1.
    assert backpropagate(leaf, config()) == pytest.approx(.9**depth * expected)


def test_depth_one_and_unevaluated_rejection_still_receive_one_penalty():
    root, leaf = chain(1)
    leaf.state = None  # e.g. an explicit rejection before transition evaluation.
    assert backpropagate(leaf, config(depth=1)) == pytest.approx(-.9)
    assert root.mean_value == pytest.approx(-.9)


def test_fixed_failure_is_discounted_relative_to_promoted_root():
    _, leaf = chain(2)
    new_root = leaf.parent.detach()
    assert leaf.depth == 1
    assert backpropagate(leaf, config()) == pytest.approx(-.9)
    assert new_root.mean_value == pytest.approx(-.9)


def test_coordinator_does_not_apply_success_bootstrap_to_failure():
    def unexpected(*args):
        pytest.fail("failed nodes must not request a success heuristic")

    root, leaf = chain(1)
    engine = MonteCarloTreeSearch(config(), None, None, None,
        SimpleNamespace(estimate=unexpected, estimate_at_depth=unexpected))
    engine._back_up(leaf)
    assert root.mean_value == pytest.approx(-.9)


def test_successful_prefix_rewards_are_preserved_before_fixed_failure():
    root, leaf = chain(3)
    first = root.children[0]
    second = first.children[0]
    first.reward = 2.
    second.reward = 3.
    result = backpropagate(leaf, config(), bootstrap_value=100.)
    assert second.mean_value == pytest.approx(3. - .9)
    assert first.mean_value == pytest.approx(2. + .9 * (3. - .9))
    assert result == pytest.approx(.9 * (2. + .9 * (3. - .9)))


def test_failure_semantics_are_recorded_but_not_configurable():
    from stacking_common.artifacts.planning_context import _values
    values = _values(config())
    assert values['failure_return_mode'] == 'fixed'
    with pytest.raises(TypeError, match='failure_return_mode'):
        MCTSConfig(max_depth=4, max_children=2, exploration_constant=1.,
                   discount=.9, failure_return_mode='horizon')
