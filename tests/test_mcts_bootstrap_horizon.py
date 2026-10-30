"""Finite-horizon leaf estimates include each future reward exactly once."""

from types import SimpleNamespace

import numpy as np
import pytest

from stacking_common.domain import Action, State
from stacking_common.search.mcts import MCTSConfig, MCTSNode, MonteCarloTreeSearch
from stacking_common.search.policies import HorizonValueEstimator


def chain(depth, stones=8):
    ids = list(range(stones))
    root = MCTSNode(state=State(ids, [], {}, [], []))
    leaf = root
    pose = np.array([0., 0., .1, 0., 0., 0., 1.])
    for index in range(depth):
        leaf = leaf.add_candidate(Action(index, pose, pose, object_id=index))
        leaf.set_result(State(ids, list(range(index + 1)), {}, [], []), None,
                        .8506 if index == depth - 1 else .2)
    return root, leaf


@pytest.mark.parametrize('depth', [1, 2, 3, 4])
@pytest.mark.parametrize('discount', [0., .5, .9, 1.])
@pytest.mark.parametrize('horizon,stones', [(4, 8), (1, 8), (4, 4)])
def test_leaf_and_ancestor_returns_follow_remaining_horizon(depth, discount, horizon, stones):
    root, leaf = chain(depth, stones)
    engine = MonteCarloTreeSearch(
        MCTSConfig(max_depth=4, max_children=2, exploration_constant=1., discount=discount),
        None, None, None, HorizonValueEstimator(1., horizon, discount))
    engine._back_up(leaf)
    remaining = min(4 - depth, horizon, stones - depth)
    expected = .8506 + sum(discount**i for i in range(1, remaining + 1))
    assert leaf.mean_value == pytest.approx(expected)
    node = leaf.parent
    while node is not None:
        expected = node.reward + discount * expected
        assert node.mean_value == pytest.approx(expected)
        node = node.parent
    assert root.visits == 1


@pytest.mark.parametrize('terminal,depth', [(True, 1), (False, 4)])
def test_terminal_or_depth_limit_does_not_call_custom_estimator(terminal, depth):
    _, leaf = chain(depth)
    leaf.terminal = terminal

    def unexpected(*args):
        pytest.fail('No future estimate is allowed at the horizon or terminal state')

    engine = MonteCarloTreeSearch(
        MCTSConfig(max_depth=4, max_children=2, exploration_constant=1., discount=.9),
        None, None, None, SimpleNamespace(estimate=unexpected))
    engine._back_up(leaf)
    assert leaf.mean_value == pytest.approx(.8506)


def test_promoted_root_uses_new_relative_depth():
    _, leaf = chain(4)
    leaf.parent.detach()
    assert leaf.depth == 1
    engine = MonteCarloTreeSearch(
        MCTSConfig(max_depth=4, max_children=2, exploration_constant=1., discount=.9),
        None, None, None, HorizonValueEstimator(1., 4, .9))
    engine._back_up(leaf)
    assert leaf.mean_value == pytest.approx(.8506 + .9 + .9**2 + .9**3)
