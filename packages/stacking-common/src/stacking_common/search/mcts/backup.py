"""Discounted value backup for MCTS trees."""

import math

from .config import MCTSConfig
from .node import MCTSNode


def backpropagate(
    leaf: MCTSNode[object],
    config: MCTSConfig,
    *,
    bootstrap_value: float = 0.0,
) -> float:
    """Back up rewards through the current root and return the root value.

    This retains the legacy convention that a failed node contributes the
    configured failure reward but receives a zero visit sample itself. The
    caller supplies any learned or horizon-based bootstrap estimate.
    """

    value = float(bootstrap_value)
    if not math.isfinite(value):
        raise ValueError("MCTS bootstrap value must be finite")

    node: MCTSNode[object] | None = leaf
    while node is not None:
        reward = config.failure_reward if node.failed else node.reward
        value = reward + config.discount * value
        node.record_visit(0.0 if node.failed else value)
        node = node.parent
    return value
