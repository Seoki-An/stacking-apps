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

    Failure returns the configured fixed penalty, with no bootstrap or future
    tail. Ancestors discount it once per edge. A failed node still receives a
    zero visit sample itself. Successful leaves retain the supplied bootstrap.
    """

    value = float(bootstrap_value)
    if not math.isfinite(value):
        raise ValueError("MCTS bootstrap value must be finite")

    node: MCTSNode[object] | None = leaf
    while node is not None:
        if node.failed:
            # Terminate this path with one failure cost, independent of the
            # computational horizon and the remaining inventory.
            value = config.failure_reward
        else:
            value = node.reward + config.discount * value
        node.record_visit(0.0 if node.failed else value)
        node = node.parent
    return value
