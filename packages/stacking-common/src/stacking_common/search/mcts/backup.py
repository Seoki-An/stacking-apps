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

    Failure incurs the configured per-step reward through the remaining search
    horizon, capped by available stones. A failed node still receives a zero
    visit sample itself. Successful leaves retain the supplied bootstrap.
    """

    value = float(bootstrap_value)
    if not math.isfinite(value):
        raise ValueError("MCTS bootstrap value must be finite")

    node: MCTSNode[object] | None = leaf
    while node is not None:
        if node.failed:
            # An absorbing failure replaces any bootstrap or descendant value.
            # This return already includes the failed placement; ancestors
            # apply their ordinary discount exactly once per edge.
            value = _failure_return(node, config)
        else:
            value = node.reward + config.discount * value
        node.record_visit(0.0 if node.failed else value)
        node = node.parent
    return value


def _failure_return(node: MCTSNode[object], config: MCTSConfig) -> float:
    future_steps = max(config.max_depth - node.depth, 0)
    # Prefer the pre-action inventory: evaluators may either append the failed
    # placement or return the unchanged parent state. Consume that stone once
    # in either representation, without trusting failure's terminal flag.
    state = node.parent.state if node.parent is not None else None
    if state is None:
        state = node.state
    if state is not None:
        remaining = set(range(len(state.object_ids))) - set(state.placed_object_indices)
        if node.action is not None:
            remaining.discard(node.action.object_index)
        future_steps = min(future_steps, len(remaining))
    return config.failure_reward * math.fsum(
        config.discount**step for step in range(future_steps + 1))
