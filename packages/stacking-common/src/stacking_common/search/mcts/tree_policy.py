"""Pure expansion and selection policies for MCTS trees."""

import math
import random
from collections.abc import Sequence
from typing import TypeVar

from .config import MCTSConfig
from .node import MCTSNode


ObservationT = TypeVar("ObservationT")


def child_limit(node: MCTSNode[object], config: MCTSConfig) -> int:
    """Return the current selectable-child limit for a node."""

    hard_limit = config.hard_child_limit(node.depth)
    widening = config.widening
    if node.depth == 0 or not widening.enabled:
        return hard_limit
    visits = max(node.visits, 1)
    widened = math.ceil(widening.coefficient * visits**widening.exponent)
    return min(hard_limit, max(widened, 1))


def attempt_limit(node: MCTSNode[object], config: MCTSConfig) -> int:
    """Return the total accepted-plus-rejected generation-attempt limit."""

    hard_limit = config.hard_child_limit(node.depth)
    return max(
        hard_limit,
        math.ceil(hard_limit * config.expansion_attempt_multiplier),
    )


def expansion_room(node: MCTSNode[object], config: MCTSConfig) -> int:
    """Return the number of candidates that may be requested immediately."""

    selectable_room = child_limit(node, config) - len(node.selectable_children())
    attempt_room = attempt_limit(node, config) - node.expansion_attempts
    return max(min(selectable_room, attempt_room), 0)


def can_expand(node: MCTSNode[object], config: MCTSConfig) -> bool:
    """Whether action generation may add candidates below a node."""

    return (
        node.evaluated
        and node.selectable
        and not node.terminal
        and not node.expansion_exhausted
        and node.depth < config.max_depth
        and expansion_room(node, config) > 0
    )


def selection_score(
    parent: MCTSNode[object],
    child: MCTSNode[object],
    prior_probability: float,
    config: MCTSConfig,
) -> float:
    """Return the legacy-compatible PUCT score for an evaluated child."""

    if child.parent is not parent:
        raise ValueError("selection child does not belong to the parent")
    if not child.evaluated or not child.selectable:
        return -math.inf
    exploration = (
        config.exploration_constant
        * float(prior_probability)
        * math.sqrt(max(parent.visits, 1))
        / (child.visits + 1.0)
    )
    return child.mean_value + exploration


def select_child(
    node: MCTSNode[ObservationT],
    config: MCTSConfig,
    *,
    epsilon: float = 0.0,
    rng: random.Random | None = None,
) -> MCTSNode[ObservationT] | None:
    """Choose a pending candidate first, otherwise choose by PUCT."""

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("MCTS selection epsilon must be between zero and one")
    pending = node.pending_children()
    if pending:
        generator = rng if rng is not None else random
        if epsilon > 0.0 and generator.random() < epsilon:
            return generator.choice(pending)
        return max(pending, key=lambda child: child.prior)

    children = node.evaluated_children()
    if not children:
        return None
    probabilities = _softmax([child.prior for child in children])
    return max(
        zip(children, probabilities),
        key=lambda pair: selection_score(node, pair[0], pair[1], config),
    )[0]


def best_child(
    node: MCTSNode[ObservationT],
) -> MCTSNode[ObservationT] | None:
    """Return the evaluated child with the highest backed-up mean value."""

    children = node.evaluated_children()
    if not children:
        return None
    return max(children, key=lambda child: child.mean_value)


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]
