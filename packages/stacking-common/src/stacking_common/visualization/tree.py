"""Lightweight retained-tree snapshots and backend-neutral tree layout."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MCTSTreeNode:
    node_id: int
    parent_id: int | None
    depth: int
    object_id: int | None
    outcome: str
    visits: int | None
    mean_value: float | None
    reward: float | None
    prior: float | None
    selected: bool = False
    candidate_id: str | None = None
    rejection_reason: str | None = None

    def __post_init__(self):
        for name in ("node_id", "depth", "object_id", "visits", "parent_id"):
            value = getattr(self, name)
            if value is None and name not in {"node_id", "depth"}:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"tree {name} must be a non-negative integer")
        if self.outcome not in {"evaluated", "failed", "pending", "unknown"}:
            raise ValueError("unknown tree node outcome")
        for name in ("mean_value", "reward", "prior"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"tree {name} must be finite")


def validate_tree(nodes):
    """Require contiguous IDs with parents before children, preventing cycles."""
    nodes = tuple(nodes)
    for index, node in enumerate(nodes):
        if not isinstance(node, MCTSTreeNode) or node.node_id != index:
            raise ValueError("tree IDs must be contiguous and ordered")
        if index == 0:
            if node.parent_id is not None or node.depth != 0:
                raise ValueError("tree must start with a depth-zero root")
        elif (node.parent_id is None or node.parent_id >= index
              or node.depth != nodes[node.parent_id].depth + 1):
            raise ValueError("tree node must follow its parent at the next depth")
    return nodes


def project_tree(root, selected, candidates):
    """Copy scalar diagnostics only, never states, meshes, or trajectories."""
    candidate_ids = {id(node): value.candidate_id
                     for node, value in zip(root.children, candidates)}
    pending = [(root, None, 0)]
    result = []
    for node, parent_id, depth in pending:
        node_id = len(result)
        result.append(MCTSTreeNode(
            node_id, parent_id, depth,
            None if parent_id is None else int(node.action.object_id),
            "failed" if node.failed else "evaluated" if node.evaluated else "pending",
            int(node.visits), float(node.mean_value),
            float(node.reward) if node.evaluated else None, float(node.prior),
            node is selected, candidate_ids.get(id(node)), node.rejection_reason,
        ))
        pending.extend((child, node_id, depth + 1) for child in node.children)
    return tuple(result)


def displayed_tree(step):
    """Older files expose only root branches; do not invent deeper structure."""
    if step.tree_nodes:
        return step.tree_nodes
    root = MCTSTreeNode(0, None, 0, None, "unknown", None, None, None, None)
    return (root,) + tuple(MCTSTreeNode(
        index, 0, 1, value.object_id,
        "evaluated" if value.outcome.value == "succeeded" else value.outcome.value,
        value.visits, value.mean_value, value.reward, value.prior,
        value.selected, value.candidate_id, value.rejection_reason,
    ) for index, value in enumerate(step.candidates, 1))


def tree_positions(nodes):
    """Place leaves left-to-right and center parents above their children."""
    nodes = validate_tree(nodes)
    children = [[] for _ in nodes]
    for node in nodes[1:]:
        children[node.parent_id].append(node.node_id)
    positions = {}
    pending = [(0, False)] if nodes else []
    leaf = 0
    while pending:
        index, visited = pending.pop()
        if not children[index]:
            positions[index] = (leaf, nodes[index].depth)
            leaf += 1
        elif visited:
            positions[index] = (
                (positions[children[index][0]][0] + positions[children[index][-1]][0]) / 2,
                nodes[index].depth,
            )
        else:
            pending.append((index, True))
            pending.extend((child, False) for child in reversed(children[index]))
    return positions
