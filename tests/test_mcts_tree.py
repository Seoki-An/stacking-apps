"""Retained-tree projection, portable storage, and operations tree diagrams."""

from dataclasses import replace
import importlib
import json
import os

import pytest

from stacking_common.artifacts import StackRunStore, decode_mcts_debug, encode_mcts_debug
from stacking_common.search import MCTSNode
from stacking_common.visualization import MCTSDebugArtifact, project_mcts_step
from stacking_common.visualization.tree import displayed_tree, tree_positions
from test_mcts_visualization import _action, _root_state, _evaluated_state


def tree_step():
    root = MCTSNode(state=_root_state())
    action = _action(1, 22, .04)
    chosen = root.add_candidate(action, prior=.6)
    chosen.set_result(_evaluated_state(action), None, 1.5)
    chosen.record_visit(2.)
    root.add_candidate(_action(2, 33, .02))
    failed = chosen.add_candidate(_action(2, 33, .05))
    failed.failed = True
    failed.rejection_reason = "unstable"
    failed.add_candidate(_action(2, 33, .06))
    root.reject_expansion("posegen_gap", count=7)
    return root, project_mcts_step(root, chosen, step=1)


def test_snapshot_includes_descendants_without_mutating_tree(tmp_path):
    root, step = tree_step()
    nodes = step.tree_nodes
    assert len(nodes) == 5
    assert [n.parent_id for n in nodes] == [None, 0, 0, 1, 3]
    assert [n.depth for n in nodes] == [0, 1, 1, 2, 3]
    assert [n.outcome for n in nodes] == ["evaluated", "evaluated", "pending", "failed", "pending"]
    assert nodes[1].selected and nodes[1].candidate_id == step.selected_candidate_id
    assert nodes[1].visits == 1 and nodes[1].mean_value == 2.
    assert nodes[3].rejection_reason == "unstable"
    assert step.rejected_expansions == {"posegen_gap": 7}
    root.children[0].record_visit(99.)
    assert nodes[1].visits == 1  # Independent snapshot, not a live tree reference.
    artifact = MCTSDebugArtifact("world", (step,))
    decoded = decode_mcts_debug(json.loads(json.dumps(encode_mcts_debug(artifact))))
    assert decoded.steps[0].tree_nodes == nodes
    path = tmp_path / "tree.stack"
    with StackRunStore.create_mcts(path, replace(artifact, steps=())) as writer:
        writer.append_mcts_step(step)
    assert StackRunStore.load_mcts(path).steps[0].tree_nodes == nodes


def test_old_files_are_explicit_root_only_and_layout_preserves_branches():
    _, step = tree_step()
    payload = encode_mcts_debug(MCTSDebugArtifact("world", (step,)))
    del payload["steps"][0]["tree_nodes"]
    old_step = decode_mcts_debug(payload).steps[0]
    assert old_step.tree_nodes == ()
    nodes = displayed_tree(old_step)
    assert len(nodes) == 3 and nodes[0].outcome == "unknown"
    assert nodes[0].visits is None
    assert nodes[1].selected
    positions = tree_positions(step.tree_nodes)
    assert len(positions) == 5
    assert positions[4][0] == positions[3][0] == positions[1][0]
    assert positions[2][0] > positions[1][0]
    assert positions[0][0] == (positions[1][0] + positions[2][0]) / 2


@pytest.mark.parametrize("changes", [
    {"node_id": 8}, {"parent_id": 2}, {"parent_id": None}, {"depth": 4},
])
def test_invalid_tree_topology_is_rejected(changes):
    _, step = tree_step()
    nodes = list(step.tree_nodes)
    nodes[1] = replace(nodes[1], **changes)
    with pytest.raises(ValueError, match="tree"):
        replace(step, tree_nodes=nodes)


def test_layout_handles_deep_trees_without_recursive_traversal():
    from stacking_common.visualization.tree import MCTSTreeNode
    nodes = tuple(MCTSTreeNode(
        index, index - 1 if index else None, index, None,
        "pending", 0, 0., None, 0.,
    ) for index in range(1500))
    positions = tree_positions(nodes)
    assert positions[1499] == (0, 1499)
    assert positions[0] == (0, 0)


@pytest.mark.parametrize("platform", ["tabletop", "excavator"])
def test_tree_dialog_follows_steps_and_refresh(platform):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from stacking_common.visualization.operations import OperationsController
    module = importlib.import_module(f"stacking_{platform}.gui.operations.search_inspector")
    app = QApplication.instance() or QApplication([])
    _, step = tree_step()
    old_step = replace(step, step=2, tree_nodes=())
    controller = OperationsController()
    inspector = module.SearchInspector(controller)
    try:
        assert not inspector.tree_button.isEnabled()
        controller.set_artifact(MCTSDebugArtifact("world", (step, old_step)))
        inspector.refresh()
        inspector.tree_button.click()
        app.processEvents()
        view = inspector._tree_view
        assert len(view.nodes) == 5
        assert "Retained search tree" in view.summary.text()
        node_item = next(item for item in view.scene.items() if item.data(0) == 3)
        node_item.setSelected(True)
        assert "unstable" in view.details.text()
        before = view.canvas.transform().m11()
        view.canvas.zoom(1.2)
        assert view.canvas.transform().m11() > before
        inspector.step_selector.setCurrentIndex(1)
        assert len(view.nodes) == 3
        assert "Partial tree" in view.summary.text()
        assert "Step 2" in view.summary.text()
        # A refreshed snapshot of the same step replaces the old diagram.
        controller.set_artifact(MCTSDebugArtifact("world", (replace(step, step=2),)))
        inspector.refresh()
        assert len(view.nodes) == 5
        inspector._tree_dialog.close()
    finally:
        inspector.close()
        app.processEvents()
