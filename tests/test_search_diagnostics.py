"""Recorded search failures remain inspectable even with no root candidates."""

from dataclasses import replace
import importlib
import json
import os
from pathlib import Path

import pytest

from stacking_common.artifacts import StackRunStore, decode_legacy_mcts_debug
from stacking_common.visualization.operations import OperationsController


ROOT = Path(__file__).resolve().parents[1]


def artifact():
    source = decode_legacy_mcts_debug(json.loads((ROOT /
        "tests/fixtures/legacy_visualization/mcts_candidate_viewer.json").read_text()))
    first = source.steps[0]
    candidates = tuple(replace(candidate, selected=False, outcome="failed",
        rejection_reason="missing_validated_descendant" if index == 0 else "target_motion_limit")
        for index, candidate in enumerate(first.candidates))
    return replace(source, steps=(
        replace(first, candidates=candidates, selected_candidate_id=None,
                succeeded=False, failure_reason="search_exhausted",
                rejected_expansions={"posegen_gap": 4}),
        replace(source.steps[1], failure_reason="search_exhausted"),
    ))


def test_recorded_search_statistics_survive_load_reload_and_step_switch(tmp_path):
    data = artifact()
    path = tmp_path / "failure.stack"
    controller = OperationsController()
    with StackRunStore.create_mcts(path, replace(data, steps=())) as writer:
        writer.append_mcts_step(data.steps[0], statistics={"step": 1,
            "timed_out": False, "generated_candidates": 12, "failed_evaluations": 9,
            "tree": {"rejection_reasons": {"target_motion_limit": 7}}})
        controller.load(path)
        controller.select_candidate(data.steps[0].candidates[0].candidate_id)
        text = controller.search_diagnostics
        assert "Step 1: search_exhausted" in text
        assert "Search time limit reached: no" in text
        assert "target_motion_limit: 7" in text
        assert "missing_validated_descendant: 1" in text
        assert "posegen_gap: 4" in text
        assert text.splitlines()[-1].endswith("missing_validated_descendant")

        writer.append_mcts_step(data.steps[1], statistics={"step": 2, "search_attempts": [
            {"attempt": 1, "timed_out": False, "generated_candidates": 3},
            {"attempt": 2, "timed_out": True, "generated_candidates": 0,
             "cem_statistics": [{"rejection_counts": {"posegen_gap": 96}}]},
        ]})
        controller.reload(path)
        assert controller.current_step.step == 1
        controller.select_step(1)
        text = controller.search_diagnostics
        assert "Step 2: search_exhausted" in text
        assert "Search attempt 2:" in text
        assert "Search time limit reached: yes" in text
        assert "Generated candidates: 0" in text
        assert "posegen_gap: 96" in text
        assert "Active candidate" not in text
        assert "missing_validated_descendant" not in text

    # JSON/legacy loads must not inherit the previous run's search statistics.
    controller.set_artifact(data)
    controller.select_step(1)
    assert "No detailed rejection reasons were recorded" in controller.search_diagnostics
    assert "Search time limit reached" not in controller.search_diagnostics


@pytest.mark.parametrize("platform", ["excavator", "tabletop"])
def test_search_inspector_displays_failure_details(platform):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    module = importlib.import_module(f"stacking_{platform}.gui.operations.search_inspector")
    app = QApplication.instance() or QApplication([])
    controller = OperationsController(artifact())
    widget = module.SearchInspector(controller)
    try:
        widget.candidate_selector.setCurrentIndex(0)
        assert "missing_validated_descendant" in widget.failure_details.toPlainText()
        assert widget.candidate_failure_reason.text() == "missing_validated_descendant"
        assert widget.candidate_selector.itemData(0, Qt.ItemDataRole.ToolTipRole) == "missing_validated_descendant"
        widget.candidate_selector.setCurrentIndex(1)
        assert widget.failure_details.toPlainText().splitlines()[-1].endswith("target_motion_limit")
        assert widget.candidate_failure_reason.text() == "target_motion_limit"
        assert widget.candidate_selector.itemData(1, Qt.ItemDataRole.ToolTipRole) == "target_motion_limit"
        widget.step_selector.setCurrentIndex(1)
        assert "Step 2: search_exhausted" in widget.failure_details.toPlainText()
        assert "No detailed rejection reasons were recorded" in widget.failure_details.toPlainText()
        assert widget.failure_details.isReadOnly()
        assert widget.candidate_failure_reason.text() == "No candidate selected."
    finally:
        widget.close()
        app.processEvents()


@pytest.mark.parametrize('nested', [False, True])
def test_backend_capacity_failure_is_visible_in_operations(tmp_path, nested):
    data = artifact()
    path = tmp_path / 'capacity.stack'
    failure = {'backend': 'posegen', 'reason': 'posegen_capacity_overflow',
               'resource': 'bodies', 'required': 33, 'capacity': 32,
               'message': 'posegen capacity exceeded for bodies: required 33, configured capacity 32'}
    attempt = {'backend_failure': failure, 'timed_out': False}
    statistics = {'search_attempts': [attempt]} if nested else attempt
    with StackRunStore.create_mcts(path, replace(data, steps=())) as writer:
        writer.append_mcts_step(replace(data.steps[0], failure_reason='posegen_capacity_overflow'),
                                statistics={'step': 1, **statistics})
    controller = OperationsController()
    controller.load(path)
    text = controller.search_diagnostics
    assert 'posegen_capacity_overflow' in text
    assert 'required: 33; capacity: 32' in text
    assert 'Search time limit reached: no' in text
