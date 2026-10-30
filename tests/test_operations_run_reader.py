"""Incremental viewer reads retain verified records without holding file locks."""

from dataclasses import replace
import json
import sqlite3
from unittest.mock import patch

import pytest

from stacking_common.artifacts import StackRunStore, PlacementPlanRecord
from stacking_common.artifacts import run_document
from stacking_common.visualization.operations import OperationsController
from test_tabletop_operations_console import _artifact
from test_tabletop_planning import _problem, _success_result


def test_refresh_decodes_only_new_steps_and_plans(tmp_path):
    artifact = _artifact()
    path = tmp_path / "growing.stack"
    with StackRunStore.create_mcts(path, replace(artifact, steps=())) as writer:
        writer.append_mcts_step(artifact.steps[0], statistics={"step": 1, "iterations": 20})
        record = PlacementPlanRecord(1, 1, _problem(), _success_result())
        writer.append_placement_plan(record)
        controller = OperationsController()
        controller.load(path)
        old_step, old_plan = controller.artifact.steps[0], controller.planning_records[0]
        with patch.object(run_document, "_arrays", wraps=run_document._arrays) as arrays:
            controller.reload(path)
            arrays.assert_not_called()
            writer.append_mcts_step(artifact.steps[1], statistics={"step": 2, "iterations": 30})
            writer.append_placement_plan(replace(record, attempt=2))
            controller.reload(path)
            assert {(call.args[1], call.args[2]) for call in arrays.call_args_list} == {
                ("mcts_step", "2"), ("mcts_search", "2"), ("placement_plan", "1:2"),
            }
        assert controller.artifact.steps[0] is old_step
        assert controller.planning_records[0] is old_plan
        assert controller.search_statistics[2]["iterations"] == 30
        assert len(controller.planning_records) == 2


def test_refresh_reloads_replaced_file_and_changed_record(tmp_path):
    path = StackRunStore.save_mcts(_artifact(), tmp_path / "run.stack")
    controller = OperationsController()
    controller.load(path)
    old_step = controller.artifact.steps[0]
    with sqlite3.connect(path) as connection:
        payload = json.loads(connection.execute(
            "SELECT payload_json FROM records WHERE kind='mcts_step' AND record_id='1'"
        ).fetchone()[0])
        payload["candidates"][0]["reward"] = 123.0
        connection.execute("UPDATE records SET payload_json=? WHERE kind='mcts_step' AND record_id='1'",
                           (json.dumps(payload),))
    controller.reload(path)
    assert controller.artifact.steps[0] is not old_step
    assert controller.artifact.steps[0].candidates[0].reward == 123.0
    old_step = controller.artifact.steps[0]
    StackRunStore.save_mcts(replace(_artifact(), steps=(_artifact().steps[0],)), path)
    controller.reload(path)
    assert len(controller.artifact.steps) == 1
    assert controller.artifact.steps[0] is not old_step
    assert controller.artifact.steps[0].candidates[0].reward != 123.0


def test_failed_refresh_preserves_previous_snapshot_and_retries_new_arrays(tmp_path):
    artifact = _artifact()
    path = tmp_path / "run.stack"
    with StackRunStore.create_mcts(path, replace(artifact, steps=())) as writer:
        writer.append_mcts_step(artifact.steps[0])
        controller = OperationsController()
        controller.load(path)
        old_artifact = controller.artifact
        writer.append_mcts_step(replace(artifact.steps[0], step=2))
        original = run_document._arrays

        def fail(connection, kind, record_id):
            if (kind, record_id) == ("mcts_step", "2"):
                raise ValueError("corrupt array")
            return original(connection, kind, record_id)

        with patch.object(run_document, "_arrays", side_effect=fail):
            with pytest.raises(ValueError, match="corrupt array"):
                controller.reload(path)
        assert controller.artifact is old_artifact
        controller.reload(path)
        assert len(controller.artifact.steps) == 2
        assert controller.artifact.steps[0] is old_artifact.steps[0]
