import csv
import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from stacking_common.artifacts import (
    ExecutionReconciliation,
    FixedCollisionGeometry,
    PlacementPlanRecord,
    PlanningRunContext,
    PlanningCheckpoint,
    StackRunStore,
    build_execution_command,
    decode_legacy_mcts_debug,
    decode_placement_plan,
    encode_placement_plan,
    export_run_summary,
    summarize_run,
    verify_context_assets,
)
from stacking_common.domain import Action, ContactPoint, StackObjectTrajectory, State
from stacking_common.execution import ExecutionEvent, ExecutionFailure, ExecutionResult, ExecutionStatus
from stacking_common.planning import (
    FrameTransform, MotionPlanSegment, PickSceneSnapshot,
    PlacementPlanningProblem, PlacementPlanningResult,
)


FIXTURE = Path(__file__).parent / "fixtures/legacy_visualization/mcts_candidate_viewer.json"
POSE = np.array([0, 0, 0.1, 0, 0, 0, 1.0])


def _context():
    return PlanningRunContext(
        "world", ("joint_a", "joint_b"),
        {"motion": {"speed": 0.1}, "robot": {"base_pose": POSE}},
        (FixedCollisionGeometry("table", "box", POSE, 0.5, (1.0, 0.5, 0.1)),),
    )


def _run(tmp_path):
    artifact = decode_legacy_mcts_debug(json.loads(FIXTURE.read_text()))
    # Use two independently committed successful steps for sequence recovery.
    first = artifact.steps[0]
    second = replace(first, step=2)
    context = _context()
    path = tmp_path / "planning.stack"
    records = []
    with StackRunStore.create_mcts(path, replace(artifact, steps=())) as writer:
        writer.append_run_context(context)
        for step in (first, second):
            candidate = next(item for item in step.candidates if item.selected)
            problem = PlacementPlanningProblem(
                "world", Action(candidate.object_index, POSE, POSE, object_id=candidate.object_id),
                POSE, POSE, (),
            )
            result = PlacementPlanningResult("success", (
                MotionPlanSegment("pick", [[0, 0], [0.1, 0.2]], [0, 1], [0.1, 0.0]),
                MotionPlanSegment("place", [[0.1, 0.2], [0, 0]], [0, 1], [0, 0.1]),
            ))
            record = PlacementPlanRecord(
                step.step, 1, problem, result, step.selected_candidate_id, context.sha256,
            )
            writer.append_placement_plan(record)
            writer.append_mcts_step(step, statistics={"step": step.step, "elapsed_seconds": 2.5})
            records.append(record)
    return path, context, records


def _event(command, sequence, status, **kwargs):
    return ExecutionEvent(command.command_id, sequence, 1000 + sequence, status, **kwargs)


def test_mock_executor_round_trip_preserves_planning_and_step_order(tmp_path):
    path, context, plans = _run(tmp_path)
    before = StackRunStore.info(path)
    command = build_execution_command(plans[0], context)
    next_command = build_execution_command(plans[1], context)

    class MockExecutor:
        def execute(self, command, *, progress):
            for index, status in enumerate(("running", "held", "running", "succeeded"), 1):
                progress(_event(command, index, status))
            return ExecutionResult(command.command_id, "succeeded")

    with StackRunStore.open_append(path) as writer:
        with pytest.raises(ValueError, match="earlier planned"):
            writer.append_execution_command(next_command)
        writer.append_execution_command(command)
        result = MockExecutor().execute(command, progress=writer.append_execution_event)
        assert result.status is ExecutionStatus.SUCCEEDED
    restored = StackRunStore.load_execution_commands(path)[0]
    np.testing.assert_array_equal(restored.segments[0].positions, command.segments[0].positions)
    np.testing.assert_array_equal(restored.segments[0].openings, command.segments[0].openings)
    assert StackRunStore.load_execution_results(path)[0].status is ExecutionStatus.SUCCEEDED
    checkpoint = StackRunStore.load_execution_checkpoint(path)
    assert checkpoint.completed_steps == (1,)
    assert checkpoint.pending_steps == (2,)
    assert not checkpoint.requires_reconciliation
    assert StackRunStore.info(path).completed_utc == before.completed_utc
    assert StackRunStore.info(path).status == before.status
    assert len(StackRunStore.load_mcts(path).steps) == 2
    assert [plan.step for plan in StackRunStore.load_committed_placement_plans(path)] == [1, 2]
    with StackRunStore.open_append(path) as writer:
        with pytest.raises(ValueError, match="already been executed"):
            writer.append_execution_command(build_execution_command(plans[0], context, execution_attempt=2))
        writer.append_execution_command(next_command)


def test_interrupted_attempt_requires_reconciliation_before_retry(tmp_path):
    path, context, plans = _run(tmp_path)
    first = build_execution_command(plans[0], context)
    retry = build_execution_command(plans[0], context, execution_attempt=2)
    with StackRunStore.open_append(path) as writer:
        writer.append_execution_command(first)
        writer.append_execution_event(_event(first, 1, "running", segment_index=0, sample_index=1))
    checkpoint = StackRunStore.load_execution_checkpoint(path)
    assert checkpoint.completed_steps == ()
    assert checkpoint.unconfirmed_command_ids == (first.command_id,)
    with StackRunStore.open_append(path) as writer:
        with pytest.raises(ValueError, match="reconciliation"):
            writer.append_execution_command(retry)
        with pytest.raises(ValueError, match="failed/cancelled"):
            writer.append_execution_reconciliation(ExecutionReconciliation(first.command_id, 2000, "checked", {}))
        writer.append_execution_event(_event(
            first, 2, "failed", failure=ExecutionFailure.INTERRUPTED, reason="controller stopped after disconnect",
        ))
        with pytest.raises(ValueError, match="reconciliation"):
            writer.append_execution_command(retry)
        writer.append_execution_reconciliation(ExecutionReconciliation(
            first.command_id, 2000, "robot stopped; pickup scene restored", {"checked_by": "mock"},
        ))
        writer.append_execution_command(retry)
        writer.append_execution_event(_event(retry, 1, "running"))
        writer.append_execution_event(_event(retry, 2, "succeeded"))
    assert not StackRunStore.load_execution_checkpoint(path).requires_reconciliation
    assert len(StackRunStore.load_execution_results(path)) == 2


def test_queued_command_and_failed_writer_context_never_imply_completion(tmp_path):
    path, context, plans = _run(tmp_path)
    command = build_execution_command(plans[0], context)
    with pytest.raises(RuntimeError):
        with StackRunStore.open_append(path) as writer:
            writer.append_execution_command(command)
            raise RuntimeError("disconnected before receiving telemetry")
    assert StackRunStore.load_execution_checkpoint(path).unconfirmed_command_ids == (command.command_id,)
    assert StackRunStore.info(path).status == "complete"  # Planning status is independent.


def test_event_validation_rolls_back_and_terminal_events_are_final(tmp_path):
    path, context, plans = _run(tmp_path)
    command = build_execution_command(plans[0], context)
    with StackRunStore.open_append(path) as writer:
        writer.append_execution_command(command)
        with pytest.raises(ValueError, match="sequence must be 1"):
            writer.append_execution_event(_event(command, 2, "running"))
        with pytest.raises(ValueError, match="first execution"):
            writer.append_execution_event(_event(command, 1, "succeeded"))
        with pytest.raises(ValueError, match="segment index"):
            writer.append_execution_event(_event(command, 1, "running", segment_index=99))
        writer.append_execution_event(_event(command, 1, "running"))
        with pytest.raises(ValueError, match="sample index"):
            writer.append_execution_event(_event(command, 2, "running", segment_index=0, sample_index=99))
        writer.append_execution_event(_event(command, 2, "succeeded"))
        with pytest.raises(ValueError, match="terminal"):
            writer.append_execution_event(_event(command, 3, "running"))
    assert len(StackRunStore.load_execution_events(path)) == 2


def test_plan_and_context_hashes_reject_changed_paths_and_profiles(tmp_path):
    path, context, plans = _run(tmp_path)
    command = build_execution_command(plans[0], context)
    with StackRunStore.open_append(path) as writer:
        with pytest.raises(ValueError, match="differs"):
            writer.append_execution_command(replace(command, plan_sha256="0" * 64))
        wrong_segment = replace(command.segments[0], positions=np.ones((2, 2)))
        with pytest.raises(ValueError, match="differs"):
            writer.append_execution_command(replace(command, segments=(wrong_segment, command.segments[1])))
    with pytest.raises(ValueError, match="not bound"):
        build_execution_command(plans[0], replace(context, profiles={"motion": {"speed": 1.0}}))
    with pytest.raises(ValueError, match="not bound"):
        build_execution_command(replace(plans[0], context_sha256=None), context)
    assert StackRunStore.load_execution_commands(path) == ()


def test_context_round_trip_is_immutable_and_detects_corruption(tmp_path):
    path, context, _ = _run(tmp_path)
    loaded = StackRunStore.load_run_context(path, context.sha256)
    assert loaded.sha256 == context.sha256
    with pytest.raises(TypeError):
        loaded.profiles["motion"]["speed"] = 2
    with pytest.raises(ValueError):
        loaded.fixed_geometry[0].pose[0] = 99
    with StackRunStore.open_append(path) as writer:
        assert writer.append_run_context(context) == context.sha256
    with sqlite3.connect(path) as connection:
        payload = connection.execute("SELECT payload_json FROM records WHERE kind='planning_context'").fetchone()[0]
        changed = json.loads(payload)
        changed["profiles"]["motion"]["speed"] = 3
        connection.execute("UPDATE records SET payload_json=? WHERE kind='planning_context'", (json.dumps(changed),))
    with pytest.raises(ValueError, match="hash mismatch"):
        StackRunStore.load_run_context(path, context.sha256)


def test_legacy_plan_decode_remains_readable_without_context(tmp_path):
    _, _, plans = _run(tmp_path)
    payload = encode_placement_plan(plans[0])
    payload.pop("context_sha256")
    assert decode_placement_plan(payload).context_sha256 is None


def test_exports_are_derived_and_preserve_existing_files(tmp_path):
    path, context, plans = _run(tmp_path)
    command = build_execution_command(plans[0], context)
    with StackRunStore.open_append(path) as writer:
        writer.append_execution_command(command)
        writer.append_execution_event(_event(command, 1, "running"))
        writer.append_execution_event(_event(command, 2, "succeeded"))
    before = path.read_bytes()
    summary = summarize_run(path)
    assert summary["executions"][0]["status"] == "succeeded"
    assert summary["checkpoint"]["completed_steps"] == (1,)
    exported = export_run_summary(path, tmp_path / "summary.json")
    assert json.loads(exported.read_text())["steps"][0]["statistics"]["elapsed_seconds"] == 2.5
    csv_path = export_run_summary(path, tmp_path / "summary.csv")
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert json.loads(rows[0]["executions_json"])[0]["status"] == "succeeded"
    with pytest.raises(FileExistsError):
        export_run_summary(path, exported)
    assert path.read_bytes() == before


def test_open_append_does_not_create_or_replace_documents(tmp_path):
    missing = tmp_path / "missing.stack"
    with pytest.raises(sqlite3.OperationalError):
        StackRunStore.open_append(missing)
    assert not missing.exists()
    wrong = tmp_path / "wrong.stack"
    wrong.write_bytes(b"not a SQLite file")
    with pytest.raises(sqlite3.DatabaseError):
        StackRunStore.open_append(wrong)
    assert wrong.read_bytes() == b"not a SQLite file"


def test_execution_values_require_valid_immutable_paths(tmp_path):
    _, context, plans = _run(tmp_path)
    command = build_execution_command(plans[0], context)
    with pytest.raises(ValueError):
        command.segments[0].positions[0, 0] = 2
    with pytest.raises(ValueError, match="strictly increase"):
        replace(command, segments=(MotionPlanSegment("move", np.zeros((2, 2)), [0, 0]),))
    with pytest.raises(ValueError, match="joint names"):
        replace(command, joint_names=("only_one",))
    with pytest.raises(ValueError, match="failure and reason"):
        _event(command, 1, "failed")
    with pytest.raises(ValueError):
        _event(command, 1, "running", diagnostics={"bad": float("nan")})


def test_two_writers_cannot_queue_the_same_step_twice(tmp_path):
    path, context, plans = _run(tmp_path)
    first = build_execution_command(plans[0], context)
    second = build_execution_command(plans[0], context)
    with StackRunStore.open_append(path) as left, StackRunStore.open_append(path) as right:
        left.append_execution_command(first)
        with pytest.raises(ValueError, match="reconciliation"):
            right.append_execution_command(second)
    assert len(StackRunStore.load_execution_commands(path)) == 1


def test_context_assets_accept_embedded_bytes_and_detect_changed_files(tmp_path):
    data = b"mesh source"
    digest = hashlib.sha256(data).hexdigest()
    context = replace(_context(), asset_hashes={"object_1": digest})
    verify_context_assets(context, {"object_1": data})
    mesh = tmp_path / "mesh.obj"
    mesh.write_bytes(data)
    verify_context_assets(context, {"object_1": mesh})
    mesh.write_bytes(b"changed mesh")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_context_assets(context, {"object_1": mesh})
    with pytest.raises(ValueError, match="roles"):
        verify_context_assets(context, {})


def test_planning_checkpoint_commits_atomically_and_preserves_domain_state(tmp_path):
    artifact = decode_legacy_mcts_debug(json.loads(FIXTURE.read_text()))
    context = _context()
    pose = np.array([0.041, 0.001, 0.061, 0, 0, 0, 1])
    # Native posegen maps contact entity IDs (integers) to vectors. This shape
    # reaches action_history in a real simulation, unlike kinematic-only tests.
    contact_diagnostics = {
        name: {np.int64(5): np.array([0., 0., 1.])}
        for name in ("posegen_contact_points", "posegen_contact_normals", "posegen_contact_forces")
    }
    action = Action(1, pose, pose, object_id=22, diagnostics=contact_diagnostics)
    trajectory = StackObjectTrajectory(22, [POSE, pose], 1.5, 2.5, 2, 0.01, 0.02,
                                       times=[0, 0.2])
    contact = ContactPoint(1, 2, 0, 1, False, False, [0, 0, 0], [0, 0, 1], [0, 0, 1], 0.001)
    state = State([11, 22], [1], {22: pose}, [{22: trajectory}], [action],
                  contact_points=[contact], pose_identified_object_ids={11, 22})
    checkpoint = PlanningCheckpoint(
        1, context.sha256, state,
        PickSceneSnapshot("world", {11: POSE, 22: POSE}, "recorded", {"capture": "test"}),
        FrameTransform("world", "stack", [0, 0, 0, 0, 0, 0, 1]),
    )
    state.object_poses[22][0] = 999
    path = tmp_path / "checkpoint.stack"
    with StackRunStore.create_mcts(path, replace(artifact, steps=())) as writer:
        writer.append_run_context(context)
        with pytest.raises(ValueError, match="missing planning_context"):
            writer.append_mcts_step(artifact.steps[0], checkpoint=replace(checkpoint, context_sha256="0" * 64))
        assert StackRunStore.load_mcts(path).steps == ()
        writer.append_mcts_step(artifact.steps[0], checkpoint=checkpoint)
    restored = StackRunStore.load_planning_checkpoint(path)
    assert restored.state.trajectories[0][22].times == [0, 0.2]
    assert restored.step == 1
    assert restored.state.placed_object_indices == [1]
    assert restored.state.pose_identified_object_ids == {11, 22}
    np.testing.assert_allclose(restored.state.object_poses[22], pose)
    assert restored.state.trajectories[0][22].vel_integral == 2.5
    assert restored.state.contact_points[0].gap == 0.001
    assert restored.pick_scene.metadata["capture"] == "test"
    for name in contact_diagnostics:
        np.testing.assert_array_equal(restored.state.action_history[0].diagnostics[name]["5"], [0., 0., 1.])
        assert 5 in checkpoint.state.action_history[0].diagnostics[name]
    # A newer committed step without a checkpoint must not silently use old state.
    with StackRunStore.open_append(path) as writer:
        writer.append_mcts_step(replace(artifact.steps[0], step=2))
    with pytest.raises(ValueError, match="missing planning_checkpoint/2"):
        StackRunStore.load_planning_checkpoint(path)


def test_committed_plan_loader_rejects_search_only_run(tmp_path):
    artifact = decode_legacy_mcts_debug(json.loads(FIXTURE.read_text()))
    path = StackRunStore.save_mcts(artifact, tmp_path / "search.stack")
    with pytest.raises(ValueError, match="selected successful motion plan"):
        StackRunStore.load_committed_placement_plans(path)
