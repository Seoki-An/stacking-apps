"""Checkpoint branching and native-worker interruption never erase source runs."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

from stacking_common.artifacts import (
    PlacementPlanRecord, PlanningCheckpoint, PlanningRunContext, StackRunStore,
    build_execution_command, decode_legacy_mcts_debug, export_run_summary,
)
from stacking_common.domain import Action, State
from stacking_common.planning import (
    FrameTransform, MotionPlanSegment, PickSceneSnapshot,
    PlacementPlanningProblem, PlacementPlanningResult,
)
from stacking_tabletop import TabletopApplication
from stacking_tabletop.planning import load_native_solver_profile
from stacking_tabletop.planning.persistence import (
    branch_planning_run, build_planning_context, validate_planning_resume,
)
from stacking_tabletop.planning.process import stop_planning_process, supervise_planning


ROOT = Path(__file__).resolve().parents[1]
POSE = np.array([0., 0., .1, 0., 0., 0., 1.])


def _checkpoint_run(path):
    fixture = ROOT / "tests/fixtures/legacy_visualization/mcts_candidate_viewer.json"
    artifact = decode_legacy_mcts_debug(json.loads(fixture.read_text()))
    step = artifact.steps[0]
    candidate = next(item for item in step.candidates if item.selected)
    context = PlanningRunContext("world", ("a", "b"), {"motion": "test"})
    pick = PickSceneSnapshot("world", {11: POSE, 22: POSE}, "test")
    action = Action(candidate.object_index, POSE, POSE, object_id=candidate.object_id)
    state = State([11, 22], [candidate.object_index], {candidate.object_id: POSE}, [], [action])
    checkpoint = PlanningCheckpoint(1, context.sha256, state, pick,
        FrameTransform("world", "stack", [0, 0, 0, 0, 0, 0, 1]))
    plan = PlacementPlanRecord(1, 1,
        PlacementPlanningProblem("world", action, POSE, POSE, ()),
        PlacementPlanningResult("success", (MotionPlanSegment("move", [[0, 0], [.1, .2]]),)),
        step.selected_candidate_id, context.sha256)
    with StackRunStore.create_mcts(path, replace(artifact, steps=()), metadata={
        "request": {"plan_motion": True},
        "config": {"path": str(ROOT / "apps/tabletop/config/mcts_cpu.yml")},
    }) as writer:
        writer.append_run_context(context)
        writer.append_placement_plan(plan)
        writer.append_mcts_step(step, statistics={"step": 1, "elapsed_seconds": 2.}, checkpoint=checkpoint)
        # A crash can leave a completed motion attempt without a selected step.
        writer.append_placement_plan(replace(plan, step=2))
    return context, checkpoint, plan


def test_branch_preserves_prefix_and_retries_partial_step_in_new_document(tmp_path):
    source, destination = tmp_path / "source.stack", tmp_path / "branch.stack"
    context, checkpoint, plan = _checkpoint_run(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    writer, restored = branch_planning_run(source, destination, context)
    with writer:
        assert restored.step == 1
        restored_plans = StackRunStore.load_placement_plans(destination)
        assert len(restored_plans) == 1
        assert restored_plans[0].context_sha256 == plan.context_sha256
        np.testing.assert_allclose(restored_plans[0].problem.place_pose, plan.problem.place_pose)
        assert len(StackRunStore.load_mcts(destination).steps) == 1
        assert StackRunStore.load_mcts_statistics(destination)[0]["elapsed_seconds"] == 2.
        # Step-2 attempt IDs are free in the branch, even though source has one.
        writer.append_placement_plan(replace(plan, step=2))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert StackRunStore.load_planning_checkpoint(destination).step == checkpoint.step
    for suffix in ("json", "csv"):
        target = tmp_path / f"summary.{suffix}"
        export_run_summary(destination, target)
        assert target.stat().st_size > 0
        with pytest.raises(FileExistsError):
            export_run_summary(destination, target)


def test_operations_summary_and_resume_controls(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QMainWindow
    from types import SimpleNamespace
    import stacking_tabletop.gui.operations.window as windows
    import stacking_tabletop.gui.operations.planning_run as dialogs

    qt = QApplication.instance() or QApplication([])
    source, output = tmp_path / "source.stack", tmp_path / "summary.json"
    _checkpoint_run(source)

    class Window(windows.StackingOperationsWindow):
        def __init__(self):
            QMainWindow.__init__(self)
            self._artifact_path = source
            self._planning_dialog = None
            self._video_exporting = False
            self.repository = ROOT

    window = Window()
    monkeypatch.setattr(windows.QFileDialog, "getSaveFileName", lambda *a: (str(output), "JSON (*.json)"))
    window._export_summary()
    assert json.loads(output.read_text())["steps"][0]["step"] == 1
    launches = []

    class Dialog:
        def __init__(self, *args):
            launches.append(args)
            self.run_available = SimpleNamespace(connect=lambda slot: None)
            self.running = False

        def show(self):
            pass

    monkeypatch.setattr(dialogs, "PlanningRunDialog", Dialog)
    destination = tmp_path / "new.stack"
    monkeypatch.setattr(windows.QFileDialog, "getSaveFileName", lambda *a: (str(destination), "Stacking runs (*.stack)"))
    window._resume_planning()
    assert launches[0][:3] == (ROOT, source, destination)
    window.close()
    qt.processEvents()


def test_resume_refuses_changed_context_existing_output_and_execution(tmp_path):
    source = tmp_path / "source.stack"
    context, _, plan = _checkpoint_run(source)
    with pytest.raises(ValueError, match="context mismatch"):
        validate_planning_resume(source, replace(context, profiles={"motion": "different"}))
    with pytest.raises(ValueError, match="context mismatch"):
        validate_planning_resume(source, replace(context, asset_hashes={"changed": "0" * 64}))
    with pytest.raises(FileExistsError):
        branch_planning_run(source, source, context)
    with StackRunStore.open_append(source) as writer:
        writer.append_execution_command(build_execution_command(plan, context))
    with pytest.raises(ValueError, match="physical reconciliation"):
        validate_planning_resume(source, context)


def test_resume_refuses_a_still_running_source(tmp_path):
    source = tmp_path / "source.stack"
    context, checkpoint, _ = _checkpoint_run(source)
    artifact = StackRunStore.load_mcts(source)
    active = tmp_path / "active.stack"
    with StackRunStore.create_mcts(active, replace(artifact, steps=())) as writer:
        writer.append_run_context(context)
        writer.append_mcts_step(artifact.steps[0], checkpoint=checkpoint)
        with pytest.raises(ValueError, match="still marked running"):
            validate_planning_resume(active, context)


def test_tabletop_context_records_native_joint_order_profiles_and_assets():
    app = TabletopApplication.from_repository(ROOT)
    pick = PickSceneSnapshot("world", {1: POSE}, "test")
    inventory = app.load_inventory([1], pick.object_poses)
    solver = load_native_solver_profile(app.solver_profiles.grasp, app.solver_profiles.motion)
    context = build_planning_context(app, inventory, pick, solver, {"iterations": 5})
    assert context.joint_names == ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                                   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    assert "stacking_core_native" in context.asset_hashes
    assert "object/1" in context.asset_hashes
    assert any("dsf" in role for role in context.asset_hashes)
    assert context.profiles["native_solver"]["grasp"]
    sampling = context.profiles["action_generation_profile"]["sampling_settings"]
    assert sampling == app.sampling_settings
    assert len(context.fixed_geometry) == len(app.planning_profile.obstacles)
    assert context.sha256 == build_planning_context(app, inventory, pick, solver, {"iterations": 5}).sha256
    changed = replace(app, planning_profile=replace(app.planning_profile, obstacle_friction=2.))
    assert context.sha256 != build_planning_context(changed, inventory, pick, solver, {"iterations": 5}).sha256
    changed = replace(app, action_generation_profile=replace(app.action_generation_profile,
        sampling_settings={**app.sampling_settings,
                           "score": {**app.sampling_settings["score"], "fill_area": 0.}}))
    assert context.sha256 != build_planning_context(changed, inventory, pick, solver, {"iterations": 5}).sha256


def test_supervisor_wall_timeout_and_normal_completion():
    started = time.monotonic()
    assert supervise_planning([sys.executable, "-c", "import time; time.sleep(60)"], max_seconds=.1) == 124
    assert time.monotonic() - started < 3
    assert supervise_planning([sys.executable, "-c", "pass"], max_seconds=3) == 0
    for value in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            supervise_planning([sys.executable, "-c", "pass"], max_seconds=value)


def test_supervisor_marks_interrupted_journal_and_checkpoint_survives(tmp_path):
    source = tmp_path / "source.stack"
    context, _, _ = _checkpoint_run(source)
    destination = tmp_path / "interrupted.stack"
    # Copying a closed fixture simulates a worker having committed one step.
    program = (
        "import shutil,time; "
        f"shutil.copyfile({str(source)!r}, {str(destination)!r}); "
        "time.sleep(60)"
    )
    assert supervise_planning([sys.executable, "-c", program], destination=destination, max_seconds=.5) == 124
    assert StackRunStore.info(destination).status == "cancelled"
    assert validate_planning_resume(destination, context).step == 1
    with pytest.raises(FileExistsError):
        supervise_planning([sys.executable, "-c", "pass"], destination=destination)


@pytest.mark.skipif(os.name != "posix", reason="tabletop process groups require POSIX")
def test_stop_escalates_when_native_worker_ignores_sigterm():
    process = subprocess.Popen([sys.executable, "-u", "-c",
        "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready'); time.sleep(60)"],
        stdout=subprocess.PIPE, start_new_session=True)
    try:
        assert process.stdout.readline() == b"ready\n"
        stop_planning_process(process, grace_seconds=.1)
        assert process.returncode == -signal.SIGKILL
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_ctrl_c_reaches_supervisor_while_worker_ignores_interrupts():
    worker = "import os,signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); print(os.getpid(),flush=True); time.sleep(60)"
    parent = (
        "import sys; from stacking_tabletop.planning.process import supervise_planning; "
        f"sys.exit(supervise_planning([sys.executable,'-u','-c',{worker!r}]))"
    )
    process = subprocess.Popen([sys.executable, "-u", "-c", parent], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        worker_pid = int(process.stdout.readline())
        process.send_signal(signal.SIGINT)
        output, _ = process.communicate(timeout=5)
        assert process.returncode == 130, output
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize('scan_enabled', [True, False])
def test_debug_cli_captures_and_resumes_planning_checkpoint(monkeypatch, tmp_path, scan_enabled):
    import importlib.util
    from stacking_tabletop.planning.coordinator import TabletopPlanningCoordinator
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location("debug_restart", ROOT / "apps/tabletop/scripts/debug/mcts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    original_solve = module._PassThroughPoseSolver.solve

    def solve_with_native_contacts(*args):
        result = original_solve(*args)
        return replace(result, diagnostics={
            **result.diagnostics,
            "posegen_contact_points": {5: np.array([0., 0., 0.])},
            "posegen_contact_normals": {5: np.array([0., 0., 1.])},
            "posegen_contact_forces": {5: np.array([0., 0., .5])},
        })

    monkeypatch.setattr(module._PassThroughPoseSolver, "solve", solve_with_native_contacts)

    def coordinator(app, inventory, **kwargs):
        def problem(state, action, *, pick_pose, metadata):
            return app.build_placement_planning_problem(state, action, pick_pose=pick_pose,
                                                        metadata=metadata)
        planner = SimpleNamespace(plan=lambda problem: PlacementPlanningResult("success", (
            MotionPlanSegment("test", [app.planning_profile.robot.home_positions] * 2),
        )))
        return TabletopPlanningCoordinator(inventory, problem, planner,
            world_T_stack=app.planning_profile.world_T_stack,
            context_sha256=kwargs.get("context_sha256"))

    monkeypatch.setattr(TabletopApplication, "build_planning_coordinator", coordinator)
    source, destination = tmp_path / "first.stack", tmp_path / "resumed.stack"
    common = ["mcts.py", "--config", str(ROOT / "apps/tabletop/config/mcts_cpu.yml")]
    monkeypatch.setattr(sys, "argv", [*common, "--object-ids", "1", "3", "--steps", "1",
        "--iterations", "2", "--pose-solver", "passthrough", "--transition-backend", "kinematic",
        "--plan-motion", "--save-run", str(source),
        "--pose-from-scan" if scan_enabled else "--no-pose-from-scan"])
    module.main()
    checkpoint = StackRunStore.load_planning_checkpoint(source)
    assert checkpoint.step == 1
    recorded = StackRunStore.load_run_context(source, checkpoint.context_sha256)
    assert recorded.profiles['action_generation_profile'].get('pose_from_scan', True) is scan_enabled
    assert "5" in checkpoint.state.action_history[0].diagnostics["posegen_contact_forces"]
    assert StackRunStore.load_committed_placement_plans(source)[0].context_sha256 == checkpoint.context_sha256
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(sys, "argv", [*common, "--resume-run", str(source), "--save-run", str(destination),
        "--no-pose-from-scan" if scan_enabled else "--pose-from-scan"])
    with pytest.raises(ValueError, match='preserves recorded pose_from_scan'):
        module.main()
    assert not destination.exists()
    monkeypatch.setattr(sys, "argv", [*common, "--resume-run", str(source), "--save-run", str(destination), "--steps", "1"])
    module.main()
    assert StackRunStore.load_planning_checkpoint(destination).step == 2
    assert StackRunStore.load_planning_checkpoint(destination, step=1).step == 1
    with pytest.raises(ValueError, match="no committed planning step 3"):
        StackRunStore.load_planning_checkpoint(destination, step=3)
    assert [item.step for item in StackRunStore.load_committed_placement_plans(destination)] == [1, 2]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
