"""Excavator CLI diagnostics and progress lifecycle at the native boundary."""

import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
import yaml

from stacking_common.artifacts import StackRunStore
from stacking_common.search import MonteCarloTreeSearch, MCTSSearchResult
from stacking_common.workflows import mcts_reporting
from stacking_excavator.planning import cli

ROOT = Path(__file__).resolve().parents[1]


def test_leaf_counts_include_pending_and_failed_nodes():
    def node(children=(), *, evaluated=True, failed=False):
        return SimpleNamespace(children=children, evaluated=evaluated, failed=failed,
                               depth=0, rejected_expansions={}, rejection_reason=None)

    pending = node(evaluated=False)
    branch = node([pending, node(evaluated=False)])
    failed = node(failed=True)
    terminal = node()
    root = node([branch, failed, terminal])
    stats = mcts_reporting.tree_statistics(root)
    assert stats["nodes"] == 6
    assert stats["leaves"] == 4
    # The evaluated branch is also a frontier leaf, but is not childless.
    assert stats["frontier_leaves"] == 3
    assert mcts_reporting.tree_statistics(node())["leaves"] == 1


@pytest.mark.parametrize("outcome", ["success", "catalog", "exhausted", "capacity", "error"])
def test_excavator_cli_reports_search_and_stops_heartbeat(tmp_path, monkeypatch, capsys, outcome):
    succeeded = outcome in ("success", "catalog")
    reference = json.loads((ROOT / "tests/fixtures/legacy_planner/reference.json").read_text())
    pickup = tmp_path / "pick.json"
    pickup.write_text(json.dumps({str(i): reference["pickup_poses"][str(i)] for i in (56, 58, 8)}))
    values = yaml.safe_load((ROOT / "apps/excavator/config/mcts_cpu.yml").read_text())
    values["parallel"].update(enabled=False, workers=1)
    # Exercise successful native validation with the narrower ablation target.
    values["root_cem"].update(population=96, elite=24, iterations=1, keep=12)
    values["tree"]["root_max_children"] = 12
    config = tmp_path / "mcts.yml"
    config.write_text(yaml.safe_dump(values))
    reporters = []

    class Reporter(mcts_reporting.ProgressReporter):
        def __init__(self):
            super().__init__()
            reporters.append(self)

    monkeypatch.setattr(cli, "ProgressReporter", Reporter)
    if not succeeded:
        def search(self, root, request):
            if outcome == "error":
                raise RuntimeError("injected native search failure")
            return MCTSSearchResult(root=root, selected=None, completed_iterations=0,
                expansion_calls=0, generated_candidates=0, evaluated_candidates=0,
                failed_evaluations=0, effective_max_depth=1, timed_out=False,
                backend_failure=({'backend': 'posegen', 'reason': 'posegen_capacity_overflow',
                    'resource': 'bodies', 'required': 33, 'capacity': 32,
                    'message': 'posegen capacity exceeded for bodies: required 33, configured capacity 32'}
                    if outcome == 'capacity' else None))
        monkeypatch.setattr(MonteCarloTreeSearch, "search", search)
    output = tmp_path / "run.stack"
    inventory_args = (["--object-ids", "56", "58", "8"] if outcome == "catalog"
                      else ["--pick-pose-data", str(pickup)])
    status = cli.main(["--worker", "--repository", str(ROOT), "--config", str(config),
        *inventory_args, "--save-run", str(output), "--search-only",
        "--iterations", "2", "--max-search-time", "60", "--seed", "0"])
    assert status == {"success": 0, "catalog": 0, "exhausted": 1, "capacity": 1, "error": 2}[outcome]
    assert reporters and all(not reporter._thread.is_alive() for reporter in reporters)
    captured = capsys.readouterr()
    assert "debug workload: objects=3, steps=1, iterations/step=2" in captured.out
    assert "[Progress] step 1: MCTS search 1 started" in captured.out
    assert StackRunStore.info(output).status == ("complete" if succeeded else "failed")
    if outcome == "catalog":
        context, = StackRunStore.load_run_contexts(output)
        assert context.profiles["pick_scene"]["format"] == "synthetic_search_inventory"
        assert set(context.profiles["pick_scene"]["object_poses"]) == {"56", "58", "8"}
        assert "no recorded pickup poses" in captured.out
    if outcome == "error":
        assert "injected native search failure" in captured.err
        assert "MCTS search finished" not in captured.out
        return
    for message in ("[Progress] step 1: MCTS search finished", "[INFO] Search:",
                    "[INFO] Tree nodes:", "[INFO] Root candidates:",
                    "[Timing] MCTS phases:", "[Timing] MCTS total wall=", "stack run:", "return="):
        assert message in captured.out
    stats, = StackRunStore.load_mcts_statistics(output)
    assert "elapsed_seconds" in stats and "tree" in stats and "simulation" in stats
    assert f"leaves: {stats['tree']['leaves']}," in captured.out
    assert f"leaves={stats['tree']['leaves']}" in captured.out
    if outcome == 'capacity':
        assert stats['backend_failure']['required'] == 33
        assert stats['backend_failure']['capacity'] == 32
        assert 'posegen_capacity_overflow' in captured.out
        assert 'Search stopped: posegen capacity exceeded' in captured.out
    elif succeeded:
        assert "[INFO] Root CEM:" in captured.out
        assert "[INFO] CEM round 1:" in captured.out
        assert "[Progress] step 1 finished with complete" in captured.out
        assert "steps=1" in captured.out.split("return=")[-1]
    else:
        assert "no feasible candidate found" in captured.out
        assert "[Progress] step 1 finished with search_exhausted" in captured.out
        assert "steps=0" in captured.out.split("return=")[-1]


def test_heartbeat_reports_current_phase_and_stops(monkeypatch):
    heartbeat = threading.Event()
    messages = []

    def emit(message, **kwargs):
        messages.append(message)
        if "still running MCTS search 1 for step 2" in message:
            heartbeat.set()

    monkeypatch.setattr(mcts_reporting, "print", emit, raising=False)
    reporter = mcts_reporting.ProgressReporter(interval_seconds=.01)
    reporter.update("search_started", {"step": 2, "search_attempt": 1})
    reporter.start()
    try:
        assert heartbeat.wait(timeout=2)
    finally:
        reporter.stop()
    assert not reporter._thread.is_alive()
    assert any("phase=" in message and "total=" in message for message in messages)
