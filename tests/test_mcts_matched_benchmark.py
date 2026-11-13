"""The matched throughput protocol must preserve work settings and stay prepare-only."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import benchmark_excavator_mcts_matched as benchmark
from stacking_common.artifacts import StackRunStore
from stacking_common.artifacts.planning_context import _values
from stacking_common.profiles.mcts import load_mcts_profile
from stacking_excavator import ExcavatorApplication


def test_prepare_never_builds_or_runs_search(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("preparation must not construct or execute a search")

    monkeypatch.setattr(benchmark.subprocess, "run", forbidden)
    monkeypatch.setattr(ExcavatorApplication, "build_mcts", forbidden)
    monkeypatch.setattr(ExcavatorApplication, "load_inventory", forbidden)
    output = tmp_path / "prepared"
    benchmark.main(["prepare", "--output", str(output), "--iterations", "512",
                    "--batch-sizes", "12", "32"])
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["status"] == "prepared"
    assert protocol["max_time_seconds"] is None
    assert protocol["target_shape"] == "default"
    assert len(protocol["trials"]) == 12
    assert not list((output / "runs").iterdir())
    assert not (output / "summary.json").exists()
    for batch in (12, 32):
        profile = load_mcts_profile(output / "configs" / f"batch-{batch}.yml")
        assert profile.request.max_time_seconds is None
        assert profile.request.iterations == 512
        assert profile.tree.generation_batch_size == profile.tree.evaluation_batch_size == batch
    for trial in protocol["trials"]:
        command = trial["command"]
        assert command[command.index("--target-shape") + 1] == "default"
        assert command[command.index("--early-stops") + 1] == "excursion"
        if trial["backend"] == "cuda":
            assert command[command.index("--placement-batch-size") + 1] == str(trial["batch_size"])
            assert command[command.index("--posegen-batch-size") + 1] == str(trial["batch_size"])
    benchmark.verify_inputs(protocol)
    with pytest.raises(FileExistsError):
        benchmark.main(["prepare", "--output", str(output)])


def test_shared_batch_profile_keeps_the_search_policy():
    base = yaml.safe_load((benchmark.ROOT / "apps/excavator/config/mcts_cpu.yml").read_text())
    snapshot = deepcopy(base)
    matched = benchmark.matched_config(base, 32, 1024)
    for key in ("root_cem", "widening", "final_selection", "bootstrap"):
        assert matched[key] == base[key]
    assert matched["tree"]["max_depth"] == base["tree"]["max_depth"]
    assert base == snapshot


def test_recorded_scene_from_another_target_is_rejected(tmp_path, monkeypatch):
    inventory = ExcavatorApplication.from_repository(benchmark.ROOT, target_shape="default").inventory_profile
    previous = _values(inventory)
    previous["target"]["width"] += 1.
    checkpoint = SimpleNamespace(context_sha256="context", state=SimpleNamespace(object_ids=inventory.object_ids))
    context = SimpleNamespace(profiles={"inventory_profile": previous})
    monkeypatch.setattr(StackRunStore, "load_planning_checkpoint", lambda *a, **kw: checkpoint)
    monkeypatch.setattr(StackRunStore, "load_run_context", lambda *a, **kw: context)
    with pytest.raises(ValueError, match="recorded target differs"):
        benchmark.export_scene("placed10=unused.stack:10", tmp_path, inventory)
    assert not list(tmp_path.iterdir())


def test_input_changes_are_rejected(tmp_path):
    path = tmp_path / "input.yml"
    path.write_text("initial")
    protocol = {"input_sha256": {str(path): benchmark.digest(path)}}
    path.write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        benchmark.verify_inputs(protocol)


def trial_results(tmp_path):
    trials = []
    for seed, nodes in enumerate((20, 80)):
        for backend, multiplier in (("cpu", 1), ("cuda", 2)):
            elapsed = 10 * (seed + 1)
            path = tmp_path / f"{backend}-{seed}.json"
            result = dict(max_time_seconds=None, timed_out=False, requested_iterations=512,
                          generated_candidates=nodes * multiplier, elapsed_seconds=elapsed,
                          expanded_nodes_per_second=nodes * multiplier / elapsed,
                          evaluation_batch_sizes=[2, 4], evaluated_candidates=nodes * multiplier,
                          failed_evaluations=1, completed_iterations=512, selected=None,
                          state_sha256="same-state", source_sha256={"source": "same"},
                          core_native_sha256="same-binary", inventory_profile={"target": "default"},
                          profiles={"mcts_profile": {"parallel": {"workers": 12 if backend == "cpu" else 2,
                                                                   "posegen_backend": backend,
                                                                   "simulation_backend": backend},
                                                     "discount": .9}, "physics": {"friction": 1.}})
            benchmark.write_json(path, result)
            trials.append(dict(scene="empty", batch_size=32, backend=backend, seed=seed, output=str(path)))
    return dict(iterations=512, trials=trials)


def test_summary_reports_mean_and_pooled_throughput(tmp_path):
    summary = benchmark.summarize(trial_results(tmp_path))
    rows = {row["backend"]: row for row in summary["groups"]}
    assert rows["cpu"]["mean_nodes_per_second"] == 3.
    assert rows["cuda"]["mean_nodes_per_second"] == 6.
    assert rows["cpu"]["pooled_nodes_per_second"] == pytest.approx(100 / 30)
    assert summary["speedups"][0]["gpu_over_cpu"] == 2.
    assert rows["cpu"]["mean_actual_evaluation_batch"] == 3.


@pytest.mark.parametrize("change", ["time_budget", "physics", "native", "backend_failure"])
def test_summary_rejects_unmatched_or_failed_trials(tmp_path, change):
    protocol = trial_results(tmp_path)
    path = Path(protocol["trials"][1]["output"])
    result = json.loads(path.read_text())
    if change == "time_budget":
        result["max_time_seconds"] = 60.
    elif change == "physics":
        result["profiles"]["physics"]["friction"] = 2.
    elif change == "native":
        result["core_native_sha256"] = "changed"
    else:
        result["backend_failure"] = "CUDA capacity failure"
    benchmark.write_json(path, result)
    with pytest.raises(ValueError):
        benchmark.summarize(protocol)
