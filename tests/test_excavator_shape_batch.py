"""Long example batches preserve full inventory, progress, and partial results."""

from pathlib import Path
import sys

import numpy as np
import pytest

from stacking_common.search import MonteCarloTreeSearch
from stacking_common.artifacts import StackRunStore
from stacking_common.visualization import MCTSDebugArtifact
from stacking_excavator.planning import cli
from stacking_excavator.planning.shape_batch import (
    ROOT, digest, inspect_run, main, prepare_batch, run_shape, run_worker, shape_progress,
)
from test_excavator_search_resume import controlled_search


@pytest.fixture
def batch(tmp_path):
    output = tmp_path / "batch"
    original = digest(ROOT / "apps/excavator/config/inventory.yml")
    manifest = prepare_batch(ROOT, output, config=ROOT / "apps/excavator/config/mcts_cpu.yml",
        shapes=["one_sided_wall", "two_sided_wall", "pyramid", "box"])
    assert digest(ROOT / "apps/excavator/config/inventory.yml") == original
    return output, manifest


def test_profiles_freeze_all_49_stones_and_represent_the_four_shapes(batch):
    output, manifest = batch
    assert len(set(manifest["object_ids"])) == 49
    jobs = {job["name"]: job for job in manifest["jobs"]}
    assert set(jobs) == {"one_sided_wall", "two_sided_wall", "pyramid", "box"}
    for name, job in jobs.items():
        for relative, expected in job["profile_hashes"].items():
            assert digest(output / relative) == expected
        angles = job["target"]["taper_degrees"]
        if name == "one_sided_wall":
            continue  # Currently a narrow untapered wall example, not a comparison shape.
        slopes = sum(value > 0 for value in angles.values())
        assert slopes == {"two_sided_wall": 2, "pyramid": 4, "box": 0}[name]
        assert 17. < job["target_volume_m3"] < 22.
        if name == "two_sided_wall":
            assert angles["width_positive"] == angles["width_negative"] == 5.
        if name == "pyramid":
            target = job["target"]
            top_width = target["width"]-2*target["height"]*np.tan(np.deg2rad(angles["width_positive"]))
            assert top_width == pytest.approx(.7)


@pytest.mark.parametrize("status", ["running", "failed", "cancelled"])
def test_status_before_first_checkpoint_reports_zero_placements(batch, status, capsys):
    output, manifest = batch
    job = manifest["jobs"][0]
    path = output / job["name"] / "runs/attempt-001.stack"
    with StackRunStore.create_mcts(path, MCTSDebugArtifact("stack", ()),
            metadata={"platform": "excavator", "mode": "search"}) as writer:
        if status != "running":
            writer.finish(status)
        # Read while the writer is still open, as happens during the first search.
        record = inspect_run(path, manifest["object_ids"])
        assert record["status"] == status
        assert record["placed"] == 0 and record["step"] == 0 and not record["complete"]
        assert main(["--output", str(output), "--status"]) == 0
        assert f"{job['name']}: 0/49 placed" in capsys.readouterr().out


def test_status_does_not_hide_invalid_checkpoint_errors(batch, monkeypatch):
    output, manifest = batch
    path = output / manifest["jobs"][0]["name"] / "runs/attempt-001.stack"
    with StackRunStore.create_mcts(path, MCTSDebugArtifact("stack", ()),
            metadata={"platform": "excavator", "mode": "search"}):
        def invalid_checkpoint(path):
            raise ValueError("checkpoint and context frames differ")
        monkeypatch.setattr("stacking_excavator.planning.shape_batch.load_resume_checkpoint", invalid_checkpoint)
        with pytest.raises(ValueError, match="checkpoint and context frames differ"):
            inspect_run(path, manifest["object_ids"])


def test_interruption_resumes_committed_state_then_skips_fully_used_inventory(batch, monkeypatch):
    output, manifest = batch
    job = next(job for job in manifest["jobs"] if job["name"] == "box")
    seen = []
    monkeypatch.setattr(MonteCarloTreeSearch, "search", controlled_search(seen, interrupt_at=3))
    commands = []

    def worker(command, *args, **kwargs):
        commands.append(command)
        try:
            return cli.main(command[4:])
        except KeyboardInterrupt:
            return 130

    monkeypatch.setattr("stacking_excavator.planning.shape_batch.run_worker", worker)
    with pytest.raises(KeyboardInterrupt):
        run_shape(output, job, manifest, seconds=30., attempts=1, lock_fd=0)
    partial = shape_progress(output, job, manifest["object_ids"])
    assert partial["placed"] == 2 and not partial["complete"]
    source = Path(partial["best_run"]["path"])
    source_hash = digest(source)
    resumed = []
    monkeypatch.setattr(MonteCarloTreeSearch, "search", controlled_search(resumed))
    result = run_shape(output, job, manifest, seconds=60., attempts=1, lock_fd=0)
    assert result["placed"] == 49 and result["complete"]
    assert len(resumed) == 47
    assert len(resumed[0].placed_object_indices) == 2
    assert commands[-1][commands[-1].index("--steps")+1] == "47"
    assert "--resume-run" in commands[-1] and "--seed" not in commands[-1]
    assert digest(source) == source_hash
    run_shape(output, job, manifest, seconds=1., attempts=1, lock_fd=0)
    assert len(commands) == 2


def test_clean_exit_after_one_step_is_incomplete_and_no_progress_is_not_retried(batch, monkeypatch):
    output, manifest = batch
    job = manifest["jobs"][0]
    monkeypatch.setattr(MonteCarloTreeSearch, "search", controlled_search([]))

    def one_step(command, *args, **kwargs):
        command = command.copy()
        command[command.index("--steps")+1] = "1"
        return cli.main(command[4:])

    monkeypatch.setattr("stacking_excavator.planning.shape_batch.run_worker", one_step)
    result = run_shape(output, job, manifest, seconds=30., attempts=1, lock_fd=0)
    assert result["last_exit_code"] == 0 and result["placed"] == 1 and not result["complete"]
    calls = []
    monkeypatch.setattr("stacking_excavator.planning.shape_batch.run_worker",
                        lambda *args, **kwargs: calls.append(args) or 2)
    run_shape(output, job, manifest, seconds=30., attempts=3, lock_fd=0)
    assert len(calls) == 1
    assert len(list((output / job["name"] / "runs").glob("*.command.json"))) == 2


def test_modified_snapshot_is_rejected_before_worker_start(batch, monkeypatch):
    output, manifest = batch
    job = manifest["jobs"][0]
    path = output / next(iter(job["profile_hashes"]))
    path.write_text(path.read_text()+"\n# changed after preparation\n")
    monkeypatch.setattr("stacking_excavator.planning.shape_batch.run_worker",
                        lambda *args, **kwargs: pytest.fail("must reject changed configuration"))
    with pytest.raises(ValueError, match="frozen profile changed"):
        run_shape(output, job, manifest, seconds=1., attempts=1, lock_fd=0)


def test_worker_time_limit_returns_without_waiting_for_native_work(tmp_path):
    with (tmp_path / "lock").open("w") as lock:
        code = run_worker([sys.executable, "-c", "import time; time.sleep(60)"],
            tmp_path / "absent.stack", tmp_path / "worker.log", .05, lock_fd=lock.fileno())
    assert code == 124
