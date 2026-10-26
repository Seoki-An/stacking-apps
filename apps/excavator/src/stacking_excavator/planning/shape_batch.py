"""Plan full-catalog target-shape examples through the Operations Console CLI.

    python -m stacking_excavator.planning.shape_batch --output .analysis/target-shapes --prepare-only
    python -m stacking_excavator.planning.shape_batch --output .analysis/target-shapes

Runs shapes sequentially, preserving every attempt and resuming the best saved
checkpoint on subsequent invocations. No robot motion or hardware is involved.
"""

import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import yaml

from stacking_common.artifacts import StackRunStore
from stacking_common.profiles.inventory import load_inventory_profile
from stacking_common.profiles.targets import load_target_registry
from stacking_common.domain import CompositeTargetGeometry
from stacking_common.environment import VoxelOverlapEvaluator
from stacking_common.workflows.search import require_search_target
from stacking_common.profiles.native import load_native_solver_profile
from stacking_common.workflows.process import stop_planning_process
from stacking_excavator import ExcavatorApplication
from .resume import load_resume_checkpoint


ROOT = Path(__file__).resolve().parents[5]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, values):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2) + "\n")
    temporary.replace(path)


def prepare_batch(repository, output, *, config=None, shapes_file=None, shapes=None, seed=0):
    """Snapshot platform profiles without editing the working configuration."""
    repository, output = Path(repository).resolve(), Path(output).resolve()
    source = repository / "apps/excavator/config"
    config = Path(config or source / "mcts_cuda.yml").resolve()
    shapes_file = Path(shapes_file or source / "target_shapes.yml").resolve()
    targets = yaml.safe_load(shapes_file.read_text())
    if "shapes" in targets:
        load_target_registry(shapes_file)
        targets = targets["shapes"]
    names = list(targets) if shapes is None else list(shapes)
    if not names or len(set(names)) != len(names) or any(name not in targets for name in names):
        raise ValueError("choose unique shape names present in the shapes YAML")
    if any(not name.replace("_", "").isalnum() for name in names):
        raise ValueError("shape names must contain only letters, numbers, or underscores")
    original = load_inventory_profile(source / "inventory.yml")
    if len(original.object_ids) != 49:
        raise ValueError("these examples require all 49 catalog stones")
    if (output / "manifest.json").exists():
        raise FileExistsError("batch already prepared")
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "source_config"
    shutil.copytree(source, snapshot)
    (snapshot / "mcts_batch.yml").write_bytes(config.read_bytes())
    inventory_values = yaml.safe_load((snapshot / "inventory.yml").read_text())
    jobs = []
    for name in names:
        root = output / name / "repository"
        profiles = root / "apps/excavator/config"
        shutil.copytree(snapshot, profiles)
        assets = root / "apps/excavator/assets"
        assets.symlink_to(repository / "apps/excavator/assets", target_is_directory=True)
        values = {key: value for key, value in inventory_values.items() if key != "target"}
        (profiles / "inventory.yml").write_text(yaml.safe_dump(values, sort_keys=False))
        (profiles / "target_shapes.yml").write_text(yaml.safe_dump(
            {"units": "si", "default": name, "shapes": targets}, sort_keys=False))
        profile = load_inventory_profile(profiles / "inventory.yml")
        # Resolve every platform profile and all 49 native object models now.
        app = ExcavatorApplication.from_repository(root, mcts_config=profiles / "mcts_batch.yml")
        app = replace(app, pose_from_scan=False)
        require_search_target(profile.target.geometry(),
            ground_z=float(app.posegen_profile.objective.get("ground_height", 0.)))
        inventory = app.load_inventory(profile.object_ids,
            {i: np.array([0., 0., 0., 0., 0., 0., 1.]) for i in profile.object_ids})
        load_native_solver_profile(app.solver_profiles.grasp, app.solver_profiles.motion)
        composite = isinstance(inventory.target, CompositeTargetGeometry)
        volume = (VoxelOverlapEvaluator(inventory.target, app.environment_profile.overlap_resolution).target_volume
                  if composite else inventory.target.volume)
        jobs.append(dict(name=name, target=targets[name],
            target_volume_m3=volume, target_volume_method="voxel_union" if composite else "analytic",
            stone_hull_volume_m3=sum(stone.volume for stone in inventory),
            profile_hashes={str(path.relative_to(output)): digest(path)
                            for path in sorted(profiles.rglob("*.yml"))}))
        (output / name / "runs").mkdir()
    manifest = dict(version=1, repository=str(repository), object_ids=list(original.object_ids),
        seed=seed, mode="search", pose_from_scan=False,
        source_config_sha256=digest(config), jobs=jobs)
    write_json(output / "manifest.json", manifest)
    return manifest


def inspect_run(path, object_ids):
    """Check real committed state, rather than treating exit zero as all placed."""
    info = StackRunStore.info(path)
    if info.metadata.get("platform") != "excavator" or info.metadata.get("mode") != "search":
        raise ValueError(f"not an excavator placement-only run: {path}")
    try:
        checkpoint = load_resume_checkpoint(path)
    except ValueError as error:
        no_committed_step = str(error) == "run has no committed planning step"
        if not no_committed_step and "planning_checkpoint/" not in str(error) and "without resume checkpoints" not in str(error):
            raise
        return dict(path=str(path), status=info.status, placed=0, step=0, complete=False)
    state = checkpoint.state
    if tuple(state.object_ids) != tuple(object_ids):
        raise ValueError(f"run inventory differs from the frozen 49-stone catalog: {path}")
    placed = [state.object_ids[i] for i in state.placed_object_indices]
    if len(set(placed)) != len(placed):
        raise ValueError(f"run contains duplicate placed stones: {path}")
    return dict(path=str(path), status=info.status, placed=len(placed), step=checkpoint.step,
                complete=set(placed) == set(object_ids))


def shape_progress(output, job, object_ids):
    attempts = [inspect_run(path, object_ids) for path in sorted(
        (output / job["name"] / "runs").glob("*.stack"))]
    best = max(attempts, key=lambda item: (item["placed"], item["path"]), default=None)
    return dict(shape=job["name"], placed=0 if best is None else best["placed"],
                complete=best is not None and best["complete"], best_run=best, attempts=attempts)


def run_worker(command, destination, log_path, seconds, *, lock_fd):
    """Bound native work, retain committed records, and reap its whole group."""
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # The snapshot contains configuration/assets, while source stays in this checkout.
    sources = [ROOT / "packages/stacking-common/src", ROOT / "apps/excavator/src"]
    environment["PYTHONPATH"] = os.pathsep.join([*map(str, sources), environment.get("PYTHONPATH", "")])
    with log_path.open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=environment, pass_fds=(lock_fd,))
        started, heartbeat = time.monotonic(), time.monotonic()
        code = None
        try:
            while process.poll() is None:
                now = time.monotonic()
                if now-started >= seconds:
                    code = 124
                    break
                if now-heartbeat >= 30:
                    print(f"  planning {destination.name}: {now-started:.0f}s; log: {log_path}", flush=True)
                    heartbeat = now
                time.sleep(.2)
        except KeyboardInterrupt:
            code = 130
        finally:
            if code is not None:
                previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
                previous_term = signal.signal(signal.SIGTERM, signal.SIG_IGN)
                try:
                    stop_planning_process(process)
                finally:
                    signal.signal(signal.SIGINT, previous)
                    signal.signal(signal.SIGTERM, previous_term)
            else:
                code = process.wait()
    if destination.is_file() and (code != 0 or StackRunStore.info(destination).status == "running"):
        with StackRunStore.open_append(destination) as writer:
            writer.finish("cancelled" if code in (124, 130) else "failed")
    return code


def run_shape(output, job, manifest, *, seconds, attempts, lock_fd):
    for relative, expected in job["profile_hashes"].items():
        if digest(output / relative) != expected:
            raise ValueError(f"frozen profile changed: {relative}; prepare a new batch")
    directory = output / job["name"]
    progress = shape_progress(output, job, manifest["object_ids"])
    # Holding the inherited batch lock proves no previous batch worker is alive.
    for record in progress["attempts"]:
        if record["status"] == "running":
            with StackRunStore.open_append(record["path"]) as writer:
                writer.finish("cancelled")
    progress = shape_progress(output, job, manifest["object_ids"])
    deadline = time.monotonic()+seconds
    for _ in range(attempts):
        if progress["complete"] or time.monotonic() >= deadline:
            break
        number = max([int(p.name.split("-")[1].split(".")[0])
                      for p in (directory / "runs").glob("attempt-*.command.json")], default=0)+1
        destination = directory / "runs" / f"attempt-{number:03d}.stack"
        log = destination.with_suffix(".log")
        command = [sys.executable, "-u", "-m", "stacking_excavator.planning.cli", "--worker",
            "--repository", str(directory / "repository"),
            "--config", str(directory / "repository/apps/excavator/config/mcts_batch.yml"),
            "--save-run", str(destination), "--search-only", "--no-pose-from-scan",
            "--steps", str(len(manifest["object_ids"])-progress["placed"])]
        if progress["placed"]:
            command += ["--resume-run", progress["best_run"]["path"]]
        else:
            command += ["--seed", str(manifest["seed"])]
        print(f"{job['name']}: {progress['placed']}/49 placed; starting {destination.name}", flush=True)
        write_json(destination.with_suffix(".command.json"), dict(argv=command))
        code = run_worker(command, destination, log, deadline-time.monotonic(), lock_fd=lock_fd)
        previous_count = progress["placed"]
        progress = shape_progress(output, job, manifest["object_ids"])
        progress.update(last_exit_code=code, last_log=str(log))
        write_json(directory / "summary.json", progress)
        print(f"{job['name']}: {progress['placed']}/49 placed; exit={code}; complete={progress['complete']}", flush=True)
        if code == 130:
            raise KeyboardInterrupt
        if progress["placed"] <= previous_count:
            break  # Repeating an identical no-progress search cannot help.
    write_json(directory / "summary.json", progress)
    return progress


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="MCTS profile for a new batch; default: excavator CUDA")
    parser.add_argument("--shapes-file", type=Path, help="target YAML for a new batch")
    parser.add_argument("--shapes", nargs="+", help="shape names to prepare/run; default: all")
    parser.add_argument("--seed", type=int, help="new batch seed (default 0); preserved on resume")
    parser.add_argument("--max-run-time", type=float, default=21600., help="total seconds per shape per invocation (default 6 hours)")
    parser.add_argument("--attempts", type=int, default=3, help="maximum advancing attempts per shape per invocation")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="freeze and validate profiles without running MCTS")
    mode.add_argument("--status", action="store_true", help="read current checkpoint counts without starting workers")
    args = parser.parse_args(argv)
    if not np.isfinite(args.max_run_time) or args.max_run_time <= 0 or args.attempts <= 0:
        parser.error("run time and attempts must be positive and finite")
    if args.seed is not None and args.seed < 0:
        parser.error("seed must be nonnegative")
    output = args.output.resolve()
    if args.status and not (output / "manifest.json").is_file():
        parser.error("no prepared batch at this output path")
    output.mkdir(parents=True, exist_ok=True)
    try:
        with (output / ".batch.lock").open("a+") as lock:
            if not args.status:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError("another planner still holds this batch lock") from None
            if (output / "manifest.json").exists():
                manifest = json.loads((output / "manifest.json").read_text())
                if args.shapes_file is not None or (args.config is not None and digest(args.config) != manifest["source_config_sha256"]):
                    raise ValueError("existing batch inputs are frozen; choose a new output directory")
                if args.seed is not None and args.seed != manifest["seed"]:
                    raise ValueError("resume preserves the batch seed")
            else:
                manifest = prepare_batch(args.repository, output, config=args.config,
                    shapes_file=args.shapes_file, shapes=args.shapes, seed=args.seed or 0)
            names = set(args.shapes) if args.shapes else {job["name"] for job in manifest["jobs"]}
            if not names.issubset({job["name"] for job in manifest["jobs"]}):
                raise ValueError("requested shape is not in the prepared batch")
            jobs = [job for job in manifest["jobs"] if job["name"] in names]
            if args.status or args.prepare_only:
                for job in jobs:
                    progress = shape_progress(output, job, manifest["object_ids"])
                    print(f"{job['name']}: {progress['placed']}/49 placed; target volume={job['target_volume_m3']:.2f} m^3")
                    if progress["best_run"] is not None:
                        print(f"  open: {progress['best_run']['path']}")
                return 0
            def interrupt(signum, frame):
                raise KeyboardInterrupt

            previous = signal.signal(signal.SIGTERM, interrupt)
            try:
                results = [run_shape(output, job, manifest, seconds=args.max_run_time,
                    attempts=args.attempts, lock_fd=lock.fileno()) for job in jobs]
            finally:
                signal.signal(signal.SIGTERM, previous)
            write_json(output / "summary.json", dict(shapes=results, complete=all(r["complete"] for r in results)))
            return 0 if all(r["complete"] for r in results) else 1
    except KeyboardInterrupt:
        print("Batch stopped; committed checkpoints retained. Rerun the same command to resume.", flush=True)
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"batch error: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
