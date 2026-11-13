"""Prepare a CPU/CUDA MCTS throughput comparison; run only with the run subcommand.

Preparation copies recorded states and writes shared iteration-only profiles.
It does not construct an inventory, start search, or allocate CUDA workspaces.
"""

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import re
import statistics
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "tests/benchmark_excavator_cuda_mcts.py"
for directory in ("packages/stacking-common/src", "apps/excavator/src"):
    sys.path.insert(0, str(ROOT / directory))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def matched_config(base, batch_size, iterations):
    values = deepcopy(base)
    values["tree"].update(iterations=iterations, max_search_time=None)
    values["generation"].update(batch_size=batch_size, evaluation_batch_size=batch_size)
    return values


def trial_command(config, state, backend, seed, batch_size, workers, noise_trials, output):
    command = [sys.executable, str(BENCHMARK), "--backend", backend,
               "--config", str(config), "--target-shape", "default",
               "--seed", str(seed), "--workers", str(workers),
               "--early-stops", "excursion", "--output", str(output)]
    if state is not None:
        command.extend(("--state", str(state)))
    if backend == "cuda":
        command.extend(("--posegen-batch-size", str(batch_size),
                        "--placement-batch-size", str(batch_size),
                        "--cuda-batch-size", str(max(batch_size, min(512, batch_size * noise_trials)))))
    return command


def export_scene(spec, directory, inventory):
    from stacking_common.artifacts import StackRunStore
    from stacking_common.artifacts.planning_context import _values

    name, separator, source = spec.partition("=")
    run, colon, step = source.rpartition(":")
    if (not separator or not colon or not step.isdigit() or name == "empty"
            or not re.fullmatch(r"[A-Za-z0-9_-]+", name)):
        raise ValueError("scene must be NAME=RUN.stack:STEP; NAME uses letters, numbers, _ or -")
    source = Path(run).resolve()
    checkpoint = StackRunStore.load_planning_checkpoint(source, step=int(step))
    context = StackRunStore.load_run_context(source, checkpoint.context_sha256)
    previous = _values(context.profiles["inventory_profile"])
    current = _values(inventory)
    for key in ("target", "density", "density_overrides"):
        if previous[key] != current[key]:
            raise ValueError(f"{name}: recorded {key} differs from the current default inventory")
    if list(checkpoint.state.object_ids) != current["object_ids"]:
        raise ValueError(f"{name}: recorded inventory/order differs from the current catalog")
    for object_id in inventory.object_ids:
        for role, suffix in (("object", ".obj"), ("object_surface", "_mesh.ply")):
            path = ROOT / inventory.model_directory / f"model_{object_id}{suffix}"
            if context.asset_hashes.get(f"{role}/{object_id}") != digest(path):
                raise ValueError(f"{name}: recorded {role}/{object_id} differs from current assets")
    path = directory / f"{name}.pkl"
    with path.open("xb") as stream:
        pickle.dump(checkpoint.state, stream)
    return name, dict(state=str(path), placed=len(checkpoint.state.placed_object_indices),
                      source_run=str(source), source_run_sha256=digest(source), step=int(step),
                      source_context_sha256=context.sha256)


def prepare(args):
    import stacking_core as core
    from stacking_common.artifacts.planning_context import _values
    from stacking_common.profiles.mcts import load_mcts_profile
    from stacking_excavator import ExcavatorApplication

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    app = ExcavatorApplication.from_repository(ROOT, target_shape="default")
    base = yaml.safe_load((ROOT / "apps/excavator/config/mcts_cpu.yml").read_text())
    output.mkdir(parents=True)
    for name in ("configs", "states", "runs"):
        (output / name).mkdir()
    scenes = {"empty": dict(state=None, placed=0)}
    for spec in args.scene:
        name = spec.partition("=")[0]
        if name in scenes:
            raise ValueError(f"duplicate scene: {name}")
        name, scene = export_scene(spec, output / "states", app.inventory_profile)
        scenes[name] = scene
    configs = {}
    for batch in args.batch_sizes:
        path = output / "configs" / f"batch-{batch}.yml"
        path.write_text(yaml.safe_dump(matched_config(base, batch, args.iterations), sort_keys=False))
        load_mcts_profile(path)
        configs[batch] = path
    trials = []
    noise_trials = app.environment_profile.place_stability.n_noise
    for batch in args.batch_sizes:
        for name, scene in scenes.items():
            for index, seed in enumerate(args.seeds):
                order = ("cpu", "cuda") if index % 2 == 0 else ("cuda", "cpu")
                for backend in order:
                    path = output / "runs" / f"{name}-batch{batch}-seed{seed}-{backend}.json"
                    workers = args.cpu_workers if backend == "cpu" else args.gpu_workers
                    trials.append(dict(scene=name, batch_size=batch, backend=backend, seed=seed,
                                       output=str(path), command=trial_command(
                                           configs[batch], scene["state"], backend, seed,
                                           batch, workers, noise_trials, path)))
    inputs = {BENCHMARK, Path(__file__).resolve(), Path(core._native.__file__).resolve()}
    for directory, pattern in ((ROOT / "packages/stacking-common/src", "*.py"),
                               (ROOT / "apps/excavator/src", "*.py"),
                               (ROOT / "apps/excavator/config", "*.yml"),
                               (Path(core.__file__).parent, "*.py"),
                               (output / "configs", "*.yml"), (output / "states", "*.pkl")):
        inputs.update(directory.rglob(pattern))
    for object_id in app.inventory_profile.object_ids:
        for suffix in (".obj", "_mesh.ply"):
            inputs.add(ROOT / app.inventory_profile.model_directory / f"model_{object_id}{suffix}")
    paper = ROOT.parent / "stacking-paper/root.pdf"
    protocol = dict(
        status="prepared", repository=str(ROOT), python=sys.executable,
        iterations=args.iterations, max_time_seconds=None, target_shape="default",
        inventory_profile=_values(app.inventory_profile),
        physical_profiles={name: _values(getattr(app, name)) for name in (
            "posegen_profile", "simulation_profile", "environment_profile", "action_generation_profile")},
        cpu_workers=args.cpu_workers, gpu_result_workers=args.gpu_workers,
        seeds=args.seeds, batch_sizes=args.batch_sizes, scenes=scenes, trials=trials,
        paper=dict(path=str(paper), sha256=digest(paper)) if paper.is_file() else None,
        metric="generated_candidates / elapsed_seconds; new tree children including physical failures",
        timing="search.search only, including first-search lazy work; setup and serialization excluded",
        input_sha256={str(path.resolve()): digest(path) for path in sorted(inputs)},
    )
    write_json(output / "protocol.json", protocol)
    print(f"Prepared {len(trials)} trials in {output}; no searches executed.")
    print(f"Run later: {sys.executable} {Path(__file__).resolve()} run --output {output}")


def verify_inputs(protocol):
    for path, expected in protocol["input_sha256"].items():
        if not Path(path).is_file() or digest(path) != expected:
            raise ValueError(f"benchmark input changed; prepare a new comparison: {path}")


def comparable_profiles(result):
    profiles = deepcopy(result["profiles"])
    parallel = profiles["mcts_profile"]["parallel"]
    for key in ("posegen_backend", "simulation_backend", "workers"):
        parallel.pop(key, None)
    return profiles


def summarize(protocol):
    groups = defaultdict(list)
    pairs = defaultdict(dict)
    for trial in protocol["trials"]:
        path = Path(trial["output"])
        if not path.is_file():
            raise ValueError(f"missing trial result: {path}")
        result = json.loads(path.read_text())
        if result["max_time_seconds"] is not None or result["timed_out"]:
            raise ValueError(f"trial used a time budget: {path}")
        if result["requested_iterations"] != protocol["iterations"]:
            raise ValueError(f"iteration budget differs: {path}")
        if result.get("backend_failure"):
            raise ValueError(f"backend failed: {path}: {result['backend_failure']}")
        groups[trial["scene"], trial["batch_size"], trial["backend"]].append(result)
        pairs[trial["scene"], trial["batch_size"], trial["seed"]][trial["backend"]] = result
    for pair in pairs.values():
        cpu, gpu = pair["cpu"], pair["cuda"]
        for key in ("state_sha256", "source_sha256", "core_native_sha256", "inventory_profile"):
            if cpu[key] != gpu[key]:
                raise ValueError(f"CPU/CUDA {key} differs")
        if comparable_profiles(cpu) != comparable_profiles(gpu):
            raise ValueError("CPU/CUDA search or physical profiles differ")
    rows = []
    for (scene, batch, backend), results in groups.items():
        rates = [r["expanded_nodes_per_second"] for r in results]
        sizes = [size for r in results for size in r["evaluation_batch_sizes"] if size]
        rows.append(dict(scene=scene, batch_size=batch, backend=backend, trials=len(results),
                         mean_nodes_per_second=statistics.mean(rates),
                         stdev_nodes_per_second=statistics.stdev(rates) if len(rates) > 1 else 0.,
                         pooled_nodes_per_second=sum(r["generated_candidates"] for r in results)
                             / sum(r["elapsed_seconds"] for r in results),
                         mean_actual_evaluation_batch=statistics.mean(sizes) if sizes else 0.,
                         max_actual_evaluation_batch=max(sizes, default=0),
                         generated=[r["generated_candidates"] for r in results],
                         evaluated=[r["evaluated_candidates"] for r in results],
                         failed=[r["failed_evaluations"] for r in results],
                         completed_iterations=[r["completed_iterations"] for r in results],
                         selected=[r["selected"] for r in results]))
    speedups = []
    for scene, batch in dict.fromkeys((row["scene"], row["batch_size"]) for row in rows):
        rates = {r["backend"]: r["mean_nodes_per_second"] for r in rows
                 if (r["scene"], r["batch_size"]) == (scene, batch)}
        speedups.append(dict(scene=scene, batch_size=batch,
                             gpu_over_cpu=rates["cuda"] / rates["cpu"] if rates["cpu"] else None))
    return dict(groups=rows, speedups=speedups)


def run(output):
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol["python"] != sys.executable:
        raise ValueError("use the Python interpreter recorded in protocol.json")
    verify_inputs(protocol)
    if any(Path(trial["output"]).exists() or Path(trial["output"]).with_suffix(".log").exists()
           for trial in protocol["trials"]):
        raise FileExistsError("trial outputs exist; prepare a new comparison")
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", QT_QPA_PLATFORM="offscreen",
                       MPLCONFIGDIR=str(output / "mpl-cache"))
    write_json(output / "host.json", dict(platform=platform.platform(), cpu=platform.processor(),
                                         logical_cpus=os.cpu_count()))
    for trial in protocol["trials"]:
        verify_inputs(protocol)
        print(f"START {trial['scene']} batch={trial['batch_size']} seed={trial['seed']} {trial['backend']}", flush=True)
        with Path(trial["output"]).with_suffix(".log").open("x") as log:
            subprocess.run(trial["command"], cwd=ROOT, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    write_json(output / "summary.json", summarize(protocol))


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    preparation = commands.add_parser("prepare", help="write inputs and protocol without running searches")
    preparation.add_argument("--output", type=Path, required=True)
    preparation.add_argument("--scene", action="append", default=[], metavar="NAME=RUN.stack:STEP")
    preparation.add_argument("--batch-sizes", type=positive, nargs="+", default=[32])
    preparation.add_argument("--iterations", type=positive, default=4096)
    preparation.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    preparation.add_argument("--cpu-workers", type=positive, default=12)
    preparation.add_argument("--gpu-workers", type=positive, default=2)
    for name in ("run", "summarize"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        if len(set(args.seeds)) != len(args.seeds) or len(set(args.batch_sizes)) != len(args.batch_sizes):
            parser.error("seeds and batch sizes must be unique")
        prepare(args)
    elif args.action == "run":
        run(args.output.resolve())
    else:
        write_json(args.output / "summary.json", summarize(json.loads((args.output / "protocol.json").read_text())))


if __name__ == "__main__":
    main()
