"""CPU-only wall sampling benchmark against a Git revision; no native solves.

Uses real excavator meshes and a saved ten-stone scene. Compares every score,
candidate pose and seeded draw before timing uncached scene maps with warm
orientation geometry. Run with the project Python, e.g.:

    OPENBLAS_NUM_THREADS=1 python tests/benchmark_two_sided_sampling.py \
        --baseline-ref 822435b --output /tmp/wall-sampling.json
"""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from stacking_common.actions.wall import WallPlanarPoseSampler
from stacking_common.domain import State
from stacking_excavator import ExcavatorApplication


ROOT = Path(__file__).resolve().parents[1]


def baseline_sampler(revision, directory):
    def load(relative, name):
        source = subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=ROOT)
        path = directory / (name.rsplit(".", 1)[-1] + ".py")
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    prefix = "packages/stacking-common/src/stacking_common/"
    geometry = load(prefix + "wall.py", "stacking_common._benchmark_wall_geometry")
    sampler = load(prefix + "actions/wall.py", "stacking_common.actions._benchmark_wall_sampler")
    sampler.WallSurfaces = geometry.WallSurfaces
    return sampler.WallPlanarPoseSampler


def make_sampler(app, inventory, cls):
    profile = app.action_generation_profile
    return cls(inventory, profile.planar, app.build_orientation_provider(),
               app.sampling_settings, construction=profile.construction,
               seed=app.mcts_profile.root_cem.seed, floor_policy=app.build_floor_policy(inventory))


def compare(before, after, inventory, state, observation, indices):
    for sampler in (before, after):
        sampler._wall_context = None
        sampler.rng.seed(0)
    for i in indices:
        old = before.score_map(state, observation, i)
        new = after.score_map(state, observation, i)
        assert len(old["candidates"]) == len(new["candidates"])
        for key in ("wall_quality", "wall_remaining"):
            np.testing.assert_allclose(old[key], new[key], rtol=1e-12, atol=1e-12)
        for a, b in zip(old["candidates"], new["candidates"]):
            for key in a:
                np.testing.assert_allclose(a[key], b[key], rtol=1e-12, atol=1e-12)
        a = before.sample(inventory, state, observation, i, 32, "lower_floor_first")
        b = after.sample(inventory, state, observation, i, 32, "lower_floor_first")
        np.testing.assert_allclose([p.pose for p in a], [p.pose for p in b], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose([p.score for p in a], [p.score for p in b], rtol=1e-12, atol=1e-12)


def measure(sampler, inventory, state, observation, indices):
    sampler._wall_context = None
    sampler.rng.seed(0)
    started = time.perf_counter()
    count = sum(len(sampler.sample(inventory, state, observation, i, 32, "lower_floor_first"))
                for i in indices)
    return time.perf_counter() - started, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="822435b")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    app = ExcavatorApplication.from_repository(ROOT)
    if not app.action_generation_profile.construction.enabled:
        parser.error("select two_sided_wall in excavator action_generation.yml")
    saved = json.loads((ROOT / "tests/fixtures/planning/excavator_floor_step11.json").read_text())
    ids = saved["object_ids"]
    inventory = app.load_inventory(ids, {i: np.array([0., 0., 0., 0., 0., 0., 1.]) for i in ids})
    placed = State(ids, [ids.index(o["object_id"]) for o in saved["placed"]],
                   {o["object_id"]: np.array(o["pose"]) for o in saved["placed"]}, [], [])
    builder = app._build_state_evaluators(inventory)[0]
    rows = []
    with tempfile.TemporaryDirectory(prefix="wall-benchmark-") as directory:
        old_class = baseline_sampler(args.baseline_ref, Path(directory))
        for name, state in (("empty", State(ids, [], {}, [], [])), ("ten_placed", placed)):
            observation = builder.build(state)
            indices = [i for i in range(len(ids)) if i not in state.placed_object_indices]
            samplers = [make_sampler(app, inventory, cls) for cls in (old_class, WallPlanarPoseSampler)]
            cold = [measure(s, inventory, state, observation, indices)[0] for s in samplers]
            compare(*samplers, inventory, state, observation, indices)
            times = [[], []]
            for repeat in range(args.repeats):
                # Alternate order to reduce drift bias.
                for slot in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                    elapsed, count = measure(samplers[slot], inventory, state, observation, indices)
                    times[slot].append(elapsed)
            rows.append(dict(scene=name, stones=len(indices), proposals=count, parity="passed",
                             cold_baseline_seconds=cold[0], cold_current_seconds=cold[1],
                             baseline_seconds=times[0], current_seconds=times[1],
                             speedup=float(np.median(times[0]) / np.median(times[1]))))
    result = json.dumps(dict(baseline_ref=args.baseline_ref, cases=rows), indent=2)
    if args.output:
        args.output.write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
