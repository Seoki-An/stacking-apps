"""Time the surface reward on saved excavator states (no simulation).

Example: OPENBLAS_NUM_THREADS=1 python tests/benchmark_surface_flatness.py \
    --scene .stacks/run.stack:4 --scene .stacks/run.stack:30

Uses the current excavator assets/target/reward profile. Perturbed poses are
synthetic cache workloads, not claims about physically valid placements.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

import numpy as np

from stacking_common.artifacts.run_document import StackRunStore
from stacking_common.environment import StackingReward, VoxelOverlapEvaluator
from stacking_common.environment.surface import SurfaceFlatness
from stacking_excavator import ExcavatorApplication


def timed(call):
    start = time.perf_counter()
    call()
    return 1000 * (time.perf_counter() - start)


def benchmark(application, spec, repeats):
    path, step = spec.rsplit(':', 1)
    state = StackRunStore.load_planning_checkpoint(path, step=int(step)).state
    if not state.placed_object_indices:
        raise ValueError('benchmark requires at least one placed stone')
    inventory = application.load_inventory(state.object_ids,
        {oid: state.object_poses.get(oid, np.array([0., 0., 0., 0., 0., 0., 1.]))
         for oid in state.object_ids})
    surface = SurfaceFlatness(inventory)
    cold_ms = timed(lambda: surface.measure(state))
    warm_ms = [timed(lambda: surface.measure(state)) for _ in range(repeats)]
    parent = state.copy()
    parent.placed_object_indices.pop()
    surface.measure(parent)
    newest = state.object_ids[state.placed_object_indices[-1]]
    oldest = state.object_ids[state.placed_object_indices[0]]
    workloads = {}
    for mode in ('new_placement', 'previous_stone_moved', 'all_stones_moved'):
        times = []
        for i in range(repeats):
            branch = state.copy()
            delta = (i + 1) * 1e-5
            if mode == 'new_placement':
                branch.object_poses[newest][0] += delta
            elif mode == 'previous_stone_moved':
                branch.object_poses[oldest][1] += delta
            else:
                for index in branch.placed_object_indices:
                    branch.object_poses[state.object_ids[index]][2] += delta
            times.append(timed(lambda: surface.measure(branch)))
        workloads[mode] = statistics.median(times)

    profile = application.environment_profile
    overlap = VoxelOverlapEvaluator(inventory.target, profile.overlap_resolution)
    reward_times = {}
    for weight in (0., 1.):
        config = replace(profile.reward, weights=replace(profile.reward.weights, surface_flatness=weight))
        reward = StackingReward(inventory, overlap, config)
        reward.compute(state)
        reward_times[str(weight)] = statistics.median(
            timed(lambda: reward.compute(state)) for _ in range(repeats))
    return dict(scene=spec, placed=len(state.placed_object_indices),
                cold_surface_ms=cold_ms, cached_surface_ms=statistics.median(warm_ms),
                median_surface_ms=workloads, cached_reward_ms_by_weight=reward_times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', action='append', required=True, help='RUN.stack:STEP')
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    app = ExcavatorApplication.from_repository(Path(__file__).resolve().parents[1])
    for spec in args.scene:
        print(json.dumps(benchmark(app, spec, args.repeats)), flush=True)


if __name__ == '__main__':
    main()
