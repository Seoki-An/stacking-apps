"""Profile one native, empty-stack excavator search without hardware execution.

Use the project Python. Timers include worker threads; their nested wall times
overlap and must not be added together. Optional cProfile covers the main thread
and adds overhead, so use the default timer-only run for wall-time attribution.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import cProfile
from functools import wraps
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

from benchmark_excavator_mcts import build
from stacking_common.actions import scored
from stacking_common.domain import StackObject
from stacking_common.domain.geometry import transform_points
from stacking_common.environment import overlap
from stacking_common.search import MCTSNode
from stacking_common.workflows.mcts_reporting import search_statistics


class Timers:
    def __init__(self):
        self.events = []

    def wrap(self, method, label):
        @wraps(method)
        def timed(*args, **kwargs):
            started = time.perf_counter()
            cpu = time.thread_time()
            try:
                return method(*args, **kwargs)
            finally:
                self.events.append({
                    'label': label,
                    'start': started,
                    'seconds': time.perf_counter() - started,
                    'thread_cpu_seconds': time.thread_time() - cpu,
                    'thread': threading.get_ident(),
                    'items': len(args[0]) if label == 'candidate_batch' else None,
                })
        return timed

    def instrument(self, owner, name, label):
        setattr(owner, name, self.wrap(getattr(owner, name), label))

    def summary(self):
        groups = defaultdict(list)
        for event in self.events:
            groups[event['label']].append(event)
        return {label: {
            'calls': len(events),
            'wall_seconds_sum': sum(e['seconds'] for e in events),
            'thread_cpu_seconds_sum': sum(e['thread_cpu_seconds'] for e in events),
            'max_call_seconds': max(e['seconds'] for e in events),
            'threads': len({e['thread'] for e in events}),
        } for label, events in sorted(groups.items())}


class TimedNativeGenerator:
    def __init__(self, generator, timers):
        self.generator = generator
        self.solve = timers.wrap(generator.solve, 'native_pose_solve')

    @property
    def config(self):
        return self.generator.config

    @config.setter
    def config(self, value):
        self.generator.config = value


def cropped_intersection(evaluator, stack_object, pose):
    """Diagnostic broad phase; not a production geometry implementation.

    Keep the original target voxel centers and pad the object bounds by a full
    voxel before applying the existing narrow-phase containment predicate.
    """
    points = transform_points(stack_object.mesh_arrays()[0], pose)
    lower = points.min(axis=0) - evaluator.resolution
    upper = points.max(axis=0) + evaluator.resolution
    grid = evaluator._grid_points
    mask = np.ones(len(grid), dtype=bool)
    for axis in range(3):
        mask &= (grid[:, axis] >= lower[axis]) & (grid[:, axis] <= upper[axis])
    local_points = overlap.inverse_transform_points(grid[mask], pose)
    inside = stack_object.contains_local_points(
        local_points, tolerance=evaluator.resolution * 1e-6)
    return float(np.count_nonzero(inside)) * evaluator.resolution**3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cprofile', action='store_true')
    parser.add_argument('--cache-prior', action='store_true',
                        help='diagnostic experiment: memoize containment within this search')
    parser.add_argument('--crop-prior', action='store_true',
                        help='diagnostic experiment: crop prior voxel tests to object bounds')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    build_args = argparse.Namespace(seed=0, mode='pmbs', prepare=False,
        force_depth3=False, iterations=200, seconds=240)
    application, environment, engine, request = build(build_args)
    try:
        reset = environment.reset()
        root = MCTSNode(state=reset.state, observation=reset.observation)
        timers = Timers()
        generator = engine.action_generator
        for method, label in (
            ('generate', 'root_cem'), ('_rank', 'cem_rank'),
            ('_deduplicate', 'cem_deduplicate'),
            ('_mutation_proposal', 'cem_mutation'),
        ):
            timers.instrument(engine.root_action_generator, method, label)
        timers.instrument(generator, '_build_jobs', 'candidate_batch')
        timers.instrument(generator, '_build_candidate_with_solver', 'candidate_build')
        timers.instrument(generator.initializer, 'initialize', 'pose_initialization')
        for placement_filter in generator.filters:
            timers.instrument(placement_filter, 'evaluate', type(placement_filter).__name__)
        timers.instrument(generator.planar_sampler, 'sample', 'planar_sample')
        timers.instrument(generator.planar_sampler, 'score_map', 'planar_score_map')
        timers.instrument(scored, 'bottom_height_kernel', 'bottom_height_kernel')
        timers.instrument(scored.floor, 'score_xy_debug_map', 'score_xy_debug_map')
        timers.instrument(engine.prior_scorer, 'score', 'prior_score')
        evaluator = engine.prior_scorer.overlap_evaluator
        if args.crop_prior:
            evaluator._intersection_volume = lambda obj, pose: cropped_intersection(
                evaluator, obj, pose)
        timers.instrument(engine.prior_scorer.overlap_evaluator,
                          'object_containment', 'prior_containment')
        containment_cache = {}
        cache_hits = 0
        if args.cache_prior:
            evaluator = engine.prior_scorer.overlap_evaluator
            original_containment = evaluator.object_containment

            def cached_containment(stack_object, pose):
                nonlocal cache_hits
                key = (stack_object.object_id, tuple(float(x) for x in pose))
                if key not in containment_cache:
                    containment_cache[key] = original_containment(stack_object, pose)
                else:
                    cache_hits += 1
                return containment_cache[key]

            evaluator.object_containment = cached_containment
        timers.instrument(overlap, 'inverse_transform_points', 'overlap_transform')
        timers.instrument(StackObject, 'contains_local_points', 'object_contains')
        for solver in generator.pose_solvers:
            timers.instrument(solver, 'solve', 'pose_solve_adapter')
            timers.instrument(solver, '_scheduled_config', 'pose_config')
            timers.instrument(solver.scene_builder, 'build', 'pose_scene_build')
            solver.generator = TimedNativeGenerator(solver.generator, timers)
        profile = cProfile.Profile() if args.cprofile else None
        started = time.perf_counter()
        cpu = time.process_time()
        if profile is not None:
            profile.enable()
        result = engine.search(root, request)
        if profile is not None:
            profile.disable()
        elapsed = time.perf_counter() - started
        row = {
            'search': search_statistics(result, elapsed, engine, step=1),
            'process_cpu_seconds': time.process_time() - cpu,
            'cprofile_enabled': args.cprofile,
            'experimental_prior_cache': args.cache_prior,
            'experimental_prior_crop': args.crop_prior,
            'prior_cache_hits': cache_hits,
            'prior_cache_entries': len(containment_cache),
            'timers': timers.summary(),
            'candidate_batches': [e for e in timers.events if e['label'] == 'candidate_batch'],
            'selected_object_id': result.selected.action.object_id if result.selected else None,
            'selected_pose': result.selected.action.pose.tolist() if result.selected else None,
            'overlap_grid_points': len(engine.prior_scorer.overlap_evaluator._grid_points),
            'thread_environment': {name: os.environ.get(name) for name in
                                   ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        }
        try:
            from threadpoolctl import threadpool_info
            row['thread_pools'] = threadpool_info()
        except ImportError:
            row['thread_pools'] = None
        (args.output / 'summary.json').write_text(json.dumps(row, indent=2) + '\n')
        (args.output / 'events.json').write_text(json.dumps(timers.events) + '\n')
        if args.cache_prior:
            (args.output / 'containment-reference.json').write_text(json.dumps([
                {'object_id': key[0], 'pose': key[1], 'containment': value}
                for key, value in containment_cache.items()
            ], indent=2) + '\n')
        if profile is not None:
            profile.dump_stats(str(args.output / 'main-thread.prof'))
        print(json.dumps(row, indent=2))
    finally:
        engine.close()
        environment.close()


if __name__ == '__main__':
    main()
