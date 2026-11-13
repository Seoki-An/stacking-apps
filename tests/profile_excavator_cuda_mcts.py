"""Profile CPU or CUDA search stages during the existing MCTS benchmark.

Example (all remaining arguments go to benchmark_excavator_cuda_mcts.py):
  python tests/profile_excavator_cuda_mcts.py --profile-output /tmp/stages.json \
    --backend cuda --seed 0 --evaluation-batch-size 128 --cuda-batch-size 128 \
    --state /tmp/state9.pkl --output /tmp/search.json

Timers run only inside search. Inclusive times overlap; exclusive times subtract
instrumented child calls. --cprofile-stage adds diagnostic overhead to that stage
and writes a sibling .prof file; use a timer-only run for wall-time attribution.
"""
from collections import defaultdict
import argparse
import cProfile
from functools import wraps
import json
from pathlib import Path
import sys
import threading
import time

import benchmark_excavator_cuda_mcts as benchmark
from stacking_common.domain import State, StackObjectTrajectory
from stacking_common.environment.core_backend import CorePlacementBackend
from stacking_common.environment.cuda_backend import CudaPlacementBackend
from stacking_common.environment.environment import StackingEnvironment
from stacking_common.environment.observation import ArrayObservationBuilder
from stacking_common.environment.overlap import VoxelOverlapEvaluator
from stacking_common.environment.reward import StackingReward
from stacking_common.environment.transition import SimulationTransition
from stacking_common.search.adapters import StackingEnvironmentEvaluator, ParallelTransitionEvaluator
from stacking_common.actions import _scored_floor
from stacking_common.actions.core_solver import CorePlacementPoseSolver
from stacking_common.actions.cuda_solver import CudaPlacementPoseSolver
from stacking_common.environment import robustness
from stacking_common.search.mcts.search import MonteCarloTreeSearch


class StageTimers:
    def __init__(self, profile_stage=None):
        self.local = threading.local()
        self.active = False
        self.totals = defaultdict(lambda: dict(calls=0, inclusive_seconds=0.,
            exclusive_seconds=0., thread_cpu_seconds=0., max_seconds=0., jobs=0))
        self.profile_stage = profile_stage
        self.profile = cProfile.Profile() if profile_stage else None

    def instrument(self, owner, name, label):
        method = getattr(owner, name)

        @wraps(method)
        def timed(*args, **kwargs):
            if not self.active:
                return method(*args, **kwargs)
            stack = getattr(self.local, 'stack', None)
            if stack is None:
                stack = self.local.stack = []
            frame = [0.]
            stack.append(frame)
            started = time.perf_counter()
            cpu_started = time.thread_time()
            profiling = label == self.profile_stage
            if profiling:
                self.profile.enable()
            try:
                return method(*args, **kwargs)
            finally:
                if profiling:
                    self.profile.disable()
                elapsed = time.perf_counter() - started
                cpu = time.thread_time() - cpu_started
                stack.pop()
                if stack:
                    stack[-1][0] += elapsed
                row = self.totals[label]
                row['calls'] += 1
                if label == 'cuda_posegen_native':
                    row['jobs'] += len(args[1])
                elif label == 'cuda_posegen_batch':
                    row['jobs'] += len(args[2])
                elif label == 'cpu_posegen_worker':
                    row['jobs'] += 1
                row['inclusive_seconds'] += elapsed
                row['exclusive_seconds'] += elapsed - frame[0]
                row['thread_cpu_seconds'] += cpu
                row['max_seconds'] = max(row['max_seconds'], elapsed)
        setattr(owner, name, timed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--profile-output', type=Path, required=True)
    parser.add_argument('--cprofile-stage')
    args, remaining = parser.parse_known_args()
    if args.profile_output.exists() or args.profile_output.with_suffix('.prof').exists():
        raise FileExistsError(args.profile_output)
    timers = StageTimers(args.cprofile_stage)
    targets = [
        (MonteCarloTreeSearch, 'search', 'search'),
        (_scored_floor, 'score_xy_debug_map', 'floor_score_maps'),
        (_scored_floor, 'build_state_map_bundle', 'floor_state_maps'),
        (MonteCarloTreeSearch, '_expand_many', 'pose_generation_and_priors'),
        (StackingEnvironmentEvaluator, 'evaluate_batch', 'transition_batch'),
        (ParallelTransitionEvaluator, 'evaluate_batch', 'cpu_transition_batch_wall'),
        (CorePlacementPoseSolver, 'solve', 'cpu_posegen_worker'),
        (CudaPlacementPoseSolver, 'solve_batch', 'cuda_posegen_batch'),
        (CorePlacementBackend, 'place', 'cpu_placement_worker'),
        (CorePlacementBackend, '_settle', 'cpu_settling_worker'),
        (CorePlacementBackend, '_simulate_structural', 'cpu_structural_worker'),
        (robustness, '_simulate_disturbance', 'cpu_robustness_trial_worker'),
        (StackingEnvironmentEvaluator, 'evaluate', 'transition_single'),
        (StackingEnvironment, 'step_batch', 'environment_batch'),
        (StackingEnvironment, '_step_results', 'reward_observation_batch_wall'),
        (StackingEnvironment, '_prepare_geometries', 'reward_geometry_batch_wall'),
        (SimulationTransition, 'step_batch', 'transition_apply_batch'),
        (SimulationTransition, '_result_state', 'state_update'),
        (SimulationTransition, 'restore', 'state_restore'),
        (SimulationTransition, 'snapshot', 'state_snapshot'),
        (State, 'copy', 'state_copy'),
        (State, '__post_init__', 'state_constructor'),
        (StackObjectTrajectory, 'copy', 'trajectory_copy'),
        (CudaPlacementBackend, 'place_batch', 'cuda_placement'),
        (CudaPlacementBackend, '_place_chunk', 'cuda_chunk'),
        (CudaPlacementBackend, '_settling', 'settling_prepare_and_run'),
        (CudaPlacementBackend, '_run', 'native_simulation_run'),
        (CorePlacementBackend, '_scene', 'scene_build'),
        (CorePlacementBackend, '_scene_from_snapshot', 'scene_from_snapshot'),
        (CorePlacementBackend, '_structural_frozen_indices', 'support_freezing'),
        (CudaPlacementBackend, '_structural_result', 'structural_sample_processing'),
        (CudaPlacementBackend, '_structural_summary', 'structural_metadata'),
        (CudaPlacementBackend, '_populate_trajectories', 'structural_trajectory_processing'),
        (CudaPlacementBackend, '_settle_trajectory', 'settling_sample_processing'),
        (CudaPlacementBackend, '_robustness', 'robustness_total'),
        (CudaPlacementBackend, '_robustness_outcome', 'robustness_outcome'),
        (CorePlacementBackend, '_finish_placement', 'physical_validation_and_diagnostics'),
        (StackingReward, 'compute', 'reward'),
        (ArrayObservationBuilder, 'build', 'observation'),
        (VoxelOverlapEvaluator, '_intersection_volume', 'voxel_overlap'),
    ]
    if hasattr(benchmark.core, 'CudaPoseGenerator'):
        targets.append((benchmark.core.CudaPoseGenerator, 'solve', 'cuda_posegen_native'))
    if args.cprofile_stage and args.cprofile_stage not in {t[2] for t in targets}:
        parser.error('unknown cProfile stage')
    for owner, name, label in targets:
        timers.instrument(owner, name, label)
    measured_search = MonteCarloTreeSearch.search

    @wraps(measured_search)
    def search(self, *search_args, **search_kwargs):
        if timers.active:
            return measured_search(self, *search_args, **search_kwargs)
        timers.active = True
        try:
            return measured_search(self, *search_args, **search_kwargs)
        finally:
            timers.active = False

    MonteCarloTreeSearch.search = search
    sys.argv = [str(Path(benchmark.__file__)), *remaining]
    benchmark.main()
    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        timers=dict(sorted(timers.totals.items())),
        cprofile_stage=args.cprofile_stage,
        profiler_sha256=benchmark.digest(__file__),
        benchmark_arguments=remaining,
        note='Search-only wall timers; inclusive values overlap. CPU worker times sum overlapping threads and are not search wall time. Exclusive values subtract instrumented child calls on the same thread; native calls include immediate return handling, not pure kernel time.',
    )
    args.profile_output.write_text(json.dumps(report, indent=2) + '\n')
    if timers.profile:
        timers.profile.dump_stats(str(args.profile_output.with_suffix('.prof')))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
