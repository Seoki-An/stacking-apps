"""Measure one CPU or combined-CUDA search in an isolated process.

Run each mode/seed sequentially with OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1.
--state accepts only a trusted, locally exported State pickle. Omit it for an
empty stack. Search budgets and platform profiles default to the current app.
"""
from dataclasses import replace
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

import stacking_core as core

ROOT = Path(__file__).resolve().parents[1]
for directory in ('packages/stacking-common/src', 'apps/excavator/src'):
    sys.path.insert(0, str(ROOT / directory))

from stacking_common.artifacts.planning_context import _values
from stacking_common.search import MCTSNode, EvaluationFidelity
from stacking_common.workflows.mcts_reporting import search_statistics
from stacking_excavator import ExcavatorApplication


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_digest(node, *, accepted_only=False):
    """Hash ordered decisions and backed-up values, excluding timing diagnostics."""
    def values(item):
        return dict(
            action=None if item.action is None else [item.action.object_id, item.action.pose.tolist()],
            poses=None if item.state is None or (accepted_only and item.failed) else {
                str(key): value.tolist() for key, value in sorted(item.state.object_poses.items())},
            prior=item.prior, reward=None if accepted_only and item.failed else item.reward,
            visits=item.visits, value_sum=item.value_sum,
            failed=item.failed, terminal=item.terminal,
            rejection_reason=None if accepted_only and item.failed else item.rejection_reason,
            children=[values(child) for child in item.children])
    return hashlib.sha256(json.dumps(values(node), sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--config', type=Path, help='Use the same profile for controlled CPU/CUDA comparisons.')
    parser.add_argument('--target-shape', help='Select a named target explicitly, independent of the registry default.')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--workers', type=int, help='CPU workers, including CUDA reward/observation workers.')
    parser.add_argument('--result-batch-size', type=int, help='CPU reward/observation chunk size, independent of CUDA placements.')
    parser.add_argument('--generation-batch-size', type=int)
    parser.add_argument('--posegen-batch-size', type=int)
    parser.add_argument('--state', type=Path)
    parser.add_argument('--early-stops', choices=('none', 'excursion', 'quiet', 'both'), default='none',
                        help='Process-local simulation shortcut experiment.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluation-batch-size', type=int)
    parser.add_argument('--cuda-batch-size', type=int)
    parser.add_argument('--placement-batch-size', type=int,
                        help='CUDA branches per streamed physics/result chunk.')
    parser.add_argument('--eager-observations', action='store_true',
                        help='Build all search observations as a step-3 control.')
    parser.add_argument('--serial-geometry', action='store_true',
                        help='Disable nominal geometry/robustness overlap for a matched control.')
    parser.add_argument('--serial-results', action='store_true',
                        help='Disable result overlap while preserving CUDA chunk boundaries.')
    parser.add_argument('--root-candidates', type=int,
                        help='Widen root retention and generation; use the same value across a batch sweep.')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for name in ('evaluation_batch_size', 'cuda_batch_size', 'root_candidates',
                 'iterations', 'generation_batch_size', 'posegen_batch_size', 'workers',
                 'placement_batch_size', 'result_batch_size'):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f'{name} must be positive')
    if args.cuda_batch_size is not None and args.backend != 'cuda':
        parser.error('--cuda-batch-size requires --backend cuda')
    if (args.placement_batch_size is not None or args.serial_results) and args.backend != 'cuda':
        parser.error('--placement-batch-size and --serial-results require --backend cuda')
    if args.posegen_batch_size is not None:
        if args.backend != 'cuda':
            parser.error('--posegen-batch-size requires --backend cuda')
        from stacking_common.actions.cuda_solver import CudaPlacementPoseSolver
        CudaPlacementPoseSolver.batch_size = args.posegen_batch_size
    if args.eager_observations:
        from stacking_common.environment.observation import ArrayObservationBuilder
        ArrayObservationBuilder.defer = ArrayObservationBuilder.build
    result_batches = []
    from stacking_common.environment.environment import StackingEnvironment
    if args.result_batch_size is not None:
        StackingEnvironment.result_batch_size = args.result_batch_size
    original_step_results = StackingEnvironment._step_results

    def measure_step_results(environment, states, *, start_index=0, **kwargs):
        started_results = time.perf_counter()
        try:
            return original_step_results(environment, states, start_index=start_index, **kwargs)
        finally:
            finished = time.perf_counter()
            result_batches.append(dict(jobs=len(states), workers=environment.result_workers,
                                       start=started_results, end=finished,
                                       seconds=finished - started_results))

    StackingEnvironment._step_results = measure_step_results
    if args.serial_results:
        def serial_step_results(environment, batches):
            results = []
            start_index = 0
            for states in batches:
                results.extend(environment._step_result_chunks(states, start_index=start_index))
                start_index += len(states)
            return tuple(results)
        StackingEnvironment._pipelined_step_results = serial_step_results
    geometry_batches = []
    original_prepare = StackingEnvironment._prepare_geometries

    def measure_prepare(environment, parents, candidates, poses, *, start_index):
        started = time.perf_counter()
        try:
            return original_prepare(environment, parents, candidates, poses, start_index=start_index)
        finally:
            ended = time.perf_counter()
            geometry_batches.append(dict(jobs=len(parents), workers=environment.result_workers,
                                         start=started, end=ended, seconds=ended - started))

    StackingEnvironment._prepare_geometries = measure_prepare
    cuda_calls = []
    if args.backend == 'cuda':
        from stacking_common.environment import cuda_backend
        if args.serial_geometry or args.serial_results:
            cuda_backend.CudaPlacementBackend.place_batch_with_preparation = None
        class MeasuredCudaBackend(cuda_backend.CudaPlacementBackend):
            early_excursion_stop = args.early_stops in ('excursion', 'both')
            early_quiet_stop = args.early_stops in ('quiet', 'both')
            batch_size = args.cuda_batch_size or cuda_backend.CudaPlacementBackend.batch_size
            placement_batch_size = args.placement_batch_size or cuda_backend.CudaPlacementBackend.placement_batch_size
            benchmark_stage = 'structural'

            def _settling(self, states, actions):
                self.benchmark_stage = 'settling'
                try:
                    return super()._settling(states, actions)
                finally:
                    self.benchmark_stage = 'structural'

            def _robustness(self, jobs):
                self.benchmark_stage = 'robustness'
                try:
                    return super()._robustness(jobs)
                finally:
                    self.benchmark_stage = 'structural'

            def _run(self, problems):
                started_run = time.perf_counter()
                outputs = super()._run(problems)
                finished = time.perf_counter()
                if problems:
                    cuda_calls.append(dict(
                        stage=self.benchmark_stage, jobs=len(problems),
                        statuses=[out.status.name for out in outputs],
                        simulated_steps=sum(p.steps for out in outputs for p in out.phases),
                        returned_samples=sum(out.sample_count if hasattr(out, "sample_count") else len(out.samples)
                                             for out in outputs),
                        chunks=[min(self.batch_size, len(problems) - i)
                                for i in range(0, len(problems), self.batch_size)],
                        start=started_run, end=finished, seconds=finished - started_run))
                return outputs

        # Process-local benchmark override; production defaults stay unchanged.
        cuda_backend.CudaPlacementBackend = MeasuredCudaBackend
    started = time.perf_counter()
    app = ExcavatorApplication.from_repository(ROOT, mcts_config=args.config,
                                               target_shape=args.target_shape)
    app = replace(app, mcts_profile=replace(app.mcts_profile,
        parallel=replace(app.mcts_profile.parallel,
                         posegen_backend=args.backend, simulation_backend=args.backend),
        root_cem=replace(app.mcts_profile.root_cem, seed=args.seed)))
    profile = app.mcts_profile
    if args.workers is not None:
        profile = replace(profile, parallel=replace(profile.parallel, enabled=True, workers=args.workers))
    if args.iterations is not None:
        profile = replace(profile, request=replace(profile.request, iterations=args.iterations))
    if args.generation_batch_size is not None:
        profile = replace(profile, tree=replace(profile.tree, generation_batch_size=args.generation_batch_size))
    if args.evaluation_batch_size is not None:
        profile = replace(profile, tree=replace(
            profile.tree, evaluation_batch_size=args.evaluation_batch_size))
    if args.root_candidates is not None:
        count = args.root_candidates
        profile = replace(profile,
                          tree=replace(profile.tree, root_max_children=count,
                                       generation_batch_size=max(profile.tree.generation_batch_size, count)),
                          root_cem=replace(profile.root_cem, keep=count,
                                           population=max(profile.root_cem.population, 2 * count),
                                           max_per_object=None))
    app = replace(app, mcts_profile=profile)
    state = pickle.loads(args.state.read_bytes()) if args.state else None
    ids = list(app.inventory_profile.object_ids) if state is None else state.object_ids
    inventory = app.load_inventory(ids, {i: [0., 0., 0., 0., 0., 0., 1.] for i in ids})
    environment = app.build_environment(inventory, app.build_core_backend(inventory, EvaluationFidelity.LONG))
    search = app.build_mcts(inventory, rng=random.Random(args.seed))
    evaluation_batch_sizes = []
    original_evaluate = search._evaluate_transitions

    def measure_evaluate(states, actions, fidelity):
        evaluation_batch_sizes.append(len(actions))
        return original_evaluate(states, actions, fidelity)

    search._evaluate_transitions = measure_evaluate
    request = app.default_mcts_request()
    try:
        reset = environment.reset()
        if state is None:
            state = reset.state
        else:
            environment.restore(state)
        root = MCTSNode(state=state.copy(), observation=environment.observation())
        setup_seconds = time.perf_counter() - started
        load_before = os.getloadavg()
        cuda_calls.clear()
        result_batches.clear()
        geometry_batches.clear()
        started_search = time.perf_counter()
        result = search.search(root, request)
        elapsed = time.perf_counter() - started_search
        # Each stream is sequential; pairwise interval intersections cannot double count.
        overlap = sum(max(0., min(cpu['end'], gpu['end']) - max(cpu['start'], gpu['start']))
                      for cpu in result_batches for gpu in cuda_calls)
        geometry_overlap = sum(max(0., min(cpu['end'], gpu['end']) - max(cpu['start'], gpu['start']))
                               for cpu in geometry_batches for gpu in cuda_calls)
        for interval in [*result_batches, *geometry_batches, *cuda_calls]:
            interval['start'] -= started_search
            interval['end'] -= started_search
        row = search_statistics(result, elapsed, search, step=len(state.placed_object_indices) + 1)
        row.update(backend=args.backend, seed=args.seed, placed=len(state.placed_object_indices),
                   object_count=len(ids), setup_seconds=setup_seconds,
                   requested_iterations=request.iterations, max_time_seconds=request.max_time_seconds,
                   expanded_nodes_per_second=result.generated_candidates / elapsed,
                   evaluation_batch_sizes=evaluation_batch_sizes,
                   expansion_calls_per_second=result.expansion_calls / elapsed,
                   evaluated_nodes_per_second=result.evaluated_candidates / elapsed,
                   nonfailed_evaluations_per_second=(result.evaluated_candidates - result.failed_evaluations) / elapsed,
                   benchmark_overrides={name: getattr(args, name) for name in (
                       'evaluation_batch_size', 'cuda_batch_size', 'root_candidates',
                       'iterations', 'generation_batch_size', 'posegen_batch_size', 'workers',
                       'placement_batch_size', 'result_batch_size', 'serial_results', 'serial_geometry',
                       'eager_observations', 'early_stops')},
                   config_path=str(args.config.resolve()) if args.config else None,
                   inventory_profile=_values(app.inventory_profile),
                   cuda_simulation_calls=cuda_calls,
                   result_processing_batches=result_batches,
                   result_cuda_overlap_seconds=overlap,
                   geometry_preparation_batches=geometry_batches,
                   geometry_cuda_overlap_seconds=geometry_overlap, tree_sha256=tree_digest(root),
                   accepted_tree_sha256=tree_digest(root, accepted_only=True),
                   benchmark_script_sha256=digest(__file__),
                   state_sha256=digest(args.state) if args.state else None,
                   load_before=load_before, load_after=os.getloadavg(),
                   core_native_sha256=digest(core._native.__file__), capabilities=core.cuda_capabilities(),
                   profiles={name: _values(getattr(app, name)) for name in (
                       'mcts_profile', 'posegen_profile', 'simulation_profile',
                       'environment_profile', 'action_generation_profile')},
                   source_sha256={str(p.relative_to(ROOT)): digest(p) for directory in (
                       ROOT / 'packages/stacking-common/src', ROOT / 'apps/excavator/src')
                       for p in sorted(directory.rglob('*.py'))},
                   core_commit=subprocess.check_output(['git','-C',str(ROOT.parent/'stacking-core'),'rev-parse','HEAD'],text=True).strip(),
                   apps_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                   selected=None if result.selected is None else dict(
                       object_id=result.selected.action.object_id,
                       pose=result.selected.action.pose.tolist(), reward=result.selected.reward,
                       target_IoU=result.selected.info.get('target_IoU'),
                       selectable=result.selected.selectable))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(row, indent=2) + '\n')
        print(json.dumps({k: row[k] for k in (
            'backend','seed','placed','elapsed_seconds','completed_iterations','timed_out',
            'generated_candidates','evaluated_candidates','evaluated_nodes_per_second',
            'nonfailed_evaluations_per_second','selected')}), flush=True)
    finally:
        search.close()
        environment.close()


if __name__ == '__main__':
    main()
