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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--config', type=Path, help='Use the same profile for controlled CPU/CUDA comparisons.')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--generation-batch-size', type=int)
    parser.add_argument('--posegen-batch-size', type=int)
    parser.add_argument('--state', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluation-batch-size', type=int)
    parser.add_argument('--cuda-batch-size', type=int)
    parser.add_argument('--root-candidates', type=int,
                        help='Widen root retention and generation; use the same value across a batch sweep.')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for name in ('evaluation_batch_size', 'cuda_batch_size', 'root_candidates',
                 'iterations', 'generation_batch_size', 'posegen_batch_size'):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f'{name} must be positive')
    if args.cuda_batch_size is not None and args.backend != 'cuda':
        parser.error('--cuda-batch-size requires --backend cuda')
    if args.posegen_batch_size is not None:
        if args.backend != 'cuda':
            parser.error('--posegen-batch-size requires --backend cuda')
        from stacking_common.actions.cuda_solver import CudaPlacementPoseSolver
        CudaPlacementPoseSolver.batch_size = args.posegen_batch_size
    cuda_calls = []
    if args.backend == 'cuda':
        from stacking_common.environment import cuda_backend

        class MeasuredCudaBackend(cuda_backend.CudaPlacementBackend):
            batch_size = args.cuda_batch_size or cuda_backend.CudaPlacementBackend.batch_size
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
                if problems:
                    cuda_calls.append(dict(
                        stage=self.benchmark_stage, jobs=len(problems),
                        chunks=[min(self.batch_size, len(problems) - i)
                                for i in range(0, len(problems), self.batch_size)],
                        seconds=time.perf_counter() - started_run))
                return outputs

        # Process-local benchmark override; production defaults stay unchanged.
        cuda_backend.CudaPlacementBackend = MeasuredCudaBackend
    started = time.perf_counter()
    app = ExcavatorApplication.from_repository(ROOT, mcts_config=args.config)
    app = replace(app, mcts_profile=replace(app.mcts_profile,
        parallel=replace(app.mcts_profile.parallel,
                         posegen_backend=args.backend, simulation_backend=args.backend),
        root_cem=replace(app.mcts_profile.root_cem, seed=args.seed)))
    profile = app.mcts_profile
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
    environment = app.build_environment(inventory, app.build_core_backend(inventory, EvaluationFidelity.SHORT))
    search = app.build_mcts(inventory, rng=random.Random(args.seed))
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
        started_search = time.perf_counter()
        result = search.search(root, request)
        elapsed = time.perf_counter() - started_search
        row = search_statistics(result, elapsed, search, step=len(state.placed_object_indices) + 1)
        row.update(backend=args.backend, seed=args.seed, placed=len(state.placed_object_indices),
                   object_count=len(ids), setup_seconds=setup_seconds,
                   requested_iterations=request.iterations, max_time_seconds=request.max_time_seconds,
                   evaluated_nodes_per_second=result.evaluated_candidates / elapsed,
                   nonfailed_evaluations_per_second=(result.evaluated_candidates - result.failed_evaluations) / elapsed,
                   benchmark_overrides={name: getattr(args, name) for name in (
                       'evaluation_batch_size', 'cuda_batch_size', 'root_candidates',
                       'iterations', 'generation_batch_size', 'posegen_batch_size')},
                   config_path=str(args.config.resolve()) if args.config else None,
                   cuda_simulation_calls=cuda_calls,
                   benchmark_script_sha256=digest(__file__),
                   state_sha256=digest(args.state) if args.state else None,
                   load_before=load_before, load_after=os.getloadavg(),
                   core_native_sha256=digest(core._native.__file__), capabilities=core.cuda_capabilities(),
                   profiles={name: _values(getattr(app, name)) for name in (
                       'mcts_profile', 'posegen_profile', 'simulation_profile', 'short_simulation_profile',
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
