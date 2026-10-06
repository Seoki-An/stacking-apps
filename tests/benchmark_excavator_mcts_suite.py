"""Run the excavator MCTS benchmark over several recorded scenes and seeds.

Each --scene is NAME=RUN.stack:STEP (a committed planning checkpoint) or
NAME=STATE.pkl (a trusted, locally exported state). States are exported once
into OUTPUT/states with their source hashes, so results stay reproducible when
a run file is later removed. Every scene and seed then runs
benchmark_excavator_cuda_mcts.py in a fresh process, sequentially, with one
BLAS/OpenMP thread. Arguments after -- go to that benchmark. Nothing is
overwritten.

Example:
  python tests/benchmark_excavator_mcts_suite.py --output /tmp/suite \\
    --scene early=.stacks/run.stack:4 --scene late=.stacks/run.stack:21 \\
    --seeds 0 1 2 -- --backend cuda --config apps/excavator/config/mcts_cuda.yml
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in ('packages/stacking-common/src', 'apps/excavator/src'):
    sys.path.insert(0, str(ROOT / directory))

from stacking_common.artifacts.run_document import StackRunStore

BENCHMARK = Path(__file__).with_name('benchmark_excavator_cuda_mcts.py')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def export_scene(name, spec, directory):
    """Write the scene's state pickle; return where it came from."""
    source, _, step = spec.rpartition(':')
    if source.endswith('.stack') and step.isdigit():
        checkpoint = StackRunStore.load_planning_checkpoint(source, step=int(step))
        state, origin = checkpoint.state, dict(run=source, step=int(step),
                                               run_sha256=digest(Path(source).read_bytes()))
    else:
        data = Path(spec).read_bytes()
        state, origin = pickle.loads(data), dict(state=spec)
    data = pickle.dumps(state)
    path = directory / f'{name}.pkl'
    path.write_bytes(data)
    return path, dict(origin, placed=len(state.placed_object_indices),
                      objects=len(state.object_ids), state_sha256=digest(data))


def summary(name, seed, result):
    sim = result['simulation']
    selected = result.get('selected') or {}
    fallback = result.get('floor_fill_fallback') or {}
    return dict(
        scene=name, seed=seed, seconds=result['elapsed_seconds'],
        timed_out=result['timed_out'], iterations=result['completed_iterations'],
        evaluated=result['evaluated_candidates'], failed=result['failed_evaluations'],
        depth=result['effective_max_depth'],
        evaluated_per_second=result['evaluated_nodes_per_second'],
        pose_generation_seconds=sim['pose_generation_seconds'],
        short_seconds=sim['short_batch_wall_seconds'], long_seconds=sim['long_batch_wall_seconds'],
        final_validation_timed_out=result['final_validation']['timed_out'],
        descendant_validation=result.get('descendant_validation'),
        root_validation=result.get('root_validation'), root_retry_rounds=result['root_retry']['rounds'],
        fallback_attempted=bool(fallback.get('attempted')), fallback_selected=bool(fallback.get('selected')),
        selected_object=selected.get('object_id'), selected_reward=selected.get('reward'),
        selected_target_iou=selected.get('target_IoU'), tree_sha256=result['tree_sha256'],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene', action='append', required=True, metavar='NAME=SOURCE')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    arguments = sys.argv[1:]
    split = arguments.index('--') if '--' in arguments else len(arguments)
    args = parser.parse_args(arguments[:split])
    forwarded = arguments[split + 1:]
    if args.output.exists():
        raise FileExistsError(args.output)
    (args.output / 'states').mkdir(parents=True)
    (args.output / 'runs').mkdir()
    scenes = {}
    for item in args.scene:
        name, _, spec = item.partition('=')
        path, origin = export_scene(name, spec, args.output / 'states')
        scenes[name] = dict(path=str(path), **origin)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       QT_QPA_PLATFORM=os.environ.get('QT_QPA_PLATFORM', 'offscreen'))
    rows = []
    for name, scene in scenes.items():
        for seed in args.seeds:
            output = args.output / 'runs' / f'{name}-seed{seed}.json'
            command = [sys.executable, str(BENCHMARK), *forwarded, '--seed', str(seed),
                       '--state', scene['path'], '--output', str(output)]
            completed = subprocess.run(command, env=environment, capture_output=True, text=True)
            (output.with_suffix('.log')).write_text(completed.stdout + completed.stderr)
            if completed.returncode:
                rows.append(dict(scene=name, seed=seed, error=completed.returncode))
            else:
                rows.append(summary(name, seed, json.loads(output.read_text())))
            row = rows[-1]
            print(f"{name:>12} seed {seed}: " + (f"error {row['error']}" if 'error' in row else
                  f"{row['seconds']:7.2f} s, depth {row['depth']}, evaluated {row['evaluated']}, "
                  f"long {row['long_seconds']:.1f} s, timed out {row['timed_out']}, "
                  f"fallback {row['fallback_attempted']}, selected {row['selected_object']}"), flush=True)
    (args.output / 'summary.json').write_text(json.dumps(dict(
        command=dict(suite=arguments, benchmark=forwarded), scenes=scenes, runs=rows), indent=1) + '\n')


if __name__ == '__main__':
    main()
