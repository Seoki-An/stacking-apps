"""Compare current CPU/CUDA profiles on identical saved placement states.

Example:
  python tests/benchmark_excavator_mcts_profiles.py \\
    --run .stacks/excavator_plan10_gpu.stack --placed 10 15 20 \\
    --output .benchmarks/excavator-profiles

Expanded nodes means newly generated child nodes, including rejected physical
placements, excluding repeat validation evaluations. Each trial starts a fresh
tree and process. Profiles keep their own search policies and budgets.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from benchmark_excavator_mcts_suite import BENCHMARK, ROOT, export_scene, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--placed', nargs='+', type=int, default=[10, 15, 20])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 1 for n in args.placed):
        parser.error('repeats and placed counts must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('states', 'runs', 'configs'):
        (args.output / name).mkdir()
    scenes = {}
    for count in args.placed:
        path, origin = export_scene(str(count), f'{args.run.resolve()}:{count}', args.output / 'states')
        if origin['placed'] != count:
            raise ValueError(f'Checkpoint {count} contains {origin["placed"]} stones')
        scenes[str(count)] = dict(path=str(path.resolve()), **origin)
    configs = {}
    for backend in ('cpu', 'cuda'):
        source = ROOT / 'apps/excavator/config' / f'mcts_{backend}.yml'
        config = args.output / 'configs' / source.name
        config.write_bytes(source.read_bytes())
        configs[backend] = str(config.resolve())
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', QT_QPA_PLATFORM='offscreen')
    document = dict(
        command=sys.argv, platform=platform.platform(), cpu_count=os.cpu_count(),
        scenes=scenes, configs=configs,
        expanded_definition='generated_candidates / elapsed_seconds; new child nodes, including physical failures',
        timing='search.search only; includes generation, physics, backup, validation, repair and retries; excludes setup',
        trials=[],
    )
    output = args.output / 'summary.json'
    output.write_text(json.dumps(document, indent=2) + '\n')
    for repeat in range(args.repeats):
        for count in args.placed:
            # Alternate order to reduce systematic first/second effects.
            backends = ('cpu', 'cuda') if repeat % 2 == 0 else ('cuda', 'cpu')
            for backend in backends:
                path = args.output / 'runs' / f'{count}-{backend}-{repeat}.json'
                command = [sys.executable, str(BENCHMARK), '--backend', backend,
                           '--config', configs[backend], '--seed', str(args.seed),
                           '--state', scenes[str(count)]['path'], '--output', str(path.resolve())]
                print(f'START placed={count} backend={backend} repeat={repeat}', flush=True)
                with path.with_suffix('.log').open('w') as log:
                    completed = subprocess.run(command, cwd=ROOT, env=environment,
                                               stdout=log, stderr=subprocess.STDOUT)
                row = dict(placed=count, backend=backend, repeat=repeat,
                           returncode=completed.returncode, result=str(path.resolve()))
                if completed.returncode == 0:
                    result = json.loads(path.read_text())
                    row.update(summary(str(count), args.seed, result))
                    row.update(generated=result['generated_candidates'],
                               expanded_per_second=result['expanded_nodes_per_second'],
                               expansion_calls=result['expansion_calls'],
                               expansion_calls_per_second=result['expansion_calls_per_second'],
                               setup_seconds=result['setup_seconds'])
                document['trials'].append(row)
                output.write_text(json.dumps(document, indent=2) + '\n')
                print(json.dumps(row), flush=True)
    if any(row['returncode'] for row in document['trials']):
        raise SystemExit('One or more trials failed; see run logs.')


if __name__ == '__main__':
    main()
