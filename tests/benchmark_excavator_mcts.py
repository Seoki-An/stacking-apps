"""Compare excavator search schedulers on identical, locally saved native states.

Run with the project Python. Pickles are generated here and must be trusted.
Preparation uses the pre-PMBS implementation from --baseline-ref. Measurements
run in separate processes; invoke one case/seed/mode at a time to avoid CPU
contention. No robot motion or hardware execution is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in ('packages/stacking-common/src', 'apps/excavator/src'):
    sys.path.insert(0, str(ROOT / directory))

from stacking_common.search import EvaluationFidelity, MCTSNode, MCTSSession
from stacking_common.workflows.mcts_reporting import search_statistics
from stacking_excavator import ExcavatorApplication


def baseline_class(directory, revision):
    package = 'packages/stacking-common/src/stacking_common/search/mcts'
    modules = {}
    for name in ('tree_policy', 'search'):
        target = directory / f'baseline_{name}.py'
        source = subprocess.check_output(
            ['git', 'show', f'{revision}:{package}/{name}.py'], cwd=ROOT)
        if target.exists() and target.read_bytes() != source:
            raise ValueError('baseline snapshot differs; choose a new output directory')
        if not target.exists():
            target.write_bytes(source)
        module_name = f'stacking_common.search.mcts._benchmark_{name}'
        spec = importlib.util.spec_from_file_location(module_name, target)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    # Use the exact historical score/selection implementation too.
    for name in ('best_child', 'can_expand', 'expansion_room', 'select_child'):
        setattr(modules['search'], name, getattr(modules['tree_policy'], name))
    return modules['search'].MonteCarloTreeSearch


def build(args):
    application = ExcavatorApplication.from_repository(ROOT)
    profile = application.mcts_profile
    application = replace(application, mcts_profile=replace(
        profile, root_cem=replace(profile.root_cem, seed=args.seed)))
    if args.force_depth3:
        application = replace(application, mcts_profile=replace(
            application.mcts_profile,
            tree=replace(application.mcts_profile.tree, initial_depth1_steps=0)))
    ids = list(application.inventory_profile.object_ids)
    inventory = application.load_inventory(ids, {i: [0., 0., 0., 0., 0., 0., 1.] for i in ids})
    environment = application.build_environment(inventory,
        application.build_core_backend(inventory, EvaluationFidelity.LONG))
    search = application.build_mcts(inventory, rng=random.Random(args.seed))
    if args.mode == 'previous' or args.prepare:
        search.__class__ = baseline_class(args.output, args.baseline_ref)
    elif args.mode == 'serial':
        # Keep worker/sibling batching; disable only ordinary multi-frontier rounds.
        search._run_descendant_frontier_batch = lambda *a: 0
    request = replace(application.default_mcts_request(),
        iterations=args.iterations, max_time_seconds=args.seconds)
    return application, environment, search, request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-ref', default='HEAD')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--placements', type=int, default=10)
    parser.add_argument('--case', type=int, default=10, help='number of placed objects in saved state')
    parser.add_argument('--mode', choices=('previous', 'pmbs', 'serial'), default='pmbs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--seconds', type=float, default=240)
    parser.add_argument('--force-depth3', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = '-depth3' if args.force_depth3 else ''
    filename = f'case{args.case}{suffix}-{args.mode}-seed{args.seed}-iter{args.iterations}-time{args.seconds:g}.json'
    destination = args.output / ('state_0.pkl' if args.prepare else filename)
    if destination.exists():
        raise FileExistsError(f'benchmark output already exists: {destination}')
    started = time.perf_counter()
    app, env, engine, request = build(args)
    try:
        if args.prepare:
            reset = env.reset()
            session = MCTSSession(engine, MCTSNode(state=reset.state, observation=reset.observation))
            (args.output / 'state_0.pkl').write_bytes(pickle.dumps(session.root.state))
            for step in range(1, args.placements + 1):
                begin = time.perf_counter()
                result = session.search(request)
                row = search_statistics(result, time.perf_counter() - begin, engine, step=step)
                row['selected_object_id'] = result.selected.action.object_id if result.selected else None
                with (args.output / 'preparation.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                print(json.dumps({k: row[k] for k in ('step','elapsed_seconds','selected_object_id')}), flush=True)
                if result.selected is None:
                    raise RuntimeError(f'prefix search exhausted at step {step}')
                session.promote(result.selected)
                (args.output / f'state_{step}.pkl').write_bytes(pickle.dumps(session.root.state))
            return
        source = args.output / f'state_{args.case}.pkl'
        state = pickle.loads(source.read_bytes())
        env.reset()
        env.restore(state)
        root = MCTSNode(state=env.snapshot(), observation=env.observation())
        setup_seconds = time.perf_counter() - started
        load_before = os.getloadavg()
        begin = time.perf_counter()
        result = engine.search(root, request)
        elapsed = time.perf_counter() - begin
        row = search_statistics(result, elapsed, engine, step=args.case + 1)
        row.update(mode=args.mode, seed=args.seed, case=args.case,
                   force_depth3=args.force_depth3, requested_iterations=args.iterations,
                   max_time_seconds=args.seconds, setup_seconds=setup_seconds,
                   load_before=load_before, load_after=os.getloadavg(),
                   state_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   tree_config=asdict(app.mcts_profile.tree), object_count=len(state.object_ids),
                   selected=None if result.selected is None else {
                       'object_id': result.selected.action.object_id,
                       'pose': result.selected.action.pose.tolist(),
                       'reward': result.selected.reward,
                       'target_IoU': result.selected.info.get('target_IoU'),
                       'failed': result.selected.failed,
                       'selection_score': result.selected.info.get('final_selection_score'),
                   })
        destination.write_text(json.dumps(row, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    finally:
        engine.close()
        env.close()


if __name__ == '__main__':
    main()
