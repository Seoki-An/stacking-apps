"""Replay saved candidates with velocity disturbances enabled/disabled on CUDA.

Measures candidate evaluation overhead, not whole-MCTS performance. Saved files
are read-only; current platform profiles determine the replay physics.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import stacking_core as core
from stacking_common.artifacts import StackRunStore
from stacking_common.environment.cuda_backend import CudaPlacementBackend
from stacking_excavator import ExcavatorApplication
from saved_placement_cases import load_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stack', type=Path, required=True)
    parser.add_argument('--steps', type=int, nargs='+', default=[1, 3, 4])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    app = ExcavatorApplication.from_repository(Path(__file__).resolve().parents[1])
    ids = StackRunStore.load_planning_checkpoint(args.stack, 1).state.object_ids
    inventory = app.load_inventory(ids, {i: np.array([0., 0., 0., 0., 0., 0., 1.]) for i in ids})
    backend = CudaPlacementBackend.from_profile(inventory.models, app.simulation_profile,
        place_stability=app.environment_profile.place_stability)
    backend.reset(ids)
    policy = backend.place_stability
    report = dict(stack=str(args.stack.resolve()),
        stack_sha256=hashlib.sha256(args.stack.read_bytes()).hexdigest(),
        native_sha256=hashlib.sha256(Path(core._native.__file__).read_bytes()).hexdigest(),
        simulation_profile=asdict(app.simulation_profile), place_stability=asdict(policy), measurements=[])
    try:
        for step in args.steps:
            state, candidates, actions = load_case(args.stack, step, ids)
            entry = dict(step=step, candidates=len(actions), modes={})
            report['measurements'].append(entry)
            timings = {'nominal_only': [], 'velocity_disturbance': []}
            results = {}
            for enabled in (False, True):
                backend.place_stability = replace(policy, enabled=enabled)
                backend.place(state, actions[0])
            for repeat in range(args.repeats):
                for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
                    mode = 'velocity_disturbance' if enabled else 'nominal_only'
                    backend.place_stability = replace(policy, enabled=enabled)
                    start = time.perf_counter()
                    values = backend.place_batch([state] * len(actions), actions)
                    timings[mode].append(time.perf_counter() - start)
                    results[mode] = values
            for mode, values in results.items():
                entry['modes'][mode] = dict(seconds=timings[mode], median_seconds=statistics.median(timings[mode]),
                    accepted=sum(not r.failed and r.settled for r in values),
                    rejections=dict(Counter(r.diagnostics.get('rejection_reason', 'passed') for r in values)),
                    trial_outcomes=dict(Counter(outcome for r in values
                        for outcome in r.diagnostics.get('place_robustness_trial_outcomes', []))))
            entry['candidates_detail'] = [dict(candidate_id=c['candidate_id'],
                saved_rejection=c.get('rejection_reason'),
                nominal_rejection=n.diagnostics.get('rejection_reason'),
                replay_rejection=v.diagnostics.get('rejection_reason'), settled=v.settled,
                evaluated=v.diagnostics['place_robustness_evaluated'],
                excursions=v.diagnostics['place_robustness_trial_displacements'],
                outcomes=v.diagnostics['place_robustness_trial_outcomes'])
                for c, n, v in zip(candidates, results['nominal_only'], results['velocity_disturbance'])]
            print(json.dumps({k: v for k, v in entry.items() if k != 'candidates_detail'}), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v)) + '\n')
    finally:
        backend.close()


if __name__ == '__main__':
    main()
