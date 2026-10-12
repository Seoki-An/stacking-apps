"""A/B replay of production CUDA robustness clearance against uncorrected noise.

Does not change production placement, profiles, or the input .stack. Uses the
current profiles and native binary, with saved root candidates and parent states.
Example (run with the environment's Python and a visible CUDA device)::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
      tests/benchmark_place_robustness_clearance.py --stack .stacks/run.stack \
      --steps 1 8 15 --batches 1 8 32 --output /tmp/clearance.json

This measures candidate evaluation, not an entire MCTS search. Changing trial
initialization can change both simulation durations and the subsequent search.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import stacking_core as core

from stacking_common.artifacts import StackRunStore
from stacking_common.domain import Action, State
from stacking_common.environment.cuda_backend import CudaPlacementBackend
from stacking_common.environment.release_clearance import ReleaseTrial
from stacking_excavator import ExcavatorApplication


class IdentityClearance:
    """Benchmark-only control: simulate the original noise without correction."""

    def correct(self, rows, poses, bases):
        return [ReleaseTrial(np.array(p, dtype=float, copy=True)) for p in poses]


class Probe(CudaPlacementBackend):
    correction = None
    corrected = False

    def _release_clearance_checker(self):
        if not self.corrected:
            return IdentityClearance()
        self.correction = super()._release_clearance_checker()
        return self.correction

    def _robustness(self, jobs):
        started = time.perf_counter()
        result = super()._robustness(jobs)
        self.last_clearance = (dict(self.correction.last)
                               if self.corrected and jobs else None)
        self.last_robustness = dict(seconds=time.perf_counter() - started,
                                   trial_reasons=dict(Counter(
                                       reason or 'passed' for _, _, reasons in result
                                       for reason in reasons)))
        return result


def load_case(path, step, ids):
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as connection:
        row = connection.execute("select payload_json from records where "
                                 "kind='mcts_step' and ordinal=?", (step,)).fetchone()
        if row is None:
            raise ValueError(f'no saved step {step}')
        data = json.loads(row[0])
    if step == 1:
        state = State(ids, [], {}, [], [])
    else:
        saved = StackRunStore.load_planning_checkpoint(path, step - 1).state
        state = State(saved.object_ids, saved.placed_object_indices,
                      saved.object_poses, [], [],
                      pose_identified_object_ids=saved.pose_identified_object_ids)
    candidates = [c for c in data['candidates'] if c.get('action_pose') is not None]
    preferred = 'step-1-candidate-9' if step == 1 else data['selected_candidate_id']
    candidates.sort(key=lambda c: c['candidate_id'] != preferred)
    actions = [Action(c['object_index'], np.array(c['action_pose']),
                      np.array(c['init_pose']), object_id=c['object_id']) for c in candidates]
    return state, candidates, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stack', type=Path, required=True)
    parser.add_argument('--steps', type=int, nargs='+', default=[1, 8, 15])
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 8, 32])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.batches) < 1 or max(args.batches) > 128 or args.repeats < 1:
        parser.error('batch must be 1..128; repeats must be positive')
    app = ExcavatorApplication.from_repository(Path(__file__).resolve().parents[1])
    if not app.environment_profile.place_stability.active:
        parser.error('the current profile must enable robustness trials')
    ids = StackRunStore.load_planning_checkpoint(args.stack, 1).state.object_ids
    inventory = app.load_inventory(ids, {i: np.array([0., 0., 0., 0., 0., 0., 1.]) for i in ids})
    backend = Probe.from_profile(inventory.models, app.simulation_profile,
                                place_stability=app.environment_profile.place_stability)
    report = dict(stack=str(args.stack.resolve()),
                  benchmark_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  stack_sha256=hashlib.sha256(args.stack.read_bytes()).hexdigest(),
                  native_sha256=hashlib.sha256(Path(core._native.__file__).read_bytes()).hexdigest(),
                  cuda=core.cuda_capabilities(),
                  simulation_profile=asdict(app.simulation_profile),
                  place_stability=asdict(app.environment_profile.place_stability),
                  measurements=[])

    def save():
        args.output.write_text(json.dumps(report, indent=2, default=lambda x:
                              x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')

    try:
        backend.reset(ids)
        for step in args.steps:
            state, candidates, available = load_case(args.stack, step, ids)
            for count in args.batches:
                if count > len(available):
                    raise ValueError(f'only {len(available)} candidates at step {step}')
                actions = available[:count]
                states = [state] * count
                measurement = dict(step=step, placed=len(state.placed_object_indices),
                                   batch=count, candidates=[c['candidate_id'] for c in candidates[:count]],
                                   baseline=[], corrected=[])
                report['measurements'].append(measurement)
                # Warm each path, then alternate order to reduce clock/thermal bias.
                for repetition in range(-1, args.repeats):
                    for corrected in ([False, True] if repetition % 2 else [True, False]):
                        backend.corrected = corrected
                        started = time.perf_counter()
                        results = backend.place_batch(states, actions)
                        elapsed = time.perf_counter() - started
                        row = dict(total_seconds=elapsed, robustness=backend.last_robustness,
                                   candidate_reasons=dict(Counter(
                                       r.diagnostics.get('rejection_reason') or 'passed'
                                       for r in results)))
                        if corrected:
                            row['clearance'] = backend.last_clearance
                        if repetition >= 0:
                            measurement['corrected' if corrected else 'baseline'].append(row)
                        print(json.dumps(dict(step=step, batch=count, repetition=repetition,
                                              corrected=corrected, **row)), flush=True)
                        save()
        report['clearance_workspace_bytes'] = {
            count: workspace.allocated_bytes
            for count, workspace in backend.correction.queries.workspaces.items()}
        save()
    finally:
        backend.close()


if __name__ == '__main__':
    main()
