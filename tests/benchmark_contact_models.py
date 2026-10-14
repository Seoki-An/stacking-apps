"""Replay saved root candidates with 4D and 6D structural contacts.

Uses current platform profiles, identical single-point settling and noise, and
the same native binary for both models. Saved scenes are opened read-only.
Measures candidate evaluation, not a complete MCTS search.

Example::

    python tests/benchmark_contact_models.py --stacks .stacks/run.stack \
      --output /tmp/contact-models.json

Use ``--patch-eps 1e-4 1e-5`` to compare patch sizes with the 6D model.
"""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import time

import numpy as np
import stacking_core as core

from stacking_common.artifacts import StackRunStore
from stacking_common.environment.cuda_backend import CudaPlacementBackend
from stacking_common.environment.robustness import pose_window_extent
from stacking_excavator import ExcavatorApplication

from saved_placement_cases import load_case


class Probe(CudaPlacementBackend):
    def _structural_result(self, scene, indices, action, settling, out):
        entity = self.scene_builder.entity_id(action.object_index)
        poses = out.sample_poses(entity)
        initial = np.asarray(scene.body_pose(entity))
        phase = out.phases[-1]
        tail_count = round(1.0 / self.config.structural.dt) + 1
        extent = pose_window_extent(poses[-tail_count:]) if len(poses) else (None, None)
        self.motion[id(action)] = dict(
            status=out.status.name,
            max_energy=phase.max_energy,
            max_motion_integral=phase.max_motion_integral,
            quiet_steps=phase.quiet_steps,
            unconverged_steps=sum(p.unconverged_steps for p in out.phases),
            tail_duration_seconds=min(max(len(poses) - 1, 0), tail_count - 1)
            * self.config.structural.dt,
            tail_position_extent_m=extent[0],
            tail_angle_extent_degrees=None if extent[1] is None else float(np.degrees(extent[1])),
            max_target_translation_m=float(np.linalg.norm(poses[:, :3] - initial[:3], axis=1).max())
            if len(poses) else None,
        )
        return super()._structural_result(scene, indices, action, settling, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stacks', type=Path, nargs='+', required=True)
    parser.add_argument('--steps', type=int, nargs='+')
    parser.add_argument('--fidelities', choices=('short', 'long'), nargs='+', default=['short', 'long'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--patch-eps', type=float, nargs='+',
                        help='compare these patch epsilons with 6D instead of comparing 4D/6D')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 128 or args.repeats < 1:
        parser.error('batch size must be 1..128 and repeats must be positive')
    if not hasattr(core.ContactModel, 'LIMIT_SURFACE_6D'):
        parser.error('use a rebuilt stacking-core with LIMIT_SURFACE_6D')
    if args.patch_eps and any(not np.isfinite(v) or v < 0 for v in args.patch_eps):
        parser.error('patch epsilons must be finite and non-negative')
    variants = ({f'limit_surface_6d/patch_eps={v:g}': dict(contact_model='limit_surface_6d', patch_eps=v)
                 for v in args.patch_eps} if args.patch_eps else
                {m: dict(contact_model=m) for m in ('limit_surface_4d', 'limit_surface_6d')})
    root = Path(__file__).resolve().parents[1]
    app = ExcavatorApplication.from_repository(root)
    report = dict(
        native_path=core._native.__file__,
        native_sha256=hashlib.sha256(Path(core._native.__file__).read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        apps_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        cuda=core.cuda_capabilities(),
        simulation_profile=asdict(app.simulation_profile),
        place_stability=asdict(app.environment_profile.place_stability),
        batch_size=args.batch_size, repeats=args.repeats, variants=variants,
        stacks=[], measurements=[],
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=lambda v:
            v.tolist() if isinstance(v, np.ndarray) else str(v)) + '\n')

    for path in args.stacks:
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
            steps = [r[0] for r in db.execute("select ordinal from records where kind='mcts_step' order by ordinal")]
        if args.steps:
            steps = [s for s in steps if s in args.steps]
        report['stacks'].append(dict(path=str(path.resolve()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), steps=steps))
        ids = StackRunStore.load_planning_checkpoint(path, 1).state.object_ids
        inventory = app.load_inventory(ids, {i: np.array([0., 0., 0., 0., 0., 0., 1.]) for i in ids})
        for fidelity in args.fidelities:
            profile = app.short_simulation_profile if fidelity == 'short' else app.simulation_profile
            # Keep the initial pose and release preparation identical between arms.
            profile = replace(profile, settling_physics=replace(profile.settling_physics, contact_model='single_point'))
            backends = {}
            try:
                for model, settings in variants.items():
                    backend = Probe.from_profile(inventory.models, replace(profile, **settings),
                        place_stability=app.environment_profile.place_stability)
                    backend.reset(ids)
                    backend.motion = {}
                    backends[model] = backend
                for step in steps:
                    state, candidates, actions = load_case(path, step, ids)
                    measurement = dict(stack=path.name, step=step, fidelity=fidelity, count=len(actions), models={})
                    report['measurements'].append(measurement)
                    # Warm both models; timings below exclude initialization.
                    for backend in backends.values():
                        backend.place(state, actions[0])
                        backend.motion.clear()
                    for repeat in range(args.repeats):
                        order = list(backends) if repeat % 2 else list(reversed(backends))
                        for model in order:
                            backend = backends[model]
                            backend.motion.clear()
                            started = time.perf_counter()
                            results = []
                            for start in range(0, len(actions), args.batch_size):
                                batch = actions[start:start + args.batch_size]
                                results.extend(backend.place_batch([state] * len(batch), batch))
                            seconds = time.perf_counter() - started
                            reasons = Counter(r.diagnostics.get('rejection_reason') or
                                ('simulation_unsettled' if not r.settled else 'accepted') for r in results)
                            row = measurement['models'].setdefault(model, dict(seconds=[], outcomes=[], candidates=[]))
                            row['seconds'].append(seconds)
                            row['outcomes'].append(dict(reasons))
                            if repeat == 0:
                                row['candidates'] = [dict(candidate_id=c['candidate_id'],
                                    recorded_rejection=c.get('rejection_reason'), recorded_selected=c.get('selected', False),
                                    failed=r.failed, settled=r.settled, diagnostics=r.diagnostics,
                                    motion=backend.motion.get(id(a)),
                                    final_pose=r.object_poses[a.object_id].tolist())
                                    for c, a, r in zip(candidates, actions, results)]
                            print(json.dumps(dict(stack=path.name, step=step, fidelity=fidelity,
                                model=model, repeat=repeat, seconds=seconds, outcomes=dict(reasons))), flush=True)
                    save()
            finally:
                for backend in backends.values():
                    backend.close()
    save()


if __name__ == '__main__':
    main()
