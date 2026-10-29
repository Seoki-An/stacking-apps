"""Time a full search, then profile its slowest native batch in each stage.

Wrap profile_excavator_cuda_mcts.py arguments. Nsight collection must use
--capture-range=cudaProfilerApi --capture-range-end=repeat:3 so CUDA tracing
starts only for the three post-search replays. Search timers do not include
replay or fingerprint work. Saved cases are owned native objects, not pickles.
"""
import argparse
from collections import Counter
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
import profile_excavator_cuda_mcts as profile
from stacking_common.environment.cuda_backend import CudaPlacementBackend


def fingerprint(outputs):
    """Exact numerical outputs, including partial histories and solver metrics."""
    digest = hashlib.sha256()

    def add(value):
        if isinstance(value, np.ndarray):
            digest.update(str((value.dtype.str, value.shape)).encode())
            digest.update(value.tobytes())
        else:
            digest.update(repr(value).encode())

    for out in outputs:
        add((out.job.request_id, out.status.name))
        if out.snapshot is not None:
            for body in out.snapshot.bodies:
                add((body.id, body.mobility.name))
                for value in (body.pose, body.linear, body.angular):
                    add(value)
        for phase in out.phases:
            for name in ('steps', 'quiet_steps', 'settled_steps', 'max_energy',
                         'max_motion_integral', 'motion_integrals', 'unconverged_steps'):
                add((name, getattr(phase, name)))
            add((phase.solver.iters, phase.solver.converged,
                 phase.solver.primal_residual, phase.solver.dual_residual))
        if out.samples:
            add(out.sample_times)
            # Compact robustness histories contain only the sampled entities.
            for body in out.samples[0].snapshot.bodies:
                add(body.id)
                add(out.sample_poses(body.id))
        for contact in out.contacts:
            for key, value in sorted(contact.items()):
                add(key)
                add(value)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--native-profile-output', type=Path, required=True)
    parser.add_argument('--replay-repeats', type=int, default=3)
    parser.add_argument('--cuda-capture', action='store_true')
    parser.add_argument('--capture-policy', choices=('slowest', 'largest'), default='slowest',
                        help='Largest selects batches deterministically for cross-build comparisons.')
    args, remaining = parser.parse_known_args()
    if args.native_profile_output.exists():
        raise FileExistsError(args.native_profile_output)
    if args.replay_repeats < 1:
        parser.error('replay-repeats must be positive')
    runtime = ctypes.CDLL('/usr/local/cuda/lib64/libcudart.so') if args.cuda_capture else None
    cases = {}
    run = CudaPlacementBackend._run

    def capture(self, problems):
        started = time.perf_counter()
        outputs = run(self, problems)
        elapsed = time.perf_counter() - started
        stage = getattr(self, 'benchmark_stage', 'structural')
        score = elapsed if args.capture_policy == 'slowest' else len(problems)
        if problems and (stage not in cases or score > cases[stage][0]):
            owner = SimpleNamespace(batch_size=self.batch_size,
                                    cuda_simulator=self.cuda_simulator)
            cases[stage] = score, elapsed, owner, problems, outputs
        return outputs

    CudaPlacementBackend._run = capture
    sys.argv = [profile.__file__, *remaining]
    try:
        profile.main()
    finally:
        CudaPlacementBackend._run = run
    report = dict(arguments=remaining, capture_policy=args.capture_policy, native_sha256=profile.benchmark.digest(
        profile.benchmark.core._native.__file__), cases={})
    for index, stage in enumerate(('settling', 'structural', 'robustness'), 1):
        _, elapsed, owner, problems, reference = cases[stage]
        expected = fingerprint(reference)
        times = []
        for _ in range(args.replay_repeats):
            started = time.perf_counter()
            outputs = run(owner, problems)
            times.append(time.perf_counter() - started)
            assert fingerprint(outputs) == expected, (stage, 'replay changed outputs')
        profiled_seconds = None
        if runtime:
            assert runtime.cudaProfilerStart() == 0
            started = time.perf_counter()
            try:
                outputs = run(owner, problems)
            finally:
                profiled_seconds = time.perf_counter() - started
                assert runtime.cudaProfilerStop() == 0
            assert fingerprint(outputs) == expected, (stage, 'profiling changed outputs')
        row = dict(
            capture_index=index, search_call_seconds=elapsed, jobs=len(problems),
            body_counts=[p.scene.body_count for p in problems],
            seconds=times, median_seconds=statistics.median(times),
            profiled_seconds=profiled_seconds, output_sha256=expected,
            statuses=dict(Counter(out.status.name for out in reference)),
            integrated_steps=sum(p.steps for out in reference for p in out.phases),
            per_job_steps=[sum(p.steps for p in out.phases) for out in reference],
            unconverged_steps=sum(p.unconverged_steps for out in reference for p in out.phases),
            contact_counts=[len(out.contacts) for out in reference],
        )
        report['cases'][stage] = row
        args.native_profile_output.write_text(json.dumps(report, indent=2) + '\n')
        print('NATIVE_REPLAY', stage, json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
