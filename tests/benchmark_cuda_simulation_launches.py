"""Replay the first actual MCTS batch of each fidelity/stage for native tuning.

Pass normal benchmark_excavator_cuda_mcts arguments after --replay-output and
--repeats. --output is a required but unused search output: capture stops once
all six native cases are owned. Replay timing excludes hashing and search setup.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import benchmark_excavator_cuda_mcts as benchmark
from stacking_common.environment.cuda_backend import CudaPlacementBackend


class Captured(BaseException):
    pass


def fingerprint(values):
    digest = hashlib.sha256()
    def add(value):
        if hasattr(value, 'tobytes'):
            digest.update(value.tobytes())
        else:
            digest.update(repr(value).encode())
    def snapshot(scene):
        if scene is None:
            add(None)
            return
        add(scene.frame_id)
        for body in scene.bodies:
            add((body.id, body.model_id, body.mobility.name))
            for field in (body.pose, body.linear, body.angular):
                add(field)
    for out in values:
        if hasattr(out, 'scene'):
            add(out.request_id)
            snapshot(out.scene)
            for phase in out.phases:
                for name in ('dt','steps','min_steps','quiet_steps','max_total_steps','energy_threshold',
                             'motion_limit','sample_stride','continue_previous','reset_motion','monitored','mobility'):
                    add((name, getattr(phase, name)))
                rule = phase.robustness
                add(None if rule is None else (rule.min_time,rule.max_time))
        else:
            add((out.job.request_id, out.status.name))
            snapshot(out.snapshot)
            for phase in out.phases:
                for name in ('status','steps','quiet_steps','settled_steps','max_energy','max_motion_integral','motion_integrals','unconverged_steps'):
                    add((name,getattr(phase,name)))
            for sample in out.samples:
                add((sample.phase,sample.step,sample.time))
                snapshot(sample.snapshot)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--replay-output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    args, remaining = parser.parse_known_args()
    if args.replay_output.exists():
        raise FileExistsError(args.replay_output)
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    cases = {}
    run = CudaPlacementBackend._run
    def capture(self, problems):
        result = run(self, problems)
        stage = getattr(self, 'benchmark_stage', 'structural')
        fidelity = 'long' if self.config.short_gate is not None else 'short'
        key = fidelity + '-' + stage
        if problems and key not in cases:
            cases[key] = (SimpleNamespace(batch_size=self.batch_size,cuda_simulator=self.cuda_simulator),
                          problems, result)
            print('Captured '+key,flush=True)
        if len(cases)==6:
            raise Captured()
        return result
    CudaPlacementBackend._run = capture
    sys.argv = [benchmark.__file__, *remaining]
    try:
        benchmark.main()
    except Captured:
        pass
    finally:
        CudaPlacementBackend._run = run
    if len(cases)!=6:
        raise RuntimeError('search did not reach all six replay stages')
    rows={}
    for name,(owner,problems,reference) in cases.items():
        expected=fingerprint(reference)
        times=[]
        for repeat in range(args.repeats):
            start=time.perf_counter()
            results=run(owner,problems)
            times.append(time.perf_counter()-start)
            assert fingerprint(results)==expected, (name,repeat,'output changed')
            assert fingerprint(reference)==expected, (name,repeat,'previous result mutated')
        rows[name]=dict(jobs=len(problems),capacity=owner.batch_size,seconds=times,
                        median_seconds=statistics.median(times),input_sha256=fingerprint(problems),
                        output_sha256=expected,
                        steps=sum(p.steps for r in reference for p in r.phases),
                        samples=sum(len(r.samples) for r in reference))
        print(name,rows[name]['median_seconds'],flush=True)
    core=benchmark.ROOT.parent/'stacking-core'
    report=dict(cases=rows,arguments=remaining,repeats=args.repeats,
                native_sha256=benchmark.digest(benchmark.core._native.__file__),
                script_sha256=benchmark.digest(__file__),
                source_sha256={str(p.relative_to(core)):benchmark.digest(p) for p in
                               [core/'src/simulation/cuda/simulator.cpp',core/'src/cuda/runtime.hpp']},
                note='Captured real MCTS inputs, then repeated sequential native calls. Capture warms the workspace. Hashing and capture excluded from replay times. Input digest covers scenes/phases; resolved physics comes from the supplied app config. Output digest covers snapshots, samples and phase metrics; no CPU numerical equivalence claim.')
    args.replay_output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
