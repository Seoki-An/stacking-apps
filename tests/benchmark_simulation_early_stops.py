"""Paired native and placement replays using saved candidates and physics.

The original file is opened read-only. Compare shortcuts independently on the
same binary; failed histories may be prefixes, accepted outputs must be exact.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import stacking_core as core
from stacking_common.artifacts import StackRunStore
from stacking_common.environment.core_backend import CoreSimulationProfile, CorePlacementConfig, CorePhysicsProfile, SimulationPhaseConfig
from stacking_common.environment.cuda_backend import CudaPlacementBackend
from stacking_common.environment.robustness import PlaceStabilityConfig
from stacking_excavator import ExcavatorApplication
from saved_placement_cases import load_case


def simulation_profile(values):
    values = dict(values)
    placement = dict(values.pop('placement'))
    for key in ('structural', 'short_gate'):
        if placement.get(key) is not None:
            placement[key] = SimulationPhaseConfig(**dict(placement[key]))
    values['placement'] = CorePlacementConfig(**placement)
    if values.get('settling_physics') is not None:
        values['settling_physics'] = CorePhysicsProfile(**dict(values['settling_physics']))
    return CoreSimulationProfile(**values)


class MeasuredBackend(CudaPlacementBackend):
    def _run(self, problems):
        started = time.perf_counter()
        outputs = super()._run(problems)
        self.calls.append(dict(wall=time.perf_counter()-started, jobs=len(problems),
            steps=sum(p.steps for out in outputs for p in out.phases),
            stops=dict(Counter(out.status.name for out in outputs))))
        return outputs


def fingerprint(result):
    if result.failed or not result.settled:
        return 'rejected'
    poses = {str(k): np.asarray(v).tolist() for k, v in result.object_poses.items()}
    data = dict(
        poses=poses,
        contacts=[asdict(contact) for contact in result.contact_points],
        robustness=result.diagnostics.get('place_robustness_trial_displacements'),
        robustness_reasons=result.diagnostics.get('place_robustness_trial_outcomes'),
        histories={str(k): dict(poses=np.asarray(t.poses), times=t.times,
                               energy=t.energy, integral=t.vel_integral)
                   for k, t in result.trajectories.items()},
    )
    return hashlib.sha256(json.dumps(data, sort_keys=True,
                                    default=lambda x: x.tolist()).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stack',type=Path,required=True)
    parser.add_argument('--steps',type=int,nargs='+',default=[10,20,30,40])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--modes', nargs='+', choices=('baseline','excursion','quiet','combined'),
                        default=['baseline','excursion','quiet','combined'])
    args=parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    cp=StackRunStore.load_planning_checkpoint(args.stack,1)
    ctx=StackRunStore.load_run_context(args.stack,cp.context_sha256)
    app=ExcavatorApplication.from_repository(Path(__file__).resolve().parents[1],
        recorded_target=ctx.profiles['inventory_profile'])
    inventory=app.load_inventory(cp.state.object_ids,cp.pick_scene.object_poses)
    profile=simulation_profile(ctx.profiles['simulation_profile'])
    stability=PlaceStabilityConfig(**dict(ctx.profiles['environment_profile']['place_stability']))
    backend=MeasuredBackend.from_profile(inventory.models,profile,place_stability=stability)
    backend.reset(cp.state.object_ids)
    report=dict(stack=str(args.stack.resolve()),
        stack_sha256=hashlib.sha256(args.stack.read_bytes()).hexdigest(),
        native_sha256=hashlib.sha256(Path(core._native.__file__).read_bytes()).hexdigest(),
        profile=asdict(profile),robustness=asdict(stability),cases=[])
    modes={'baseline':(False,False),'excursion':(True,False),'quiet':(False,True),'combined':(True,True)}
    modes={key:modes[key] for key in args.modes}
    try:
        for step in args.steps:
            state,candidates,actions=load_case(args.stack,step,cp.state.object_ids)
            backend.calls=[]
            backend.place(state,actions[0]) # Warm model caches.
            case=dict(step=step,candidates=len(actions),measurements={k:[] for k in modes})
            reference=None
            for repeat in range(args.repeats):
                order=list(modes) if repeat%2==0 else list(reversed(modes))
                for mode in order:
                    backend.early_excursion_stop,backend.early_quiet_stop=modes[mode]
                    backend.calls=[]
                    start=time.perf_counter()
                    results=backend.place_batch([state]*len(actions),actions)
                    wall=time.perf_counter()-start
                    fingerprints=[fingerprint(r) for r in results]
                    if reference is None:
                        reference=fingerprints
                    assert fingerprints==reference,(step,mode,'acceptance or accepted-result mismatch')
                    measurement=dict(wall=wall,calls=backend.calls,
                        fingerprint=hashlib.sha256(json.dumps(fingerprints).encode()).hexdigest(),accepted=sum(f!='rejected' for f in fingerprints),
                        parity=True,reasons=dict(Counter(r.diagnostics.get('rejection_reason','passed') for r in results)))
                    case['measurements'][mode].append(measurement)
                    print(json.dumps(dict(step=step,mode=mode,repeat=repeat,**measurement)),flush=True)
            case['median_wall']={k:statistics.median(x['wall'] for x in v) for k,v in case['measurements'].items()}
            report['cases'].append(case)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2,default=lambda x: x.tolist())+'\n')
    finally:
        backend.close()


if __name__=='__main__':
    main()
