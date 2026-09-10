"""Run a journaled placement simulation without ROS or a NUC."""
import argparse
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
from stacking_common.execution import ExecutionCommand
from stacking_common.planning import MotionPlanSegment, GraspEvent
from stacking_excavator.acquisition.capture import digest
from .runtime import SimulationRuntime
from .saved_plan import load_command
from .simulation import SimulatedDevice


class SimulationClock:
    def __init__(self):
        self.seconds = 0.
    def __call__(self):
        return self.seconds
    def advance(self, seconds=.05):
        self.seconds += seconds


def demonstration_command(device, *, regrasp=False):
    """Small joint excursions for workflow testing; not a placement solution."""
    names = ['pick_approach', 'pick_retreat', 'place_approach', 'place_retreat']
    if regrasp:
        names[2:2] = ['handoff_place', 'handoff_retreat', 'handoff_approach', 'handoff_pick']
    segments, events = [], []
    q = device.positions.copy()
    for i, name in enumerate(names):
        end = q.copy()
        end[0] += .01 if i % 2 == 0 else -.01
        segments.append(MotionPlanSegment(name, np.stack([q, end])))
        if name in ('pick_approach', 'handoff_approach', 'handoff_place', 'place_approach'):
            acquire = name in ('pick_approach', 'handoff_approach')
            events.append(GraspEvent('acquire' if acquire else 'release', i, 1,
                                    np.array([0., 0., 0., 0., 0., 0., 1.]), -.4 if acquire else 0.))
        q = end
    return ExecutionCommand(uuid4().hex, 1, 1, 1, digest({'demo': names}), digest({'mode': 'simulation'}),
                            'world', device.joint_names, tuple(segments), tuple(events))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path.cwd())
    parser.add_argument('--session', type=Path, required=True, help='durable simulation/review directory')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--demo', choices=('direct', 'regrasp'))
    source.add_argument('--run', type=Path, help='committed .stack placement')
    parser.add_argument('--step', type=int)
    args = parser.parse_args(argv)
    clock = SimulationClock()
    runtime = None
    try:
        device = SimulatedDevice(args.repository, clock=clock)
        runtime = SimulationRuntime(args.repository, args.session, device=device, clock=clock)
        command = (demonstration_command(device, regrasp=args.demo == 'regrasp') if args.demo else
                   load_command(args.repository, args.run, step=args.step))
        runtime.start(command, expected_revision=runtime.review.snapshot()['revision'])
        for _ in range(200000):
            clock.advance()
            snapshot = runtime.tick()
            if snapshot['status'] not in ('running', 'held'):
                break
            runtime.heartbeat()
        else:
            runtime.cancel()
            snapshot = runtime.snapshot()
        print(json.dumps(snapshot, indent=2))
        return 0 if snapshot['status'] == 'succeeded' else 1
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'simulation error: {error}\n')
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == '__main__':
    raise SystemExit(main())
