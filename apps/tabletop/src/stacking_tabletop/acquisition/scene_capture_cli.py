"""Preview and execute a nine-view wrist scan bracketed by fixed FEMTO captures."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import signal
import sys

import numpy as np

from stacking_tabletop.execution.kinematics import UR5eKinematics
from .hand_eye import HandEyeCalibration
from .hand_eye_capture import atomic_json, capture_view, digest, inspect_capture
from .scene_capture import capture_lock, execute_capture
from .scene_views import build_views, kinematic_urdf, render_preview, validate_views

ROOT = Path(__file__).resolve().parents[5]


def robot_description(transport):
    """Read the running driver's description; never assume nominal factory FK."""
    return kinematic_urdf(transport.read_robot_description())


def ensure_ownership(transport):
    if transport.node.count_publishers('/dish_planning/pos_planner') > 1:
        raise RuntimeError('another robot reference publisher is active; stop its execution client, not UR bringup')
    if transport.feedback.gripper_state().hold_goal is not None:
        raise RuntimeError('scene scan requires an empty tool; reconcile the existing gripper hold first')


@contextmanager
def connected_arm(kinematics):
    from stacking_tabletop.execution.ur5e import UR5eAdapter
    from stacking_tabletop.ros.hardware import TabletopRosTransport
    with TabletopRosTransport(node_name='tabletop_scene_capture') as transport:
        arm = UR5eAdapter(transport, kinematics)
        arm.runtime.poll = lambda: ensure_ownership(transport)
        arm.wait_until_ready()
        yield arm, transport


def prepare_plan(args):
    directory = args.output.expanduser().resolve()
    if directory.exists():
        raise ValueError('plan output already exists')
    if args.seed_view is not None and args.urdf is None:
        raise ValueError('offline --seed-view needs --urdf matching the recorded robot')
    calibration = HandEyeCalibration.load(args.profile)
    directory.mkdir(parents=True)
    calibration.snapshot(directory / 'calibration')
    if args.seed_view is not None:
        root = args.seed_view.expanduser().resolve()
        manifest = inspect_capture(root)
        recorded = HandEyeCalibration.load(root / 'calibration/profile.yml')
        if recorded.report()['artifacts'] != calibration.report()['artifacts']:
            raise ValueError('seed capture and current hand-eye calibration differ')
        raw = json.loads((root / 'manifest.json').read_text())
        frame = json.loads((root / (raw['frames'][0] + '.json')).read_text())
        seed = np.asarray(frame['robot']['joint_positions'][-1])
        description = kinematic_urdf(args.urdf.expanduser().read_text())
        source = dict(kind='recorded_view', directory=str(root), serial=manifest['serial'])
    else:
        from stacking_tabletop.ros.hardware import TabletopRosTransport
        from stacking_tabletop.execution.runtime import AdapterRuntime
        with TabletopRosTransport(node_name='tabletop_scene_preview') as transport:
            description = robot_description(transport)
            runtime = AdapterRuntime()
            def ready():
                try:
                    transport.feedback.arm_state()
                    return transport.feedback.ready
                except RuntimeError:
                    return False
            runtime.until(ready, runtime.deadline(15), lambda: runtime.check(transport),
                          'need ready UR bringup and current joint/tool feedback')
            seed = transport.feedback.arm_state()[0]
        source = dict(kind='live_robot_description')
    urdf = directory / 'robot.urdf'
    urdf.write_text(description)
    kinematics = UR5eKinematics(urdf)
    plan = build_views(kinematics, calibration, seed, distance=args.distance,
                       offset=args.offset, oblique_offset=args.oblique_offset)
    plan.update(robot_urdf_sha256=digest(urdf), source=source,
                hand_eye_profile_sha256=digest(directory / 'calibration/profile.yml'))
    render_preview(plan, kinematics, calibration, directory / 'preview.html')
    atomic_json(directory / 'plan.json', plan)
    return dict(plan=str(directory), preview=str(directory / 'preview.html'),
                approval_sha256=digest(directory / 'plan.json'), motion_sent=False,
                collision_checked=False)


def load_reviewed_plan(directory, approval):
    directory = Path(directory).expanduser().resolve()
    if digest(directory / 'plan.json') != approval:
        raise ValueError('--approve must match the SHA256 printed for the reviewed plan')
    plan = json.loads((directory / 'plan.json').read_text())
    if digest(directory / 'robot.urdf') != plan['robot_urdf_sha256']:
        raise ValueError('planned robot kinematics changed')
    if digest(directory / 'calibration/profile.yml') != plan['hand_eye_profile_sha256']:
        raise ValueError('planned hand-eye profile changed')
    calibration = HandEyeCalibration.load(directory / 'calibration/profile.yml')
    kinematics = UR5eKinematics(directory / 'robot.urdf')
    validate_views(plan, kinematics, calibration)
    plan['approved_sha256'] = approval
    return plan, calibration, kinematics


def run_scan(args):
    from stacking_tabletop.config.femto import load_femto_profile
    from .femto import FemtoCalibration, FemtoCamera
    from .field_workflow import camera_driver
    from .storage import CaptureSession
    from .hand_eye_replay import assemble_session
    from .realsense import RealSenseCamera
    from stacking_tabletop.ros.femto import FemtoRosTransport
    from stacking_tabletop.ros.hand_eye import HandEyeRobot

    args.plan = args.plan.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.femto_config = args.femto_config.expanduser().resolve()
    if args.output.is_relative_to(args.plan) or args.plan.is_relative_to(args.output):
        raise ValueError('plan and capture output must be separate directories')
    if args.output.exists():
        raise ValueError('capture output already exists')
    if args.count < 1 or args.fps < 1 or min(args.depth_size) < 1:
        raise ValueError('count, fps and depth dimensions must be positive')
    plan, calibration, kinematics = load_reviewed_plan(args.plan, args.approve)
    femto = FemtoCalibration(load_femto_profile(args.femto_config, args.camera))
    if femto.profile.base_frame != 'base_link':
        raise ValueError('combined scene capture requires FEMTO observations in base_link')
    with capture_lock(), connected_arm(kinematics) as (arm, transport):
        live_hash = hashlib.sha256(robot_description(transport).encode()).hexdigest()
        if live_hash != plan['robot_urdf_sha256']:
            raise ValueError('running robot_description differs from the reviewed plan; prepare a new plan')
        namespace = femto.profile.namespace
        if transport.node.count_publishers(namespace + '/depth/image_raw'):
            raise ValueError('a FEMTO driver is already running; stop it once so this command can own camera scheduling')

        def femto_capture(directory):
            # Keep UR bringup running. Own only a short FEMTO driver interval;
            # no active FEMTO depth emitter remains during wrist acquisition.
            print(f'{directory.name}: starting FEMTO driver; stream initialization normally takes 15–20 s',
                  flush=True)
            with camera_driver(ROOT, args.femto_config, args.camera,
                               directory.parent / (directory.name + '_driver.log')) as process:
                def check():
                    arm._check_motion()
                    if process.poll() is not None:
                        raise RuntimeError('owned FEMTO driver exited during capture')
                with FemtoRosTransport(femto) as camera_transport:
                    sensor = FemtoCamera(camera_transport, femto, cancel_event=arm.runtime.cancel_event)
                    with CaptureSession(directory, femto) as session:
                        session.add(sensor.capture(check=check, progress=lambda done, total:
                            print(f'{directory.name}: FEMTO exposure {done}/{total}', flush=True)))
            print(f'{directory.name}: capture complete and driver stopped', flush=True)

        @contextmanager
        def wrist_context():
            with HandEyeRobot(calibration) as robot:
                with RealSenseCamera(calibration, depth_size=args.depth_size, fps=args.fps) as camera:
                    class CheckedCamera:
                        report = camera.report
                        def read(self):
                            arm._check_motion()
                            pair = camera.read()
                            arm._check_motion()
                            return pair
                    checked = CheckedCamera()
                    yield lambda directory: capture_view(directory, calibration, checked, robot, count=args.count)

        def snapshot(directory):
            # Snapshot before motion, including on failed/incomplete runs.
            destination = directory / 'reviewed_plan'
            shutil.copytree(args.plan, destination)
            if (digest(destination / 'plan.json') != args.approve
                    or digest(destination / 'robot.urdf') != plan['robot_urdf_sha256']
                    or digest(destination / 'calibration/profile.yml') != plan['hand_eye_profile_sha256']):
                raise ValueError('reviewed plan changed before capture')
            copied = HandEyeCalibration.load(destination / 'calibration/profile.yml')
            if copied.report()['artifacts'] != calibration.report()['artifacts']:
                raise ValueError('reviewed calibration changed before capture')

        return execute_capture(plan, args.output, arm, femto_capture, wrist_context,
                               assemble_session, snapshot=snapshot)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    plan = commands.add_parser('plan', help='read-only: generate a full path and HTML preview')
    plan.add_argument('--profile', type=Path, default=ROOT / 'apps/tabletop/config/hand_eye.yml')
    plan.add_argument('--distance', type=float, required=True, help='target distance along seed RGB optical axis, metres')
    plan.add_argument('--offset', type=float, default=.04, help='translated offset from center in metres')
    plan.add_argument('--oblique-offset', type=float, default=.08,
                      help='radial offset of four diagonal oblique views in metres')
    plan.add_argument('--seed-view', type=Path, help='offline planning from a recorded view instead of live feedback')
    plan.add_argument('--urdf', type=Path, help='verified URDF for offline --seed-view planning')
    plan.add_argument('--output', type=Path, required=True)
    run = commands.add_parser('run', help='MOVE ROBOT through a reviewed plan and capture both sensors')
    run.add_argument('--plan', type=Path, required=True)
    run.add_argument('--approve', required=True, help='SHA256 of the previewed plan; confirms empty tool and full-path clearance review')
    run.add_argument('--femto-config', type=Path, default=ROOT / 'apps/tabletop/config/femto.yml')
    run.add_argument('--camera', choices=('primary', 'secondary'), default='primary')
    run.add_argument('--depth-size', type=int, nargs=2, default=(848, 480))
    run.add_argument('--fps', type=int, default=15)
    run.add_argument('--count', type=int, default=10)
    run.add_argument('--output', type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    def terminate(_signum, _frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        result = prepare_plan(args) if args.command == 'plan' else run_scan(args)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print('Scene capture interrupted; partial observations retained and no automatic return.', file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        print(f'Scene capture failed: {error}', file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
