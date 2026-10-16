"""Preview and execute a wrist scan bracketed by fixed FEMTO captures."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import signal
import sys

import numpy as np
import yaml

from stacking_tabletop.execution.kinematics import UR5eKinematics
from .hand_eye import HandEyeCalibration
from .hand_eye_capture import atomic_json, digest, inspect_capture
from .scene_capture import capture_lock, execute_capture
from .scene_views import build_capture_route, camera_pose, kinematic_urdf, render_preview, validate_views

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
    capture_config = yaml.safe_load((ROOT / 'apps/tabletop/config/scene_capture.yml').read_text())
    capture_init = np.deg2rad(np.asarray(capture_config['q_capture_init_deg'], dtype=float))
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
    request = dict(start_joints=np.asarray(seed).tolist(), seed_joints=capture_init.tolist(),
                   q_capture_init_deg=capture_config['q_capture_init_deg'], distance_m=args.distance,
                   offset_m=args.offset, oblique_offset_m=args.oblique_offset,
                   viewpoint_pattern=getattr(args, 'viewpoint_pattern', 'coverage9'),
                   source=source, motion_sent=False, status='planning')
    atomic_json(directory / 'planning.json', request)
    from .scene_views import camera_view_targets
    from stacking_common.multiview.coverage import plane_coverage
    from stacking_common.rendering.capture_coverage import coverage_summary, render_coverage
    poses, reference, target = camera_view_targets(kinematics, calibration, capture_init,
        distance=args.distance, offset=args.offset, oblique_offset=args.oblique_offset,
        pattern=request['viewpoint_pattern'])
    coverage = plane_coverage(poses, calibration.color_matrix, calibration.color_distortion,
        (calibration.width, calibration.height), target, reference[:3, :3])
    coverage.update(distance_m=args.distance, frame='base_link', route_status='not_validated',
                    camera_pose_source='requested_optical_targets',
                    reference_plane='perpendicular_to_seed_optical_axis_at_assumed_target_depth')
    def save_coverage():
        atomic_json(directory / 'coverage.json', coverage)
        render_coverage(coverage, directory / 'coverage.html')
    save_coverage()
    print(coverage_summary(coverage), flush=True)
    print(f'Capture region preview: {directory / "coverage.html"}', flush=True)
    try:
        plan = build_capture_route(kinematics, calibration, seed, capture_init, distance=args.distance,
                           offset=args.offset, oblique_offset=args.oblique_offset,
                           pattern=request['viewpoint_pattern'])
    except (ValueError, KeyboardInterrupt) as error:
        request.update(status='cancelled' if isinstance(error, KeyboardInterrupt) else 'failed',
                       error=str(error))
        atomic_json(directory / 'planning.json', request)
        coverage.update(route_status=request['status'], route_error=str(error))
        save_coverage()
        raise
    request['status'] = 'complete'
    atomic_json(directory / 'planning.json', request)
    coverage['route_status'] = 'validated'
    save_coverage()
    plan.update(robot_urdf_sha256=digest(urdf), source=source,
                hand_eye_profile_sha256=digest(directory / 'calibration/profile.yml'))
    render_preview(plan, kinematics, calibration, directory / 'preview.html')
    atomic_json(directory / 'plan.json', plan)
    return dict(plan=str(directory), preview=str(directory / 'preview.html'),
                coverage_preview=str(directory / 'coverage.html'),
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
    from stacking_tabletop.perception.service_client import scene_wrist
    from stacking_tabletop.ros.femto import FemtoRosTransport

    args.plan = args.plan.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    wrist_only = getattr(args, 'sensor_mode', 'wrist-femto') == 'wrist-only'
    if args.output.is_relative_to(args.plan) or args.plan.is_relative_to(args.output):
        raise ValueError('plan and capture output must be separate directories')
    if args.output.exists():
        raise ValueError('capture output already exists')
    if args.count < 1 or args.fps < 1 or min(args.depth_size) < 1:
        raise ValueError('count, fps and depth dimensions must be positive')
    plan, calibration, kinematics = load_reviewed_plan(args.plan, args.approve)
    femto = None
    if not wrist_only:
        args.femto_config = args.femto_config.expanduser().resolve()
        femto = FemtoCalibration(load_femto_profile(args.femto_config, args.camera))
    if femto is not None and femto.profile.base_frame != 'base_link':
        raise ValueError('combined scene capture requires FEMTO observations in base_link')
    with capture_lock(), connected_arm(kinematics) as (arm, transport):
        live_hash = hashlib.sha256(robot_description(transport).encode()).hexdigest()
        if live_hash != plan['robot_urdf_sha256']:
            raise ValueError('running robot_description differs from the reviewed plan; prepare a new plan')
        if femto is not None and transport.node.count_publishers(femto.profile.namespace + '/depth/image_raw'):
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
            with scene_wrist(arm, calibration, ROOT, depth_size=args.depth_size,
                             fps=args.fps, count=args.count) as capture:
                yield capture

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

        return execute_capture(plan, args.output, arm, None if wrist_only else femto_capture, wrist_context,
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
    plan.add_argument('--viewpoint-pattern', choices=('coverage9', 'coverage13', 'coverage17'),
                      default='coverage9', help='coverage13 adds 30-degree side views; coverage17 also adds 45/60-degree left/right views; review path clearance')
    plan.add_argument('--seed-view', type=Path, help='offline starting/return pose from a recorded view instead of live feedback; capture center comes from scene_capture.yml')
    plan.add_argument('--urdf', type=Path, help='verified URDF for offline --seed-view planning')
    plan.add_argument('--output', type=Path, required=True)
    run = commands.add_parser('run', help='MOVE ROBOT through a reviewed plan and capture both sensors')
    run.add_argument('--plan', type=Path, required=True)
    run.add_argument('--approve', required=True, help='SHA256 of the previewed plan; confirms empty tool and full-path clearance review')
    run.add_argument('--femto-config', type=Path, default=ROOT / 'apps/tabletop/config/femto.yml')
    run.add_argument('--sensor-mode', choices=('wrist-femto', 'wrist-only'), default='wrist-femto')
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
