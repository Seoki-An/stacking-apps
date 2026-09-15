"""Capture, filter, optionally fuse, and identify a fixed FEMTO field."""

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
from uuid import uuid4

from stacking_tabletop.config.femto import load_femto_profile
from stacking_tabletop.perception.field_cli import add_identification_arguments, identify_field, load_inputs
from stacking_tabletop.perception.fusion import DualFemtoFusion, write_dual_femto_fusion
from .femto import FemtoCalibration, FemtoCamera
from .storage import CaptureSession


@dataclass(frozen=True)
class FieldWorkflowResult:
    capture_dir: Path
    pose_input_dir: Path
    pose_data: Path


@contextmanager
def camera_driver(repository, config, camera, log_path, *, process_factory=subprocess.Popen):
    """Own only this launch process group; never stop an existing driver."""
    ros2 = shutil.which('ros2')
    if ros2 is None:
        raise RuntimeError('source ROS Jazzy and ~/ros2_ws/install/setup.bash before capture')
    command = [sys.executable, ros2, 'launch',
               str(repository / 'apps/tabletop/launch/femto.launch.py'),
               f'config:={config}', f'camera:={camera}']
    with Path(log_path).open('w') as log:
        process = process_factory(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            yield process
            if process.poll() is not None:
                raise RuntimeError(f'camera driver exited during capture; inspect {log_path}')
        finally:
            # Include the launch's component container, even if the parent died.
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=5)
                    # A dead launch parent can leave its component container.
                    # Escalate for surviving members of our own process group.
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        break
                except subprocess.TimeoutExpired:
                    continue
            process.wait(timeout=5)


def capture_field(args, camera, calibration, directory, log_path):
    driver = nullcontext() if args.use_running_camera else camera_driver(
        args.repository, args.femto_config, camera, log_path)
    with driver as process:
        def check_driver():
            if process is not None and process.poll() is not None:
                raise RuntimeError(f'camera driver exited; inspect {log_path}')
        from stacking_tabletop.ros.femto import FemtoRosTransport
        with CaptureSession(directory, calibration) as session:
            with FemtoRosTransport(calibration) as transport:
                sensor = FemtoCamera(transport, calibration, cancel_event=threading.Event())
                session.add(sensor.capture(check=check_driver, progress=lambda done, total:
                    print(f'{camera}: FEMTO {done}/{total}', flush=True)))
    return directory


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_identification_arguments(parser)
    parser.add_argument('--femto-config', type=Path)
    parser.add_argument('--camera', choices=('primary', 'secondary'), default='primary')
    parser.add_argument('--both-cameras', action='store_true', help='capture sequentially and register secondary to primary')
    parser.add_argument('--use-running-camera', action='store_true', help='single-camera capture using an externally owned driver')
    parser.add_argument('--output-dir', type=Path, help='new workflow directory; existing directories are refused')
    return parser


def run_workflow(args, *, capture_runner=capture_field, pose_runner=identify_field, fusion_factory=DualFemtoFusion):
    if args.both_cameras and (args.use_running_camera or args.camera != 'primary'):
        raise ValueError('--both-cameras requires owned drivers and primary as the reference camera')
    args.repository = args.repository.expanduser().resolve()
    args.femto_config = (args.femto_config or args.repository / 'apps/tabletop/config/femto.yml').expanduser().resolve()
    profile, _ = load_inputs(args)  # Fail bad models/options before launching a camera.
    cameras = ('primary', 'secondary') if args.both_cameras else (args.camera,)
    calibrations = {camera: FemtoCalibration(load_femto_profile(args.femto_config, camera)) for camera in cameras}
    if len({value.profile.serial for value in calibrations.values()}) != len(cameras):
        raise ValueError('two-camera capture requires distinct device serials')
    for calibration in calibrations.values():
        if calibration.profile.base_frame != 'base_link':
            raise ValueError('field pose identification requires base_link captures')
    output = (args.output_dir or args.repository / '.data/field_pcd' /
              f'field_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}').expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = {'status': 'running', 'cameras': list(cameras)}
    manifest = output / 'field_workflow.json'

    def save_status():
        manifest.write_text(json.dumps(status, indent=2) + '\n')
        print(f'Field workflow: {status.get("stage", "starting")} ({status["status"]})', flush=True)

    save_status()
    try:
        for camera in cameras:
            status['stage'] = f'capture_{camera}'
            save_status()
            capture_runner(args, camera, calibrations[camera], output / camera, output / f'{camera}_driver.log')
        pose_input = output / cameras[0]
        if args.both_cameras:
            status['stage'] = 'fusion'
            save_status()
            result = fusion_factory(profile.dual_femto_fusion, profile.segmentation).fuse_recorded(
                output / 'primary', output / 'secondary')
            pose_input = output / 'merged'
            write_dual_femto_fusion(pose_input, result, tuple(calibrations[c].profile.serial for c in cameras))
        status['stage'] = 'identification'
        save_status()
        pose_data = Path(pose_runner(args, pose_input))
        if not pose_data.is_file():
            raise FileNotFoundError(f'identification did not produce pickup poses: {pose_data}')
        status.update(status='complete', pose_input_dir=str(pose_input), pose_data=str(pose_data))
        save_status()
        return FieldWorkflowResult(output, pose_input, pose_data)
    except BaseException as error:
        status.update(status='cancelled' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        save_status()
        raise


def main(argv=None):
    args = build_parser().parse_args(argv)
    def terminate(_signum, _frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        result = run_workflow(args)
        print(f'Field workflow complete. Planning pickup poses: {result.pose_data}', flush=True)
        return 0
    except KeyboardInterrupt:
        print('Field workflow cancelled; partial data retained.', flush=True)
        return 130
    except Exception as error:
        print(f'Field workflow failed: {error}', flush=True)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
