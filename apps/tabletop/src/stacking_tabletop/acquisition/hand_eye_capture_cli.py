"""Inspect D435 profiles, verify live RGB-D, or record one stationary wrist view."""

import argparse
import json
from pathlib import Path
import sys

from .hand_eye import HandEyeCalibration
from .hand_eye_capture import atomic_json, capture_view, inspect_capture, preflight
from .realsense import RealSenseCamera, list_profiles


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=Path('apps/tabletop/config/hand_eye.yml'))
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('profiles', help='list native modes without starting streams')
    for name in ('preflight', 'capture'):
        command = commands.add_parser(name)
        command.add_argument('--depth-size', type=int, nargs=2, default=(848, 480), metavar=('WIDTH', 'HEIGHT'))
        command.add_argument('--fps', type=int, default=15)
        command.add_argument('--count', type=int, default=10)
        command.add_argument('--output', type=Path, required=True,
                             help='new output directory; never overwrite existing data')
    commands.add_parser('inspect', help='verify raw capture offline').add_argument('directory', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'inspect':
            result = inspect_capture(args.directory)
        else:
            calibration = HandEyeCalibration.load(args.profile)
            if args.command == 'profiles':
                result = list_profiles(calibration)
            else:
                if args.count < 1 or args.fps < 1 or min(args.depth_size) < 1:
                    raise ValueError('count, fps and image dimensions must be positive')
                if args.output.exists():
                    raise ValueError('output directory already exists')
                if args.command == 'preflight':
                    with RealSenseCamera(calibration, depth_size=args.depth_size, fps=args.fps) as camera:
                        result = preflight(camera, count=args.count)
                    args.output.mkdir(parents=True, exist_ok=False)
                    calibration.snapshot(args.output / 'calibration')
                    atomic_json(args.output / 'preflight.json', result)
                else:
                    # Lazy transport import: profiles/preflight/inspect need no ROS.
                    from stacking_tabletop.ros.hand_eye import HandEyeRobot
                    try:
                        with HandEyeRobot(calibration) as robot:
                            with RealSenseCamera(calibration, depth_size=args.depth_size, fps=args.fps) as camera:
                                result = capture_view(args.output, calibration, camera, robot, count=args.count)
                    except BaseException as error:
                        manifest_path = args.output / 'manifest.json'
                        if manifest_path.exists():
                            failed = json.loads(manifest_path.read_text())
                            failed.update(status='incomplete', error=f'{type(error).__name__}: {error}')
                            atomic_json(manifest_path, failed)
                        raise
                    result = dict(status=result['status'], frame_count=len(result['frames']),
                                  directory=str(args.output), accuracy_1mm_verified=False)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print('Wrist capture interrupted; any partial view is incomplete.', file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f'Wrist capture failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
