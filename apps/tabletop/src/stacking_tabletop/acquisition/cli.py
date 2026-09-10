"""Capture a fixed field or one already-settled in-hand view from FEMTO."""

import argparse
from pathlib import Path
import threading

from stacking_tabletop.config.femto import load_femto_profile
from .femto import FemtoCalibration, FemtoCamera
from .storage import CaptureSession


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('apps/tabletop/config/femto.yml'))
    parser.add_argument('--camera', choices=('primary', 'secondary'), default='primary')
    parser.add_argument('--output', type=Path, required=True, help='new capture directory; existing directories are refused')
    parser.add_argument('--in-hand', action='store_true', help='record one stationary view with exact tool TF and timestamp-matched joints/gripper; sends no motion')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        profile = load_femto_profile(args.config, args.camera)
        calibration = FemtoCalibration(profile)
        # Validate calibration and reserve output before creating a ROS node.
        with CaptureSession(args.output, calibration, in_hand=args.in_hand) as session:
            from stacking_tabletop.ros.femto import FemtoRosTransport
            with FemtoRosTransport(calibration) as transport:
                camera = FemtoCamera(transport, calibration, cancel_event=threading.Event())
                session.add(camera.capture(in_hand=args.in_hand,
                    progress=lambda done, total: print(f'FEMTO {done}/{total}', flush=True)))
        print(session.directory)
        return 0
    except KeyboardInterrupt:
        print('FEMTO capture interrupted; incomplete data retained.')
        return 130
    except Exception as error:
        print(f'FEMTO capture failed: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
