"""Audit a pinned wrist calibration and optionally create a portable snapshot."""

import argparse
import json
from pathlib import Path
import sys

from .hand_eye import HandEyeCalibration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=Path('apps/tabletop/config/hand_eye.yml'))
    parser.add_argument('--compare', type=Path, action='append', default=[],
                        help='diagnose an alternative hand-eye export without selecting it')
    parser.add_argument('--output', type=Path, help='new directory for immutable calibration snapshots')
    args = parser.parse_args(argv)
    try:
        calibration = HandEyeCalibration.load(args.profile)
        report = calibration.report()
        comparisons = [calibration.compare(path) for path in args.compare]
        report['comparisons'] = comparisons
        if args.output is not None:
            report['snapshot_profile'] = str(calibration.snapshot(args.output, comparisons=comparisons))
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f'Hand-eye calibration audit failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
