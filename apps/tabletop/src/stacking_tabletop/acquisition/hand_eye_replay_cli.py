"""Assemble and replay wrist observations without ROS or a camera SDK."""

import argparse
import json
from pathlib import Path
import sys

from .hand_eye_replay import assemble_session, inspect_session, replay_session


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    assemble = commands.add_parser('assemble')
    assemble.add_argument('--views', nargs='+', type=Path, required=True, help='captures in source-time order')
    assemble.add_argument('--femto-before', type=Path)
    assemble.add_argument('--femto-after', type=Path)
    assemble.add_argument('--reference', type=Path, help='optional independently supplied reference JSON')
    assemble.add_argument('--output', type=Path, required=True)
    commands.add_parser('inspect').add_argument('session', type=Path)
    replay = commands.add_parser('replay')
    replay.add_argument('session', type=Path)
    replay.add_argument('--output', type=Path, required=True)
    replay.add_argument('--no-visuals', action='store_true')
    demo = commands.add_parser('demo')
    demo.add_argument('--output', type=Path, required=True)
    demo.add_argument('--no-visuals', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'assemble':
            result = assemble_session(args.output, args.views, femto_before=args.femto_before,
                                      femto_after=args.femto_after, reference=args.reference)
        elif args.command == 'inspect':
            result = inspect_session(args.session)
        elif args.command == 'replay':
            result = replay_session(args.session, args.output, visualize=not args.no_visuals)
        else:
            from .hand_eye_synthetic import generate_demo
            result = generate_demo(args.output, visualize=not args.no_visuals)
        print(json.dumps({key: result[key] for key in ('status', 'data_origin', 'point_count', 'accuracy_1mm_verified')
                          if key in result}, indent=2))
        return 0
    except KeyboardInterrupt:
        print('Offline replay interrupted; check output completion status.', file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        print(f'Offline replay failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
