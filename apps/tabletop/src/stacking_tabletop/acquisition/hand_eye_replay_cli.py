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
    diagnose = commands.add_parser('diagnose', help='compare uncorrected table heights in common XY cells')
    diagnose.add_argument('session', type=Path)
    diagnose.add_argument('--region', type=Path, required=True, help='base_link table-region JSON')
    diagnose.add_argument('--output', type=Path, required=True)
    diagnose.add_argument('--no-visuals', action='store_true')
    rgb = commands.add_parser('rgb', help='test fixed-pose RGB triangulation against FEMTO')
    rgb.add_argument('session', type=Path)
    rgb.add_argument('--region', type=Path, required=True, help='base_link ROI, stone boxes and support plane JSON')
    rgb.add_argument('--output', type=Path, required=True)
    refine = commands.add_parser('rgb-refine', help='refine RGB camera poses with metric priors and unused-view checks')
    refine.add_argument('session', type=Path)
    refine.add_argument('--baseline', type=Path, required=True, help='completed rgb experiment directory')
    refine.add_argument('--output', type=Path, required=True)
    dense = commands.add_parser('rgb-dense', help='compare dense RGB MVS using original and refined camera poses')
    dense.add_argument('session', type=Path)
    dense.add_argument('--baseline', type=Path, required=True)
    dense.add_argument('--refinement', type=Path, required=True)
    dense.add_argument('--output', type=Path, required=True)
    confidence = commands.add_parser('rgb-filter', help='filter dense RGB with disjoint-pair support and classify reference coverage')
    confidence.add_argument('session', type=Path)
    confidence.add_argument('--dense', type=Path, required=True, help='dense experiment including independent_pairs control')
    confidence.add_argument('--output', type=Path, required=True)
    scene = commands.add_parser('scene-assemble', help='assemble portable FEMTO and filtered RGB evidence layers')
    scene.add_argument('session', type=Path)
    scene.add_argument('--filtered', type=Path, required=True)
    scene.add_argument('--output', type=Path, required=True)
    scene.add_argument('--no-visuals', action='store_true')
    commands.add_parser('scene-inspect', help='verify a portable layered scene').add_argument('scene', type=Path)
    scene_replay = commands.add_parser('scene-replay', help='render a portable layered scene')
    scene_replay.add_argument('scene', type=Path)
    scene_replay.add_argument('--output', type=Path, required=True)
    surface = commands.add_parser('surface-check', help='compare stone interiors before and after a shared rigid diagnostic fit')
    surface.add_argument('scene', type=Path)
    surface.add_argument('--output', type=Path, required=True)
    catalog = commands.add_parser('catalog-check', help='compare scene regions to measured stone PCDs at fixed scale')
    catalog.add_argument('scene', type=Path)
    catalog.add_argument('--models', type=Path, required=True)
    catalog.add_argument('--stone-ids', type=int, nargs='+', required=True)
    catalog.add_argument('--output', type=Path, required=True)
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
        elif args.command == 'diagnose':
            from .hand_eye_diagnostics import diagnose_session
            result = diagnose_session(args.session, args.region, args.output, visualize=not args.no_visuals)
        elif args.command == 'rgb':
            from .hand_eye_rgb import rgb_experiment
            result = rgb_experiment(args.session, args.region, args.output)
        elif args.command == 'rgb-refine':
            from .hand_eye_rgb_refinement import refine_rgb_session
            result = refine_rgb_session(args.session, args.baseline, args.output)
        elif args.command == 'rgb-dense':
            from .hand_eye_dense import dense_rgb_session
            result = dense_rgb_session(args.session, args.baseline, args.refinement, args.output)
        elif args.command == 'rgb-filter':
            from .hand_eye_confidence import filter_dense_session
            result = filter_dense_session(args.session, args.dense, args.output)
        elif args.command == 'scene-assemble':
            from .hand_eye_scene import assemble_scene
            result = assemble_scene(args.session, args.filtered, args.output, visualize=not args.no_visuals)
        elif args.command == 'scene-inspect':
            from .hand_eye_scene import inspect_scene
            result = inspect_scene(args.scene)
        elif args.command == 'scene-replay':
            from .hand_eye_scene import replay_scene
            result = replay_scene(args.scene, args.output)
        elif args.command == 'surface-check':
            from .hand_eye_surface_check import check_scene_surfaces
            result = check_scene_surfaces(args.scene, args.output)
        elif args.command == 'catalog-check':
            from .hand_eye_catalog_check import compare_scene_catalog
            result = compare_scene_catalog(args.scene, args.models, args.stone_ids, args.output)
        else:
            from .hand_eye_synthetic import generate_demo
            result = generate_demo(args.output, visualize=not args.no_visuals)
        print(json.dumps({key: result[key] for key in ('status', 'data_origin', 'point_count', 'common_cell_count', 'accuracy_1mm_verified')
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
