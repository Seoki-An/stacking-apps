"""Identify separated field stones and export world-frame planning pickup poses."""

import argparse
from dataclasses import replace
import json
import hashlib
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial.transform import Rotation

from stacking_common.pointcloud import Open3dRegistrationBackend
from stacking_common.artifacts.planning_context import _values
from stacking_tabletop.config import load_inventory_profile, load_perception_profile
from stacking_tabletop.acquisition.calibration import BASE_LINK_T_UR_BASE
from .assignment import TabletopFieldMatcher
from .catalog import TabletopPcdCatalog
from .fusion import load_recorded_femto_scan
from .field_source import resolve_field_capture_directory
from .segmentation import TabletopFieldSegmenter


def add_identification_arguments(parser, *, include_output=True):
    parser.add_argument('--repository', type=Path, default=Path.cwd())
    parser.add_argument('--perception-config', type=Path)
    parser.add_argument('--stone-directory', type=Path, help='directory containing measured model_<id>.pcd files')
    if include_output:
        parser.add_argument('--identification-output', type=Path, help='new result directory for re-identifying an existing capture')
    parser.add_argument('--pick-table-height', type=float, help='designated pickup height in metres; default: perception profile')
    parser.add_argument('--automatic-ordering', action='store_true', help='match all models to clusters without the ID assignment window')
    parser.add_argument('--manual-init-gui', action='store_true')
    parser.add_argument('--manual-init-stones', type=int, nargs='+')
    parser.add_argument('--manual-init-translation-step', type=float, default=.005)
    parser.add_argument('--manual-init-rotation-step', type=float, default=5.)
    parser.add_argument('--visualize', action='store_true')


def load_inputs(args):
    import open3d  # Check the processing dependency before starting a camera.
    repository = args.repository.expanduser().resolve()
    profile = load_perception_profile(args.perception_config or repository / 'apps/tabletop/config/perception.yml')
    if args.pick_table_height is not None:
        profile = replace(profile, segmentation=replace(profile.segmentation,
                          designated_table_height=args.pick_table_height))
    for value in (args.manual_init_translation_step, args.manual_init_rotation_step):
        if not np.isfinite(value) or value <= 0:
            raise ValueError('manual initialization increments must be positive and finite')
    if args.stone_directory is None:
        inventory = load_inventory_profile(repository / 'apps/tabletop/config/inventory.yml')
        directory = repository / inventory.model_directory
    else:
        directory = args.stone_directory.expanduser().resolve()
    ids = sorted(int(path.stem.removeprefix('model_')) for path in directory.glob('model_*.pcd')
                 if path.stem.removeprefix('model_').isdigit())
    catalog = TabletopPcdCatalog(directory, ids)
    if args.manual_init_stones is not None and not set(args.manual_init_stones).issubset(ids):
        raise ValueError('manual initialization IDs must exist in the selected model directory')
    return profile, catalog


def _cloud(points):
    import open3d as o3d
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.array(points, copy=True))
    return cloud


def _write_cloud(path, cloud):
    import open3d as o3d
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise OSError(f'could not save {path}')


def identify_field(args, scan_directory, *, ordering=None, initializer=None):
    """Use migrated segmentation/registration; raw captures remain untouched."""
    import open3d as o3d
    profile, catalog = load_inputs(args)
    scan_directory = resolve_field_capture_directory(scan_directory)
    print(f'Using field capture: {scan_directory}', flush=True)
    scan = load_recorded_femto_scan(scan_directory)
    if scan.frame_id != 'base_link':
        raise ValueError('field identification requires a base_link capture')
    # Planning world uses UR base axes, not the ROS base_link axes.
    scene = scan.cloud.transformed(BASE_LINK_T_UR_BASE)
    segmented = TabletopFieldSegmenter(profile.segmentation).segment(scene)
    if not segmented.clusters:
        raise ValueError('no separated stone clusters found in the field')
    result_root = scan_directory
    if args.identification_output is not None:
        result_root = args.identification_output.expanduser().resolve()
        result_root.mkdir(parents=True, exist_ok=False)
    output = result_root / 'pose_identification'
    pose_path = result_root / 'pose_data.pkl'
    if pose_path.exists():
        raise FileExistsError(f'pickup poses already exist: {pose_path}')
    output.mkdir(exist_ok=False)
    print(f'Table: measured {segmented.measured_table_height:.4f} m, '
          f'designated {segmented.designated_table_height:.4f} m, '
          f'Z correction {segmented.height_correction:+.4f} m', flush=True)
    clusters = [_cloud(cluster.points) for cluster in segmented.clusters]
    _write_cloud(output / 'scene_cropped.pcd', _cloud(segmented.cropped.points))
    _write_cloud(output / 'stone_points.pcd', _cloud(segmented.object_points.points))
    for index, cluster in enumerate(clusters):
        _write_cloud(output / f'cluster_{index:02d}.pcd', cluster)
    if args.visualize:
        o3d.visualization.draw_geometries([_cloud(segmented.cropped.points)], window_name='Filtered field in planning world')
    matcher = TabletopFieldMatcher(catalog, Open3dRegistrationBackend(profile.field_registration),
                                  profile.field_quality, profile.assignment)
    candidates = {}
    if args.automatic_ordering:
        print(f'Matching {len(catalog.object_ids)} models against {len(clusters)} clusters…', flush=True)
        result = matcher.assign(catalog.object_ids, segmented.clusters, segmented.support_plane)
        assignments = result.assignments
        candidates = {(cluster, catalog.object_ids[model]): candidate
                      for (model, cluster), candidate in result.candidates.items()}
    else:
        if ordering is None:
            from stacking_tabletop.gui.field_stone_ordering_gui import assign_cluster_ids_gui
            ordering = assign_cluster_ids_gui
        assignments = ordering(clusters, list(catalog.object_ids), {})
    if not assignments:
        raise ValueError('no stone IDs were assigned or accepted')
    if (len(set(assignments.values())) != len(assignments)
            or any(index not in range(len(clusters)) or object_id not in catalog.object_ids
                   for index, object_id in assignments.items())):
        raise ValueError('cluster assignments must use unique known stone IDs and valid cluster indices')
    models = {object_id: _cloud(catalog.load(object_id).points) for object_id in assignments.values()}
    seeds = {}
    if args.manual_init_gui or args.manual_init_stones is not None:
        if initializer is None:
            from stacking_tabletop.gui.field_stone_manual_init_gui import run_manual_initial_pose_gui
            initializer = run_manual_initial_pose_gui
        seeds = initializer(assignments, clusters, models, segmented.support_plane,
            target_stone_ids=None if args.manual_init_stones is None else set(args.manual_init_stones),
            translation_step_m=args.manual_init_translation_step,
            rotation_step_deg=args.manual_init_rotation_step)
    poses, rejected = [], []
    review = [_cloud(segmented.cropped.points)]
    for cluster_index, object_id in sorted(assignments.items()):
        print(f'Identifying stone {object_id} in cluster {cluster_index}…', flush=True)
        candidate = candidates.get((cluster_index, object_id))
        if candidate is None or object_id in seeds:
            candidate = matcher.register_best(object_id, segmented.clusters[cluster_index],
                segmented.support_plane, initial_transform=seeds.get(object_id))
        if not candidate.decision.accepted:
            result = candidate.registration
            decision = candidate.decision
            rejected.append({'stone_id': object_id, 'cluster_index': cluster_index,
                             'reasons': list(decision.reasons),
                             'source_coverage': result.source_coverage,
                             'target_coverage': result.target_coverage,
                             'rmse_m': result.rmse,
                             'correction_translation_m': decision.correction_translation,
                             'correction_rotation_degrees': decision.correction_rotation_degrees,
                             'initial_transform': candidate.initial_transform.tolist(),
                             'planning_T_model': result.transform.tolist()})
            quality = profile.field_quality
            print(f'Rejected stone {object_id} in cluster {cluster_index}: '
                  f'{", ".join(decision.reasons)}. '
                  f'RMSE {result.rmse * 1000:.2f} mm '
                  f'(maximum {quality.maximum_rmse * 1000:.2f}); '
                  f'target coverage {result.target_coverage:.1%} '
                  f'(minimum {quality.minimum_target_coverage:.1%}); '
                  f'source coverage {result.source_coverage:.1%} '
                  f'(minimum {quality.minimum_source_coverage:.1%}).', flush=True)
            continue
        result = candidate.registration
        pose = np.r_[result.transform[:3, 3], Rotation.from_matrix(result.transform[:3, :3]).as_quat()]
        poses.append({'stone_id': object_id, 'cluster_index': cluster_index, 'pose_xyzw': pose.tolist(),
                      'planning_T_model': result.transform.tolist(), 'score': candidate.score,
                      'source_coverage': result.source_coverage, 'target_coverage': result.target_coverage,
                      'rmse_m': result.rmse, 'model_file': str(catalog.path(object_id)),
                      'model_sha256': hashlib.sha256(catalog.path(object_id).read_bytes()).hexdigest()})
        aligned = _cloud(catalog.load(object_id).points).transform(result.transform)
        _write_cloud(output / f'stone_{object_id}_aligned.pcd', aligned)
        aligned.paint_uniform_color([1., .35, .05])
        review.append(aligned)
    report = {'version': 1, 'frame_id': 'world', 'source_frame_id': scan.frame_id,
              'planning_T_base_link': BASE_LINK_T_UR_BASE.tolist(), 'source_directory': str(scan_directory),
              'perception_profile': _values(profile), 'capture_metadata': scan.metadata,
              'depth_processing': 'already_corrected_capture', 'poses': poses, 'rejected': rejected,
              'cluster_count': len(clusters), 'cluster_assignments': assignments,
              'table_plane': segmented.support_plane.tolist(),
              'fitted_table_plane': segmented.fitted_plane.tolist(),
              'table_height_correction_m': segmented.height_correction,
              'status': 'complete' if poses and not rejected else 'failed'}
    (output / 'stone_poses.json').write_text(json.dumps(report, indent=2) + '\n')
    if rejected or not poses:
        rejected_ids = ', '.join(str(item['stone_id']) for item in rejected)
        raise ValueError(f'field identification rejected assigned stones [{rejected_ids}]; '
                         f'inspect {output / "stone_poses.json"}. '
                         'Retry Identify existing capture with manual alignment after verifying stone IDs.')
    with pose_path.open('xb') as stream:
        pickle.dump({item['stone_id']: (np.array(item['pose_xyzw'][:3]), np.array(item['pose_xyzw'][3:]))
                     for item in poses}, stream, protocol=pickle.HIGHEST_PROTOCOL)
    if args.visualize:
        o3d.visualization.draw_geometries(review, window_name='Identified field stones')
    print(f'Saved {len(poses)} pickup poses: {pose_path}', flush=True)
    return pose_path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scan_dir', type=Path)
    add_identification_arguments(parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        identify_field(args, args.scan_dir)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f'Field identification failed: {error}', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
