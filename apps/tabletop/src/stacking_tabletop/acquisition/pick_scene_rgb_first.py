"""Coarse FEMTO regions, then first catalog fits from wrist RGB plus FEMTO."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial.transform import Rotation

from stacking_common.pointcloud import Open3dRegistrationBackend, PointCloudData, evaluate_registration
from stacking_tabletop.acquisition.calibration import BASE_LINK_T_UR_BASE
from stacking_tabletop.perception.assignment import TabletopFieldMatcher
from stacking_tabletop.perception.fusion import load_recorded_femto_scan
from stacking_tabletop.perception.segmentation import TabletopFieldSegmenter
from .hand_eye_capture import atomic_json, digest
from .pick_scene_fusion import FusedPoseRejected, _balanced_observation, _metric, _read_cloud, _transform
from .storage import write_xyz_pcd


def _json(path):
    return json.loads(Path(path).read_text())


def _planning_T_base(geometry):
    correction = np.eye(4)
    correction[2, 3] = geometry['table_height_correction_m']
    return correction @ np.asarray(geometry['planning_T_base_link'], dtype=float)


def prepare_femto_regions(capture, profile, output, region_file, *, padding=.01):
    """Segment the reference scan and save MVS masks; no model is fitted here."""
    capture, output, region_file = map(lambda p: Path(p).resolve(), (capture, output, region_file))
    scan_dir = capture / 'femto_before'
    scan = load_recorded_femto_scan(scan_dir)
    if scan.frame_id != 'base_link':
        raise ValueError('FEMTO reference must be in base_link')
    segmented = TabletopFieldSegmenter(profile.segmentation).segment(
        scan.cloud.transformed(BASE_LINK_T_UR_BASE))
    if not segmented.clusters:
        raise ValueError('FEMTO found no coarse stone regions')
    output.mkdir(parents=True, exist_ok=False)
    geometry = dict(schema_version=1, kind='femto_pick_scene_geometry', status='complete',
                    frame_id='world', source_frame_id='base_link',
                    planning_T_base_link=BASE_LINK_T_UR_BASE.tolist(),
                    table_height_correction_m=float(segmented.height_correction),
                    table_plane=segmented.support_plane.tolist(),
                    fitted_table_plane=segmented.fitted_plane.tolist(),
                    cluster_count=len(segmented.clusters),
                    source_sha256={'field_scan.json': digest(scan_dir / 'field_scan.json')},
                    catalog_pose_fitted=False, accuracy_1mm_verified=False)
    planning_T_base = _planning_T_base(geometry)
    base_T_planning = np.linalg.inv(planning_T_base)
    fitted_base = planning_T_base.T @ segmented.fitted_plane
    support = np.array([-fitted_base[0]/fitted_base[2],
                        -fitted_base[1]/fitted_base[2], -fitted_base[3]/fitted_base[2]])
    boxes, all_points, stones = [], [], []
    for index, cluster in enumerate(segmented.clusters):
        write_xyz_pcd(output / f'cluster_{index:02d}.pcd', cluster.points)
        points = _transform(cluster.points, base_T_planning)
        boxes.append([(points[:, :2].min(axis=0)-padding).tolist(),
                      (points[:, :2].max(axis=0)+padding).tolist()])
        all_points.append(points)
        stones.append(dict(cluster_index=index, stone_id=None))
    write_xyz_pcd(output / 'scene_cropped.pcd', segmented.cropped.points)
    metadata = _json(scan_dir / 'field_scan.json')
    crop = metadata.get('base_frame_crop', {})
    points = np.concatenate(all_points)
    low = np.asarray(crop.get('min', points.min(axis=0)-.05), dtype=float)
    high = np.asarray(crop.get('max', points.max(axis=0)+.05), dtype=float)
    low = np.minimum(low, points.min(axis=0)-[.02, .02, .02])
    high = np.maximum(high, points.max(axis=0)+[.02, .02, .02])
    region = dict(schema_version=1, kind='coarse_femto_pick_regions', frame='base_link',
                  minimum=low.tolist(), maximum=high.tolist(), stone_boxes_xy=boxes,
                  support_plane=support.tolist(), stones=stones,
                  source_sha256={'field_scan.json': digest(scan_dir / 'field_scan.json')},
                  limitations=['FEMTO clusters define projected RGB masks, not catalog identity or pose.'])
    atomic_json(output / 'geometry.json', geometry)
    atomic_json(region_file, region)
    return geometry


def _editable_open3d_cloud(points):
    """Copy immutable point-cloud data for Open3D's editable GUI geometry."""
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64, copy=True))
    return cloud


def render_joint_pose_result(identification):
    """Visualize saved RGB, FEMTO and fitted-model clouds for each accepted pose."""
    from stacking_common.rendering.pointcloud import render_clouds

    identification = Path(identification).resolve()
    report_dir = identification / 'pose_identification'
    report = _json(report_dir / 'stone_poses.json')
    if report.get('status') != 'complete' or not report.get('poses'):
        raise ValueError('a complete joint pose result is required for visualization')
    layers = {}
    object_poses = {}
    for item in report['poses']:
        stone = int(item['stone_id'])
        object_poses[f'Stone {stone} pose'] = np.asarray(item['planning_T_model'], dtype=float)
        layers[f'FEMTO stone {stone}'] = _read_cloud(report_dir / f'stone_{stone}_femto.pcd')
        layers[f'RGB stone {stone}'] = _read_cloud(report_dir / f'stone_{stone}_rgb.pcd')
        layers[f'Fitted model {stone}'] = _read_cloud(report_dir / f'stone_{stone}_aligned.pcd')
    viewer_dir = identification / 'comparison'
    render_clouds(viewer_dir, layers, {},
                  title='Joint RGB/FEMTO pickup pose: observations and fitted models',
                  frame='world', object_poses=object_poses)
    return viewer_dir / 'overview.html'


def identify_from_rgb_and_femto(dense, region_file, geometry_dir, catalog, profile, output, *,
                                automatic_ordering=False, manual_init_gui=False,
                                manual_init_stones=None, manual_init_translation_step=.005,
                                manual_init_rotation_step=5., visualize=False, assignments=None):
    """Make the first catalog pose fit against both measured point clouds."""
    dense, region_file, geometry_dir, output = map(lambda p: Path(p).resolve(),
                                                   (dense, region_file, geometry_dir, output))
    summary = _json(dense / 'summary.json')
    region = _json(region_file)
    geometry = _json(geometry_dir / 'geometry.json')
    if summary.get('status') != 'complete' or 'refined' not in summary.get('modes', {}):
        raise ValueError('complete RGB MVS is required before catalog identification')
    if summary.get('region') != region:
        raise ValueError('RGB MVS was reconstructed with different coarse FEMTO regions')
    if geometry.get('catalog_pose_fitted') is not False:
        raise ValueError('coarse FEMTO input must not contain a catalog pose fit')
    if automatic_ordering:
        raise ValueError('automatic RGB/FEMTO stone ID assignment is not validated; review cluster IDs manually')
    rgb_scene = _read_cloud(dense / 'refined/rgb_dense_voxel_1mm.pcd')
    plane = np.asarray(region['support_plane'], dtype=float)
    planning_T_base = _planning_T_base(geometry)
    voxel = profile.field_registration.voxel_sizes[-1]
    minimum = profile.field_registration.minimum_points
    observations = {}
    gui_clusters = []
    for index, box in enumerate(region['stone_boxes_xy']):
        low, high = np.asarray(box, dtype=float)
        selected = rgb_scene[:, 2] > rgb_scene[:, :2] @ plane[:2] + plane[2] + .008
        selected &= np.all((rgb_scene[:, :2] >= low) & (rgb_scene[:, :2] <= high), axis=1)
        rgb_base = rgb_scene[selected]
        femto = _read_cloud(geometry_dir / f'cluster_{index:02d}.pcd')
        rgb = _transform(rgb_base, planning_T_base)
        observations[index] = (femto, rgb)
        combined = np.concatenate((femto, rgb))
        gui_clusters.append(_editable_open3d_cloud(combined))
    if assignments is None:
        from stacking_tabletop.gui.field_stone_ordering_gui import assign_cluster_ids_gui
        assignments = assign_cluster_ids_gui(gui_clusters, list(catalog.object_ids), {})
    assignments = {int(cluster): int(stone) for cluster, stone in assignments.items()}
    if not assignments or len(set(assignments.values())) != len(assignments):
        raise ValueError('assign unique stone IDs to at least one RGB/FEMTO cluster')
    if (any(index not in observations for index in assignments)
            or any(stone not in catalog.object_ids for stone in assignments.values())):
        raise ValueError('stone assignments reference unknown clusters or catalog models')
    seeds = {}
    if manual_init_gui or manual_init_stones is not None:
        from stacking_tabletop.gui.field_stone_manual_init_gui import run_manual_initial_pose_gui
        models = {}
        for stone in assignments.values():
            models[stone] = _editable_open3d_cloud(catalog.load(stone).points)
        seeds = run_manual_initial_pose_gui(assignments, gui_clusters, models,
            np.asarray(geometry['table_plane']), target_stone_ids=None if manual_init_stones is None
            else set(manual_init_stones), translation_step_m=manual_init_translation_step,
            rotation_step_deg=manual_init_rotation_step)
    quality = replace(profile.field_quality, maximum_correction_translation=None,
                      maximum_correction_rotation_degrees=None)
    backend = Open3dRegistrationBackend(profile.field_registration)
    matcher = TabletopFieldMatcher(catalog, backend, quality, profile.assignment)
    output.mkdir(parents=True, exist_ok=False)
    report_dir = output / 'pose_identification'
    report_dir.mkdir()
    report = dict(version=2, kind='femto_rgb_joint_pose_identification', status='running',
                  pose_source='rgb_femto_first_joint_icp', frame_id='world',
                  source_frame_id='base_link',
                  planning_T_base_link=geometry['planning_T_base_link'],
                  table_height_correction_m=geometry['table_height_correction_m'],
                  table_plane=geometry['table_plane'],
                  fitted_table_plane=geometry['fitted_table_plane'],
                  source_sha256={'dense_summary.json': digest(dense / 'summary.json'),
                                 'rgb_region.json': digest(region_file),
                                 'femto_geometry.json': digest(geometry_dir / 'geometry.json')},
                  cluster_assignments=assignments, poses=[], rejected=[], warnings=[],
                  accuracy_1mm_verified=False)
    atomic_json(report_dir / 'stone_poses.json', report)
    try:
        for cluster_index, stone_id in sorted(assignments.items()):
            femto_raw, rgb_raw = observations[cluster_index]
            if len(rgb_raw) < minimum:
                report['rejected'].append(dict(stone_id=stone_id, cluster_index=cluster_index,
                                               reasons=['too_few_rgb_points'], rgb_input_points=len(rgb_raw)))
                continue
            femto, rgb, combined = _balanced_observation(femto_raw, rgb_raw, voxel, minimum)
            model = catalog.load(stone_id).points
            candidate = matcher.register_best(stone_id, PointCloudData(combined),
                                              np.asarray(geometry['table_plane']),
                                              initial_transform=seeds.get(stone_id))
            final = candidate.registration.transform
            results = dict(combined=candidate.registration,
                           femto=backend.evaluate(PointCloudData(model), PointCloudData(femto), final),
                           rgb=backend.evaluate(PointCloudData(model), PointCloudData(rgb), final))
            reasons = sorted({f'{name}_{reason}' for name, result in results.items()
                              for reason in evaluate_registration(result, candidate.initial_transform, quality).reasons})
            if reasons:
                report['rejected'].append(dict(stone_id=stone_id, cluster_index=cluster_index,
                                               reasons=reasons, candidate_planning_T_model=final.tolist(),
                                               sensors={name: _metric(result) for name, result in results.items()}))
                continue
            pose = np.r_[final[:3, 3], Rotation.from_matrix(final[:3, :3]).as_quat()]
            entry = dict(stone_id=stone_id, cluster_index=cluster_index, pose_xyzw=pose.tolist(),
                         planning_T_model=final.tolist(),
                         initial_joint_planning_T_model=candidate.initial_transform.tolist(),
                         rgb_input_points=len(rgb_raw), rgb_used_points=len(rgb),
                         femto_used_points=len(femto), balanced_points=len(combined),
                         sensors={name: _metric(result) for name, result in results.items()},
                         model_file=str(catalog.path(stone_id)),
                         model_sha256=hashlib.sha256(catalog.path(stone_id).read_bytes()).hexdigest())
            report['poses'].append(entry)
            write_xyz_pcd(report_dir / f'stone_{stone_id}_femto.pcd', femto)
            write_xyz_pcd(report_dir / f'stone_{stone_id}_rgb.pcd', rgb)
            write_xyz_pcd(report_dir / f'stone_{stone_id}_balanced_observation.pcd', combined)
            write_xyz_pcd(report_dir / f'stone_{stone_id}_aligned.pcd', _transform(model, final))
        report['status'] = 'complete' if report['poses'] and not report['rejected'] else 'failed'
        report['limitations'] = [
            'Catalog identity is reviewed against clusters containing both FEMTO and RGB points.',
            'The first catalog pose fit uses equally sampled FEMTO and RGB points without a FEMTO-only pose gate.',
            'FEMTO still supplies segmentation, table frame and metric scale.',
            'Passing field gates does not verify absolute 1 mm accuracy.',
        ]
        atomic_json(report_dir / 'stone_poses.json', report)
        if report['status'] != 'complete':
            raise FusedPoseRejected('joint RGB/FEMTO identification rejected assigned stones')
        pose_path = output / 'pose_data.pkl'
        with pose_path.open('xb') as stream:
            pickle.dump({item['stone_id']: (np.asarray(item['pose_xyzw'][:3]),
                                            np.asarray(item['pose_xyzw'][3:]))
                         for item in report['poses']}, stream, protocol=pickle.HIGHEST_PROTOCOL)
        render_joint_pose_result(output)
        return pose_path
    except BaseException as error:
        if report['status'] == 'running':
            report.update(status='failed', error=f'{type(error).__name__}: {error}')
            atomic_json(report_dir / 'stone_poses.json', report)
        raise


def validate_selected_femto_stability(capture, geometry_dir, region_file, identification,
                                      profile, output):
    """Compare FEMTO brackets only for clusters selected by the joint solve."""
    from .pick_scene_workflow import _fixed_pose_metrics
    capture, geometry_dir, region_file, identification, output = map(
        lambda p: Path(p).resolve(), (capture, geometry_dir, region_file, identification, output))
    geometry = _json(geometry_dir / 'geometry.json')
    region = _json(region_file)
    poses = _json(identification / 'pose_identification/stone_poses.json')['poses']
    after = load_recorded_femto_scan(capture / 'femto_after')
    if after.frame_id != 'base_link':
        raise ValueError('FEMTO after reference must be in base_link')
    base_T_planning = np.linalg.inv(_planning_T_base(geometry))
    points = after.cloud.points
    plane = np.asarray(region['support_plane'])
    threshold = profile.field_registration.evaluation_distance
    quality = profile.field_quality
    results = []
    retained = []
    output.mkdir(parents=True, exist_ok=False)
    for item in poses:
        index = int(item['cluster_index'])
        low, high = np.asarray(region['stone_boxes_xy'][index])
        selected = points[:, 2] > points[:, :2] @ plane[:2] + plane[2] + .008
        selected &= np.all((points[:, :2] >= low) & (points[:, :2] <= high), axis=1)
        observed = points[selected]
        before = _transform(_read_cloud(geometry_dir / f'cluster_{index:02d}.pcd'), base_T_planning)
        reasons = []
        if len(observed) < profile.field_registration.minimum_points:
            reasons.append('too_few_femto_after_points')
            metrics = dict(model_coverage=0., observed_coverage=0., rmse_m=None,
                           observed_median_mm=None, observed_p95_mm=None,
                           observed_points=len(observed), model_points=len(before))
        else:
            metrics = _fixed_pose_metrics(before, observed, np.eye(4), threshold)
            if metrics['observed_coverage'] < quality.minimum_target_coverage:
                reasons.append('low_femto_after_coverage')
            if metrics['model_coverage'] < quality.minimum_source_coverage:
                reasons.append('low_femto_before_coverage')
            if metrics['rmse_m'] is None or metrics['rmse_m'] > quality.maximum_rmse:
                reasons.append('high_femto_bracket_rmse')
        retained.append(observed)
        results.append(dict(stone_id=item['stone_id'], cluster_index=index,
                            stable=not reasons, reasons=reasons, **metrics))
    write_xyz_pcd(output / 'femto_after_stone_points.pcd',
                  np.concatenate(retained) if retained else np.empty((0, 3)))
    stable = all(item['stable'] for item in results)
    report = dict(schema_version=1, kind='femto_pick_scene_stability',
                  status='complete' if stable else 'rejected', scene_stable=stable,
                  output_frame='base_link', accuracy_1mm_verified=False,
                  after_sha256=digest(capture / 'femto_after/field_scan.json'),
                  region_sha256=digest(region_file), stones=results)
    atomic_json(output / 'summary.json', report)
    return report
