"""Capture a pick scene with wrist RGB and FEMTO, then publish reviewed pickup poses."""

import argparse
from copy import copy
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import signal
import time

import numpy as np
from scipy.spatial import cKDTree

from stacking_tabletop.config import load_perception_profile
from stacking_tabletop.perception.catalog import TabletopPcdCatalog
from stacking_tabletop.perception.field_cli import (
    add_identification_arguments,
    identify_field,
    load_inputs,
)
from stacking_tabletop.perception.fusion import load_recorded_femto_scan
from .hand_eye_capture import atomic_json, digest
from .hand_eye_replay import inspect_session
from .storage import write_xyz_pcd


ROOT = Path(__file__).resolve().parents[5]
STANDARD_DISTANCE_M = .4
STANDARD_NEAR_OFFSET_M = .04
STANDARD_OBLIQUE_OFFSET_M = .08


@dataclass(frozen=True)
class PickSceneWorkflowResult:
    root: Path
    capture: Path
    pose_data: Path
    validation: Path


def _json(path):
    return json.loads(Path(path).read_text())


def _corrected_planning_T_base(report):
    planning_T_base = np.asarray(report['planning_T_base_link'], dtype=float)
    if planning_T_base.shape != (4, 4):
        raise ValueError('identification planning/base transform is invalid')
    correction = np.eye(4)
    correction[2, 3] = float(report['table_height_correction_m'])
    return correction @ planning_T_base


def _read_cloud(path):
    import open3d as o3d
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=float)
    if not len(points):
        raise ValueError(f'point cloud is empty: {path}')
    return points


def build_rgb_region(capture, identification, output, *, padding=.01):
    """Derive base_link MVS masks from FEMTO clusters without changing their poses."""
    capture, identification, output = map(lambda value: Path(value).resolve(),
                                           (capture, identification, output))
    report_path = identification / 'pose_identification/stone_poses.json'
    report = _json(report_path)
    if report.get('status') != 'complete' or not report.get('poses'):
        raise ValueError('complete FEMTO stone identification is required before RGB reconstruction')
    if not np.isfinite(padding) or padding <= 0:
        raise ValueError('RGB stone-region padding must be positive')
    corrected_planning_T_base = _corrected_planning_T_base(report)
    base_T_corrected_planning = np.linalg.inv(corrected_planning_T_base)
    fitted = np.asarray(report['fitted_table_plane'], dtype=float)
    if fitted.shape != (4,) or not np.all(np.isfinite(fitted)):
        raise ValueError('identified support plane is invalid')
    base_plane = corrected_planning_T_base.T @ fitted
    if abs(base_plane[2]) < 1e-9:
        raise ValueError('identified support plane is vertical in base_link')
    support = np.array([-base_plane[0] / base_plane[2],
                        -base_plane[1] / base_plane[2],
                        -base_plane[3] / base_plane[2]])
    poses = sorted(report['poses'], key=lambda item: item['cluster_index'])
    boxes = []
    stones = []
    all_points = []
    for item in poses:
        cluster_path = identification / 'pose_identification' / f'cluster_{item["cluster_index"]:02d}.pcd'
        corrected = _read_cloud(cluster_path)
        points = corrected @ base_T_corrected_planning[:3, :3].T + base_T_corrected_planning[:3, 3]
        low = points[:, :2].min(axis=0) - padding
        high = points[:, :2].max(axis=0) + padding
        boxes.append([low.tolist(), high.tolist()])
        base_T_model = base_T_corrected_planning @ np.asarray(item['planning_T_model'], dtype=float)
        stones.append(dict(stone_id=int(item['stone_id']), cluster_index=int(item['cluster_index']),
                           base_link_T_model=base_T_model.tolist()))
        all_points.append(points)
    metadata_path = capture / 'femto_before/field_scan.json'
    metadata = _json(metadata_path)
    crop = metadata.get('base_frame_crop', {})
    points = np.concatenate(all_points)
    low = np.asarray(crop.get('min', points.min(axis=0) - .05), dtype=float)
    high = np.asarray(crop.get('max', points.max(axis=0) + .05), dtype=float)
    low = np.minimum(low, points.min(axis=0) - [.02, .02, .02])
    high = np.maximum(high, points.max(axis=0) + [.02, .02, .02])
    if low.shape != (3,) or high.shape != (3,) or np.any(high <= low):
        raise ValueError('FEMTO field crop cannot define a valid RGB reconstruction region')
    region = dict(schema_version=1, kind='identified_pick_scene_region', frame='base_link',
                  minimum=low.tolist(), maximum=high.tolist(), stone_boxes_xy=boxes,
                  support_plane=support.tolist(), stones=stones,
                  source_sha256={'field_scan.json': digest(metadata_path),
                                 'stone_poses.json': digest(report_path)},
                  limitations=['Stone boxes are padded FEMTO cluster XY bounds used as MVS masks.',
                               'The support plane is transformed back to the uncorrected base_link capture frame.'])
    atomic_json(output, region)
    return region


def _fixed_pose_metrics(model, observed, transform, threshold):
    aligned = model @ transform[:3, :3].T + transform[:3, 3]
    model_distances = cKDTree(observed).query(aligned)[0]
    observed_distances = cKDTree(aligned).query(observed)[0]
    model_inliers = model_distances <= threshold
    observed_inliers = observed_distances <= threshold
    inliers = np.r_[model_distances[model_inliers], observed_distances[observed_inliers]]
    return dict(model_coverage=float(np.mean(model_inliers)),
                observed_coverage=float(np.mean(observed_inliers)),
                rmse_m=float(np.sqrt(np.mean(inliers**2))) if len(inliers) else None,
                observed_median_mm=float(np.median(observed_distances) * 1000),
                observed_p95_mm=float(np.percentile(observed_distances, 95) * 1000),
                observed_points=len(observed), model_points=len(model))


def validate_rgb_poses(dense, region_file, identification, models, profile, output, *, visualize=False):
    """Validate final catalog poses against refined wrist RGB geometry."""
    dense, region_file, identification, models, output = map(
        lambda value: Path(value).resolve(), (dense, region_file, identification, models, output))
    summary = _json(dense / 'summary.json')
    if summary.get('status') != 'complete' or 'refined' not in summary.get('modes', {}):
        raise ValueError('complete refined RGB MVS output is required for pose validation')
    region = _json(region_file)
    points = _read_cloud(dense / 'refined/rgb_dense_voxel_1mm.pcd')
    pose_report_path = identification / 'pose_identification/stone_poses.json'
    pose_report = _json(pose_report_path)
    if pose_report.get('status') != 'complete' or not pose_report.get('poses'):
        raise ValueError('complete pose identification is required for RGB validation')
    catalog = TabletopPcdCatalog(models, [item['stone_id'] for item in pose_report['poses']])
    poses = {int(item['cluster_index']): item for item in pose_report['poses']}
    regions = {int(item['cluster_index']): index for index, item in enumerate(region['stones'])}
    base_T_corrected_planning = np.linalg.inv(_corrected_planning_T_base(pose_report))
    plane = np.asarray(region['support_plane'])
    quality = profile.field_quality
    threshold = profile.field_registration.evaluation_distance
    output.mkdir(parents=True, exist_ok=False)
    results = []
    retained = []
    layers = {}
    try:
        for stone in pose_report['poses']:
            index = regions[int(stone['cluster_index'])]
            low, high = np.asarray(region['stone_boxes_xy'][index])
            selected = (points[:, 2] > points[:, :2] @ plane[:2] + plane[2] + .008)
            selected &= np.all((points[:, :2] >= low) & (points[:, :2] <= high), axis=1)
            observed = points[selected]
            reasons = []
            if len(observed) < profile.field_registration.minimum_points:
                reasons.append('too_few_rgb_points')
                metrics = dict(model_coverage=0., observed_coverage=0., rmse_m=None,
                               observed_median_mm=None, observed_p95_mm=None,
                               observed_points=len(observed), model_points=0)
            else:
                model = catalog.load(stone['stone_id']).points
                pose = poses.get(int(stone['cluster_index']))
                if pose is None or int(pose['stone_id']) != int(stone['stone_id']):
                    raise ValueError('pose identification and RGB region cluster mapping differ')
                transform = base_T_corrected_planning @ np.asarray(pose['planning_T_model'])
                metrics = _fixed_pose_metrics(model, observed, transform, threshold)
                if metrics['observed_coverage'] < quality.minimum_target_coverage:
                    reasons.append('low_rgb_observed_coverage')
                if metrics['model_coverage'] < quality.minimum_source_coverage:
                    reasons.append('low_rgb_model_coverage')
                if metrics['rmse_m'] is None or metrics['rmse_m'] > quality.maximum_rmse:
                    reasons.append('high_rgb_rmse')
                aligned = model @ transform[:3, :3].T + transform[:3, 3]
                layers[f'RGB stone {stone["stone_id"]}'] = observed
                layers[f'Catalog model {stone["stone_id"]}'] = aligned
            retained.append(observed)
            results.append(dict(stone_id=stone['stone_id'], cluster_index=stone['cluster_index'],
                                accepted=not reasons, reasons=reasons, **metrics))
        write_xyz_pcd(output / 'rgb_stone_points.pcd', np.concatenate(retained) if retained else np.empty((0, 3)))
        accepted = all(item['accepted'] for item in results)
        report = dict(schema_version=1, kind='wrist_rgb_fixed_pose_validation',
                      status='complete' if accepted else 'rejected', output_frame='base_link',
                      pose_source=pose_report.get('pose_source', 'femto_identification'),
                      rgb_role='fixed_pose_validation',
                      production_pose_accepted=accepted, accuracy_1mm_verified=False,
                      dense_sha256=digest(dense / 'summary.json'), region_sha256=digest(region_file),
                      identification_sha256=digest(pose_report_path),
                      thresholds=dict(distance_m=threshold,
                                      minimum_rgb_observed_coverage=quality.minimum_target_coverage,
                                      minimum_rgb_model_coverage=quality.minimum_source_coverage,
                                      maximum_rgb_rmse_m=quality.maximum_rmse,
                                      minimum_rgb_points=profile.field_registration.minimum_points),
                      stones=results,
                      limitations=['RGB validates the accepted poses without further translation or rotation.',
                                   'The field registration thresholds are reused as operational cross-sensor gates.',
                                   'Passing these gates does not verify absolute 1-mm pose accuracy.'])
        atomic_json(output / 'summary.json', report)
        if visualize and layers:
            from stacking_common.rendering.pointcloud import render_clouds
            render_clouds(output / 'comparison', layers, {},
                          title='Wrist RGB validation of joint pickup poses')
        return report
    except BaseException as error:
        atomic_json(output / 'summary.json', dict(schema_version=1,
                    kind='wrist_rgb_fixed_pose_validation', status='failed', error=str(error),
                    production_pose_accepted=False, accuracy_1mm_verified=False))
        raise


def validate_femto_stability(capture, identification, region_file, profile, output):
    """Reject a scene whose second fixed-camera bracket differs around a stone."""
    capture, identification, region_file, output = map(
        lambda value: Path(value).resolve(), (capture, identification, region_file, output))
    region = _json(region_file)
    after = load_recorded_femto_scan(capture / 'femto_after')
    if after.frame_id != 'base_link':
        raise ValueError('FEMTO stability check requires a base_link after capture')
    points = after.cloud.points
    plane = np.asarray(region['support_plane'], dtype=float)
    quality = profile.field_quality
    threshold = profile.field_registration.evaluation_distance
    output.mkdir(parents=True, exist_ok=False)
    results = []
    retained = []
    pose_report = _json(identification / 'pose_identification/stone_poses.json')
    base_T_corrected = np.linalg.inv(_corrected_planning_T_base(pose_report))
    try:
        for index, stone in enumerate(region['stones']):
            low, high = np.asarray(region['stone_boxes_xy'][index], dtype=float)
            selected = points[:, 2] > points[:, :2] @ plane[:2] + plane[2] + .008
            selected &= np.all((points[:, :2] >= low) & (points[:, :2] <= high), axis=1)
            observed = points[selected]
            before_path = (identification / 'pose_identification' /
                           f'cluster_{stone["cluster_index"]:02d}.pcd')
            before_corrected = _read_cloud(before_path)
            before = before_corrected @ base_T_corrected[:3, :3].T + base_T_corrected[:3, 3]
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
            results.append(dict(stone_id=stone['stone_id'], cluster_index=stone['cluster_index'],
                                stable=not reasons, reasons=reasons, **metrics))
        write_xyz_pcd(output / 'femto_after_stone_points.pcd',
                      np.concatenate(retained) if retained else np.empty((0, 3)))
        stable = all(item['stable'] for item in results)
        report = dict(schema_version=1, kind='femto_pick_scene_stability',
                      status='complete' if stable else 'rejected', output_frame='base_link',
                      scene_stable=stable, accuracy_1mm_verified=False,
                      after_sha256=digest(capture / 'femto_after/field_scan.json'),
                      region_sha256=digest(region_file),
                      thresholds=dict(distance_m=threshold,
                                      minimum_after_coverage=quality.minimum_target_coverage,
                                      minimum_before_coverage=quality.minimum_source_coverage,
                                      maximum_rmse_m=quality.maximum_rmse,
                                      minimum_after_points=profile.field_registration.minimum_points),
                      stones=results,
                      limitations=['The check compares fixed-camera repeatability around each identified stone.',
                                   'It does not establish absolute accuracy or detect motion below its thresholds.'])
        atomic_json(output / 'summary.json', report)
        return report
    except BaseException as error:
        atomic_json(output / 'summary.json', dict(schema_version=1,
                    kind='femto_pick_scene_stability', status='failed', error=str(error),
                    scene_stable=False, accuracy_1mm_verified=False))
        raise


def _verify_capture(capture):
    capture = Path(capture).resolve()
    status = _json(capture / 'capture.json')
    if status.get('status') != 'complete':
        raise ValueError('pick-scene capture is incomplete')
    session = inspect_session(capture / 'session')
    if session.get('references') != ['femto_before', 'femto_after']:
        raise ValueError('pick-scene capture requires FEMTO before/after references')
    return capture


def _publish_poses(root, identification, validation, *, validation_dir=None):
    if not validation.get('production_pose_accepted'):
        rejected = [str(item['stone_id']) for item in validation['stones'] if not item['accepted']]
        raise ValueError(f'wrist RGB rejected pickup poses for stones {", ".join(rejected)}')
    source_pose = identification / 'pose_data.pkl'
    source_report = identification / 'pose_identification'
    pose = root / 'pose_data.pkl'
    reports = root / 'pose_identification'
    if pose.exists() or reports.exists():
        raise FileExistsError('published pickup poses already exist in workflow output')
    shutil.copytree(source_report, reports)
    shutil.copyfile(source_pose, pose)
    if digest(source_pose) != digest(pose):
        raise ValueError('published pickup pose file differs from joint identification result')
    source = _json(reports / 'stone_poses.json')
    atomic_json(root / 'pick_scene_poses.json', dict(schema_version=1,
                kind='validated_pick_scene_poses', status='complete', pose_data=str(pose),
                data_origin=_json(root / 'pick_scene_workflow.json').get('data_origin', 'hardware'),
                pose_source=source.get('pose_source', 'unknown'),
                joint_identification=str(reports / 'stone_poses.json'),
                preliminary_femto_identification=(
                    'femto_identification/pose_identification/stone_poses.json'
                    if (root / 'femto_identification/pose_identification/stone_poses.json').is_file() else None),
                coarse_femto_geometry=(
                    'femto_geometry/geometry.json' if (root / 'femto_geometry/geometry.json').is_file() else None),
                rgb_validation=str(((validation_dir or root / 'rgb_validation') / 'summary.json').relative_to(root)),
                pose_viewer=(str((identification / 'comparison/overview.html').relative_to(root))
                             if (identification / 'comparison/overview.html').is_file() else None),
                production_pose_accepted=True,
                pose_warnings=source.get('warnings', []),
                accuracy_1mm_verified=False))
    return pose


def _run_femto_first_workflow(args, *, plan_runner=None, capture_runner=None, identify_runner=identify_field,
                 region_runner=build_rgb_region, sparse_runner=None, refine_runner=None,
                 dense_runner=None, fusion_runner=None, fallback_dense_runner=None,
                 stability_runner=validate_femto_stability,
                 validation_runner=validate_rgb_poses,
                 preflight=load_inputs, clock=time.monotonic):
    """Execute or reprocess one capture with injectable hardware/GPU stages."""
    root = Path(args.output).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f'pick-scene workflow output already exists: {root}')
    args.repository = Path(args.repository).expanduser().resolve()
    args.perception_config = (args.perception_config or
                              args.repository / 'apps/tabletop/config/perception.yml').resolve()
    profile, catalog = preflight(args)
    root.mkdir(parents=True)
    manifest = root / 'pick_scene_workflow.json'
    status = dict(schema_version=1, kind='wrist_rgb_femto_pick_workflow', status='running',
                  mode=args.command, stages=[], production_pose_accepted=False,
                  scene_stability='not_evaluated', accuracy_1mm_verified=False)
    started = clock()
    current = None

    def stage(name):
        nonlocal current
        now = clock()
        if current is not None:
            current['elapsed_s'] = now - started - current['started_s']
        current = dict(name=name, started_s=now - started)
        status['stages'].append(current)
        status['stage'] = name
        atomic_json(manifest, status)
        print(name, flush=True)

    try:
        stage('preflight')
        if args.command == 'run':
            if plan_runner is None:
                from .scene_capture_cli import prepare_plan
                plan_runner = prepare_plan
            if capture_runner is None:
                from .scene_capture_cli import run_scan
                capture_runner = run_scan
            plan = root / 'capture_plan'
            plan_args = argparse.Namespace(
                profile=ROOT / 'apps/tabletop/config/hand_eye.yml',
                distance=STANDARD_DISTANCE_M,
                offset=STANDARD_NEAR_OFFSET_M,
                oblique_offset=STANDARD_OBLIQUE_OFFSET_M,
                seed_view=None,
                urdf=None,
                output=plan,
            )
            stage('capture_route')
            planned = plan_runner(plan_args)
            approval = planned.get('approval_sha256')
            if not approval or digest(plan / 'plan.json') != approval:
                raise ValueError('integrated capture route did not produce a valid plan hash')
            status.update(capture_plan_dir=str(plan), capture_plan_sha256=approval,
                          capture_route='standard_coverage9')
            capture = root / 'capture'
            scene_args = argparse.Namespace(plan=plan, approve=approval,
                femto_config=args.femto_config, camera=args.camera, depth_size=args.depth_size,
                fps=args.fps, count=args.count, output=capture)
            stage('capture')
            capture_runner(scene_args)
        else:
            capture = Path(args.capture).expanduser().resolve()
        stage('verify_capture')
        capture = _verify_capture(capture)
        status['capture_dir'] = str(capture)
        status['data_origin'] = _json(capture / 'session/session.json')['data_origin']
        identification = root / 'femto_identification'
        identify_args = copy(args)
        identify_args.identification_output = identification
        stage('femto_identification')
        femto_pose = Path(identify_runner(identify_args, capture / 'femto_before'))
        if femto_pose != identification / 'pose_data.pkl' or not femto_pose.is_file():
            raise ValueError('FEMTO identification did not create the expected pose_data.pkl')
        region_file = root / 'rgb_region.json'
        stage('rgb_region')
        region_runner(capture, identification, region_file)
        stability_dir = root / 'femto_stability'
        stage('femto_stability')
        stability = stability_runner(capture, identification, region_file, profile, stability_dir)
        if not stability.get('scene_stable'):
            status['scene_stability'] = 'rejected'
            rejected = [str(item['stone_id']) for item in stability['stones'] if not item['stable']]
            raise ValueError(f'FEMTO brackets changed around stones {", ".join(rejected)}')
        status['scene_stability'] = 'verified_repeatable'
        if sparse_runner is None:
            from .hand_eye_rgb import rgb_experiment
            sparse_runner = rgb_experiment
        if refine_runner is None:
            from .hand_eye_rgb_refinement import refine_rgb_session
            refine_runner = refine_rgb_session
        if dense_runner is None:
            from .hand_eye_dense import dense_rgb_production_session, dense_rgb_full_session
            dense_runner = dense_rgb_production_session
            if fallback_dense_runner is None:
                fallback_dense_runner = dense_rgb_full_session
        if fusion_runner is None:
            from .pick_scene_fusion import refine_fused_pick_poses
            fusion_runner = refine_fused_pick_poses
        sparse = root / 'rgb_sparse'
        stage('rgb_sparse')
        sparse_runner(capture / 'session', region_file, sparse)
        refined = root / 'rgb_refinement'
        stage('rgb_refinement')
        refine_runner(capture / 'session', sparse, refined)
        from .pick_scene_fusion import FusedPoseRejected
        attempts = [('', dense_runner)]
        if fallback_dense_runner is not None:
            attempts.append(('_full', fallback_dense_runner))
        status['dense_fallback_used'] = False
        for suffix, reconstruct in attempts:
            dense = root / ('rgb_dense' + suffix)
            stage('rgb_dense' + suffix)
            reconstruct(capture / 'session', sparse, refined, dense)
            fused_identification = root / ('fused_identification' + suffix)
            validation_dir = root / ('rgb_validation' + suffix)
            try:
                stage('fused_pose_identification' + suffix)
                fused_pose = Path(fusion_runner(dense, region_file, identification, catalog.model_directory,
                                               profile, fused_identification, visualize=args.visualize))
                if fused_pose != fused_identification / 'pose_data.pkl' or not fused_pose.is_file():
                    raise ValueError('joint FEMTO/RGB identification did not create the expected pose_data.pkl')
                stage('rgb_pose_validation' + suffix)
                validation = validation_runner(dense, region_file, fused_identification,
                                               catalog.model_directory, profile, validation_dir,
                                               visualize=args.visualize)
                if not validation.get('production_pose_accepted'):
                    rejected = [str(item['stone_id']) for item in validation['stones'] if not item['accepted']]
                    raise FusedPoseRejected(f'wrist RGB rejected pickup poses for stones {", ".join(rejected)}')
            except FusedPoseRejected as error:
                if suffix or fallback_dense_runner is None:
                    raise
                status.update(dense_fallback_used=True, dense_fallback_reason=str(error))
                atomic_json(manifest, status)
                print(f'Fast RGB pose check rejected: {error}; retrying full images and all sources', flush=True)
                continue
            break
        stage('publish')
        pose = _publish_poses(root, fused_identification, validation, validation_dir=validation_dir)
        pose_warnings = _json(fused_identification / 'pose_identification/stone_poses.json').get('warnings', [])
        now = clock()
        current['elapsed_s'] = now - started - current['started_s']
        status.update(status='complete', stage='complete', elapsed_s=now - started,
                      pose_data=str(pose), validation=str(validation_dir / 'summary.json'),
                      preliminary_femto_pose_data=str(femto_pose),
                      fused_pose_data=str(fused_pose), dense_dir=str(dense), production_pose_accepted=True,
                      pose_warnings=pose_warnings)
        atomic_json(manifest, status)
        return PickSceneWorkflowResult(root, capture, pose, validation_dir / 'summary.json')
    except BaseException as error:
        now = clock()
        if current is not None:
            current['elapsed_s'] = now - started - current['started_s']
        status.update(status='cancelled' if isinstance(error, KeyboardInterrupt) else 'failed',
                      elapsed_s=now - started, error=f'{type(error).__name__}: {error}')
        atomic_json(manifest, status)
        raise


def run_rgb_first_workflow(args, *, plan_runner=None, capture_runner=None,
                           dense_runner=None, fallback_dense_runner=None,
                           preflight=load_inputs, clock=time.monotonic):
    """Build coarse FEMTO masks, reconstruct RGB, then make the first catalog fit jointly."""
    from .hand_eye_dense import dense_rgb_production_session, dense_rgb_full_session
    from .hand_eye_rgb import rgb_experiment
    from .hand_eye_rgb_refinement import refine_rgb_session
    from .pick_scene_fusion import FusedPoseRejected
    from .pick_scene_rgb_first import (prepare_femto_regions, identify_from_rgb_and_femto,
                                       validate_selected_femto_stability)

    if args.automatic_ordering:
        raise ValueError('automatic RGB/FEMTO stone ID assignment is not validated; review cluster IDs manually')
    root = Path(args.output).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f'pick-scene workflow output already exists: {root}')
    args.repository = Path(args.repository).expanduser().resolve()
    args.perception_config = (args.perception_config or
                              args.repository / 'apps/tabletop/config/perception.yml').resolve()
    profile, catalog = preflight(args)
    backend = getattr(args, 'dense_backend', 'fast-foundation-stereo')
    pose_backend = getattr(args, 'pose_backend', 'foundationpose')
    if pose_backend == 'foundationpose' and backend != 'fast-foundation-stereo':
        raise ValueError('FoundationPose requires --dense-backend fast-foundation-stereo; COLMAP supports --pose-backend icp')
    if pose_backend == 'foundationpose' and (args.manual_init_gui or args.manual_init_stones is not None):
        raise ValueError('manual model alignment requires --pose-backend icp')
    if pose_backend == 'foundationpose':
        from .foundation_pose_runtime import check_runtime
        check_runtime(args.repository)
    if backend == 'fast-foundation-stereo' and dense_runner is None:
        from functools import partial
        from .hand_eye_fast_stereo import reconstruct_with_runtime
        vendor = args.repository / '.data/vendor/Fast-FoundationStereo'
        dense_runner = partial(reconstruct_with_runtime, repository=vendor,
            checkpoint=vendor / 'weights/23-36-37/model_best_bp2_serialize.pth')
    root.mkdir(parents=True)
    manifest = root / 'pick_scene_workflow.json'
    status = dict(schema_version=2, kind='wrist_rgb_femto_pick_workflow', status='running',
                  mode=args.command, stages=[], production_pose_accepted=False,
                  scene_stability='not_evaluated', accuracy_1mm_verified=False,
                  pose_sequence='femto_regions_rgb_mvs_joint_catalog_fit', pose_backend=pose_backend)
    started = clock()
    current = None

    def stage(name):
        nonlocal current
        now = clock()
        if current is not None:
            current['elapsed_s'] = now - started - current['started_s']
        current = dict(name=name, started_s=now-started)
        status['stages'].append(current)
        status['stage'] = name
        atomic_json(manifest, status)
        print(name, flush=True)

    try:
        stage('preflight')
        if args.command == 'run':
            if plan_runner is None:
                from .scene_capture_cli import prepare_plan
                plan_runner = prepare_plan
            if capture_runner is None:
                from .scene_capture_cli import run_scan
                capture_runner = run_scan
            plan = root / 'capture_plan'
            plan_args = argparse.Namespace(
                profile=ROOT / 'apps/tabletop/config/hand_eye.yml',
                distance=STANDARD_DISTANCE_M, offset=STANDARD_NEAR_OFFSET_M,
                oblique_offset=STANDARD_OBLIQUE_OFFSET_M,
                viewpoint_pattern=args.viewpoint_pattern, seed_view=None,
                urdf=None, output=plan)
            stage('capture_route')
            planned = plan_runner(plan_args)
            approval = planned.get('approval_sha256')
            if not approval or digest(plan / 'plan.json') != approval:
                raise ValueError('integrated capture route did not produce a valid plan hash')
            status.update(capture_plan_dir=str(plan), capture_plan_sha256=approval,
                          capture_route=f'standard_{args.viewpoint_pattern}')
            capture = root / 'capture'
            scene_args = argparse.Namespace(plan=plan, approve=approval,
                femto_config=args.femto_config, camera=args.camera, depth_size=args.depth_size,
                fps=args.fps, count=args.count, output=capture)
            stage('capture')
            capture_runner(scene_args)
        else:
            capture = Path(args.capture).expanduser().resolve()
        stage('verify_capture')
        capture = _verify_capture(capture)
        status['capture_dir'] = str(capture)
        status['data_origin'] = _json(capture / 'session/session.json')['data_origin']
        geometry_dir = root / 'femto_geometry'
        region_file = root / 'rgb_region.json'
        stage('femto_regions')
        prepare_femto_regions(capture, profile, geometry_dir, region_file)
        sparse = root / 'rgb_sparse'
        stage('rgb_sparse')
        rgb_experiment(capture / 'session', region_file, sparse)
        refined = root / 'rgb_refinement'
        stage('rgb_refinement')
        refine_rgb_session(capture / 'session', sparse, refined)
        status['dense_backend'] = backend
        status['accepted_dense_backend'] = None
        status['dense_fallback_used'] = False
        for suffix, reconstruct in (('', dense_runner or dense_rgb_production_session),
                                    ('_full', fallback_dense_runner or dense_rgb_full_session)):
            dense = root / ('rgb_dense'+suffix)
            stage('rgb_dense'+suffix)
            reconstruct(capture / 'session', sparse, refined, dense)
            identification = root / ('fused_identification'+suffix)
            validation_dir = root / ('rgb_validation'+suffix)
            try:
                stage('joint_pose_identification'+suffix)
                fused_pose = Path(identify_from_rgb_and_femto(
                    dense, region_file, geometry_dir, catalog, profile, identification,
                    pose_backend=pose_backend, repository=args.repository,
                    automatic_ordering=args.automatic_ordering,
                    manual_init_gui=args.manual_init_gui,
                    manual_init_stones=args.manual_init_stones,
                    manual_init_translation_step=args.manual_init_translation_step,
                    manual_init_rotation_step=args.manual_init_rotation_step,
                    visualize=args.visualize))
                print(f'Pose-identification viewer: {identification / "comparison/overview.html"}', flush=True)
                stability_dir = root / ('femto_stability'+suffix)
                stage('femto_stability'+suffix)
                stability = validate_selected_femto_stability(
                    capture, geometry_dir, region_file, identification, profile, stability_dir)
                if not stability['scene_stable']:
                    status['scene_stability'] = 'rejected'
                    raise ValueError('FEMTO brackets changed around selected stones')
                status['scene_stability'] = 'verified_repeatable'
                stage('rgb_pose_validation'+suffix)
                validation = validate_rgb_poses(dense, region_file, identification,
                    catalog.model_directory, profile, validation_dir, visualize=args.visualize)
                if not validation.get('production_pose_accepted'):
                    rejected = [str(item['stone_id']) for item in validation['stones'] if not item['accepted']]
                    raise FusedPoseRejected(f'wrist RGB rejected pickup poses for stones {", ".join(rejected)}')
            except FusedPoseRejected as error:
                if suffix or pose_backend == 'foundationpose':
                    raise
                status.update(dense_fallback_used=True, dense_fallback_reason=str(error))
                atomic_json(manifest, status)
                print(f'Joint RGB/FEMTO pose check rejected: {error}; retrying COLMAP with full images', flush=True)
                continue
            break
        status['accepted_dense_backend'] = 'colmap' if suffix else backend
        stage('publish')
        pose = _publish_poses(root, identification, validation, validation_dir=validation_dir)
        now = clock()
        current['elapsed_s'] = now-started-current['started_s']
        report = _json(identification / 'pose_identification/stone_poses.json')
        status.update(status='complete', stage='complete', elapsed_s=now-started,
                      pose_data=str(pose), fused_pose_data=str(fused_pose),
                      femto_geometry_dir=str(geometry_dir), dense_dir=str(dense),
                      pose_viewer=str(identification / 'comparison/overview.html'),
                      validation=str(validation_dir / 'summary.json'),
                      production_pose_accepted=True, pose_warnings=report.get('warnings', []))
        atomic_json(manifest, status)
        return PickSceneWorkflowResult(root, capture, pose, validation_dir / 'summary.json')
    except BaseException as error:
        now = clock()
        if current is not None:
            current['elapsed_s'] = now-started-current['started_s']
        status.update(status='cancelled' if isinstance(error, KeyboardInterrupt) else 'failed',
                      elapsed_s=now-started, error=f'{type(error).__name__}: {error}')
        atomic_json(manifest, status)
        raise


def _prepare_reidentification(args):
    """Branch a completed fit, copying only final clouds/viewers, not MVS workspaces."""
    source = Path(args.workflow).expanduser().resolve()
    previous = _json(source / 'pick_scene_workflow.json')
    if (previous.get('schema_version') != 2 or previous.get('status') != 'complete'
            or not previous.get('production_pose_accepted')):
        raise ValueError('reidentify requires a completed RGB-first workflow')
    dense = Path(previous['dense_dir']).resolve()
    if not dense.is_relative_to(source):
        raise ValueError('RGB reconstruction is outside the source workflow')
    report = _json(source / 'pose_identification/stone_poses.json')
    hashes = report.get('source_sha256', {})
    for label, path in [('dense_summary.json', dense / 'summary.json'),
                        ('femto_geometry.json', source / 'femto_geometry/geometry.json'),
                        ('rgb_region.json', source / 'rgb_region.json')]:
        if hashes.get(label) != digest(path):
            raise ValueError(f'cached input changed since pose identification: {label}')
    if not (dense / 'refined/rgb_dense_voxel_1mm.pcd').is_file():
        raise ValueError('completed RGB point cloud is missing')
    capture = _verify_capture(previous['capture_dir'])
    root = Path(args.output).expanduser().resolve()
    if root == source or root.is_relative_to(source):
        raise ValueError('choose a new output directory outside the source workflow')
    root.mkdir(parents=True, exist_ok=False)
    cached = root / 'rgb_dense'
    cached.mkdir()
    for name in ('summary.json', 'refined/rgb_dense_voxel_1mm.pcd', 'rgb_only', 'comparison'):
        src, dst = dense / name, cached / name
        if src.is_dir():
            shutil.copytree(src, dst)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for pattern in ('pair_*.npz', 'pair_*.png'):
        for src in (dense / 'refined').glob(pattern):
            shutil.copy2(src, cached / 'refined' / src.name)
    shutil.copytree(source / 'femto_geometry', root / 'femto_geometry')
    shutil.copy2(source / 'rgb_region.json', root / 'rgb_region.json')
    status = dict(schema_version=2, kind='wrist_rgb_femto_pick_workflow',
                  mode='reidentify', status='ready', stage='joint_pose_identification', stages=[],
                  capture_dir=str(capture), data_origin=previous.get('data_origin', 'hardware'),
                  production_pose_accepted=False, scene_stability='not_evaluated',
                  accuracy_1mm_verified=False, pose_sequence=previous['pose_sequence'],
                  reidentification_source=str(source),
                  source_workflow_sha256=digest(source / 'pick_scene_workflow.json'),
                  suggested_assignments=report.get('cluster_assignments', {}),
                  reused_dense_dir=str(dense), resume_dense_dir=str(cached))
    atomic_json(root / 'pick_scene_workflow.json', status)
    return root, status


def resume_joint_workflow(args, *, preflight=load_inputs, clock=time.monotonic):
    """Review IDs and fit again using cached RGB MVS and FEMTO geometry."""
    from .pick_scene_rgb_first import identify_from_rgb_and_femto, validate_selected_femto_stability

    if args.automatic_ordering:
        raise ValueError('automatic RGB/FEMTO stone ID assignment is not validated; review cluster IDs manually')
    if args.command == 'reidentify':
        root, status = _prepare_reidentification(args)
    else:
        root = Path(args.workflow).expanduser().resolve()
        status = _json(root / 'pick_scene_workflow.json')
        if (status.get('schema_version') != 2 or status.get('status') not in ('failed', 'cancelled')
                or not status.get('stage', '').startswith('joint_pose_identification')
                or status.get('production_pose_accepted')
                or (root / 'pose_data.pkl').exists()):
            raise ValueError('resume requires a failed RGB-first workflow at joint pose identification')
    manifest = root / 'pick_scene_workflow.json'
    failed_stage = status.get('stage', '')
    capture = _verify_capture(status['capture_dir'])
    dense = Path(status.get('resume_dense_dir') or
                 root / ('rgb_dense_full' if 'full' in failed_stage else 'rgb_dense')).resolve()
    if not dense.is_relative_to(root):
        raise ValueError('cached RGB reconstruction is outside the workflow')
    if not (dense / 'summary.json').is_file():
        raise ValueError('completed RGB MVS output is missing from the failed workflow')
    geometry_dir = root / 'femto_geometry'
    region_file = root / 'rgb_region.json'
    if not (geometry_dir / 'geometry.json').is_file() or not region_file.is_file():
        raise ValueError('coarse FEMTO geometry is missing from the failed workflow')
    args.repository = Path(args.repository).expanduser().resolve()
    args.perception_config = (args.perception_config or
                              args.repository / 'apps/tabletop/config/perception.yml').resolve()
    profile, catalog = preflight(args)
    attempt = int(status.get('resume_count', 0)) + 1
    identification = root / f'fused_identification_resume_{attempt:02d}'
    stability_dir = root / f'femto_stability_resume_{attempt:02d}'
    validation_dir = root / f'rgb_validation_resume_{attempt:02d}'
    if any(path.exists() for path in (identification, stability_dir, validation_dir)):
        raise FileExistsError('resume output already exists; inspect the failed workflow before retrying')
    previous_elapsed = float(status.get('elapsed_s', 0))
    started = clock()
    current = None
    status.update(status='running', resume_count=attempt, resumed_from=failed_stage,
                  previous_failure=status.pop('error', None),
                  resume_dense_dir=str(dense), pose_backend=args.pose_backend)
    atomic_json(manifest, status)

    def stage(name):
        nonlocal current
        now = clock()
        if current is not None:
            current['elapsed_s'] = previous_elapsed + now - started - current['started_s']
        current = dict(name=name, started_s=previous_elapsed + now - started)
        status['stages'].append(current)
        status['stage'] = name
        atomic_json(manifest, status)
        print(name, flush=True)

    try:
        stage('joint_pose_identification_resume')
        fused_pose = Path(identify_from_rgb_and_femto(
            dense, region_file, geometry_dir, catalog, profile, identification,
            pose_backend=args.pose_backend, repository=args.repository,
            manual_init_gui=args.manual_init_gui,
            manual_init_stones=args.manual_init_stones,
            manual_init_translation_step=args.manual_init_translation_step,
            manual_init_rotation_step=args.manual_init_rotation_step,
            visualize=args.visualize,
            suggested_assignments=status.get('suggested_assignments')))
        print(f'Pose-identification viewer: {identification / "comparison/overview.html"}', flush=True)
        stage('femto_stability_resume')
        stability = validate_selected_femto_stability(
            capture, geometry_dir, region_file, identification, profile, stability_dir)
        if not stability['scene_stable']:
            status['scene_stability'] = 'rejected'
            raise ValueError('FEMTO brackets changed around selected stones')
        status['scene_stability'] = 'verified_repeatable'
        stage('rgb_pose_validation_resume')
        validation = validate_rgb_poses(dense, region_file, identification,
            catalog.model_directory, profile, validation_dir, visualize=args.visualize)
        if not validation.get('production_pose_accepted'):
            rejected = [str(item['stone_id']) for item in validation['stones'] if not item['accepted']]
            raise ValueError(f'wrist RGB rejected pickup poses for stones {", ".join(rejected)}')
        stage('publish_resume')
        pose = _publish_poses(root, identification, validation, validation_dir=validation_dir)
        now = clock()
        current['elapsed_s'] = previous_elapsed + now - started - current['started_s']
        report = _json(identification / 'pose_identification/stone_poses.json')
        status.update(status='complete', stage='complete', elapsed_s=previous_elapsed + now-started,
                      pose_data=str(pose), fused_pose_data=str(fused_pose),
                      femto_geometry_dir=str(geometry_dir), dense_dir=str(dense),
                      pose_viewer=str(identification / 'comparison/overview.html'),
                      validation=str(validation_dir / 'summary.json'),
                      production_pose_accepted=True, pose_warnings=report.get('warnings', []))
        atomic_json(manifest, status)
        return PickSceneWorkflowResult(root, capture, pose, validation_dir / 'summary.json')
    except BaseException as error:
        now = clock()
        if current is not None:
            current['elapsed_s'] = previous_elapsed + now - started - current['started_s']
        status.update(status='cancelled' if isinstance(error, KeyboardInterrupt) else 'failed',
                      elapsed_s=previous_elapsed + now-started,
                      error=f'{type(error).__name__}: {error}')
        atomic_json(manifest, status)
        raise


def run_workflow(args, *, plan_runner=None, capture_runner=None, identify_runner=identify_field,
                 region_runner=build_rgb_region, sparse_runner=None, refine_runner=None,
                 dense_runner=None, fusion_runner=None, fallback_dense_runner=None,
                 stability_runner=validate_femto_stability,
                 validation_runner=validate_rgb_poses, preflight=load_inputs, clock=time.monotonic):
    """The public pick workflow always makes its first catalog fit with RGB and FEMTO."""
    if (identify_runner is not identify_field or region_runner is not build_rgb_region
            or sparse_runner is not None or refine_runner is not None
            or fusion_runner is not None or stability_runner is not validate_femto_stability
            or validation_runner is not validate_rgb_poses):
        raise ValueError('legacy stage overrides cannot be used with RGB-first identification')
    return run_rgb_first_workflow(args, plan_runner=plan_runner,
                                  capture_runner=capture_runner, dense_runner=dense_runner,
                                  fallback_dense_runner=fallback_dense_runner,
                                  preflight=preflight, clock=clock)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    resume = commands.add_parser('resume', help='finish a failed joint pose fit using saved RGB MVS')
    add_identification_arguments(resume, include_output=False)
    resume.add_argument('workflow', type=Path, help='failed RGB-first workflow directory')
    reidentify = commands.add_parser('reidentify', help='review model IDs again using saved RGB/FEMTO clouds')
    add_identification_arguments(reidentify, include_output=False)
    reidentify.add_argument('workflow', type=Path, help='completed RGB-first workflow directory')
    reidentify.add_argument('--output', type=Path, required=True, help='new identification workflow directory')
    for name in ('run', 'process'):
        command = commands.add_parser(name, help=('capture and identify with the standard live route' if name == 'run'
                                      else 'identify an existing complete scene capture'))
        add_identification_arguments(command, include_output=False)
        command.add_argument('--output', type=Path, required=True, help='new workflow directory')
        command.add_argument('--dense-backend', choices=('colmap', 'fast-foundation-stereo'),
                             default='fast-foundation-stereo', help='Fast-FoundationStereo (default) or COLMAP MVS')
        if name == 'run':
            command.add_argument('--viewpoint-pattern', choices=('coverage9', 'coverage13', 'coverage17'),
                                 default='coverage13',
                                 help='coverage17 adds 45/60-degree left/right views; coverage13 uses 30-degree sides; coverage9 is compact')
            command.add_argument('--femto-config', type=Path,
                                 default=ROOT / 'apps/tabletop/config/femto.yml')
            command.add_argument('--camera', choices=('primary', 'secondary'), default='primary')
            command.add_argument('--depth-size', type=int, nargs=2, default=(848, 480))
            command.add_argument('--fps', type=int, default=15)
            command.add_argument('--count', type=int, default=10)
        else:
            command.add_argument('capture', type=Path)
    for command in (resume, reidentify, *[commands.choices[name] for name in ('run', 'process')]):
        command.add_argument('--pose-backend', choices=('foundationpose', 'icp'), default='foundationpose',
                             help='FoundationPose (default, learned stereo required) or multistart ICP')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    def terminate(_signum, _frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        result = resume_joint_workflow(args) if args.command in ('resume', 'reidentify') else run_workflow(args)
        print(json.dumps(dict(status='complete', output=str(result.root),
                              pose_data=str(result.pose_data), validation=str(result.validation)), indent=2))
        return 0
    except KeyboardInterrupt:
        print('Pick-scene workflow cancelled; partial artifacts retained.', flush=True)
        return 130
    except Exception as error:
        print(f'Pick-scene workflow failed: {error}', flush=True)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
