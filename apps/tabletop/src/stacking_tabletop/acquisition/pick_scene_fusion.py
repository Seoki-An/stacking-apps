"""Joint FEMTO/RGB catalog-pose refinement for an identified pick scene."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from stacking_common.pointcloud import (
    Open3dRegistrationBackend,
    PointCloudData,
    evaluate_registration,
)
from stacking_common.pointcloud.processing import fixed_grid_centroids
from stacking_tabletop.perception.catalog import TabletopPcdCatalog
from .hand_eye_capture import atomic_json, digest
from .storage import write_xyz_pcd


ROTATION_WARNING_DEGREES = 10.0


class FusedPoseRejected(ValueError):
    """A reconstructed candidate failed a non-rotation pose acceptance check."""


def _classify_pose_reasons(decisions):
    """Keep rotation outliers visible without making them pose rejections."""
    reasons = sorted({f'{sensor}_{reason}' for sensor, decision in decisions.items()
                      for reason in decision.reasons})
    warnings = [reason for reason in reasons if reason.endswith('_large_rotation_correction')]
    rejected = [reason for reason in reasons if reason not in warnings]
    return rejected, warnings


def _json(path):
    return json.loads(Path(path).read_text())


def _read_cloud(path):
    import open3d as o3d
    points = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=float)
    if not len(points) or not np.all(np.isfinite(points)):
        raise ValueError(f'point cloud is empty or invalid: {path}')
    return points


def _corrected_planning_T_base(report):
    transform = np.asarray(report['planning_T_base_link'], dtype=float)
    if transform.shape != (4, 4):
        raise ValueError('FEMTO planning/base transform is invalid')
    correction = np.eye(4)
    correction[2, 3] = float(report['table_height_correction_m'])
    return correction @ transform


def _transform(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def _metric(result):
    return dict(
        model_coverage=float(result.source_coverage),
        observed_coverage=float(result.target_coverage),
        rmse_m=float(result.rmse) if np.isfinite(result.rmse) else None,
        correspondences=int(result.correspondences),
    )


def _balanced_observation(femto, rgb, voxel_size, minimum_points):
    femto = fixed_grid_centroids(PointCloudData(femto), voxel_size).points
    rgb = fixed_grid_centroids(PointCloudData(rgb), voxel_size).points
    count = min(len(femto), len(rgb))
    if count < minimum_points:
        raise ValueError(
            f'joint pose refinement needs at least {minimum_points} points from each sensor; '
            f'got FEMTO={len(femto)}, RGB={len(rgb)}'
        )

    def sample(points):
        if len(points) == count:
            return points
        return points[np.linspace(0, len(points) - 1, count, dtype=int)]

    return femto, rgb, np.concatenate((sample(femto), sample(rgb)))


def refine_fused_pick_poses(
    dense,
    region_file,
    femto_identification,
    models,
    profile,
    output,
    *,
    visualize=False,
):
    """Refine known FEMTO identities using an equally sampled two-sensor observation."""
    dense = Path(dense).resolve()
    region_file = Path(region_file).resolve()
    femto_identification = Path(femto_identification).resolve()
    output = Path(output).resolve()
    dense_summary = _json(dense / 'summary.json')
    region = _json(region_file)
    femto_report_path = femto_identification / 'pose_identification/stone_poses.json'
    femto_report = _json(femto_report_path)
    if dense_summary.get('status') != 'complete' or 'refined' not in dense_summary.get('modes', {}):
        raise ValueError('complete refined RGB MVS output is required for fused pose refinement')
    if femto_report.get('status') != 'complete' or not femto_report.get('poses'):
        raise ValueError('complete FEMTO identification is required for fused pose refinement')
    if [int(item['stone_id']) for item in region['stones']] != [
            int(item['stone_id']) for item in sorted(femto_report['poses'], key=lambda value: value['cluster_index'])]:
        raise ValueError('RGB regions and FEMTO identification contain different stones')

    rgb_scene = _read_cloud(dense / 'refined/rgb_dense_voxel_1mm.pcd')
    stone_ids = [int(item['stone_id']) for item in femto_report['poses']]
    catalog = TabletopPcdCatalog(models, stone_ids)
    corrected_planning_T_base = _corrected_planning_T_base(femto_report)
    base_T_corrected_planning = np.linalg.inv(corrected_planning_T_base)
    threshold = profile.field_registration.evaluation_distance
    voxel_size = profile.field_registration.voxel_sizes[-1]
    minimum_points = profile.field_registration.minimum_points
    quality = replace(
        profile.field_quality,
        maximum_correction_translation=threshold,
        maximum_correction_rotation_degrees=ROTATION_WARNING_DEGREES,
    )
    backend = Open3dRegistrationBackend(profile.field_registration)
    plane = np.asarray(region['support_plane'], dtype=float)
    region_by_cluster = {int(item['cluster_index']): (index, item)
                         for index, item in enumerate(region['stones'])}

    output.mkdir(parents=True, exist_ok=False)
    report_dir = output / 'pose_identification'
    report_dir.mkdir()
    report = dict(
        version=1,
        kind='femto_rgb_joint_pose_identification',
        frame_id='world',
        source_frame_id='base_link',
        status='running',
        pose_source='balanced_femto_rgb_joint_icp',
        planning_T_base_link=femto_report['planning_T_base_link'],
        table_height_correction_m=float(femto_report['table_height_correction_m']),
        table_plane=femto_report['table_plane'],
        fitted_table_plane=femto_report['fitted_table_plane'],
        source_sha256={
            'dense_summary.json': digest(dense / 'summary.json'),
            'rgb_region.json': digest(region_file),
            'femto_stone_poses.json': digest(femto_report_path),
        },
        thresholds=dict(
            model_consistency_distance_m=threshold,
            registration_evaluation_distance_m=threshold,
            maximum_translation_from_femto_m=threshold,
            rotation_warning_from_femto_degrees=ROTATION_WARNING_DEGREES,
            balance_voxel_size_m=voxel_size,
            minimum_points_per_sensor=minimum_points,
        ),
        poses=[],
        rejected=[],
        warnings=[],
        accuracy_1mm_verified=False,
    )
    atomic_json(report_dir / 'stone_poses.json', report)
    layers = {}
    try:
        for femto_pose in sorted(femto_report['poses'], key=lambda value: value['cluster_index']):
            stone_id = int(femto_pose['stone_id'])
            cluster_index = int(femto_pose['cluster_index'])
            region_index, region_stone = region_by_cluster[cluster_index]
            if int(region_stone['stone_id']) != stone_id:
                raise ValueError(f'stone {stone_id} cluster mapping differs between FEMTO and RGB region')
            low, high = np.asarray(region['stone_boxes_xy'][region_index], dtype=float)
            selected = rgb_scene[:, 2] > rgb_scene[:, :2] @ plane[:2] + plane[2] + .008
            selected &= np.all((rgb_scene[:, :2] >= low) & (rgb_scene[:, :2] <= high), axis=1)
            rgb_raw_base = rgb_scene[selected]
            model = catalog.load(stone_id).points
            initial = np.asarray(femto_pose['planning_T_model'], dtype=float)
            base_T_model = base_T_corrected_planning @ initial
            initial_model_base = _transform(model, base_T_model)
            distances = cKDTree(initial_model_base).query(rgb_raw_base)[0] if len(rgb_raw_base) else np.empty(0)
            rgb_consistent_base = rgb_raw_base[distances <= threshold]
            rgb_planning = _transform(rgb_consistent_base, corrected_planning_T_base)
            femto_path = (femto_identification / 'pose_identification' /
                          f'cluster_{cluster_index:02d}.pcd')
            femto_planning = _read_cloud(femto_path)
            femto_voxel, rgb_voxel, combined = _balanced_observation(
                femto_planning, rgb_planning, voxel_size, minimum_points
            )

            observed_to_model = backend.register(
                PointCloudData(combined), PointCloudData(model), np.linalg.inv(initial)
            )
            final = np.linalg.inv(observed_to_model.transform)
            initial_results = {
                'femto': backend.evaluate(
                    PointCloudData(model), PointCloudData(femto_voxel), initial),
                'rgb': backend.evaluate(
                    PointCloudData(model), PointCloudData(rgb_voxel), initial),
                'combined': backend.evaluate(
                    PointCloudData(model), PointCloudData(combined), initial),
            }
            combined_result = backend.evaluate(
                PointCloudData(model), PointCloudData(combined), final,
                scales=observed_to_model.scales,
                correspondences=observed_to_model.correspondences,
            )
            femto_result = backend.evaluate(
                PointCloudData(model), PointCloudData(femto_voxel), final
            )
            rgb_result = backend.evaluate(
                PointCloudData(model), PointCloudData(rgb_voxel), final
            )
            decisions = {
                'combined': evaluate_registration(combined_result, initial, quality),
                'femto': evaluate_registration(femto_result, initial, quality),
                'rgb': evaluate_registration(rgb_result, initial, quality),
            }
            reasons, warnings = _classify_pose_reasons(decisions)
            correction = np.linalg.inv(initial) @ final
            correction_rotation_degrees = float(
                np.rad2deg(Rotation.from_matrix(correction[:3, :3]).magnitude())
            )
            if warnings:
                report['warnings'].append(dict(
                    stone_id=stone_id,
                    cluster_index=cluster_index,
                    correction_rotation_degrees=correction_rotation_degrees,
                    reasons=warnings,
                ))
            if combined_result.rmse > initial_results['combined'].rmse + 1e-12:
                reasons.append('combined_rmse_not_improved')
            if reasons:
                report['rejected'].append(dict(
                    stone_id=stone_id,
                    cluster_index=cluster_index,
                    reasons=reasons,
                    warnings=warnings,
                    initial_planning_T_model=initial.tolist(),
                    candidate_planning_T_model=final.tolist(),
                    initial_sensors={name: _metric(result)
                                     for name, result in initial_results.items()},
                    sensors=dict(
                        femto=_metric(femto_result),
                        rgb=_metric(rgb_result),
                        combined=_metric(combined_result),
                    ),
                ))
                continue

            pose = np.r_[final[:3, 3], Rotation.from_matrix(final[:3, :3]).as_quat()]
            entry = dict(
                stone_id=stone_id,
                cluster_index=cluster_index,
                pose_xyzw=pose.tolist(),
                planning_T_model=final.tolist(),
                initial_femto_planning_T_model=initial.tolist(),
                correction_translation_m=float(np.linalg.norm(correction[:3, 3])),
                correction_rotation_degrees=correction_rotation_degrees,
                warnings=warnings,
                rgb_input_points=len(rgb_raw_base),
                rgb_model_consistent_points=len(rgb_consistent_base),
                femto_voxel_points=len(femto_voxel),
                rgb_voxel_points=len(rgb_voxel),
                balanced_points=len(combined),
                initial_sensors={name: _metric(result)
                                 for name, result in initial_results.items()},
                sensors=dict(
                    femto=_metric(femto_result),
                    rgb=_metric(rgb_result),
                    combined=_metric(combined_result),
                ),
                model_file=str(catalog.path(stone_id)),
                model_sha256=hashlib.sha256(catalog.path(stone_id).read_bytes()).hexdigest(),
            )
            report['poses'].append(entry)
            write_xyz_pcd(report_dir / f'stone_{stone_id}_femto.pcd', femto_voxel)
            write_xyz_pcd(report_dir / f'stone_{stone_id}_rgb.pcd', rgb_voxel)
            write_xyz_pcd(report_dir / f'stone_{stone_id}_balanced_observation.pcd', combined)
            aligned = _transform(model, final)
            write_xyz_pcd(report_dir / f'stone_{stone_id}_aligned.pcd', aligned)
            layers[f'FEMTO stone {stone_id}'] = femto_voxel
            layers[f'RGB stone {stone_id}'] = rgb_voxel
            layers[f'Joint model {stone_id}'] = aligned

        report['status'] = 'complete' if report['poses'] and not report['rejected'] else 'failed'
        report['limitations'] = [
            'FEMTO supplies segmentation, identity, metric frame and the initial pose.',
            'RGB points enter fitting only when they lie within the field registration distance of the initial catalog pose.',
            'FEMTO and RGB are voxelized and sampled equally so RGB density cannot dominate ICP.',
            'The joint fit must independently pass the existing field quality gates for FEMTO and RGB.',
            'The joint fit is rejected unless its balanced two-sensor RMSE improves over the FEMTO initialization.',
            'Rotation corrections above 10 degrees are reported as warnings, not rejections.',
            'Acceptance is an operational consistency result; absolute 1 mm accuracy remains unverified.',
        ]
        atomic_json(report_dir / 'stone_poses.json', report)
        if report['status'] != 'complete':
            rejected = ', '.join(str(item['stone_id']) for item in report['rejected'])
            raise FusedPoseRejected(f'joint FEMTO/RGB pose refinement rejected stones [{rejected}]')
        pose_path = output / 'pose_data.pkl'
        with pose_path.open('xb') as stream:
            pickle.dump({
                item['stone_id']: (
                    np.asarray(item['pose_xyzw'][:3]),
                    np.asarray(item['pose_xyzw'][3:]),
                )
                for item in report['poses']
            }, stream, protocol=pickle.HIGHEST_PROTOCOL)
        if visualize and layers:
            from stacking_common.rendering.pointcloud import render_clouds
            render_clouds(output / 'comparison', layers, {},
                          title='Joint FEMTO / wrist RGB catalog-pose refinement')
        return pose_path
    except BaseException as error:
        if report.get('status') == 'running':
            report.update(status='failed', error=f'{type(error).__name__}: {error}')
            atomic_json(report_dir / 'stone_poses.json', report)
        raise
