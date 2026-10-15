"""Reconstruct the configured pickup workspace, then segment RGB observations."""

from itertools import product
from pathlib import Path

import numpy as np

from stacking_common.pointcloud import PointCloudData
from stacking_tabletop.perception.segmentation import TabletopFieldSegmenter
from .calibration import BASE_LINK_T_UR_BASE
from .hand_eye_capture import atomic_json, digest
from .hand_eye_replay import read_json
from .pick_scene_fusion import _read_cloud, _transform
from .storage import write_xyz_pcd


def workspace_region(profile, output):
    """Use configured workspace bounds, without consulting any camera depth."""
    config = profile.segmentation
    corners = np.array(list(product(*zip(config.workspace_min, config.workspace_max))))
    base = _transform(corners, np.linalg.inv(BASE_LINK_T_UR_BASE))
    low, high = base.min(axis=0), base.max(axis=0)
    region = dict(schema_version=1, kind='wrist_pick_workspace', frame='base_link',
                  minimum=low.tolist(), maximum=high.tolist(),
                  stone_boxes_xy=[[low[:2].tolist(), high[:2].tolist()]],
                  support_plane=[0., 0., config.designated_table_height],
                  source='configured pickup workspace; recorded robot/hand-eye camera poses')
    atomic_json(output, region)
    return region


def prepare_rgb_regions(dense, profile, output, region_file, *, padding=.01):
    """Fit the observed table and stone clusters after RGB reconstruction."""
    dense, output = Path(dense), Path(output)
    summary = read_json(dense / 'summary.json')
    if summary.get('status') != 'complete':
        raise ValueError('complete wrist RGB reconstruction is required for segmentation')
    source = dense / 'refined/rgb_dense_voxel_1mm.pcd'
    segmented = TabletopFieldSegmenter(profile.segmentation).segment(
        PointCloudData(_read_cloud(source)).transformed(BASE_LINK_T_UR_BASE))
    if not segmented.clusters:
        raise ValueError('wrist RGB found no stone clusters above the observed table')
    output.mkdir(parents=True, exist_ok=False)
    hashes = {'dense_summary.json': digest(dense / 'summary.json'), 'rgb_pcd': digest(source)}
    geometry = dict(schema_version=1, kind='wrist_rgb_pick_scene_geometry', status='complete',
                    frame_id='world', source_frame_id='base_link',
                    planning_T_base_link=BASE_LINK_T_UR_BASE.tolist(),
                    table_height_correction_m=float(segmented.height_correction),
                    table_plane=segmented.support_plane.tolist(),
                    fitted_table_plane=segmented.fitted_plane.tolist(),
                    cluster_count=len(segmented.clusters), source_sha256=hashes,
                    catalog_pose_fitted=False, accuracy_1mm_verified=False)
    correction = np.eye(4)
    correction[2, 3] = segmented.height_correction
    planning_T_base = correction @ BASE_LINK_T_UR_BASE
    base_T_planning = np.linalg.inv(planning_T_base)
    plane = planning_T_base.T @ segmented.fitted_plane
    boxes = []
    for index, cluster in enumerate(segmented.clusters):
        write_xyz_pcd(output / f'cluster_{index:02d}.pcd', cluster.points)
        points = _transform(cluster.points, base_T_planning)
        boxes.append([(points[:, :2].min(axis=0)-padding).tolist(),
                      (points[:, :2].max(axis=0)+padding).tolist()])
    write_xyz_pcd(output / 'scene_cropped.pcd', segmented.cropped.points)
    region = dict(schema_version=1, kind='observed_wrist_rgb_pick_regions', frame='base_link',
                  minimum=summary['region']['minimum'], maximum=summary['region']['maximum'],
                  stone_boxes_xy=boxes, support_plane=(-plane[[0, 1, 3]]/plane[2]).tolist(),
                  stones=[dict(cluster_index=i, stone_id=None) for i in range(len(boxes))],
                  source_sha256=hashes,
                  limitations=['Clusters and fitted table come from wrist RGB stereo only; no FEMTO anchor.'])
    atomic_json(output / 'geometry.json', geometry)
    atomic_json(region_file, region)
    return geometry


def region_matches_dense(dense, region, summary):
    """Accept original ROI or hash-bound segmentation of that reconstruction."""
    if summary.get('region') == region:
        return True
    if region.get('kind') != 'observed_wrist_rgb_pick_regions' or region.get('frame') != 'base_link':
        return False
    hashes = region.get('source_sha256', {})
    return (hashes.get('dense_summary.json') == digest(Path(dense) / 'summary.json')
            and hashes.get('rgb_pcd') == digest(Path(dense) / 'refined/rgb_dense_voxel_1mm.pcd')
            and all(region.get(key) == summary['region'].get(key) for key in ('minimum', 'maximum')))
