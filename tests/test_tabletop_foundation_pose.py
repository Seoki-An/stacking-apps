"""Learned pose publication must preserve estimates and keep sensor gates."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stacking_common.pointcloud import PointCloudData
from stacking_tabletop.acquisition import foundation_pose_runtime, pick_scene_rgb_first as joint
from stacking_tabletop.acquisition.hand_eye_capture import atomic_json
from stacking_tabletop.acquisition.pick_scene_fusion import FusedPoseRejected
from stacking_tabletop.acquisition.pick_scene_workflow import build_parser
from stacking_tabletop.acquisition.storage import write_xyz_pcd
from stacking_tabletop.config import load_perception_profile

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('shift', [0., .5])
def test_default_preserves_learned_pose_and_rejects_sensor_disagreement(tmp_path, monkeypatch, shift):
    dense, geometry, output = [tmp_path/name for name in ('dense', 'geometry', 'output')]
    (dense/'refined').mkdir(parents=True)
    geometry.mkdir()
    region = dict(stone_boxes_xy=[[[-.1, -.1], [.1, .1]]], support_plane=[0, 0, 0])
    region_file = tmp_path/'region.json'
    atomic_json(region_file, region)
    atomic_json(dense/'summary.json', dict(status='complete', modes={'refined': {}},
                kind='fast_foundation_stereo_experiment', region=region))
    frame = dict(catalog_pose_fitted=False, table_height_correction_m=.013,
                 planning_T_base_link=np.diag([-1, -1, 1, 1]).tolist(),
                 table_plane=[0, 0, 1, 0], fitted_table_plane=[0, 0, 1, 0])
    atomic_json(geometry/'geometry.json', frame)
    points = np.random.default_rng(4).uniform(-.02, .02, (300, 3))
    base_pose = np.eye(4); base_pose[2, 3] = .05
    rgb = points + base_pose[:3, 3]
    write_xyz_pcd(dense/'refined/rgb_dense_voxel_1mm.pcd', rgb)
    write_xyz_pcd(geometry/'cluster_00.pcd', joint._transform(rgb, joint._planning_T_base(frame)))
    model_file = tmp_path/'model_9.pcd'; write_xyz_pcd(model_file, points)
    catalog = SimpleNamespace(object_ids=(9,), path=lambda stone: model_file,
                              load=lambda stone: PointCloudData(points))
    base_pose[0, 3] = shift
    def estimate(*args):
        result = output/'foundation_pose_results'
        result.mkdir()
        atomic_json(result/'summary.json', {'status': 'complete'})
        return {9: {'base_link_T_model': base_pose.tolist()}}
    monkeypatch.setattr(foundation_pose_runtime, 'estimate_assigned', estimate)
    def no_icp(*args, **kwargs):
        pytest.fail('learned default invoked ICP')
    monkeypatch.setattr(joint.TabletopFieldMatcher, 'register_best', no_icp)
    monkeypatch.setattr(joint, 'render_joint_pose_result', lambda p: None)
    profile = load_perception_profile(ROOT/'apps/tabletop/config/perception.yml')
    def run():
        return joint.identify_from_rgb_and_femto(dense, region_file, geometry, catalog, profile,
                                                output, assignments={0: 9})
    if shift:
        with pytest.raises(FusedPoseRejected):
            run()
        assert not (output/'pose_data.pkl').exists()
    else:
        assert run().is_file()
    report = json.loads((output/'pose_identification/stone_poses.json').read_text())
    assert report['pose_source'] == 'foundationpose_rgb_femto_validated'
    assert report['icp_refinement_applied'] is False
    if not shift:
        np.testing.assert_allclose(report['poses'][0]['planning_T_model'],
                                   joint._planning_T_base(frame) @ base_pose, atol=0, rtol=0)
    else:
        assert report['rejected'][0]['reasons']


@pytest.mark.parametrize('command', [['run', '--output', 'new'], ['process', 'capture', '--output', 'new'],
                                    ['resume', 'failed'], ['reidentify', 'previous', '--output', 'new']])
def test_all_pick_commands_default_to_foundationpose(command):
    assert build_parser().parse_args(command).pose_backend == 'foundationpose'


def test_missing_runtime_requires_explicit_icp_selection(tmp_path):
    with pytest.raises(ValueError, match='explicitly select --pose-backend icp'):
        foundation_pose_runtime.check_runtime(tmp_path)
