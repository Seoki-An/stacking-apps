"""Wrist-only workflow contracts with synthetic RGB geometry and learned poses."""
import json
from pathlib import Path

import numpy as np
import pytest

from stacking_tabletop.acquisition import foundation_pose_runtime, pick_scene_rgb_first, pick_scene_workflow
from stacking_tabletop.acquisition.hand_eye_capture import atomic_json
from stacking_tabletop.acquisition.hand_eye_replay import assemble_session
from stacking_tabletop.acquisition.storage import write_xyz_pcd
from stacking_tabletop.acquisition.wrist_pick_regions import region_matches_dense
from stacking_tabletop.config import load_perception_profile
from stacking_tabletop.perception.catalog import TabletopPcdCatalog
from stacking_tabletop.gui.operations.field_scan import inspect_joint_pick_output
from test_tabletop_pick_scene_workflow import complete_capture

ROOT = Path(__file__).resolve().parents[1]


def test_wrist_only_reconstruct_segment_identify_publish_and_reidentify(complete_capture, tmp_path, monkeypatch):
    capture = tmp_path / 'capture'
    capture.mkdir()
    source = complete_capture / 'session'
    manifest = json.loads((source / 'session.json').read_text())
    assemble_session(capture / 'session', [source / v['path'] for v in manifest['views']])
    atomic_json(capture / 'capture.json', dict(status='complete', sensor_mode='wrist-only'))
    assert json.loads((capture / 'session/session.json').read_text())['references'] == []
    with pytest.raises(ValueError, match='requires FEMTO'):
        pick_scene_workflow._verify_capture(capture)

    models = tmp_path / 'models'
    models.mkdir()
    model = np.random.default_rng(12).uniform(-.02, .02, (3000, 3))
    write_xyz_pcd(models / 'model_9.pcd', model)
    catalog = TabletopPcdCatalog(models, [9])
    profile = load_perception_profile(ROOT / 'apps/tabletop/config/perception.yml')
    pose = np.eye(4)
    pose[:3, 3] = [.1, .6, .15]
    x, y = np.meshgrid(np.linspace(-.2, .2, 100), np.linspace(.45, .85, 100))
    table = np.c_[x.ravel(), y.ravel(), np.full(x.size, .11)]
    events = []

    def forbidden(*args, **kwargs):
        pytest.fail('wrist-only workflow accessed FEMTO')
    monkeypatch.setattr(pick_scene_rgb_first, 'prepare_femto_regions', forbidden)
    monkeypatch.setattr(pick_scene_rgb_first, 'validate_selected_femto_stability', forbidden)
    monkeypatch.setattr(pick_scene_rgb_first, 'load_recorded_femto_scan', forbidden)
    monkeypatch.setattr(foundation_pose_runtime, 'check_runtime', lambda repository: None)
    monkeypatch.setattr('stacking_tabletop.gui.field_stone_ordering_gui.assign_cluster_ids_gui',
                        lambda clusters, ids, suggested: {0: 9})

    def reconstruct(session, baseline, refinement, output, *, region_file):
        events.append('reconstruct')
        assert baseline is refinement is None
        (output / 'refined').mkdir(parents=True)
        write_xyz_pcd(output / 'refined/rgb_dense_voxel_1mm.pcd', np.r_[table, model + pose[:3, 3]])
        atomic_json(output / 'summary.json', dict(status='complete', modes={'refined': {}},
                    kind='fast_foundation_stereo_experiment', data_origin='synthetic',
                    region=json.loads(region_file.read_text())))
        (output / 'rgb_only').mkdir()
        (output / 'rgb_only/overview.html').write_text('<html>Synthetic RGB fixture</html>')

    def estimate(dense, region, assignments, catalog, output, repository):
        events.append('identify')
        assert region_matches_dense(dense, json.loads(region.read_text()),
                                    json.loads((dense / 'summary.json').read_text()))
        assert assignments == {0: 9}
        (output / 'foundation_pose_results').mkdir()
        atomic_json(output / 'foundation_pose_results/summary.json', dict(status='complete'))
        return {9: {'base_link_T_model': pose.tolist()}}
    monkeypatch.setattr(foundation_pose_runtime, 'estimate_assigned', estimate)
    output = tmp_path / 'result'
    args = pick_scene_workflow.build_parser().parse_args([
        'process', str(capture), '--output', str(output), '--sensor-mode', 'wrist-only'])
    result = pick_scene_workflow.run_rgb_first_workflow(args, dense_runner=reconstruct,
                                                       preflight=lambda a: (profile, catalog))
    assert events == ['reconstruct', 'identify']
    assert result.pose_data.is_file()
    status = json.loads((output / 'pick_scene_workflow.json').read_text())
    assert status['scene_stability'] == 'not_evaluated_without_femto'
    assert not (output / 'femto_geometry').exists()
    report = json.loads((output / 'pose_identification/stone_poses.json').read_text())
    assert report['table_height_correction_m'] == pytest.approx(-.01, abs=1e-5)
    assert report['pose_source'] == 'foundationpose_wrist_rgb'
    assert report['poses'][0]['femto_used_points'] == 0
    assert set(report['poses'][0]['sensors']) == {'rgb'}
    inspected = inspect_joint_pick_output(output)
    assert inspected[0] == result.pose_data and inspected[1].is_file() and inspected[2].is_file()

    reviewed = tmp_path / 'reviewed'
    args = pick_scene_workflow.build_parser().parse_args(['reidentify', str(output), '--output', str(reviewed)])
    result = pick_scene_workflow.resume_joint_workflow(args, preflight=lambda a: (profile, catalog))
    assert events == ['reconstruct', 'identify', 'identify']
    assert inspect_joint_pick_output(reviewed)[0] == result.pose_data

    dense = output / 'rgb_dense'
    region = json.loads((output / 'rgb_region.json').read_text())
    summary = json.loads((dense / 'summary.json').read_text())
    assert region_matches_dense(dense, region, summary)
    with (dense / 'refined/rgb_dense_voxel_1mm.pcd').open('a') as stream:
        stream.write('\n')
    assert not region_matches_dense(dense, region, summary)
