"""Field command, pose export, driver ownership, and operator workflow regression."""
from dataclasses import replace
import json
from pathlib import Path
import signal
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from stacking_tabletop.acquisition import field_workflow as workflow
from stacking_tabletop.perception import field_cli
from stacking_tabletop.perception.fusion import load_recorded_femto_scan
from stacking_tabletop.planning.legacy_pose_data import load_legacy_pick_scene

ROOT = Path(__file__).resolve().parents[1]


def write_cloud(path, points):
    o3d = pytest.importorskip('open3d')
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.array(points, copy=True))
    assert o3d.io.write_point_cloud(str(path), cloud)


@pytest.fixture
def field(tmp_path):
    rng = np.random.default_rng(7)
    model = rng.normal(size=(2400, 3))
    model /= np.linalg.norm(model, axis=1)[:, None]
    model *= [.035, .024, .02]
    pose = np.eye(4)
    pose[:3, 3] = [.13, -.66, .1 - np.quantile(model[:, 2], .01)]
    xy = rng.uniform([-.4, -.85], [.4, -.45], size=(10000, 2))
    scene = np.vstack([np.c_[xy, np.full(len(xy), .1)], model + pose[:3, 3]])
    models = tmp_path / 'models'
    models.mkdir()
    write_cloud(models / 'model_7.pcd', model)
    scan = tmp_path / 'scan'
    scan.mkdir()
    write_cloud(scan / 'field_fused_base_link.pcd', scene * [-1, -1, 1])
    (scan / 'field_scan.json').write_text(json.dumps(dict(output_frame='base_link', camera_serial='A', fused_depth_corrected=True)))
    args = field_cli.build_parser().parse_args([str(scan), '--repository', str(ROOT), '--stone-directory', str(models), '--manual-init-gui'])
    return args, scan, pose


@pytest.mark.parametrize('automatic', [False, True])
def test_identification_exports_planning_poses_without_recorrecting_depth(field, automatic):
    args, scan, seed = field
    args.automatic_ordering = automatic
    args.manual_init_gui = not automatic
    before = (scan / 'field_fused_base_link.pcd').read_bytes()
    initializer = Mock(return_value={7: seed})
    path = field_cli.identify_field(args, scan, ordering=lambda *a: {0: 7}, initializer=initializer)
    picked = load_legacy_pick_scene(path).pose(7)
    np.testing.assert_allclose(picked[:3], seed[:3, 3], atol=.001)
    assert picked[1] < 0  # base_link -> planning world rotates the pickup side.
    assert (scan / 'field_fused_base_link.pcd').read_bytes() == before
    report = json.loads((scan / 'pose_identification/stone_poses.json').read_text())
    assert report['frame_id'] == 'world'
    assert report['status'] == 'complete'
    assert abs(report['table_height_correction_m']) < .001
    if not automatic:
        assert initializer.call_args.kwargs['translation_step_m'] == .005
    with pytest.raises(FileExistsError):
        field_cli.identify_field(args, scan)
    args.identification_output = scan.parent / 'retry'
    retried = field_cli.identify_field(args, scan, ordering=lambda *a: {0: 7}, initializer=initializer)
    assert retried == args.identification_output / 'pose_data.pkl'
    assert path.is_file()


def test_rejected_assignment_never_exports_pickup_file(field, monkeypatch, capsys):
    args, scan, _ = field
    args.manual_init_gui = False
    monkeypatch.setattr(field_cli.TabletopFieldMatcher, 'register_best', lambda *a, **k:
                        NS(decision=NS(accepted=False, reasons=('low_target_coverage',),
                                       correction_rotation_degrees=25.64, correction_translation=.0083),
                           registration=NS(source_coverage=.525, target_coverage=.2,
                                           rmse=.00425, transform=np.eye(4)),
                           initial_transform=np.eye(4)))
    with pytest.raises(ValueError, match=r'rejected assigned stones \[7\]'):
        field_cli.identify_field(args, scan, ordering=lambda *a: {0: 7})
    assert not (scan / 'pose_data.pkl').exists()
    report = json.loads((scan / 'pose_identification/stone_poses.json').read_text())
    assert report['status'] == 'failed'
    assert report['rejected'][0]['stone_id'] == 7
    assert report['rejected'][0]['correction_rotation_degrees'] == 25.64
    assert report['rejected'][0]['rmse_m'] == .00425
    log = capsys.readouterr().out
    assert 'Rejected stone 7 in cluster 0: low_target_coverage' in log
    assert 'target coverage 20.0% (minimum 35.0%)' in log


def test_assignment_and_manual_selection_preserve_legacy_validation():
    from stacking_tabletop.gui.field_stone_ordering_gui import validate_cluster_assignments
    from stacking_tabletop.gui.field_stone_manual_init_gui import selected_manual_assignments, _rotation_transform
    assert validate_cluster_assignments([7, None, 9], {1}) == {0: 7, 2: 9}
    with pytest.raises(ValueError):
        validate_cluster_assignments([7, 7], set())
    assert selected_manual_assignments({2: 14, 0: 3, 1: 9}, {14, 3}) == [(0, 3), (2, 14)]
    center = np.array([.2, -.6, .1])
    transform = _rotation_transform([0, 0, 1], np.pi / 2, center)
    np.testing.assert_allclose(transform[:3, :3] @ center + transform[:3, 3], center)


@pytest.fixture
def run_args(tmp_path, monkeypatch):
    args = workflow.build_parser().parse_args(['--repository', str(ROOT), '--output-dir', str(tmp_path / 'run')])
    # Camera identity and calibration validity have independent acquisition tests.
    monkeypatch.setattr(workflow, 'FemtoCalibration', lambda profile: NS(profile=profile))
    return args


@pytest.mark.parametrize('both', [False, True])
def test_workflow_sequences_capture_fusion_and_identification(run_args, monkeypatch, both):
    args = run_args
    args.both_cameras = both
    events = []
    def capture(args, camera, calibration, directory, log):
        events.append(camera)
        directory.mkdir()
    class Fusion:
        def __init__(self, *args): pass
        def fuse_recorded(self, primary, secondary):
            assert primary.is_dir() and secondary.is_dir()
            events.append('fusion')
            return 'merged-cloud'
    def write(directory, result, serials):
        assert result == 'merged-cloud' and serials == ('CL83C53005L', 'CL8956200P5')
        directory.mkdir()
    def identify(args, directory):
        events.append('identify')
        path = directory / 'pose_data.pkl'
        path.write_bytes(b'test-result')
        return path
    monkeypatch.setattr(workflow, 'write_dual_femto_fusion', write)
    result = workflow.run_workflow(args, capture_runner=capture, pose_runner=identify, fusion_factory=Fusion)
    assert events == (['primary', 'secondary', 'fusion', 'identify'] if both else ['primary', 'identify'])
    assert result.pose_input_dir.name == ('merged' if both else 'primary')
    assert json.loads((result.capture_dir / 'field_workflow.json').read_text())['status'] == 'complete'
    with pytest.raises(FileExistsError):
        workflow.run_workflow(args, capture_runner=capture, pose_runner=identify)


@pytest.mark.parametrize('failure', [RuntimeError('capture failed'), KeyboardInterrupt()])
def test_capture_failure_stops_downstream_and_retains_status(run_args, failure):
    pose = Mock()
    def capture(*args): raise failure
    with pytest.raises(type(failure)):
        workflow.run_workflow(run_args, capture_runner=capture, pose_runner=pose)
    pose.assert_not_called()
    status = json.loads((run_args.output_dir / 'field_workflow.json').read_text())
    assert status['stage'] == 'capture_primary'
    assert status['status'] == ('cancelled' if isinstance(failure, KeyboardInterrupt) else 'failed')


def test_preflight_refuses_ambiguous_driver_ownership(run_args):
    run_args.both_cameras = run_args.use_running_camera = True
    with pytest.raises(ValueError, match='owned drivers'):
        workflow.run_workflow(run_args)
    assert not run_args.output_dir.exists()


@pytest.mark.parametrize('fail', [False, True])
def test_owned_driver_is_stopped_on_success_or_error(tmp_path, monkeypatch, fail):
    process = Mock(pid=12345)
    process.poll.return_value = None
    factory = Mock(return_value=process)
    signals = []
    def killpg(pid, sig):
        assert pid == process.pid
        if sig == 0: raise ProcessLookupError
        signals.append(sig)
    monkeypatch.setattr(workflow.shutil, 'which', lambda name: '/opt/ros/jazzy/bin/ros2')
    monkeypatch.setattr(workflow.os, 'killpg', killpg)
    def run():
        with workflow.camera_driver(ROOT, ROOT / 'apps/tabletop/config/femto.yml', 'primary', tmp_path / 'driver.log', process_factory=factory):
            if fail: raise RuntimeError('camera unavailable')
    if fail:
        with pytest.raises(RuntimeError): run()
    else:
        run()
    assert factory.call_args.kwargs['start_new_session']
    assert signals == [signal.SIGINT]
    assert process.wait.called


def test_merged_capture_can_be_loaded_for_identification(tmp_path):
    from stacking_common.pointcloud import PointCloudData
    from stacking_tabletop.perception.fusion import write_dual_femto_fusion
    result = NS(cloud=PointCloudData([[.1, .5, .1], [.2, .5, .12]]), secondary_to_primary=np.eye(4),
                registration=NS(target_coverage=.9, source_coverage=.8, rmse=.001))
    write_dual_femto_fusion(tmp_path / 'merged', result, ('A', 'B'))
    scan = load_recorded_femto_scan(tmp_path / 'merged')
    assert scan.frame_id == 'base_link'
    np.testing.assert_allclose(scan.cloud.points, result.cloud.points)
