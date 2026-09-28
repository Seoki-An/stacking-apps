"""Integrated wrist-RGB/FEMTO pick workflow and fixed-pose validation."""

import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

import stacking_tabletop.acquisition.pick_scene_workflow as workflow
from stacking_tabletop.acquisition.hand_eye_synthetic import generate_demo
from stacking_tabletop.acquisition.pick_scene_workflow import (
    _fixed_pose_metrics,
    build_parser,
    build_rgb_region,
    _run_femto_first_workflow as run_legacy_workflow,
    run_workflow,
    validate_femto_stability,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_pose_runtime(monkeypatch):
    # Workflow sequencing tests do not require installed GPU model weights.
    from stacking_tabletop.acquisition import foundation_pose_runtime
    monkeypatch.setattr(foundation_pose_runtime, 'check_runtime', lambda repository: None)


@pytest.fixture(scope='module')
def complete_capture(tmp_path_factory):
    root = tmp_path_factory.mktemp('pick-workflow') / 'fixture'
    generate_demo(root, visualize=False)
    capture = root / 'capture'
    capture.mkdir()
    (capture / 'session').symlink_to(root / 'session', target_is_directory=True)
    (capture / 'femto_before').symlink_to(root / 'inputs/femto_before', target_is_directory=True)
    (capture / 'femto_after').symlink_to(root / 'inputs/femto_after', target_is_directory=True)
    (capture / 'capture.json').write_text(json.dumps({'status': 'complete'}))
    return capture


def workflow_args(capture, output):
    return build_parser().parse_args(['process', str(capture), '--repository', str(ROOT),
                                      '--output', str(output), '--automatic-ordering'])


def test_live_run_materializes_standard_route_and_uses_its_hash(tmp_path):
    output = tmp_path / 'live'
    args = build_parser().parse_args(['run', '--repository', str(ROOT), '--output', str(output)])
    assert not hasattr(args, 'plan') and not hasattr(args, 'approve')

    def preflight(_args):
        return NS(), NS(model_directory=tmp_path / 'models')

    def plan(values):
        assert values.distance == .4 and values.offset == .04 and values.oblique_offset == .08
        assert values.viewpoint_pattern == 'coverage13'
        values.output.mkdir()
        (values.output / 'plan.json').write_text('{}')
        return {'approval_sha256': workflow.digest(values.output / 'plan.json')}

    class CaptureReached(RuntimeError):
        pass

    def capture(values):
        assert values.plan == output / 'capture_plan'
        assert values.approve == workflow.digest(values.plan / 'plan.json')
        raise CaptureReached

    with pytest.raises(CaptureReached):
        run_workflow(args, plan_runner=plan, capture_runner=capture, preflight=preflight)
    status = json.loads((output / 'pick_scene_workflow.json').read_text())
    assert status['capture_route'] == 'standard_coverage13'
    assert status['capture_plan_dir'] == str(output / 'capture_plan')


def test_process_sequences_identification_mvs_validation_and_atomic_publish(complete_capture, tmp_path):
    output = tmp_path / 'result'
    args = workflow_args(complete_capture, output)
    events = []

    def preflight(_args):
        return NS(field_registration='registration', field_quality='quality'), NS(model_directory=tmp_path / 'models')

    def identify(args, scan):
        events.append(('identify', scan.name))
        root = args.identification_output
        report = root / 'pose_identification'
        report.mkdir(parents=True)
        (report / 'stone_poses.json').write_text(json.dumps({'status': 'complete', 'poses': [{'stone_id': 7}]}))
        pose = root / 'pose_data.pkl'
        pose.write_bytes(b'accepted-femto-pose')
        return pose

    def region(_capture, _identification, path):
        events.append(('region', path.name))
        path.write_text('{}')

    def sparse(_session, _region, path):
        events.append(('sparse', path.name)); path.mkdir()

    def stable(_capture, _identification, _region, _profile, path):
        events.append(('stability', path.name)); path.mkdir()
        (path / 'summary.json').write_text(json.dumps({'status': 'complete', 'scene_stable': True}))
        return {'status': 'complete', 'scene_stable': True, 'stones': []}

    def refine(_session, _sparse, path):
        events.append(('refine', path.name)); path.mkdir()

    def dense(_session, _sparse, _refined, path):
        events.append(('dense', path.name)); path.mkdir()

    def fuse(_dense, _region, _femto, _models, _profile, path, *, visualize):
        events.append(('fuse', visualize))
        report = path / 'pose_identification'
        report.mkdir(parents=True)
        (report / 'stone_poses.json').write_text(json.dumps({
            'status': 'complete', 'pose_source': 'balanced_femto_rgb_joint_icp',
            'poses': [{'stone_id': 7}],
            'warnings': [{'stone_id': 7, 'correction_rotation_degrees': 12.0,
                          'reasons': ['combined_large_rotation_correction']}]}))
        pose = path / 'pose_data.pkl'
        pose.write_bytes(b'accepted-fused-pose')
        return pose

    def validate(_dense, _region, _identification, _models, _profile, path, *, visualize):
        events.append(('validate', visualize)); path.mkdir()
        report = {'status': 'complete', 'production_pose_accepted': True,
                  'stones': [{'stone_id': 7, 'accepted': True}]}
        (path / 'summary.json').write_text(json.dumps(report))
        return report

    ticks = iter(range(20))
    result = run_legacy_workflow(args, identify_runner=identify, region_runner=region,
                          sparse_runner=sparse, refine_runner=refine, dense_runner=dense,
                          fusion_runner=fuse, stability_runner=stable,
                          validation_runner=validate, preflight=preflight,
                          clock=lambda: next(ticks))
    assert [event[0] for event in events] == [
        'identify', 'region', 'stability', 'sparse', 'refine', 'dense', 'fuse', 'validate']
    assert result.pose_data.read_bytes() == b'accepted-fused-pose'
    assert (output / 'pose_identification/stone_poses.json').is_file()
    status = json.loads((output / 'pick_scene_workflow.json').read_text())
    assert status['status'] == 'complete' and status['production_pose_accepted']
    assert status['scene_stability'] == 'verified_repeatable'
    assert status['pose_warnings'][0]['stone_id'] == 7
    published = json.loads((output / 'pick_scene_poses.json').read_text())
    assert published['pose_warnings'] == status['pose_warnings']
    assert not status['dense_fallback_used']
    assert all(stage['elapsed_s'] == 1 for stage in status['stages'])


def test_rgb_rejection_withholds_planning_pose(complete_capture, tmp_path):
    output = tmp_path / 'rejected'
    args = workflow_args(complete_capture, output)

    def preflight(_args):
        return NS(), NS(model_directory=tmp_path / 'models')

    def identify(args, _scan):
        report = args.identification_output / 'pose_identification'
        report.mkdir(parents=True)
        (report / 'stone_poses.json').write_text(json.dumps({'status': 'complete', 'poses': [{'stone_id': 9}]}))
        pose = args.identification_output / 'pose_data.pkl'; pose.write_bytes(b'unpublished'); return pose

    def make(*values):
        path = Path(values[-1]); path.mkdir() if path.suffix == '' else path.write_text('{}')

    def fuse(_dense, _region, _femto, _models, _profile, path, **_kwargs):
        report = path / 'pose_identification'
        report.mkdir(parents=True)
        (report / 'stone_poses.json').write_text(json.dumps({
            'status': 'complete', 'pose_source': 'balanced_femto_rgb_joint_icp',
            'poses': [{'stone_id': 9}]}))
        pose = path / 'pose_data.pkl'; pose.write_bytes(b'fused-unpublished'); return pose

    def reject(*values, **_kwargs):
        path = Path(values[-1]); path.mkdir()
        report = {'status': 'rejected', 'production_pose_accepted': False,
                  'stones': [{'stone_id': 9, 'accepted': False}]}
        (path / 'summary.json').write_text(json.dumps(report)); return report

    def stable(*values):
        path = Path(values[-1]); path.mkdir()
        return {'status': 'complete', 'scene_stable': True, 'stones': []}

    with pytest.raises(ValueError, match='stones 9'):
        run_legacy_workflow(args, identify_runner=identify, region_runner=make,
                     sparse_runner=make, refine_runner=make, dense_runner=make,
                     fusion_runner=fuse, stability_runner=stable,
                     validation_runner=reject, preflight=preflight)
    assert not (output / 'pose_data.pkl').exists()
    assert json.loads((output / 'pick_scene_workflow.json').read_text())['status'] == 'failed'


def test_region_returns_corrected_identification_to_raw_base_frame(tmp_path):
    o3d = pytest.importorskip('open3d')
    capture = tmp_path / 'capture/femto_before'
    capture.mkdir(parents=True)
    (capture / 'field_scan.json').write_text(json.dumps({'base_frame_crop': {
        'min': [-.5, .1, .02], 'max': [.5, .8, .3]}}))
    identification = tmp_path / 'identification/pose_identification'
    identification.mkdir(parents=True)
    planning_T_base = np.diag([-1., -1., 1., 1.])
    pose = np.eye(4); pose[:3, 3] = [.1, -.2, .12]
    report = {'status': 'complete', 'planning_T_base_link': planning_T_base.tolist(),
              'table_height_correction_m': .02, 'fitted_table_plane': [0., 0., 1., -.1],
              'poses': [{'stone_id': 7, 'cluster_index': 0, 'planning_T_model': pose.tolist()}]}
    (identification / 'stone_poses.json').write_text(json.dumps(report))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector([[.09, -.21, .11], [.11, -.19, .13]])
    assert o3d.io.write_point_cloud(str(identification / 'cluster_00.pcd'), cloud)
    region = build_rgb_region(tmp_path / 'capture', tmp_path / 'identification', tmp_path / 'region.json')
    np.testing.assert_allclose(region['support_plane'], [0., 0., .08])
    np.testing.assert_allclose(np.asarray(region['stones'][0]['base_link_T_model'])[:3, 3], [-.1, .2, .1])
    low, high = np.asarray(region['stone_boxes_xy'][0])
    assert np.all(low < [-.1, .2]) and np.all(high > [-.1, .2])


def test_fixed_pose_metrics_detect_alignment_and_misalignment():
    rng = np.random.default_rng(4)
    model = rng.normal(scale=.01, size=(300, 3))
    transform = np.eye(4); transform[:3, 3] = [.1, .2, .3]
    observed = model + transform[:3, 3]
    good = _fixed_pose_metrics(model, observed, transform, .002)
    assert good['model_coverage'] == good['observed_coverage'] == 1
    assert good['rmse_m'] < 1e-10
    bad = _fixed_pose_metrics(model, observed + [.02, 0, 0], transform, .002)
    assert bad['model_coverage'] < .1 and bad['observed_coverage'] < .1


def test_femto_bracket_stability_accepts_repeat_and_rejects_movement(tmp_path, monkeypatch):
    capture = tmp_path / 'capture/femto_after'
    capture.mkdir(parents=True)
    (capture / 'field_scan.json').write_text('{}')
    identification = tmp_path / 'identification/pose_identification'
    identification.mkdir(parents=True)
    (identification / 'stone_poses.json').write_text(json.dumps({
        'planning_T_base_link': np.eye(4).tolist(), 'table_height_correction_m': 0.}))
    region = tmp_path / 'region.json'
    region.write_text(json.dumps({'support_plane': [0., 0., 0.],
                                  'stone_boxes_xy': [[[-.1, -.1], [.1, .1]]],
                                  'stones': [{'stone_id': 9, 'cluster_index': 0}]}))
    rng = np.random.default_rng(7)
    before = np.c_[rng.uniform(-.02, .02, (100, 2)), rng.uniform(.02, .04, 100)]
    monkeypatch.setattr(workflow, '_read_cloud', lambda _path: before)
    observed = before + rng.normal(scale=.0002, size=before.shape)
    scan = NS(frame_id='base_link', cloud=NS(points=observed))
    monkeypatch.setattr(workflow, 'load_recorded_femto_scan', lambda _path: scan)
    profile = NS(field_registration=NS(evaluation_distance=.002, minimum_points=10),
                 field_quality=NS(minimum_target_coverage=.8, minimum_source_coverage=.8,
                                  maximum_rmse=.002))
    accepted = validate_femto_stability(tmp_path / 'capture', tmp_path / 'identification',
                                        region, profile, tmp_path / 'stable')
    assert accepted['scene_stable'] and accepted['stones'][0]['stable']

    moved = NS(frame_id='base_link', cloud=NS(points=observed + [.04, 0., 0.]))
    monkeypatch.setattr(workflow, 'load_recorded_femto_scan', lambda _path: moved)
    rejected = validate_femto_stability(tmp_path / 'capture', tmp_path / 'identification',
                                        region, profile, tmp_path / 'moved')
    assert not rejected['scene_stable'] and not rejected['stones'][0]['stable']


def test_joint_observation_balances_sensor_density():
    from stacking_tabletop.acquisition.pick_scene_fusion import _balanced_observation

    femto = np.column_stack((np.arange(20) * .002, np.zeros(20), np.zeros(20)))
    rgb = np.column_stack((np.arange(80) * .002, np.ones(80), np.zeros(80)))
    femto_voxel, rgb_voxel, combined = _balanced_observation(
        femto, rgb, voxel_size=.001, minimum_points=10)
    expected = min(len(femto_voxel), len(rgb_voxel))
    assert combined.shape == (2 * expected, 3)
    assert np.allclose(combined[:expected, 1], 0.)
    assert np.allclose(combined[expected:, 1], 1.)


def test_production_dense_uses_refined_pose_and_coverage_views(monkeypatch):
    import stacking_tabletop.acquisition.hand_eye_dense as dense

    called = {}

    def reconstruct(*args, **kwargs):
        called.update(args=args, kwargs=kwargs)
        return {'status': 'complete'}

    monkeypatch.setattr(dense, 'dense_rgb_session', reconstruct)
    monkeypatch.setattr(dense, 'production_reference_views', lambda _session: (0, 5, 6, 7, 8))
    result = dense.dense_rgb_production_session('session', 'baseline', 'refinement', 'output')
    assert result == {'status': 'complete'}
    assert called['args'] == ('session', 'baseline', 'refinement', 'output')
    assert called['kwargs'] == {
        'mode_names': ('refined',),
        'reference_views': (0, 5, 6, 7, 8),
        'crop_to_stones': True,
        'maximum_sources': 4,
    }


def test_dense_workspace_crops_pixels_and_rebuilds_each_camera(tmp_path):
    pycolmap = pytest.importorskip('pycolmap')
    from PIL import Image
    from stacking_common.multiview.dense import write_pinhole_cameras
    from stacking_tabletop.acquisition.hand_eye_dense import _prepare_dense_workspace

    names = ['a.png', 'b.png', 'c.png']
    matrix = np.array([[100., 0, 64.5], [0, 100., 48.5], [0, 0, 1.]])
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, 0, 3] = [-.04, 0., .04]
    image = np.arange(128*96*3, dtype=np.uint8).reshape(96, 128, 3)
    (tmp_path/'images').mkdir()
    for name in names:
        Image.fromarray(image).save(tmp_path/'images'/name)
    write_pinhole_cameras(tmp_path/'sparse', [matrix]*3, [(128, 96)]*3, poses, names)
    crops, sources = _prepare_dense_workspace(
        tmp_path, names, [[[-.03, -.03, .4], [.03, .03, .5]]],
        crop_images=True, maximum_sources=2)
    rebuilt = pycolmap.Reconstruction(tmp_path/'sparse')
    for index, item in enumerate(sorted(rebuilt.images.values(), key=lambda value:value.name)):
        x0, y0, x1, y1 = crops[index]['crop_xyxy']
        with Image.open(tmp_path/'images'/item.name) as cropped:
            np.testing.assert_array_equal(np.asarray(cropped), image[y0:y1, x0:x1])
        camera = rebuilt.cameras[item.camera_id]
        expected = matrix.copy(); expected[:2, 2] -= [x0, y0]
        np.testing.assert_allclose(camera.calibration_matrix(), expected)
        assert (camera.width, camera.height) == (x1-x0, y1-y0)
        np.testing.assert_allclose(item.cam_from_world().matrix(), np.linalg.inv(poses[index])[:3])
        assert set(sources[index]) == set(range(3)) - {index}


@pytest.mark.parametrize('rejection', ['fusion', 'validation', 'both_attempts'])
def test_fast_pose_rejection_retries_once_and_publishes_only_accepted_output(
        complete_capture, tmp_path, rejection):
    from stacking_tabletop.acquisition.pick_scene_fusion import FusedPoseRejected
    output = tmp_path/'retry'
    args = workflow_args(complete_capture, output)
    attempts = []

    def identify(values, _scan):
        path = values.identification_output
        path.mkdir()
        (path/'pose_data.pkl').write_bytes(b'femto')
        return path/'pose_data.pkl'

    def create(*values):
        path = Path(values[-1])
        if path.suffix:
            path.write_text('{}')
        else:
            path.mkdir()

    def dense(*values):
        attempts.append(Path(values[-1]).name)
        create(*values)

    def fuse(_dense, _region, _femto, _models, _profile, path, **kwargs):
        report = path/'pose_identification'
        report.mkdir(parents=True)
        if rejection == 'both_attempts' or (rejection == 'fusion' and not path.name.endswith('_full')):
            (report/'stone_poses.json').write_text(json.dumps({'status':'failed','rejected':[{'stone_id':9}]}))
            raise FusedPoseRejected('stone 9 coverage check')
        (report/'stone_poses.json').write_text(json.dumps({'status':'complete','pose_source':'joint'}))
        (path/'pose_data.pkl').write_bytes(path.name.encode())
        return path/'pose_data.pkl'

    def validate(_dense, _region, _identification, _models, _profile, path, **kwargs):
        path.mkdir()
        accepted = path.name.endswith('_full')
        result = {'production_pose_accepted':accepted,'stones':[{'stone_id':9,'accepted':accepted}]}
        (path/'summary.json').write_text(json.dumps(result))
        return result

    def execute():
        return run_legacy_workflow(args, identify_runner=identify, region_runner=create,
            sparse_runner=create, refine_runner=create, dense_runner=dense,
            fallback_dense_runner=dense, fusion_runner=fuse, validation_runner=validate,
            stability_runner=lambda *values: {'scene_stable':True},
            preflight=lambda values: (NS(), NS(model_directory=tmp_path/'models')))

    if rejection == 'both_attempts':
        with pytest.raises(FusedPoseRejected):
            execute()
        assert not (output/'pose_data.pkl').exists()
    else:
        result = execute()
        assert result.pose_data.read_bytes() == b'fused_identification_full'
        published = json.loads((output/'pick_scene_poses.json').read_text())
        assert published['rgb_validation'] == 'rgb_validation_full/summary.json'
        assert result.validation == output/'rgb_validation_full/summary.json'
    assert attempts == ['rgb_dense', 'rgb_dense_full']
    manifest = json.loads((output/'pick_scene_workflow.json').read_text())
    assert manifest['dense_fallback_used']
    assert manifest['dense_fallback_reason']
    assert (output/'fused_identification/pose_identification/stone_poses.json').exists()


def test_fusion_rotation_outlier_warns_but_other_quality_failures_reject():
    from stacking_tabletop.acquisition.pick_scene_fusion import _classify_pose_reasons
    decisions = {
        'combined': NS(reasons=('large_rotation_correction',)),
        'rgb': NS(reasons=('large_rotation_correction', 'low_target_coverage')),
        'femto': NS(reasons=()),
    }
    rejected, warnings = _classify_pose_reasons(decisions)
    assert rejected == ['rgb_low_target_coverage']
    assert warnings == ['combined_large_rotation_correction', 'rgb_large_rotation_correction']


def test_public_workflow_keeps_rgb_first_with_custom_dense_runner(tmp_path, monkeypatch):
    args = workflow_args(tmp_path / 'capture', tmp_path / 'output')
    marker = object()
    observed = {}

    def rgb_first(received_args, **kwargs):
        observed.update(kwargs)
        assert received_args is args
        return marker

    monkeypatch.setattr(workflow, 'run_rgb_first_workflow', rgb_first)
    dense = lambda *_args: None
    assert run_workflow(args, dense_runner=dense) is marker
    assert observed['dense_runner'] is dense
    with pytest.raises(ValueError, match='legacy stage overrides'):
        run_workflow(args, identify_runner=lambda *_args: None)


def test_joint_identification_rejects_rgb_reconstructed_with_other_regions(tmp_path):
    from stacking_tabletop.acquisition.pick_scene_rgb_first import identify_from_rgb_and_femto
    dense = tmp_path / 'dense'
    geometry = tmp_path / 'geometry'
    dense.mkdir(); geometry.mkdir()
    (dense / 'summary.json').write_text(json.dumps({
        'status': 'complete', 'modes': {'refined': {}}, 'region': {'stone_boxes_xy': []}}))
    region = tmp_path / 'region.json'
    region.write_text(json.dumps({'stone_boxes_xy': [[[0, 0], [1, 1]]]}))
    (geometry / 'geometry.json').write_text(json.dumps({'catalog_pose_fitted': False}))
    with pytest.raises(ValueError, match='different coarse FEMTO regions'):
        identify_from_rgb_and_femto(dense, region, geometry, None, None, tmp_path / 'out')


def test_manual_gui_cloud_copies_immutable_catalog_points():
    from stacking_tabletop.acquisition.pick_scene_rgb_first import _editable_open3d_cloud
    source = np.arange(12., dtype=np.float64).reshape(4, 3)
    source.setflags(write=False)
    cloud = _editable_open3d_cloud(source)
    editable = np.asarray(cloud.points)
    assert editable.flags.writeable
    editable[0, 0] = 99.
    assert source[0, 0] == 0.


def test_resume_rejects_unfinished_dense_stage(tmp_path):
    from stacking_tabletop.acquisition.pick_scene_workflow import resume_joint_workflow
    output = tmp_path / 'failed'
    output.mkdir()
    (output / 'pick_scene_workflow.json').write_text(json.dumps({
        'schema_version': 2, 'status': 'failed', 'stage': 'rgb_dense',
        'production_pose_accepted': False}))
    args = build_parser().parse_args(['resume', str(output), '--repository', str(ROOT)])
    with pytest.raises(ValueError, match='joint pose identification'):
        resume_joint_workflow(args)


@pytest.mark.parametrize('failure', [None, ValueError('fit rejected'), KeyboardInterrupt()])
def test_reidentify_reuses_clouds_reviews_ids_and_preserves_previous_result(
        complete_capture, tmp_path, monkeypatch, failure):
    import stacking_tabletop.acquisition.pick_scene_rgb_first as rgb_first
    source = tmp_path / 'accepted'
    dense = source / 'rgb_dense_full'
    (dense / 'refined/workspace').mkdir(parents=True)
    (dense / 'summary.json').write_text('{"status":"complete","modes":{"refined":{}}}')
    (dense / 'refined/rgb_dense_voxel_1mm.pcd').write_bytes(b'saved-rgb')
    (dense / 'refined/workspace/large-intermediate').write_bytes(b'not-needed')
    (dense / 'refined/pair_000_001.npz').write_bytes(b'learned-depth')
    (dense / 'refined/pair_000_001.png').write_bytes(b'rectified-rgb')
    geometry = source / 'femto_geometry'
    geometry.mkdir()
    (geometry / 'geometry.json').write_text('{"catalog_pose_fitted":false}')
    (geometry / 'cluster_00.pcd').write_bytes(b'saved-femto')
    (source / 'rgb_region.json').write_text('{}')
    report_dir = source / 'pose_identification'
    report_dir.mkdir()
    report = dict(status='complete', pose_source='rgb_femto_first_joint_icp',
                  kind='femto_rgb_joint_pose_identification',
                  cluster_assignments={'0': 9}, source_sha256={
                      'dense_summary.json': workflow.digest(dense / 'summary.json'),
                      'femto_geometry.json': workflow.digest(geometry / 'geometry.json'),
                      'rgb_region.json': workflow.digest(source / 'rgb_region.json')})
    (report_dir / 'stone_poses.json').write_text(json.dumps(report))
    (source / 'pose_data.pkl').write_bytes(b'previous-pose')
    (source / 'pick_scene_workflow.json').write_text(json.dumps(dict(
        schema_version=2, status='complete', production_pose_accepted=True,
        capture_dir=str(complete_capture), dense_dir=str(dense), data_origin='synthetic',
        pose_sequence='femto_regions_rgb_mvs_joint_catalog_fit')))
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    output = tmp_path / 'reviewed'
    args = build_parser().parse_args(['reidentify', str(source), '--output', str(output), '--pose-backend', 'icp'])
    events = []

    def identify(cached, region, coarse, catalog, profile, result, **kwargs):
        events.append('identify')
        assert kwargs['suggested_assignments'] == {'0': 9}
        assert (cached / 'refined/rgb_dense_voxel_1mm.pcd').read_bytes() == b'saved-rgb'
        assert (coarse / 'cluster_00.pcd').read_bytes() == b'saved-femto'
        assert not (cached / 'refined/workspace').exists()
        assert (cached / 'refined/pair_000_001.npz').read_bytes() == b'learned-depth'
        assert (cached / 'refined/pair_000_001.png').read_bytes() == b'rectified-rgb'
        if failure is not None:
            raise failure
        result.mkdir()
        (result / 'pose_identification').mkdir()
        report['cluster_assignments'] = {'0': 11}
        (result / 'pose_identification/stone_poses.json').write_text(json.dumps(report))
        pose = result / 'pose_data.pkl'
        pose.write_bytes(b'reviewed-pose')
        return pose

    monkeypatch.setattr(rgb_first, 'identify_from_rgb_and_femto', identify)
    monkeypatch.setattr(rgb_first, 'validate_selected_femto_stability',
                        lambda *a: events.append('stability') or {'scene_stable': True})
    monkeypatch.setattr(workflow, 'validate_rgb_poses',
                        lambda *a, **kw: events.append('validate') or {'production_pose_accepted': True})
    preflight = lambda args: (NS(), NS(model_directory=tmp_path / 'models'))
    if failure is None:
        result = workflow.resume_joint_workflow(args, preflight=preflight)
        assert result.pose_data.read_bytes() == b'reviewed-pose'
        assert events == ['identify', 'stability', 'validate']
        published = json.loads((output / 'pick_scene_poses.json').read_text())
        assert published['data_origin'] == 'synthetic'
        assert published['production_pose_accepted']
        from stacking_tabletop.gui.operations.field_scan import inspect_joint_pick_output
        assert inspect_joint_pick_output(output)[0] == result.pose_data
    else:
        with pytest.raises(type(failure)):
            workflow.resume_joint_workflow(args, preflight=preflight)
        assert not (output / 'pose_data.pkl').exists()
        assert events == ['identify']
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()} == before
    manifest = json.loads((output / 'pick_scene_workflow.json').read_text())
    assert manifest['reidentification_source'] == str(source)
    assert manifest['resume_dense_dir'] == str(output / 'rgb_dense')


def test_fast_foundation_backend_reaches_rgb_reconstruction_before_pose_fit(complete_capture, tmp_path, monkeypatch):
    from stacking_tabletop.acquisition import hand_eye_fast_stereo, hand_eye_rgb, hand_eye_rgb_refinement, pick_scene_rgb_first
    args = build_parser().parse_args(['process', str(complete_capture), '--output', str(tmp_path/'fast')])
    monkeypatch.setattr(pick_scene_rgb_first, 'prepare_femto_regions', lambda *a: None)
    monkeypatch.setattr(hand_eye_rgb, 'rgb_experiment', lambda *a: None)
    monkeypatch.setattr(hand_eye_rgb_refinement, 'refine_rgb_session', lambda *a: None)
    calls = []
    def reconstruct(session, baseline, refinement, output, **kwargs):
        calls.append((session, output, kwargs))
        raise RuntimeError('reached learned stereo')
    monkeypatch.setattr(hand_eye_fast_stereo, 'reconstruct_with_runtime', reconstruct)
    with pytest.raises(RuntimeError, match='reached learned stereo'):
        workflow.run_rgb_first_workflow(args, preflight=lambda a: (NS(), NS(model_directory=tmp_path)))
    assert calls[0][0] == complete_capture/'session'
    assert calls[0][1] == tmp_path/'fast/rgb_dense'
    assert calls[0][2]['checkpoint'].name == 'model_best_bp2_serialize.pth'
    status = json.loads((tmp_path/'fast/pick_scene_workflow.json').read_text())
    assert status['dense_backend'] == 'fast-foundation-stereo'
    assert status['stage'] == 'rgb_dense'
    assert not (tmp_path/'fast/pose_data.pkl').exists()


@pytest.mark.parametrize('pose_backend', ['icp', 'foundationpose'])
def test_rgb_first_only_icp_retries_colmap_after_pose_rejection(complete_capture, tmp_path, monkeypatch, pose_backend):
    from stacking_tabletop.acquisition import hand_eye_dense, hand_eye_fast_stereo, hand_eye_rgb, hand_eye_rgb_refinement, pick_scene_rgb_first
    from stacking_tabletop.acquisition.pick_scene_fusion import FusedPoseRejected
    args = build_parser().parse_args(['process', str(complete_capture), '--output', str(tmp_path/'retry-default'), '--pose-backend', pose_backend])
    assert args.dense_backend == 'fast-foundation-stereo'
    assert build_parser().parse_args(['run', '--output', 'unused', '--dense-backend', 'colmap']).dense_backend == 'colmap'
    for module, name in [(pick_scene_rgb_first, 'prepare_femto_regions'),
                         (hand_eye_rgb, 'rgb_experiment'), (hand_eye_rgb_refinement, 'refine_rgb_session')]:
        monkeypatch.setattr(module, name, lambda *a: None)
    attempts = []
    def fast(*a, **kw):
        attempts.append('fast')
    def colmap(*a, **kw):
        attempts.append('colmap')
    monkeypatch.setattr(hand_eye_fast_stereo, 'reconstruct_with_runtime', fast)
    monkeypatch.setattr(hand_eye_dense, 'dense_rgb_full_session', colmap)
    def identify(dense, region, geometry, catalog, profile, output, **kw):
        if not str(dense).endswith('_full'):
            raise FusedPoseRejected('test quality rejection')
        report = output/'pose_identification'
        report.mkdir(parents=True)
        (report/'stone_poses.json').write_text(json.dumps({'status': 'complete', 'warnings': []}))
        return output/'pose_data.pkl'
    monkeypatch.setattr(pick_scene_rgb_first, 'identify_from_rgb_and_femto', identify)
    monkeypatch.setattr(pick_scene_rgb_first, 'validate_selected_femto_stability', lambda *a: {'scene_stable': True})
    monkeypatch.setattr(workflow, 'validate_rgb_poses', lambda *a, **kw: {'production_pose_accepted': True})
    monkeypatch.setattr(workflow, '_publish_poses', lambda root, *a, **kw: root/'pose_data.pkl')
    if pose_backend == 'foundationpose':
        with pytest.raises(FusedPoseRejected, match='test quality rejection'):
            workflow.run_rgb_first_workflow(args, preflight=lambda a: (NS(), NS(model_directory=tmp_path)))
        assert attempts == ['fast']
        assert not (Path(args.output)/'pose_data.pkl').exists()
        return
    workflow.run_rgb_first_workflow(args, preflight=lambda a: (NS(), NS(model_directory=tmp_path)))
    status = json.loads((Path(args.output)/'pick_scene_workflow.json').read_text())
    assert attempts == ['fast', 'colmap']
    assert status['dense_fallback_used'] and status['accepted_dense_backend'] == 'colmap'
    assert status['production_pose_accepted']
