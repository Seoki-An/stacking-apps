"""Post-release stacked-scene collection is serialized with plan execution."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
import json
import shutil

import numpy as np
import pytest

from stacking_common.artifacts import StackRunStore
from stacking_tabletop.acquisition import stack_scene_capture as capture
from stacking_tabletop.acquisition import stack_scene_processing as processing
from stacking_tabletop.acquisition.hand_eye_capture import atomic_json, digest
from stacking_tabletop.acquisition.hand_eye_replay import inspect_session
from stacking_tabletop.acquisition.hand_eye_synthetic import generate_demo
from stacking_tabletop.acquisition.storage import write_xyz_pcd
from stacking_tabletop.execution.sequence import execute_saved_run
from test_tabletop_saved_execution import saved, executor_for, ROOT
from test_tabletop_hand_eye_calibration import profile_path


def target_region(minimum=(.44, -.09, .115), maximum=(.66, .09, .23)):
    return dict(
        minimum=list(minimum), maximum=list(maximum),
        target_structure_origin=[.55, 0., .115],
        target_structure_axes=capture.BASE_LINK_T_UR_BASE[:3, :3].tolist(),
        view_target=[.55, 0., .19],
    )


@pytest.mark.parametrize('fail', [False, True])
def test_capture_runs_from_pre_place_before_home_and_next_pickup(saved, fail):
    transport, executor = executor_for(saved)
    calls, finished = [], []
    class Processing:
        def __init__(self, step):
            self.step = step
        def finish(self, worker):
            np.testing.assert_allclose(
                worker.arm.feedback.arm_state()[0],
                saved.planning.robot.home_positions, atol=.01,
            )
            finished.append(self.step)
        def abort(self):
            pass
    def collect(worker):
        assert worker._attached is False
        assert worker._release_pose is not None
        assert not worker.gripper.feedback.gripper_state().holding
        np.testing.assert_allclose(
            worker.arm.feedback.arm_state()[0],
            worker._command.segments[4].positions[saved.grasp_steps - 1],
            atol=.01,
        )
        events = StackRunStore.load_execution_events(saved.path)
        assert any(e.diagnostics.get('release_completed') for e in events)
        assert not any(e.status.value == 'succeeded' and e.diagnostics['step'] == worker._command.step for e in events)
        calls.append(worker._command.step)
        if fail:
            raise RuntimeError('camera failed after placement')
        return Processing(worker._command.step)
    executor.stack_scene_handler = collect
    results = execute_saved_run(saved, executor)
    if fail:
        assert calls == [1] and len(results) == 1
        assert results[0].status.value == 'failed'
        assert results[0].diagnostics['phase'] == 'stack_scene'
        assert StackRunStore.load_execution_checkpoint(saved.path).requires_reconciliation
    else:
        assert calls == finished == [1, 2]
        assert all(r.status.value == 'succeeded' for r in results)
        np.testing.assert_allclose(
            executor.arm.feedback.arm_state()[0],
            saved.planning.robot.home_positions,
            atol=.01,
        )


def test_stack_scene_route_uses_faster_entry_and_view_motion_limits():
    from stacking_tabletop.acquisition.scene_capture import SCAN_LIMITS

    entry = capture.STACK_SCENE_ENTRY_LIMITS
    views = capture.STACK_SCENE_VIEW_LIMITS
    assert entry.rate_hz == views.rate_hz == SCAN_LIMITS.rate_hz
    assert entry.max_joint_speed == SCAN_LIMITS.max_joint_speed
    assert entry.max_tcp_speed == SCAN_LIMITS.max_tcp_speed
    assert entry.max_angular_speed == SCAN_LIMITS.max_angular_speed
    assert views.max_joint_speed == pytest.approx(1.5 * entry.max_joint_speed)
    assert views.max_tcp_speed == pytest.approx(1.5 * entry.max_tcp_speed)
    assert views.max_angular_speed == pytest.approx(1.5 * entry.max_angular_speed)


def test_live_wrist_reopens_stalled_pipeline_before_motion(monkeypatch):
    from stacking_tabletop.acquisition import realsense
    from stacking_tabletop.ros import hand_eye as ros_hand_eye

    cameras, events, waits = [], [], []
    class Camera:
        report = {}
        def __init__(self, *args, **kwargs):
            self.index = len(cameras)
            self.closed = False
            cameras.append(self)
        def read(self):
            if self.index == 0:
                raise RuntimeError("Frame didn't arrive within 1000")
            return object()
        def close(self):
            self.closed = True
    class Robot:
        def __init__(self, calibration):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(realsense, 'RealSenseCamera', Camera)
    monkeypatch.setattr(ros_hand_eye, 'HandEyeRobot', Robot)
    executor = NS(
        _health=lambda: None,
        _emit=lambda status, **values: events.append((status, values)),
        runtime=NS(
            paused_seconds=0.,
            wait=lambda seconds, check: waits.append(seconds) or check(),
        ),
    )
    with capture.live_wrist(executor, NS()) as record:
        assert callable(record)
    assert len(cameras) == 2
    assert all(camera.closed for camera in cameras)
    assert waits == [1.]
    assert events[0][1]['stack_scene_camera_reopen'] == 1


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp('stack-wrist')/'demo'
    generate_demo(root, visualize=False)
    return root


@pytest.mark.parametrize('failure', [None, 'capture', 'reconstruction'])
def test_session_collection_returns_before_reconstruction_and_retains_failures(tmp_path, monkeypatch, synthetic, failure):
    sources = sorted((synthetic/'inputs').glob('view_*'))
    assert len(sources) >= 3
    calibration = capture.HandEyeCalibration.load(sources[0]/'calibration/profile.yml')
    monkeypatch.setattr(capture.HandEyeCalibration, 'load', lambda p: calibration)
    monkeypatch.setattr(capture, 'render_preview', lambda *a: None)
    region = dict(frame='base_link', minimum=[-.2,-.2,0], maximum=[.2,.2,.3],
                  stone_boxes_xy=[[[-.2,-.2],[.2,.2]]], support_plane=[0,0,0])
    monkeypatch.setattr(capture, 'stack_region', lambda *a: region)
    anchor = np.zeros(6); seed = np.full(6, .1)
    views = [dict(label=f'v{i}', path=[seed.tolist(), (seed+i*.01).tolist()]) for i in range(3)]
    views[1]['return_path'] = []
    views[2]['path'] = [views[1]['path'][-1], (seed+.02).tolist()]
    views[2]['return_path'] = [views[2]['path'][-1], views[1]['path'][-1], seed.tolist()]
    plan = dict(views=views, entry_path=[anchor.tolist(),seed.tolist()],
                return_path=[seed.tolist(),anchor.tolist()])
    motions, phases, gates = [], [], []
    def move(path, **kw):
        expected = (capture.STACK_SCENE_ENTRY_LIMITS
                    if executor._phase.endswith(('entry', 'return_to_execution'))
                    else capture.STACK_SCENE_VIEW_LIMITS)
        assert kw['limits'] is expected
        motions.append(np.array(path))
        kw['progress'](0, 1, path[-1])
    urdf = tmp_path/'robot.urdf'; urdf.write_text('<robot/>')
    arm = NS(kinematics=NS(urdf_path=urdf), execute_path=move,
             feedback=NS(arm_state=lambda: (anchor, None)))
    executor = NS(arm=arm, _attached=False, _release_pose=np.array([0,0,.1,0,0,0,1]),
                  _plan=NS(), _command=NS(step=1), active_command_id='fixture',
                  saved=NS(path=tmp_path/'run.stack', context=NS(sha256='context'),verify_assets=lambda: None),
                  gripper=NS(feedback=NS(gripper_state=lambda:NS(holding=False,hold_goal=None))),
                  _health=lambda: None, _profile=lambda value: None)
    def gate(name, message, **kw):
        gates.append(name)
        if 'prepared' in kw:kw['prepared']()
    executor._gate_wait=gate
    executor._emit=lambda status,**kw: phases.append(kw)
    monkeypatch.setattr(capture, '_settle', lambda *a:None)
    @contextmanager
    def wrist(*a):
        count=0
        def record(destination):
            nonlocal count
            if failure=='capture':raise RuntimeError('capture failure')
            shutil.copytree(sources[count],destination);count+=1
        yield record
    def reconstruct(session, baseline, refinement, output, **kwargs):
        assert gates == ['stack_scan_review']
        assert baseline is refinement is None
        assert inspect_session(session)['references'] == []
        np.testing.assert_array_equal(motions[-1][-1],anchor)
        # Every leg starts where the preceding leg ended; no return between mates.
        assert len(motions) == 7
        for first, second in zip(motions, motions[1:]):
            np.testing.assert_allclose(first[-1], second[0])
        if failure=='reconstruction':raise RuntimeError('reconstruction failure')
        (output/'refined').mkdir(parents=True)
        write_xyz_pcd(output/'refined/rgb_dense_voxel_1mm.pcd', np.array([[0.,0.,.1]]))
    def run():
        processing = capture.collect_stack_scene(
            executor, ROOT, wrist_factory=wrist,
            planner=lambda *a, **kw: plan, reconstruct=reconstruct,
        )
        return processing.finish(executor)
    if failure:
        with pytest.raises(RuntimeError,match=failure+' failure'):run()
    else:
        result=run();assert result['status']=='complete'
        assert gates==['stack_scan_review','stack_scene_result']
    report=json.loads((tmp_path/'run_scene_captures/step_001_fixture/capture.json').read_text())
    assert report['robot_returned'] == (failure!='capture')
    if failure=='capture':
        assert len(motions)==2  # Entry and first view; no automatic return on failure.
        assert not (tmp_path/'run_scene_captures/step_001_fixture/session').exists()
    if failure:assert report['status']=='failed'


def test_scene_observation_uses_correct_frame_and_hash(saved,tmp_path):
    from stacking_tabletop.gui.operations.execution_scene import ExecutionScene
    scene=ExecutionScene(saved,ROOT)
    path=tmp_path/'scene.pcd';points=np.array([[.4,.2,.3],[.5,.3,.4]])
    write_xyz_pcd(path,points)
    event=NS(status=NS(value='held'),diagnostics=dict(stack_scene_pcd=str(path),stack_scene_pcd_sha256=digest(path)))
    scene.event(event)
    np.testing.assert_allclose(scene.frame().observed_scene_points,points*[-1,-1,1])
    assert not scene.frame().observed_scene_points.flags.writeable
    path.write_text('changed')
    with pytest.raises(ValueError,match='PCD changed'):scene.event(event)


def test_stack_region_uses_recorded_target_structure_frame(saved):
    target = {'origin': [.01, -.02, .03], 'height': .16}
    context = NS(profiles={'inventory_profile': {'target': target}})
    source = NS(planning=saved.planning, assets=saved.assets, context=context)
    plan = saved.plans[0]
    region = capture.stack_region(source, plan, plan.problem.place_pose)
    stack_T_target = np.eye(4)
    stack_T_target[:3, 3] = target['origin']
    expected = (capture.BASE_LINK_T_UR_BASE
                @ capture._matrix(saved.planning.world_T_stack.parent_T_child)
                @ stack_T_target)
    np.testing.assert_allclose(region['target_structure_origin'], expected[:3, 3])
    np.testing.assert_allclose(region['target_structure_axes'], expected[:3, :3])
    np.testing.assert_allclose(
        region['view_target'], (expected @ [0., 0., target['height']/2., 1.])[:3],
    )
    assert region['schema_version'] == 2


def test_isolated_stack_processing_reconstructs_then_identifies(tmp_path, monkeypatch):
    capture_directory = tmp_path/'capture'
    (capture_directory/'session').mkdir(parents=True)
    atomic_json(capture_directory/'capture.json', {
        'kind': 'stacked_scene_wrist_capture', 'status': 'running',
        'step': 2, 'context_sha256': 'context', 'output_frame': 'base_link',
    })
    saved_run = NS(path=tmp_path/'run.stack', context=NS(sha256='context'))
    monkeypatch.setattr(
        'stacking_tabletop.execution.saved_plan.load_execution_run',
        lambda run, repository: saved_run,
    )
    order = []
    def reconstruct(session, baseline, refinement, output, **kwargs):
        order.append('reconstruct')
        (output/'refined').mkdir(parents=True)
        (output/'rgb_only').mkdir()
        write_xyz_pcd(
            output/'refined/rgb_dense_voxel_1mm.pcd',
            np.array([[0., 0., .1]]),
        )
        (output/'rgb_only/overview.html').write_text('<html/>')
    def identify(saved, repository, directory, step):
        assert json.loads((directory/'capture.json').read_text())['pcd']
        order.append('identify')
        return {'status': 'identified', 'accepted_snapshot': True}
    report = processing.process_stack_scene(
        saved_run.path, ROOT, capture_directory, 2,
        reconstruct=reconstruct, identify=identify,
    )
    status = json.loads((capture_directory/'capture.json').read_text())
    assert order == ['reconstruct', 'identify']
    assert report['status'] == status['status'] == status['stage'] == 'complete'
    assert status['scene_identification']['status'] == 'identified'
    assert Path(status['scene_identification']['path']).is_file()


def test_stack_route_aims_at_stack_and_returns_to_execution_anchor(profile_path):
    from test_tabletop_hand_eye_calibration import HandEyeCalibration
    from stacking_tabletop.config import load_planning_profile
    from stacking_tabletop.execution.kinematics import UR5eKinematics
    from stacking_tabletop.acquisition.scene_views import camera_pose
    profile=load_planning_profile(ROOT/'apps/tabletop/config/planning.yml')
    kin=UR5eKinematics(ROOT/profile.robot.urdf)
    calibration=HandEyeCalibration.load(profile_path)
    anchor=profile.robot.home_positions
    region=target_region()
    route=capture.build_stack_route(kin,calibration,anchor,region)
    assert len(route['views'])>=3
    np.testing.assert_array_equal(route['entry_path'][0],anchor)
    np.testing.assert_array_equal(route['return_path'][-1],anchor)
    assert 3 <= len(route['views']) <= 7
    assert len(route['views']) == 3 + 2*len(route['side_stereo_pairs'])
    assert 2*len(route['side_stereo_pairs']) + len(route['omitted_views']) == 4
    assert route['viewpoint_pattern'] == 'stack_paired7'
    assert route['viewpoint_frame'] == 'target_structure'
    np.testing.assert_allclose(route['target'], region['view_target'], atol=1e-12)
    assert not route['collision_checked']
    for view in route['views']:
        pose=camera_pose(kin,calibration,view['path'][-1])
        target=np.linalg.solve(pose,np.r_[route['target'],1])
        np.testing.assert_allclose(target[:2],0,atol=1e-5)
        assert target[2]>0


def test_stack_route_uses_angled_center_if_top_down_is_unreachable(profile_path, monkeypatch):
    from stacking_tabletop.config import load_planning_profile
    from stacking_tabletop.execution.kinematics import UR5eKinematics
    profile = load_planning_profile(ROOT/'apps/tabletop/config/planning.yml')
    kin = UR5eKinematics(ROOT/profile.robot.urdf)
    calibration = capture.HandEyeCalibration.load(profile_path)
    solve = kin.solve
    calls = []
    def reject_top_down(target, seed, **kwargs):
        calls.append(target.copy())
        if len(calls) == 1:
            raise ValueError('top-down center outside reach')
        return solve(target, seed, **kwargs)
    monkeypatch.setattr(kin, 'solve', reject_top_down)
    region = target_region()
    route = capture.build_stack_route(kin, calibration, profile.robot.home_positions, region)
    assert route['central_tilt_deg'] in (25., 40.)
    assert 'top-down center outside reach' in route['central_view_attempts'][0]
    np.testing.assert_allclose(route['target'], region['view_target'], atol=1e-5)
    assert len(route['views']) >= 3


def test_synthetic_stack_export_is_wrist_only_with_ideal_calibration(tmp_path):
    from stacking_tabletop.acquisition.hand_eye_synthetic import calibration_fixture
    from stacking_tabletop.acquisition.simulation_capture import SimulatedCapture
    calibration = calibration_fixture(tmp_path/'source_calibration', color_size=(32,24))
    pose = np.eye(4); pose[:3,:3] = np.diag([1.,-1.,-1.]); pose[2,3] = .6
    views = [dict(label=f'view_{i}',path=[[0.]*6],base_link_T_color=pose.tolist()) for i in range(3)]
    exporter = SimulatedCapture(tmp_path/'captures',calibration,dict(views=views))
    for view in views:
        exporter.add(view,np.full((24,32,3),120,np.uint8),np.full((24,32),.45))
    root = exporter.finish()
    manifest = inspect_session(root/'session')
    assert manifest['data_origin'] == 'synthetic'
    assert manifest['references'] == [] and len(manifest['views']) == 3
    assert not (root/'femto_before').exists() and not (root/'femto_after').exists()
    replay_calibration = capture.HandEyeCalibration.load(root/'session/views/000/calibration/profile.yml')
    np.testing.assert_allclose(replay_calibration.color_distortion,0)


def test_stack_side_pairs_have_stereo_baseline_and_keep_region_in_frame(profile_path):
    from itertools import product
    from stacking_tabletop.config import load_planning_profile
    from stacking_tabletop.execution.kinematics import UR5eKinematics
    from stacking_tabletop.acquisition.scene_views import camera_pose
    from stacking_common.multiview.learned_stereo import select_pairs
    profile = load_planning_profile(ROOT/'apps/tabletop/config/planning.yml')
    kin = UR5eKinematics(ROOT/profile.robot.urdf)
    calibration = capture.HandEyeCalibration.load(profile_path)
    region = target_region((.44, -.15, .115), (.67, .13, .26))
    route = capture.build_stack_route(kin, calibration, profile.robot.home_positions, region)
    assert route['side_stereo_pairs']
    views = route['views']
    poses = np.array([v['base_link_T_color'] for v in views])
    selected = select_pairs(poses)
    labels = [v['label'] for v in views]
    corners = np.array(list(product(*zip(region['minimum'], region['maximum']))))
    for pair in route['side_stereo_pairs']:
        indices = tuple(labels.index(label) for label in pair)
        assert indices in selected
        assert np.linalg.norm(poses[indices[0], :3, 3]-poses[indices[1], :3, 3]) >= .02
        for index in indices:
            view = views[index]
            assert view['tilt_from_vertical_deg'] >= 50
            path = np.array(view['path'])
            assert np.max(np.abs(np.diff(path, axis=0))) <= np.deg2rad(10)
            assert np.all(path >= kin.lower) and np.all(path <= kin.upper)
            if index == indices[0]:
                np.testing.assert_allclose(path[0], route['seed_joints'])
                assert view['return_path'] == []
            else:
                np.testing.assert_allclose(path[0], views[indices[0]]['path'][-1])
                np.testing.assert_allclose(view['return_path'][0], path[-1])
                np.testing.assert_allclose(view['return_path'][-1], route['seed_joints'])
            for q in path[::max(1, len(path)//10)].tolist() + [path[-1].tolist()]:
                pose = camera_pose(kin, calibration, q)
                local = (corners-pose[:3,3]) @ pose[:3,:3]
                assert np.all(local[:,2] > 0)
                uv = local @ calibration.color_matrix.T
                uv = uv[:,:2]/uv[:,2,None]
                assert np.all(uv >= 0)
                assert np.all(uv < [calibration.width, calibration.height])


def test_incomplete_side_pair_is_omitted_as_a_pair(profile_path, monkeypatch):
    from stacking_tabletop.config import load_planning_profile
    from stacking_tabletop.execution.kinematics import UR5eKinematics
    profile = load_planning_profile(ROOT/'apps/tabletop/config/planning.yml')
    kin = UR5eKinematics(ROOT/profile.robot.urdf)
    calibration = capture.HandEyeCalibration.load(profile_path)
    side_view = capture._stack_side_view
    def fail_second_mate(*args):
        if args[6].endswith('_1'):
            raise ValueError('second mate unreachable')
        return side_view(*args)
    monkeypatch.setattr(capture, '_stack_side_view', fail_second_mate)
    route = capture.build_stack_route(kin, calibration, profile.robot.home_positions,
        target_region())
    assert len(route['views']) == 3
    assert not route['side_stereo_pairs']
    assert len(route['omitted_views']) == 4
    assert any('second mate unreachable' in a['reason'] for a in route['side_view_attempts'])


def test_stack_route_propagates_cancellation(profile_path):
    from stacking_tabletop.config import load_planning_profile
    from stacking_tabletop.execution.kinematics import UR5eKinematics
    profile = load_planning_profile(ROOT/'apps/tabletop/config/planning.yml')
    kin = UR5eKinematics(ROOT/profile.robot.urdf)
    calibration = capture.HandEyeCalibration.load(profile_path)
    def cancel():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        capture.build_stack_route(kin, calibration, profile.robot.home_positions,
            target_region(), check=cancel)
