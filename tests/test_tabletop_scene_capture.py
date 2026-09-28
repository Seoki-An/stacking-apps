"""Automatic wrist geometry and lifecycle using offline kinematics/fake devices."""

from contextlib import contextmanager
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from stacking_tabletop.acquisition.hand_eye import HandEyeCalibration
from stacking_tabletop.acquisition.hand_eye_capture import atomic_json, digest
from stacking_tabletop.acquisition.scene_views import (
    build_views, camera_pose, kinematic_urdf, validate_views,
)
from stacking_tabletop.acquisition.scene_capture import execute_capture, capture_lock, settle_arm
from stacking_tabletop.acquisition.scene_capture_cli import load_reviewed_plan, ensure_ownership, main
from stacking_tabletop.execution.kinematics import UR5eKinematics
from stacking_tabletop.execution.runtime import AdapterRuntime
from test_tabletop_hand_eye_calibration import profile_path

ROOT = Path(__file__).resolve().parents[1]
SEED = np.deg2rad([90, -90, 90, -90, -90, 30])


@pytest.fixture
def route(profile_path, tmp_path):
    calibration = HandEyeCalibration.load(profile_path)
    urdf = tmp_path / 'robot.urdf'
    urdf.write_text(kinematic_urdf((ROOT / 'apps/tabletop/assets/ur5e/ur5e_ik.urdf').read_text()))
    kin = UR5eKinematics(urdf)
    plan = build_views(kin, calibration, SEED, distance=.4)
    return plan, kin, calibration


def test_views_translate_look_at_one_target_and_return_on_same_branch(route):
    plan, kin, calibration = route
    reference = camera_pose(kin, calibration, SEED)
    centers = []
    for view in plan['views']:
        path = np.asarray(view['path'])
        np.testing.assert_array_equal(path[0], SEED)
        assert np.all(path >= kin.lower) and np.all(path <= kin.upper)
        pose = camera_pose(kin, calibration, path[-1])
        target_in_camera = np.linalg.solve(pose, np.r_[plan['target'], 1])
        np.testing.assert_allclose(target_in_camera[:2], 0, atol=1e-5)
        assert target_in_camera[2] > 0
        centers.append(pose[:3, 3])
        # Reversing the whole saved leg returns exactly to the physical seed.
        np.testing.assert_array_equal(path[::-1][-1], SEED)
    assert [view['label'] for view in plan['views']] == [
        'center', 'left', 'right', 'up', 'down', 'oblique_left_up',
        'oblique_right_up', 'oblique_right_down', 'oblique_left_down']
    assert len({tuple(np.round(p, 5)) for p in centers}) == 9
    radial = np.linalg.norm(np.array(centers)[1:] - reference[:3, 3], axis=1)
    np.testing.assert_allclose(radial[:4], .04, atol=1e-5)
    np.testing.assert_allclose(radial[4:], .08, atol=1e-5)
    assert [view['role'] for view in plan['views']] == ['overview'] + ['near']*4 + ['oblique']*4
    assert max(view['view_angle_deg'] for view in plan['views']) == pytest.approx(np.rad2deg(np.arctan2(.08, .4)))
    validate_views(plan, kin, calibration)
    changed = copy.deepcopy(plan)
    changed['views'][1]['path'][-1][0] += .05
    with pytest.raises(ValueError, match='saved path differs'):
        validate_views(changed, kin, calibration)



def test_side_route_adds_tilted_views_with_kinematic_paths(route):
    _, kin, calibration = route
    plan = build_views(kin, calibration, SEED, distance=.4, pattern='coverage13')
    assert plan['schema_version'] == 3
    assert len(plan['views']) == 13
    assert [view['label'] for view in plan['views'][9:]] == [
        'side_left', 'side_right', 'side_up', 'side_down']
    reference = camera_pose(kin, calibration, SEED)
    target = np.asarray(plan['target'])
    for view in plan['views'][9:]:
        path = np.asarray(view['path'])
        assert len(path) > 20
        np.testing.assert_array_equal(path[0], SEED)
        assert np.all(path >= kin.lower) and np.all(path <= kin.upper)
        reached = camera_pose(kin, calibration, path[-1])
        np.testing.assert_allclose(np.linalg.norm(reached[:3, 3] - target), .4, atol=2e-4)
        np.testing.assert_allclose(view['view_angle_deg'], 30.)
        np.testing.assert_allclose(np.linalg.norm(view['camera_offset_xy_m']), .2)
        assert (reached[:3, 3] - reference[:3, 3]) @ reference[:3, 2] > 0
    validate_views(plan, kin, calibration)
    changed = copy.deepcopy(plan)
    changed['side_angle_deg'] = 35.
    with pytest.raises(ValueError, match='side angle'):
        validate_views(changed, kin, calibration)

def test_extended_route_preserves_old_views_and_adds_wider_stereo_pairs(route):
    _, kin, calibration = route
    seed = np.deg2rad([90, -131, 117, -114, -90, 0])
    old = build_views(kin, calibration, seed, distance=.3, pattern='coverage13')
    wide = build_views(kin, calibration, seed, distance=.3, pattern='coverage17')
    assert wide['schema_version'] == 4
    assert wide['views'][:13] == old['views']
    assert [v['label'] for v in wide['views'][13:]] == [
        'side_left_45', 'side_right_45', 'side_left_60', 'side_right_60']
    for view, degrees in zip(wide['views'][13:], [45, 45, 60, 60]):
        path = np.array(view['path'])
        np.testing.assert_array_equal(path[0], seed)
        assert np.max(np.abs(np.diff(path, axis=0))) <= np.deg2rad(10)
        assert view['view_angle_deg'] == degrees
        np.testing.assert_allclose(np.linalg.norm(view['camera_offset_xy_m']),
                                   .3 * np.sin(np.deg2rad(degrees)))
        reached = camera_pose(kin, calibration, path[-1])
        local = np.linalg.solve(reached, np.r_[wide['target'], 1.])
        np.testing.assert_allclose(local[:3], [0, 0, .3], atol=2e-4)
    validate_views(wide, kin, calibration)
    changed = copy.deepcopy(wide)
    changed['views'][-1]['view_angle_deg'] = 30.
    with pytest.raises(ValueError, match='saved path differs'):
        validate_views(changed, kin, calibration)


def test_invalid_geometry_and_ik_branches_fail_before_execution(route):
    _, kin, calibration = route
    for distance, offset in ((float('nan'), .04), (.4, -.1), (.4, .5)):
        with pytest.raises(ValueError):
            build_views(kin, calibration, SEED, distance=distance, offset=offset)
    for oblique in (.04, .21, float('nan')):
        with pytest.raises(ValueError, match='oblique'):
            build_views(kin, calibration, SEED, distance=.4, oblique_offset=oblique)
    original = kin.solve
    kin.solve = lambda pose, q: original(pose, q) + np.array([0, 0, 0, 0, 0, 2 * np.pi])
    with pytest.raises(ValueError, match='branch'):
        build_views(kin, calibration, SEED, distance=.4)


def test_legacy_five_view_plan_remains_reviewable(route):
    _, kin, calibration = route
    legacy = build_views(kin, calibration, SEED, distance=.4, pattern='legacy5')
    legacy['schema_version'] = 1
    legacy.pop('viewpoint_pattern')
    legacy.pop('oblique_offset_m')
    for view in legacy['views']:
        view.pop('role'); view.pop('camera_offset_xy_m'); view.pop('view_angle_deg')
    validate_views(legacy, kin, calibration)


def test_kinematic_snapshot_preserves_fk_without_external_geometry(route):
    _, stripped, _ = route
    original = UR5eKinematics(ROOT / 'apps/tabletop/assets/ur5e/ur5e_ik.urdf')
    for q in (SEED, SEED + .1):
        np.testing.assert_allclose(stripped.pose(q), original.pose(q), atol=1e-12)
    text = stripped.urdf_path.read_text()
    assert '<visual' not in text and '<collision' not in text
    with pytest.raises(ValueError, match='canonical six'):
        kinematic_urdf('<robot name="bad"><link name="base"/></robot>')


def test_approval_and_asset_mismatch_rejected_offline(route, tmp_path):
    plan, kin, calibration = route
    directory = tmp_path / 'saved'
    directory.mkdir()
    calibration.snapshot(directory / 'calibration')
    (directory / 'robot.urdf').write_bytes(kin.urdf_path.read_bytes())
    plan['robot_urdf_sha256'] = digest(directory / 'robot.urdf')
    plan['hand_eye_profile_sha256'] = digest(directory / 'calibration/profile.yml')
    atomic_json(directory / 'plan.json', plan)
    approval = digest(directory / 'plan.json')
    loaded, _, _ = load_reviewed_plan(directory, approval)
    assert loaded['approved_sha256'] == approval
    with pytest.raises(ValueError, match='approve'):
        load_reviewed_plan(directory, 'unreviewed')
    profile = directory / 'calibration/profile.yml'
    original_profile = profile.read_bytes()
    profile.write_bytes(original_profile + b'\n')
    with pytest.raises(ValueError, match='hand-eye profile changed'):
        load_reviewed_plan(directory, approval)
    profile.write_bytes(original_profile)
    with (directory / 'robot.urdf').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='kinematics changed'):
        load_reviewed_plan(directory, approval)


class Arm:
    def __init__(self, events):
        self.events = events
        self.q = SEED.copy()
        self.paths = []
        self.holds = 0
    def validate_start(self, q):
        np.testing.assert_allclose(q, self.q)
    def set_rigid_tracking(self):
        self.events.append('rigid')
    def hold_actual_pose(self):
        self.holds += 1
        self.events.append('hold')
        return True
    def execute_path(self, path, **kwargs):
        self.validate_start(path[0])
        self.paths.append(np.array(path))
        self.q = np.array(path[-1])
        self.events.append('move')


def run_fake(route, output, failure=None):
    plan = copy.deepcopy(route[0])
    plan['approved_sha256'] = 'reviewed'
    events = []
    arm = Arm(events)
    open_camera = [False]
    def femto(directory):
        assert not open_camera[0]
        np.testing.assert_allclose(arm.q, SEED)
        events.append(directory.name)
        if failure == directory.name:
            raise RuntimeError(failure)
    @contextmanager
    def wrist():
        open_camera[0] = True
        events.append('wrist_open')
        try:
            def capture(directory):
                events.append(directory.name)
                if failure == directory.name:
                    raise KeyboardInterrupt
            yield capture
        finally:
            open_camera[0] = False
            events.append('wrist_close')
    def assemble(directory, views, **kwargs):
        assert len(views) == len(plan['views']) and not open_camera[0]
        assert events[-1] == 'femto_after'
        events.append('assemble')
    try:
        execute_capture(plan, output, arm, femto, wrist, assemble,
                        settle=lambda arm, q: arm.validate_start(q), progress=lambda _: None)
    except BaseException as error:
        return arm, events, error
    return arm, events, None


def test_complete_sequence_brackets_wrist_and_keeps_camera_open(route, tmp_path):
    arm, events, error = run_fake(route, tmp_path / 'run')
    assert error is None
    assert events.index('femto_before') < events.index('wrist_open')
    assert events.index('wrist_close') < events.index('femto_after') < events.index('assemble')
    assert events.count('wrist_open') == 1
    assert len(arm.paths) == 18
    np.testing.assert_array_equal(arm.q, SEED)
    status = json.loads((tmp_path / 'run/capture.json').read_text())
    assert status['status'] == 'complete' and len(status['completed_views']) == 9
    assert status['scene_stability'] == 'not_evaluated'
    assert not status['accuracy_1mm_verified']


@pytest.mark.parametrize('failure,moves,status', [('femto_before', 0, 'failed'),
    ('view_001_left', 3, 'cancelled'), ('femto_after', 18, 'failed')])
def test_failure_holds_without_return_or_further_acquisition(route, tmp_path, failure, moves, status):
    arm, events, error = run_fake(route, tmp_path / 'run', failure)
    assert error is not None and len(arm.paths) == moves
    assert arm.holds == 2 and events[-1] == 'hold'
    assert 'assemble' not in events
    record = json.loads((tmp_path / 'run/capture.json').read_text())
    assert record['status'] == status and record['hold_requested']
    if failure == 'view_001_left':
        assert 'wrist_close' in events and 'femto_after' not in events
        assert not np.allclose(arm.q, SEED)  # no automatic abort return


def test_exclusive_lock_and_external_command_or_hold_rejection():
    with capture_lock():
        with pytest.raises(RuntimeError, match='another automated'):
            with capture_lock():
                pass
    transport = NS(node=NS(count_publishers=lambda _: 2))
    with pytest.raises(RuntimeError, match='reference publisher'):
        ensure_ownership(transport)
    transport.node.count_publishers = lambda _: 1
    transport.feedback = NS(gripper_state=lambda: NS(hold_goal=500))
    with pytest.raises(RuntimeError, match='empty tool'):
        ensure_ownership(transport)


def test_settling_checks_measured_joints_and_times_out():
    now = [0.]
    runtime = AdapterRuntime(clock=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0] + dt))
    arm = NS(runtime=runtime, _check_motion=lambda: None,
             feedback=NS(arm_state=lambda: (SEED.copy(), np.eye(4))))
    settle_arm(arm, SEED)
    assert now[0] >= .75
    with pytest.raises(RuntimeError, match='did not settle'):
        settle_arm(arm, SEED + .02, timeout=.1)


def test_offline_plan_requires_explicit_robot_model(tmp_path):
    assert main(['plan', '--distance', '.4', '--seed-view', str(tmp_path),
                 '--output', str(tmp_path / 'out')]) == 1


def test_simulated_pick_stones_use_pose_solver_and_seat_render_meshes_on_table():
    from stacking_tabletop.acquisition.scene_capture_simulation import synthesize_stones, matrix, transformed
    from stacking_tabletop.composition import TabletopApplication
    import open3d as o3d

    app = TabletopApplication.from_repository(ROOT)
    meshes, records = synthesize_stones(app, (9, 10, 11), np.array([.1, -.65, .2]))
    table = next(box for box in app.planning_profile.obstacles if box.obstacle_id == 'pick_table')
    for (vertices, faces), record in zip(meshes, records):
        np.testing.assert_allclose(vertices[:, 2].min(), table.maximum[2], atol=1e-12)
        original = np.asarray(record['posegen_world_pose'])
        adjusted = np.asarray(record['world_pose'])
        np.testing.assert_allclose(adjusted - original, [0, 0, record['mesh_seating_z_m'], 0, 0, 0, 0])
        source = o3d.io.read_triangle_mesh(record['mesh'])
        np.testing.assert_allclose(vertices, transformed(source.vertices, matrix(adjusted)))
        assert record['pose_solver_converged'] is True
        assert record['posegen_ground_height_m'] == table.maximum[2]
        assert record['solver_diagnostics']['posegen_force_converged'] is True
        assert np.isfinite(record['c_feq']) and np.isfinite(record['c_gap_m'])
        assert len(faces) > 0


def test_simulation_exports_gui_capture_contract(tmp_path):
    from stacking_tabletop.acquisition.hand_eye_synthetic import calibration_fixture, geometry
    from stacking_tabletop.acquisition.simulation_capture import SimulatedCapture
    from stacking_tabletop.acquisition.pick_scene_workflow import _verify_capture
    from stacking_tabletop.acquisition.hand_eye_replay import inspect_session
    from stacking_tabletop.acquisition.calibration import BASE_LINK_T_UR_BASE

    calibration = calibration_fixture(tmp_path/'calibration', color_size=(32, 24))
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1., -1., -1.])
    pose[:3, 3] = [0., 0., .7]
    views = [dict(label=f'view_{index}', path=[SEED.tolist()], base_link_T_color=pose.tolist())
             for index in range(3)]
    plan = dict(kind='tabletop_wrist_scene_plan', views=views)
    capture = SimulatedCapture(tmp_path/'captures', calibration, plan,
        np.full((24, 32), .6), pose, geometry(32, 24), ([-1., -1., 0.], [1., 1., 1.]))
    for view in views:
        capture.add(view, np.full((24, 32, 3), 120, dtype=np.uint8), np.full((24, 32), .6))
    capture.finish()
    assert _verify_capture(tmp_path/'captures') == tmp_path/'captures'
    manifest = inspect_session(tmp_path/'captures/session')
    assert manifest['data_origin'] == 'synthetic'
    assert len(manifest['views']) == 3
    frame = json.loads((tmp_path/'captures/session/views/000/frames/0000.json').read_text())
    np.testing.assert_allclose(frame['base_link_T_color'], pose, atol=1e-12)
    np.testing.assert_allclose(frame['robot']['joint_positions'][-1], SEED)
    assert (tmp_path/'captures/femto_before/field_scan.json').is_file()
    assert (tmp_path/'captures/femto_after/field_scan.json').is_file()
