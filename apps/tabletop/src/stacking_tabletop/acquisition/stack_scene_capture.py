"""Collect stacked-scene wrist RGB evidence inside the execution worker."""
from contextlib import contextmanager
from itertools import product
from pathlib import Path

import numpy as np

from stacking_common.execution import ExecutionFailure
from stacking_tabletop.execution.runtime import HardwareError
from stacking_tabletop.execution.sequence import _matrix
from .calibration import BASE_LINK_T_UR_BASE
from .hand_eye import HandEyeCalibration
from .hand_eye_capture import atomic_json, capture_view, digest
from .hand_eye_replay import assemble_session
from .scene_capture import SCAN_LIMITS, capture_lock
from .scene_views import build_views, camera_pose, render_preview
from .scan_paths import joint_leg
from .scanning import _settle


def stack_region(saved, plan, release_pose):
    """Use recorded scene extents for capture coverage, never as observed geometry."""
    import open3d as o3d

    table = next(box for box in saved.planning.obstacles if box.obstacle_id == 'place_table')
    objects = {item.object_id: item.pose for item in plan.problem.place_scene_objects}
    objects[plan.problem.action.object_id] = release_pose
    points = []
    for stone, pose in objects.items():
        mesh = o3d.io.read_triangle_mesh(str(saved.assets[f'object/{stone}']))
        transform = _matrix(pose)
        points.append(np.asarray(mesh.vertices) @ transform[:3, :3].T + transform[:3, 3])
    points = np.concatenate(points)
    low, high = points.min(axis=0)-.06, points.max(axis=0)+.06
    low[2] = table.maximum[2]
    high[2] = max(high[2], low[2]+.08)
    if np.any(high <= low):
        raise ValueError('stack capture region has invalid bounds')
    corners = np.array(list(product(*zip(low, high))))
    base = corners @ BASE_LINK_T_UR_BASE[:3, :3].T + BASE_LINK_T_UR_BASE[:3, 3]
    minimum, maximum = base.min(axis=0), base.max(axis=0)
    return dict(schema_version=1, kind='stack_scene_capture_region', frame='base_link',
                minimum=minimum.tolist(), maximum=maximum.tolist(),
                stone_boxes_xy=[[minimum[:2].tolist(), maximum[:2].tolist()]],
                support_plane=[0., 0., float(table.maximum[2])],
                source='recorded placement extents plus 60 mm coverage margin',
                observed_geometry=False)


def _stack_camera_pose(target, corners, reference_x, direction, calibration):
    """Fit every ROI corner into a look-at view, including at low elevations."""
    z = -np.asarray(direction)
    x = reference_x - z*(reference_x @ z)
    if np.linalg.norm(x) < .1:
        x = np.cross([0., 1., 0.], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    k = calibration.color_matrix
    # Keep a margin on both sides even when the principal point is off-center.
    available = .84*np.array([min(k[0, 2], calibration.width-1-k[0, 2]),
                              min(k[1, 2], calibration.height-1-k[1, 2])])
    if np.any(available <= 0):
        raise ValueError('camera principal point must lie inside the image')
    distance = max(.4, float(np.max(np.maximum(
        np.abs(corners @ x)*k[0, 0]/available[0],
        np.abs(corners @ y)*k[1, 1]/available[1]) - corners @ z)))
    if distance > .8:
        raise ValueError('region needs more than 0.8 m standoff')
    pose = np.eye(4)
    pose[:3, 3] = target + distance*np.asarray(direction)
    pose[:3, :3] = np.column_stack((x, y, z))
    return pose, distance


def _stack_side_view(kinematics, calibration, seed, target, corners, direction,
                     label, tilt, azimuth, check):
    """Follow a bounded look-at arc from the center; keep the full ROI in frame."""
    reference = camera_pose(kinematics, calibration, seed)
    start = -reference[:3, 2]
    angle = np.arccos(np.clip(start @ direction, -1., 1.))
    # Conservative angular sampling at the maximum allowed standoff (0.8 m).
    steps = max(1, int(np.ceil(.8*angle/.004)))
    q = np.asarray(seed).copy()
    path = [q.copy()]
    for fraction in np.linspace(0, 1, steps+1)[1:]:
        check()
        outward = (1-fraction)*start + fraction*direction
        outward /= np.linalg.norm(outward)
        pose, distance = _stack_camera_pose(target, corners, reference[:3, 0], outward, calibration)
        tool = BASE_LINK_T_UR_BASE @ pose @ np.linalg.inv(calibration.tool_T_color)
        next_q = kinematics.solve(tool, q, check=check)
        if np.max(np.abs(next_q-q)) > np.deg2rad(10):
            raise ValueError('IK changed joint branch or exceeded the local step limit')
        q = next_q
        path.append(q.copy())
    path = np.asarray(path)
    if np.any(path < kinematics.lower) or np.any(path > kinematics.upper):
        raise ValueError('side path exceeds robot joint limits')
    reached = camera_pose(kinematics, calibration, q)
    if not np.allclose(reached, pose, atol=2e-4, rtol=0):
        raise ValueError('side camera target FK does not match IK')
    return dict(label=label, role='side', tilt_from_vertical_deg=tilt,
                azimuth_from_robot_deg=azimuth, distance_m=distance,
                view_angle_deg=float(np.rad2deg(angle)), path=path.tolist(),
                base_link_T_color=reached.tolist())


def build_stack_route(kinematics, calibration, anchor, region, *, check=lambda: None):
    """Use three overview views and two reachable low-side stereo pairs (max 7)."""
    low, high = np.array(region['minimum']), np.array(region['maximum'])
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(high <= low):
        raise ValueError('stack capture region has invalid bounds')
    target = (low+high)/2
    reference = camera_pose(kinematics, calibration, anchor)
    inward = np.r_[-target[:2], 0.]
    if np.linalg.norm(inward) < 1e-9:
        inward = np.array([1., 0., 0.])
    inward /= np.linalg.norm(inward)
    tangent = np.cross([0., 0., 1.], inward)
    corners = np.array(list(product(*zip(low, high)))) - target
    failures = []
    # A usable central pose must also have its two nearby stereo views.
    for tilt_deg in (0., 25., 40.):
        check()
        angle = np.deg2rad(tilt_deg)
        outward = np.cos(angle)*np.array([0., 0., 1.]) + np.sin(angle)*inward
        try:
            pose, distance = _stack_camera_pose(target, corners, reference[:3, 0], outward, calibration)
            tool = BASE_LINK_T_UR_BASE @ pose @ np.linalg.inv(calibration.tool_T_color)
            seed = kinematics.solve(tool, anchor, check=check)
            partial = build_views(kinematics, calibration, seed, distance=distance,
                                  pattern='coverage13', check=check, labels=('center', 'left', 'right'))
            entry = joint_leg(anchor, seed)
            if np.any(entry < kinematics.lower) or np.any(entry > kinematics.upper):
                raise ValueError('stack capture entry exceeds joint limits')
            break
        except ValueError as error:
            failures.append(f'{tilt_deg:g} degrees: {error}')
    else:
        raise ValueError('stack camera center and stereo neighbors are unreachable: ' + '; '.join(failures))
    views, omitted, attempts, pairs = list(partial['views']), [], [], []
    for side, sign in (('left', -1), ('right', 1)):
        # Prefer low views; try less oblique geometry if either mate is unreachable.
        # The two tilt angles give a ~4 cm baseline at 0.4 m without wide-angle matching.
        for azimuth, tilt in ((45., 60.), (60., 60.), (45., 50.), (30., 50.)):
            azimuth *= sign
            a = np.deg2rad(azimuth)
            pair = []
            try:
                for index, t in enumerate((tilt, tilt+6.)):
                    angle = np.deg2rad(t)
                    direction = (np.cos(angle)*np.array([0., 0., 1.]) +
                                 np.sin(angle)*(np.cos(a)*inward+np.sin(a)*tangent))
                    pair.append(_stack_side_view(kinematics, calibration, seed, target, corners,
                        direction, f'low_{side}_{index}', t, azimuth, check))
                # Visit both mates before returning to center. The short joint
                # transition is previewed too; it is not a collision-checked arc.
                first_q, second_q = [np.asarray(v['path'][-1]) for v in pair]
                if np.max(np.abs(second_q-first_q)) > np.deg2rad(20):
                    raise ValueError('side mates do not share a nearby joint branch')
                transition = joint_leg(first_q, second_q)
                pair[0]['return_path'] = []
                pair[1]['return_path'] = transition[::-1].tolist() + pair[0]['path'][::-1][1:]
                pair[1]['path'] = transition.tolist()
                views.extend(pair)
                pairs.append([v['label'] for v in pair])
                break
            except ValueError as error:
                attempts.append(dict(side=side, azimuth_deg=azimuth, tilt_deg=tilt, reason=str(error)))
        else:
            omitted.extend(dict(label=f'low_{side}_{i}', reason='no complete reachable stereo pair')
                           for i in range(2))
    plan = dict(partial, kind='tabletop_stack_scene_plan', viewpoint_pattern='stack_paired7',
                views=views, omitted_views=omitted, side_view_attempts=attempts, side_stereo_pairs=pairs)
    plan.update(return_policy='return_after_overview_or_side_pair', central_tilt_deg=tilt_deg,
                central_view_attempts=failures, maximum_views=7,
                entry_path=entry.tolist(), return_path=entry[::-1].tolist(),
                stack_region=region, entry_collision_checked=False)
    return plan


@contextmanager
def live_wrist(executor, calibration):
    from stacking_tabletop.ros.hand_eye import HandEyeRobot
    from .realsense import RealSenseCamera

    with HandEyeRobot(calibration) as robot, RealSenseCamera(calibration, depth_size=(848, 480), fps=15) as camera:
        class CheckedCamera:
            report = camera.report
            def read(self):
                executor._health()
                pair = camera.read()
                executor._health()
                return pair
        checked = CheckedCamera()
        def capture(directory):
            paused = executor.runtime.paused_seconds
            capture_view(directory, calibration, checked, robot, count=3)
            if executor.runtime.paused_seconds != paused:
                raise RuntimeError('hold interrupted wrist capture; partial scene retained for review')
        yield capture


def check_stack_capture_runtime(repository):
    repository = Path(repository).resolve()
    vendor = repository/'.data/vendor/Fast-FoundationStereo'
    required = (repository/'.data/venvs/fast-stereo/bin/python',
                vendor/'weights/23-36-37/model_best_bp2_serialize.pth')
    if any(not p.is_file() for p in required):
        raise ValueError('stack scene capture requires the installed Fast-FoundationStereo runtime and weights')
    return HandEyeCalibration.load(repository/'apps/tabletop/config/hand_eye.yml')


def collect_stack_scene(executor, repository, *, prepared=lambda value: None,
                        wrist_factory=live_wrist, planner=build_stack_route, reconstruct=None):
    """Run only after release/retreat. Failures hold and prevent the next pickup."""
    from .hand_eye_fast_stereo import reconstruct_with_runtime

    repository = Path(repository).resolve()
    vendor = repository/'.data/vendor/Fast-FoundationStereo'
    checkpoint = vendor/'weights/23-36-37/model_best_bp2_serialize.pth'
    if reconstruct is None:
        check_stack_capture_runtime(repository)
        reconstruct = reconstruct_with_runtime
    if executor._attached or executor._release_pose is None:
        raise ValueError('stack scene capture requires a released object and empty tool')
    hand = executor.gripper.feedback.gripper_state()
    if hand.holding or hand.hold_goal is not None:
        raise ValueError('stack scene capture cannot run with a held object')
    executor.saved.verify_assets()
    calibration = HandEyeCalibration.load(repository/'apps/tabletop/config/hand_eye.yml')
    anchor = executor.arm.feedback.arm_state()[0].copy()
    region = stack_region(executor.saved, executor._plan, executor._release_pose)
    plan = planner(executor.arm.kinematics, calibration, anchor, region, check=executor._health)
    output = (executor.saved.path.parent / (executor.saved.path.stem+'_scene_captures') /
              f'step_{executor._command.step:03d}_{executor.active_command_id}')
    output.mkdir(parents=True, exist_ok=False)
    status = dict(schema_version=1, kind='stacked_scene_wrist_capture', status='running',
                  command_id=executor.active_command_id, step=executor._command.step,
                  context_sha256=executor.saved.context.sha256, output_frame='base_link',
                  release_pose=executor._release_pose.tolist(), accuracy_1mm_verified=False,
                  robot_returned=False, completed_views=[])
    def stage(name, **values):
        executor._phase = 'stack_scene/'+name
        status.update(stage=name, **values)
        atomic_json(output/'capture.json', status)
        executor._emit('running', stack_scene_stage=name, stack_scene_directory=str(output))
    def move(path):
        executor.arm.execute_path(np.asarray(path), attached=False, limits=SCAN_LIMITS,
            progress=lambda _i, _n, q: setattr(executor, '_reference', q))
        _settle(executor, np.asarray(path[-1]))
    try:
        calibration.snapshot(output/'calibration')
        atomic_json(output/'plan.json', plan)
        atomic_json(output/'region.json', region)
        (output/'robot.urdf').write_bytes(executor.arm.kinematics.urdf_path.read_bytes())
        render_preview(plan, executor.arm.kinematics, calibration, output/'preview.html')
        preview = [*plan['entry_path']]
        for view in plan['views']:
            preview.extend(view['path'])
            preview.extend(view.get('return_path', view['path'][::-1]))
        preview.extend(plan['return_path'])
        # Keep the camera open for the whole reviewed scan and serialize access.
        with capture_lock(), wrist_factory(executor, calibration) as capture:
            executor._gate_wait('stack_scan_review',
                'The stone is released and the arm has retreated. Review the stacked-scene wrist '
                'capture entry, viewpoints and return in the cyan preview; paths are not collision checked. '
                f'{len(plan["views"])} views are reachable; {len(plan.get("omitted_views", []))} were omitted. Continue starts the scan.',
                prepared=lambda: prepared(dict(kind='stacked scene RGB', positions=preview,
                    capture_count=len(plan['views']), output=str(output))))
            executor._profile(False)
            stage('entry'); move(plan['entry_path'])
            views = []
            for index, view in enumerate(plan['views']):
                stage('move_'+view['label']); move(view['path'])
                stage('capture_'+view['label'])
                directory = output/'wrist'/f'view_{index:03d}_{view["label"]}'
                capture(directory)
                views.append(directory)
                status['completed_views'].append(str(directory.relative_to(output)))
                return_path = view.get('return_path', view['path'][::-1])
                if return_path:
                    stage('return_'+view['label']); move(return_path)
            stage('return_to_execution'); move(plan['return_path'])
            status['robot_returned'] = True
        executor._reference = anchor
        stage('assembly')
        assemble_session(output/'session', views)
        stage('reconstruction')
        reconstruct(output/'session', None, None, output/'rgb_dense', repository=vendor,
                    checkpoint=checkpoint, region_file=output/'region.json', check=executor._health)
        pcd = output/'rgb_dense/refined/rgb_dense_voxel_1mm.pcd'
        if not pcd.is_file():
            raise ValueError('stack reconstruction produced no PCD')
        stage('complete', status='complete', pcd=str(pcd), pcd_sha256=digest(pcd),
              viewer=str(output/'rgb_dense/rgb_only/overview.html'))
        executor._emit('held', stack_scene_completed=str(output), stack_scene_pcd=str(pcd),
                       stack_scene_pcd_sha256=digest(pcd), stack_scene_viewer=status['viewer'])
        executor._gate_wait('stack_scene_result',
            'Stacked-scene PCD is saved and displayed. Inspect the observation, then Continue. '
            'The saved placement plan has not been changed by this capture.')
        return status
    except BaseException as error:
        cancelled = isinstance(error, KeyboardInterrupt) or (isinstance(error, HardwareError)
                     and error.failure is ExecutionFailure.CANCELLED)
        status.update(status='cancelled' if cancelled else 'failed', error=f'{type(error).__name__}: {error}')
        atomic_json(output/'capture.json', status)
        if (not cancelled and not isinstance(error, HardwareError)
                and status.get('stage', '').startswith(('capture_', 'assembly', 'reconstruction'))):
            raise HardwareError(ExecutionFailure.PERCEPTION_FAILED, str(error)) from error
        raise
