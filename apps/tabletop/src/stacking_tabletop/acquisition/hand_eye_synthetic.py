"""Explicitly synthetic wrist/FEMTO captures for the offline experiment path."""

import io
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from stacking_common.pointcloud.depth import optical_rays, backproject_depth
from stacking_common.pointcloud.synthetic_depth import raycast_ellipsoids
from .calibration import BASE_LINK_T_UR_BASE
from .hand_eye import HandEyeCalibration
from .hand_eye_capture import atomic_json, capture_view, digest
from .hand_eye_replay import assemble_session, replay_session, read_json, file_hashes
from .storage import CaptureSession


SCENE = dict(table_z=.12, table_bounds=[[-.24, -.20], [.24, .20]],
    centres=[[-.095, -.065, .151], [.065, -.055, .158], [-.06, .07, .15], [.09, .07, .162]],
    radii=[[.057, .039, .031], [.052, .036, .038], [.045, .05, .03], [.061, .044, .042]])


class SyntheticClock:
    def __init__(self):
        self.now = 1_800_000_000_000_000_000
    def time_ns(self):
        return self.now
    def monotonic(self):
        return (self.now-1_800_000_000_000_000_000)/1e9


def look_at(position, target=(0., 0., .15)):
    forward = np.asarray(target)-position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
    pose[:3, 3] = position
    return pose


def geometry(width, height):
    return dict(width=width, height=height, fps=15, format='format.z16',
        K=[[width*.9, 0., (width-1)/2], [0., width*.9, (height-1)/2], [0., 0., 1.]],
        distortion_model='distortion.none', distortion=[0.]*5)


def rays_for(info):
    return optical_rays(info['width'], info['height'], info['K'])


def calibration_fixture(directory, *, color_size=(320, 240), color_matrix=None, tool_T_color=None):
    directory.mkdir(parents=True)
    color = geometry(*color_size)
    matrix = np.asarray(color['K'] if color_matrix is None else color_matrix)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('xyz', [8., -12., 17.], degrees=True).as_matrix()
    transform[:3, 3] = [-.03, -.075, .083]
    if tool_T_color is not None:
        transform = np.asarray(tool_T_color).copy()
    np.savez(directory / 'hand_eye.npz', T_gripper_to_camera=transform, T_EE_to_cam1=transform,
        T_gripper_to_camera_tf=transform, cameraMatrix=matrix, distCoeffs=np.zeros(5), method='Tsai',
        robot_pose_source='tf', robot_pose_parent_frame='base', robot_pose_child_frame='tool0',
        camera_serial='SYNTHETIC-D435', synthetic_fixture=True)
    np.savez(directory / 'intrinsics.npz', K=matrix, dist=np.zeros(5), K_factory=matrix,
        dist_factory=np.zeros(5), camera='D435_COLOR', method='d435_handeye_log_charuco_intrinsic',
        image_size=list(color_size), distortion_model='k3', camera_serial='SYNTHETIC-D435', synthetic_fixture=True)
    params = dict(rs_serial='SYNTHETIC-D435', rs_width=color_size[0], rs_height=color_size[1],
                  tf_base_frame='base', tf_tool_frame='tool0', robot_pose_source='tf', synthetic_fixture=True)
    (directory / 'installation.yml').write_text(yaml.safe_dump(dict(handeye_calib=dict(ros__parameters=params))))
    profile = dict(schema_version=1, serial='SYNTHETIC-D435', color_frame='synthetic_color_optical_frame',
                   base_frame='base', tool_frame='tool0', robot_pose_source='tf')
    for name, file in [('hand_eye', 'hand_eye.npz'), ('intrinsics', 'intrinsics.npz'), ('installation', 'installation.yml')]:
        profile[name] = dict(path=file, sha256=digest(directory / file))
    (directory / 'profile.yml').write_text(yaml.safe_dump(profile))
    return HandEyeCalibration.load(directory / 'profile.yml')


def shade_color(info, pose):
    rays = rays_for(info)
    depth, labels = raycast_ellipsoids(rays, pose, **SCENE)
    world = (rays*depth[..., None]) @ pose[:3, :3].T + pose[:3, 3]
    normals = np.zeros_like(world)
    normals[..., 2] = 1.
    palette = np.array([[233, 239, 247], [177, 187, 199], [112, 122, 137],
                        [171, 141, 109], [128, 147, 137], [150, 132, 148]], dtype=float)
    for index, (centre, radii) in enumerate(zip(SCENE['centres'], SCENE['radii'])):
        mask = labels == index+2
        normals[mask] = (world[mask]-centre)/np.square(radii)
        normals[mask] /= np.linalg.norm(normals[mask], axis=-1)[:, None]
    light = np.array([-.3, -.5, 1.])
    light /= np.linalg.norm(light)
    shading = .45 + .55*np.maximum(normals @ light, 0)
    rgb = palette[labels]*shading[..., None]
    grid = ((np.floor(world[..., 0]/.025)+np.floor(world[..., 1]/.025)) % 2 == 0) & (labels == 1)
    rgb[grid] *= .9
    rgb[labels == 0] = palette[0]
    return np.clip(rgb, 0, 255).astype(np.uint8)


class SyntheticCamera:
    def __init__(self, calibration, pose, clock, seed):
        self.clock, self.rng, self.number = clock, np.random.default_rng(seed), 0
        color, depth = geometry(320, 240), geometry(160, 120)
        color['format'] = 'format.rgb8'
        extrinsic = np.eye(4)
        extrinsic[0, 3] = .012
        self.report = dict(schema_version=1, data_origin='synthetic',
            device=dict(serial_number='SYNTHETIC-D435', name='analytic fixture'),
            streams=dict(color=color, depth=depth), depth_metres_per_unit=.00025,
            color_T_depth=extrinsic.tolist(), accuracy_1mm_verified=False,
            timestamp_mapping='fabricated deterministic sequence; no physical clocks',
            synthetic_noise=dict(seed=seed, axial_std_m=.00035, dropout_probability=.025))
        self.truth, _ = raycast_ellipsoids(rays_for(depth), pose, **SCENE)
        self.rgb = shade_color(color, pose @ np.linalg.inv(extrinsic))
        self.base_T_tool = BASE_LINK_T_UR_BASE @ pose @ np.linalg.inv(extrinsic) @ np.linalg.inv(calibration.tool_T_color)

    def read(self):
        self.clock.now += 67_000_000
        self.number += 1
        noisy = self.truth + self.rng.normal(0, .00035, self.truth.shape)
        raw = np.clip(np.rint(noisy/.00025), 0, 65535).astype(np.uint16)
        raw[(self.truth == 0) | (self.rng.random(raw.shape) < .025)] = 0
        pair = dict(receipt_ns=self.clock.now+10_000_000,
                    receipt_monotonic_ns=int(self.clock.monotonic()*1e9)+10_000_000)
        for name, image in [('color', self.rgb), ('depth', raw)]:
            stamp = self.clock.now + (1_000_000 if name == 'depth' else 0)
            pair[name] = dict(image=image, frame_number=self.number, timestamp_ns=stamp,
                timestamp_ms=stamp/1e6, timestamp_domain='timestamp_domain.global_time',
                metadata=dict(synthetic=True))
        return pair

    def measure(self, start, end, stamps):
        times = list(range(start, end, 20_000_000)) + [end]
        return dict(joint_stamps_ns=times, joint_positions=np.zeros((len(times), 6)).tolist(),
            interval_start_ns=start, interval_end_ns=end, pose_stamps_ns=stamps,
            color_base_T_tool=self.base_T_tool.tolist(), depth_base_T_tool=self.base_T_tool.tolist(),
            pose_source='synthetic_exact_pose; joints are fixture values, not robot FK')


def synthetic_femto(directory, clock, *, pose=None, info=None, depth=None,
                    crop_min=(-.25, -.21, 0.), crop_max=(.25, .21, .3), voxel_size=.002):
    """Exercise the existing seven-frame fixed-field writer with synthetic inputs."""
    pose = look_at(np.array([.025, -.025, .82])) if pose is None else pose
    info = geometry(160, 120) if info is None else info
    rays = rays_for(info)
    if depth is None:
        depth, _ = raycast_ellipsoids(rays, pose, **SCENE)
    raw = np.rint(depth/.00025).astype(np.uint16)
    points, _ = backproject_depth(raw, rays, .00025)
    artifacts = {}
    for name in ('camera_pose.npz', 'camera_depth.npz'):
        buffer = io.BytesIO()
        np.savez(buffer, synthetic_fixture=True, base_link_T_depth=pose)
        artifacts[name] = buffer.getvalue()
    import hashlib
    calibration = NS(profile=NS(serial='SYNTHETIC-FEMTO', base_frame='base_link',
        crop_min=crop_min, crop_max=crop_max, voxel_size=voxel_size),
        depth=NS(depth_mode=f'SYNTHETIC_{info["width"]}x{info["height"]}', depth_frame='synthetic_depth_optical_frame', scale=1., offset=0.),
        pose=NS(correction='synthetic_exact'), artifacts=artifacts,
        hashes={name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()}, base_T_depth=pose)
    camera_info = dict(width=info['width'], height=info['height'], k=np.asarray(info['K']).ravel().tolist(), d=[0.]*5,
                       distortion_model='plumb_bob', frame_id='synthetic_depth_optical_frame')
    burst = []
    for _ in range(7):
        clock.now += 67_000_000
        frame = NS(frame_id=camera_info['frame_id'], camera_info=camera_info, encoding='16UC1',
            depth_stamp_ns=clock.now, cloud_stamp_ns=clock.now, depth=raw, points=points.points, depth_unit=.00025)
        burst.append(NS(frame=frame, robot=None, base_T_cloud=pose))
    with CaptureSession(directory, calibration) as capture:
        capture.add(burst)
    for name in ('field_scan.json', 'camera_frame.json'):
        atomic_json(directory / name, dict(read_json(directory / name), data_origin='synthetic'))


def input_preview(session, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    views = read_json(session / 'session.json')['views']
    figure, axes = plt.subplots(len(views), 2, figsize=(10, len(views)*2.8), squeeze=False)
    for index, view in enumerate(views):
        root = session / view['path']
        stem = read_json(root / 'manifest.json')['frames'][0]
        with np.load(root / (stem + '.npz')) as data:
            axes[index, 0].imshow(data['color'])
            depth = data['depth'].astype(float)*.00025
        depth[depth == 0] = np.nan
        axes[index, 1].imshow(depth, cmap='viridis', vmin=.25, vmax=.65)
        for column, label in enumerate(('RGB', 'Native depth · 0.25–0.65 m; white = invalid')):
            axes[index, column].set_title(f'View {index+1} · {label}', fontsize=10)
            axes[index, column].axis('off')
    figure.suptitle('SYNTHETIC · Four stone proxies on a table', fontsize=16)
    figure.tight_layout()
    figure.savefig(output / 'inputs.png', dpi=160)
    plt.close(figure)


def generate_demo(directory, *, visualize=True):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    calibration = calibration_fixture(directory / 'fixture_calibration')
    clock = SyntheticClock()
    inputs = directory / 'inputs'
    synthetic_femto(inputs / 'femto_before', clock)
    views = []
    positions = [(.19, -.16, .56), (-.18, -.14, .57), (-.16, .17, .55), (.18, .15, .58)]
    for index, position in enumerate(positions):
        camera = SyntheticCamera(calibration, look_at(np.asarray(position)), clock, seed=20260915+index)
        view = inputs / f'view_{index:03d}'
        capture_view(view, calibration, camera, camera, count=3, clock=clock)
        views.append(view)
    synthetic_femto(inputs / 'femto_after', clock)
    atomic_json(directory / 'reference.json', dict(schema_version=1, data_origin='synthetic',
        kind='analytic_scene', frame='base_link', scene=SCENE,
        note='Exact generated geometry; not physical reference measurements or real stone models'))
    assemble_session(directory / 'session', views, femto_before=inputs / 'femto_before',
                     femto_after=inputs / 'femto_after', reference=directory / 'reference.json')
    report = replay_session(directory / 'session', directory / 'replay', visualize=visualize)
    if visualize:
        input_preview(directory / 'session', directory / 'replay')
        report['output_hashes'] = {name: value for name, value in file_hashes(directory / 'replay').items()
                                   if name != 'replay.json'}
        atomic_json(directory / 'replay/replay.json', report)
    return report
