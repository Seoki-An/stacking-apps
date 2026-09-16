"""Timestamp checks and portable, single-view wrist RGB-D recordings."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from stacking_common.pointcloud import validate_rigid_transform
from stacking_tabletop.execution.motion import pose_error
from .hand_eye import HandEyeCalibration


# Initial acquisition guards, not a demonstrated 1 mm measurement budget.
LIMITS = dict(max_frame_age_ns=500_000_000, max_future_ns=20_000_000,
              max_pair_skew_ns=100_000_000, max_clock_step_ns=20_000_000,
              max_joint_gap_ns=50_000_000, settle_ns=500_000_000,
              max_joint_excursion_rad=.0005, max_tool_translation_m=.0002,
              max_tool_rotation_rad=.0005)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TimestampGuard:
    def __init__(self, not_before_ns):
        self.not_before_ns = not_before_ns
        self.previous = {}
        self.clock_offset = None

    def check(self, pair):
        offset = pair['receipt_ns'] - pair['receipt_monotonic_ns']
        if self.clock_offset is not None and abs(offset - self.clock_offset) > LIMITS['max_clock_step_ns']:
            raise ValueError('host system clock stepped relative to monotonic clock')
        self.clock_offset = offset
        for name in ('color', 'depth'):
            frame = pair[name]
            stamp, number = frame['timestamp_ns'], frame['frame_number']
            if frame['timestamp_domain'] != 'timestamp_domain.global_time':
                raise ValueError('unmapped camera clock; SDK global_time required')
            if stamp <= self.not_before_ns or number <= 0:
                raise ValueError('frame predates the capture gate or has invalid frame number')
            age = pair['receipt_ns'] - stamp
            if not -LIMITS['max_future_ns'] <= age <= LIMITS['max_frame_age_ns']:
                raise ValueError('stale/future source timestamp')
            if name in self.previous:
                old_stamp, old_number = self.previous[name]
                if stamp <= old_stamp or number <= old_number:
                    raise ValueError('duplicate frame or camera clock/counter reset')
            self.previous[name] = stamp, number
        if abs(pair['color']['timestamp_ns'] - pair['depth']['timestamp_ns']) > LIMITS['max_pair_skew_ns']:
            raise ValueError('color/depth source timestamps are too far apart')


def validate_images(pair, report):
    for name, dtype, channels in (('color', np.uint8, (3,)), ('depth', np.uint16, ())):
        geometry = report['streams'][name]
        image = pair[name]['image']
        if image.dtype != dtype or image.shape != (geometry['height'], geometry['width']) + channels:
            raise ValueError('native ' + name + ' image shape/encoding mismatch')


def warmup(camera):
    # Startup discard only. Exposure convergence must still be measured on hardware.
    for _ in range(30):
        camera.read()


def read_after_gate(camera, not_before_ns, *, clock=time):
    """Drain exposures already in flight at the gate, with a bounded wait."""
    deadline = clock.monotonic() + 5.
    while clock.monotonic() < deadline:
        pair = camera.read()
        if all(pair[name]['timestamp_ns'] > not_before_ns for name in ('color', 'depth')):
            return pair
    raise ValueError('no new exposures after source-time capture gate')


def preflight(camera, *, count=10):
    if count < 1:
        raise ValueError('preflight count must be positive')
    warmup(camera)
    guard = TimestampGuard(time.time_ns())
    samples = []
    for index in range(count):
        pair = read_after_gate(camera, guard.not_before_ns) if index == 0 else camera.read()
        guard.check(pair)
        validate_images(pair, camera.report)
        samples.append({key: ({k: v for k, v in value.items() if k != 'image'}
                             if key in ('color', 'depth') else value)
                        for key, value in pair.items()})
    return dict(camera.report, status='live_camera_preflight_passed', samples=samples,
                limits=LIMITS, robot_timing_verified=False)


def stationary_evidence(packet, baseline_joints, baseline_pose):
    joints = np.asarray(packet['joint_positions'], dtype=float)
    stamps = np.asarray(packet['joint_stamps_ns'], dtype=np.int64)
    if (joints.shape != (len(stamps), 6) or len(stamps) < 2 or not np.isfinite(joints).all()
            or np.any(np.diff(stamps) <= 0)
            or np.any(np.diff(stamps) > LIMITS['max_joint_gap_ns'])):
        raise ValueError('missing, invalid or gapped joint evidence')
    if (stamps[0] > packet['interval_start_ns'] or stamps[-1] < packet['interval_end_ns']
            or packet['interval_start_ns'] - stamps[0] > LIMITS['max_joint_gap_ns']
            or stamps[-1] - packet['interval_end_ns'] > LIMITS['max_joint_gap_ns']):
        raise ValueError('joint evidence does not bracket the observation interval')
    if np.max(np.abs(joints - baseline_joints)) > LIMITS['max_joint_excursion_rad']:
        raise ValueError('robot joint motion exceeds stationary capture guard')
    for name in ('color', 'depth'):
        pose = validate_rigid_transform(packet[name + '_base_T_tool'])
        translation, rotation = pose_error(pose, baseline_pose)
        if translation > LIMITS['max_tool_translation_m'] or rotation > LIMITS['max_tool_rotation_rad']:
            raise ValueError('tool motion exceeds stationary capture guard')


def capture_view(directory, calibration, camera, robot, *, count=10, clock=None):
    """Capture one operator-positioned view; a failure never becomes complete."""
    clock = time if clock is None else clock
    if count < 1:
        raise ValueError('capture count must be positive')
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, kind='hand_eye_native_view', status='recording',
        frame_count_requested=count, frames=[], files={}, limits=LIMITS,
        created_ns=clock.time_ns(), output_frame='base_link',
        data_origin=camera.report.get('data_origin', 'hardware'),
        pose_time_semantics='SDK global frame timestamp; exposure midpoint mapping unverified',
        accuracy_1mm_verified=False)
    source = Path(__file__).parent
    manifest['recorder_source_sha256'] = {name: digest(source / name) for name in
        ('hand_eye_capture.py', 'hand_eye.py', 'realsense.py', '../ros/hand_eye.py')}
    def save():
        atomic_json(directory / 'manifest.json', manifest)
    save()
    try:
        calibration.snapshot(directory / 'calibration')
        atomic_json(directory / 'device.json', camera.report)
        for path in sorted((directory / 'calibration').iterdir()):
            manifest['files'][str(path.relative_to(directory))] = digest(path)
        manifest['files']['device.json'] = digest(directory / 'device.json')
        (directory / 'frames').mkdir()
        warmup(camera)
        guard = TimestampGuard(clock.time_ns())
        baseline_joints = baseline_pose = None
        previous_stamp = None
        for index in range(count + 1):
            pair = read_after_gate(camera, guard.not_before_ns, clock=clock) if index <= 1 else camera.read()
            guard.check(pair)
            validate_images(pair, camera.report)
            stamps = {name: pair[name]['timestamp_ns'] for name in ('color', 'depth')}
            start = min(stamps.values()) - LIMITS['settle_ns'] if previous_stamp is None else previous_stamp
            packet = robot.measure(start, max(stamps.values()), stamps)
            if (packet['pose_stamps_ns'] != stamps or packet['interval_start_ns'] != start
                    or packet['interval_end_ns'] != max(stamps.values())):
                raise ValueError('robot evidence refers to different source timestamps')
            if baseline_joints is None:
                baseline_joints = np.asarray(packet['joint_positions'])[0].copy()
                baseline_pose = validate_rigid_transform(packet['color_base_T_tool'])
            stationary_evidence(packet, baseline_joints, baseline_pose)
            previous_stamp = min(stamps.values())
            if index == 0:
                # Preserve the full initial settling evidence and require later
                # source times, so a buffered pre-settle exposure cannot be saved.
                manifest['settling'] = packet
                guard.not_before_ns = max(stamps.values())
                save()
                continue
            record = {key: ({k: v for k, v in value.items() if k != 'image'}
                            if key in ('color', 'depth') else value) for key, value in pair.items()}
            record['robot'] = packet
            record['base_link_T_color'] = calibration.base_link_T_color(packet['color_base_T_tool']).tolist()
            record['base_link_T_depth'] = (calibration.base_link_T_color(packet['depth_base_T_tool'])
                                           @ np.asarray(camera.report['color_T_depth'])).tolist()
            stem = f'frames/{index - 1:04d}'
            image_path = directory / (stem + '.npz')
            with image_path.with_suffix('.npz.tmp').open('wb') as stream:
                np.savez_compressed(stream, color=pair['color']['image'], depth=pair['depth']['image'])
            image_path.with_suffix('.npz.tmp').replace(image_path)
            atomic_json(directory / (stem + '.json'), record)
            manifest['frames'].append(stem)
            for suffix in ('.npz', '.json'):
                manifest['files'][stem + suffix] = digest(directory / (stem + suffix))
            save()
        manifest['status'] = 'complete'
        save()
    except BaseException as error:
        manifest.update(status='incomplete', error=f'{type(error).__name__}: {error}')
        save()
        raise
    return manifest


def inspect_capture(directory):
    """Verify and read a portable raw capture without SDK, ROS or source files."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest['schema_version'] != 1 or manifest['kind'] != 'hand_eye_native_view'
            or manifest['status'] != 'complete'
            or len(manifest['frames']) != manifest['frame_count_requested']):
        raise ValueError('unsupported or incomplete native wrist capture')
    required = {'device.json', 'calibration/profile.yml', 'calibration/hand_eye.npz',
                'calibration/color_intrinsics.npz', 'calibration/installation.yml'}
    for stem in manifest['frames']:
        required.update((stem + '.npz', stem + '.json'))
    if not required.issubset(manifest['files']):
        raise ValueError('manifest is missing required file hashes')
    for relative, expected in manifest['files'].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or digest(path) != expected:
            raise ValueError('capture path/hash mismatch: ' + relative)
    calibration = HandEyeCalibration.load(directory / 'calibration/profile.yml')
    device = json.loads((directory / 'device.json').read_text())
    color = device['streams']['color']
    calibration.validate_color_stream(serial=device['device']['serial_number'],
        width=color['width'], height=color['height'], factory_matrix=color['K'],
        factory_distortion=color['distortion'])
    if not np.isfinite(device['depth_metres_per_unit']) or device['depth_metres_per_unit'] <= 0:
        raise ValueError('invalid recorded depth units')
    color_T_depth = validate_rigid_transform(device['color_T_depth'])
    settling = manifest['settling']
    baseline_joints = np.asarray(settling['joint_positions'])[0]
    baseline_pose = validate_rigid_transform(settling['color_base_T_tool'])
    stationary_evidence(settling, baseline_joints, baseline_pose)
    guard = TimestampGuard(max(settling['pose_stamps_ns'].values()))
    records = []
    for stem in manifest['frames']:
        record = json.loads((directory / (stem + '.json')).read_text())
        guard.check(record)
        stationary_evidence(record['robot'], baseline_joints, baseline_pose)
        if record['robot']['pose_stamps_ns'] != {name: record[name]['timestamp_ns'] for name in ('color', 'depth')}:
            raise ValueError('recorded robot and camera timestamps disagree')
        with np.load(directory / (stem + '.npz'), allow_pickle=False) as images:
            validate_images({name: dict(image=images[name]) for name in ('color', 'depth')}, device)
        for name in ('color', 'depth'):
            expected = calibration.base_link_T_color(record['robot'][name + '_base_T_tool'])
            if name == 'depth':
                expected = expected @ color_T_depth
            if not np.allclose(validate_rigid_transform(record['base_link_T_' + name]), expected,
                               rtol=0, atol=1e-10):
                raise ValueError('stored camera transform disagrees with snapshot/robot pose')
        records.append(record)
    return dict(status='verified_raw_capture', frame_count=len(records),
                serial=device['device']['serial_number'], accuracy_1mm_verified=False)
