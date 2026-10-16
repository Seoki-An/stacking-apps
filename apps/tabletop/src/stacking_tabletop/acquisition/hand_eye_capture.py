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


class RobotEvidenceUnavailable(ValueError):
    """A specific exposure lacks robot evidence; never use a nearby pose."""

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__('missing timestamp-bracketed joints or exact-time TF for capture: '
                         + json.dumps(diagnostics, sort_keys=True))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RepeatedFrame(ValueError):
    """An unchanged exposure appeared again in a composite frameset."""


class TimestampGuard:
    def __init__(self, not_before_ns, *, streams=('color', 'depth')):
        self.streams = tuple(streams)
        if self.streams not in (('color',), ('color', 'depth')):
            raise ValueError('unsupported wrist streams')
        self.not_before_ns = not_before_ns
        self.previous = {}
        self.seen = {}
        self.repeated_pairs = 0
        self.clock_offset = None

    def check(self, pair):
        offset = pair['receipt_ns'] - pair['receipt_monotonic_ns']
        if self.clock_offset is not None and abs(offset - self.clock_offset) > LIMITS['max_clock_step_ns']:
            raise ValueError('host system clock stepped relative to monotonic clock')
        self.clock_offset = offset
        current = {}
        repeated = []
        for name in self.streams:
            frame = pair[name]
            stamp, number = frame['timestamp_ns'], frame['frame_number']
            if frame['timestamp_domain'] != 'timestamp_domain.global_time':
                raise ValueError('unmapped camera clock; SDK global_time required')
            if stamp <= 0 or number <= 0:
                raise ValueError('invalid camera timestamp or invalid frame number')
            if stamp - pair['receipt_ns'] > LIMITS['max_future_ns']:
                raise ValueError('stale/future source timestamp')
            if name in self.seen:
                old_stamp, old_number = self.seen[name]
                if (stamp < old_stamp or number < old_number
                        or ((stamp == old_stamp) != (number == old_number))):
                    raise ValueError(f'{name} camera clock/counter reset or inconsistent frame identity: '
                                     f'previous=({old_stamp}, {old_number}), current=({stamp}, {number})')
            current[name] = stamp, number
            if current[name] == self.previous.get(name):
                repeated.append(name)
        self.seen = current
        if repeated:
            self.repeated_pairs += 1
            raise RepeatedFrame(f'duplicate frame in {", ".join(repeated)}; '
                                f'previous={self.previous}, current={current}')
        # Clock/domain/identity checks above apply even to discarded framesets.
        # Age, gate and skew determine whether two NEW exposures can be saved;
        # a repeated exposure becoming older is not a new clock failure.
        for name in self.streams:
            stamp = current[name][0]
            if stamp <= self.not_before_ns:
                raise ValueError('frame predates the capture gate')
            if pair['receipt_ns'] - stamp > LIMITS['max_frame_age_ns']:
                raise ValueError('stale/future source timestamp')
        if 'depth' in self.streams and abs(pair['color']['timestamp_ns'] - pair['depth']['timestamp_ns']) > LIMITS['max_pair_skew_ns']:
            raise ValueError('color/depth source timestamps are too far apart')
        # Commit both accepted streams together; a repeated depth exposure must
        # not consume a fresh color exposure which a later frameset can reuse.
        self.previous = current


def validate_images(pair, report):
    for name, dtype, channels in (('color', np.uint8, (3,)), ('depth', np.uint16, ())):
        if name not in report['streams']:
            continue
        geometry = report['streams'][name]
        image = pair[name]['image']
        if image.dtype != dtype or image.shape != (geometry['height'], geometry['width']) + channels:
            raise ValueError('native ' + name + ' image shape/encoding mismatch')


def _read_camera(camera, *, attempts=5):
    """Tolerate a short native-frame gap without hiding persistent failures."""
    last_error = None
    for _ in range(attempts):
        try:
            return camera.read()
        except RuntimeError as error:
            if "Frame didn't arrive within" not in str(error):
                raise
            last_error = error
    raise RuntimeError(
        f'D435 delivered no frame after {attempts} consecutive 1000 ms waits; '
        'check camera power, USB 3 connection, and competing camera processes'
    ) from last_error


def warmup(camera, *, count=30):
    # Startup discard only. Exposure convergence must still be measured on hardware.
    if count < 1:
        raise ValueError('warmup count must be positive')
    for _ in range(count):
        _read_camera(camera)


def read_after_gate(camera, not_before_ns, *, clock=time, streams=('color', 'depth')):
    """Drain exposures already in flight at the gate, with a bounded wait."""
    deadline = clock.monotonic() + 5.
    while clock.monotonic() < deadline:
        pair = _read_camera(camera)
        if all(pair[name]['timestamp_ns'] > not_before_ns for name in streams):
            return pair
    raise ValueError('no new exposures after source-time capture gate')


def read_unique_pair(camera, guard, *, drain_gate=False, clock=time):
    """Discard only identical exposures, with a bounded number of reads."""
    last_repeat = None
    for _ in range(10):
        pair = (read_after_gate(camera, guard.not_before_ns, clock=clock, streams=guard.streams)
                if drain_gate else _read_camera(camera))
        try:
            guard.check(pair)
        except RepeatedFrame as error:
            last_repeat = error
            continue
        except ValueError as error:
            identities = {name: {key: pair[name][key] for key in
                          ('timestamp_ns', 'frame_number', 'timestamp_domain')}
                          for name in guard.streams}
            raise ValueError(f'{error}; received_frames={identities}; '
                             f'previous_accepted={guard.previous}; receipt_ns={pair["receipt_ns"]}') from error
        return pair
    raise ValueError(f'no unique RGB/depth pair after 10 framesets: {last_repeat}')


def preflight(camera, *, count=10):
    if count < 1:
        raise ValueError('preflight count must be positive')
    warmup(camera)
    guard = TimestampGuard(time.time_ns(), streams=camera.report['streams'])
    samples = []
    for index in range(count):
        pair = read_unique_pair(camera, guard, drain_gate=index == 0, clock=time)
        validate_images(pair, camera.report)
        samples.append({key: ({k: v for k, v in value.items() if k != 'image'}
                             if key in ('color', 'depth') else value)
                        for key, value in pair.items()})
    return dict(camera.report, status='live_camera_preflight_passed', samples=samples,
                limits=LIMITS, repeated_pairs_discarded=guard.repeated_pairs, robot_timing_verified=False)


def stationary_evidence(packet, baseline_joints, baseline_pose, *, streams=('color', 'depth')):
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
    for name in streams:
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
    streams = tuple(camera.report['streams'])
    if streams not in (('color',), ('color', 'depth')):
        raise ValueError('unsupported wrist streams')
    manifest = dict(schema_version=2 if streams == ('color',) else 1, streams=list(streams),
        kind='hand_eye_native_view', status='recording',
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
        guard = TimestampGuard(clock.time_ns(), streams=streams)
        baseline_joints = baseline_pose = None
        previous_stamp = None
        for index in range(count + 1):
            for attempt in range(3):
                pair = read_unique_pair(camera, guard, drain_gate=index <= 1, clock=clock)
                manifest['repeated_pairs_discarded'] = guard.repeated_pairs
                validate_images(pair, camera.report)
                stamps = {name: pair[name]['timestamp_ns'] for name in streams}
                start = min(stamps.values()) - LIMITS['settle_ns'] if previous_stamp is None else previous_stamp
                try:
                    packet = robot.measure(start, max(stamps.values()), stamps)
                    break
                except RobotEvidenceUnavailable as error:
                    # Only restart initial settling, before accepting any pose or
                    # frame. A gap in an established burst remains a failure.
                    if index != 0 or attempt == 2:
                        raise
                    manifest.setdefault('settling_retries', []).append(error.diagnostics)
                    save()
                    print(f'wrist capture: retrying initial synchronization ({attempt + 1}/2): {error}', flush=True)
                    guard.not_before_ns = clock.time_ns() + LIMITS['settle_ns']
            if (packet['pose_stamps_ns'] != stamps or packet['interval_start_ns'] != start
                    or packet['interval_end_ns'] != max(stamps.values())):
                raise ValueError('robot evidence refers to different source timestamps')
            if baseline_joints is None:
                baseline_joints = np.asarray(packet['joint_positions'])[0].copy()
                baseline_pose = validate_rigid_transform(packet['color_base_T_tool'])
            stationary_evidence(packet, baseline_joints, baseline_pose, streams=streams)
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
            if 'depth' in streams:
                record['base_link_T_depth'] = (calibration.base_link_T_color(packet['depth_base_T_tool'])
                                               @ np.asarray(camera.report['color_T_depth'])).tolist()
            stem = f'frames/{index - 1:04d}'
            image_path = directory / (stem + '.npz')
            with image_path.with_suffix('.npz.tmp').open('wb') as stream:
                np.savez_compressed(stream, **{name: pair[name]['image'] for name in streams})
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
        if 'guard' in locals():
            manifest['repeated_pairs_discarded'] = guard.repeated_pairs
        if isinstance(error, RobotEvidenceUnavailable):
            manifest['robot_timing_failure'] = error.diagnostics
        save()
        raise
    return manifest


def inspect_capture(directory):
    """Verify and read a portable raw capture without SDK, ROS or source files."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    streams = tuple(manifest.get('streams', ('color', 'depth')))
    expected_streams = {1: ('color', 'depth'), 2: ('color',)}
    if (streams != expected_streams.get(manifest['schema_version']) or manifest['kind'] != 'hand_eye_native_view'
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
    if set(device['streams']) != set(streams):
        raise ValueError('recorded device streams differ from manifest')
    color = device['streams']['color']
    calibration.validate_color_stream(serial=device['device']['serial_number'],
        width=color['width'], height=color['height'], factory_matrix=color['K'],
        factory_distortion=color['distortion'])
    if 'depth' in streams:
        if not np.isfinite(device['depth_metres_per_unit']) or device['depth_metres_per_unit'] <= 0:
            raise ValueError('invalid recorded depth units')
        color_T_depth = validate_rigid_transform(device['color_T_depth'])
    settling = manifest['settling']
    baseline_joints = np.asarray(settling['joint_positions'])[0]
    baseline_pose = validate_rigid_transform(settling['color_base_T_tool'])
    stationary_evidence(settling, baseline_joints, baseline_pose, streams=streams)
    guard = TimestampGuard(max(settling['pose_stamps_ns'].values()), streams=streams)
    records = []
    for stem in manifest['frames']:
        record = json.loads((directory / (stem + '.json')).read_text())
        guard.check(record)
        stationary_evidence(record['robot'], baseline_joints, baseline_pose, streams=streams)
        if record['robot']['pose_stamps_ns'] != {name: record[name]['timestamp_ns'] for name in streams}:
            raise ValueError('recorded robot and camera timestamps disagree')
        with np.load(directory / (stem + '.npz'), allow_pickle=False) as images:
            validate_images({name: dict(image=images[name]) for name in streams}, device)
        for name in streams:
            expected = calibration.base_link_T_color(record['robot'][name + '_base_T_tool'])
            if name == 'depth':
                expected = expected @ color_T_depth
            if not np.allclose(validate_rigid_transform(record['base_link_T_' + name]), expected,
                               rtol=0, atol=1e-10):
                raise ValueError('stored camera transform disagrees with snapshot/robot pose')
        records.append(record)
    return dict(status='verified_raw_capture', frame_count=len(records),
                serial=device['device']['serial_number'], accuracy_1mm_verified=False)
