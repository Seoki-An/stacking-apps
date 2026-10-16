"""Native wrist recording checks with deterministic clocks and no hardware."""

import copy
import json
import shutil
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stacking_tabletop.acquisition.hand_eye import HandEyeCalibration
from stacking_tabletop.acquisition import hand_eye_capture as capture
from stacking_tabletop.acquisition.hand_eye_capture_cli import main
from stacking_tabletop.acquisition.realsense import RealSenseCamera, extrinsic_matrix
from stacking_tabletop.ros.hand_eye import bracketed_joints, HandEyeRobot
from test_tabletop_hand_eye_calibration import profile_path
from test_tabletop_femto_ros import bindings as ros_bindings, header
from test_tabletop_ros_hardware import Node
from stacking_tabletop.execution.motion import UR_JOINT_NAMES


class Clock:
    now = 1_800_000_000_000_000_000

    def time_ns(self):
        return self.now

    def monotonic(self):
        return (self.now - 1_799_999_990_000_000_000) / 1e9


def pair(stamp=1_800_000_000_000_000_000, number=1):
    result = dict(receipt_ns=stamp + 10_000_000, receipt_monotonic_ns=10_000_000_000 + number * 67_000_000)
    for name, shape, dtype in (('color', (720, 1280, 3), np.uint8), ('depth', (2, 3), np.uint16)):
        result[name] = dict(timestamp_ns=stamp + (1_000_000 if name == 'depth' else 0),
            frame_number=number, timestamp_domain='timestamp_domain.global_time', metadata={},
            image=np.full(shape, number, dtype=dtype))
    return result


class Camera:
    def __init__(self, calibration, clock):
        self.clock, self.number = clock, 0
        self.report = dict(device=dict(serial_number=calibration.profile.serial),
            streams=dict(color=dict(width=1280, height=720, K=calibration.factory_matrix.tolist(),
                                    distortion=calibration.factory_distortion.tolist()),
                         depth=dict(width=3, height=2)),
            color_T_depth=np.eye(4).tolist(), depth_metres_per_unit=.00025)
        self.report['color_T_depth'][0][3] = .014

    def read(self):
        self.number += 1
        self.clock.now += 67_000_000
        return pair(self.clock.now, self.number)


class Robot:
    def __init__(self):
        self.calls = 0
        self.fail = None

    def measure(self, start, end, stamps):
        self.calls += 1
        if self.fail is not None and self.calls == 3:
            raise self.fail
        times = list(range(start, end, 20_000_000)) + [end]
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler('z', 30, degrees=True).as_matrix()
        pose[:3, 3] = [.3, .2, .6]
        depth_pose = pose.copy()
        depth_pose[0, 3] += .0001
        return dict(joint_stamps_ns=times, joint_positions=np.zeros((len(times), 6)).tolist(),
                    color_base_T_tool=pose.tolist(), depth_base_T_tool=depth_pose.tolist(),
                    interval_start_ns=start, interval_end_ns=end, pose_stamps_ns=stamps)


@pytest.fixture
def setup(profile_path, monkeypatch):
    calibration = HandEyeCalibration.load(profile_path)
    clock = Clock()
    monkeypatch.setattr(capture, 'time', clock)
    return calibration, Camera(calibration, clock), Robot()


def test_portable_capture_pixels_source_time_and_separate_pose_composition(setup, tmp_path):
    calibration, camera, robot = setup
    directory = tmp_path / 'view'
    result = capture.capture_view(directory, calibration, camera, robot, count=2)
    assert result['status'] == 'complete'
    record = json.loads((directory / 'frames/0000.json').read_text())
    assert record['color']['timestamp_ns'] != record['receipt_ns']
    assert record['color']['timestamp_ns'] != record['depth']['timestamp_ns']
    np.testing.assert_allclose(record['base_link_T_depth'],
        calibration.base_link_T_color(record['robot']['depth_base_T_tool']) @ np.asarray(camera.report['color_T_depth']))
    with np.load(directory / 'frames/0000.npz') as data:
        assert data['depth'].dtype == np.uint16
        np.testing.assert_array_equal(data['depth'], np.full((2, 3), 32))
        assert data['color'].dtype == np.uint8
    shutil.move(directory, tmp_path / 'relocated')
    for artifact in calibration.profile.hand_eye, calibration.profile.intrinsics, calibration.profile.installation:
        artifact.path.unlink()
    assert capture.inspect_capture(tmp_path / 'relocated')['frame_count'] == 2
    assert main(['inspect', str(tmp_path / 'relocated')]) == 0


@pytest.mark.parametrize('error', [ValueError('missing TF'), KeyboardInterrupt()])
def test_interruption_preserves_partial_frames_as_incomplete(setup, tmp_path, error):
    calibration, camera, robot = setup
    robot.fail = error
    directory = tmp_path / 'view'
    with pytest.raises(type(error)):
        capture.capture_view(directory, calibration, camera, robot, count=3)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'incomplete' and len(manifest['frames']) == 1
    with pytest.raises(ValueError, match='incomplete'):
        capture.inspect_capture(directory)
    with pytest.raises(FileExistsError):
        capture.capture_view(directory, calibration, camera, robot)


@pytest.mark.parametrize('target', ['frames/0000.npz', 'frames/0000.json', 'device.json', 'calibration/hand_eye.npz'])
def test_replay_rejects_corrupted_inputs(setup, tmp_path, target):
    calibration, camera, robot = setup
    directory = tmp_path / 'view'
    capture.capture_view(directory, calibration, camera, robot, count=1)
    with (directory / target).open('ab') as stream:
        stream.write(b'corrupted')
    with pytest.raises(ValueError, match='hash mismatch'):
        capture.inspect_capture(directory)


@pytest.mark.parametrize('change,reason', [
    (lambda p: p['color'].update(timestamp_domain='timestamp_domain.hardware_clock'), 'unmapped'),
    (lambda p: p['depth'].update(timestamp_ns=p['receipt_ns'] - 600_000_000), 'stale'),
    (lambda p: p['depth'].update(timestamp_ns=p['receipt_ns'] + 30_000_000), 'future'),
    (lambda p: p['depth'].update(timestamp_ns=p['color']['timestamp_ns'] - 150_000_000), 'too far apart'),
    (lambda p: p['color'].update(frame_number=0), 'invalid frame'),
])
def test_timestamp_failures(change, reason):
    value = pair()
    change(value)
    with pytest.raises(ValueError, match=reason):
        capture.TimestampGuard(0).check(value)


def test_duplicate_reset_host_step_and_post_settle_source_gate():
    first = pair()
    guard = capture.TimestampGuard(0)
    guard.check(first)
    with pytest.raises(ValueError, match='duplicate'):
        guard.check(first)
    second = pair(first['color']['timestamp_ns'] + 67_000_000, 2)
    second['receipt_ns'] += 30_000_000
    with pytest.raises(ValueError, match='clock stepped'):
        guard.check(second)
    with pytest.raises(ValueError, match='predates'):
        capture.TimestampGuard(first['color']['timestamp_ns']).check(first)


def test_native_frame_timeout_is_retried_but_other_camera_errors_are_not():
    class Camera:
        calls = 0

        def read(self):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("Frame didn't arrive within 1000")
            return 'frame'

    camera = Camera()
    assert capture._read_camera(camera) == 'frame'
    assert camera.calls == 3

    class FailedCamera:
        calls = 0

        def read(self):
            self.calls += 1
            raise RuntimeError('device disconnected')

    failed = FailedCamera()
    with pytest.raises(RuntimeError, match='device disconnected'):
        capture._read_camera(failed)
    assert failed.calls == 1

    camera = Camera()
    camera.calls = 0
    with pytest.raises(RuntimeError, match='2 consecutive 1000 ms waits'):
        capture._read_camera(camera, attempts=2)


def test_stationarity_uses_all_joint_samples_and_both_camera_poses():
    packet = Robot().measure(0, 500_000_000, dict(color=490_000_000, depth=500_000_000))
    baseline = np.asarray(packet['color_base_T_tool'])
    capture.stationary_evidence(packet, np.zeros(6), baseline)
    moving = copy.deepcopy(packet)
    moving['joint_positions'][5][0] = .001
    with pytest.raises(ValueError, match='joint motion'):
        capture.stationary_evidence(moving, np.zeros(6), baseline)
    moving = copy.deepcopy(packet)
    moving['depth_base_T_tool'][2][3] += .001
    with pytest.raises(ValueError, match='tool motion'):
        capture.stationary_evidence(moving, np.zeros(6), baseline)
    packet['joint_stamps_ns'] = packet['joint_stamps_ns'][4:]
    packet['joint_positions'] = packet['joint_positions'][4:]
    with pytest.raises(ValueError, match='does not bracket'):
        capture.stationary_evidence(packet, np.zeros(6), baseline)


def test_bracketed_feedback_rejects_nearest_only_and_gaps():
    samples = [(i, np.zeros(6)) for i in range(0, 200_000_001, 20_000_000)]
    result = bracketed_joints(samples, 21_000_000, 99_000_000)
    assert result['joint_stamps_ns'] == [20_000_000, 40_000_000, 60_000_000, 80_000_000, 100_000_000]
    assert bracketed_joints(samples, -1, 90_000_000) is None
    assert bracketed_joints(samples, 1, 201_000_000) is None
    assert bracketed_joints([samples[0], samples[-1]], 1, 190_000_000) is None


def test_sdk_extrinsics_column_major_direction():
    rotation = Rotation.from_euler('xyz', [10, 20, 30], degrees=True).as_matrix()
    transform = extrinsic_matrix(NS(rotation=rotation.flatten(order='F'), translation=[.01, -.02, .03]))
    point = np.array([.1, .2, .3])
    np.testing.assert_allclose((transform @ np.r_[point, 1])[:3], rotation @ point + [.01, -.02, .03])


def test_ros_reader_has_no_commands_and_uses_both_exact_timestamps(profile_path, monkeypatch):
    import time
    class SystemNode(Node):
        def get_parameter(self, name):
            return NS(value=False)
        def get_clock(self):
            return NS(now=lambda: NS(nanoseconds=time.time_ns()))
    api = ros_bindings()
    api.Node = SystemNode
    api.sensor_qos = NS(depth=5)
    robot = HandEyeRobot(HandEyeCalibration.load(profile_path), bindings=api)
    node = robot.node
    try:
        now = time.time_ns()
        for stamp in range(now - 600_000_000, now + 20_000_001, 20_000_000):
            robot._joint(NS(header=header(stamp), name=UR_JOINT_NAMES, position=np.zeros(6)))
        stamps = dict(color=now - 50_000_000, depth=now - 40_000_000)
        evidence = robot.measure(now - 550_000_000, now - 40_000_000, stamps)
        assert evidence['pose_stamps_ns'] == stamps
        assert robot.tf.calls == [('base', 'tool0', stamps['color']), ('base', 'tool0', stamps['depth'])]
        assert set(node.subscriptions) == {'/joint_states'}
        assert node.subscriptions['/joint_states'].qos.depth == 1000
        assert api.sensor_qos.depth == 5
        assert not node.publishers and not node.clients
        robot.tf.fail = True
        with pytest.raises(capture.RobotEvidenceUnavailable, match='exact-time TF') as failure:
            robot.measure(now - 550_000_000, now - 40_000_000, stamps)
        assert failure.value.diagnostics['reason'] == 'exact_time_tf_unavailable'
        assert failure.value.diagnostics['tf_error'] == 'no exact transform'
        assert all(call[2] in stamps.values() for call in robot.tf.calls)
    finally:
        robot.close()
    assert node.destroyed and not robot.context.ok() and not robot.thread.is_alive()


def test_initial_sync_retry_uses_fresh_settling_interval(setup, tmp_path):
    calibration, camera, robot = setup
    original = robot.measure
    intervals = []
    def measure(start, end, stamps):
        intervals.append((start, end))
        if len(intervals) == 1:
            raise capture.RobotEvidenceUnavailable({'reason': 'joint_gap'})
        return original(start, end, stamps)
    robot.measure = measure
    result = capture.capture_view(tmp_path / 'view', calibration, camera, robot, count=1)
    assert result['status'] == 'complete'
    assert result['settling_retries'] == [{'reason': 'joint_gap'}]
    assert intervals[1][0] > intervals[0][1]
    assert result['settling']['interval_start_ns'] == intervals[1][0]
    assert capture.inspect_capture(tmp_path / 'view')['frame_count'] == 1


def test_initial_sync_retry_is_bounded_and_retains_diagnostics(setup, tmp_path):
    calibration, camera, robot = setup
    attempts = []
    def measure(*args):
        attempts.append(args)
        raise capture.RobotEvidenceUnavailable({'reason': 'exact_time_tf_unavailable', 'tf_error': 'no TF'})
    robot.measure = measure
    with pytest.raises(capture.RobotEvidenceUnavailable):
        capture.capture_view(tmp_path / 'view', calibration, camera, robot, count=1)
    manifest = json.loads((tmp_path / 'view/manifest.json').read_text())
    assert len(attempts) == 3
    assert manifest['status'] == 'incomplete' and not manifest['frames']
    assert manifest['robot_timing_failure']['tf_error'] == 'no TF'


def test_sync_gap_after_first_saved_frame_is_not_retried(setup, tmp_path):
    calibration, camera, robot = setup
    robot.fail = capture.RobotEvidenceUnavailable({'reason': 'joint_gap'})
    with pytest.raises(capture.RobotEvidenceUnavailable):
        capture.capture_view(tmp_path / 'view', calibration, camera, robot, count=3)
    manifest = json.loads((tmp_path / 'view/manifest.json').read_text())
    assert len(manifest['frames']) == 1 and robot.calls == 3
    assert 'settling_retries' not in manifest


@pytest.mark.parametrize('reason,offsets', [
    ('joint_bracket_missing', [-400_000_000, 0]),
    ('joint_gap', [-600_000_000, -300_000_000, 0]),
])
def test_robot_timing_error_distinguishes_missing_bounds_and_gaps(reason, offsets):
    import threading
    import time
    robot = HandEyeRobot.__new__(HandEyeRobot)
    robot._error = None
    robot._lock = threading.Lock()
    now = time.time_ns()
    robot._joints = [(now + offset, np.zeros(6), time.monotonic_ns()) for offset in offsets]
    robot.context = NS(ok=lambda: True)
    robot.subscription = NS(get_publisher_count=lambda: 1)
    robot.node = NS(get_clock=lambda: NS(now=lambda: NS(nanoseconds=time.time_ns())))
    stamps = dict(color=now-81_000_000, depth=now-80_000_000)
    with pytest.raises(capture.RobotEvidenceUnavailable) as failure:
        robot.measure(now-500_000_000, stamps['depth'], stamps)
    report = failure.value.diagnostics
    assert report['reason'] == reason
    assert report['joint_samples'] == len(offsets)
    assert report['pose_stamps_ns'] == stamps
    if reason == 'joint_gap':
        assert report['max_interval_joint_gap_ms'] == 300.


def test_partial_repeat_does_not_consume_the_other_stream():
    first = pair()
    guard = capture.TimestampGuard(0)
    guard.check(first)
    second = pair(first['color']['timestamp_ns'] + 67_000_000, 2)
    mixed = copy.deepcopy(second)
    mixed['depth'] = copy.deepcopy(first['depth'])
    with pytest.raises(capture.RepeatedFrame):
        guard.check(mixed)
    assert guard.previous['color'][1] == 1
    # The next frameset can reuse the new color with a new depth exposure.
    guard.check(second)
    assert all(number == 2 for stamp, number in guard.previous.values())


@pytest.mark.parametrize('stream', ['color', 'depth'])
@pytest.mark.parametrize('change', ['counter_backward', 'time_backward', 'same_counter_new_time', 'new_counter_same_time'])
def test_real_reset_or_inconsistent_identity_is_never_skipped(stream, change):
    first = pair(number=10)
    guard = capture.TimestampGuard(0)
    guard.check(first)
    second = copy.deepcopy(first)
    if change == 'counter_backward':
        second[stream]['frame_number'] -= 1
    elif change == 'time_backward':
        second[stream]['timestamp_ns'] -= 1
    elif change == 'same_counter_new_time':
        second[stream]['timestamp_ns'] += 1
    else:
        second[stream]['frame_number'] += 1
    camera = NS(read=lambda: second)
    with pytest.raises(ValueError, match=stream + ' camera clock/counter'):
        capture.read_unique_pair(camera, guard)
    assert guard.repeated_pairs == 0


def test_repeated_frameset_wait_is_bounded():
    first = pair()
    guard = capture.TimestampGuard(0)
    guard.check(first)
    with pytest.raises(ValueError, match='no unique RGB/depth pair after 10'):
        capture.read_unique_pair(NS(read=lambda: first), guard)
    assert guard.repeated_pairs == 10


def test_repeat_does_not_hide_other_stream_clock_reset():
    first = pair(number=10)
    guard = capture.TimestampGuard(0)
    guard.check(first)
    second = pair(first['color']['timestamp_ns'] + 67_000_000, 11)
    second['color'] = copy.deepcopy(first['color'])
    with pytest.raises(capture.RepeatedFrame):
        guard.check(second)
    # Depth regressed to an earlier observed exposure while color repeats.
    with pytest.raises(ValueError, match='depth camera clock/counter'):
        capture.read_unique_pair(NS(read=lambda: first), guard)


def test_capture_skips_exact_repeat_and_replay_has_only_unique_exposures(setup, tmp_path):
    calibration, camera, robot = setup
    original = camera.read
    previous = []
    injected = []
    def read():
        if camera.number == 33 and not injected:
            injected.append(True)
            return previous[-1]
        result = original()
        previous.append(result)
        return result
    camera.read = read
    result = capture.capture_view(tmp_path / 'view', calibration, camera, robot, count=3)
    assert result['repeated_pairs_discarded'] == 1
    assert capture.inspect_capture(tmp_path / 'view')['frame_count'] == 3
    records = [json.loads((tmp_path / 'view' / (stem + '.json')).read_text()) for stem in result['frames']]
    for stream in ('color', 'depth'):
        assert [r[stream]['frame_number'] for r in records] == [32, 33, 34]


def test_reported_133ms_repeated_depth_is_discarded_before_pair_skew():
    # Reproduce field_20261016_141743_f8399c69's actual frame identities.
    def sample(color_stamp, color_number, depth_stamp, depth_number, receipt):
        value = pair(color_stamp)
        value['color']['frame_number'] = color_number
        value['depth'].update(timestamp_ns=depth_stamp, frame_number=depth_number)
        value['receipt_ns'] = receipt
        value['receipt_monotonic_ns'] = receipt - 1_700_000_000_000_000_000
        return value
    previous = sample(1792127892906342400, 583, 1792127892906298112, 581,
                      1792127892981355645)
    repeated = sample(1792127893039788800, 585, 1792127892906298112, 581,
                      1792127893114802045)
    fresh = sample(1792127893039788800, 585, 1792127893039744512, 583,
                   1792127893114802046)
    guard = capture.TimestampGuard(0)
    guard.check(previous)
    frames = iter([repeated, fresh])
    assert capture.read_unique_pair(NS(read=lambda: next(frames)), guard) is fresh
    assert guard.repeated_pairs == 1
    assert guard.previous == {'color': (1792127893039788800, 585),
                              'depth': (1792127893039744512, 583)}


@pytest.mark.parametrize('repeated_stream', ['color', 'depth'])
def test_repeated_stream_with_excessive_skew_checks_reset_and_discards_aged_repeat(repeated_stream):
    first = pair(number=10)
    guard = capture.TimestampGuard(0)
    guard.check(first)
    other = 'depth' if repeated_stream == 'color' else 'color'
    mixed = pair(first['color']['timestamp_ns'] + 134_000_000, 12)
    mixed[repeated_stream] = copy.deepcopy(first[repeated_stream])
    with pytest.raises(capture.RepeatedFrame):
        guard.check(mixed)
    assert guard.previous[other][1] == 10
    reset = copy.deepcopy(mixed)
    reset[other]['frame_number'] -= 1
    with pytest.raises(ValueError, match='clock/counter reset'):
        guard.check(reset)
    stale = copy.deepcopy(mixed)
    stale['receipt_ns'] += 500_000_000
    stale['receipt_monotonic_ns'] += 500_000_000
    with pytest.raises(capture.RepeatedFrame):
        guard.check(stale)


def test_two_new_exposures_with_excessive_skew_still_fail():
    first = pair(number=10)
    guard = capture.TimestampGuard(0)
    guard.check(first)
    fresh = pair(first['color']['timestamp_ns'] + 134_000_000, 12)
    fresh['depth'].update(timestamp_ns=first['depth']['timestamp_ns'] + 1_000_000,
                          frame_number=11)
    with pytest.raises(ValueError, match='too far apart'):
        capture.read_unique_pair(NS(read=lambda: fresh), guard)
    assert guard.repeated_pairs == 0


def test_reported_599ms_repeated_depth_is_discarded_and_fresh_pair_used():
    def sample(color_stamp, color_number, depth_stamp, depth_number, receipt):
        value = pair(color_stamp)
        value['color']['frame_number'] = color_number
        value['depth'].update(timestamp_ns=depth_stamp, frame_number=depth_number)
        value['receipt_ns'] = receipt
        value['receipt_monotonic_ns'] = receipt - 1_700_000_000_000_000_000
        return value
    previous = sample(1792128105098894592, 1182, 1792128105098850560, 1180,
                      1792128105163894592)
    repeated = sample(1792128105299062528, 1185, 1792128105098850560, 1180,
                      1792128105698033191)
    fresh = sample(1792128105699062528, 1191, 1792128105699018496, 1189,
                   1792128105764062528)
    guard = capture.TimestampGuard(0)
    guard.check(previous)
    frames = iter([repeated, fresh])
    assert capture.read_unique_pair(NS(read=lambda: next(frames)), guard) is fresh
    assert guard.repeated_pairs == 1
    assert guard.previous['depth'] == (1792128105699018496, 1189)


@pytest.mark.parametrize('streams', [('color',), ('depth',), ('color', 'depth')])
@pytest.mark.parametrize('elapsed_ms', [67, 200, 600, 1200])
def test_exact_repeats_age_without_becoming_accepted(streams, elapsed_ms):
    first = pair(number=10)
    guard = capture.TimestampGuard(0)
    guard.check(first)
    next_pair = copy.deepcopy(first)
    next_pair['receipt_ns'] += elapsed_ms * 1_000_000
    next_pair['receipt_monotonic_ns'] += elapsed_ms * 1_000_000
    for name in ('color', 'depth'):
        if name not in streams:
            next_pair[name]['timestamp_ns'] += elapsed_ms * 1_000_000
            next_pair[name]['frame_number'] += 1
    with pytest.raises(capture.RepeatedFrame):
        guard.check(next_pair)
    assert all(number == 10 for stamp, number in guard.previous.values())


def test_stalled_old_stream_exhausts_bounded_reads():
    first = pair()
    guard = capture.TimestampGuard(0)
    guard.check(first)
    old = copy.deepcopy(first)
    old['receipt_ns'] += 1_000_000_000
    old['receipt_monotonic_ns'] += 1_000_000_000
    with pytest.raises(ValueError, match='no unique RGB/depth pair after 10'):
        capture.read_unique_pair(NS(read=lambda: old), guard)
    assert guard.repeated_pairs == 10


@pytest.mark.parametrize('fault', ['future', 'domain', 'host_clock', 'counter_reset', 'identity'])
def test_aged_repeat_never_masks_clock_or_identity_failure_in_other_stream(fault):
    first = pair(number=10)
    guard = capture.TimestampGuard(0)
    guard.check(first)
    mixed = copy.deepcopy(first)
    mixed['receipt_ns'] += 600_000_000
    mixed['receipt_monotonic_ns'] += 600_000_000
    mixed['color'].update(timestamp_ns=first['color']['timestamp_ns'] + 600_000_000,
                           frame_number=19)
    if fault == 'future':
        mixed['color']['timestamp_ns'] = mixed['receipt_ns'] + 100_000_000
    elif fault == 'domain':
        mixed['color']['timestamp_domain'] = 'timestamp_domain.hardware_clock'
    elif fault == 'host_clock':
        mixed['receipt_ns'] += 100_000_000
    elif fault == 'counter_reset':
        mixed['color']['frame_number'] = 1
    else:
        mixed['color']['frame_number'] = 10
    with pytest.raises(ValueError) as failure:
        guard.check(mixed)
    assert not isinstance(failure.value, capture.RepeatedFrame)
    assert guard.repeated_pairs == 0


def test_two_new_but_stale_exposures_are_never_saved():
    first = pair(number=10)
    guard = capture.TimestampGuard(0)
    guard.check(first)
    new = pair(first['color']['timestamp_ns'] + 67_000_000, 11)
    new['receipt_ns'] += 600_000_000
    new['receipt_monotonic_ns'] += 600_000_000
    with pytest.raises(ValueError, match='stale/future'):
        capture.read_unique_pair(NS(read=lambda: new), guard)
    assert all(number == 10 for stamp, number in guard.previous.values())


def test_stale_repeat_capture_preserves_continuous_robot_evidence(setup, tmp_path):
    calibration, camera, robot = setup
    original = camera.read
    last = []
    delay = [0]
    def read():
        if camera.number == 33 and not delay[0]:
            delay[0] = 600_000_000
            camera.clock.now += delay[0]
            repeated = copy.deepcopy(last[-1])
            repeated['receipt_ns'] += delay[0]
            repeated['receipt_monotonic_ns'] += delay[0]
            return repeated
        result = original()
        result['receipt_monotonic_ns'] += delay[0]
        last.append(result)
        return result
    camera.read = read
    result = capture.capture_view(tmp_path / 'view', calibration, camera, robot, count=3)
    assert result['repeated_pairs_discarded'] == 1
    assert capture.inspect_capture(tmp_path / 'view')['frame_count'] == 3
    second = json.loads((tmp_path / 'view/frames/0001.json').read_text())
    third = json.loads((tmp_path / 'view/frames/0002.json').read_text())
    assert third['robot']['interval_start_ns'] == min(second[name]['timestamp_ns'] for name in ('color', 'depth'))
    assert third['robot']['interval_end_ns'] - third['robot']['interval_start_ns'] > 600_000_000
    assert third['receipt_ns'] - third['color']['timestamp_ns'] < capture.LIMITS['max_frame_age_ns']


def test_rgb_only_capture_assembly_and_rgb_inputs_without_native_depth(setup, tmp_path, monkeypatch):
    from stacking_tabletop.acquisition.hand_eye_replay import assemble_session, inspect_session, replay_session
    calibration, camera, robot = setup
    camera.report['streams'].pop('depth')
    camera.report.pop('color_T_depth')
    camera.report.pop('depth_metres_per_unit')
    original = camera.read
    def read():
        value = original()
        value.pop('depth')
        return value
    camera.read = read
    original_measure = robot.measure
    def measure(start, end, stamps):
        assert set(stamps) == {'color'}
        result = original_measure(start, end, stamps)
        result.pop('depth_base_T_tool')
        return result
    robot.measure = measure
    views = [tmp_path / f'view_{i}' for i in range(2)]
    for view in views:
        result = capture.capture_view(view, calibration, camera, robot, count=2)
        assert result['schema_version'] == 2 and result['streams'] == ['color']
        assert capture.inspect_capture(view)['frame_count'] == 2
        record = json.loads((view/'frames/0000.json').read_text())
        assert 'depth' not in record and 'base_link_T_depth' not in record
        with np.load(view/'frames/0000.npz') as data:
            assert data.files == ['color']
    session = tmp_path / 'session'
    assemble_session(session, views)
    manifest = inspect_session(session)
    assert set(manifest['identity']['streams']) == {'color'}
    # This is the same RGB + calibrated pose input consumed by learned stereo.
    for view in manifest['views']:
        root = session / view['path']
        stem = json.loads((root/'manifest.json').read_text())['frames'][0]
        with np.load(root/(stem+'.npz')) as arrays:
            assert arrays['color'].shape == (720, 1280, 3)
        assert np.asarray(json.loads((root/(stem+'.json')).read_text())['base_link_T_color']).shape == (4, 4)
    with pytest.raises(ValueError, match='no native depth'):
        replay_session(session, tmp_path/'depth_replay')
    assert not (tmp_path/'depth_replay').exists()
    # Exercise the actual learned-stereo input loader up to the GPU boundary.
    # No native depth archive or depth camera geometry is available to it.
    from pathlib import Path
    from stacking_tabletop.acquisition import hand_eye_fast_stereo as stereo
    def model_boundary(*args):
        raise RuntimeError('RGB inputs reached model initialization')
    monkeypatch.setattr(stereo, 'FastFoundationStereo', model_boundary)
    region = tmp_path/'region.json'
    capture.atomic_json(region, dict(frame='base_link', minimum=[0., 0., 0.],
        maximum=[1., 1., 1.], stone_boxes_xy=[[[0., 0.], [1., 1.]]]))
    checkpoint = tmp_path/'checkpoint'
    checkpoint.write_bytes(b'test checkpoint; no inference')
    with pytest.raises(RuntimeError, match='RGB inputs reached model initialization'):
        stereo.reconstruct(session, None, None, tmp_path/'stereo',
            repository=Path(__file__).resolve().parents[1], checkpoint=checkpoint, region_file=region)
