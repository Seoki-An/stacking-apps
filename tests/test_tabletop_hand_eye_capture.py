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
        assert not node.publishers and not node.clients
        robot.tf.fail = True
        with pytest.raises(ValueError, match='exact-time TF'):
            robot.measure(now - 550_000_000, now - 40_000_000, stamps)
        assert all(call[2] in stamps.values() for call in robot.tf.calls)
    finally:
        robot.close()
    assert node.destroyed and not robot.context.ok() and not robot.thread.is_alive()
