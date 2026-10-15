"""FEMTO acquisition against synthetic frames, clocks and calibration artifacts."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace as NS

import numpy as np
import pytest

from stacking_tabletop.acquisition.femto import (
    FemtoCalibration, FemtoBuffer, FemtoCamera, CaptureError, CaptureCancelled, DEPTH_MODE,
)
from stacking_tabletop.acquisition.storage import CaptureSession
from stacking_tabletop.config.femto import FemtoProfile, load_femto_profile
from stacking_tabletop.perception.capture import TabletopRawCapture
from stacking_tabletop.perception.organized_depth import DepthFilterConfig, OrganizedDepthDataset


@pytest.fixture
def calibration(tmp_path):
    pose, depth = tmp_path / 'pose.npz', tmp_path / 'depth.npz'
    # Distinct optical origin catches wrong multiplication direction.
    depth_T_color = np.eye(4)
    depth_T_color[0, 3] = .02
    np.savez(pose, T_base_to_camera=np.eye(4), T_base_to_observation=np.eye(4),
             parent_frame='base_link', child_frame='camera_link', observation_frame='camera_color_optical_frame',
             camera_serial='test-camera', camera='FEMTO_BOLT', method='new_frame_aware', ee_pose_frame='base_link')
    np.savez(depth, method='femto_charuco_plane_robust_offset_v2', depth_scale=1., depth_offset_m=.01,
             depth_frame='camera_depth_optical_frame', color_frame='camera_color_optical_frame',
             depth_mode=DEPTH_MODE, camera_serial='test-camera', capture_depth_T_color=np.array([depth_T_color, depth_T_color]))
    profile = FemtoProfile('test-camera', pose, depth, '/camera', 'base_link', (-2., -2., 0.), (2., 2., 2.), .001)
    return FemtoCalibration(profile)


def info():
    return {'frame_id': 'camera_depth_optical_frame', 'width': 1024, 'height': 1024,
            'k': [1000., 0., 512., 0., 1000., 512., 0., 0., 1.], 'd': [],
            'distortion_model': 'plumb_bob', 'r': list(np.eye(3).ravel()), 'p': [0.] * 12,
            'binning_x': 0, 'binning_y': 0, 'roi': {'x_offset': 0, 'y_offset': 0, 'width': 0, 'height': 0, 'do_rectify': False}}


def frame_values(stamp=1_000_000_000):
    depth = np.zeros((1024, 1024), dtype=np.uint16)
    depth[510:515, 510:515] = 500
    return dict(depth_stamp_ns=stamp, cloud_stamp_ns=stamp + 1_000_000,
                frame_id='camera_depth_optical_frame', encoding='16UC1', depth=depth,
                points=np.array([[0., 0., .5]], np.float32), camera_info=info())


class FakeTransport:
    def __init__(self, calibration):
        self.now = 10.
        self.buffer = FemtoBuffer(calibration, clock=lambda: self.now)
        self.serial = calibration.profile.serial
        self.frames = True
        self.transforms = True
        self.robot = True
        self.joints = np.zeros(6)
        self.position = 700.
        self.tool = np.eye(4)
        self.on_sleep = None

    def ok(self):
        return True

    def verify_serial(self, expected, deadline, check):
        check()
        if expected != self.serial:
            raise CaptureError('serial mismatch')

    def lookup(self, *args):
        return np.eye(4) if self.transforms else None

    def robot_at(self, stamp, base_frame):
        return {'joints': self.joints.copy(), 'gripper_position': self.position,
                'base_T_tool': self.tool.copy(), 'joint_stamp_ns': stamp - 1_000_000,
                'gripper_stamp_ns': stamp + 1_000_000, 'joint_sample_mode': 'interpolated'} if self.robot else None

    def sleep(self, _duration):
        self.now += .21
        if self.on_sleep:
            self.on_sleep()
        if self.frames:
            self.buffer.push(**frame_values(int(self.now * 1e9)))


def camera(calibration):
    transport = FakeTransport(calibration)
    return transport, FemtoCamera(transport, calibration, clock=lambda: transport.now, sleep=transport.sleep)


def test_profiles_use_explicit_camera_identities_without_loading_ros():
    root = Path(__file__).resolve().parents[1]
    primary = load_femto_profile(root / 'apps/tabletop/config/femto.yml')
    secondary = load_femto_profile(root / 'apps/tabletop/config/femto.yml', 'secondary')
    assert primary.serial == 'CL83C53005L'
    assert secondary.serial == 'CL8956200P5'
    assert primary.namespace == secondary.namespace == '/camera'


def test_depth_origin_and_immutable_snapshot(calibration):
    assert calibration.base_T_depth[0, 3] == pytest.approx(-.02)
    np.testing.assert_allclose(calibration.transform(np.eye(4)), calibration.base_T_depth)
    bad = np.eye(4)
    bad[2, 3] = .001
    with pytest.raises(CaptureError, match='optical transform'):
        calibration.transform(bad)
    calibration.profile.pose_calibration.write_bytes(b'changed source')
    assert calibration.artifacts['camera_pose.npz'] != b'changed source'


@pytest.mark.parametrize('change', ['serial', 'mode', 'dynamic_extrinsic', 'observation_frame'])
def test_bad_calibration_rejected(calibration, change):
    profile = calibration.profile
    path = profile.pose_calibration if change == 'observation_frame' else profile.depth_calibration
    with np.load(path, allow_pickle=False) as data:
        values = dict(data)
    if change == 'serial':
        values['camera_serial'] = 'wrong'
    elif change == 'mode':
        values['depth_mode'] = 'NFOV'
    elif change == 'observation_frame':
        values['observation_frame'] = 'wrong'
    else:
        values['capture_depth_T_color'][1, 0, 3] += .001
    np.savez(path, **values)
    with pytest.raises(ValueError):
        FemtoCalibration(profile)


def test_buffer_rejects_repeated_unsynchronized_and_changed_stream(calibration):
    buffer = FemtoBuffer(calibration)
    values = frame_values()
    assert buffer.push(**values)
    assert not buffer.push(**values)
    assert not buffer.push(**{**values, 'cloud_stamp_ns': 2_000_000_000})
    frame = buffer.after(0)[0]
    values['depth'][:] = 999
    assert frame.depth[0, 0] == 0
    assert not frame.depth.flags.writeable
    with pytest.raises(CaptureError, match='encoding changed'):
        buffer.push(**{**frame_values(3_000_000_000), 'encoding': '32FC1', 'depth': np.zeros((1024, 1024), np.float32)})
    values = frame_values(4_000_000_000)
    values['camera_info']['k'][0] = 999.
    with pytest.raises(CaptureError, match='intrinsics changed'):
        buffer.push(**values)


def test_capture_warmup_distinct_frames_exact_transform_and_cancellation(calibration):
    transport, adapter = camera(calibration)
    transport.buffer.push(**frame_values(1))  # Pre-request frame is not captured.
    progress = []
    burst = adapter.capture(progress=lambda n, total: progress.append((n, total)))
    assert len(burst) == 7
    assert burst[0].frame.sequence >= 12
    assert len({frame.frame.depth_stamp_ns for frame in burst}) == 7
    assert progress == [(n, 7) for n in range(1, 8)]
    adapter.cancel.set()
    with pytest.raises(CaptureCancelled):
        adapter.capture()


@pytest.mark.parametrize('missing', ['frames', 'transforms', 'robot'])
def test_capture_timeout_never_falls_back_to_latest_or_unmatched_state(calibration, missing):
    transport, adapter = camera(calibration)
    setattr(transport, missing, False)
    expected = {'frames': 'fresh synchronized depth/cloud frames',
                'transforms': 'exact transform camera_link',
                'robot': 'timestamp-matched robot/gripper state'}
    with pytest.raises(CaptureError, match=expected[missing]) as error:
        adapter.capture(in_hand=True, timeout=5.)
    assert 'captured=0/7' in str(error.value)


def test_capture_timeout_includes_transport_receipt_counts(calibration):
    transport, adapter = camera(calibration)
    transport.frames = False
    transport.diagnostics = lambda: {'depth_messages': 0, 'cloud_messages': 0,
                                    'camera_info_received': False}
    with pytest.raises(CaptureError, match='depth_messages=0, cloud_messages=0'):
        adapter.capture(timeout=1.)


def test_wrong_live_camera_and_motion_during_burst_fail(calibration):
    transport, adapter = camera(calibration)
    transport.serial = 'other-camera'
    with pytest.raises(CaptureError, match='serial'):
        adapter.capture()
    transport.serial = calibration.profile.serial
    def progress(n, _total):
        if n == 1:
            transport.joints[5] += np.deg2rad(.5)
    with pytest.raises(CaptureError, match='robot moved'):
        adapter.capture(in_hand=True, progress=progress)


def test_field_capture_reprocesses_and_loads_for_dual_fusion(calibration, tmp_path):
    from stacking_tabletop.perception.fusion import load_recorded_femto_scan
    _, adapter = camera(calibration)
    directory = tmp_path / 'field'
    with CaptureSession(directory, calibration) as session:
        session.add(adapter.capture())
    restored = load_recorded_femto_scan(directory)
    assert restored.camera_serial == calibration.profile.serial
    assert np.mean(restored.cloud.points[:, 2]) == pytest.approx(.51, abs=1e-6)
    metadata = json.loads((directory / 'field_scan.json').read_text())
    dataset = OrganizedDepthDataset(directory, calibration.depth.depth_frame, DEPTH_MODE)
    raw, stats = dataset.points_for_view(metadata['fused_file'], DepthFilterConfig())
    assert stats.kept_pixels == 25
    assert np.mean(raw[:, 2]) == pytest.approx(.5)
    assert len(list((directory / 'raw_pcd').glob('*.pcd'))) == 7
    with pytest.raises(FileExistsError):
        CaptureSession(directory, calibration)


def test_in_hand_roundtrip_is_relocatable_and_detects_modified_snapshot(calibration, tmp_path):
    _, adapter = camera(calibration)
    original, moved = tmp_path / 'inhand', tmp_path / 'relocated'
    with CaptureSession(original, calibration, in_hand=True) as session:
        session.add(adapter.capture(in_hand=True))
    shutil.move(original, moved)
    calibration.profile.depth_calibration.unlink()
    loaded = TabletopRawCapture.load(moved)
    views = loaded.load_tool_views(DepthFilterConfig(), np.array([-1, -1, 0.]), np.ones(3))
    assert len(views) == 1
    assert np.mean(views[0].cloud.points[:, 2]) == pytest.approx(.51, abs=1e-6)
    snapshot = moved / 'calibration_artifacts/camera_depth.npz'
    snapshot.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        loaded.load_tool_views(DepthFilterConfig(), -np.ones(3), np.ones(3))


def test_failed_capture_is_marked_incomplete(calibration, tmp_path):
    directory = tmp_path / 'failed'
    with pytest.raises(CaptureCancelled):
        with CaptureSession(directory, calibration, in_hand=True):
            raise CaptureCancelled('operator cancelled')
    assert json.loads((directory / 'capture_status.json').read_text())['status'] == 'failed'
    with pytest.raises(ValueError, match='incomplete'):
        TabletopRawCapture.load(directory)


def test_field_voxels_keep_legacy_origin_anchored_floor_cells():
    from stacking_common.pointcloud import PointCloudData
    from stacking_common.pointcloud.processing import fixed_grid_centroids
    points = np.array([[-.01, 0, 0], [.01, 0, 0], [.09, 0, 0], [.11, 0, 0]])
    actual = fixed_grid_centroids(PointCloudData(points), .1).points
    np.testing.assert_allclose(actual, [[-.01, 0, 0], [.05, 0, 0], [.11, 0, 0]])


def test_cli_rejects_existing_output_before_ros_creation(calibration, tmp_path, monkeypatch):
    from stacking_tabletop.acquisition import cli
    from stacking_tabletop.ros import femto
    monkeypatch.setattr(cli, 'load_femto_profile', lambda *_: calibration.profile)
    monkeypatch.setattr(femto, 'FemtoRosTransport', lambda *_: pytest.fail('must not create a ROS node'))
    existing = tmp_path / 'existing'
    existing.mkdir()
    assert cli.main(['--output', str(existing)]) == 1


def test_second_settled_view_retains_distinct_bursts(calibration, tmp_path):
    transport, adapter = camera(calibration)
    directory = tmp_path / 'two-views'
    with CaptureSession(directory, calibration, in_hand=True) as session:
        session.add(adapter.capture(in_hand=True))
        transport.joints[5] = .1  # Only between bursts.
        session.add(adapter.capture(in_hand=True))
    loaded = TabletopRawCapture.load(directory)
    assert [view.capture_index for view in loaded.views] == [0, 1]
    assert loaded.views[1].joints[5] == .1
    assert len(loaded.depth.views) == 2


def test_external_health_and_callback_fault_checked_during_identity_wait(calibration):
    transport, adapter = camera(calibration)
    checked = []
    def external():
        checked.append(True)
        raise CaptureError('held object lost')
    with pytest.raises(CaptureError, match='held object lost'):
        adapter.capture(in_hand=True, check=external)
    assert checked == [True]
    transport.buffer.fail('invalid depth data')
    with pytest.raises(CaptureError, match='invalid depth data'):
        adapter.capture()
