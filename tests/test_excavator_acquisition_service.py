"""No-device acquisition ownership, outage capacity and remote transfer faults."""

from pathlib import Path
import subprocess
import shutil

import numpy as np
import pytest

from stacking_excavator.acquisition.capture import CaptureFrame, CropRequest, digest
from stacking_excavator.acquisition.spool import CaptureSpool
from stacking_excavator.acquisition.stationary import StationaryCaptureService
from stacking_excavator.acquisition.delivery import pull_product
from stacking_excavator.acquisition.storage import create_capture, prepare_crop, load_product
from stacking_excavator.acquisition.legacy import import_legacy_log

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scan():
    sha = digest({"calibration": "test"})
    frame = CaptureFrame("frame", "lidar", 1, 1, np.eye(4), np.eye(4), sha,
                         np.array([[0., 0., 1.]]), (0.,)*6, -.4)
    crop = CropRequest("field", "world", np.eye(4), (2.,)*3, (.1,)*3, sha, 0, (10,))
    return frame, crop


def test_retries_do_not_recapture_or_evict_retained_raw(tmp_path, scan):
    frame, crop = scan

    class Source:
        calls = 0
        def frames(self, job_id):
            self.calls += 1
            yield frame

    source = Source()
    spool = CaptureSpool(tmp_path, capacity_bytes=40000)
    service = StationaryCaptureService(spool, source)
    result = service.capture("scan", crop, maximum_bytes=30000, maximum_frames=1)
    assert service.capture("scan", crop, maximum_bytes=30000, maximum_frames=1) == result
    assert source.calls == 1
    assert CaptureSpool(tmp_path, capacity_bytes=40000).status("scan")["state"] == "retained"
    with pytest.raises(ValueError, match="capacity"):
        service.capture("next", crop, maximum_bytes=30000, maximum_frames=1)
    assert (tmp_path / "scan/raw/manifest.json").exists()


def test_interrupted_source_requires_reconciliation(tmp_path, scan):
    frame, crop = scan

    class Source:
        calls = 0
        def frames(self, job_id):
            self.calls += 1
            yield frame
            raise RuntimeError("sensor disconnected")

    source = Source()
    spool = CaptureSpool(tmp_path, capacity_bytes=50000)
    service = StationaryCaptureService(spool, source)
    with pytest.raises(RuntimeError, match="disconnected"):
        service.capture("scan", crop, maximum_bytes=30000, maximum_frames=2)
    with pytest.raises(ValueError, match="reconciliation"):
        service.capture("scan", crop, maximum_bytes=30000, maximum_frames=2)
    assert source.calls == 1
    assert not (tmp_path / "scan/raw/manifest.json").exists()


def test_declared_frame_bound_is_enforced(tmp_path, scan):
    frame, crop = scan

    class Source:
        def frames(self, job_id):
            yield frame
            yield frame

    service = StationaryCaptureService(CaptureSpool(tmp_path, capacity_bytes=100000), Source())
    with pytest.raises(ValueError, match="frame/storage bound"):
        service.capture("scan", crop, maximum_bytes=80000, maximum_frames=1)
    assert not (tmp_path / "scan/raw/manifest.json").exists()


def test_rsync_stages_then_verifies_and_uses_bounded_options(tmp_path, scan):
    frame, crop = scan
    create_capture(tmp_path / "raw", "scan", "field", 0, [frame])
    prepare_crop(tmp_path / "raw", tmp_path / "product", "v1", crop, maximum_timestamp_skew_ns=0)
    calls = []

    class Process:
        returncode = 0
        def communicate(self, timeout=None):
            return None, b""
        def poll(self):
            return self.returncode

    def spawn(args, **kwargs):
        calls.append(args)
        shutil.copytree(tmp_path / "product", Path(args[-1]), dirs_exist_ok=True)
        return Process()

    assert pull_product("nuc", "/spool/job/crop-v1", tmp_path / "staging", tmp_path / "received",
        bandwidth_kib=256, control_fresh=lambda: True, spawn=spawn)
    assert "--bwlimit=256" in calls[0]
    assert "--partial-dir=.rsync-partial" in calls[0]
    assert "--include=/*.npz" not in calls[0]
    assert "--include=/*.npz" in calls[1]
    assert load_product(tmp_path / "received")[0]["product_id"] == "v1"


def test_stale_control_terminates_bulk_process(tmp_path):
    freshness = iter([True, True, False])

    class Process:
        terminated = False
        def communicate(self, timeout=None):
            if not self.terminated:
                raise subprocess.TimeoutExpired("rsync", timeout)
            return None, b""
        def poll(self):
            return 1 if self.terminated else None
        def terminate(self):
            self.terminated = True

    process = Process()
    with pytest.raises(ValueError, match="stale"):
        pull_product("nuc", "/spool/crop", tmp_path / "staging", tmp_path / "received",
            bandwidth_kib=128, control_fresh=lambda: next(freshness), spawn=lambda *a, **k: process)
    assert process.terminated
    assert not (tmp_path / "received/manifest.json").exists()


def test_remote_raw_capture_is_rejected_before_point_payload_transfer(tmp_path, scan):
    frame, _ = scan
    create_capture(tmp_path / "raw", "scan", "field", 0, [frame])
    calls = []
    class Process:
        returncode = 0
        def communicate(self, timeout=None):
            return None, b""
        def poll(self):
            return 0
    def spawn(args, **kwargs):
        calls.append(args)
        shutil.copyfile(tmp_path / "raw/manifest.json", Path(args[-1]) / "manifest.json")
        return Process()
    with pytest.raises(ValueError, match="not raw"):
        pull_product("nuc", "/raw", tmp_path / "staging", tmp_path / "received",
            bandwidth_kib=128, control_fresh=lambda: True, spawn=spawn)
    assert len(calls) == 1 and "--include=/*.npz" not in calls[0]


def test_legacy_field_transform_is_used_exactly_once(tmp_path):
    source = tmp_path / "logs"
    source.mkdir()
    sensor = np.eye(4)
    sensor[:3, 3] = [4., 2., 1.]
    np.save(source / "field_raw_iv_points21_swing0.npy", np.array([[1., 0., 0.]]))
    np.save(source / "field_lidar_frame_iv_points21_swing0.npy", sensor)
    np.save(source / "field_scan_q_actual_swing0.npy", np.zeros(6))
    manifest = import_legacy_log(source, tmp_path / "raw", repository=ROOT,
        kind="field", capture_id="legacy", scene_revision=0)
    assert not manifest["recording"]["timing_verified"]
    assert manifest["frames"][0]["timestamp_ns"] is None
    assert manifest["frames"][0]["gripper_opening"] is None
    crop = CropRequest("field", "world", sensor, (2.,)*3, (0.,)*3,
        manifest["frames"][0]["calibration_sha256"], 0, (10,))
    prepare_crop(tmp_path / "raw", tmp_path / "crop", "v1", crop, maximum_timestamp_skew_ns=0)
    _, frames = load_product(tmp_path / "crop")
    f = frames[0]
    np.testing.assert_allclose(f.points @ f.world_T_sensor[:3, :3].T + f.world_T_sensor[:3, 3], [[5,2,1]])


def test_unverified_recording_cannot_be_a_live_stationary_source(tmp_path, scan):
    from dataclasses import replace
    frame, crop = scan
    class Source:
        def frames(self, job_id):
            yield replace(frame, timestamp_ns=None, joint_timestamp_ns=None,
                          timestamp_source="recorded_unknown")
    service = StationaryCaptureService(CaptureSpool(tmp_path, capacity_bytes=50000), Source())
    with pytest.raises(ValueError, match="sensor timing"):
        service.capture("scan", crop, maximum_bytes=30000, maximum_frames=1)


@pytest.mark.parametrize("kind", ["field", "scene"])
def test_real_legacy_frame_fixture_matches_saved_world_cloud(kind):
    import json
    from stacking_excavator.acquisition.storage import file_hash
    fixture = ROOT / "tests/fixtures/legacy_planner/captures.npz"
    metadata = json.loads(fixture.with_suffix(".json").read_text())
    assert file_hash(fixture) == metadata["sha256"]
    assert metadata["timestamp_source"] == "recorded_unknown"
    with np.load(fixture, allow_pickle=False) as arrays:
        sensor = arrays[kind + "_world_T_sensor"]
        world = arrays[kind + "_points"] @ sensor[:3, :3].T + sensor[:3, 3]
        np.testing.assert_allclose(world, arrays[kind + "_world_points"], atol=1e-6)


def test_real_inhand_fixture_is_not_transformed_twice(tmp_path):
    o3d = pytest.importorskip("open3d")
    fixture = ROOT / "tests/fixtures/legacy_planner/captures.npz"
    source = tmp_path / "logs"
    source.mkdir()
    with np.load(fixture, allow_pickle=False) as arrays:
        points = arrays["inhand_points"]
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        assert o3d.io.write_point_cloud(str(source / "pcd_0.ply"), cloud)
        np.save(source / "q_list.npy", arrays["inhand_joints"][None, :])
    manifest = import_legacy_log(source, tmp_path / "raw", repository=ROOT,
        kind="inhand", capture_id="inhand", scene_revision=0)
    assert not manifest["recording"]["original_sensor_frame"]
    from stacking_excavator.acquisition.storage import read_frame
    frame = read_frame(tmp_path / "raw", manifest["frames"][0])
    transform = np.linalg.inv(frame.world_T_gripper) @ frame.world_T_sensor
    np.testing.assert_allclose(frame.points @ transform[:3, :3].T + transform[:3, 3], points, atol=1e-10)
