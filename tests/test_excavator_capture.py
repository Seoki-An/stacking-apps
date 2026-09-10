"""Purpose crops and disconnected transfer, with no ROS, NUC or robot calls."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stacking_common.pointcloud.cropping import box_mask
from stacking_excavator.acquisition.capture import CaptureFrame, CropRequest, GRIPPER_FRAME, digest
from stacking_excavator.acquisition.storage import (
    create_capture, file_hash, load_product, prepare_crop, receive_local,
)


def transform(xyz, angles):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    result[:3, 3] = xyz
    return result


@pytest.fixture
def capture():
    # Geometry frame has a rotation AND an offset: using the bare link is wrong.
    gripper = transform([4., -3., 2.], [.3, -.5, 1.2])
    sensor = transform([-2., 1., 3.], [-.1, .4, -.7])
    geometry_points = np.array([[0., 0., 0.], [.4, .4, 0.], [1.1, 0., 0.],
                                [0., -1.1, 0.], [0., 0., 2.]])
    sensor_T_gripper = np.linalg.inv(sensor) @ gripper
    points = geometry_points @ sensor_T_gripper[:3, :3].T + sensor_T_gripper[:3, 3]
    frame = CaptureFrame("frame-0", "lidar-2", 1000, 1000, sensor, gripper,
        digest({"calibration": "test"}), points, (0.,) * 6, -.45,
        attributes={"intensity": np.arange(5, dtype=np.uint16),
                    "time_ns": np.arange(5, dtype=np.uint64) + 2**60})
    crop = CropRequest("inhand", GRIPPER_FRAME, np.eye(4), (.5, .5, .5),
                       (.1, .1, .1), frame.calibration_sha256, 4, (10,))
    return frame, crop


def product(tmp_path, capture):
    frame, crop = capture
    raw, cropped = tmp_path / "raw", tmp_path / "crop"
    second = replace(frame, frame_id="frame-1")
    create_capture(raw, "scan", "inhand", 4, [frame, second])
    prepare_crop(raw, cropped, "crop-v1", crop, maximum_timestamp_skew_ns=0)
    return raw, cropped


def test_capture_time_geometry_crop_preserves_exact_channels(tmp_path, capture):
    frame, crop = capture
    np.testing.assert_array_equal(crop.select(frame, maximum_timestamp_skew_ns=0),
                                  [True, True, False, False, False])
    raw, cropped = product(tmp_path, capture)
    manifest, frames = load_product(cropped, expected_revision=4)
    assert len(frames) == 2
    for selected in frames:
        assert selected.gripper_opening == frame.gripper_opening
        np.testing.assert_array_equal(selected.points, frame.points[:2])
        for channel in frame.attributes:
            np.testing.assert_array_equal(selected.attributes[channel], frame.attributes[channel][:2])
            assert selected.attributes[channel].dtype == frame.attributes[channel].dtype
    assert manifest["source_manifest_sha256"] == json.loads((raw / "manifest.json").read_text())["sha256"]


@pytest.mark.parametrize("changes,match", [
    ({"calibration_sha256": "0" * 64}, "calibration"),
    ({"joint_timestamp_ns": 500}, "stale"),
    ({"stationary": False}, "moving"),
])
def test_unusable_capture_is_rejected(capture, changes, match):
    frame, crop = capture
    with pytest.raises(ValueError, match=match):
        crop.select(replace(frame, **changes), maximum_timestamp_skew_ns=10)


def test_frame_choice_and_margin(capture):
    frame, crop = capture
    expanded = replace(crop, margins=(.7, .7, .1))
    np.testing.assert_array_equal(expanded.select(frame, maximum_timestamp_skew_ns=0),
                                  [True, True, True, True, False])
    with pytest.raises(ValueError, match="geometry frame"):
        replace(crop, reference_frame="world")
    wrong_geometry = replace(frame, world_T_gripper=np.eye(4))
    assert not crop.select(wrong_geometry, maximum_timestamp_skew_ns=0).any()


def test_recrop_uses_immutable_raw_and_keeps_newly_requested_context(tmp_path, capture):
    frame, crop = capture
    raw, cropped = product(tmp_path, capture)
    hashes = {p.name: file_hash(p) for p in raw.iterdir()}
    wider = tmp_path / "wider"
    prepare_crop(raw, wider, "crop-v2", replace(crop, margins=(2., 2., 2.)), maximum_timestamp_skew_ns=0)
    _, frames = load_product(wider)
    np.testing.assert_array_equal(frames[0].points, frame.points)
    assert hashes == {p.name: file_hash(p) for p in raw.iterdir()}
    with pytest.raises(ValueError, match="retained raw"):
        prepare_crop(cropped, tmp_path / "invalid", "v3", crop, maximum_timestamp_skew_ns=0)
    with pytest.raises(FileExistsError):
        prepare_crop(raw, wider, "v2", crop, maximum_timestamp_skew_ns=0)


def test_interruption_restart_and_lost_ack_are_idempotent(tmp_path, capture):
    raw, cropped = product(tmp_path, capture)
    receiver = tmp_path / "receiver"
    assert not receive_local(cropped, receiver, byte_budget=71)
    assert (receiver / "frame-0.npz.partial").stat().st_size == 71
    assert not (receiver / "receipt.json").exists()
    with pytest.raises(FileNotFoundError):
        load_product(receiver)
    assert not receive_local(cropped, receiver, byte_budget=31)
    assert (receiver / "frame-0.npz.partial").stat().st_size == 102
    assert receive_local(cropped, receiver)
    before = {p.name: file_hash(p) for p in receiver.iterdir()}
    assert receive_local(cropped, receiver)
    assert before == {p.name: file_hash(p) for p in receiver.iterdir()}
    assert file_hash(raw / "frame-0.npz")  # ACK never deletes raw captures.
    _, frames = load_product(receiver, expected_revision=4)
    assert len(frames) == 2
    with pytest.raises(ValueError, match="stale scene"):
        load_product(receiver, expected_revision=5)


@pytest.mark.parametrize("failure", ["corrupt_partial", "truncated_source", "missing_frame", "wrong_product"])
def test_incomplete_or_corrupt_transfer_is_never_published(tmp_path, capture, failure):
    raw, cropped = product(tmp_path, capture)
    receiver = tmp_path / "receiver"
    receive_local(cropped, receiver, byte_budget=71)
    if failure == "corrupt_partial":
        (receiver / "frame-0.npz.partial").write_bytes(b"x" * 71)
    elif failure == "truncated_source":
        (cropped / "frame-0.npz").write_bytes(b"short")
    elif failure == "missing_frame":
        (cropped / "frame-1.npz").unlink()
    else:
        _, crop = capture
        other = tmp_path / "other"
        prepare_crop(raw, other, "different", crop, maximum_timestamp_skew_ns=0)
        cropped = other
    with pytest.raises((ValueError, FileNotFoundError)):
        receive_local(cropped, receiver)
    assert not (receiver / "manifest.json").exists()
    assert not (receiver / "receipt.json").exists()


def test_empty_crops_remain_explicit_and_channel_aligned(tmp_path, capture):
    frame, crop = capture
    raw = tmp_path / "raw"
    create_capture(raw, "scan", "inhand", 4, [frame])
    shifted = np.eye(4)
    shifted[:3, 3] = [100., 0., 0.]
    prepare_crop(raw, tmp_path / "empty", "empty", replace(crop, reference_T_box=shifted),
                 maximum_timestamp_skew_ns=0)
    _, frames = load_product(tmp_path / "empty")
    assert frames[0].points.shape == (0, 3)
    assert frames[0].attributes["intensity"].shape == (0,)


def test_box_membership_is_batch_independent():
    points = np.random.default_rng(2).normal(size=(123, 3))
    box_T_sensor = transform([.1, -.2, .3], [.2, -.3, .5])
    expected = np.all(np.abs(points @ box_T_sensor[:3, :3].T + box_T_sensor[:3, 3]) <= [.5, .7, .9], axis=1)
    np.testing.assert_array_equal(box_mask(points, box_T_sensor, [.5, .7, .9], batch_size=7), expected)


@pytest.mark.parametrize("field,value", [("points", [[np.nan, 0, 0]]), ("gripper_opening", np.nan),
    ("joint_positions", (0.,) * 5), ("timestamp_ns", 0), ("frame_id", "../escape")])
def test_invalid_capture_metadata(capture, field, value):
    with pytest.raises(ValueError):
        replace(capture[0], **{field: value})
