"""Legacy LiDAR parity and an end-to-end desktop capture exercise."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from stacking_common.pointcloud.raycast import simulate_lidar
from stacking_common.pointcloud.types import PointCloudData
from stacking_common.pointcloud.registration import (
    MultiscaleRegistrationConfig, RegistrationQualityConfig, evaluate_registration,
)
from stacking_common.pointcloud.open3d_registration import Open3dRegistrationBackend
from scipy.spatial.transform import Rotation
from stacking_excavator.acquisition.capture import CaptureFrame
from stacking_excavator.acquisition.emulation import demonstration, run_demonstration
from stacking_excavator.acquisition.storage import file_hash, load_product

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/legacy_planner/lidar.npz"
META = json.loads(FIXTURE.with_suffix(".json").read_text())


def test_reference_provenance():
    assert file_hash(FIXTURE) == META["fixture_sha256"]
    for name, expected in META["asset_hashes"].items():
        assert file_hash(ROOT / name) == expected


@pytest.mark.parametrize("kind", ["field", "inhand", "scene"])
def test_real_mesh_rays_and_sensor_coordinates_match_legacy(kind):
    pytest.importorskip("open3d")
    meshes, sensors, gripper, joints, crop = demonstration(ROOT, kind)
    reference = META["cases"][kind]
    np.testing.assert_allclose(sensors, reference["sensors"], atol=1e-12)
    np.testing.assert_allclose(gripper, reference["world_T_gripper"], atol=1e-12)
    retained = []
    with np.load(FIXTURE, allow_pickle=False) as expected:
        for index, sensor in enumerate(sensors):
            points, rays = simulate_lidar(meshes, sensor, fov=META["fov"])
            # Float32 ray intersections may vary slightly across Open3D builds.
            np.testing.assert_allclose(points, expected[f"{kind}_{index}"], atol=2e-6, rtol=0)
            assert len(points) > 500
            assert rays.dtype == np.uint32
            assert np.all(np.diff(rays.astype(np.int64)) > 0)
            frame = CaptureFrame(f"f{index}", "lidar", 1, 1, sensor, gripper,
                crop.calibration_sha256, points, joints, -.45)
            selected = crop.select(frame, maximum_timestamp_skew_ns=0)
            # Independent reference calculation in the requested geometry frame.
            world = points @ sensor[:3, :3].T + sensor[:3, 3]
            world_T_box = (gripper if kind == "inhand" else np.eye(4)) @ crop.reference_T_box
            local = (world - world_T_box[:3, 3]) @ world_T_box[:3, :3]
            expected_mask = np.all(np.abs(local) <= np.array(crop.half_extents) + crop.margins, axis=1)
            np.testing.assert_array_equal(selected, expected_mask)
            retained.append(selected.sum())
    assert min(retained) > 100


def test_stone_grab_opening_changes_occlusion():
    pytest.importorskip("open3d")
    closed, sensors, *_ = demonstration(ROOT, "inhand", opening=0.)
    opened, *_ = demonstration(ROOT, "inhand", opening=-.8)
    points_closed, ids_closed = simulate_lidar(closed, sensors[0], fov=(40., 30.))
    points_open, ids_open = simulate_lidar(opened, sensors[0], fov=(40., 30.))
    _, closed_index, open_index = np.intersect1d(ids_closed, ids_open, return_indices=True)
    distances = np.linalg.norm(points_closed[closed_index] - points_open[open_index], axis=1)
    assert np.count_nonzero(distances > .01) > 20


def test_desktop_registration_recovers_known_cloud_displacement():
    pytest.importorskip("open3d")
    # Isolate numeric registration from pose identification: same visible cloud,
    # with a known imposed displacement, not a claim of full PoseID acceptance.
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        observed = fixture["field_0"]
    sensor = np.array(META["cases"]["field"]["sensors"][0])
    world = observed @ sensor[:3, :3].T + sensor[:3, 3]
    # Ground removal is essential: a large plane can hide a yaw error in ICP.
    observed = observed[(world[:, 2] > .1) & (np.abs(world[:, 1]) < .8)]
    source = PointCloudData(observed)
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_euler("xyz", [.008, -.012, .005]).as_matrix()
    expected[:3, 3] = [.018, -.012, .009]
    target = source.transformed(expected)
    backend = Open3dRegistrationBackend(MultiscaleRegistrationConfig(
        (.04, .015, .005), (60, 40, 30), 3., .01, 100))
    result = backend.register(source, target, np.eye(4))
    np.testing.assert_allclose(result.transform, expected, atol=5e-4)
    quality = RegistrationQualityConfig(.95, .95, .001, .1, 5.)
    assert evaluate_registration(result, np.eye(4), quality).accepted
    assert result.rmse < .001


def test_desktop_only_end_to_end_artifacts(tmp_path):
    o3d = pytest.importorskip("open3d")
    output = tmp_path / "emulation"
    report = run_demonstration(ROOT, output)
    assert not report["commissioned_calibration"]
    for kind, values in report["purposes"].items():
        assert values["interrupted_then_resumed"]
        assert 100 < values["cropped_points"] < values["raw_points"]
        assert 0 < values["byte_fraction"] < .5
        manifest, frames = load_product(output / kind / "desktop", expected_revision=1)
        assert len(frames) == 2
        assert all(f.timestamp_source == "emulated" for f in frames)
        assert manifest["sha256"] == values["manifest_sha256"]
        world = np.concatenate([f.points @ f.world_T_sensor[:3, :3].T + f.world_T_sensor[:3, 3]
                                for f in frames])
        exported = o3d.io.read_point_cloud(str(output / kind / "crop-world.ply"))
        np.testing.assert_allclose(np.asarray(exported.points), world, atol=1e-12)
        assert (output / kind / "raw/manifest.json").exists()
    with pytest.raises(FileExistsError):
        run_demonstration(ROOT, output)


def test_emulator_has_no_ros_gui_or_legacy_imports():
    roots = set()
    files = list((ROOT / "apps/excavator/src/stacking_excavator/acquisition").glob("*.py"))
    files.append(ROOT / "packages/stacking-common/src/stacking_common/pointcloud/raycast.py")
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
    assert not roots.intersection({"rclpy", "rospy", "sensor_msgs", "pyvista", "PySide6",
                                   "diffsimpy", "inrol_urdf_parser", "model"})
