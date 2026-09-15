import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stacking_common.pointcloud import CapturedPointCloud, PointCloudData
from stacking_tabletop import TabletopApplication
from stacking_tabletop.perception import (
    ORIENTATION_BANK,
    TabletopCaptureBuffer,
    TabletopFieldSegmenter,
    load_femto_depth_calibration,
    load_recorded_femto_scan,
)
from stacking_tabletop.ros import RosPointCloudAdapter, RosPointCloudSubscription


ROOT = Path(__file__).resolve().parents[1]


def test_orientation_bank_has_24_unique_proper_rotations() -> None:
    rotations = np.asarray(ORIENTATION_BANK)

    assert rotations.shape == (24, 3, 3)
    assert len({tuple(np.round(value, 12).flat) for value in rotations}) == 24
    assert np.allclose(
        rotations.transpose(0, 2, 1) @ rotations,
        np.broadcast_to(np.eye(3), rotations.shape),
        atol=1e-12,
    )
    assert np.allclose(np.linalg.det(rotations), 1.0, atol=1e-12)


def test_depth_calibration_loads_recorded_artifact_and_preserves_rays(
    tmp_path: Path,
) -> None:
    path = tmp_path / "depth.npz"
    np.savez(
        path,
        method="femto_charuco_plane_robust_offset_v2",
        depth_scale=1.0,
        depth_offset_m=-0.1,
        depth_frame="camera_depth_optical_frame",
        depth_mode="WFOV_UNBINNED_1024x1024_15",
        camera_serial="PRIMARY",
    )
    calibration = load_femto_depth_calibration(
        path,
        expected_camera_serial="PRIMARY",
        expected_depth_mode="WFOV_UNBINNED_1024x1024_15",
    )

    corrected = calibration.correct(PointCloudData([[1.0, 0.0, 2.0]]))

    assert np.allclose(corrected.points, [[0.95, 0.0, 1.9]])


def test_capture_buffer_applies_warmup_period_and_count() -> None:
    application = TabletopApplication.from_repository(ROOT)
    profile = replace(
        application.perception_profile.capture,
        frame_count=2,
        warmup_frames=1,
        period=0.2,
    )
    camera = application.perception_profile.primary_camera
    buffer = TabletopCaptureBuffer(camera, profile)
    cloud = PointCloudData([[0.0, 0.0, 1.0]])

    assert not buffer.accept(CapturedPointCloud(cloud, "camera", 1, camera.serial))
    assert buffer.accept(
        CapturedPointCloud(cloud, "camera", 1_000_000_000, camera.serial)
    )
    assert not buffer.accept(
        CapturedPointCloud(cloud, "camera", 1_100_000_000, camera.serial)
    )
    assert buffer.accept(
        CapturedPointCloud(cloud, "camera", 1_300_000_000, camera.serial)
    )
    assert buffer.complete


def test_ros_adapter_is_only_a_message_converter() -> None:
    camera = TabletopApplication.from_repository(
        ROOT
    ).perception_profile.primary_camera
    message = SimpleNamespace(
        header=SimpleNamespace(
            frame_id="camera_depth_optical_frame",
            stamp=SimpleNamespace(sec=3, nanosec=5),
        )
    )
    calls = []

    def reader(value, **kwargs):
        calls.append((value, kwargs))
        return np.array([[1.0, 2.0, 3.0]])

    captured = RosPointCloudAdapter(camera).convert(message, reader=reader)

    assert np.allclose(captured.cloud.points, [[1.0, 2.0, 3.0]])
    assert captured.timestamp_ns == 3_000_000_005
    assert captured.sensor_id == camera.serial
    assert calls[0][1]["field_names"] == ("x", "y", "z")


def test_ros_subscription_wraps_an_existing_node() -> None:
    camera = TabletopApplication.from_repository(
        ROOT
    ).perception_profile.primary_camera
    created = []
    received = []

    class Node:
        def create_subscription(self, message_type, topic, callback, qos):
            created.append((message_type, topic, callback, qos))
            return "subscription"

    reader = lambda message, **kwargs: np.array([[0.0, 0.0, 1.0]])
    wrapper = RosPointCloudSubscription(
        Node(),
        camera,
        received.append,
        message_type=object,
        reader=reader,
    )
    message = SimpleNamespace(
        header=SimpleNamespace(
            frame_id="camera",
            stamp=SimpleNamespace(sec=1, nanosec=2),
        )
    )
    created[0][2](message)

    assert wrapper.subscription == "subscription"
    assert created[0][1] == camera.pointcloud_topic
    assert received[0].sensor_id == camera.serial


def test_field_segmentation_fits_table_and_separates_clusters() -> None:
    pytest.importorskip("open3d")
    application = TabletopApplication.from_repository(ROOT)
    profile = replace(
        application.perception_profile.segmentation,
        workspace_min=np.array([-0.3, -0.3, -0.02]),
        workspace_max=np.array([0.3, 0.3, 0.2]),
        designated_table_height=0.10,
        minimum_cluster_points=20,
    )
    rng = np.random.default_rng(2)
    table_xy = rng.uniform(-0.25, 0.25, size=(5000, 2))
    table = np.column_stack(
        [table_xy, rng.normal(0.08, 0.0005, size=len(table_xy))]
    )
    first = rng.normal([0.10, 0.10, 0.115], 0.006, size=(500, 3))
    second = rng.normal([-0.12, -0.08, 0.125], 0.007, size=(600, 3))

    result = TabletopFieldSegmenter(profile).segment(
        PointCloudData(np.vstack([table, first, second]))
    )

    assert abs(result.measured_table_height - 0.08) < 0.002
    assert abs(result.height_correction - 0.02) < 0.002
    assert len(result.clusters) == 2
    assert np.allclose(result.support_plane, [0.0, 0.0, 1.0, -0.10])


def test_field_matcher_assigns_real_catalog_models_one_to_one() -> None:
    pytest.importorskip("open3d")
    application = TabletopApplication.from_repository(ROOT)
    catalog = application.build_pcd_catalog()
    table_plane = np.array([0.0, 0.0, 1.0, -0.10])
    clusters = []
    for object_id, translation in ((1, [0.10, 0.0, 0.0]), (2, [-0.10, 0.0, 0.0])):
        model = catalog.load(object_id)
        initial = np.eye(4)
        initial[:3, 3] = translation
        points = model.transformed(initial).points
        bottom = np.quantile(points[:, 2], 0.01)
        points = points + np.array([0.0, 0.0, 0.10 - bottom])
        clusters.append(PointCloudData(points))

    result = application.build_field_matcher().assign(
        (1, 2),
        tuple(clusters),
        table_plane,
        rotations=(np.eye(3),),
    )

    assert result.assignments == {0: 1, 1: 2}


def test_field_icp_recovers_partial_surface_without_snapping_solved_pose() -> None:
    pytest.importorskip('open3d')
    from scipy.spatial.transform import Rotation
    from stacking_tabletop.perception.assignment import TabletopFieldMatcher
    from stacking_tabletop.perception.registration import Open3dRegistrationBackend

    profile = TabletopApplication.from_repository(ROOT).perception_profile
    rng = np.random.default_rng(81)
    points = rng.normal(size=(5000, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    points *= [.038, .025, .020]
    points[:, 2] += .003 * np.sin(points[:, 0] * 90) * np.cos(points[:, 1] * 75)
    model = PointCloudData(points)
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_euler('xyz', [12, -8, 33], degrees=True).as_matrix()
    expected[:3, 3] = [.12, -.62, .128]
    # Only the upper surface is visible; hidden points must not pull the fit.
    observation = PointCloudData(points[points[:, 2] > .006]).transformed(expected)
    initial = expected.copy()
    initial[:3, 3] += [.001, -.001, .001]
    matcher = TabletopFieldMatcher(
        SimpleNamespace(load=lambda _: model),
        Open3dRegistrationBackend(profile.field_registration),
        profile.field_quality, profile.assignment,
    )
    result = matcher.register_best(
        1, observation, np.array([0., 0., 1., -.1]), initial_transform=initial,
    )
    assert result.decision.accepted
    np.testing.assert_allclose(result.registration.transform[:3, 3], expected[:3, 3], atol=.0001)
    error_rotation = result.registration.transform[:3, :3].T @ expected[:3, :3]
    assert Rotation.from_matrix(error_rotation).magnitude() < np.deg2rad(.2)
    # A reference plane offset must be scored, not imposed on the ICP solution.
    assert result.bottom_height > .005


def test_recorded_dual_femto_fusion_loads_and_aligns(tmp_path: Path) -> None:
    o3d = pytest.importorskip("open3d")
    application = TabletopApplication.from_repository(ROOT)
    rng = np.random.default_rng(12)
    table_xy = rng.uniform([-0.25, -0.25], [0.25, 0.25], (4000, 2))
    table = np.column_stack([table_xy, np.zeros(len(table_xy))])
    objects = rng.normal([0.04, -0.03, 0.05], [0.03, 0.02, 0.012], (1200, 3))
    primary_points = np.vstack([table, objects])
    correction = np.eye(4)
    correction[:3, 3] = [0.004, -0.003, 0.0]
    inverse = np.linalg.inv(correction)
    secondary_points = PointCloudData(primary_points).transformed(inverse).points

    directories = []
    for index, (serial, points) in enumerate(
        (("PRIMARY", primary_points), ("SECONDARY", secondary_points))
    ):
        directory = tmp_path / f"scan_{index}"
        directory.mkdir()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.array(points, copy=True))
        o3d.io.write_point_cloud(
            str(directory / "field_fused_base_link.pcd"), cloud
        )
        (directory / "field_scan.json").write_text(
            json.dumps(
                {
                    "output_frame": "base_link",
                    "camera_serial": serial,
                    "fused_depth_corrected": True,
                    "fused_file": "field_fused_base_link.pcd",
                }
            )
        )
        directories.append(directory)

    recorded = load_recorded_femto_scan(directories[0])
    result = application.build_dual_femto_fusion().fuse_recorded(*directories)

    assert recorded.camera_serial == "PRIMARY"
    translation_error = np.linalg.norm(
        result.secondary_to_primary[:3, 3] - correction[:3, 3]
    )
    assert translation_error < 0.002
    assert result.registration.rmse < 0.003
    assert len(result.cloud) > 1000
