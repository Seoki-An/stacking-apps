import numpy as np
import pytest

from stacking_common.pointcloud import (
    CapturedPointCloud,
    MultiViewVoxelMerge,
    MultiscaleRegistrationConfig,
    PointCloudData,
    RegistrationResult,
    VoxelMergeConfig,
    evaluate_registration,
)
from stacking_common.pointcloud import RegistrationQualityConfig


def test_point_cloud_data_owns_immutable_finite_points() -> None:
    values = np.array([[1.0, 2.0, 3.0]])
    cloud = PointCloudData(values)
    values[0] = 0.0

    assert np.allclose(cloud.points, [[1.0, 2.0, 3.0]])
    assert cloud.points.flags.writeable is False
    with pytest.raises(ValueError, match="finite"):
        PointCloudData([[0.0, np.nan, 0.0]])


def test_point_cloud_transform_is_rigid_and_non_mutating() -> None:
    cloud = PointCloudData([[1.0, 0.0, 0.0]])
    transform = np.eye(4)
    transform[:3, 3] = [0.1, 0.2, 0.3]

    transformed = cloud.transformed(transform)

    assert np.allclose(transformed.points, [[1.1, 0.2, 0.3]])
    assert np.allclose(cloud.points, [[1.0, 0.0, 0.0]])
    invalid = np.eye(4)
    invalid[0, 0] = 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        cloud.transformed(invalid)


def test_multiscale_config_requires_explicit_decreasing_resolutions() -> None:
    with pytest.raises(ValueError, match="decrease"):
        MultiscaleRegistrationConfig(
            voxel_sizes=(0.002, 0.004),
            max_iterations=(20, 10),
            correspondence_distance_scale=2.5,
            evaluation_distance=0.005,
            minimum_points=10,
        )


def test_registration_result_owns_validated_transform() -> None:
    transform = np.eye(4)
    result = RegistrationResult(
        transform=transform,
        target_coverage=1.0,
        source_coverage=1.0,
        rmse=0.0,
        correspondences=10,
        scales=(),
    )
    transform[0, 3] = 1.0

    assert np.allclose(result.transform, np.eye(4))
    assert result.transform.flags.writeable is False


def test_registration_quality_is_workflow_specific() -> None:
    transform = np.eye(4)
    transform[0, 3] = 0.03
    result = RegistrationResult(
        transform=transform,
        target_coverage=0.8,
        source_coverage=0.7,
        rmse=0.002,
        correspondences=100,
        scales=(),
    )
    field = RegistrationQualityConfig(0.35, 0.25, 0.0075, 0.05, 20.0)
    in_hand = RegistrationQualityConfig(0.35, 0.25, 0.005, 0.02, 10.0)

    assert evaluate_registration(result, np.eye(4), field).accepted is True
    decision = evaluate_registration(result, np.eye(4), in_hand)
    assert decision.accepted is False
    assert decision.reasons == ("large_translation_correction",)


def test_captured_cloud_requires_frame_sensor_and_timestamp() -> None:
    cloud = PointCloudData([[0.0, 0.0, 1.0]])
    capture = CapturedPointCloud(cloud, "camera", 10, "SERIAL")

    assert capture.cloud is cloud
    with pytest.raises(ValueError, match="timestamp"):
        CapturedPointCloud(cloud, "camera", 0, "SERIAL")


def test_multi_view_merge_counts_distinct_views_not_raw_samples() -> None:
    merger = MultiViewVoxelMerge(VoxelMergeConfig(0.01, 2))
    progress = []
    first = PointCloudData(
        [[0.001, 0.0, 0.0], [0.002, 0.0, 0.0], [0.03, 0.0, 0.0]]
    )
    second = PointCloudData([[0.003, 0.0, 0.0], [0.06, 0.0, 0.0]])

    result = merger.merge(
        (first, second),
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert np.allclose(result.cloud.points, [[0.00225, 0.0, 0.0]])
    assert result.stats.view_count == 2
    assert result.stats.input_point_count == 5
    assert result.stats.supported_voxel_count == 1
    assert result.stats.rejected_voxel_count == 2
    assert progress == [(1, 2), (2, 2)]


def test_multi_view_merge_rejects_impossible_support() -> None:
    merger = MultiViewVoxelMerge(VoxelMergeConfig(0.01, 2))
    with pytest.raises(ValueError, match="exceeds"):
        merger.merge((PointCloudData([[0.0, 0.0, 0.0]]),))
