"""Execution-time known-object identification from completed hand scans."""

import json
from types import SimpleNamespace

import numpy as np

from stacking_common.pointcloud import (
    PointCloudData,
    RegistrationDecision,
    RegistrationResult,
)
from stacking_tabletop.perception.inhand_execution import identify_inhand_capture


def test_identification_crops_merges_registers_and_persists(tmp_path, monkeypatch):
    model = PointCloudData(
        np.asarray(
            [
                [x, y, z]
                for x in (-0.01, 0.0, 0.01)
                for y in (-0.01, 0.0, 0.01)
                for z in (0.08, 0.09)
            ]
        )
    )
    clutter = np.asarray([[0.4, 0.4, 0.4]])
    observation = PointCloudData(np.vstack((model.points, clutter)))
    capture = SimpleNamespace(
        load_tool_views=lambda *args, **kwargs: (
            SimpleNamespace(cloud=observation),
            SimpleNamespace(cloud=observation),
        )
    )
    monkeypatch.setattr(
        "stacking_tabletop.perception.inhand_execution.TabletopRawCapture.load",
        lambda _directory: capture,
    )
    identified = np.eye(4)
    identified[0, 3] = 0.004
    registration = RegistrationResult(
        identified,
        target_coverage=0.8,
        source_coverage=0.9,
        rmse=0.001,
        correspondences=18,
        scales=(),
    )
    decision = RegistrationDecision(True, (), 0.004, 0.0)
    registrar = SimpleNamespace(
        register_with_decision=lambda *_args: (registration, decision)
    )
    capture_profile = SimpleNamespace(
        minimum_temporal_support=2,
        temporal_tolerance=0.005,
        enable_boundary_filter=True,
        discontinuity_threshold=0.015,
        discontinuity_radius_pixels=2,
        boundary_dilation_pixels=2,
        crop_min=np.asarray([-0.1, -0.1, -0.1]),
        crop_max=np.asarray([0.1, 0.1, 0.2]),
        reference_voxel_size=0.001,
    )
    application = SimpleNamespace(
        perception_profile=SimpleNamespace(
            in_hand_capture=capture_profile,
            in_hand_registration=SimpleNamespace(minimum_points=3),
            reconstruction=SimpleNamespace(minimum_view_support=2),
        ),
        build_pcd_catalog=lambda: SimpleNamespace(load=lambda _object_id: model),
        build_in_hand_registrar=lambda: registrar,
    )

    result = identify_inhand_capture(application, tmp_path, 7, np.eye(4))

    assert result.accepted and result.object_id == 7
    np.testing.assert_allclose(result.identified_tool_T_object, identified)
    assert result.observation_points == len(model)
    assert (tmp_path / "inhand_observation_tool0.pcd").is_file()
    report = json.loads((tmp_path / "inhand_identification.json").read_text())
    assert report["status"] == "accepted"
    assert report["merge"]["view_count"] == 2
    assert report["observation_points"] == len(model)
