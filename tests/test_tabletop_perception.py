from pathlib import Path

import numpy as np
import pytest

from stacking_common.pointcloud import PointCloudData
from stacking_tabletop import TabletopApplication
from stacking_tabletop.config import load_perception_profile
from stacking_tabletop.perception import Open3dRegistrationBackend


ROOT = Path(__file__).resolve().parents[1]


def rotation_z(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def transform(angle: float, translation) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = rotation_z(angle)
    value[:3, 3] = translation
    return value


def test_tabletop_pcd_catalog_resolves_only_configured_objects() -> None:
    application = TabletopApplication.from_repository(ROOT)
    catalog = application.build_pcd_catalog()

    assert catalog.object_ids == tuple(range(1, 21))
    assert catalog.path(1).name == "model_1.pcd"
    with pytest.raises(KeyError, match="not in the PCD catalog"):
        catalog.path(999)


def test_field_and_in_hand_registration_have_independent_scales() -> None:
    profile = load_perception_profile(
        ROOT / "apps" / "tabletop" / "config" / "perception.yml"
    )

    assert profile.field_registration.voxel_sizes == (
        0.006,
        0.003,
        0.0015,
    )
    assert profile.in_hand_registration.voxel_sizes == (
        0.006,
        0.003,
        0.002,
    )
    assert (
        profile.field_registration.evaluation_distance
        != profile.in_hand_registration.evaluation_distance
    )


def test_complete_tabletop_pcd_catalog_loads_finite_models() -> None:
    pytest.importorskip("open3d")
    catalog = TabletopApplication.from_repository(ROOT).build_pcd_catalog()

    models = catalog.load_many(catalog.object_ids)

    assert tuple(models) == catalog.object_ids
    assert all(len(model) > 100 for model in models.values())
    assert all(np.all(np.isfinite(model.points)) for model in models.values())


def test_multiscale_registration_recovers_synthetic_rigid_transform() -> None:
    pytest.importorskip("open3d")
    profile = load_perception_profile(
        ROOT / "apps" / "tabletop" / "config" / "perception.yml"
    )
    rng = np.random.default_rng(23)
    source = PointCloudData(
        rng.uniform([-0.04, -0.02, -0.01], [0.05, 0.03, 0.04], (1200, 3))
    )
    expected = transform(np.deg2rad(4.0), [0.012, -0.006, 0.004])
    target = source.transformed(expected)
    initial = transform(np.deg2rad(2.0), [0.009, -0.004, 0.003])

    result = Open3dRegistrationBackend(
        profile.field_registration
    ).register(source, target, initial)

    assert np.linalg.norm(result.transform[:3, 3] - expected[:3, 3]) < 5e-4
    assert np.linalg.norm(
        result.transform[:3, :3] - expected[:3, :3]
    ) < 0.01
    assert result.target_coverage > 0.99
    assert result.source_coverage > 0.99
    assert result.rmse < 5e-4
    assert len(result.scales) == 3


def test_application_registrar_uses_catalog_model() -> None:
    pytest.importorskip("open3d")
    application = TabletopApplication.from_repository(ROOT)
    catalog = application.build_pcd_catalog()
    model = catalog.load(1)
    expected = transform(np.deg2rad(2.0), [0.01, -0.005, 0.003])
    observation = model.transformed(expected)

    result = application.build_field_registrar().register(
        1,
        observation,
        expected,
    )

    assert np.linalg.norm(result.transform[:3, 3] - expected[:3, 3]) < 1e-4
    assert np.linalg.norm(
        result.transform[:3, :3] - expected[:3, :3]
    ) < 0.002
    assert result.target_coverage > 0.99
    assert result.source_coverage > 0.99
    assert result.rmse < 5e-4
