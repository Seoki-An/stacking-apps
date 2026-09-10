import hashlib
import json
from pathlib import Path

import numpy as np

from stacking_common.reconstruction import estimate_hull_inertial, load_point_cloud
from stacking_tabletop.environment.assets import _parse_object_asset


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/legacy_reconstruction/model_1_hull.json"


def test_migrated_model_matches_recorded_legacy_hull_properties() -> None:
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    asset = ROOT / expected["source"]["asset"]
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == expected["source"]["sha256"]

    points = load_point_cloud(asset).points
    inertial = estimate_hull_inertial(points)

    assert len(points) == expected["point_count"]
    assert np.allclose(points.min(axis=0), expected["bounds_min"], atol=1e-12)
    assert np.allclose(points.max(axis=0), expected["bounds_max"], atol=1e-12)
    assert np.allclose(points.mean(axis=0), expected["centroid"], atol=1e-12)
    assert np.isclose(inertial.mass, expected["unit_density_mass"], rtol=1e-12)
    assert np.allclose(inertial.center_of_mass, expected["center_of_mass"], atol=1e-12)
    assert np.allclose(inertial.inertia, expected["inertia"], rtol=1e-12, atol=1e-18)


def test_migrated_dsf_asset_preserves_recorded_legacy_commissioning() -> None:
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))["legacy_dsf_asset"]
    asset = ROOT / expected["asset"]
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == expected["sha256"]

    value = _parse_object_asset(asset)

    assert np.isclose(value.mass, expected["unit_density_mass"], rtol=1e-12)
    assert len(value.components) == expected["component_count"]
    assert [
        len(component.geometry.points) for component in value.components
    ] == expected["node_counts"]
    assert [component.geometry.sharpness for component in value.components] == expected["sharpness"]
