"""Reject invalid solver settings before native sampling can mask the error."""

from dataclasses import replace
from pathlib import Path

import pytest
import stacking_core as core

from stacking_common.profiles.native import load_native_solver_profile


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("platform", ["excavator", "tabletop"])
def test_platform_grasp_radii_resolve_to_valid_native_configuration(platform):
    directory = ROOT / "apps" / platform / "config/solvers"
    profile = load_native_solver_profile(directory / "grasp.yml", directory / "motion.yml")
    trust_region = profile.native_config(core).direct.grasp_generation.trust_region
    assert 0 < trust_region.radius_init <= trust_region.radius_max


def test_reject_recorded_invalid_excavator_grasp_radii():
    directory = ROOT / "apps/excavator/config/solvers"
    profile = load_native_solver_profile(directory / "grasp.yml", directory / "motion.yml")
    grasp = {**profile.grasp, "trust_region": {
        **profile.grasp["trust_region"], "radius_init": 1.0, "radius_max": 0.1,
    }}
    with pytest.raises(ValueError, match="radius_init=1.0, radius_max=0.1"):
        replace(profile, grasp=grasp)
