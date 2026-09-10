"""Native configurations checked against recorded legacy Python schedule output."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import stacking_core as core

from stacking_common.actions.core_solver import CorePlacementPoseSolver
from stacking_common.profiles.posegen import load_posegen_profile
from stacking_common.profiles.native import load_native_solver_profile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "tests/fixtures/legacy_planner/posegen_schedule.json").read_text())


@pytest.mark.parametrize("platform", ["excavator", "tabletop"])
def test_native_schedule_matches_legacy_stages(platform):
    profile = load_posegen_profile(ROOT / f"apps/{platform}/config/solvers/posegen.yml")
    solver = SimpleNamespace(profile=profile, core=core)
    for stage in REFERENCE[platform]["stages"]:
        config = CorePlacementPoseSolver._scheduled_config(
            solver, stage["time"], stage["c_gap"], final=stage["time"] == 1.)
        for name, value in stage["objective"].items():
            np.testing.assert_allclose(getattr(config.objective, name), value, atol=1e-12)
        assert config.trust_region.max_iters == stage["trust_region"]["max_iter"]
        assert config.trust_region.tol == pytest.approx(stage["trust_region"]["tol_eps"])
        assert config.force_solver.tol_abs == .001
        assert config.force_solver.tol_rel == .0001


def test_old_profile_without_weights_preserves_previous_schedule():
    profile = load_posegen_profile(ROOT / "apps/tabletop/config/solvers/posegen.yml")
    schedule = dict(profile.schedule)
    schedule.pop("objective_weights")
    restored = replace(profile, schedule=schedule)
    assert restored.schedule == profile.schedule


@pytest.mark.parametrize("invalid", [
    {}, {"k_comp": {"unknown": 1}}, {"k_comp": {"ramp": -1}},
    {"k_comp": {"ramp": float("nan")}}, {"k_comp": {"ramp": float("inf")}},
])
def test_schedule_rejects_incomplete_or_invalid_coefficients(invalid):
    profile = load_posegen_profile(ROOT / "apps/excavator/config/solvers/posegen.yml")
    weights = {**profile.schedule["objective_weights"], **invalid} if invalid else {}
    with pytest.raises(ValueError, match="schedule"):
        replace(profile, schedule={**profile.schedule, "objective_weights": weights})


def test_excavator_native_grasp_retreat_matches_legacy_attachment():
    directory = ROOT / "apps/excavator/config/solvers"
    config = load_native_solver_profile(directory / "grasp.yml", directory / "motion.yml").native_config(core)
    # planning.py: gripper.height=1.5, surface_grasp_offset_scale=0.5;
    # diffsim grasp_initializer_t::sampleParameters multiplies these in metres.
    assert config.direct.grasp_sampling.retreat_distance == pytest.approx(1.5 * .5)
    assert config.direct.approach_distance == 1.5
