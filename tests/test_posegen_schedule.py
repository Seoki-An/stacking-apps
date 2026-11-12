"""Current platform defaults and historical native schedule compatibility."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import stacking_core as core

from stacking_common.actions.core_solver import CorePlacementPoseSolver
from stacking_common.actions.cuda_solver import CudaPlacementPoseSolver
from stacking_common.profiles.posegen import load_posegen_profile
from stacking_common.profiles.native import load_native_solver_profile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "tests/fixtures/legacy_planner/posegen_schedule.json").read_text())


@pytest.mark.parametrize("platform", ["excavator", "tabletop"])
def test_native_schedule_matches_legacy_stages(platform):
    profile = load_posegen_profile(ROOT / f"apps/{platform}/config/solvers/posegen.yml")
    # Reconstruct the historical profile explicitly; current defaults have
    # intentionally changed since the legacy outputs were recorded.
    weights = dict(profile.schedule["objective_weights"])
    if platform == "excavator":
        weights.update(k_wrench={"ramp": 1.0},
                       k_gap_c={"constant": 40.0, "ramp": 150.0, "gap_feedback": 1e5},
                       k_gap={"sine": 10.0}, k_xy={"anti_ramp": 1.0},
                       k_lower={"anti_ramp": 5.0})
    else:
        weights.update(k_wrench={"ramp": 5.0}, k_lower={"anti_ramp": 0.0})
    objective = {k: v for k, v in profile.objective.items()
                 if k not in ("k_robust", "robust_load")}
    profile = replace(profile, objective=objective,
                      schedule={**profile.schedule, "objective_weights": weights})
    solver = SimpleNamespace(profile=profile, core=core)
    assert profile.native_config(core).objective.k_robust == 0.
    for stage in REFERENCE[platform]["stages"]:
        config = CorePlacementPoseSolver._scheduled_config(
            solver, stage["time"], stage["c_gap"], final=stage["time"] == 1.)
        for name, value in stage["objective"].items():
            np.testing.assert_allclose(getattr(config.objective, name), value, atol=1e-12)
        assert config.trust_region.max_iters == stage["trust_region"]["max_iter"]
        assert config.trust_region.tol == pytest.approx(stage["trust_region"]["tol_eps"])
        assert config.force_solver.tol_abs == .001
        assert config.force_solver.tol_rel == .001


@pytest.mark.parametrize("platform", ["excavator", "tabletop"])
@pytest.mark.parametrize("solver_type", [CorePlacementPoseSolver, CudaPlacementPoseSolver])
def test_current_platform_wrench_and_robustness_defaults_survive_schedule(platform, solver_type):
    profile = load_posegen_profile(ROOT / f"apps/{platform}/config/solvers/posegen.yml")
    solver = SimpleNamespace(profile=profile, core=core)
    for stage in REFERENCE["excavator"]["stages"]:
        config = solver_type._scheduled_config(
            solver, stage["time"], stage["c_gap"], final=stage["time"] == 1.)
        np.testing.assert_allclose(config.objective.k_wrench,
                                   np.asarray(stage["objective"]["k_wrench"]), atol=1e-12)
        assert config.objective.k_robust == 0.
        assert config.objective.robust_load == pytest.approx(.2)


def test_old_profile_without_weights_preserves_previous_schedule():
    profile = load_posegen_profile(ROOT / "apps/tabletop/config/solvers/posegen.yml")
    schedule = dict(profile.schedule)
    schedule.pop("objective_weights")
    restored = replace(profile, schedule=schedule)
    solver = SimpleNamespace(profile=restored, core=core)
    for stage in REFERENCE["tabletop"]["stages"]:
        config = CorePlacementPoseSolver._scheduled_config(
            solver, stage["time"], stage["c_gap"], final=stage["time"] == 1.)
        for name, value in stage["objective"].items():
            np.testing.assert_allclose(getattr(config.objective, name), value, atol=1e-12)


@pytest.mark.parametrize("solver_type", [CorePlacementPoseSolver, CudaPlacementPoseSolver])
def test_excavator_rho_continuation_is_stage_fixed_and_gap_independent(solver_type):
    profile = load_posegen_profile(ROOT / "apps/excavator/config/solvers/posegen.yml")
    solver = SimpleNamespace(profile=profile, core=core)
    configs = [solver_type._scheduled_config(solver, t, gap, final=t == 1.)
               for gap in (0., .1) for t in profile.schedule["times"]]
    assert len(configs) == 8
    # Inspect after every stage has been built: later stages must not mutate
    # configurations already passed to native force/objective/gradient solves.
    assert [c.objective.rho for c in configs] == pytest.approx(
        [.01, .00875, .00625, .005] * 2)
    assert profile.objective["rho"] == .01
    assert [c.force_solver.max_iters for c in configs] == [1000] * 8


def test_saved_profile_without_rho_schedule_keeps_constant_rho():
    profile = load_posegen_profile(ROOT / "apps/excavator/config/solvers/posegen.yml")
    weights = {k: v for k, v in profile.schedule["objective_weights"].items() if k != "rho"}
    restored = replace(profile, schedule={**profile.schedule, "objective_weights": weights})
    assert "rho" not in restored.schedule["objective_weights"]
    solver = SimpleNamespace(profile=restored, core=core)
    for t in restored.schedule["times"]:
        assert CorePlacementPoseSolver._scheduled_config(
            solver, t, .1, final=t == 1.).objective.rho == .01


@pytest.mark.parametrize("rho", [
    {"constant": 0.}, {"ramp": .01}, {"anti_ramp": .01},
    {"constant": .005, "gap_feedback": .005},
    {"constant": .005, "sine": .005},
])
def test_rho_schedule_requires_positive_stage_only_values(rho):
    profile = load_posegen_profile(ROOT / "apps/excavator/config/solvers/posegen.yml")
    weights = {**profile.schedule["objective_weights"], "rho": rho}
    with pytest.raises(ValueError, match="schedule rho"):
        replace(profile, schedule={**profile.schedule, "objective_weights": weights})


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
