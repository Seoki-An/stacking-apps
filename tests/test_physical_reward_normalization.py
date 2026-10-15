"""Physical reward units across object sizes, densities, and platform profiles."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stacking_common.actions.core_solver import CorePlacementPoseSolver
from stacking_common.actions.cuda_solver import CudaPlacementPoseSolver
from stacking_common.domain import Action, State
from stacking_common.environment import RewardConfig, RewardWeights, StackingReward
from stacking_common.profiles import load_environment_profile
from stacking_common.search import ActionCandidate, PlacementPriorScorer


ROOT = Path(__file__).resolve().parents[1]
POSE = np.array([0., 0., 0., 0., 0., 0., 1.])
KEY = "posegen_equilibrium_residual_normalized"
WEIGHTS = RewardWeights(2., 0., 0., 2., 0., 0.)


class Inventory(list):
    @property
    def objects(self):
        return self

    @property
    def object_ids(self):
        return tuple(obj.object_id for obj in self)

    def initial_pose(self, _):
        return POSE.copy()


def inventory(scale=1., density=1.):
    objects = Inventory([
        NS(object_id=10, mass=density * scale**3, volume=scale**3,
           representative_face_normals=lambda: ([], [])),
        NS(object_id=90, mass=8 * density * scale**3, volume=8 * scale**3,
           representative_face_normals=lambda: ([], [])),
    ])
    objects.target = NS(origin=np.zeros(3), height=scale, volume=10 * scale**3)
    return objects


def placement_values(objects, wrenches, solver_type=CorePlacementPoseSolver, gravity=9.81):
    solver = solver_type.__new__(solver_type)
    solver.inventory = objects
    solver.gravity = gravity
    result = NS(
        solver=NS(iters=1, objective_evals=1, grad_norm=0., converged=True,
                  force_solver=NS(converged=True), scene_graph_rebuilds=1,
                  scene_graph_reuses=0),
        c_comp=0., contact_point={}, contact_normal={}, contact_force={},
        net_wrench=wrenches, c_feq=.4, c_gap=0., optimal_pose=POSE,
    )
    return solver._placement_values(result)


@pytest.mark.parametrize("solver_type", [CorePlacementPoseSolver, CudaPlacementPoseSolver])
@pytest.mark.parametrize("scale,density,gravity", [(.1, 2., 9.81), (1., 1., 9.81), (10., 5., 3.7)])
def test_equilibrium_is_invariant_to_scale_density_and_world_rotation(solver_type, scale, density, gravity):
    objects = inventory(scale, density)
    rotation = Rotation.from_euler("xyz", [23., 47., 11.], degrees=True).as_matrix()
    wrenches = {}
    for index, obj in enumerate(objects):
        weight = obj.mass * gravity
        length = np.cbrt(obj.volume)
        # Body 2 has the worst relative residual despite unequal mass and size.
        force = rotation @ np.array([.1 * (index + 1) * weight, 0., 0.])
        torque = rotation @ np.array([0., .2 * weight * length, 0.])
        wrenches[index + 1] = np.r_[force, torque]
    values = placement_values(objects, wrenches, solver_type, gravity)
    assert values["diagnostics"][KEY] == pytest.approx(.4)
    # Existing feasibility thresholds still receive precisely the legacy metric.
    assert values["c_feq"] == pytest.approx(
        max(np.linalg.norm(w, ord=1) for w in wrenches.values()) / max(o.mass for o in objects))
    objects.append(NS(object_id=123, mass=1e12, volume=1e6))
    assert placement_values(objects, wrenches, solver_type, gravity)["diagnostics"][KEY] == pytest.approx(.4)


def score(config, displacement, diagnostics=None, c_feq=.5):
    objects = inventory()
    overlap = NS(object_containment=lambda *_: 1.)
    action = Action(0, POSE, POSE, object_id=10, c_feq=c_feq,
                    place_robustness_displacement=displacement,
                    diagnostics=diagnostics or {})
    state = State([10, 90], [0], {}, [], [action])
    reward = StackingReward(objects, overlap, config, support_evaluator=lambda *_: (1, True))
    prior = PlacementPriorScorer(objects, overlap, posegen_threshold=config.posegen_threshold,
                                 stability_weight=2., containment_weight=0.)
    return reward.compute(state), prior.score(state, None, [ActionCandidate(action)])[0]


@pytest.mark.parametrize("platform,reference", [("excavator", .15), ("tabletop", .015)])
def test_platforms_assign_same_reward_to_equal_relative_motion(platform, reference):
    profile = load_environment_profile(ROOT / f"apps/{platform}/config/environment.yml")
    config = replace(profile.reward, weights=WEIGHTS)
    assert config.displacement_reference == reference
    assert config.displacement_reference == profile.place_stability.fail_displacement
    (value, info), prior = score(config, reference / 2., {KEY: .25}, c_feq=100.)
    assert info["place_stability"] == pytest.approx(-.5)
    assert info["place_robustness_displacement"] == reference / 2.
    assert info["place_robustness_displacement_reference"] == reference
    assert info["stability"] == pytest.approx(-.25)
    assert info["equilibrium_legacy_scaling"] is False
    assert value == pytest.approx(-1.5)
    assert prior == pytest.approx(-.5)


@pytest.mark.parametrize("displacement,nonfinite", [(np.inf, True), (.2, False)])
def test_displacement_fallback_and_clipping_precede_normalization(displacement, nonfinite):
    config = RewardConfig(1., .5, WEIGHTS, nonfinite_displacement=.2,
                          max_displacement=.1, displacement_reference=.02)
    (value, info), _ = score(config, displacement, {KEY: 2.})
    assert info["place_robustness_nonfinite"] == nonfinite
    assert info["place_robustness_clipped"] is True
    assert info["place_robustness_displacement"] == .1
    assert info["place_stability"] == -5.
    assert info["stability"] == -1.
    assert value == -12.


def test_missing_wrenches_and_historical_actions_keep_explicit_legacy_scoring():
    assert KEY not in placement_values(inventory(), {})["diagnostics"]
    (value, info), prior = score(RewardConfig(2., .5, WEIGHTS), .1)
    assert info["equilibrium_legacy_scaling"] is True
    assert info["stability"] == -.25
    assert value == pytest.approx(-.7)
    assert prior == -.5


@pytest.mark.parametrize("relative_wrench,expected", [
    ([0., 0., 0., 0., 0., 0.], 0.),
    ([0., 0., -1., 0., 0., 0.], 1.),
    ([0., 0., 0., .5, 0., 0.], .5),
])
def test_equilibrium_has_physical_reference_values(relative_wrench, expected):
    objects = inventory(scale=.2, density=3.)
    body = objects[0]
    wrench = np.asarray(relative_wrench) * body.mass * 9.81
    wrench[3:] *= np.cbrt(body.volume)
    assert placement_values(objects, {1: wrench})["diagnostics"][KEY] == pytest.approx(expected)


@pytest.mark.parametrize("reference", [0., -1., np.nan, np.inf])
def test_invalid_displacement_reference_is_rejected(reference):
    with pytest.raises(ValueError, match="displacement reference"):
        RewardConfig(1., .5, WEIGHTS, displacement_reference=reference)
