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


def placement_values(objects, wrenches, solver_type=CorePlacementPoseSolver, gravity=9.81,
                     native_mass_scale=3.7, contact_force=None):
    """Return adapter values for physical wrenches, as native posegen reports them."""
    solver = solver_type.__new__(solver_type)
    solver.inventory = objects
    solver.gravity = gravity
    result = NS(
        solver=NS(iters=1, objective_evals=1, grad_norm=0., converged=True,
                  force_solver=NS(converged=True), scene_graph_rebuilds=1,
                  scene_graph_reuses=0),
        c_comp=0., contact_point={}, contact_normal={},
        contact_force={key: [np.asarray(force) / native_mass_scale for force in forces]
                       for key, forces in (contact_force or {}).items()},
        net_wrench={key: np.asarray(wrench) / native_mass_scale for key, wrench in wrenches.items()},
        c_gap=0., optimal_pose=POSE,
    )
    return solver._placement_values(result, native_mass_scale)


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
    # Admission, reward, and prior share the same per-body normalized residual.
    assert values["c_feq"] == pytest.approx(.4)
    objects.append(NS(object_id=123, mass=1e12, volume=1e6))
    extra = placement_values(objects, wrenches, solver_type, gravity)
    assert extra["diagnostics"][KEY] == pytest.approx(.4)
    assert extra["c_feq"] == pytest.approx(.4)


def test_native_mass_normalization_is_removed_from_wrenches_and_forces():
    objects = inventory()
    wrenches = {1: np.array([0., 0., -2., 0., .5, 0.])}
    force = np.array([1., 0., 9.])
    values = placement_values(objects, wrenches, native_mass_scale=40., contact_force={1: [force]})
    np.testing.assert_allclose(values["diagnostics"]["posegen_contact_forces"][1], [force])
    assert values["diagnostics"][KEY] == pytest.approx(2. / (objects[0].mass * 9.81)
                                                       + .5 / (objects[0].mass * 9.81))
    assert values["c_feq"] == pytest.approx(values["diagnostics"][KEY])
    np.testing.assert_allclose(values["diagnostics"]["posegen_residual_wrenches"][1], wrenches[1])


def test_native_mass_scale_is_mean_mass_of_candidate_and_placed_bodies():
    solver = CorePlacementPoseSolver.__new__(CorePlacementPoseSolver)
    solver.inventory = Inventory([NS(mass=2.), NS(mass=4.), NS(mass=9.)])
    state = NS(placed_object_indices=[0, 2])
    assert solver._native_mass_scale(state, 1) == pytest.approx(5.)
    assert solver._native_mass_scale(NS(placed_object_indices=[]), 1) == pytest.approx(4.)


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


def test_missing_wrenches_are_invalid_and_historical_actions_keep_legacy_scoring():
    with pytest.raises(ValueError, match="residual wrenches"):
        placement_values(inventory(), {})
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


@pytest.mark.parametrize("solver_type", [CorePlacementPoseSolver, CudaPlacementPoseSolver])
@pytest.mark.parametrize("relative,accepted", [(.99, True), (1., False), (1.1, False)])
def test_admission_reward_and_prior_share_the_residual(solver_type, relative, accepted):
    from stacking_common.actions import PosegenFeasibilityFilter
    from stacking_common.actions.types import PlacementSolveResult
    objects = inventory()
    body = objects[0]
    values = placement_values(objects, {1: np.array([relative * body.mass * 9.81, 0., 0., 0., 0., 0.])}, solver_type)
    solved = PlacementSolveResult(**values)
    decision = PosegenFeasibilityFilter(1., 1., reject_equilibrium_at_threshold=True).evaluate(
        objects, None, 0, None, solved)
    assert decision.accepted is accepted
    assert decision.diagnostics[KEY] == pytest.approx(relative)
    (_, info), prior = score(RewardConfig(1., .5, WEIGHTS), 0., values["diagnostics"], values["c_feq"])
    assert info["stability"] == pytest.approx(-min(relative, 1.))
    assert prior == pytest.approx(-2. * min(relative, 1.))


@pytest.mark.parametrize("wrenches,message", [
    ({1: np.full(6, np.nan)}, "finite six-vector"),
    ({1: np.zeros(3)}, "finite six-vector"),
    ({999: np.zeros(6)}, "unknown body"),
])
def test_invalid_residual_wrenches_are_not_replaced_by_a_scalar_cost(wrenches, message):
    with pytest.raises(ValueError, match=message):
        placement_values(inventory(), wrenches)
