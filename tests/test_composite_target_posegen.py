"""Compound targets use one native target entity with one DSF per component."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

core = pytest.importorskip("stacking_core")

from stacking_common.actions.config import CudaPosegenCapacityConfig
from stacking_common.actions.core_solver import CorePlacementPoseSolver, CorePosegenProfile
from stacking_common.actions.cuda_solver import CudaPlacementPoseSolver
from stacking_common.actions.types import InitializedPlacement
from stacking_common.domain import (
    CompositeTargetGeometry, ConvexGeometry, DirectionalTaper, ObjectInventory,
    StackObject, State, TargetComponent, TargetGeometry,
)


def pose(x=0., y=0., z=0., yaw=0.):
    return np.array([x, y, z, 0., 0., np.sin(yaw / 2.), np.cos(yaw / 2.)])


def box(width=.8, length=.6, height=.8):
    return TargetGeometry(width, length, height, np.zeros(3), DirectionalTaper(),
                          support_overlap=0.)


def composite():
    region = box()
    geometry = ConvexGeometry(region.points, region.faces)
    return CompositeTargetGeometry((
        TargetComponent("west", geometry, pose(-1.5)),
        TargetComponent("north-east", geometry, pose(1.5, .2, yaw=np.pi / 3.)),
    ))


def inventory_for(target, count=3):
    shape = box(.1, .1, .1)
    points = shape.points - np.array([0., 0., .05])
    objects = [StackObject(10 + index, core.BodyModel.dsf(
        20 + index, points, mass=1., inertia=np.eye(3) * .001,
        friction=.5, sharpness=20,
    ), [ConvexGeometry(points, shape.faces)]) for index in range(count)]
    inventory = ObjectInventory(objects, {obj.object_id: pose() for obj in objects}, target)
    return inventory, State(list(inventory.object_ids), [], {}, [], [])


def profile(**changes):
    return replace(CorePosegenProfile(
        frame_id=3, target_model_id=100, target_entity_id=1000,
        target_friction=.7, target_sharpness=16, target_epsilon=1e-7,
        trust_region={"max_iters": 2}, hausdorff={"max_iters": 8},
        force_solver={"max_iters": 32}, objective={"gravity": 9.81},
        schedule={"times": (1.,), "max_iters_warmup": 2, "max_iters_final": 2},
    ), **changes)


def recording_native():
    calls = []

    def factory(kind):
        def create(*args, **kwargs):
            calls.append((kind, args, kwargs))
            return getattr(core.BodyModel, kind)(*args, **kwargs)
        return create

    native = SimpleNamespace(**{name: getattr(core, name) for name in (
        "BodyInstance", "Mobility", "PosegenConfig", "PoseGenerator",
        "SceneSnapshot", "PosegenProblem",
    )})
    native.BodyModel = SimpleNamespace(dsf=factory("dsf"), dsf_components=factory("dsf_components"))
    return native, calls


def test_components_preserve_order_world_transforms_and_native_settings():
    target = composite()
    inventory, _ = inventory_for(target)
    native, calls = recording_native()
    configured = profile()
    solver = CorePlacementPoseSolver(inventory, configured, core_module=native)
    assert len(calls) == 1
    kind, args, kwargs = calls[0]
    assert kind == "dsf_components"
    assert args[0] == configured.target_model_id
    assert len(args[1]) == len(target.components) == solver.target_model.geometry_count == 2
    for actual, component in zip(args[1], target.components, strict=True):
        np.testing.assert_array_equal(actual, component.world_geometry.points)
    assert kwargs["frictions"] == [configured.target_friction] * 2
    assert kwargs["sharpnesses"] == [configured.target_sharpness] * 2
    assert kwargs["epsilon"] == configured.target_epsilon
    np.testing.assert_array_equal(solver.target_body.pose, pose())
    assert solver.target_body.mobility == core.Mobility.STATIC


@pytest.mark.parametrize("prepared", [False, True])
def test_compound_target_is_one_target_not_a_collision_boundary(prepared):
    inventory, state = inventory_for(composite())
    state.placed_object_indices = [0]
    state.object_poses[10] = pose(-1.5, z=.05)
    solver = CorePlacementPoseSolver(inventory, profile())
    context = solver._prepare_problem_scene(state) if prepared else None
    problem = solver._problem(state, 1, pose(1.5, .2, .05), scene_context=context)
    assert problem.targets == [1000]
    assert problem.boundaries == []
    assert problem.candidate == 2
    assert [body.id for body in problem.scene.bodies] == [1000, 1, 2]
    assert [body.model.geometry_count for body in problem.scene.bodies] == [2, 1, 1]


def test_boundary_radius_uses_components_not_their_enclosing_rectangle():
    inventory, state = inventory_for(composite(), count=4)
    state.placed_object_indices = [0, 1, 2]
    state.object_poses = {
        10: pose(-1.5, z=.05),  # Inside the western component.
        11: pose(1.5, .2, .05),  # Inside the transformed eastern component.
        12: pose(0., z=.05),  # Empty gap, though inside aggregate bounds.
    }
    solver = CorePlacementPoseSolver(inventory, profile(boundary_radius=.1))
    _, boundaries = solver._scene_objects(state)
    assert boundaries == [3]
    state.placed_object_indices = []
    assert solver._scene_objects(state) == ([], [])


def test_legacy_single_target_keeps_exact_dsf_inputs_and_boundary_semantics():
    target = box()
    inventory, state = inventory_for(target)
    native, calls = recording_native()
    configured = profile(boundary_radius=.1)
    solver = CorePlacementPoseSolver(inventory, configured, core_module=native)
    kind, args, kwargs = calls[0]
    assert kind == "dsf"
    np.testing.assert_array_equal(args[1], target.points)
    assert kwargs["friction"] == configured.target_friction
    assert kwargs["sharpness"] == configured.target_sharpness
    state.placed_object_indices = [0, 1]
    state.object_poses = {10: pose(.45), 11: pose(.51)}
    assert solver._scene_objects(state)[1] == [2]


@pytest.mark.parametrize("component_index", [0, 1])
def test_native_cpu_can_solve_at_either_component_and_reports_containment_mode(component_index):
    target = composite()
    inventory, state = inventory_for(target, count=1)
    solver = CorePlacementPoseSolver(inventory, profile())
    initial_pose = target.components[component_index].pose.copy()
    initial_pose[2] = .07
    result = solver.solve(inventory, state, 0, InitializedPlacement(initial_pose, 0.))
    assert np.isfinite(result.pose).all()
    assert np.isfinite(result.c_gap)
    assert target.contains(result.pose[None, :3])[0]
    assert result.diagnostics["posegen_target_component_ids"] == ("west", "north-east")
    assert result.diagnostics["posegen_target_component_count"] == 2
    assert result.diagnostics["posegen_target_containment"] == "whole_stone_in_one_component"


def test_one_component_compound_matches_legacy_native_solve():
    target = box()
    compound = CompositeTargetGeometry((
        TargetComponent("single", ConvexGeometry(target.points, target.faces)),
    ))
    results = []
    for geometry in (target, compound):
        inventory, state = inventory_for(geometry, count=1)
        solver = CorePlacementPoseSolver(inventory, profile())
        results.append(solver.solve(inventory, state, 0,
                                    InitializedPlacement(pose(z=.07), 0.)))
    legacy, actual = results
    np.testing.assert_allclose(actual.pose, legacy.pose, rtol=0., atol=1e-12)
    assert actual.c_gap == pytest.approx(legacy.c_gap, abs=1e-12)
    assert actual.c_feq == pytest.approx(legacy.c_feq, abs=1e-12)
    assert actual.converged == legacy.converged


def test_cuda_workspace_counts_all_target_components_without_more_target_bodies():
    inventory, _ = inventory_for(composite())
    native, _ = recording_native()
    capacities = []
    native.cuda_capabilities = lambda: {"posegen_solve": True, "devices": [0]}
    native.CudaCapacity = SimpleNamespace
    native.CudaPoseGenerator = lambda capacity: capacities.append(capacity)
    solver = CudaPlacementPoseSolver(inventory, profile(), core_module=native,
        capacity=CudaPosegenCapacityConfig(batch=2, bodies=1, geometries=1))
    assert solver.target_model.geometry_count == 2
    assert capacities[0].bodies == 4
    assert capacities[0].geometries == 5
