"""Excavator platform boundaries using commissioned models and recorded inputs."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stacking_common.artifacts import StackRunStore, PlanningCheckpoint, PlacementPlanRecord
from stacking_common.artifacts.pick_scene import load_pick_scene
from stacking_common.artifacts.planning_context import branch_planning_run
from stacking_common.domain import Action, State
from stacking_common.environment.transition import PlacementResult
from stacking_common.planning import FrameTransform, MotionPlanSegment, PlacementPlanningResult, PlanningStatus
from stacking_common.profiles.native import load_native_solver_profile
from stacking_common.profiles.planning import PlanningObstacle
from stacking_common.search import MCTSNode
from stacking_common.visualization import MCTSDebugArtifact, project_mcts_step
from stacking_common.workflows.planning_sequence import MCTSPlanningSequenceCoordinator
from stacking_excavator import ExcavatorApplication
from stacking_excavator.planning.cli import planning_context
from stacking_excavator.planning.coordinator import ExcavatorPlanningCoordinator
from stacking_excavator.planning.native import retry_handoff

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "tests/fixtures/legacy_planner/reference.json").read_text())


def pose(x=0., y=0., z=0.):
    return np.array([x, y, z, 0., 0., 0., 1.])


@pytest.fixture
def application():
    return ExcavatorApplication.from_repository(ROOT)


@pytest.fixture
def inventory(application):
    return application.load_inventory([55, 50], {
        i: np.concatenate(REFERENCE["pickup_poses"][str(i)]) for i in (55, 50)})


def test_phase_scenes_and_commissioned_gripper(application, inventory):
    application = replace(application, planning_profile=replace(application.planning_profile,
        obstacles=(PlanningObstacle("fence", [10., 10., 0.], [11., 11., 2.]),)),
        excavator_profile=replace(application.excavator_profile, pick_ground_height=.1,
            place_ground_height=.2, handoff_ground_height=.3,
            handoff_positions=((-2., -5., .3),)))
    backend = application.build_native_motion_planner(inventory)
    action = Action(0, pose(.3, .4, .5), pose(), object_id=55)
    problem = application.build_placement_planning_problem(State([55, 50], [], {}, [], []),
        action, pick_pose=inventory.initial_pose(55),
        pick_scene_poses={i: inventory.initial_pose(i) for i in (55, 50)})
    native = backend.build_problem(problem)
    assert backend.robot.initial_state.model.dof_count == 6
    np.testing.assert_array_equal(application.planning_profile.gripper.opening_direction, [1, 1])
    assert len(backend.gripper.contact_geometries) == 8
    assert set(native.direct.pick.scene.entity_ids) == {0, 1, 2, 10000}
    assert set(native.direct.place.scene.entity_ids) == {0, 1, 10000}
    for phase, height in ((native.direct.pick, .1), (native.direct.place, .2), (native.handoff, .3)):
        assert phase.scene.snapshot.body_pose(0)[2] == height
        assert phase.target_id == 1
    np.testing.assert_allclose(native.direct.place.scene.snapshot.body_pose(1), problem.place_pose)
    np.testing.assert_allclose(problem.place_pose[:3], [-5.7, 3.4, .7])


def test_rotated_frame_and_lift_are_applied_once(application):
    transform = FrameTransform("world", "stack", np.r_[[2., 3., .4],
        Rotation.from_euler("z", 90, degrees=True).as_quat()])
    application = replace(application, planning_profile=replace(application.planning_profile,
        world_T_stack=transform))
    action = Action(1, pose(1., 0., .5), pose(), object_id=50)
    state = State([55, 50], [0], {55: pose(0., 1., .2)}, [], [])
    problem = application.build_placement_planning_problem(state, action, pick_pose=pose(3., -4., .5))
    np.testing.assert_allclose(problem.place_pose[:3], [2., 4., 1.1])
    np.testing.assert_allclose(problem.place_scene_objects[0].pose[:3], [1., 3., .6])
    np.testing.assert_array_equal(problem.action.pose, action.pose)
    np.testing.assert_array_equal(state.object_poses[55], pose(0., 1., .2))


@pytest.mark.parametrize("code,retryable,expected", [
    ("regrasp_motion", True, True), ("regrasp_a_grasp_generation", True, True),
    ("regrasp_handoff_target_boundary", True, True),
    ("regrasp_ac_ik_recovery", True, True),
    ("regrasp_bc_ik_recovery", True, True),
    ("regrasp_pick_ik_recovery", True, False),
    ("regrasp_place_ik_recovery", True, False),
    ("regrasp_pick_approach_motion", True, False),
    ("regrasp_place_target_boundary", True, False),
    ("regrasp_b_grasp_generation", True, False), ("regrasp_place_ik", True, False),
    ("direct_grasp_generation", True, False), ("regrasp_motion", False, False)])
def test_handoff_retry_policy(code, retryable, expected):
    result = PlacementPlanningResult(PlanningStatus.INFEASIBLE, reason=code,
        diagnostics={"failure_code": code, "retryable": retryable})
    assert bool(retry_handoff(result)) is expected


def test_handoff_retry_order_and_diagnostics(application, inventory, monkeypatch):
    import stacking_core as core
    positions = []

    def solve(problem, config):
        positions.append(np.asarray(problem.handoff_position).copy())
        if len(positions) < 3:
            return NS(status="infeasible", candidates=[], selected_index=None,
                      failure=NS(code="regrasp_to_handoff_motion", message="no path", retryable=True))
        sample = NS(robot=NS(positions=np.zeros(6), gripper_opening=-.4),
                    frame_from_target=pose())
        segment = NS(stage="handoff_pick", mode="attached", trajectory=NS(samples=[sample]))
        return NS(status="success", candidates=[NS(segments=[segment], score=1., grasp_events=[])],
                  selected_index=0)

    monkeypatch.setattr(core, "solve_pick_place", solve)
    backend = application.build_native_motion_planner(inventory)
    problem = application.build_placement_planning_problem(State([55, 50], [], {}, [], []),
        Action(0, pose(), pose(), object_id=55), pick_pose=inventory.initial_pose(55))
    result = backend.plan(problem)
    assert result.succeeded
    np.testing.assert_array_equal(positions, application.excavator_profile.handoff_positions[:3])
    assert len(result.diagnostics["handoff_attempts"]) == 3
    assert result.diagnostics["elapsed_seconds"] == sum(
        item["diagnostics"]["elapsed_seconds"] for item in result.diagnostics["handoff_attempts"])
    assert result.segments[0].scene_phase == "handoff"
    assert result.segments[0].openings[0] == -.4


def test_recovery_policies_are_excavator_only(application, inventory):
    from stacking_tabletop import TabletopApplication
    import stacking_core as core
    excavator = application.build_native_motion_planner(inventory).config
    tabletop = TabletopApplication.from_repository(ROOT)
    reference = load_native_solver_profile(tabletop.solver_profiles.grasp, tabletop.solver_profiles.motion).native_config(core)
    for config in (excavator.direct, excavator.regrasp):
        assert config.joint_grasp_recovery and config.scene_approach
        assert config.minimum_approach_up_component == .5
    for config in (reference.direct, reference.regrasp):
        assert not config.joint_grasp_recovery and not config.scene_approach
    np.testing.assert_allclose(excavator.direct.grasp_sampling.fallback_retreat_distances, [.975, 1.2, 1.8])
    assert reference.direct.grasp_sampling.fallback_retreat_distances == []


def test_catalog_assets_apply_density_once(application):
    from stacking_common.assets import _parse_object_asset
    ids = application.inventory_profile.object_ids
    inventory = application.load_inventory(ids, {i: pose() for i in ids})
    for obj in inventory:
        asset = _parse_object_asset(ROOT / application.inventory_profile.model_directory / f"model_{obj.object_id}.obj")
        assert obj.body_model.mass == pytest.approx(asset.mass * application.inventory_profile.density)


@pytest.mark.parametrize("value", [[], {"55": [1, 2, 3]}, {"55": [0, 0, 0, 0, 0, 0, 0]}])
def test_pick_scene_rejects_malformed_inputs(tmp_path, value):
    path = tmp_path / "pick.json"
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError, TypeError)):
        load_pick_scene(path)


@pytest.mark.parametrize("failed", [False, True])
def test_fixed_scene_failure_and_unlifted_commit(application, inventory, failed):
    # This fixture tests pose bookkeeping with disconnected synthetic stones;
    # strict floor rejection is covered by test_excavator_floor_first.
    state = State([55, 50], [0], {55: pose(.2, .3, .4)}, [], [])
    action = Action(1, pose(.5, .6, .8), pose(), object_id=50)
    calls = []
    settled = pose(.51, .61, .7)

    def settle(fixed, prepared):
        assert 55 in fixed.pose_identified_object_ids
        return PlacementResult({55: state.object_poses[55], 50: settled}, failed=failed,
                               diagnostics={"rejection_reason": "unsupported"} if failed else {})

    def plan(problem):
        calls.append(problem)
        return PlacementPlanningResult(PlanningStatus.SUCCESS,
                                      (MotionPlanSegment("transfer", np.zeros((2, 6))),))

    backend = NS(reset=lambda _: None, settle_fixed=settle)
    coordinator = ExcavatorPlanningCoordinator(application, inventory, NS(plan=plan), placement_backend=backend)
    record = coordinator.attempt(state, action, step=2, attempt=1)
    assert record.result.succeeded is not failed
    assert bool(calls) is not failed
    assert 50 not in state.object_poses
    np.testing.assert_array_equal(action.pose, pose(.5, .6, .8))
    if failed:
        assert record.result.reason == "fixed_scene_resettle_unsupported"
    else:
        np.testing.assert_allclose(record.problem.place_pose[2], .9)
        from stacking_common.search import MCTSSearchRequest, MCTSSession
        root = MCTSNode(state=state, observation=object())
        child = root.add_candidate(action)
        child.set_result(State([55, 50], [0, 1], {55: pose(), 50: action.pose.copy()}, [], [action.copy()]),
                         object(), 0.)
        sequence = MCTSPlanningSequenceCoordinator(MCTSSession(NS(), root),
            MCTSSearchRequest(iterations=1), coordinator, frame_id="stack",
            committed_state_evaluator=application.build_committed_state_evaluator(inventory))
        sequence._commit_planned_pose(state, child, record)
        np.testing.assert_array_equal(child.state.object_poses[50], settled)
        np.testing.assert_array_equal(child.state.object_poses[55], state.object_poses[55])
        np.testing.assert_allclose(child.observation.object_poses[1], settled)
        np.testing.assert_allclose(child.observation.object_poses[0], state.object_poses[55])


def test_portable_pick_scene_and_excavator_resume_context(application, tmp_path):
    source = tmp_path / "pick.json"
    source.write_text(json.dumps({55: REFERENCE["pickup_poses"]["55"]}))
    scene = load_pick_scene(source)
    inventory = application.load_inventory(scene.object_ids, {55: scene.pose(55)})
    solver = load_native_solver_profile(application.solver_profiles.grasp, application.solver_profiles.motion)
    context = planning_context(application, inventory, scene, solver, {"seed": 0})
    state = State([55], [0], {55: pose(0., 0., .4)}, [], [])
    path = tmp_path / "source.stack"
    writer = StackRunStore.create_mcts(path, MCTSDebugArtifact("stack", ()))
    writer.append_run_context(context)
    action = Action(0, state.object_poses[55], pose(), object_id=55)
    root = MCTSNode(state=State([55], [], {}, [], []))
    child = MCTSNode(state=state, action=action, info={"candidate_id": "candidate-1"})
    root.children.append(child)
    plan = PlacementPlanRecord(1, 1, application.build_placement_planning_problem(
        root.state, action, pick_pose=scene.pose(55)), PlacementPlanningResult(PlanningStatus.SUCCESS,
        (MotionPlanSegment("transfer", np.zeros((2, 6))),)), "candidate-1", context.sha256)
    writer.append_placement_plan(plan)
    writer.append_mcts_step(project_mcts_step(root, child, step=1, frame_id="stack"),
        checkpoint=PlanningCheckpoint(1, context.sha256, state, scene, application.planning_profile.world_T_stack))
    writer.finish("complete")
    writer.close()
    # Context includes support planes, model closure, physical values, and seed.
    restored = StackRunStore.load_run_context(path, context.sha256)
    assert restored.sha256 == context.sha256
    assert len(restored.fixed_geometry) == 3
    branched, checkpoint = branch_planning_run(path, tmp_path / "branch.stack", context)
    branched.close()
    assert checkpoint.step == 1
    changed = replace(context, profiles={**context.profiles, "pose_from_scan": True})
    with pytest.raises(ValueError, match="context|profile"):
        branch_planning_run(path, tmp_path / "bad.stack", changed)


def test_native_planner_accepts_seeded_stationary_excavator_motion(application, inventory):
    """Exercise actual motion solving with the commissioned kinematic/gripper models."""
    import stacking_core as core
    application = replace(application, excavator_profile=replace(application.excavator_profile, place_z_offset=0.))
    backend = application.build_native_motion_planner(inventory)
    q_home = np.asarray(backend.robot.initial_state.positions)
    tool_pose = np.asarray(backend.robot.initial_state.link_pose(backend.robot.tool_link))
    # Put the real stone clear of the gripper; this tests motion, not grasp quality.
    target = tool_pose.copy()
    target[:3] += Rotation.from_quat(tool_pose[3:]).apply([0., 2., 0.])
    local = target.copy()
    local[:3] -= application.planning_profile.world_T_stack.parent_T_child[:3]
    problem = application.build_placement_planning_problem(State([55, 50], [], {}, [], []),
        Action(0, local, local, object_id=55), pick_pose=target)
    native = backend.build_problem(problem)
    candidate = core.GraspCandidate()
    candidate.grasp = core.Grasp(tool_pose, opening=-.4)
    candidate.score = 1.
    joint = core.JointGraspCandidate()
    joint.grasp, joint.positions = candidate, [q_home, q_home]
    native.direct.grasp_candidates = [joint]
    config = backend.config
    config.allow_regrasp = False
    config.direct.simulation_refinement = False
    config.direct.approach_distance = 0.
    config.direct.move_steps = config.direct.grasp_steps = 2
    config.direct.motion.max_iters = 2
    result = core.solve_pick_place(native, config)
    assert result.status.name == "SUCCESS", result.failure.message
    assert {s.stage.name for s in result.candidates[0].segments} >= {"PICK_APPROACH", "PLACE_APPROACH"}
