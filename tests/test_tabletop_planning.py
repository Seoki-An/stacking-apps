import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from stacking_common.artifacts import (
    PlacementPlanRecord,
    StackRunStore,
    decode_placement_plan,
    encode_placement_plan,
)
from stacking_common.domain import Action, ConvexGeometry, State
from stacking_common.planning import (
    MotionPlanSegment,
    PlacementPlanningProblem,
    PlacementPlanningResult,
    PlanningStatus,
    SceneMobility,
    SceneObjectInstance,
)
from stacking_common.artifacts import decode_legacy_mcts_debug
from stacking_tabletop.gui.operations import OperationsController
from stacking_tabletop import TabletopApplication
from stacking_tabletop.config import planning_profile_with_pick_height
from stacking_tabletop.planning import (
    TabletopMotionPlannerAdapter,
    TabletopNativePlanningBackend,
    TabletopPlanningCoordinator,
    TabletopPlacementSafety,
    TabletopRobotTrajectoryProjector,
    infer_legacy_pick_support,
    load_legacy_pick_scene,
    load_native_solver_profile,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "tests/fixtures/legacy_visualization/mcts_candidate_viewer.json"
)


def _pose(x=0.0, y=0.0, z=0.0):
    return np.array([x, y, z, 0.0, 0.0, 0.0, 1.0])


def _problem():
    action = Action(0, _pose(-0.5, 0.0, 0.2), _pose(), object_id=11)
    return PlacementPlanningProblem(
        frame_id="world",
        action=action,
        pick_pose=_pose(0.2, -0.7, 0.1),
        place_pose=_pose(-0.5, 0.0, 0.2),
        scene_objects=(
            SceneObjectInstance(
                0,
                11,
                _pose(0.2, -0.7, 0.1),
                SceneMobility.KINEMATIC,
            ),
        ),
        metadata={"planning_T_search": _pose(-0.55, 0.0, 0.115)},
    )


def _success_result():
    return PlacementPlanningResult(
        PlanningStatus.SUCCESS,
        segments=(
            MotionPlanSegment(
                "pick_approach",
                np.arange(18, dtype=float).reshape(3, 6),
                openings=np.array([0.7, 0.4, 0.1]),
            ),
        ),
        score=2.5,
        diagnostics={"native_status": "success"},
    )


def _artifact():
    return decode_legacy_mcts_debug(
        json.loads(FIXTURE.read_text(encoding="utf-8"))
    )


def _state(object_ids=(11,)):
    return State(list(object_ids), [], {}, [], [])


def test_tabletop_motion_adapter_normalizes_native_plan() -> None:
    samples = [
        SimpleNamespace(
            robot=SimpleNamespace(
                positions=np.full(6, index),
                gripper_opening=0.2 + index,
            ),
            frame_from_target=_pose(0.1 * index, 0.0, 0.2),
        )
        for index in range(3)
    ]
    segment = SimpleNamespace(
        stage=SimpleNamespace(name="PICK_APPROACH"),
        mode=SimpleNamespace(name="FREE"),
        trajectory=SimpleNamespace(samples=samples),
    )
    candidate = SimpleNamespace(segments=[segment], score=4.0, grasp_events=[SimpleNamespace(
        event=SimpleNamespace(name="ACQUIRE"), segment_index=0, sample_index=2,
        grasp=SimpleNamespace(frame_from_grasp=_pose(.2, 0., .2), opening=.4),
    )])
    native = SimpleNamespace(
        status=SimpleNamespace(name="SUCCESS"),
        candidates=[candidate],
        selected_index=0,
        failure=SimpleNamespace(code="", message="", retryable=False),
        timings=SimpleNamespace(
            grasp_generation_seconds=1.0,
            simulation_refinement_seconds=2.0,
            trajectory_optimization_seconds=3.0,
            total_seconds=6.0,
            grasp_candidates=30,
            refined_candidates=4,
            motion_candidates=2,
        ),
    )

    result = TabletopMotionPlannerAdapter(lambda problem: native).plan(
        _problem()
    )

    assert result.succeeded
    assert result.segments[0].name == "pick_approach"
    assert result.segments[0].mode == "free"
    assert result.grasp_events[0].event == "acquire"
    assert result.grasp_events[0].sample_index == 2
    assert result.grasp_events[0].opening == .4
    assert result.segments[0].positions.shape == (3, 6)
    assert result.segments[0].times is None
    assert np.allclose(result.segments[0].openings, [0.2, 1.2, 2.2])
    assert result.score == 4.0
    assert result.diagnostics["native_timings"] == {
        "grasp_generation_seconds": 1.0,
        "simulation_refinement_seconds": 2.0,
        "trajectory_optimization_seconds": 3.0,
        "total_seconds": 6.0,
        "grasp_candidates": 30,
        "refined_candidates": 4,
        "motion_candidates": 2,
    }
    assert np.allclose(result.diagnostics["achieved_place_pose"], _pose(0.2, 0.0, 0.2))


def test_tabletop_motion_adapter_preserves_native_failure() -> None:
    native = SimpleNamespace(
        status=SimpleNamespace(name="INFEASIBLE"),
        candidates=[],
        selected_index=None,
        failure=SimpleNamespace(
            code="place_collision",
            message="collision entities=1001,101 sample=4 gap_m=-0.003 required_gap_m=0 violation_m=0.003",
            retryable=True,
        ),
    )

    result = TabletopMotionPlannerAdapter(lambda problem: native).plan(
        _problem()
    )

    assert result.status is PlanningStatus.INFEASIBLE
    assert result.reason.startswith("place_collision:")
    assert result.diagnostics["failure_message"] == native.failure.message
    assert native.failure.message in result.reason
    assert result.diagnostics["retryable"]


def test_stacking_core_adapter_reports_missing_nanobind_planner_surface() -> None:
    adapter = TabletopMotionPlannerAdapter.from_stacking_core(
        lambda problem, core: object(),
        core_module=SimpleNamespace(),
    )

    result = adapter.plan(_problem())

    assert result.status is PlanningStatus.FAILED
    assert "does not expose solve_pick_place yet" in result.reason


def test_stack_run_and_controller_keep_multiple_planning_attempts(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    header = type(artifact)(
        frame_id=artifact.frame_id,
        target=artifact.target,
        camera=artifact.camera,
        steps=(),
    )
    path = tmp_path / "plan.stack"
    writer = StackRunStore.create_mcts(path, header)
    writer.append_mcts_step(artifact.steps[0])
    writer.append_placement_plan(
        PlacementPlanRecord(
            1,
            1,
            _problem(),
            _success_result(),
            candidate_id="step-1-candidate-1",
        )
    )
    writer.append_placement_plan(
        PlacementPlanRecord(
            1,
            2,
            _problem(),
            PlacementPlanningResult(
                PlanningStatus.INFEASIBLE,
                reason="ik_failure",
            ),
            candidate_id="step-1-candidate-1",
        )
    )
    writer.finish()

    records = StackRunStore.load_placement_plans(path)
    controller = OperationsController()
    controller.load(path)

    assert [record.attempt for record in records] == [1, 2]
    assert records[0].result.segments[0].times is None
    assert np.allclose(
        records[0].result.segments[0].openings,
        _success_result().segments[0].openings,
    )
    assert np.array_equal(
        records[0].result.segments[0].positions,
        _success_result().segments[0].positions,
    )
    assert StackRunStore.info(path).content_types == (
        "mcts_debug",
        "placement_plan",
    )
    assert controller.current_plan.attempt == 2
    controller.select_plan_attempt(1)
    assert controller.current_plan.result.succeeded
    frame = controller.planning_frame()
    assert [value.layer for value in frame.overlays] == [
        "candidate_initial",
        "candidate_solved",
    ]
    assert np.allclose(frame.overlays[0].pose, _problem().pick_pose)
    assert np.allclose(frame.overlays[1].pose, _problem().place_pose)
    assert np.allclose(
        frame.target.origin,
        artifact.target.origin + np.array([-0.55, 0.0, 0.115]),
    )


def test_planning_artifact_preserves_distinct_pick_and_place_scenes() -> None:
    target = SceneObjectInstance(
        0,
        11,
        _pose(0.2, -0.7, 0.1),
        SceneMobility.KINEMATIC,
    )
    pickup_neighbor = SceneObjectInstance(
        1,
        12,
        _pose(-0.2, -0.7, 0.1),
        SceneMobility.STATIC,
    )
    placed_neighbor = SceneObjectInstance(
        2,
        13,
        _pose(-0.5, 0.0, 0.1),
        SceneMobility.STATIC,
    )
    problem = PlacementPlanningProblem(
        "world",
        Action(0, _pose(-0.5, 0.0, 0.2), _pose(), object_id=11),
        target.pose,
        _pose(-0.5, 0.0, 0.2),
        (placed_neighbor, target),
        pick_scene_objects=(target, pickup_neighbor),
        place_scene_objects=(placed_neighbor, target),
    )
    record = PlacementPlanRecord(
        1,
        1,
        problem,
        PlacementPlanningResult(PlanningStatus.INFEASIBLE, reason="grasp"),
    )

    decoded = decode_placement_plan(encode_placement_plan(record))

    assert [value.object_id for value in decoded.problem.pick_scene_objects] == [
        11,
        12,
    ]
    assert [value.object_id for value in decoded.problem.place_scene_objects] == [
        13,
        11,
    ]


def test_planning_artifact_reads_legacy_single_scene_problem() -> None:
    record = PlacementPlanRecord(
        1,
        1,
        _problem(),
        PlacementPlanningResult(PlanningStatus.INFEASIBLE, reason="grasp"),
    )
    encoded = encode_placement_plan(record)
    encoded["problem"].pop("pick_scene_objects")
    encoded["problem"].pop("place_scene_objects")

    decoded = decode_placement_plan(encoded)

    assert decoded.problem.pick_scene_objects == decoded.problem.scene_objects
    assert decoded.problem.place_scene_objects == decoded.problem.scene_objects


def test_native_solver_profiles_map_to_nanobind_config() -> None:
    import stacking_core

    application = TabletopApplication.from_repository(ROOT)
    profile = load_native_solver_profile(
        application.solver_profiles.grasp,
        application.solver_profiles.motion,
    )
    config = profile.native_config(stacking_core)

    assert config.direct.grasp_sampling.max_seeds == 30
    assert config.direct.grasp_sampling.max_candidates == 16
    assert config.direct.grasp_sampling.spin_samples == 12
    assert config.direct.grasp_generation.cost.enclosure == 10.0
    assert config.direct.motion.grasped_boundary_pos_scale == 30.0
    for motion in (config.direct.motion, config.regrasp.motion):
        assert motion.collision_alm_enabled
        assert motion.collision_alm_max_iters == 5
        assert motion.collision_alm_beta_init == 100.0
        assert motion.collision_alm_beta_increase == 5.0
        assert motion.collision_alm_tol == 0.001
        assert motion.collision_penetration_clamp == 0.01
        assert motion.plane_feasibility_margin == -0.005
    assert config.direct.inverse_kinematics.max_iters == 2000
    assert config.direct.worker_count == 20
    assert config.regrasp.yaw_samples == 8
    assert config.regrasp.worker_count == 20
    assert not config.allow_regrasp


def test_native_problem_builder_uses_tabletop_models_and_phase_scenes() -> None:
    application, inventory, backend, problem = _native_fixture()

    native = backend.build_problem(problem)

    assert application.planning_profile.frame_id == 1
    assert native.direct.pick.scene.frame_id == 1
    assert native.direct.pick.target_id == 1
    assert native.direct.pick.scene.entity_ids == [100, 101, 102, 1]
    assert native.direct.place.scene.entity_ids == [100, 101, 102, 1]
    assert np.allclose(
        native.direct.robot.initial_state.positions,
        application.planning_profile.robot.home_positions,
    )
    assert native.direct.gripper.opening_upper == 0.743
    assert inventory.object_ids == (1,)


def test_legacy_pose_data_loads_selected_world_pick_scene(tmp_path: Path) -> None:
    path = tmp_path / "pose_data.pkl"
    with path.open("wb") as stream:
        pickle.dump(
            {
                1: (np.array([0.1, -0.7, 0.12]), np.array([0.0, 0.0, 0.0, 2.0])),
                2: (np.array([-0.1, -0.7, 0.13]), np.array([0.0, 0.0, 1.0, 1.0])),
            },
            stream,
        )

    scene = load_legacy_pick_scene(path, object_ids=(2, 1))

    assert scene.frame_id == "world"
    assert scene.object_ids == (2, 1)
    assert np.allclose(scene.pose(1), _pose(0.1, -0.7, 0.12))
    assert np.isclose(np.linalg.norm(scene.pose(2)[3:]), 1.0)
    assert scene.metadata["format"] == "legacy_pose_data_pickle"
    assert len(scene.metadata["sha256"]) == 64


def test_legacy_pick_support_updates_runtime_pick_table(tmp_path: Path) -> None:
    path = tmp_path / "pose_data.pkl"
    with path.open("wb") as stream:
        pickle.dump(
            {
                1: (np.array([0.0, -0.7, 0.12]), np.array([0.0, 0.0, 0.0, 1.0])),
                2: (np.array([0.2, -0.7, 0.13]), np.array([0.0, 0.0, 0.0, 1.0])),
            },
            stream,
        )
    scene = load_legacy_pick_scene(path)

    class Catalog:
        def load(self, object_id):
            bottom = -0.02 if object_id == 1 else -0.03
            return SimpleNamespace(
                points=np.array(
                    [[0.0, 0.0, bottom], [0.01, 0.0, bottom], [0.0, 0.01, bottom]]
                )
            )

    support = infer_legacy_pick_support(scene, Catalog())
    application = TabletopApplication.from_repository(ROOT)
    profile = planning_profile_with_pick_height(
        application.planning_profile,
        support.height,
    )
    pick_table = next(
        value for value in profile.obstacles if value.obstacle_id == "pick_table"
    )

    assert support.height == 0.1
    assert np.allclose(tuple(support.bottom_by_object.values()), [0.1, 0.1])
    assert pick_table.maximum[2] == 0.1


def test_legacy_pick_support_skips_pose_without_matching_model(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pose_data.pkl"
    with path.open("wb") as stream:
        pickle.dump(
            {
                1: (np.array([0.0, -0.7, 0.12]), np.array([0.0, 0.0, 0.0, 1.0])),
                99: (np.array([0.2, -0.7, 0.80]), np.array([0.0, 0.0, 0.0, 1.0])),
            },
            stream,
        )
    scene = load_legacy_pick_scene(path)

    class Catalog:
        def load(self, object_id):
            if object_id == 99:
                raise KeyError(object_id)
            return SimpleNamespace(points=np.array([[0.0, 0.0, -0.02]]))

    support = infer_legacy_pick_support(scene, Catalog())

    assert np.isclose(support.height, 0.1)
    assert np.isclose(support.bottom_by_object[1], 0.1)


def test_native_problem_uses_remaining_pick_objects_only() -> None:
    application = TabletopApplication.from_repository(ROOT)
    initial = {
        1: _pose(0.0, -0.7, 0.12),
        2: _pose(0.2, -0.7, 0.13),
    }
    inventory = application.load_inventory((1, 2), initial)
    solver = load_native_solver_profile(
        application.solver_profiles.grasp,
        application.solver_profiles.motion,
    )
    backend = TabletopNativePlanningBackend(
        inventory,
        application.planning_profile,
        solver,
        ROOT,
    )
    action = Action(0, _pose(0.0, 0.0, 0.02), _pose(), object_id=1)
    problem = application.build_placement_planning_problem(
        _state((1, 2)),
        action,
        pick_pose=initial[1],
        pick_scene_poses=initial,
    )

    native = backend.build_problem(problem)

    assert native.direct.pick.scene.entity_ids == [100, 101, 102, 1, 2]
    assert native.direct.place.scene.entity_ids == [100, 101, 102, 1]


def test_robot_trajectory_projector_moves_gripper_with_opening() -> None:
    application = TabletopApplication.from_repository(ROOT)
    projector = TabletopRobotTrajectoryProjector(
        application.planning_profile,
        ROOT,
    )

    closed = projector.project(
        application.planning_profile.robot.home_positions,
        0.1,
    )
    opened = projector.project(
        application.planning_profile.robot.home_positions,
        0.7,
    )

    assert len(closed) == 12
    assert all(value.geometry_kind == "visual" for value in closed)
    assert all(value.mesh_path.is_file() for value in closed)
    closed_by_id = {value.part_id: value.pose for value in closed}
    opened_by_id = {value.part_id: value.pose for value in opened}
    assert np.allclose(
        closed_by_id["robot/shoulder_link/visual/0"],
        opened_by_id["robot/shoulder_link/visual/0"],
    )
    assert not np.allclose(
        closed_by_id["gripper/left_finger_tip/visual/0"],
        opened_by_id["gripper/left_finger_tip/visual/0"],
    )


def test_native_planner_failure_crosses_common_boundary() -> None:
    _, _, backend, problem = _native_fixture()
    config = backend.config
    config.allow_regrasp = False
    config.direct.grasp_sampling.max_seeds = 1
    config.direct.grasp_sampling.spin_samples = 1
    config.direct.grasp_generation.trust_region.max_iters = 3
    config.direct.grasp_generation.alm.max_iters = 1
    config.direct.grasp_generation.force.enabled = False
    config.direct.simulation_refinement = False
    config.direct.inverse_kinematics.max_iters = 10
    config.direct.move_steps = 2
    config.direct.grasp_steps = 2
    config.direct.motion.max_iters = 2

    result = TabletopMotionPlannerAdapter(backend.solve).plan(problem)

    assert result.status is PlanningStatus.INFEASIBLE
    assert result.reason.startswith("direct_grasp_generation:")
    assert result.diagnostics["retryable"]


def test_native_planner_accepts_known_collision_free_motion() -> None:
    import stacking_core

    application, _, backend, _ = _native_fixture()
    q_home = np.asarray(backend.robot.initial_state.positions)
    tool_pose = np.asarray(
        backend.robot.initial_state.link_pose(backend.robot.tool_link)
    )
    target_pose = tool_pose.copy()
    target_pose[:3] += Rotation.from_quat(tool_pose[3:]).apply(
        [0.0, 0.15, 0.0]
    )
    place_in_stack = target_pose.copy()
    place_in_stack[:3] -= (
        application.planning_profile.world_T_stack.parent_T_child[:3]
    )
    action = Action(0, place_in_stack, _pose(), object_id=1)
    problem = application.build_placement_planning_problem(
        _state((1,)),
        action,
        pick_pose=target_pose,
    )
    native = backend.build_problem(problem)
    grasp = stacking_core.Grasp(tool_pose, opening=0.1)
    candidate = stacking_core.GraspCandidate()
    candidate.grasp = grasp
    candidate.score = 1.0
    joint_candidate = stacking_core.JointGraspCandidate()
    joint_candidate.grasp = candidate
    joint_candidate.positions = [q_home, q_home]
    native.direct.grasp_candidates = [joint_candidate]
    config = backend.config
    config.allow_regrasp = False
    config.direct.simulation_refinement = False
    config.direct.approach_distance = 0.0
    config.direct.move_steps = 2
    config.direct.grasp_steps = 2
    config.direct.motion.max_iters = 2

    result = stacking_core.solve_pick_place(native, config)

    assert result.status.name == "SUCCESS"
    assert len(result.candidates) == 1


def test_planning_coordinator_builds_and_appends_attempt() -> None:
    application = TabletopApplication.from_repository(ROOT)
    initial = {1: _pose(0.0, -0.7, 0.2)}
    inventory = application.load_inventory((1,), initial)
    action = Action(0, _pose(-0.55, 0.0, 0.2), _pose(), object_id=1)
    planner = TabletopMotionPlannerAdapter(
        lambda problem: SimpleNamespace(
            status=SimpleNamespace(name="INFEASIBLE"),
            candidates=[],
            selected_index=None,
            failure=SimpleNamespace(
                code="ik_failure",
                message="no branch",
                retryable=True,
            ),
        )
    )
    coordinator = TabletopPlanningCoordinator(
        inventory,
        application.build_placement_planning_problem,
        planner,
    )
    written = []
    writer = SimpleNamespace(append_placement_plan=written.append)

    record = coordinator.attempt(
        _state((1,)),
        action,
        step=2,
        attempt=3,
        candidate_id="candidate-4",
        writer=writer,
    )

    assert written == [record]
    assert record.problem.metadata["planning_attempt"] == 3
    assert record.problem.metadata["search_frame_id"] == "stack"
    assert record.result.status is PlanningStatus.INFEASIBLE


def test_planning_coordinator_uses_evaluated_then_fixed_scene_pose() -> None:
    class Inventory:
        object_ids = (1,)

        def __len__(self):
            return 1

        def initial_pose(self, object_id):
            assert object_id == 1
            return _pose(0.0, -0.7, 0.2)

    class FixedBackend:
        def place(self, state, action):
            assert state.pose_identified_object_ids == set()
            settled = action.pose.copy()
            settled[2] -= 0.01
            return SimpleNamespace(
                failed=False,
                object_poses={1: settled},
                diagnostics={"simulation_settled": True},
            )

    captured = []

    def build_problem(state, action, *, pick_pose, metadata):
        del state
        captured.append(action.copy())
        return PlacementPlanningProblem(
            "world",
            action,
            pick_pose,
            action.pose,
            (
                SceneObjectInstance(
                    0,
                    1,
                    pick_pose,
                    SceneMobility.KINEMATIC,
                ),
            ),
            metadata,
        )

    planner = SimpleNamespace(
        plan=lambda problem: PlacementPlanningResult(
            PlanningStatus.INFEASIBLE,
            reason="ik_failure",
            diagnostics={"retryable": True},
        )
    )
    coordinator = TabletopPlanningCoordinator(
        Inventory(),
        build_problem,
        planner,
        placement_backend=FixedBackend(),
    )
    action = Action(
        0,
        _pose(0.1, 0.0, 0.2),
        _pose(),
        solved_pose=_pose(0.2, 0.0, 0.2),
        object_id=1,
    )
    evaluated = State([1], [0], {1: _pose(0.3, 0.0, 0.15)}, [], [])

    record = coordinator.attempt(
        _state((1,)),
        action,
        step=1,
        attempt=1,
        evaluated_state=evaluated,
    )

    assert np.allclose(captured[0].pose, _pose(0.3, 0.0, 0.14))
    assert np.allclose(record.problem.place_pose, _pose(0.3, 0.0, 0.14))
    assert record.problem.metadata["fixed_scene_resettle_enabled"]
    assert record.result.diagnostics["motion_planning_attempted"] is True


def test_tabletop_active_floor_rejects_floating_and_disconnected_poses() -> None:
    half = np.array([0.01, 0.02, 0.01])
    points = np.asarray(
        [
            [x, y, z]
            for x in (-half[0], half[0])
            for y in (-half[1], half[1])
            for z in (-half[2], half[2])
        ]
    )
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ]
    )
    geometry = ConvexGeometry(points, faces)

    class StackObject:
        geometries = (geometry,)

        def mesh_arrays(self):
            return points, faces

    class Inventory:
        target = SimpleNamespace(
            origin=np.zeros(3),
            width=0.08,
            length=0.17,
        )

        def __getitem__(self, index):
            assert index in (0, 1)
            return StackObject()

    safety = TabletopPlacementSafety(Inventory(), bottom_tolerance=.02, fill_ratio=.9,
        grid_size=32, coverage_mode='bottom_slice')
    parent = State(
        [1, 2],
        [0],
        {1: _pose(-0.025, 0.0, 0.01)},
        [],
        [],
    )
    floating = Action(1, _pose(0.0, 0.0, 0.08), _pose(), object_id=2)
    disconnected = Action(1, _pose(0.025, 0.06, 0.01), _pose(), object_id=2)

    floating_reason, _ = safety.active_floor_rejection(parent, floating)
    disconnected_reason, _ = safety.active_floor_rejection(parent, disconnected)

    assert floating_reason == "above_active_layer"
    assert disconnected_reason == "disconnected_active_layer"


def test_operations_controller_projects_selected_plan_sample() -> None:
    class Projector:
        def __init__(self):
            self.calls = []

        def project(self, positions, opening, *, geometry_mode="visual"):
            assert geometry_mode == "visual"
            self.calls.append((np.asarray(positions), opening))
            return ("robot-frame",)

    projector = Projector()
    record = PlacementPlanRecord(
        1,
        1,
        _problem(),
        _success_result(),
        candidate_id="step-1-candidate-1",
    )
    controller = OperationsController(trajectory_projector=projector)
    controller.set_artifact(_artifact(), planning_records=(record,))
    controller.select_plan_segment(0)
    controller.set_plan_sample(2)

    frame = controller.planning_frame()

    assert frame.robot_parts == ("robot-frame",)
    assert np.array_equal(
        projector.calls[-1][0],
        _success_result().segments[0].positions[2],
    )
    assert projector.calls[-1][1] == 0.1


def _native_fixture():
    application = TabletopApplication.from_repository(ROOT)
    initial = {1: _pose(0.0, -0.7, 0.2)}
    inventory = application.load_inventory((1,), initial)
    solver = load_native_solver_profile(
        application.solver_profiles.grasp,
        application.solver_profiles.motion,
    )
    backend = TabletopNativePlanningBackend(
        inventory,
        application.planning_profile,
        solver,
        ROOT,
    )
    action = Action(0, _pose(-0.55, 0.0, 0.2), _pose(), object_id=1)
    problem = application.build_placement_planning_problem(
        _state((1,)),
        action,
        pick_pose=initial[1],
    )
    return application, inventory, backend, problem
