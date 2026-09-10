from pathlib import Path

import numpy as np
import pytest

from stacking_common.domain import Action, State
from stacking_common.planning import (
    FrameTransform,
    MotionPlanSegment,
    PlacementPlanningProblem,
    PlacementPlanningResult,
    PlanningStatus,
    SceneMobility,
)
from stacking_tabletop.config import load_planning_profile


ROOT = Path(__file__).resolve().parents[1]


def test_pick_scene_normalization_preserves_repeated_checkpoint_copies():
    from stacking_common.planning import PickSceneSnapshot
    # This recorded orientation oscillated by one ULP under repeated division.
    pose = np.array([0., 0., 0., .5392192987778156, -.5540334524286524,
                     -.29579898436322427, .5610636704200641])
    scene = PickSceneSnapshot('world', {12: pose}, 'scan')
    np.testing.assert_array_equal(scene.pose(12), pose)
    scaled = pose.copy()
    scaled[3:] *= 2
    normalized = PickSceneSnapshot('world', {12: scaled}, 'scan')
    assert np.linalg.norm(normalized.pose(12)[3:]) == pytest.approx(1.)
    original = normalized.artifact_metadata()
    for _ in range(20):
        normalized = normalized.select([12])
        assert normalized.artifact_metadata() == original


def _pose(x=0.0, y=0.0, z=0.0, yaw_degrees=0.0):
    half = np.deg2rad(yaw_degrees) / 2.0
    return np.array([x, y, z, 0.0, 0.0, np.sin(half), np.cos(half)])


def _state():
    return State(
        object_ids=[10, 20],
        placed_object_indices=[0],
        object_poses={10: _pose(0.1, 0.0, 0.02)},
        trajectories=[],
        action_history=[],
    )


def test_frame_transform_round_trip_includes_rotation() -> None:
    world_T_stack = FrameTransform(
        "world",
        "stack",
        _pose(1.0, 2.0, 3.0, yaw_degrees=90.0),
    )
    stack_T_object = _pose(0.2, 0.0, 0.4, yaw_degrees=-30.0)

    world_T_object = world_T_stack.child_pose_to_parent(stack_T_object)
    recovered = world_T_stack.parent_pose_to_child(world_T_object)

    assert np.allclose(world_T_object[:3], [1.0, 2.2, 3.4])
    assert np.allclose(recovered, stack_T_object)


def test_search_action_becomes_world_frame_planning_problem() -> None:
    world_T_stack = FrameTransform("world", "stack", _pose(-0.55, 0.0, 0.115))
    action = Action(
        object_index=1,
        object_id=20,
        pose=_pose(0.0, 0.0, 0.08),
        init_pose=_pose(0.0, 0.0, 0.1),
        solved_pose=_pose(0.01, 0.0, 0.075),
    )
    pick_pose = _pose(0.2, -0.7, 0.12)

    problem = PlacementPlanningProblem.from_search(
        _state(),
        action,
        pick_pose=pick_pose,
        world_T_stack=world_T_stack,
    )

    assert problem.frame_id == "world"
    assert np.allclose(problem.pick_pose, pick_pose)
    assert np.allclose(problem.place_pose[:3], [-0.55, 0.0, 0.195])
    by_id = {value.object_id: value for value in problem.scene_objects}
    assert np.allclose(by_id[10].pose[:3], [-0.45, 0.0, 0.135])
    assert by_id[10].mobility is SceneMobility.STATIC
    assert by_id[20].mobility is SceneMobility.KINEMATIC
    assert np.allclose(by_id[20].pose, pick_pose)


def test_search_problem_separates_pick_inventory_from_placed_scene() -> None:
    world_T_stack = FrameTransform("world", "stack", _pose(-0.55, 0.0, 0.115))
    state = State(
        object_ids=[10, 20, 30],
        placed_object_indices=[0],
        object_poses={10: _pose(0.1, 0.0, 0.02)},
        trajectories=[],
        action_history=[],
    )
    action = Action(1, _pose(0.0, 0.0, 0.08), _pose(), object_id=20)
    pick_pose = _pose(0.2, -0.7, 0.12)

    problem = PlacementPlanningProblem.from_search(
        state,
        action,
        pick_pose=pick_pose,
        pick_scene_poses={
            20: pick_pose,
            30: _pose(-0.2, -0.7, 0.13),
        },
        world_T_stack=world_T_stack,
    )

    assert [value.object_id for value in problem.pick_scene_objects] == [20, 30]
    assert [value.object_id for value in problem.place_scene_objects] == [10, 20]
    assert problem.pick_scene_objects[0].mobility is SceneMobility.KINEMATIC
    assert problem.pick_scene_objects[1].mobility is SceneMobility.STATIC
    assert problem.place_scene_objects[0].mobility is SceneMobility.STATIC


def test_tabletop_planning_profile_preserves_legacy_boundary_transform() -> None:
    profile = load_planning_profile(ROOT / "apps/tabletop/config/planning.yml")

    assert profile.world_T_stack.parent_frame == "world"
    assert profile.world_T_stack.child_frame == "stack"
    assert np.allclose(
        profile.world_T_stack.parent_T_child,
        [-0.55, 0.0, 0.115, 0.0, 0.0, 0.0, 1.0],
    )


def test_planning_result_has_typed_success_and_failure_shapes() -> None:
    segment = MotionPlanSegment(
        "transfer",
        positions=np.zeros((2, 6)),
        times=np.array([0.0, 1.0]),
        openings=np.array([0.7, 0.1]),
    )
    success = PlacementPlanningResult(
        PlanningStatus.SUCCESS,
        segments=(segment,),
        score=1.5,
    )
    failure = PlacementPlanningResult(
        PlanningStatus.INFEASIBLE,
        reason="collision",
    )

    assert success.succeeded
    assert np.allclose(success.segments[0].openings, [0.7, 0.1])
    assert not failure.succeeded
    with pytest.raises(ValueError, match="requires motion segments"):
        PlacementPlanningResult(PlanningStatus.SUCCESS)
    with pytest.raises(ValueError, match="openings must align"):
        MotionPlanSegment(
            "transfer",
            positions=np.zeros((2, 6)),
            openings=np.zeros(3),
        )
