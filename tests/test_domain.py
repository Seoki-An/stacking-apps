import numpy as np
import pytest

from stacking_common.artifacts import decode_action, encode_action
from stacking_common.domain import (
    Action,
    ActionList,
    ContactPoint,
    StackObjectTrajectory,
    State,
)


def pose(x: float = 0.0) -> np.ndarray:
    return np.array([x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def test_action_defaults_to_an_independent_solved_pose() -> None:
    action = Action(
        object_index=1,
        object_id=7,
        pose=pose(1.0),
        init_pose=pose(2.0),
    )

    assert np.array_equal(action.solved_pose, action.pose)
    assert action.solved_pose is not action.pose


def test_action_copy_owns_mutable_values() -> None:
    action = Action(
        object_index=1,
        object_id=7,
        pose=pose(1.0),
        init_pose=pose(2.0),
        diagnostics={"source": {"name": "posegen"}},
    )

    copied = action.copy()
    copied.pose[0] = 3.0
    copied.solved_pose[0] = 4.0
    copied.diagnostics["source"]["name"] = "copy"

    assert action.pose[0] == 1.0
    assert action.solved_pose[0] == 1.0
    assert action.diagnostics == {"source": {"name": "posegen"}}


def test_action_validates_and_owns_input_poses() -> None:
    source = pose(1.0)
    action = Action(0, source, pose(2.0))
    source[0] = 3.0

    assert action.pose[0] == 1.0
    with pytest.raises(ValueError, match="shape"):
        Action(0, np.zeros(6), pose())
    with pytest.raises(ValueError, match="quaternion"):
        Action(0, np.zeros(7), pose())


def test_action_artifact_conversion_and_masking() -> None:
    first = decode_action(
        {
            "object_index": 0,
            "object_id": 4,
            "pose": pose(1.0),
            "init_pose": pose(2.0),
            "future_metadata": "ignored",
        }
    )
    second = Action(
        object_index=1,
        object_id=5,
        pose=pose(3.0),
        init_pose=pose(4.0),
    )
    actions = ActionList([first, second])

    assert actions.mask(np.array([0.0, -np.inf])) == [first]
    arrays = actions.get_dict_array()
    assert arrays["pose"].shape == (2, 7)
    assert arrays["object_id"].tolist() == [4, 5]
    encoded = encode_action(first)
    assert set(encoded) == {
        "object_index",
        "object_id",
        "pose",
        "solved_pose",
        "init_pose",
        "c_feq",
        "c_gap",
        "place_robustness_displacement",
        "diagnostics",
    }
    encoded["pose"][0] = 9.0
    assert first.pose[0] == 1.0
    with pytest.raises(ValueError, match="one value per action"):
        actions.mask(np.array([0.0]))


def test_action_diagnostic_entity_keys_are_serialized_without_mutating_domain():
    action = Action(0, pose(0.), pose(0.), object_id=7, diagnostics={
        "contacts": [{np.int64(5): np.array([1., 2., 3.])}],
    })
    encoded = encode_action(action)
    assert set(encoded["diagnostics"]["contacts"][0]) == {"5"}
    encoded["diagnostics"]["contacts"][0]["5"][0] = 99.
    assert action.diagnostics["contacts"][0][5][0] == 1.


def test_action_diagnostic_key_collisions_are_not_silently_overwritten():
    action = Action(0, pose(0.), pose(0.), object_id=7,
                    diagnostics={"contacts": {5: "first", "5": "second"}})
    with pytest.raises(ValueError, match="duplicate serialized"):
        encode_action(action)


def test_stack_object_trajectory_tracks_object_and_sequence() -> None:
    trajectory = StackObjectTrajectory(7)
    trajectory.add_pose(pose(1.0))
    trajectory.add_pose(pose(2.0))

    assert trajectory.object_id == 7
    assert len(trajectory) == 2
    assert trajectory[0][0] == 1.0
    assert [value[0] for value in trajectory] == [1.0, 2.0]


def test_state_copy_and_latest_velocity_integrals() -> None:
    action = Action(
        object_index=0,
        object_id=7,
        pose=pose(1.0),
        init_pose=pose(2.0),
    )
    trajectory = StackObjectTrajectory(
        7, poses=[pose(1.0)], vel_integral=0.125
    )
    contact = ContactPoint(
        entity_id_first=1,
        entity_id_second=0,
        object_index_first=0,
        object_index_second=None,
        ground_first=False,
        ground_second=True,
        point_first=np.zeros(3),
        point_second=np.zeros(3),
        normal=np.array([0.0, 0.0, 1.0]),
        gap=0.0,
    )
    state = State(
        object_ids=[7, 8],
        placed_object_indices=[0],
        object_poses={7: pose(1.0)},
        trajectories=[{7: trajectory}],
        action_history=[action],
        contact_points=[contact],
        pose_identified_object_ids={7},
    )

    copied = state.copy()
    copied.object_ids.append(9)
    copied.object_poses[7][0] = 4.0
    copied.trajectories[0][7].poses[0][0] = 5.0
    copied.action_history[0].pose[0] = 3.0

    assert state.object_ids == [7, 8]
    assert state.object_poses[7][0] == 1.0
    assert state.trajectories[0][7].poses[0][0] == 1.0
    assert state.action_history[0].pose[0] == 1.0
    with pytest.raises(ValueError, match="read-only"):
        copied.contact_points[0].point_first[0] = 1.0
    assert state.latest_velocity_integrals() == {7: 0.125}
