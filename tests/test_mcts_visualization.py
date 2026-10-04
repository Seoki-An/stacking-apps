import numpy as np
import pytest

from stacking_common.domain import (
    Action,
    ContactPoint,
    StackObjectTrajectory,
    State,
)
from stacking_common.search import MCTSNode
from stacking_common.visualization import (
    CandidateOutcome,
    CandidatePoseMode,
    ScoreField,
    project_mcts_step,
)


def _pose(x: float, z: float) -> np.ndarray:
    return np.array([x, 0.0, z, 0.0, 0.0, 0.0, 1.0])


def _action(index: int, object_id: int, x: float) -> Action:
    return Action(
        object_index=index,
        object_id=object_id,
        init_pose=_pose(x, 0.10),
        solved_pose=_pose(x, 0.07),
        pose=_pose(x, 0.065),
    )


def _trajectory(object_id: int, *poses: np.ndarray) -> StackObjectTrajectory:
    return StackObjectTrajectory(object_id, list(poses))


def _root_state() -> State:
    return State(
        object_ids=[11, 22, 33],
        placed_object_indices=[0],
        object_poses={11: _pose(0.0, 0.025)},
        trajectories=[],
        action_history=[],
    )


def _evaluated_state(
    action: Action,
    *,
    failed: bool = False,
    move_scene: bool = False,
) -> State:
    scene_final = _pose(0.015 if move_scene else 0.001, 0.025)
    target_final = _pose(float(action.pose[0]) + 0.001, 0.061)
    return State(
        object_ids=[11, 22, 33],
        placed_object_indices=[0, action.object_index],
        object_poses={11: scene_final, action.object_id: target_final},
        trajectories=[
            {
                11: _trajectory(
                    11,
                    _pose(0.0, 0.025),
                    scene_final,
                ),
                action.object_id: _trajectory(
                    action.object_id,
                    action.pose,
                    target_final,
                ),
            }
        ],
        action_history=[action],
        failed=failed,
        contact_points=[
            ContactPoint(
                entity_id_first=2,
                entity_id_second=1,
                object_index_first=action.object_index,
                object_index_second=0,
                ground_first=False,
                ground_second=False,
                point_first=[0.0, 0.0, 0.04],
                point_second=[0.0, 0.0, 0.04],
                normal=[0.0, 0.0, 1.0],
                gap=-0.0001,
            )
        ],
    )


@pytest.mark.parametrize("compact", [False, True])
def test_projection_uses_existing_tree_state_without_mutating_search(compact, monkeypatch) -> None:
    if compact:
        monkeypatch.setitem(globals(), '_trajectory', lambda object_id, *poses:
            StackObjectTrajectory.from_samples(object_id, poses, np.arange(len(poses)) * .1))
    root = MCTSNode(state=_root_state(), observation="root")
    selected_action = _action(1, 22, 0.04)
    selected = root.add_candidate(
        selected_action,
        prior=0.8,
        info={
            "candidate_id": "selected",
            "cem_score": 1.25,
            "posegen_contact_points": {"target": [0.04, 0.0, 0.04]},
            "posegen_contact_normals": {"target": [0.0, 0.0, 1.0]},
            "posegen_contact_forces": {"target": [0.0, 0.0, 4.5]},
        },
    )
    selected.set_result(
        _evaluated_state(selected_action),
        "selected observation",
        1.75,
        info=selected.info,
    )
    selected.record_visit(2.4)
    selected.record_visit(2.4)

    failed_action = _action(2, 33, -0.02)
    failed = root.add_candidate(failed_action, prior=0.6)
    failed.set_result(
        _evaluated_state(failed_action, failed=True, move_scene=True),
        "failed observation",
        0.0,
        failed=True,
        rejection_reason="scene_motion_limit",
    )
    root.add_candidate(_action(1, 22, 0.05), prior=0.45)
    root.reject_expansion("posegen_gap", count=2)
    score_field = ScoreField(
        "state-score",
        [-0.05, 0.0, 0.05],
        [-0.04, 0.0],
        [[0.1, 0.6, 0.2], [0.2, 0.9, 0.4]],
        [[True, True, False], [True, True, True]],
        [[0.0, 0.0]],
    )

    step = project_mcts_step(
        root,
        selected,
        step=1,
        score_field=score_field,
    )

    assert step.selected_candidate_id == "selected"
    assert [value.outcome for value in step.candidates] == [
        CandidateOutcome.SUCCEEDED,
        CandidateOutcome.FAILED,
        CandidateOutcome.PENDING,
    ]
    assert step.rejected_expansions == {"posegen_gap": 2}
    assert step.scene.objects[0].object_id == 11
    projected = step.candidates[0]
    assert projected.score == 1.25
    assert projected.visits == 2
    assert np.isclose(projected.mean_value, 2.4)
    assert projected.timeline.frame_count == 2
    assert len(projected.simulation_contacts) == 1
    assert len(projected.pose_solve_forces) == 1
    assert np.allclose(
        projected.pose(CandidatePoseMode.SIMULATION_START),
        selected_action.pose,
    )
    assert step.candidates[1].scene_motion.object_id == 11
    assert len(root.children) == 3
    assert root.children[0] is selected


def test_projection_reports_empty_root_without_inventing_rejected_geometry() -> None:
    root = MCTSNode(state=_root_state(), observation="root")
    root.reject_expansion("outside_target", count=3)

    step = project_mcts_step(root, None, step=2)

    assert not step.succeeded
    assert step.failure_reason == "no_candidates"
    assert step.candidates == ()
    assert step.rejected_expansions == {"outside_target": 3}


def test_projection_flattens_native_batched_posegen_contacts() -> None:
    root = MCTSNode(state=_root_state(), observation="root")
    action = _action(1, 22, 0.04)
    child = root.add_candidate(
        action,
        info={
            "posegen_contact_points": {
                7: [[0.04, 0.00, 0.04], [0.05, 0.00, 0.04]],
            },
            "posegen_contact_normals": {
                7: [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            },
            "posegen_contact_forces": {
                7: [[0.0, 0.0, 4.5], [0.0, 1.2, 0.0]],
            },
        },
    )
    child.set_result(
        _evaluated_state(action),
        "observation",
        1.0,
        info=child.info,
    )

    projected = project_mcts_step(root, child, step=1).candidates[0]

    assert len(projected.pose_solve_forces) == 2
    assert np.allclose(
        projected.pose_solve_forces[1].point,
        [0.05, 0.0, 0.04],
    )
    assert np.allclose(
        projected.pose_solve_forces[1].force,
        [0.0, 1.2, 0.0],
    )


def test_projection_skips_incomplete_posegen_force_rows() -> None:
    root = MCTSNode(state=_root_state(), observation="root")
    action = _action(1, 22, 0.04)
    child = root.add_candidate(
        action,
        info={
            "posegen_contact_points": {
                7: [[0.04, 0.00, 0.04], [0.05, 0.00, 0.04]],
                8: [[0.01, 0.00, 0.02]],
            },
            "posegen_contact_normals": {
                7: [[0.0, 0.0, 1.0], [np.nan, 0.0, 1.0]],
                8: [],
            },
            "posegen_contact_forces": {
                7: [[0.0, 0.0, 4.5]],
                8: [[0.0, 0.0, 1.0]],
            },
        },
    )
    child.set_result(
        _evaluated_state(action),
        "observation",
        1.0,
        info=child.info,
    )

    projected = project_mcts_step(root, child, step=1).candidates[0]

    assert len(projected.pose_solve_forces) == 1
    assert np.allclose(projected.pose_solve_forces[0].force, [0.0, 0.0, 4.5])
