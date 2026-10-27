from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stacking_common.artifacts import PlacementPlanRecord, PlanningCheckpoint
from stacking_common.domain import Action, State
from stacking_common.domain import ContactPoint, StackObjectTrajectory
from stacking_common.planning import (
    FrameTransform,
    PickSceneSnapshot,
    MotionPlanSegment,
    PlacementPlanningProblem,
    PlacementPlanningResult,
    PlanningStatus,
    SceneMobility,
    SceneObjectInstance,
)
from stacking_common.search import MCTSNode, MCTSSearchRequest, MCTSSession
from stacking_common.workflows import (
    MCTSPlanningSequenceCoordinator,
    PlanningSequenceStatus,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("platform", ["excavator", "tabletop"])
def test_two_step_sequence_rebuilds_committed_observation_and_reward(platform):
    from test_action_generation import _inventory
    from stacking_excavator import ExcavatorApplication
    from stacking_tabletop import TabletopApplication

    app_type = ExcavatorApplication if platform == "excavator" else TabletopApplication
    app = app_type.from_repository(ROOT)
    inventory = _inventory()
    evaluate = app.build_committed_state_evaluator(inventory)
    environment = app.build_environment(inventory, object())
    initial = State(list(inventory.object_ids), [], {}, [], [])
    root = MCTSNode(state=initial, observation=environment.observation_builder.build(initial))
    simulated_observations = []
    simulated_rewards = []
    checkpoints = []
    old_descendants = []

    class Search:
        def search(self, root, request):
            del request
            index = len(root.state.placed_object_indices)
            expected = environment.observation_builder.build(root.state)
            np.testing.assert_array_equal(root.observation.object_poses, expected.object_poses)
            np.testing.assert_array_equal(root.observation.scene_height_map, expected.scene_height_map)
            if index == 1:
                assert root.state.object_poses[10][0] == .2
                assert np.count_nonzero(root.observation.scene_height_map !=
                                        simulated_observations[0].scene_height_map) > 0
                assert root.reward == pytest.approx(evaluate(root.state)[1])
            action = Action(index, _pose(z=.02), _pose(z=.02), object_id=inventory[index].object_id)
            state = root.state.copy()
            state.placed_object_indices.append(index)
            state.object_poses[action.object_id] = action.pose.copy()
            state.action_history.append(action.copy())
            state.terminated = index == 1
            if index == 1:
                # The long simulation also displaced a previously placed stone.
                state.object_poses[10][0] = .45
            state.contact_points = [ContactPoint(0, 1, index, None, False, True,
                                                np.zeros(3), np.zeros(3), [0, 0, 1], 0.)]
            state.trajectories.append({action.object_id: StackObjectTrajectory(
                action.object_id, [action.pose.copy()])})
            observation, reward, info = evaluate(state)
            simulated_observations.append(observation)
            simulated_rewards.append(reward)
            child = root.add_candidate(action)
            child.set_result(state, observation, reward, info={**info, "authoritative_fidelity": "long"})
            if index == 0:
                old_descendants.append(child.add_candidate(
                    Action(1, _pose(z=.04), _pose(z=.04), object_id=inventory[1].object_id)))
            return SimpleNamespace(root=root, selected=child, ranked_candidates=(child,),
                completed_iterations=1, generated_candidates=1, evaluated_candidates=1, failed_evaluations=0)

    class AdjustingPlanner:
        def attempt(self, state, action, *, step, attempt, evaluated_state, candidate_id, writer):
            planned = action.copy()
            planned.pose[0] = .2 if step == 1 else .04
            problem = PlacementPlanningProblem("world", planned, _pose(), planned.pose, ())
            record = PlacementPlanRecord(step, attempt, problem,
                PlacementPlanningResult(PlanningStatus.SUCCESS,
                    segments=(MotionPlanSegment("transfer", np.zeros((2, 6))),)), candidate_id)
            writer.append_placement_plan(record)
            return record

    class Writer(_Writer):
        def append_mcts_step(self, step, *, statistics=None, checkpoint=None):
            super().append_mcts_step(step, statistics=statistics)
            checkpoints.append(checkpoint)

    writer = Writer()
    session = MCTSSession(Search(), root)
    result = MCTSPlanningSequenceCoordinator(session, MCTSSearchRequest(iterations=1),
        AdjustingPlanner(), frame_id="stack", writer=writer,
        committed_state_evaluator=evaluate,
        checkpoint_builder=lambda step, state: PlanningCheckpoint(step, "0" * 64, state,
            PickSceneSnapshot("world", {10: _pose(), 20: _pose()}, "recorded"),
            FrameTransform("world", "stack", _pose())),
    ).run(2)

    assert result.succeeded
    assert len(result.steps) == 2
    assert old_descendants and all(node.parent is None for node in old_descendants)
    assert result.final_state.object_poses[10][0] == .2
    assert result.final_state.object_poses[20][0] == .04
    for (debug, _), checkpoint in zip(writer.steps, checkpoints):
        _, expected_reward, _ = evaluate(checkpoint.state)
        selected = next(candidate for candidate in debug.candidates if candidate.selected)
        assert selected.reward == pytest.approx(expected_reward)
        np.testing.assert_array_equal(selected.final_pose, checkpoint.state.object_poses[selected.object_id])
        assert selected.simulation_contacts == ()
        assert selected.timeline.tracks == ()
    np.testing.assert_array_equal(session.root.observation.scene_height_map,
                                  evaluate(result.final_state)[0].scene_height_map)
    assert session.root.info["committed_contacts_available"] is False
    assert "support_count" not in session.root.info
    assert "authoritative_fidelity" not in session.root.info
    assert writer.steps[0][0].candidates[0].reward != pytest.approx(simulated_rewards[0])


def test_changed_commit_requires_evaluator_without_mutating_candidate():
    root = MCTSNode(state=_state(), observation=object())
    child = root.add_candidate(Action(0, _pose(), _pose(), object_id=1))
    child.set_result(_state(placed=True), object(), 1.)
    runner = _AttemptRunner()
    runner.calls = 1
    record = runner.attempt(root.state, child.action, step=1, attempt=1,
                            evaluated_state=_state(placed=True, x=.2))
    coordinator = MCTSPlanningSequenceCoordinator(MCTSSession(_SearchEngine(), root),
        MCTSSearchRequest(iterations=1), runner, frame_id="stack")
    with pytest.raises(ValueError, match="requires a committed-state evaluator"):
        coordinator._commit_planned_pose(root.state, child, record)
    assert child.state.object_poses[1][0] == 0.
    assert child.action.pose[0] == 0.


def test_restarted_sequence_preserves_step_numbers_and_checkpoints_committed_state():
    root = MCTSNode(state=_state(), observation=object())
    for x in (0.0, 0.5):
        child = root.add_candidate(Action(0, _pose(x), _pose(), object_id=1))
        child.set_result(_state(placed=True, x=x), object(), 1.0)

    class CheckpointWriter(_Writer):
        def append_mcts_step(self, step, *, statistics=None, checkpoint=None):
            super().append_mcts_step(step, statistics=statistics)
            self.checkpoint = checkpoint

    writer = CheckpointWriter()
    coordinator = MCTSPlanningSequenceCoordinator(
        MCTSSession(_SearchEngine(), root), MCTSSearchRequest(iterations=1),
        _AttemptRunner(), frame_id="stack", writer=writer,
        checkpoint_builder=lambda step, state: PlanningCheckpoint(
            step, "0" * 64, state,
            PickSceneSnapshot("world", {1: _pose()}, "recorded"),
            FrameTransform("world", "stack", _pose()),
        ),
    )
    result = coordinator.run(1, first_step=4)
    assert result.steps[0].step == 4
    assert writer.checkpoint.step == 4
    assert writer.checkpoint.state.placed_object_indices == [0]
    assert writer.plans[1].candidate_id == "step-4-candidate-2"
    np.testing.assert_array_equal(writer.checkpoint.state.object_poses[1], result.final_state.object_poses[1])


def _pose(x=0.0, y=0.0, z=0.0):
    return np.array([x, y, z, 0.0, 0.0, 0.0, 1.0])


def _state(*, placed=False, x=0.0):
    return State(
        [1],
        [0] if placed else [],
        {1: _pose(x, 0.0, 0.1)} if placed else {},
        [],
        [],
        terminated=placed,
    )


class _SearchEngine:
    def __init__(self):
        self.calls = 0

    def search(self, root, request):
        del request
        self.calls += 1
        selected = next(
            (value for value in root.evaluated_children() if value.selectable),
            None,
        )
        return SimpleNamespace(
            root=root,
            selected=selected,
            ranked_candidates=tuple(root.evaluated_children()),
            completed_iterations=1,
            generated_candidates=2,
            evaluated_candidates=2,
            failed_evaluations=0,
        )


class _Writer:
    def __init__(self):
        self.plans = []
        self.steps = []

    def append_placement_plan(self, record):
        self.plans.append(record)

    def append_mcts_step(self, step, *, statistics=None):
        self.steps.append((step, statistics))


class _AttemptRunner:
    def __init__(self):
        self.calls = 0

    def attempt(
        self,
        state,
        action,
        *,
        step,
        attempt,
        evaluated_state=None,
        candidate_id=None,
        writer=None,
    ):
        self.calls += 1
        assert evaluated_state is not None
        planned_action = action.copy()
        planned_action.pose = evaluated_state.object_poses[action.object_id].copy()
        problem = PlacementPlanningProblem(
            "world",
            planned_action,
            _pose(0.0, -0.7, 0.1),
            planned_action.pose,
            (
                SceneObjectInstance(
                    action.object_index,
                    action.object_id,
                    _pose(0.0, -0.7, 0.1),
                    SceneMobility.KINEMATIC,
                ),
            ),
        )
        result = (
            PlacementPlanningResult(
                PlanningStatus.INFEASIBLE,
                reason="motion_collision",
                diagnostics={"retryable": True},
            )
            if self.calls == 1
            else PlacementPlanningResult(
                PlanningStatus.SUCCESS,
                segments=(MotionPlanSegment("transfer", np.zeros((2, 6))),),
            )
        )
        record = PlacementPlanRecord(
            step,
            attempt,
            problem,
            result,
            candidate_id,
        )
        if writer is not None:
            writer.append_placement_plan(record)
        return record


def test_sequence_rejects_failed_motion_and_plans_next_mcts_candidate() -> None:
    root = MCTSNode(state=_state(), observation=object())
    first = root.add_candidate(Action(0, _pose(0.0), _pose(), object_id=1))
    first.set_result(_state(placed=True, x=0.0), object(), 1.0)
    # Nearby alternative must still be attempted within the returned batch.
    second = root.add_candidate(Action(0, _pose(0.05), _pose(), object_id=1))
    second.set_result(_state(placed=True, x=0.5), object(), 0.9)
    session = MCTSSession(_SearchEngine(), root)
    writer = _Writer()
    progress = []
    coordinator = MCTSPlanningSequenceCoordinator(
        session,
        MCTSSearchRequest(iterations=1),
        _AttemptRunner(),
        frame_id="stack",
        writer=writer,
        progress=lambda event, details: progress.append((event, details)),
    )

    result = coordinator.run(1, max_planning_attempts_per_step=2)

    assert result.status is PlanningSequenceStatus.COMPLETE
    assert result.final_state.object_poses[1][0] == 0.5
    assert len(result.steps[0].search_attempts) == 1
    # Snapshot the completed search before motion rejection and root promotion.
    search_attempt = result.steps[0].search_attempts[0]
    assert (search_attempt.tree_nodes, search_attempt.tree_leaves) == (3, 2)
    saved_attempt = writer.steps[0][1]["search_attempts"][0]
    assert (saved_attempt["tree_nodes"], saved_attempt["tree_leaves"]) == (3, 2)
    completed = next(details for event, details in progress if event == "search_completed")
    assert (completed["tree_nodes"], completed["tree_leaves"]) == (3, 2)
    assert [value.result.status for value in result.steps[0].planning_attempts] == [
        PlanningStatus.INFEASIBLE,
        PlanningStatus.SUCCESS,
    ]
    assert [value.candidate_id for value in writer.plans] == [
        "step-1-candidate-1",
        "step-1-candidate-2",
    ]
    assert writer.steps[0][0].selected_candidate_id == "step-1-candidate-2"
    assert writer.steps[0][1]["status"] == "complete"
    assert writer.plans[1].problem.action.pose[0] == 0.5
    assert result.steps[0].selected_action.pose[0] == 0.5
    failed = next(
        value
        for value in writer.steps[0][0].candidates
        if value.candidate_id == "step-1-candidate-1"
    )
    assert failed.rejection_reason == "execution_rejected_action"
    assert not session.execution_rejections.actions
    assert not session.execution_rejections.object_ids
    assert [event for event, _ in progress] == [
        "step_started",
        "search_started",
        "search_completed",
        "candidate_started",
        "candidate_completed",
        "candidate_started",
        "candidate_completed",
        "step_completed",
    ]


class _AlwaysFailRunner(_AttemptRunner):
    def attempt(self, *args, **kwargs):
        record = super().attempt(*args, **kwargs)
        return PlacementPlanRecord(
            record.step,
            record.attempt,
            record.problem,
            PlacementPlanningResult(
                PlanningStatus.INFEASIBLE,
                reason="motion_collision",
                diagnostics={"retryable": True},
            ),
            record.candidate_id,
        )


def test_sequence_defers_object_rejection_until_ranked_batch_is_tried() -> None:
    root = MCTSNode(state=_state(), observation=object())
    for x in (0.0, 0.05, 0.1):
        action = Action(0, _pose(x), _pose(x), object_id=1)
        child = root.add_candidate(action)
        child.set_result(_state(placed=True, x=x), object(), 1.0 - x)
    engine = _SearchEngine()
    session = MCTSSession(engine, root)
    coordinator = MCTSPlanningSequenceCoordinator(
        session,
        MCTSSearchRequest(iterations=1),
        _AlwaysFailRunner(),
        frame_id="stack",
    )

    result = coordinator.run(1, max_planning_attempts_per_step=4)

    assert result.status is PlanningSequenceStatus.SEARCH_EXHAUSTED
    assert len(result.steps[0].planning_attempts) == 3
    assert len(result.steps[0].search_attempts) == 2
    assert session.execution_rejections.object_ids == frozenset({1})


def test_sequence_reports_skipped_candidates_and_search_exclusions():
    class Search(_SearchEngine):
        def search(self, root, request):
            result = super().search(root, request)
            result.execution_rejected_candidates = 3
            result.ranked_candidates = tuple(root.children)
            return result

    root = MCTSNode(state=_state(), observation=object())
    for x in (0.0, 0.05, 0.1):
        child = root.add_candidate(Action(0, _pose(x), _pose(x), object_id=1))
        child.set_result(_state(placed=True, x=x), object(), 1.0)
    root.children[0].reject("simulation_unsettled")
    progress = []
    writer = _Writer()
    result = MCTSPlanningSequenceCoordinator(
        MCTSSession(Search(), root), MCTSSearchRequest(iterations=1),
        _AttemptRunner(), frame_id="stack", writer=writer,
        progress=lambda event, details: progress.append((event, details)),
    ).run(1)
    assert result.succeeded
    skipped = [details for event, details in progress if event == "candidate_skipped"]
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "simulation_unsettled"
    assert writer.steps[0][1]["skipped_candidates"] == skipped
    assert writer.steps[0][1]["search_attempts"][0]["execution_rejected_candidates"] == 3


def test_pre_motion_rejections_do_not_exhaust_object_failure_budget():
    from dataclasses import replace

    class PreparationFailureRunner(_AttemptRunner):
        def attempt(self, *args, **kwargs):
            record = super().attempt(*args, **kwargs)
            if self.calls <= 2:
                return replace(record, result=PlacementPlanningResult(
                    PlanningStatus.INFEASIBLE,
                    reason="active_floor_above_active_layer" if self.calls == 1 else "fixed_scene_resettle_invalid",
                    diagnostics={"motion_planning_attempted": False},
                ))
            return record

    root = MCTSNode(state=_state(), observation=object())
    for x in (0.0, 0.5, 1.0):
        child = root.add_candidate(Action(0, _pose(x), _pose(x), object_id=1))
        child.set_result(_state(placed=True, x=x), object(), 1.0 - x)
    session = MCTSSession(_SearchEngine(), root)
    result = MCTSPlanningSequenceCoordinator(
        session, MCTSSearchRequest(iterations=1), PreparationFailureRunner(), frame_id="stack",
    ).run(1, max_planning_attempts_per_step=4)
    assert result.status is PlanningSequenceStatus.COMPLETE
    assert len(result.steps[0].planning_attempts) == 3
    assert not session.execution_rejections.object_ids


def test_empty_search_preserves_cem_rejections_in_progress_and_stack_statistics():
    from stacking_common.actions.cem import CEMStatistics, CEMRoundStatistics
    statistics = CEMStatistics(
        rounds=(CEMRoundStatistics(0, 96, 0, 96, 0, 0, fresh_attempts=96),
                CEMRoundStatistics(1, 64, 0, 64, 0, 0, fresh_attempts=64)),
        rejection_counts={"stone_outside_target": 100, "active_floor_above_active_layer": 60},
        fresh_exhausted=True, next_fresh_offset=160,
    )
    engine = _SearchEngine()
    engine.root_action_generator = SimpleNamespace(statistics_history=[statistics])
    writer = _Writer()
    events = []
    result = MCTSPlanningSequenceCoordinator(
        MCTSSession(engine, MCTSNode(state=_state(), observation=object())),
        MCTSSearchRequest(iterations=1), _AttemptRunner(), frame_id="stack",
        writer=writer, progress=lambda event, details: events.append((event, details)),
    ).run(1)
    assert result.status is PlanningSequenceStatus.SEARCH_EXHAUSTED
    assert result.steps[0].search_attempts[0].cem_statistics == (statistics,)
    persisted = writer.steps[0][1]["search_attempts"][0]["cem_statistics"][0]
    assert persisted["next_fresh_offset"] == 160
    assert persisted["rejection_counts"]["stone_outside_target"] == 100
    completed = next(details for event, details in events if event == "search_completed")
    assert completed["cem_statistics"][0] == persisted


def test_sequence_records_backend_failure_without_planning_or_search_retry():
    from stacking_common.errors import BackendCapacityError
    failure = BackendCapacityError("posegen", "contacts", 700, 512).diagnostics

    class Engine(_SearchEngine):
        def search(self, root, request):
            result = super().search(root, request)
            result.backend_failure = failure
            return result
    engine = Engine()
    root = MCTSNode(state=_state(), observation=object())
    writer = _Writer()
    coordinator = MCTSPlanningSequenceCoordinator(MCTSSession(engine, root),
        MCTSSearchRequest(iterations=1), _AttemptRunner(), writer=writer, frame_id="stack")
    result = coordinator.run(2, max_planning_attempts_per_step=4)
    assert engine.calls == 1
    assert result.status is PlanningSequenceStatus.SEARCH_BACKEND_FAILED
    assert result.reason == failure["message"]
    assert not writer.plans
    assert result.steps[0].search_attempts[0].backend_failure == failure
    assert result.steps[0].debug_step.failure_reason == "search_backend_failed"
