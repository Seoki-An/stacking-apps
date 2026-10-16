"""Execution-history preservation for local scene revisions; fake motion only."""

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from stacking_common.artifacts import StackRunStore, PlanningCheckpoint, build_execution_command
from stacking_common.domain import State
from stacking_common.planning import PickSceneSnapshot
from stacking_tabletop.execution.scene_refinement import pending_scene_refinement, write_refined_branch
from stacking_tabletop.execution.saved_plan import load_execution_run, validate_command
from stacking_tabletop.execution.sequence import execute_saved_run
from test_tabletop_saved_execution import ROOT, saved, executor_for


@pytest.fixture
def observed(saved, tmp_path):
    artifact = StackRunStore.load_mcts(saved.path)
    assets = dict(saved.context.asset_hashes)
    assets['object/2'] = hashlib.sha256((ROOT/'apps/tabletop/assets/objects/model_2.obj').read_bytes()).hexdigest()
    assets['object/3'] = hashlib.sha256((ROOT/'apps/tabletop/assets/objects/model_3.obj').read_bytes()).hexdigest()
    context = replace(saved.context, asset_hashes=assets)
    plans = []
    states = []
    for step, old in enumerate(saved.plans, 1):
        action = old.problem.action.copy()
        action.object_id, action.object_index = step, step - 1
        plans.append(replace(old, context_sha256=context.sha256,
                             problem=replace(old.problem, action=action)))
        states.append(State([1, 2, 3], list(range(step)),
                            {i: action.pose.copy() for i in range(1, step + 1)}, [], []))
    pickup = PickSceneSnapshot('world', {1: plans[0].problem.pick_pose,
                                        2: plans[1].problem.pick_pose,
                                        3: plans[1].problem.pick_pose}, 'test')
    source = tmp_path/'source.stack'
    with StackRunStore.create_mcts(source, replace(artifact, steps=())) as writer:
        writer.append_run_context(context)
        for debug, plan, state in zip(artifact.steps, plans, states):
            selected = next(item for item in debug.candidates if item.selected)
            selected = replace(selected, object_id=plan.problem.action.object_id,
                               object_index=plan.problem.action.object_index, score_field_id=None)
            writer.append_placement_plan(plan)
            writer.append_mcts_step(replace(debug, candidates=(selected,), score_field=None),
                checkpoint=PlanningCheckpoint(plan.step, context.sha256, state, pickup,
                                               saved.planning.world_T_stack))
    current = load_execution_run(source, ROOT)
    capture = tmp_path/'capture'
    capture.mkdir()
    report = capture/'sceneid_result.json'
    observed_pose = current.planning.world_T_stack.child_pose_to_parent(states[0].object_poses[1])
    observed_pose[0] += .00639
    report.write_text(json.dumps(dict(kind='tabletop_stack_scene_identification',
        status='SceneIdStatus.IDENTIFIED', accepted_snapshot=True, support_warning=False,
        entity_ids=[1], proposed_poses=[observed_pose.tolist()],
        diagnostics={'nonpenetrating': True})))
    pcd = capture/'observed.pcd'
    pcd.write_text('fake observation; numerical fitting is covered separately')
    report_hash = hashlib.sha256(report.read_bytes()).hexdigest()
    pcd_hash = hashlib.sha256(pcd.read_bytes()).hexdigest()
    _, executor = executor_for(current)
    def finish(ex):
        status = dict(kind='stacked_scene_wrist_capture', status='complete', step=1,
                      context_sha256=context.sha256, command_id=ex.active_command_id,
                      robot_home=True, pcd=str(pcd), pcd_sha256=pcd_hash,
                      scene_identification=dict(path=str(report), sha256=report_hash))
        (capture/'capture.json').write_text(json.dumps(status))
        request = dict(capture_directory=str(capture), step=1,
                       sceneid_sha256=report_hash, pcd_sha256=pcd_hash)
        ex._emit('held', scene_refinement_required=request)
        ex.pending_scene_refinement = request
    executor.stack_scene_handler = lambda ex: SimpleNamespace(finish=finish, abort=lambda: None)
    results = execute_saved_run(current, executor)
    assert len(results) == 1 and results[0].status.value == 'succeeded'
    corrected = states[0].copy()
    corrected.object_poses[1][0] += .002
    following = states[1].copy()
    following.object_poses[1] = corrected.object_poses[1].copy()
    refinement = SimpleNamespace(corrected_state=corrected, records=(plans[1],),
                                  states=(following,), summary={'max_translation_m': .002})
    return current, capture, refinement


def test_revision_preserves_completed_prefix_and_never_replays_first_stone(observed, tmp_path):
    current, capture, refinement = observed
    assert pending_scene_refinement(current.path)['step'] == 1
    before_commands = StackRunStore.load_execution_commands(current.path)
    before_events = StackRunStore.load_execution_events(current.path)
    destination = write_refined_branch(current, ROOT, capture, 1, refinement,
                                        destination=tmp_path/'refined.stack')
    revised = load_execution_run(destination, ROOT)
    assert pending_scene_refinement(destination) is None
    assert pending_scene_refinement(current.path) is not None
    checkpoint = StackRunStore.load_execution_checkpoint(destination)
    assert checkpoint.completed_steps == (1,) and checkpoint.pending_steps == (2,)
    assert [v.command_id for v in StackRunStore.load_execution_commands(destination)] == [v.command_id for v in before_commands]
    assert [v.diagnostics for v in StackRunStore.load_execution_events(destination)] == [v.diagnostics for v in before_events]
    np.testing.assert_allclose(StackRunStore.load_planning_checkpoint(destination, 1).state.object_poses[1],
                               refinement.corrected_state.object_poses[1])
    selected = []
    _, executor = executor_for(revised)
    execute_saved_run(revised, executor, next_step=lambda plan: selected.append(plan.problem.action.object_id))
    assert selected == [2]
    assert StackRunStore.load_execution_checkpoint(current.path).completed_steps == (1,)


def test_cancelled_partial_revision_is_not_executable(observed, tmp_path):
    current, capture, refinement = observed
    checks = 0
    def check():
        nonlocal checks
        checks += 1
        if checks == 3:
            raise InterruptedError('cancelled after suffix write')
    destination = tmp_path/'partial.stack'
    with pytest.raises(InterruptedError):
        write_refined_branch(current, ROOT, capture, 1, refinement,
                             destination=destination, check=check)
    assert not destination.exists()
    temporary = list(tmp_path.glob('.partial.pending-*.stack'))
    assert len(temporary) == 1
    assert StackRunStore.info(temporary[0]).status == 'failed'
    with pytest.raises(ValueError, match='incomplete scene refinement'):
        load_execution_run(temporary[0], ROOT)
    assert pending_scene_refinement(current.path) is not None


def test_source_cannot_execute_stale_suffix_after_acceptance(observed):
    current, _, _ = observed
    transport, executor = executor_for(current)
    calls = len(transport.calls)
    with pytest.raises(ValueError, match='refin'):
        execute_saved_run(current, executor)
    assert len(transport.calls) == calls


def test_accepted_scene_updates_before_motion_refinement_and_survives_release_replay(observed):
    from stacking_tabletop.gui.operations.execution_scene import ExecutionScene
    current, capture, _ = observed
    scene = ExecutionScene(current, ROOT)
    for event in StackRunStore.load_execution_events(current.path):
        scene.event(event)
    expected = np.asarray(json.loads((capture/'sceneid_result.json').read_text())['proposed_poses'][0])
    np.testing.assert_allclose(scene.released[1], expected)
    assert scene._identified_through_step == 1


def test_partial_revision_executes_only_checked_prefix(observed, tmp_path):
    current, capture, refinement = observed
    artifact = StackRunStore.load_mcts(current.path)
    original = current.plans[-1]
    action = original.problem.action.copy()
    action.object_id, action.object_index = 3, 2
    plan = replace(original, step=3, problem=replace(original.problem, action=action))
    selected = replace(artifact.steps[-1].candidates[0], object_id=3, object_index=2)
    checkpoint = StackRunStore.load_planning_checkpoint(current.path, 2)
    state = checkpoint.state.copy()
    state.placed_object_indices.append(2)
    state.object_poses[3] = action.pose.copy()
    with StackRunStore.open_append(current.path) as writer:
        writer.append_placement_plan(plan)
        writer.append_mcts_step(replace(artifact.steps[-1], step=3, candidates=(selected,)),
                               checkpoint=replace(checkpoint, step=3, state=state))
    current = load_execution_run(current.path, ROOT)
    destination = write_refined_branch(current, ROOT, capture, 1, refinement,
                                       destination=tmp_path/'next-step-ready.stack')
    revised = load_execution_run(destination, ROOT)
    assert StackRunStore.load_execution_checkpoint(destination).pending_steps == (2, 3)
    blocked = revised.plans[-1]
    assert blocked.result.diagnostics['motion_requires_replan']
    assert all(segment.name.startswith('unvalidated/') for segment in blocked.result.segments)
    with pytest.raises(ValueError, match='five direct-plan segments'):
        validate_command(build_execution_command(blocked, revised.context),
                         revised.planning, revised.grasp_steps)
    _, executor = executor_for(revised)
    selected_steps = []
    result = execute_saved_run(revised, executor, next_step=lambda plan: selected_steps.append(plan.step))
    assert selected_steps == [2] and result[-1].status.value == 'succeeded'
    assert [c.step for c in StackRunStore.load_execution_commands(destination)] == [1, 2]
    with pytest.raises(ValueError, match='Next motion needs planning'):
        execute_saved_run(revised, executor)


def test_revision_excludes_uncommitted_failed_search_tail(observed, tmp_path):
    current, capture, refinement = observed
    last = StackRunStore.load_mcts(current.path).steps[-1]
    failed = replace(last, step=3, candidates=(), selected_candidate_id=None,
                     succeeded=False, score_field=None, failure_reason='search exhausted')
    with StackRunStore.open_append(current.path) as writer:
        writer.append_mcts_step(failed)
    destination = write_refined_branch(current, ROOT, capture, 1, refinement,
                                       destination=tmp_path/'committed-only.stack')
    assert [plan.step for plan in load_execution_run(destination, ROOT).plans] == [1, 2]
    assert [step.step for step in StackRunStore.load_mcts(destination).steps] == [1, 2]


@pytest.mark.parametrize('change', ['report', 'context', 'order', 'cancel'])
def test_invalid_revision_never_publishes_or_unblocks_source(observed, tmp_path, change):
    current, capture, refinement = observed
    check = lambda: None
    if change == 'report':
        (capture/'sceneid_result.json').write_text('{"status":"changed"}')
    elif change == 'context':
        status = json.loads((capture/'capture.json').read_text())
        status['context_sha256'] = 'a'*64
        (capture/'capture.json').write_text(json.dumps(status))
    elif change == 'order':
        action = refinement.records[0].problem.action.copy()
        action.object_id = 1
        refinement.records = (replace(refinement.records[0],
                               problem=replace(refinement.records[0].problem, action=action)),)
    else:
        def check():
            raise InterruptedError('operator cancelled')
    destination = tmp_path/'rejected.stack'
    with pytest.raises((ValueError, InterruptedError)):
        write_refined_branch(current, ROOT, capture, 1, refinement, destination=destination, check=check)
    assert not destination.exists()
    assert pending_scene_refinement(current.path) is not None
