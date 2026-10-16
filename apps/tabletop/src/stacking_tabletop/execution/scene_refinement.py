"""Execution-aware, immutable plan revisions after an accepted stack observation.

Only the already completed prefix is copied as execution history. The original
document stays blocked at its accepted observation; only a validated revision
can supply new commands for the remaining objects. This module sends no motion.
"""

from dataclasses import replace
import json
import os
from pathlib import Path
from uuid import uuid4

from stacking_common.artifacts import StackRunStore, build_execution_command
from stacking_common.visualization import SceneObjectPose, SceneSnapshot, SceneTimeline
from stacking_tabletop.acquisition.hand_eye_capture import digest

from .saved_plan import load_execution_run, validate_command


def pending_scene_refinement(path):
    """Find a durable accepted observation not incorporated by this revision."""
    resolved = set(StackRunStore.info(path).metadata.get('scene_refinement_resolved', ()))
    pending = None
    for event in StackRunStore.load_execution_events(path):
        request = event.diagnostics.get('scene_refinement_required')
        token = f'{event.command_id}:{event.sequence}'
        if request is not None and token not in resolved:
            pending = {**dict(request), 'command_id': event.command_id, 'sequence': event.sequence}
    return pending


def _validate_request(saved, capture, step):
    request = pending_scene_refinement(saved.path)
    if (request is None or request['step'] != step
            or Path(request['capture_directory']).resolve() != capture):
        raise ValueError('scene refinement requires the current operator-accepted capture')
    checkpoint = StackRunStore.load_execution_checkpoint(saved.path)
    if (checkpoint.requires_reconciliation
            or checkpoint.completed_steps != tuple(range(1, step + 1))):
        raise ValueError('finish/reconcile the released step before refining its remaining plan')
    commands = StackRunStore.load_execution_commands(saved.path)
    if any(command.step > step for command in commands):
        raise ValueError('cannot refine a suffix that already has execution commands')
    status = json.loads((capture / 'capture.json').read_text())
    if (status.get('kind') != 'stacked_scene_wrist_capture'
            or status.get('status') != 'complete' or status.get('step') != step
            or status.get('context_sha256') != saved.context.sha256
            or status.get('command_id') != request['command_id']
            or not status.get('robot_home')):
        raise ValueError('scene capture does not match the completed command/context at home')
    identification = status['scene_identification']
    report = capture / 'sceneid_result.json'
    if (Path(identification['path']).resolve() != report
            or identification['sha256'] != digest(report)
            or request.get('sceneid_sha256') != identification['sha256']
            or status['pcd_sha256'] != digest(Path(status['pcd']))
            or request.get('pcd_sha256') != status['pcd_sha256']):
        raise ValueError('accepted scene-identification/PCD evidence has changed')
    return request


def _scene(state, frame, transform):
    if frame not in (transform.parent_frame, transform.child_frame):
        raise ValueError('refinement viewer frame differs from the recorded planning frames')
    objects = []
    for index in state.placed_object_indices:
        object_id = state.object_ids[index]
        pose = state.object_poses[object_id]
        if frame == transform.parent_frame:
            pose = transform.child_pose_to_parent(pose)
        objects.append(SceneObjectPose(object_id, pose, index))
    return SceneSnapshot(frame, tuple(objects))


def _debug_step(original, record, state, transform):
    """Retain action identity, not stale search scores or simulated trajectories."""
    selected = next(value for value in original.candidates if value.selected)
    pose = record.problem.action.pose
    if original.scene.frame_id == transform.parent_frame:
        pose = transform.child_pose_to_parent(pose)
    selected = replace(
        selected, rank=1, init_pose=pose, solved_pose=pose, action_pose=pose,
        final_pose=pose, score=None, reward=None, prior=0., visits=0, mean_value=0.,
        score_field_id=None, timeline=SceneTimeline(()), simulation_contacts=(),
        pose_solve_forces=(), best_sequence=(), scene_motion=None,
    )
    return replace(original, scene=_scene(state, original.scene.frame_id, transform),
                   candidates=(selected,), score_field=None, rejected_expansions={})


def write_refined_branch(saved, repository, capture, step, refinement, *,
                         destination=None, check=lambda: None):
    """Publish a separately reviewed branch, preserving original execution bytes."""
    capture = Path(capture).resolve()
    request = _validate_request(saved, capture, step)
    saved.verify_assets()
    records, states = tuple(refinement.records), tuple(refinement.states)
    remaining = tuple(plan for plan in saved.plans if plan.step > step)
    if not remaining or len(records) != len(remaining) or len(states) != len(records):
        raise ValueError('refinement must cover the complete unexecuted suffix')
    for old, record in zip(remaining, records):
        if (record.step != old.step or record.context_sha256 != saved.context.sha256
                or record.candidate_id != old.candidate_id
                or record.problem.action.object_id != old.problem.action.object_id
                or record.problem.action.object_index != old.problem.action.object_index
                or not record.result.succeeded):
            raise ValueError('refinement changed object order/context or includes failed motion')
        validate_command(build_execution_command(record, saved.context), saved.planning, saved.grasp_steps)

    metadata = dict(StackRunStore.info(saved.path).metadata)
    origin = Path(metadata.get('scene_refinement_origin', str(saved.path)))
    destination = (saved.path.with_name(
        f'{origin.stem}-refined-after-{step}-{uuid4().hex[:8]}.stack')
        if destination is None else Path(destination).resolve())
    if destination == saved.path or destination.exists():
        raise FileExistsError('scene refinement requires a new .stack document')
    artifact = StackRunStore.load_mcts(saved.path)
    checkpoints = {plan.step: StackRunStore.load_planning_checkpoint(saved.path, plan.step)
                   for plan in saved.plans}
    statistics = {int(value['step']): value for value in StackRunStore.load_mcts_statistics(saved.path)}
    metadata['scene_refinement_origin'] = str(origin)
    # A scan-only recovery may supersede an accepted capture of the same released
    # step. The successfully validated newest observation resolves those older
    # tokens too; later-step requests are never covered by this revision.
    metadata['scene_refinement_resolved'] = sorted(set(
        metadata.get('scene_refinement_resolved', ())) | {
        f'{event.command_id}:{event.sequence}'
        for event in StackRunStore.load_execution_events(saved.path)
        if event.diagnostics.get('scene_refinement_required', {}).get('step', step + 1) <= step
    })
    metadata['scene_refinement'] = {
        'parent': str(saved.path), 'completed_step': step, 'capture': str(capture),
        'sceneid_sha256': request['sceneid_sha256'], 'pcd_sha256': request['pcd_sha256'],
        'policy': 'fixed_order_local_refinement', 'mcts_rerun': False,
        'summary': refinement.summary,
    }
    # Collect an immutable prefix before opening the destination. All old physical
    # observations remain unchanged; corrected scene state belongs to checkpoints.
    commands = StackRunStore.load_execution_commands(saved.path)
    events = StackRunStore.load_execution_events(saved.path)
    reconciliations = StackRunStore.load_execution_reconciliations(saved.path)
    prior_plans = StackRunStore.load_placement_plans(saved.path)
    revised = {record.step: (record, state) for record, state in zip(records, states)}
    check()
    temporary = destination.with_name(f'.{destination.stem}.pending-{uuid4().hex[:8]}.stack')
    with StackRunStore.create_mcts(temporary, replace(artifact, steps=()), metadata=metadata) as writer:
        writer.append_run_context(saved.context)
        for plan in prior_plans:
            if plan.step <= step:
                writer.append_placement_plan(plan)
        for original in artifact.steps:
            if original.step > step:
                continue
            checkpoint = checkpoints[original.step]
            if original.step == step:
                checkpoint = replace(checkpoint, state=refinement.corrected_state)
            writer.append_mcts_step(original, statistics=statistics.get(original.step), checkpoint=checkpoint)
        # Replay command/event/reconciliation order before adding pending plans;
        # there can be no window with pending objects but missing completion history.
        for command in commands:
            writer.append_execution_command(command)
            for event in events:
                if event.command_id == command.command_id:
                    writer.append_execution_event(event)
            for reconciliation in reconciliations:
                if reconciliation.command_id == command.command_id:
                    writer.append_execution_reconciliation(reconciliation)
        before = refinement.corrected_state
        for original in artifact.steps:
            if original.step not in revised:
                continue
            check()
            record, state = revised[original.step]
            checkpoint = replace(checkpoints[original.step], state=state)
            writer.append_placement_plan(record)
            writer.append_mcts_step(
                _debug_step(original, record, before, saved.planning.world_T_stack),
                statistics={'step': original.step, 'kind': 'scene_local_refinement',
                            'mcts_rerun': False, 'source_step': original.step},
                checkpoint=checkpoint,
            )
            before = state
        # Refuse a changed capture/source or cancellation before finalizing.
        if _validate_request(saved, capture, step) != request:
            raise ValueError('scene refinement request changed during planning')
        check()
    try:
        validated = load_execution_run(temporary, repository)
        after = StackRunStore.load_execution_checkpoint(validated.path)
        if (after.completed_steps != tuple(range(1, step + 1)) or after.requires_reconciliation
                or after.pending_steps != tuple(plan.step for plan in remaining)):
            raise ValueError('refined execution history does not preserve the complete suffix/prefix')
        check()
        # Link without replacing any concurrently created destination, then
        # remove only our temporary alias. SQLite has closed its DELETE journal.
        os.link(temporary, destination)
    except BaseException:
        with StackRunStore.open_append(temporary) as writer:
            writer.finish('failed')
        raise
    temporary.unlink()
    return destination


def refine_execution_run(run, repository, capture, step, *, check=lambda: None):
    """Perception-service entry point; computation only, never automatic execution."""
    from stacking_tabletop.planning.scene_refinement import refine_remaining
    saved = load_execution_run(run, repository)
    capture, step = Path(capture).resolve(), int(step)
    _validate_request(saved, capture, step)
    check()
    refinement = refine_remaining(saved, repository, capture, step, check=check)
    check()
    path = write_refined_branch(saved, repository, capture, step, refinement, check=check)
    return {'run': str(path), 'summary': refinement.summary}
