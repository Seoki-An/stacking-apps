"""Checkpoint-based branching for excavator search and motion-planning runs."""
from dataclasses import replace
from pathlib import Path

import numpy as np

from stacking_common.artifacts import StackRunStore
from stacking_common.artifacts.planning_context import _values, branch_planning_run, validate_planning_resume
from stacking_common.planning import PickSceneSnapshot


def restore_pick_scene(checkpoint, context):
    """Restore exact provenance, allowing only old quaternion normalization noise."""
    recorded = _values(context.profiles['pick_scene'])
    current = checkpoint.pick_scene.artifact_metadata()
    poses = recorded.pop('object_poses')
    current_poses = current.pop('object_poses')
    if recorded != _values(current) or poses.keys() != current_poses.keys():
        raise ValueError('checkpoint pickup scene differs from the recorded run context')
    for object_id, pose in poses.items():
        actual = current_poses[object_id]
        if (not np.array_equal(pose[:3], actual[:3]) or
                not np.allclose(pose[3:], actual[3:], rtol=0, atol=8 * np.finfo(float).eps)):
            raise ValueError(f'checkpoint pickup pose differs from the recorded context: object {object_id}')
    return PickSceneSnapshot(recorded.pop('frame_id'), poses, recorded.pop('source'), recorded)


def load_resume_checkpoint(path):
    try:
        return StackRunStore.load_planning_checkpoint(path)
    except ValueError as error:
        if ('missing planning_checkpoint/' in str(error) and
                StackRunStore.info(path).metadata.get('mode') == 'search'):
            raise ValueError(
                'This search-only run was saved without resume checkpoints. Its completed '
                'steps can be viewed, but their full search state cannot be restored. '
                'Start a new run with the updated application; new search-only runs save '
                'a checkpoint with every selected step. The original file is unchanged.'
            ) from error
        raise


def branch_excavator_run(source, destination, context):
    if not context.profiles['runtime'].get('search_only', False):
        return branch_planning_run(source, destination, context)
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists():
        raise FileExistsError('resume requires a new output .stack path')
    checkpoint = validate_planning_resume(source, context)
    if StackRunStore.load_placement_plans(source):
        raise ValueError('search-only resume cannot retain robot motion plans')
    artifact = StackRunStore.load_mcts(source)
    statistics = {int(item['step']): item for item in StackRunStore.load_mcts_statistics(source)}
    writer = StackRunStore.create_mcts(destination, replace(artifact, steps=()), metadata={
        **dict(StackRunStore.info(source).metadata),
        'planning_parent': {'path': str(source), 'checkpoint_step': checkpoint.step},
    })
    try:
        writer.append_run_context(context)
        for step in artifact.steps:
            if step.step <= checkpoint.step:
                writer.append_mcts_step(step, statistics=statistics.get(step.step),
                    checkpoint=checkpoint if step.step == checkpoint.step else None)
    except BaseException:
        writer.finish('failed')
        writer.close()
        raise
    return writer, checkpoint
