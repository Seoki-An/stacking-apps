"""Bounded scene correction never reorders stones or accepts partial suffixes."""

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from stacking_common.domain import State
from stacking_common.artifacts import StackRunStore
from stacking_common.planning import PlacementPlanningProblem
from stacking_tabletop.planning import scene_refinement as refinement
from test_tabletop_saved_execution import ROOT, saved


@pytest.fixture
def case(saved, monkeypatch, tmp_path):
    context = replace(saved.context, profiles={**saved.context.profiles, 'simulation_profile': {}})
    frame = saved.planning.world_T_stack
    template = saved.plans[0]
    plans = []
    for index in range(3):
        action = template.problem.action.copy()
        action.object_index, action.object_id = index, index + 1
        action.pose = frame.parent_pose_to_child(template.problem.place_pose)
        action.pose[0] += .1 * index
        problem = replace(template.problem, action=action,
                          place_pose=frame.child_pose_to_parent(action.pose))
        plans.append(replace(template, step=index + 1, problem=problem,
                             candidate_id=f'candidate-{index + 1}', context_sha256=context.sha256))
    saved = replace(saved, context=context, plans=tuple(plans))
    state = State([1, 2, 3], [0], {1: plans[0].problem.action.pose}, [],
                  [plans[0].problem.action])
    checkpoint = NS(step=1, context_sha256=context.sha256, state=state,
                    pick_scene=NS(object_poses={i: template.problem.pick_pose.copy() for i in (1, 2, 3)}))
    monkeypatch.setattr(StackRunStore, 'load_planning_checkpoint', lambda *args: checkpoint)
    inventory = NS(object_ids=(1, 2, 3))
    monkeypatch.setattr(refinement, '_inventory', lambda *args: inventory)
    simulation = NS(placement=NS(settle_dt=.001, settle_steps=5), native_config=lambda: object())
    monkeypatch.setattr(refinement, '_recorded_simulation', lambda values: simulation)
    observed = frame.child_pose_to_parent(state.object_poses[1])
    observed[2] += .001
    evidence = NS(entity_id=1, prior_carried=False, inliers=10, rmse=.001, min_information=1.)
    candidate = NS(ready_for_review=True, scene_frame='world', source_hashes={'pcd': 'a' * 64},
                   result=NS(status='IDENTIFIED', support_warning=False, entity_ids=[1],
                             proposed_poses=[observed], evidence=[evidence],
                             gap_screen=NS(nonpenetrating=True)))
    monkeypatch.setattr(refinement, '_scene_candidate', lambda *args: candidate)
    rollout_calls = []

    def rollout(candidate, **kwargs):
        assert kwargs['review'](candidate)
        rollout_calls.append(kwargs)
        branch = kwargs['state'].copy()
        for entity, pose in zip(candidate.result.entity_ids, candidate.result.proposed_poses):
            object_id = kwargs['entity_to_object_id'][entity]
            branch.object_poses[object_id] = frame.parent_pose_to_child(pose)
            if not candidate.result.evidence[0].prior_carried:
                branch.pose_identified_object_ids.add(object_id)
        return kwargs['replan'](branch), {'steps': 5, 'max_translation': 0., 'max_rotation': 0.}
    monkeypatch.setattr(refinement, 'review_rollout_and_replan', rollout)
    capture = tmp_path / 'capture'
    capture.mkdir()
    metadata = {'kind': 'stacked_scene_wrist_capture', 'status': 'complete',
                'step': 1, 'context_sha256': context.sha256}

    def write_review():
        report = {'kind': 'tabletop_stack_scene_identification', 'accepted_snapshot': True,
                  'status': str(candidate.result.status), 'entity_ids': candidate.result.entity_ids,
                  'proposed_poses': [pose.tolist() for pose in candidate.result.proposed_poses],
                  'source_hashes': dict(candidate.source_hashes)}
        path = capture / 'sceneid_result.json'
        path.write_text(json.dumps(report))
        metadata['scene_identification'] = {'path': str(path),
                                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        (capture / 'capture.json').write_text(json.dumps(metadata))
    write_review()
    attempts, closed = [], []
    options = {'drift': .0005, 'failure_step': None, 'residual': 0., 'missing_achieved': False,
               'unsafe_wrist': False, 'structural_failure': None, 'structural_drift': 0.}

    def attempt(branch, action, **kwargs):
        attempts.append((branch.copy(), action.copy(), dict(kwargs)))
        action = action.copy()
        action.pose[2] += options['drift']
        problem = PlacementPlanningProblem.from_search(
            branch, action, pick_pose=kwargs['pick_pose'], world_T_stack=frame,
            pick_scene_poses={i: pose for i, pose in checkpoint.pick_scene.object_poses.items()
                              if i not in branch.object_poses})
        achieved = problem.place_pose.copy()
        achieved[2] += options['residual']
        result = replace(template.result, diagnostics={} if options['missing_achieved'] else
                         {'achieved_place_pose': achieved.tolist(), 'place_support_gap': 0.})
        if kwargs['step'] == options['failure_step']:
            result = replace(result, status='infeasible', reason='collision')
        if options['unsafe_wrist']:
            segments = list(result.segments)
            positions = segments[2].positions.copy()
            positions[1, 5] = 6.1
            segments[2] = replace(segments[2], positions=positions)
            result = replace(result, segments=tuple(segments))
        return replace(template, step=kwargs['step'], attempt=kwargs['attempt'],
                       candidate_id=kwargs['candidate_id'], context_sha256=context.sha256,
                       problem=problem, result=result)
    coordinator = NS(attempt=attempt, placement_backend=NS(close=lambda: closed.append(True)))
    monkeypatch.setattr(refinement, '_coordinator', lambda *args: coordinator)
    structural_calls, structural_closed = [], []

    def validate_structure(branch, action):
        assert not branch.pose_identified_object_ids
        structural_calls.append(branch.copy())
        poses = {**branch.object_poses, action.object_id: action.pose}
        trajectories = {key: NS(poses=[pose.copy()]) for key, pose in poses.items()}
        # The support returns to its initial pose; its transient excursion must
        # still reject, not be hidden by only checking terminal poses.
        peak = poses[1].copy()
        peak[2] += options['structural_drift']
        trajectories[1].poses.append(peak)
        return NS(failed=action.object_id == options['structural_failure'], settled=True,
                  object_poses=poses, trajectories=trajectories,
                  diagnostics={'rejection_reason': 'support_collapse', 'simulation_frozen_object_count': 0})
    structural = NS(place=validate_structure, close=lambda: structural_closed.append(True))
    monkeypatch.setattr(refinement, '_structural_backend', lambda *args: structural)
    return NS(saved=saved, checkpoint=checkpoint, candidate=candidate, capture=capture,
              metadata=metadata, write_review=write_review, attempts=attempts,
              closed=closed, options=options, rollout_calls=rollout_calls,
              structural_calls=structural_calls, structural_closed=structural_closed,
              run=lambda **kwargs: refinement.refine_remaining(saved, ROOT, capture, 1, **kwargs))


def test_small_correction_refines_full_suffix_in_order_and_keeps_input_immutable(case):
    initial = case.checkpoint.state.copy()
    result = case.run()
    assert result.summary['mcts_used'] is False
    assert result.summary['remaining_steps'] == [2, 3]
    assert result.summary['object_order'] == [2, 3]
    assert result.summary['entity_to_object_id'] == {'1': 1}
    assert result.records[0].candidate_id == 'candidate-2'
    assert result.records[0].attempt == 2
    assert result.states[0].placed_object_indices == [0, 1]
    assert result.states[1].placed_object_indices == [0, 1, 2]
    assert case.attempts[0][0].placed_object_indices == [0]
    assert case.attempts[1][0].placed_object_indices == [0, 1]
    np.testing.assert_allclose(case.attempts[1][0].object_poses[2], result.records[0].problem.action.pose)
    np.testing.assert_array_equal(case.checkpoint.state.object_poses[1], initial.object_poses[1])
    assert result.corrected_state.placed_object_indices == [0]
    assert set(result.corrected_state.object_poses) == {1}
    assert not result.corrected_state.trajectories and not result.corrected_state.contact_points
    assert case.closed == [True]
    assert len(case.structural_calls) == 2 and case.structural_closed == [True]
    assert result.records[0].result.diagnostics['scene_refinement_structure']['all_stones_dynamic']
    json.dumps(result.summary, allow_nan=False)


@pytest.mark.parametrize('kind', ['translation', 'rotation', 'support', 'prior', 'evidence', 'coverage'])
def test_unqualified_or_large_scene_changes_never_start_motion_refinement(case, kind):
    result = case.candidate.result
    if kind == 'translation':
        result.proposed_poses[0][2] += .006
    elif kind == 'rotation':
        from scipy.spatial.transform import Rotation
        result.proposed_poses[0][3:] = Rotation.from_euler('z', 10, degrees=True).as_quat()
    elif kind == 'support':
        result.support_warning = True
    elif kind == 'prior':
        result.evidence[0].prior_carried = True
    elif kind == 'evidence':
        result.evidence[0].inliers = 0
    else:
        result.entity_ids = [1, 1]
    case.write_review()
    with pytest.raises(refinement.RefinementNeedsReview):
        case.run()
    assert not case.attempts


def test_prior_carried_unchanged_is_allowed(case):
    case.candidate.result.proposed_poses[0] = case.saved.planning.world_T_stack.child_pose_to_parent(
        case.checkpoint.state.object_poses[1])
    case.candidate.result.evidence[0].prior_carried = True
    case.write_review()
    assert case.run().summary['scene_changes']['1']['translation_m'] == pytest.approx(0.)


@pytest.mark.parametrize('options,match', [
    ({'drift': .006}, 'target'),
    ({'residual': .006}, 'release'),
    ({'drift': -.004, 'residual': .008}, 'residual'),
    ({'missing_achieved': True}, 'achieved'),
    ({'failure_step': 3}, 'collision'),
    ({'unsafe_wrist': True}, 'wrist_3'),
])
def test_bad_future_motion_or_target_returns_no_partial_suffix(case, options, match):
    case.options.update(options)
    with pytest.raises(ValueError, match=match):
        case.run()
    assert case.closed == [True]
    assert case.checkpoint.state.placed_object_indices == [0]


def test_pose_recomputed_differently_from_operator_review_is_rejected(case):
    case.candidate.result.proposed_poses[0][0] += .0001
    with pytest.raises(refinement.RefinementNeedsReview, match='review'):
        case.run()
    assert not case.attempts


def test_sign_equivalent_review_quaternion_is_accepted(case):
    case.candidate.result.proposed_poses[0][3:] *= -1
    assert len(case.run().records) == 2


def test_changed_report_hash_or_capture_context_is_rejected(case):
    (case.capture / 'sceneid_result.json').write_text('{}')
    with pytest.raises(refinement.RefinementNeedsReview, match='hash'):
        case.run()
    case.write_review()
    case.metadata['context_sha256'] = '0' * 64
    (case.capture / 'capture.json').write_text(json.dumps(case.metadata))
    with pytest.raises(refinement.RefinementNeedsReview, match='context'):
        case.run()


def test_native_rollout_failure_stops_before_motion(case, monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError('SceneID rollout moved a placed stone')
    monkeypatch.setattr(refinement, 'review_rollout_and_replan', reject)
    with pytest.raises(ValueError, match='rollout'):
        case.run()
    assert not case.attempts


def test_cancellation_between_operations_never_returns_partial_result(case):
    def check():
        if case.attempts:
            raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        case.run(check=check)
    assert case.closed == [True]


def test_missing_recorded_inputs_fail_clearly(case, monkeypatch):
    def missing(*args):
        raise KeyError('short_simulation_profile')
    monkeypatch.setattr(refinement, '_coordinator', missing)
    with pytest.raises(refinement.RefinementNeedsReview, match='incomplete.*short_simulation_profile'):
        case.run()
    assert not case.attempts


@pytest.mark.parametrize('options', [
    {'structural_failure': 2}, {'structural_failure': 3}, {'structural_drift': .002},
])
def test_support_stack_failure_or_transient_motion_rejects_entire_suffix(case, options):
    case.options.update(options)
    with pytest.raises(refinement.RefinementNeedsReview, match='full-stack'):
        case.run()
    assert case.closed == case.structural_closed == [True]
    assert case.checkpoint.state.placed_object_indices == [0]


def test_recorded_simulation_reconstructs_resolved_nested_dataclasses():
    from stacking_common.artifacts.planning_context import _values
    from stacking_tabletop.composition import TabletopApplication
    application = TabletopApplication.from_repository(ROOT)
    for original in (application.simulation_profile, application.short_simulation_profile):
        restored = refinement._recorded_simulation(_values(original))
        assert _values(restored) == _values(original)
        assert restored.placement.structural.dt == original.placement.structural.dt
