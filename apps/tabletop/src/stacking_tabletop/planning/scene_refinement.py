"""Accepted scene correction and checked motion continuation in fixed stone order.

This computes a proposal only: it never writes a run or authorizes robot motion.
The caller must review and persist the complete result before execution resumes.
"""

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from stacking_common.artifacts import StackRunStore, build_execution_command
from stacking_common.assets import ObjectAssetLoader
from stacking_common.domain import DirectionalTaper, TargetGeometry
from stacking_common.domain._validation import pose_array
from stacking_common.environment.core_backend import (
    CorePhysicsProfile, CorePlacementBackend, CorePlacementConfig,
    CoreSimulationProfile, SimulationPhaseConfig,
)
from stacking_common.environment.robustness import PlaceStabilityConfig
from stacking_common.planning import PlacementPlanningProblem
from stacking_common.planning.scene_identification import review_rollout_and_replan
from stacking_common.profiles.native import NativeSolverProfile
from stacking_tabletop.execution.saved_plan import validate_command

from .adapter import TabletopMotionPlannerAdapter
from .coordinator import TabletopPlanningCoordinator
from .native import TabletopNativePlanningBackend
from .safety import TabletopPlacementSafety
from .scene_identification import prepare_saved_stack_candidate


MAX_TRANSLATION = .005
MAX_ROTATION = np.deg2rad(3.)
ROLLOUT_TRANSLATION = .001
ROLLOUT_ROTATION = np.deg2rad(.5)


class RefinementNeedsReview(ValueError):
    """The local correction is unsafe or needs a broader planning decision."""


@dataclass(frozen=True)
class SceneRefinement:
    corrected_state: object
    records: tuple
    states: tuple
    summary: dict


def _pose_delta(before, after):
    before, after = pose_array(before), pose_array(after)
    cosine = abs(float(before[3:] @ after[3:]))
    cosine /= np.linalg.norm(before[3:]) * np.linalg.norm(after[3:])
    return (float(np.linalg.norm(after[:3] - before[:3])),
            float(2 * np.arccos(np.clip(cosine, 0., 1.))))


def _delta(before, after):
    translation, rotation = _pose_delta(before, after)
    return {'translation_m': translation, 'rotation_rad': rotation}


def _is_local(delta):
    return (delta['translation_m'] <= MAX_TRANSLATION + 1e-10
            and delta['rotation_rad'] <= MAX_ROTATION + 1e-10)


def _recorded_simulation(values):
    """The saved context contains resolved dataclasses, not the YAML schema."""
    values = dict(values)
    placement = dict(values.pop('placement'))
    for key in ('structural', 'short_gate'):
        if placement.get(key) is not None:
            placement[key] = SimulationPhaseConfig(**dict(placement[key]))
    values['placement'] = CorePlacementConfig(**placement)
    if values.get('settling_physics') is not None:
        values['settling_physics'] = CorePhysicsProfile(**dict(values['settling_physics']))
    return CoreSimulationProfile(**values)


def _inventory(saved, repository, checkpoint):
    values = saved.context.profiles['inventory_profile']
    target = dict(values['target'])
    target['taper'] = DirectionalTaper(**dict(target['taper']))
    loader = ObjectAssetLoader(
        Path(repository) / values['model_directory'], density=values['density'],
        density_overrides=dict(values['density_overrides']),
        target=TargetGeometry(**target),
    )
    return loader.load(checkpoint.state.object_ids, checkpoint.pick_scene.object_poses)


def _scene_candidate(saved, repository, capture, step, checkpoint):
    import stacking_core as core
    from stacking_common.planning.scene_identification import load_reviewed_stack_capture
    from stacking_common.reconstruction import load_point_cloud

    # Reload the actual hashed PCD and re-solve rather than treating report JSON
    # as a trusted native snapshot. SceneID is CPU-capable and never invokes MCTS.
    load_reviewed_stack_capture(capture, voxel_size=.001)
    status = json.loads((capture / 'capture.json').read_text())
    points = load_point_cloud(status['pcd']).points
    config = core.SceneIdConfig()
    config.scan_sigma = .005
    config.outlier_cutoff = .025
    config.reference_scan_points = max(1., len(points) / len(checkpoint.state.placed_object_indices))
    capabilities = core.cuda_capabilities() if hasattr(core, 'cuda_capabilities') else {}
    cuda_scan = bool(capabilities.get('sceneid_scan') and capabilities.get('devices'))
    return prepare_saved_stack_candidate(
        saved, repository, capture, config, step,
        scan_voxel_size=.001, mesh_voxel_size=.001, cuda_scan=cuda_scan,
    )


def _correct_scene(saved, checkpoint, candidate, simulation):
    if not candidate.ready_for_review or candidate.result.support_warning:
        raise RefinementNeedsReview('SceneID is unqualified or reports a support warning')
    if not candidate.result.gap_screen.nonpenetrating:
        raise RefinementNeedsReview('SceneID reports penetrating stones')
    if candidate.scene_frame != saved.planning.world_T_stack.parent_frame:
        raise RefinementNeedsReview('SceneID frame differs from the saved planning world')
    state = checkpoint.state.copy()
    expected = {index + 1: state.object_ids[index] for index in state.placed_object_indices}
    entities = [int(value) for value in candidate.result.entity_ids]
    poses = list(candidate.result.proposed_poses)
    evidence = list(candidate.result.evidence)
    if (set(entities) != set(expected) or len(entities) != len(expected)
            or len(poses) != len(entities)
            or {int(value.entity_id) for value in evidence} != set(expected)
            or len(evidence) != len(expected)):
        raise RefinementNeedsReview('SceneID does not cover exactly the placed inventory')
    by_entity = {int(value.entity_id): value for value in evidence}
    changes = {}
    for entity, pose in zip(entities, poses):
        object_id = expected[entity]
        pose = saved.planning.world_T_stack.parent_pose_to_child(pose_array(pose))
        delta = _delta(state.object_poses[object_id], pose)
        item = by_entity[entity]
        if item.prior_carried:
            if delta['translation_m'] > 1e-8 or delta['rotation_rad'] > 1e-6:
                raise RefinementNeedsReview(f'Occluded stone {object_id} moved without scan evidence')
        elif (item.inliers < 3 or not np.isfinite(item.rmse)
              or not np.isfinite(item.min_information) or item.min_information <= 0):
            raise RefinementNeedsReview(f'Stone {object_id} has insufficient finite scan evidence')
        changes[str(object_id)] = delta
    placement = simulation.placement
    if placement.settle_steps <= 0:
        raise RefinementNeedsReview('Recorded simulation disables the required stability rollout')
    # This callback approves computation on a copy, not operator execution. The
    # caller still presents the complete proposal for explicit review afterwards.
    corrected, rollout = review_rollout_and_replan(
        candidate, review=lambda _: True,
        simulation_config=simulation.native_config(),
        dt=placement.settle_dt, steps=placement.settle_steps,
        max_translation=ROLLOUT_TRANSLATION, max_rotation=ROLLOUT_ROTATION,
        state=state, entity_to_object_id=expected,
        world_T_stack=saved.planning.world_T_stack, replan=lambda branch: branch,
    )
    corrected.object_poses = {object_id: corrected.object_poses[object_id].copy()
                              for object_id in expected.values()}
    corrected.contact_points.clear()
    corrected.trajectories.clear()
    return corrected, changes, rollout


def _match_reviewed_estimate(capture, metadata, candidate):
    review = metadata['scene_identification']
    path = Path(review['path']).resolve()
    if not path.is_relative_to(capture) or not path.is_file():
        raise RefinementNeedsReview('Reviewed SceneID report is missing or outside its capture')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != review['sha256']:
        raise RefinementNeedsReview('Reviewed SceneID report hash changed')
    report = json.loads(path.read_text())
    entities, poses = report['entity_ids'], report['proposed_poses']
    if (report.get('kind') != 'tabletop_stack_scene_identification'
            or report.get('accepted_snapshot') is not True
            or report.get('status') != str(candidate.result.status)
            or len(entities) != len(poses) or len(set(entities)) != len(entities)
            or set(entities) != set(candidate.result.entity_ids)
            or report.get('source_hashes') != dict(candidate.source_hashes)):
        raise RefinementNeedsReview('Re-solved SceneID differs from the reviewed report; review again')
    reviewed = dict(zip(entities, poses))
    for entity, pose in zip(candidate.result.entity_ids, candidate.result.proposed_poses):
        translation, rotation = _pose_delta(reviewed[int(entity)], pose)
        if translation > 1e-6 or rotation > 1e-6:
            raise RefinementNeedsReview('Re-solved SceneID poses changed; review the new estimate first')


def _coordinator(saved, repository, checkpoint, inventory, progress):
    profiles = saved.context.profiles
    solver = NativeSolverProfile(**dict(profiles['native_solver']))
    simulation = _recorded_simulation(profiles['short_simulation_profile'])
    robustness = PlaceStabilityConfig(**dict(profiles['environment_profile']['place_stability']))
    action_profile = profiles['action_generation_profile']
    safety = None
    if action_profile['strict_floor_first']:
        from stacking_common.actions.floor import floor_policy_settings
        safety = TabletopPlacementSafety(
            inventory, **floor_policy_settings(action_profile['sampling_settings']['floor']))
    backend = TabletopNativePlanningBackend(inventory, saved.planning, solver, repository)
    placement = CorePlacementBackend.from_profile(
        inventory.models, simulation, place_stability=robustness)
    placement.reset(inventory.object_ids)

    def problem(state, action, *, pick_pose, metadata):
        pickup = {object_id: checkpoint.pick_scene.object_poses[object_id]
                  for index, object_id in enumerate(state.object_ids)
                  if index not in state.placed_object_indices}
        return PlacementPlanningProblem.from_search(
            state, action, pick_pose=pick_pose, world_T_stack=saved.planning.world_T_stack,
            pick_scene_poses=pickup, metadata={**metadata, 'scene_refinement': True})

    return TabletopPlanningCoordinator(
        inventory, problem, TabletopMotionPlannerAdapter(backend.solve),
        placement_backend=placement, safety=safety,
        world_T_stack=saved.planning.world_T_stack, progress=progress,
        context_sha256=saved.context.sha256,
    )


def _structural_backend(saved, inventory, simulation):
    # Use the recorded full structural phase, not the short gate's frozen
    # neighborhood. This validator is allowed to move every simulated stone.
    placement = replace(simulation.placement, short_gate=None,
                        structural=replace(simulation.placement.structural, freeze_radius=None))
    profile = replace(simulation, placement=placement)
    backend = CorePlacementBackend.from_profile(
        inventory.models, profile,
        place_stability=PlaceStabilityConfig(**dict(
            saved.context.profiles['environment_profile']['place_stability'])),
    )
    backend.reset(inventory.object_ids)
    return backend


def _validate_structure(backend, state, record, world_T_stack, stability):
    """Validate the corrected supporting stack under the new stone's load."""
    dynamic = state.copy()
    dynamic.pose_identified_object_ids.clear()
    action = record.problem.action.copy()
    action.pose = world_T_stack.parent_pose_to_child(
        record.result.diagnostics['achieved_place_pose'])
    result = backend.place(dynamic, action)
    if result.failed or not result.settled:
        reason = result.diagnostics.get('rejection_reason', 'not_settled')
        raise RefinementNeedsReview(
            f'Step {record.step} full-stack structural validation failed: {reason}')
    if result.diagnostics.get('simulation_frozen_object_count', 0):
        raise RefinementNeedsReview('Full-stack validation unexpectedly froze supporting stones')
    reference = {**state.object_poses, action.object_id: action.pose}
    if set(result.object_poses) != set(reference) or set(result.trajectories) != set(reference):
        raise RefinementNeedsReview('Full-stack validation omitted a stone or its motion history')
    maximum_translation = maximum_rotation = maximum_displacement = 0.
    limit = float(stability['fail_displacement'])
    rotation_weight = float(stability['rotation_weight'])
    if not np.isfinite(limit) or limit <= 0 or not np.isfinite(rotation_weight) or rotation_weight < 0:
        raise RefinementNeedsReview('Recorded structural displacement limits are invalid')
    for object_id, before in reference.items():
        poses = [*result.trajectories[object_id].poses, result.object_poses[object_id]]
        for pose in poses:
            translation, rotation = _pose_delta(before, pose)
            maximum_translation = max(maximum_translation, translation)
            maximum_rotation = max(maximum_rotation, rotation)
            displacement = translation + rotation_weight * rotation
            maximum_displacement = max(maximum_displacement, displacement)
            if displacement > limit:
                raise RefinementNeedsReview(
                    f'Step {record.step} full-stack validation moved stone {object_id} '
                    f'beyond recorded displacement limit {limit:g} m')
    # These are forecasts after a future placement, not replacements for the
    # currently accepted observed scene. Subsequent future plans use the forecast.
    return {'max_translation_m': maximum_translation, 'max_rotation_rad': maximum_rotation,
            'max_weighted_displacement_m': maximum_displacement, 'displacement_limit_m': limit,
            'settled_poses': {str(key): np.asarray(pose).tolist() for key, pose in result.object_poses.items()},
            'all_stones_dynamic': True, 'full_structural_phase': True}


def _check_containment(saved, inventory, record, pose):
    from stacking_common.environment.overlap import VoxelOverlapEvaluator
    profile = saved.context.profiles['environment_profile']
    overlap = VoxelOverlapEvaluator(inventory.target, profile['overlap_resolution'])
    fraction = overlap.object_containment(inventory[record.problem.action.object_index], pose)
    if fraction < profile['reward']['containment_threshold']:
        raise RefinementNeedsReview(f'Step {record.step} settled target leaves the target structure')
    return fraction


def _refine_one(saved, checkpoint, branch, original, coordinator, structural, check, step):
    action = original.problem.action.copy()
    if action.object_index in branch.placed_object_indices:
        raise RefinementNeedsReview(f'Step {original.step} picks an already placed stone')
    pickup = checkpoint.pick_scene.object_poses[action.object_id]
    if not np.allclose(pickup, original.problem.pick_pose, atol=1e-9, rtol=0):
        raise RefinementNeedsReview('Saved pickup pose differs from its checkpoint')
    record = coordinator.attempt(
        branch, action, step=original.step, attempt=original.attempt + 1,
        pick_pose=pickup, candidate_id=original.candidate_id,
    )
    check()
    if not record.result.succeeded:
        raise RefinementNeedsReview(
            f'Step {original.step} motion planning failed: {record.result.reason}')
    # The old target is a planning seed. Its correction is not a reason to
    # reject an otherwise supported, collision-checked new placement.
    delta = _delta(action.pose, record.problem.action.pose)
    achieved = record.result.diagnostics.get('achieved_place_pose')
    if achieved is None:
        raise RefinementNeedsReview('Refined motion lacks achieved release-pose validation')
    residual = _delta(record.problem.place_pose, achieved)
    limits = saved.context.profiles['native_solver']['motion']['direct']
    if (residual['translation_m'] > limits['target_pos_tol']
            or residual['rotation_rad'] > limits['target_rot_tol']):
        raise RefinementNeedsReview(
            f'Step {original.step} release residual exceeds recorded motion tolerances')
    validate_command(build_execution_command(record, saved.context),
                     saved.planning, saved.grasp_steps)
    check()
    structure = _validate_structure(structural, branch, record, saved.planning.world_T_stack,
                                   saved.context.profiles['environment_profile']['place_stability'])
    settled = structure['settled_poses'][str(action.object_id)]
    containment = _check_containment(saved, coordinator.inventory, record, settled)
    check()
    record = replace(record, result=replace(record.result, diagnostics={
        **record.result.diagnostics, 'scene_refinement': True,
        'scene_refinement_source_step': step, 'scene_refinement_target_delta': delta,
        'scene_refinement_structure': structure,
                    'scene_refinement_release_residual': residual,
        'scene_refinement_target_containment': containment,
    }))
    return record, delta


def refine_remaining(saved, repository, capture_directory, step, *,
                     check=lambda: None, progress=None):
    """Update a reviewed scene and plan a checked prefix of remaining motions.

    Failure of a later placement does not invalidate a checked next placement;
    unplanned later steps stay explicitly blocked in the resulting revision. Native
    operations are bounded by their recorded solver limits; cancellation is
    checked around each operation rather than interrupting native code.
    """
    check()
    saved.verify_assets()
    checkpoint = StackRunStore.load_planning_checkpoint(saved.path, step)
    if checkpoint.context_sha256 != saved.context.sha256 or checkpoint.step != step:
        raise RefinementNeedsReview('Scene refinement checkpoint/context mismatch')
    capture = Path(capture_directory).resolve()
    metadata = json.loads((capture / 'capture.json').read_text())
    if (metadata.get('kind') != 'stacked_scene_wrist_capture'
            or metadata.get('status') != 'complete' or metadata.get('step') != step
            or metadata.get('context_sha256') != saved.context.sha256):
        raise RefinementNeedsReview('Capture does not match the completed saved step/context')
    if not checkpoint.state.placed_object_indices:
        raise RefinementNeedsReview('Scene refinement needs a placed stone')
    try:
        simulation = _recorded_simulation(saved.context.profiles['simulation_profile'])
        inventory = _inventory(saved, repository, checkpoint)
        check()
        candidate = _scene_candidate(saved, repository, capture, step, checkpoint)
        check()
        _match_reviewed_estimate(capture, metadata, candidate)
        corrected, changes, rollout = _correct_scene(saved, checkpoint, candidate, simulation)
        check()
        future = tuple(record for record in saved.plans if record.step > step)
        if tuple(record.step for record in future) != tuple(range(step + 1, step + 1 + len(future))):
            raise RefinementNeedsReview('Remaining saved steps are not contiguous')
        branch = corrected.copy()
        records, states, target_changes = [], [], {}
        deferred = None
        coordinator = _coordinator(saved, repository, checkpoint, inventory, progress) if future else None
        structural = None
        try:
            if future:
                structural = _structural_backend(saved, inventory, simulation)
            for original in future:
                check()
                try:
                    record, delta = _refine_one(saved, checkpoint, branch, original,
                                               coordinator, structural, check, step)
                except ValueError as error:
                    if not records:
                        raise
                    deferred = {'step': original.step, 'reason': str(error)}
                    break
                refined = record.problem.action.copy()
                forecast = record.result.diagnostics['scene_refinement_structure']['settled_poses']
                refined.pose = pose_array(forecast[str(refined.object_id)])
                branch.placed_object_indices.append(refined.object_index)
                branch.object_poses.update({int(key): pose_array(pose) for key, pose in forecast.items()})
                branch.action_history.append(refined)
                branch.trajectories.append({})
                branch.contact_points.clear()
                branch.simulation_settled = True
                branch.failed = False
                branch.terminated = len(branch.placed_object_indices) == len(branch.object_ids)
                records.append(record)
                states.append(branch.copy())
                target_changes[str(original.step)] = delta
        finally:
            if structural is not None:
                structural.close()
            if coordinator is not None:
                coordinator.placement_backend.close()
        check()
        summary = {
            'kind': 'tabletop_fixed_order_scene_refinement', 'step': step,
            'max_translation_m': MAX_TRANSLATION, 'max_rotation_rad': float(MAX_ROTATION),
            'planning_mode': ('local_refinement' if all(_is_local(d) for d in
                              [*changes.values(), *target_changes.values()]) else 'motion_regeneration'),
            'scene_changes': changes, 'target_changes': target_changes,
            'rollout': rollout, 'source_hashes': dict(candidate.source_hashes),
            'entity_to_object_id': {
                str(index + 1): corrected.object_ids[index]
                for index in corrected.placed_object_indices},
            'observed_world_poses': {
                str(corrected.object_ids[int(entity) - 1]): np.asarray(pose).tolist()
                for entity, pose in zip(candidate.result.entity_ids, candidate.result.proposed_poses)},
            'remaining_steps': [record.step for record in records],
            'object_order': [record.problem.action.object_id for record in records],
            'validated_through_step': records[-1].step if records else step,
            'unvalidated_steps': [record.step for record in future[len(records):]],
            'deferred_failure': deferred,
            'mcts_used': False,
        }
        return SceneRefinement(corrected, tuple(records), tuple(states), summary)
    except (KeyError, TypeError) as error:
        raise RefinementNeedsReview(f'Recorded scene-refinement inputs are incomplete: {error}') from error
