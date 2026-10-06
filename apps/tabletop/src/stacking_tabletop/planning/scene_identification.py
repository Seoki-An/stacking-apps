"""Offline tabletop handoff from reviewed stacked-scene RGB capture to replanning."""

from pathlib import Path

import numpy as np

from stacking_common.planning import CoreSceneBuilder, SceneMobility, SceneObjectInstance
from stacking_common.planning.frames import FrameTransform
from stacking_common.planning.scene_identification import (
    load_mesh_surfaces, load_reviewed_stack_capture,
    propose_scene_identification, review_rollout_and_replan,
)


_IDENTITY = np.array([0., 0., 0., 0., 0., 0., 1.])
# Calibrated tabletop base_link and planning world differ by a half-turn about Z.
_WORLD_T_BASE_LINK = np.array([0., 0., 0., 0., 0., 1., 0.])


def prepare_tabletop_stack_candidate(application, inventory, state,
                                     capture_directory, sceneid_config,
                                     *, scan_voxel_size, mesh_voxel_size,
                                     cuda_scan=False):
    """Keep observed PCD, mesh assets and DSF bodies in explicit frames."""
    import stacking_core as core

    planning = application.planning_profile
    world_T_stack = planning.world_T_stack
    placed = tuple(state.placed_object_indices)
    if not placed:
        raise ValueError('SceneID requires at least one placed stone')
    if tuple(state.object_ids) != inventory.object_ids:
        raise ValueError('scene state and inventory IDs differ')
    if len(set(placed)) != len(placed):
        raise ValueError('duplicate placed stone index')
    cloud, frame, hashes = load_reviewed_stack_capture(
        capture_directory, voxel_size=scan_voxel_size)
    if frame != 'base_link':
        raise ValueError('tabletop stacked-scene capture must be in base_link')
    builder = CoreSceneBuilder(inventory.models, planning.frame_id, core)
    objects = []
    mesh_paths = {}
    for index in placed:
        object_id = state.object_ids[index]
        pose = world_T_stack.child_pose_to_parent(state.object_poses[object_id])
        objects.append(SceneObjectInstance(index, object_id, pose,
                                           SceneMobility.DYNAMIC))
        mesh_paths[builder.entity_id(index)] = (
            Path(application.repository_root) / application.inventory_profile.model_directory /
            f'model_{object_id}_mesh.ply')
    surfaces, mesh_hashes = load_mesh_surfaces(
        mesh_paths, voxel_size=mesh_voxel_size)
    hashes.update(mesh_hashes)
    ground_pose = _IDENTITY.copy()
    ground_pose[2] = application.simulation_profile.placement.ground_z
    ground_pose = world_T_stack.child_pose_to_parent(ground_pose)
    ground = core.BodyInstance(0, core.BodyModel.plane(
        0, friction=application.simulation_profile.placement.ground_friction,
        geometry_id=0), ground_pose, mobility=core.Mobility.STATIC)
    snapshot = builder.build(objects, fixed_bodies=[ground])
    return propose_scene_identification(
        snapshot, scene_frame=world_T_stack.parent_frame,
        observation=cloud, observation_frame=frame,
        scene_T_observation=FrameTransform(
            world_T_stack.parent_frame, frame, _WORLD_T_BASE_LINK),
        surfaces=surfaces, placed_entity=builder.entity_id(placed[-1]),
        ground_entity=0, ground_geometry=0, config=sceneid_config,
        source_hashes=hashes, cuda_scan=cuda_scan)


def accept_tabletop_stack_candidate(application, inventory, state, candidate,
                                    *, review, replan,
                                    max_translation, max_rotation):
    """Review, run the application physics profile, then branch planning."""
    import stacking_core as core

    placement = application.simulation_profile.placement
    return review_rollout_and_replan(
        candidate, review=review,
        simulation_config=application.simulation_profile.native_config(core),
        dt=placement.settle_dt, steps=placement.settle_steps,
        max_translation=max_translation, max_rotation=max_rotation,
        state=state,
        entity_to_object_id={index + 1: inventory.object_ids[index]
                             for index in state.placed_object_indices},
        world_T_stack=application.planning_profile.world_T_stack,
        replan=replan)
