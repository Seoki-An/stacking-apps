"""Reviewable SceneID proposal and physical handoff for placed stacks."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from stacking_common.planning.frames import FrameTransform


@dataclass(frozen=True)
class SceneIdentificationCandidate:
    problem: object
    result: object
    scene_frame: str
    observation_frame: str
    scene_T_observation: FrameTransform | None
    source_hashes: Mapping[str, str]

    @property
    def ready_for_review(self) -> bool:
        import stacking_core as core
        return (self.result.status == core.SceneIdStatus.IDENTIFIED
                and self.result.snapshot is not None)


def _points(value, name):
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError(f'{name} must be a nonempty finite N x 3 array')
    return np.ascontiguousarray(points)


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_mesh_surfaces(paths: Mapping[int, Path], *, voxel_size):
    """Sample known visual triangles once in their catalog body frames."""
    import open3d as o3d

    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError('mesh voxel_size must be positive')
    surfaces, hashes = {}, {}
    for entity, source in paths.items():
        path = Path(source).resolve()
        mesh = o3d.io.read_triangle_mesh(str(path))
        vertices = _points(np.asarray(mesh.vertices), f'stone {entity} vertices')
        faces = np.asarray(mesh.triangles, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
            raise ValueError(f'stone {entity} mesh has no triangles')
        points = np.concatenate((vertices, vertices[faces].mean(axis=1)))
        cells = np.floor(points / voxel_size).astype(np.int64)
        _, indices = np.unique(cells, axis=0, return_index=True)
        surfaces[int(entity)] = points[np.sort(indices)]
        hashes[f'mesh/{int(entity)}'] = _digest(path)
    return surfaces, hashes


def load_reviewed_stack_capture(directory, *, voxel_size):
    """Load only a complete, hash-matched post-placement PCD and its frame."""
    from stacking_common.reconstruction import load_point_cloud

    directory = Path(directory).resolve()
    metadata = json.loads((directory / 'capture.json').read_text())
    if metadata.get('kind') != 'stacked_scene_wrist_capture' or metadata.get('status') != 'complete':
        raise ValueError('stacked-scene capture is not complete')
    frame = metadata.get('output_frame')
    if not isinstance(frame, str) or not frame:
        raise ValueError('stacked-scene capture has no output frame')
    path = Path(metadata['pcd']).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError('stacked-scene PCD is missing or outside the capture')
    digest = _digest(path)
    if digest != metadata['pcd_sha256']:
        raise ValueError('stacked-scene PCD hash differs from the reviewed capture')
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError('voxel_size must be positive')
    points = _points(load_point_cloud(path).points, 'capture PCD')
    # One deterministic sample per voxel; scan crop remains fixed throughout a solve.
    cells = np.floor(points / voxel_size).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(indices)], frame, {'pcd': digest}


def transform_cloud(points, source_frame, scene_frame, scene_T_source):
    points = _points(points, 'observation')
    if source_frame == scene_frame:
        if scene_T_source is not None:
            raise ValueError('same-frame observation must not provide a transform')
        return points.copy()
    if (scene_T_source is None or
            scene_T_source.parent_frame != scene_frame or
            scene_T_source.child_frame != source_frame):
        raise ValueError('observation transform must explicitly map source to scene frame')
    pose = scene_T_source.parent_T_child
    transformed = Rotation.from_quat(pose[3:]).apply(points) + pose[:3]
    return _points(transformed, 'transformed observation')


def propose_scene_identification(snapshot, *, scene_frame, observation,
                                 observation_frame, scene_T_observation,
                                 surfaces, placed_entity, ground_entity,
                                 ground_geometry, config, source_hashes,
                                 cuda_scan=False):
    """Fit a fixed scan against supplied mesh samples and one canonical scene."""
    import stacking_core as core

    cloud = transform_cloud(observation, observation_frame, scene_frame,
                            scene_T_observation)
    body_ids = [int(body.id) for body in snapshot.bodies]
    stone_ids = tuple(int(value) for value in surfaces)
    if set(stone_ids) - set(body_ids) or placed_entity not in stone_ids:
        raise ValueError('SceneID surfaces must name scene stones and the placed stone')
    if ground_entity not in body_ids:
        raise ValueError('SceneID ground is not in the scene')
    stones = [core.SceneIdStone(entity, core.SceneIdSurface(
        _points(samples, f'stone {entity} mesh samples'))) for entity, samples in surfaces.items()]
    problem = core.SceneIdProblem(
        core.SceneView(snapshot, body_ids), snapshot.frame_id, cloud,
        ground_entity, ground_geometry, placed_entity, stones,
    )
    result = (core.SceneIdCudaScan(problem, config).solve() if cuda_scan
              else core.SceneIdentifier(config).solve(problem))
    return SceneIdentificationCandidate(
        problem, result, scene_frame, observation_frame,
        scene_T_observation, dict(source_hashes),
    )


def check_scene_rollout(candidate, *, simulation_config, dt, steps,
                        max_translation, max_rotation):
    """Require a settled native rollout before an observed pose reaches search."""
    import stacking_core as core

    if not candidate.ready_for_review:
        raise ValueError('SceneID has no evidence-qualified, gap-screened snapshot')
    if (not np.isfinite(dt) or dt <= 0 or steps <= 0 or
            not np.isfinite(max_translation) or max_translation < 0 or
            not np.isfinite(max_rotation) or max_rotation < 0):
        raise ValueError('rollout duration and drift thresholds are invalid')
    simulator = core.Simulator(simulation_config)
    simulator.clear_warm_start()
    initial = candidate.result.snapshot
    current = initial
    stone_ids = candidate.result.entity_ids
    worst_translation = worst_rotation = 0.
    for _ in range(steps):
        outcome = simulator.step(current, dt)
        if not outcome.solver.converged:
            raise ValueError('SceneID rollout solver did not converge')
        current = outcome.snapshot
        for entity in stone_ids:
            before = initial.body_pose(entity)
            after = current.body_pose(entity)
            worst_translation = max(worst_translation, float(np.linalg.norm(after[:3] - before[:3])))
            dot = abs(float(np.dot(before[3:], after[3:])))
            worst_rotation = max(worst_rotation, 2 * float(np.arccos(np.clip(dot, -1., 1.))))
        if worst_translation > max_translation or worst_rotation > max_rotation:
            raise ValueError('SceneID rollout moved a placed stone beyond the accepted tolerance')
    return {'steps': steps, 'dt': dt, 'max_translation': worst_translation,
            'max_rotation': worst_rotation}


def review_rollout_and_replan(candidate, *, review: Callable,
                              simulation_config, dt, steps,
                              max_translation, max_rotation,
                              state, entity_to_object_id: Mapping[int, int],
                              world_T_stack: FrameTransform, replan: Callable):
    """Call replanning only after human review and native rollout pass."""
    if not candidate.ready_for_review:
        raise ValueError('SceneID candidate is not qualified for review')
    if not review(candidate):
        raise ValueError('SceneID candidate was not accepted in review')
    rollout = check_scene_rollout(
        candidate, simulation_config=simulation_config, dt=dt, steps=steps,
        max_translation=max_translation, max_rotation=max_rotation)
    if world_T_stack.parent_frame != candidate.scene_frame:
        raise ValueError('world_T_stack does not start in the SceneID scene frame')
    branch = state.copy()
    for entity, pose in zip(candidate.result.entity_ids, candidate.result.proposed_poses):
        object_id = entity_to_object_id[int(entity)]
        if object_id not in branch.object_ids:
            raise ValueError('SceneID entity has no matching planning object')
        branch.object_poses[object_id] = world_T_stack.parent_pose_to_child(
            np.asarray(pose, dtype=float))
    for evidence in candidate.result.evidence:
        if not evidence.prior_carried:
            branch.pose_identified_object_ids.add(entity_to_object_id[int(evidence.entity_id)])
    branch.contact_points.clear()
    branch.simulation_settled = True
    return replan(branch), rollout
