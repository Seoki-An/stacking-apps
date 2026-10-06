import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest

import stacking_core as core
from stacking_common.domain import State
from stacking_common.planning.frames import FrameTransform
from stacking_common.pointcloud import PointCloudData
from stacking_common.reconstruction import write_point_cloud
from stacking_common.planning.scene_identification import (
    load_reviewed_stack_capture, propose_scene_identification,
    review_rollout_and_replan,
)
from stacking_tabletop.planning.scene_identification import prepare_tabletop_stack_candidate


IDENTITY = np.array([0., 0., 0., 0., 0., 0., 1.])
SAMPLES = np.array([[0., 0., .5], [.2, 0., .5],
                    [0., .2, .5], [.2, .2, .6]])


def _cube_model():
    nodes = np.array([[x, y, z] for x in (-.4, .4)
                      for y in (-.4, .4) for z in (-.4, .4)])
    return core.BodyModel.dsf(1, nodes, 1., np.eye(3))


def _config():
    config = core.SceneIdConfig()
    config.scan_sigma = .1
    config.outlier_cutoff = .25
    config.reference_scan_points = 4.
    return config


def test_occluded_lower_stone_and_handoff_gates():
    lower = IDENTITY.copy(); lower[2] = .4
    upper = IDENTITY.copy(); upper[2] = 1.15  # Initially penetrates lower.
    model = _cube_model()
    scene = core.SceneSnapshot(7, [
        core.BodyInstance(0, core.BodyModel.plane(2), IDENTITY,
                          mobility=core.Mobility.STATIC),
        core.BodyInstance(1, model, lower),
        core.BodyInstance(2, model, upper),
    ])
    top = np.array([[x, y, .4] for x in (-.3, .3) for y in (-.3, .3)])
    observed = top + np.array([0., 0., 1.2])
    candidate = propose_scene_identification(
        scene, scene_frame='world', observation=observed,
        observation_frame='world', scene_T_observation=None,
        surfaces={1: top, 2: top}, placed_entity=2,
        ground_entity=0, ground_geometry=1, config=_config(),
        source_hashes={'pcd': 'a' * 64})
    result = candidate.result
    assert candidate.ready_for_review
    assert result.gap_screen.nonpenetrating
    assert result.evidence[0].prior_carried
    assert not result.evidence[1].prior_carried
    assert result.proposed_poses[0][2] > lower[2]
    assert result.proposed_poses[1][2] > upper[2]

    state = State([11, 22], [0, 1], {11: lower, 22: upper}, [], [])
    called = []
    arguments = dict(
        candidate=candidate, simulation_config=core.SimulationConfig(),
        dt=.001, steps=1, max_translation=.01, max_rotation=.01,
        state=state, entity_to_object_id={1: 11, 2: 22},
        world_T_stack=FrameTransform('world', 'stack', IDENTITY),
        replan=lambda branch: called.append(branch) or branch,
    )
    with pytest.raises(ValueError, match='not accepted'):
        review_rollout_and_replan(review=lambda _: False, **arguments)
    assert not called
    with pytest.raises(ValueError, match='moved a placed stone'):
        review_rollout_and_replan(
            review=lambda _: True, **{**arguments, 'max_translation': 0.})
    assert not called
    branch, rollout = review_rollout_and_replan(review=lambda _: True, **arguments)
    assert branch is called[0]
    assert rollout['steps'] == 1
    assert 22 in branch.pose_identified_object_ids
    assert 11 not in branch.pose_identified_object_ids
    assert np.allclose(branch.object_poses[22], result.proposed_poses[1])
    assert np.allclose(state.object_poses[22], upper)


def test_reviewed_capture_and_tabletop_frame(tmp_path):
    assets = tmp_path / 'assets'
    assets.mkdir()
    vertices = SAMPLES.copy()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [1, 2, 3]]))
    mesh_path = assets / 'model_1_mesh.ply'
    assert o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    capture = tmp_path / 'capture'
    capture.mkdir()
    world = vertices + np.array([0., 0., .5])
    base_link = world * np.array([-1., -1., 1.])
    pcd = capture / 'stack.pcd'
    write_point_cloud(pcd, PointCloudData(base_link))
    digest = hashlib.sha256(pcd.read_bytes()).hexdigest()
    record = dict(kind='stacked_scene_wrist_capture', status='complete',
                  output_frame='base_link', pcd=str(pcd), pcd_sha256=digest)
    (capture / 'capture.json').write_text(json.dumps(record))
    points, frame, hashes = load_reviewed_stack_capture(capture, voxel_size=.001)
    assert frame == 'base_link' and hashes['pcd'] == digest
    assert len(points) == len(vertices)
    (capture / 'capture.json').write_text(json.dumps({**record, 'pcd_sha256': '0' * 64}))
    with pytest.raises(ValueError, match='hash differs'):
        load_reviewed_stack_capture(capture, voxel_size=.001)
    (capture / 'capture.json').write_text(json.dumps(record))

    pose = IDENTITY.copy(); pose[2] = .5
    state = State([1], [0], {1: pose}, [], [])
    profile = SimpleNamespace(world_T_stack=FrameTransform('world', 'stack', IDENTITY),
                              frame_id=7)
    placement = SimpleNamespace(ground_z=0., ground_friction=1.)
    application = SimpleNamespace(
        repository_root=tmp_path, planning_profile=profile,
        inventory_profile=SimpleNamespace(model_directory='assets'),
        simulation_profile=SimpleNamespace(placement=placement))
    inventory = SimpleNamespace(object_ids=(1,), models={1: _cube_model()})
    candidate = prepare_tabletop_stack_candidate(
        application, inventory, state, capture, _config(),
        scan_voxel_size=.001, mesh_voxel_size=.001)
    assert candidate.ready_for_review
    assert candidate.observation_frame == 'base_link'
    assert candidate.source_hashes['pcd'] == digest
    assert 'mesh/1' in candidate.source_hashes
    assert candidate.result.placed_evidence.inliers >= 3
    if hasattr(core, 'SceneIdCudaScan') and core.cuda_capabilities()['devices']:
        accelerated = prepare_tabletop_stack_candidate(
            application, inventory, state, capture, _config(),
            scan_voxel_size=.001, mesh_voxel_size=.001, cuda_scan=True)
        assert accelerated.ready_for_review
        assert accelerated.result.placed_evidence.inliers == candidate.result.placed_evidence.inliers
