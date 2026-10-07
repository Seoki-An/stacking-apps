"""Synthetic observation visibility, explicit frames, and portable SceneID inputs."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from stacking_common.planning.sceneid_fixture import load_fixture, pose_errors
from stacking_tabletop.acquisition.stack_scene_fixture import observed_pixels, voxel_rgb


def test_observation_keeps_first_hit_stones_and_transforms_camera_frame():
    depth = np.array([[1., 2., 0.], [3., 4., np.inf]])
    labels = np.array([[0, 1, 0], [2, 4294967295, 0]], dtype=np.uint32)
    rgb = np.arange(18, dtype=np.uint8).reshape(2,3,3)
    pose = np.diag([-1.,-1.,1.,1.]); pose[:3,3] = [.5,.2,.1]
    xyz, colors = observed_pixels(depth, labels, np.eye(3), pose, 2, rgb, 0., np.random.default_rng(7))
    np.testing.assert_allclose(xyz, [[.5,.2,1.1],[-1.5,.2,2.1]])
    np.testing.assert_array_equal(colors, rgb[0,:2])


def test_noise_is_reproducible_and_rgb_voxels_remain_aligned():
    depth = np.ones((2,2)); labels=np.zeros((2,2),int); rgb=np.full((2,2,3),127,np.uint8)
    args = depth, labels, np.eye(3), np.eye(4), 1, rgb, .0005
    first = observed_pixels(*args, np.random.default_rng(9))
    second = observed_pixels(*args, np.random.default_rng(9))
    np.testing.assert_array_equal(first[0],second[0])
    assert not np.all(first[0][:,2] == 1)
    points, colors = voxel_rgb(np.array([[0.,0.,0.],[.0002,0.,0.],[.01,0.,0.]]),
                               np.array([[255,0,0],[0,255,0],[0,0,255]]),.001)
    np.testing.assert_allclose(points, [[.0001,0,0],[.01,0,0]])
    np.testing.assert_allclose(colors, [[.5,.5,0],[0,0,1]])


def write_json(path, value):
    path.write_text(json.dumps(value))


def rehash(directory):
    manifest=json.loads((directory/'fixture.json').read_text())
    manifest['files']={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in directory.rglob('*') if p.is_file() and p.name != 'fixture.json'}
    write_json(directory/'fixture.json',manifest)


@pytest.fixture
def portable(tmp_path):
    directory=tmp_path/'fixture';directory.mkdir();(directory/'models').mkdir()
    cube=np.array([[x,y,z] for x in (-.02,.02) for y in (-.02,.02) for z in (-.02,.02)])
    pose=np.array([.4,.2,.2,0,0,0,1.]);prior=pose.copy();prior[0]+=.003
    np.savez(directory/'models/stone.npz',component_0=cube, mass=.1,inertia=np.eye(3)*1e-4,
             sharpnesses=[10],frictions=[.5],surface_points=cube)
    np.savez(directory/'observation.npz',points=cube+pose[:3],colors=np.ones_like(cube))
    for name,p in [('initial_poses',prior),('ground_truth',pose)]:
        write_json(directory/(name+'.json'),dict(frame='base_link',pose_convention='xyz_qxqyqzqw',
            stones=[dict(entity_id=1,stone_id=9,pose=p.tolist())]))
    write_json(directory/'fixture.json',dict(schema_version=1,kind='synthetic_sceneid_fixture',
        frame='base_link',frame_id=1,placed_entity_id=1,
        ground=dict(entity_id=0,geometry_id=0,friction=.5,pose=[0,0,.1,0,0,0,1]),
        stones=[dict(entity_id=1,stone_id=9,model_file='models/stone.npz')]))
    rehash(directory)
    return directory


def test_fixture_moves_without_source_assets_and_does_not_initialize_from_truth(portable,tmp_path):
    import stacking_core as core
    moved=tmp_path/'moved';shutil.move(portable,moved)
    problem,metadata,initial,truth=load_fixture(moved)
    config=core.SceneIdConfig()
    prior=core.evaluate_sceneid_scan(problem,config)
    exact=core.evaluate_sceneid_scan(problem,config,np.array([truth['stones'][0]['pose']]))
    assert exact.loss < prior.loss
    errors=pose_errors([initial['stones'][0]['pose']],[truth['stones'][0]['pose']])
    assert errors[0]['translation_mm']==pytest.approx(3)
    # Modifying evaluation-only GT does not alter the solver's initial scene.
    truth['stones'][0]['pose'][0]+=.1;write_json(moved/'ground_truth.json',truth);rehash(moved)
    other,_,_,_=load_fixture(moved)
    assert core.evaluate_sceneid_scan(other,config).loss==pytest.approx(prior.loss)


def test_fixture_rejects_changed_assets_and_mixed_frames(portable):
    with (portable/'models/stone.npz').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):load_fixture(portable)
    rehash(portable)
    gt=json.loads((portable/'ground_truth.json').read_text());gt['frame']='stack'
    write_json(portable/'ground_truth.json',gt);rehash(portable)
    with pytest.raises(ValueError,match='frame or convention'):load_fixture(portable)


def test_rotation_error_handles_quaternion_sign():
    pose=np.array([[1.,2.,3.,0.,0.,1.,0.]])
    other=pose.copy();other[:,3:]*=-1
    assert pose_errors(pose,other)==[dict(translation_mm=0.,rotation_deg=0.)]


def test_sceneid_visualization_uses_entity_ids_and_preserves_rgb(portable):
    from scipy.spatial.transform import Rotation
    from stacking_common.rendering.sceneid import comparison_layers, result_title

    _, metadata, initial, truth = load_fixture(portable)
    metadata['stones'].append(dict(metadata['stones'][0], entity_id=2, stone_id=10))
    rotated = [.1, .2, .3, 0, 0, np.sqrt(.5), np.sqrt(.5)]
    result = dict(frame='base_link', status='SceneIdStatus.ITERATION_LIMIT',
                  accepted_snapshot=False, proposed_poses=[
                      dict(entity_id=2, pose=rotated),
                      dict(entity_id=1, pose=truth['stones'][0]['pose'])])
    layers = comparison_layers(portable, metadata, initial, truth, result)
    with np.load(portable/'observation.npz') as data:
        np.testing.assert_array_equal(layers[0]['points'], data['points'])
        np.testing.assert_array_equal(layers[0]['colors'], data['colors'])
    proposed = [x for x in layers if x['group'] == 'Proposed']
    assert 'stone 10 (entity 2)' in proposed[0]['label']
    with np.load(portable/'models/stone.npz') as data:
        expected = data['surface_points'] @ Rotation.from_quat(rotated[3:]).as_matrix().T + rotated[:3]
    np.testing.assert_allclose(proposed[0]['points'], expected)
    np.testing.assert_allclose(proposed[1]['points'], layers[0]['points'])
    assert 'NO accepted snapshot' in result_title(result)
    result['frame'] = 'other'
    with pytest.raises(ValueError, match='frame'):
        comparison_layers(portable, metadata, initial, truth, result)


def test_sceneid_html_handles_missing_proposal_and_rejects_unknown_entities(portable):
    from stacking_common.rendering.sceneid import comparison_layers, write_comparison

    _, metadata, initial, truth = load_fixture(portable)
    result = dict(frame='base_link', status='failed', proposed_poses=[])
    layers = comparison_layers(portable, metadata, initial, truth, result)
    assert {x['group'] for x in layers} == {'Captured RGB', 'Initial', 'Ground truth'}
    path = write_comparison(portable/'visualization/overview.html', layers, result)
    assert 'NO accepted snapshot' in path.read_text()
    result['proposed_poses'] = [dict(entity_id=99, pose=truth['stones'][0]['pose'])]
    with pytest.raises(ValueError, match='unknown or duplicate'):
        comparison_layers(portable, metadata, initial, truth, result)
