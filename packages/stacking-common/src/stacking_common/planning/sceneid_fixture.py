"""Load portable synthetic SceneID fixtures without ROS or platform imports."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def pose_errors(estimated, truth):
    estimated, truth = np.asarray(estimated), np.asarray(truth)
    if estimated.shape != truth.shape or estimated.ndim != 2 or estimated.shape[1] != 7:
        raise ValueError('pose arrays must both have shape (N, 7)')
    translation = np.linalg.norm(estimated[:, :3]-truth[:, :3], axis=1)*1000
    rotation = (Rotation.from_quat(estimated[:, 3:]) *
                Rotation.from_quat(truth[:, 3:]).inv()).magnitude()*180/np.pi
    return [dict(translation_mm=float(t), rotation_deg=float(r)) for t, r in zip(translation, rotation)]


def load_fixture(directory):
    """Build the native problem from initial poses; GT is returned only for scoring."""
    import stacking_core as core

    directory = Path(directory).resolve()
    manifest = json.loads((directory/'fixture.json').read_text())
    if manifest.get('schema_version') != 1 or manifest.get('kind') != 'synthetic_sceneid_fixture':
        raise ValueError('unsupported SceneID fixture')
    for name, expected in manifest['files'].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError('fixture file is missing or outside fixture directory')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'fixture hash mismatch: {name}')
    def checked(name):
        if name not in manifest['files']:
            raise ValueError(f'unverified fixture file: {name}')
        return directory/name

    with np.load(checked('observation.npz'), allow_pickle=False) as data:
        points = data['points'].copy()
    initial = json.loads(checked('initial_poses.json').read_text())
    truth = json.loads(checked('ground_truth.json').read_text())
    records = manifest['stones']
    ids = [s['entity_id'] for s in records]
    for poses in (initial, truth):
        if poses['frame'] != manifest['frame'] or poses['pose_convention'] != 'xyz_qxqyqzqw':
            raise ValueError('fixture pose frame or convention mismatch')
        if [s['entity_id'] for s in poses['stones']] != ids:
            raise ValueError('fixture entity order mismatch')
    ground = manifest['ground']
    bodies = [core.BodyInstance(ground['entity_id'], core.BodyModel.plane(
        0, friction=ground['friction'], geometry_id=ground['geometry_id']),
        np.asarray(ground['pose'], dtype=float), mobility=core.Mobility.STATIC)]
    stones = []
    for record, start in zip(records, initial['stones']):
        with np.load(checked(record['model_file']), allow_pickle=False) as model:
            body_model = core.BodyModel.dsf_components(record['stone_id'],
                [model[f'component_{i}'].copy() for i in range(len(model['frictions']))],
                mass=float(model['mass']), inertia=model['inertia'].copy(),
                frictions=model['frictions'].tolist(), sharpnesses=model['sharpnesses'].tolist())
            surface = core.SceneIdSurface(model['surface_points'].copy(),
                                          np.array([0., 0., 0., 0., 0., 0., 1.]))
        bodies.append(core.BodyInstance(record['entity_id'], body_model,
            np.asarray(start['pose'], dtype=float), mobility=core.Mobility.DYNAMIC))
        stones.append(core.SceneIdStone(record['entity_id'], surface))
    snapshot = core.SceneSnapshot(manifest['frame_id'], bodies)
    problem = core.SceneIdProblem(core.SceneView(snapshot, [ground['entity_id'], *ids]),
        manifest['frame_id'], points, ground['entity_id'], ground['geometry_id'],
        manifest['placed_entity_id'], stones)
    return problem, manifest, initial, truth
