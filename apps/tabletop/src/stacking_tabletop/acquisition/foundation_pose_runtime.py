"""Run learned pickup poses in the isolated FoundationPose environment."""
import argparse
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from .hand_eye_capture import atomic_json, digest
from .hand_eye_replay import read_json, inside


def check_runtime(repository):
    repository = Path(repository).resolve()
    python = repository / '.data/venvs/foundation-pose/bin/python'
    vendor = repository / '.data/vendor/FoundationPose'
    required = [python, vendor/'estimater.py'] + [vendor/'weights'/name/'model_best.pth'
        for name in ('2023-10-28-18-33-37', '2024-01-11-20-02-45')]
    if any(not p.is_file() for p in required):
        raise ValueError('FoundationPose runtime or weights are missing under .data; '
                         'install the tested runtime or explicitly select --pose-backend icp')
    return python, vendor


def infer(inputs, output, repository):
    """Child process: learned poses only, with no registration fallback."""
    from stacking_common.pointcloud.foundation_pose import FoundationPoseEstimator

    inputs, output, repository = [Path(p).resolve() for p in (inputs, output, repository)]
    manifest = read_json(inputs/'inputs.json')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(kind='foundation_pose_estimates', status='running', stones=[],
                  inputs_sha256=digest(inputs/'inputs.json'), icp_refinement_applied=False,
                  checkpoint_sha256={str(p.relative_to(repository)): digest(p)
                                     for p in (repository/'weights').glob('*/*.pth')})
    started = time.monotonic()
    try:
        estimator = FoundationPoseEstimator(repository)
        report['load_models_s'] = time.monotonic() - started
        for item in manifest['stones']:
            stone = item['stone_id']
            folder = inside(inputs, item['path'])
            mesh = Path(item['mesh_file'])
            for path, expected in [(folder/'input.npz', item['input_sha256']),
                                   (folder/'rgb.png', item['rgb_sha256']), (mesh, item['mesh_sha256'])]:
                if digest(path) != expected:
                    raise ValueError(f'FoundationPose input changed: {path}')
            print(f'FoundationPose: stone {stone}', flush=True)
            with np.load(folder/'input.npz', allow_pickle=False) as data:
                with Image.open(folder/'rgb.png') as image:
                    rgb = np.array(image.convert('RGB'))
                pose, elapsed = estimator.estimate(mesh, rgb, data['depth'], data['mask'],
                                                   data['matrix'], output/f'stone_{stone}/debug')
                base_pose = data['base_link_T_camera'] @ pose
            report['stones'].append(dict(stone_id=stone, camera_T_model=pose.tolist(),
                base_link_T_model=base_pose.tolist(), elapsed_s=elapsed,
                mesh_file=str(mesh), mesh_sha256=item['mesh_sha256']))
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['elapsed_s'] = time.monotonic()-started
        atomic_json(output/'summary.json', report)
    return report


def estimate_assigned(dense, region_file, assignments, catalog, output, repository):
    from .hand_eye_foundation_pose import prepare_assigned

    python, vendor = check_runtime(repository)
    meshes = {stone: catalog.path(stone).with_name(f'model_{stone}_mesh.ply').resolve()
              for stone in assignments.values()}
    for mesh in meshes.values():
        if not mesh.is_file():
            raise ValueError(f'FoundationPose needs a matching mesh in catalog coordinates: {mesh}; '
                             'supply it or select --pose-backend icp')
    inputs = Path(output)/'foundation_pose_inputs'
    results = Path(output)/'foundation_pose_results'
    manifest = prepare_assigned(dense, region_file, assignments, inputs)
    for item in manifest['stones']:
        mesh = meshes[item['stone_id']]
        item.update(mesh_file=str(mesh), mesh_sha256=digest(mesh))
    atomic_json(inputs/'inputs.json', manifest)
    command = [str(python), '-m', __name__, '--inputs', str(inputs), '--output', str(results),
               '--repository', str(vendor)]
    env = dict(os.environ, PYOPENGL_PLATFORM='egl')
    child = subprocess.Popen(command, env=env)
    try:
        code = child.wait()
        if code:
            raise RuntimeError(f'FoundationPose inference failed (exit {code}); see retained input/results')
    except BaseException:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise
    report = read_json(results/'summary.json')
    if (report.get('status') != 'complete' or report.get('icp_refinement_applied') is not False
            or report.get('inputs_sha256') != digest(inputs/'inputs.json')
            or len(report['stones']) != len(assignments)
            or {s['stone_id'] for s in report['stones']} != set(assignments.values())):
        raise ValueError('FoundationPose results do not match the reviewed inputs')
    return {item['stone_id']: item for item in report['stones']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('inputs', 'output', 'repository'):
        parser.add_argument('--'+key, type=Path, required=True)
    infer(**vars(parser.parse_args()))
