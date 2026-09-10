"""Owned capture/delivery workers. Desktop identification runs separately.

Only capture receives the virtual-world snapshot. Identification receives a
verified cropped product and explicit plan/observation priors, never truth poses.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from stacking_common.pointcloud.raycast import simulate_lidar
from stacking_excavator.acquisition.capture import CaptureFrame, digest
from stacking_excavator.acquisition.emulation import look_at
from stacking_excavator.acquisition.spool import CaptureSpool
from stacking_excavator.acquisition.stationary import StationaryCaptureService
from stacking_excavator.acquisition.storage import atomic_json, prepare_crop, receive_local
from stacking_excavator.perception.models import ExcavatorModels
from stacking_excavator.perception.roi import enclosing_crop


def capture(request):
    import open3d as o3d
    root = Path(request['repository'])
    snapshot = request['snapshot']
    world = snapshot['world']
    priors = {int(k): np.array(v) for k, v in request['priors'].items()}
    models = ExcavatorModels(root)
    gripper = np.array(snapshot['world_T_gripper'])
    calibration = digest({'mode': 'emulated', 'rig': 'purpose_views_v1'})
    crop = enclosing_crop(request['kind'], models, priors,
        calibration_sha256=calibration, scene_revision=request['revision'],
        gripper_points=models.gripper(world['opening']).points if request['kind'] == 'inhand' else None,
        target_corners=request['target_corners'], ground_height=request['ground_height'])
    meshes = []
    for object_id, pose in world['poses'].items():
        mesh = o3d.io.read_triangle_mesh(str(root / f'apps/excavator/assets/objects/model_{object_id}_mesh.ply'))
        mesh.transform(np.array(pose))
        meshes.append(mesh)
    for part in snapshot['robot_parts']:
        mesh = o3d.io.read_triangle_mesh(part['path'])
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * part['scale'])
        mesh.transform(np.array(part['pose']))
        meshes.append(mesh)
    # Explicit emulated scan rig, positioned by the requested crop, not stone
    # truth. Each dwell records its actual sensor and gripper transforms.
    reference = gripper if request['kind'] == 'inhand' else np.eye(4)
    center = crop.reference_T_box[:3, 3]
    offsets = ([-2., 2., 1.], [2., 2., 1.], [0., 2.8, -.6]) if request['kind'] == 'inhand' else (
        [-2., -2., 1.5], [2., -2., 1.5], [0., 2.5, 1.5])
    class Source:
        def frames(self, job_id):
            for i, offset in enumerate(offsets):
                sensor = reference @ look_at(center + offset, center)
                points, rays = simulate_lidar(meshes, sensor)
                stamp = snapshot['timestamp_ns'] + i * 100_000_000
                yield CaptureFrame(f'frame-{i}', f'virtual-lidar-{i}', stamp, stamp,
                    sensor, gripper, calibration, points, tuple(world['joints']),
                    world['opening'], attributes={'ray_index': rays})
    spool = CaptureSpool(request['spool'], capacity_bytes=1024**3)
    service = StationaryCaptureService(spool, Source())
    raw = service.capture(request['job_id'], crop, maximum_bytes=128*1024**2, maximum_frames=3)
    directory = Path(request['spool']) / request['job_id']
    product = directory / 'crop'
    cropped = prepare_crop(directory / 'raw', product, request['job_id']+'-crop', crop,
                           maximum_timestamp_skew_ns=0)
    return {'product': str(product), 'manifest_sha256': cropped['sha256'],
            'raw_bytes': sum(f['bytes'] for f in raw['frames']),
            'cropped_bytes': sum(f['bytes'] for f in cropped['frames'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('capture', 'deliver'))
    parser.add_argument('request', type=Path)
    parser.add_argument('response', type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    result = (capture(request) if args.operation == 'capture' else
              {'delivered': receive_local(request['product'], request['destination'],
                                          byte_budget=request.get('byte_budget'))})
    atomic_json(args.response, result)


if __name__ == '__main__':
    main()
