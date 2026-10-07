"""Export occluded synthetic stack observations and GT for stacking-core SceneID."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation

from stacking_common.assets import _parse_object_asset
from stacking_common.multiview.synthetic import sample_surface
from .calibration import BASE_LINK_T_UR_BASE
from .hand_eye import HandEyeCalibration
from .hand_eye_capture import atomic_json, digest
from .stack_scene_simulation import ROOT, simulate
from stacking_tabletop.composition import TabletopApplication


def observed_pixels(depth, geometry_ids, matrix, camera, stone_count, rgb, noise_m, rng):
    """First-hit stone pixels only; never sample hidden catalog faces as observations."""
    valid = np.isfinite(depth) & (depth > 0) & (geometry_ids >= 0) & (geometry_ids < stone_count)
    y, x = np.nonzero(valid)
    z = depth[valid].astype(float)
    z += rng.normal(0., noise_m, len(z))
    keep = z > 0
    pixels = np.column_stack((x[keep], y[keep], np.ones(keep.sum())))
    points = (pixels @ np.linalg.inv(matrix).T)*z[keep, None]
    return points @ camera[:3, :3].T + camera[:3, 3], rgb[valid][keep]


def voxel_rgb(points, colors, size):
    _, inverse = np.unique(np.floor(points/size).astype(np.int64), axis=0, return_inverse=True)
    count = np.bincount(inverse)
    sums = np.zeros((len(count), 3)); rgb = np.zeros_like(sums)
    np.add.at(sums, inverse, points); np.add.at(rgb, inverse, colors)
    return sums/count[:, None], rgb/count[:, None]/255.


def write_cloud(path, points, colors):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise RuntimeError(f'could not write {path}')


def export_fixture(simulation, output, *, repository=ROOT, noise_mm=.5, seed=7):
    if not np.isfinite(noise_mm) or noise_mm < 0:
        raise ValueError('noise_mm must be finite and nonnegative')
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    simulation, output, repository = [Path(p).resolve() for p in (simulation, output, repository)]
    report = json.loads((simulation/'summary.json').read_text())
    if report.get('kind') != 'stacked_scene_capture_simulation' or report.get('status') != 'complete':
        raise ValueError('requires a complete stacked-scene simulation')
    if report.get('scene_frame') != 'world_ur_base' or report.get('data_origin') != 'synthetic':
        raise ValueError('unsupported simulation frame or data origin')
    plan = json.loads((simulation/'plan.json').read_text())
    region = json.loads((simulation/'region.json').read_text())
    if region['frame'] != 'base_link' or plan['output_frame'] != 'base_link':
        raise ValueError('camera route and region must use base_link')
    calibration = HandEyeCalibration.load(simulation/'calibration/profile.yml')
    app = TabletopApplication.from_repository(repository)
    records = report['stones']
    ids = [s['stone_id'] for s in records]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('fixture needs distinct catalog stone IDs')
    output.mkdir(parents=True, exist_ok=False)
    (output/'models').mkdir(); (output/'views').mkdir()
    rng = np.random.default_rng(seed)
    clouds, colors, view_records = [], [], []
    for view in plan['views']:
        label = view['label']; source = simulation/'views'/label
        with np.load(source/'ideal_depth.npz', allow_pickle=False) as data:
            depth, labels = data['depth_m'], data['geometry_ids']
        rgb = np.array(Image.open(source/'rgb.png').convert('RGB'))
        camera = np.asarray(view['base_link_T_color'])
        points, color = observed_pixels(depth, labels, calibration.color_matrix, camera,
                                       len(records), rgb, noise_mm/1000., rng)
        if not len(points):
            raise ValueError(f'view {label} sees no stones')
        write_cloud(output/'views'/f'{label}.pcd', points, color/255.)
        shutil.copyfile(source/'rgb.png', output/'views'/f'{label}.png')
        clouds.append(points); colors.append(color)
        view_records.append(dict(label=label, pcd=f'views/{label}.pcd', rgb=f'views/{label}.png',
            point_count=len(points), base_link_T_color=camera.tolist(), K=calibration.color_matrix.tolist()))
    points, colors = voxel_rgb(np.concatenate(clouds), np.concatenate(colors), .001)
    np.savez_compressed(output/'observation.npz', points=points, colors=colors)
    write_cloud(output/'observation.pcd', points, colors)
    truth, initial, models = [], [], []
    # Use a separate RNG so pose perturbations do not depend on image size/noise.
    prior_rng = np.random.default_rng(seed+1)
    for index, record in enumerate(records):
        stone = record['stone_id']; entity = index+1
        mesh_path = Path(record['mesh'])
        if digest(mesh_path) != record['mesh_sha256']:
            raise ValueError(f'mesh changed since rendering: stone {stone}')
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        surface, _ = sample_surface(np.asarray(mesh.vertices), np.asarray(mesh.triangles), count=12000, seed=stone)
        obj_path = repository/app.inventory_profile.model_directory/f'model_{stone}.obj'
        asset = _parse_object_asset(obj_path)
        density = dict(app.inventory_profile.density_overrides).get(stone, app.inventory_profile.density)
        arrays = dict(surface_points=surface, mass=density*asset.mass, inertia=density*asset.inertia,
            frictions=[c.friction for c in asset.components],
            sharpnesses=[int(round(c.geometry.sharpness)) for c in asset.components])
        arrays.update({f'component_{i}': c.geometry.points for i, c in enumerate(asset.components)})
        model_file = f'models/stone_{stone}.npz'
        np.savez_compressed(output/model_file, **arrays)
        shutil.copyfile(mesh_path, output/'models'/f'stone_{stone}.ply')
        world = np.asarray(record['world_pose'])
        rotation = Rotation.from_matrix(BASE_LINK_T_UR_BASE[:3, :3])*Rotation.from_quat(world[3:])
        position = BASE_LINK_T_UR_BASE[:3, :3]@world[:3]+BASE_LINK_T_UR_BASE[:3, 3]
        pose = np.r_[position, rotation.as_quat()]
        translation = prior_rng.normal(size=3); translation *= .003/np.linalg.norm(translation)
        angle = prior_rng.normal(size=3); angle *= np.deg2rad(3)/np.linalg.norm(angle)
        start = np.r_[position+translation, (Rotation.from_rotvec(angle)*rotation).as_quat()]
        truth.append(dict(entity_id=entity, stone_id=stone, pose=pose.tolist()))
        initial.append(dict(entity_id=entity, stone_id=stone, pose=start.tolist()))
        models.append(dict(entity_id=entity, stone_id=stone, model_file=model_file,
            mesh_file=f'models/stone_{stone}.ply', source_mesh_sha256=digest(mesh_path),
            source_dsf_sha256=digest(obj_path), density=density))
    common = dict(frame='base_link', units='metres', pose_convention='xyz_qxqyqzqw')
    atomic_json(output/'ground_truth.json', dict(**common, stones=truth, physical_stability_verified=False))
    atomic_json(output/'initial_poses.json', dict(**common, stones=initial,
        perturbation='3 mm translation norm and 3 degrees rotation about the body origin, in scene axes'))
    ground_z = float(region['support_plane'][2])
    metadata = dict(schema_version=1, kind='synthetic_sceneid_fixture', data_origin='synthetic',
        **common, frame_id=1, placed_entity_id=models[-1]['entity_id'], stones=models,
        ground=dict(entity_id=0, geometry_id=0, pose=[0,0,ground_z,0,0,0,1],
                    friction=app.simulation_profile.placement.ground_friction),
        observation='observation.npz', observation_pcd='observation.pcd', views=view_records,
        point_count=len(points), voxel_size_m=.001, axial_depth_noise_std_m=noise_mm/1000., seed=seed,
        source_simulation=str(simulation), source_summary_sha256=digest(simulation/'summary.json'),
        source_snapshot_step=report['source_snapshot_step'],
        limitations=['Observations are raycast first-hit surfaces with optional independent axial Gaussian noise; not RGB stereo reconstruction.',
            'Renderer stone/background labels provide an ideal foreground mask; per-stone labels are not supplied to the solver.',
            'Ground truth is the geometry used for rendering; physical stability and agreement of DSF and visual surfaces are not guaranteed.',
            'Initial poses are synthetic local perturbations of ground truth, not blind identification estimates.',
            'All poses and observations are in base_link, in metres; model surface samples stay in their body/catalog frame.'])
    metadata['files'] = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    atomic_json(output/'fixture.json', metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--simulation', type=Path, help='reuse rendered stacked-scene simulation')
    source.add_argument('--run', type=Path, help='render a saved tabletop .stack snapshot first; no GPU model needed')
    parser.add_argument('--step', type=int, help='snapshot step when using --run')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--noise-mm', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if args.step is not None and args.run is None:
        parser.error('--step requires --run')
    if not np.isfinite(args.noise_mm) or args.noise_mm < 0 or args.seed < 0:
        parser.error('noise must be finite and nonnegative; seed must be nonnegative')
    if args.output.exists():
        parser.error('choose a new output directory')
    simulation = args.simulation
    if args.run:
        simulation = args.output.with_name(args.output.name+'_render')
        simulate(args.run, simulation, step=args.step, capture_only=True)
    result = export_fixture(simulation, args.output, noise_mm=args.noise_mm, seed=args.seed)
    print(json.dumps(dict(fixture=str(args.output), stones=len(result['stones']), points=result['point_count']), indent=2))


if __name__ == '__main__':
    main()
