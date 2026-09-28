"""Offline FoundationPose initializer experiment using saved Fast-FoundationStereo depth."""
import argparse
from dataclasses import replace
import inspect
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from stacking_common.multiview.pose_inputs import stereo_pose_input
from stacking_common.pointcloud import PointCloudData, Open3dRegistrationBackend, evaluate_registration
from stacking_common.rendering.pointcloud import render_clouds
from stacking_tabletop.perception.assignment import TabletopFieldMatcher
from stacking_tabletop.perception.field_cli import load_inputs
from .hand_eye_capture import atomic_json, digest
from .hand_eye_replay import read_json, inside
from .pick_scene_fusion import _balanced_observation, _metric, _read_cloud, _transform
from .pick_scene_rgb_first import _planning_T_base
from .pick_scene_workflow import build_parser


def prepare(dense, reference, output):
    dense, reference, output = [Path(p).resolve() for p in (dense, reference, output)]
    previous = read_json(reference/'fused_identification/pose_identification/stone_poses.json')
    if previous.get('status') != 'complete':
        raise ValueError('complete reviewed assignments are required')
    report = prepare_assigned(dense, reference/'rgb_region.json', previous['cluster_assignments'], output)
    report.update(reference=str(reference),
                  geometry_sha256=digest(reference/'femto_geometry/geometry.json'),
                  prior_assignment_sha256=digest(reference/'fused_identification/pose_identification/stone_poses.json'))
    atomic_json(output/'inputs.json', report)
    return report


def prepare_assigned(dense, region_file, assignments, output):
    """Prepare learned pose inputs from reviewed IDs, without a previous pose fit."""
    dense, region_file, output = [Path(p).resolve() for p in (dense, region_file, output)]
    summary = read_json(dense/'summary.json')
    region = read_json(region_file)
    if (summary.get('status') != 'complete' or summary.get('kind') != 'fast_foundation_stereo_experiment'
            or summary.get('region') != region):
        raise ValueError('FoundationPose requires matching Fast-FoundationStereo pair data; '
                         'reconstruct with --dense-backend fast-foundation-stereo or select --pose-backend icp')
    if output.is_relative_to(dense) or dense.is_relative_to(output):
        raise ValueError('pose input output must be separate from dense inputs')
    assignments = {int(k): int(v) for k, v in assignments.items()}
    if not assignments or len(set(assignments.values())) != len(assignments):
        raise ValueError('reviewed unique stone assignments are required')
    best = {}
    for pair in summary['pairs']:
        i, j = pair['views']
        path = dense/'refined'/f'pair_{i:03d}_{j:03d}.npz'
        with np.load(path, allow_pickle=False) as data:
            k = data['matrix']; pose = data['base_link_T_camera']
            for cluster, stone in assignments.items():
                depth, mask = stereo_pose_input(data['disparity'], data['valid'], k, pose,
                    float(data['baseline_m']), region['stone_boxes_xy'][cluster], region['support_plane'])
                count = int(mask.sum())
                if stone not in best or count > best[stone]['count']:
                    best[stone] = dict(count=count, depth=depth, mask=mask, matrix=k.copy(),
                                      pose=pose.copy(), source=path, cluster=cluster)
    if set(best) != set(assignments.values()) or any(v['count'] < 100 for v in best.values()):
        raise ValueError('a selected stone has fewer than 100 valid stereo mask pixels')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, kind='foundation_pose_inputs', status='complete',
                  dense=str(dense), data_origin=summary['data_origin'],
                  dense_sha256=digest(dense/'summary.json'),
                  region_sha256=digest(region_file),
                  selection='largest visible valid-depth mask; no fitted catalog pose used',
                  source_sha256={'adapter': digest(__file__), 'geometry': digest(inspect.getfile(stereo_pose_input))},
                  accuracy_1mm_verified=False, stones=[])
    for stone, entry in sorted(best.items()):
        folder = output/f'stone_{stone}'; folder.mkdir()
        with Image.open(entry['source'].with_suffix('.png')) as rgb_pair:
            h, w = entry['depth'].shape
            if rgb_pair.size != (2*w, h):
                raise ValueError('rectified image differs from saved disparity geometry')
            rgb = np.asarray(rgb_pair)[:, :w, :3].copy()
        Image.fromarray(rgb).save(folder/'rgb.png')
        Image.fromarray(entry['mask'].astype(np.uint8)*255).save(folder/'mask.png')
        overlay = rgb.copy()
        overlay[entry['mask']] = (.55*overlay[entry['mask']] + .45*np.array([0,255,80])).astype(np.uint8)
        Image.fromarray(overlay).save(folder/'mask_overlay.png')
        np.savez_compressed(folder/'input.npz', depth=entry['depth'], mask=entry['mask'],
                            matrix=entry['matrix'], base_link_T_camera=entry['pose'])
        report['stones'].append(dict(stone_id=stone, cluster_index=entry['cluster'],
            path=folder.name, mask_pixels=entry['count'], source_pair=str(entry['source']),
            source_pair_sha256=digest(entry['source']), input_sha256=digest(folder/'input.npz'),
            rgb_sha256=digest(folder/'rgb.png')))
    atomic_json(output/'inputs.json', report)
    return report


def run(inputs, output, repository, model_repository):
    inputs, output, repository, model_repository = [Path(p).resolve() for p in (inputs, output, repository, model_repository)]
    manifest = read_json(inputs/'inputs.json')
    dense, reference = Path(manifest['dense']), Path(manifest['reference'])
    if (manifest.get('status') != 'complete' or digest(dense/'summary.json') != manifest['dense_sha256']
            or digest(reference/'rgb_region.json') != manifest['region_sha256']
            or digest(reference/'femto_geometry/geometry.json') != manifest['geometry_sha256']
            or digest(reference/'fused_identification/pose_identification/stone_poses.json') != manifest['prior_assignment_sha256']):
        raise ValueError('FoundationPose inputs have changed')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, kind='foundation_pose_initializer_benchmark', status='running',
                  data_origin=manifest['data_origin'], accuracy_1mm_verified=False, poses_published=False,
                  inputs_sha256=digest(inputs/'inputs.json'), source_sha256=digest(__file__),
                  model_repository_commit=subprocess.check_output(['git','-C',str(model_repository),'rev-parse','HEAD'],text=True).strip(),
                  checkpoint_sha256={str(p.relative_to(model_repository)): digest(p) for p in (model_repository/'weights').glob('*/*.pth')},
                  mesh_appearance='untextured catalog meshes', refinement_iterations=5, stones=[], timings_s={})
    atomic_json(output/'summary.json', report)
    try:
        started = time.monotonic()
        sys.path.insert(0, str(model_repository))
        import torch
        import trimesh
        import nvdiffrast.torch as dr
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
        scorer, refiner = ScorePredictor(), PoseRefinePredictor()
        context = dr.RasterizeCudaContext()
        torch.cuda.synchronize()
        report['timings_s']['load_models_and_renderer'] = time.monotonic()-started
        args = build_parser().parse_args(['process','unused','--repository',str(repository),'--output','unused'])
        profile, catalog = load_inputs(args)
        quality = replace(profile.field_quality, maximum_correction_translation=None, maximum_correction_rotation_degrees=None)
        backend = Open3dRegistrationBackend(profile.field_registration)
        matcher = TabletopFieldMatcher(catalog, backend, quality, profile.assignment)
        geometry = read_json(reference/'femto_geometry/geometry.json')
        planning_T_base = _planning_T_base(geometry)
        region = read_json(reference/'rgb_region.json')
        scene = _read_cloud(dense/'refined/rgb_dense_voxel_1mm.pcd')
        old = {p['stone_id']: np.array(p['planning_T_model']) for p in read_json(reference/'fused_identification/pose_identification/stone_poses.json')['poses']}
        layers = {}
        for item in manifest['stones']:
            stone, cluster = item['stone_id'], item['cluster_index']
            print(f'FoundationPose: stone {stone}', flush=True)
            folder = inside(inputs,item['path'])
            if digest(folder/'input.npz') != item['input_sha256'] or digest(folder/'rgb.png') != item['rgb_sha256']:
                raise ValueError('prepared RGB/depth input changed')
            with np.load(folder/'input.npz',allow_pickle=False) as arrays:
                data = {key: arrays[key].copy() for key in arrays.files}
            rgb = np.array(Image.open(folder/'rgb.png').convert('RGB'))
            mesh_file = repository/f'apps/tabletop/assets/objects/model_{stone}_mesh.ply'
            mesh = trimesh.load(mesh_file,process=False)
            t = time.monotonic()
            estimator = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                mesh=mesh, scorer=scorer, refiner=refiner, glctx=context, debug=0, debug_dir=str(output/f'stone_{stone}/debug'))
            torch.cuda.synchronize(); setup_s=time.monotonic()-t
            t=time.monotonic()
            camera_T_model=estimator.register(K=data['matrix'],rgb=rgb,depth=data['depth'],ob_mask=data['mask'],iteration=5)
            torch.cuda.synchronize(); inference_s=time.monotonic()-t
            initial=planning_T_base @ data['base_link_T_camera'] @ camera_T_model
            lo,hi=region['stone_boxes_xy'][cluster]; plane=np.asarray(region['support_plane'])
            mask=np.all((scene[:,:2]>=lo)&(scene[:,:2]<=hi),axis=1)&(scene[:,2]>scene[:,:2]@plane[:2]+plane[2]+.008)
            rgb_points=_transform(scene[mask],planning_T_base)
            femto_points=_read_cloud(reference/f'femto_geometry/cluster_{cluster:02d}.pcd')
            femto,rgb_points,combined=_balanced_observation(femto_points,rgb_points,profile.field_registration.voxel_sizes[-1],profile.field_registration.minimum_points)
            model=catalog.load(stone).points
            def metrics(pose):
                values={name:backend.evaluate(PointCloudData(model),PointCloudData(p),pose) for name,p in [('rgb',rgb_points),('femto',femto),('combined',combined)]}
                return {name:_metric(v) for name,v in values.items()}, sorted({f'{name}_{reason}' for name,v in values.items() for reason in evaluate_registration(v,initial,quality).reasons})
            initial_metrics,initial_reasons=metrics(initial)
            t=time.monotonic()
            candidate=matcher.register_best(stone,PointCloudData(combined),np.asarray(geometry['table_plane']),initial_transform=initial)
            icp_s=time.monotonic()-t; final=candidate.registration.transform
            final_metrics,reasons=metrics(final)
            t=time.monotonic()
            multistart=matcher.register_best(stone,PointCloudData(combined),np.asarray(geometry['table_plane']))
            multistart_s=time.monotonic()-t
            baseline_metrics,baseline_reasons=metrics(multistart.registration.transform)
            def difference(pose):
                return dict(translation_mm=float(np.linalg.norm(pose[:3,3]-old[stone][:3,3])*1000),
                            rotation_deg=float(np.rad2deg(Rotation.from_matrix(pose[:3,:3]@old[stone][:3,:3].T).magnitude())))
            result=dict(stone_id=stone,cluster_index=cluster,mesh_sha256=digest(mesh_file),
                        hypothesis_count=len(estimator.rot_grid),
                        camera_T_model=np.asarray(camera_T_model).tolist(),initial_planning_T_model=initial.tolist(),
                        refined_planning_T_model=final.tolist(),initial_metrics=initial_metrics,refined_metrics=final_metrics,
                        initial_rejection_reasons=initial_reasons,refined_rejection_reasons=reasons,
                        refined_accepted=not reasons,
                        multistart_planning_T_model=multistart.registration.transform.tolist(),
                        multistart_metrics=baseline_metrics,multistart_rejection_reasons=baseline_reasons,
                        initial_difference_from_previous=difference(initial),
                        refined_difference_from_previous=difference(final),timings_s=dict(object_setup=setup_s,inference=inference_s,icp=icp_s,multistart_icp=multistart_s))
            report['stones'].append(result);atomic_json(output/'summary.json',report)
            parts={'RGB':rgb_points,'FEMTO':femto,'FoundationPose initial':_transform(model,initial),
                   'FoundationPose + ICP':_transform(model,final),'24-start ICP (same points)':_transform(model,multistart.registration.transform),
                   'Previous workflow fit':_transform(model,old[stone])}
            render_clouds(output/f'stone_{stone}/comparison',parts,{},frame='world',title=f'Stone {stone}: learned initializer and joint ICP')
            for name,points in parts.items():layers[f'{name} stone {stone}']=points
        render_clouds(output/'comparison',layers,{},frame='world',title=f'FoundationPose initial poses vs joint ICP — {manifest["data_origin"]} capture')
        render_clouds(output/'initial', {name:p for name,p in layers.items() if name.startswith(('RGB ', 'FEMTO ', 'FoundationPose initial '))},
                      {}, frame='world', title='FoundationPose initial poses before ICP')
        render_clouds(output/'refined', {name:p for name,p in layers.items() if name.startswith(('RGB ', 'FEMTO ', 'FoundationPose + ICP '))},
                      {}, frame='world', title='FoundationPose initialization followed by joint ICP')
        report.update(status='complete',all_refined_accepted=all(s['refined_accepted'] for s in report['stones']))
        report['timings_s']['total_with_viewers']=time.monotonic()-started
        report['limitations']=['Previous fitted poses are comparisons, not ground truth.',
            'Masks use observed 3D cluster regions; catalog meshes have no textures.',
            'One selected stereo view per stone; first inference includes cold compilation.',
            'Existing millimetre-scale sensor discrepancies remain; operational gates are not 1-mm accuracy verification.']
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        atomic_json(output/'summary.json',report)
    return report



def evaluate_truth(inputs, results, simulation, output):
    """Score a completed synthetic experiment in the original UR-base scene frame."""
    import html
    import trimesh
    from stacking_common.pointcloud.pose_metrics import pose_errors
    from .calibration import BASE_LINK_T_UR_BASE

    inputs, results, simulation, output = [Path(p).resolve() for p in (inputs, results, simulation, output)]
    prepared = read_json(inputs/'inputs.json')
    benchmark = read_json(results/'summary.json')
    truth = read_json(simulation/'summary.json')
    if (prepared.get('data_origin') != 'synthetic' or benchmark.get('data_origin') != 'synthetic'
            or truth.get('data_origin') != 'synthetic' or truth.get('scene_frame') != 'world_ur_base'
            or benchmark.get('status') != 'complete' or truth.get('status') != 'complete'
            or benchmark.get('inputs_sha256') != digest(inputs/'inputs.json')):
        raise ValueError('matching complete synthetic benchmark and simulation are required')
    reference = Path(prepared['reference'])
    geometry_file = reference/'femto_geometry/geometry.json'
    if digest(geometry_file) != prepared['geometry_sha256']:
        raise ValueError('planning-frame geometry changed')
    capture = Path(read_json(reference/'pick_scene_workflow.json')['capture_dir']).resolve()
    dense = read_json(Path(prepared['dense'])/'summary.json')
    if dense['session_sha256'] != digest(capture/'session/session.json'):
        raise ValueError('benchmark does not belong to the workflow capture')
    metadata = read_json(capture/'capture.json')
    source_capture = capture
    if metadata.get('kind') == 'offline_scene_capture_subset':
        source_capture = Path(metadata['source_capture']).resolve()
        if digest(source_capture/'capture.json') != metadata['source_capture_sha256']:
            raise ValueError('source simulation capture changed')
    if source_capture != Path(truth['capture_directory']).resolve():
        raise ValueError('ground truth belongs to a different simulation capture')
    # Undo the estimated table-height correction and convert base_link to the
    # original simulator UR-base frame. No fit to the ground truth is performed.
    scene_T_planning = np.linalg.inv(BASE_LINK_T_UR_BASE) @ np.linalg.inv(_planning_T_base(read_json(geometry_file)))
    truth_by_id = {item['stone_id']: item for item in truth['stones']}
    output.mkdir(parents=True, exist_ok=False)
    report = dict(kind='foundation_pose_simulation_ground_truth', status='complete',
                  output_frame='world_ur_base', ground_truth_used_for_inference=False,
                  accuracy_1mm_verified=False, scene_T_planning=scene_T_planning.tolist(),
                  simulation_sha256=digest(simulation/'summary.json'), benchmark_sha256=digest(results/'summary.json'),
                  geometry_sha256=digest(geometry_file), source_sha256=digest(__file__),
                  metrics_source_sha256=digest(inspect.getfile(pose_errors)), stones=[],
                  limitations=['Synthetic scene only; no conclusion about real-camera absolute accuracy.',
                    'Rotation error uses the exact simulator orientation, without assumed symmetries.',
                    'ADD-S may hide orientation ambiguity; inspect rotation and corresponding-point ADD together.',
                    'Mesh vertices are the common evaluation samples; ICP fits the measured catalog PCD, which differs from the rendered mesh.'])
    method_keys = {'FoundationPose initial':'initial_planning_T_model',
                   'FoundationPose + ICP':'refined_planning_T_model',
                   '24-start ICP':'multistart_planning_T_model'}
    layers = {name:{} for name in method_keys}
    table_rows = []
    for stone in benchmark['stones']:
        ident = stone['stone_id']; item = truth_by_id[ident]
        if digest(item['mesh']) != item['mesh_sha256'] or stone['mesh_sha256'] != item['mesh_sha256']:
            raise ValueError('benchmark mesh differs from the mesh used by the simulator')
        mesh = trimesh.load(item['mesh'],process=False)
        gt = np.eye(4); gt[:3,3] = item['world_pose'][:3]
        gt[:3,:3] = Rotation.from_quat(item['world_pose'][3:]).as_matrix()
        row = dict(stone_id=ident, scene_T_model=gt.tolist(), metrics={})
        per_stone = {'Ground truth':_transform(mesh.vertices,gt)}
        for name, key in method_keys.items():
            pose = scene_T_planning @ np.asarray(stone[key])
            metrics = pose_errors(mesh.vertices,pose,gt)
            row['metrics'][name] = dict(**metrics,scene_T_model=pose.tolist())
            layers[name][f'Ground truth {ident}'] = per_stone['Ground truth']
            layers[name][f'{name} {ident}'] = _transform(mesh.vertices,pose)
            per_stone[name] = _transform(mesh.vertices,pose)
            table_rows.append(f'<tr><td>{ident}</td><td>{html.escape(name)}</td>'
                f'<td>{metrics["translation_mm"]:.2f}</td><td>{metrics["rotation_deg"]:.2f}</td>'
                f'<td>{metrics["add_mean_mm"]:.2f}</td><td>{metrics["adds_mean_mm"]:.2f}</td></tr>')
        report['stones'].append(row)
        render_clouds(output/f'stone_{ident}',per_stone,{},frame='world_ur_base',
                      title=f'Stone {ident}: estimated poses vs simulation ground truth')
    for (name, points), folder in zip(layers.items(),('initial','refined','icp')):
        render_clouds(output/folder,points,{},frame='world_ur_base',title=name+' vs ground truth')
    atomic_json(output/'summary.json',report)
    (output/'overview.html').write_text('<!doctype html><meta charset="utf-8"><title>Ground-truth pose comparison</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}td,th{padding:10px;border-bottom:1px solid #ccd;text-align:left}</style>'
        '<h1>FoundationPose vs simulation ground truth</h1><p>Seven-view synthetic scene, stones 9/10/11. Ground truth is used only for evaluation.</p>'
        '<p><a href="initial/overview.html">Initial poses</a> · <a href="refined/overview.html">After ICP</a> · <a href="icp/overview.html">24-start ICP</a></p>'
        '<table><tr><th>Stone</th><th>Method</th><th>Translation (mm)</th><th>Rotation (°)</th><th>ADD (mm)</th><th>ADD-S (mm)</th></tr>'
        + ''.join(table_rows) + '</table><p>ADD: mean displacement of corresponding mesh vertices. ADD-S: mean nearest-vertex distance, which can hide incorrect orientation.</p>'
        '<p>Per-stone overlays: <a href="stone_9/overview.html">9</a> · <a href="stone_10/overview.html">10</a> · <a href="stone_11/overview.html">11</a>.</p>'
        '<p>Pose transforms are converted back to the original simulation frame, undoing only the recorded planning-frame correction. No alignment to ground truth is fitted. The meshes are untextured and differ slightly from the catalog PCDs used by ICP. This experiment does not verify real-world accuracy.</p>')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare')
    for key in ('dense','reference','output'):prep.add_argument('--'+key,type=Path,required=True)
    infer=commands.add_parser('run')
    for key in ('inputs','output'):infer.add_argument('--'+key,type=Path,required=True)
    infer.add_argument('--repository',type=Path,default=Path.cwd())
    infer.add_argument('--model-repository',type=Path,default=Path('.data/vendor/FoundationPose'))
    evaluate=commands.add_parser('evaluate-truth')
    for key in ('inputs','results','simulation','output'):evaluate.add_argument('--'+key,type=Path,required=True)
    args=vars(parser.parse_args());command=args.pop('command')
    result={'prepare':prepare,'run':run,'evaluate-truth':evaluate_truth}[command](**args)
    print(result['status'],flush=True)


if __name__=='__main__':main()
