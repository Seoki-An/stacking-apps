"""Render a saved synthesized stack through the post-placement wrist scan route."""
import argparse
import html
from pathlib import Path
import time
from types import SimpleNamespace as NS

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree

from stacking_common.artifacts import StackRunStore
from stacking_common.multiview.synthetic import TriangleScene, sample_surface, supported_samples
from stacking_common.planning import compose_poses
from stacking_common.rendering.pointcloud import render_clouds
from stacking_tabletop.composition import TabletopApplication
from stacking_tabletop.execution.kinematics import UR5eKinematics
from stacking_tabletop.gui.operations.environment import planning_scene_boxes
from .calibration import BASE_LINK_T_UR_BASE
from .hand_eye import HandEyeCalibration
from .hand_eye_capture import atomic_json, digest
from .hand_eye_fast_stereo import reconstruct_with_runtime
from .scene_capture_simulation import RobotMeshes, matrix, transformed, render_rgb, render_scene
from .scene_views import render_preview
from .simulation_capture import SimulatedCapture
from .stack_scene_capture import stack_region, build_stack_route
from .storage import write_xyz_pcd

ROOT = Path(__file__).resolve().parents[5]


def load_stack(run, app, step=None):
    """Snapshot poses are synthetic scene inputs, never observations or live state."""
    artifact = StackRunStore.load_mcts(run)
    choices = [s for s in artifact.steps if s.scene.objects and (step is None or s.step == step)]
    if not choices:
        raise ValueError('select a search snapshot with placed objects')
    snapshot = choices[-1]
    if snapshot.scene.frame_id != 'stack':
        raise ValueError('simulation currently requires a stack-frame search snapshot')
    records = []
    for obj in snapshot.scene.objects:
        mesh = app.repository_root/app.inventory_profile.model_directory/f'model_{obj.object_id}_mesh.ply'
        pose = compose_poses(app.planning_profile.world_T_stack.parent_T_child, obj.pose)
        records.append(dict(stone_id=obj.object_id, world_pose=pose.tolist(),
                            mesh=str(mesh), mesh_sha256=digest(mesh)))
    return snapshot.step, records


def evaluate(output, meshes, samples, visibility, cameras, records, region):
    """Compare reconstructed points with known triangles; no ICP alignment."""
    cloud = np.asarray(o3d.io.read_point_cloud(str(output/'rgb_dense/refined/rgb_dense_voxel_1mm.pcd')).points)
    ground = region['support_plane'][2]
    rgb = cloud[cloud[:, 2] > ground+.008]
    if not len(rgb):
        raise ValueError('no reconstructed points above the tabletop margin')
    truth = TriangleScene([(transformed(v, BASE_LINK_T_UR_BASE), f) for v, f in meshes])
    distances = truth.scene.compute_distance(o3d.core.Tensor(rgb.astype(np.float32))).numpy()*1000
    supported = supported_samples(visibility, np.concatenate([p for p, _ in samples]),
                                  np.asarray(cameras)[:, :3, 3])
    tree = cKDTree(rgb)
    rows, layers, start = [], {'RGB reconstruction': rgb}, 0
    for record, (points, normals) in zip(records, samples):
        count = len(points)
        seen = visibility[:, start:start+count]
        support = supported[start:start+count]; start += count
        base = transformed(points, BASE_LINK_T_UR_BASE)
        retained = support & (base[:, 2] > ground+.008)
        nearest = tree.query(base[retained])[0]*1000
        side = np.abs(normals[:, 2]) < .5
        rows.append(dict(stone_id=record['stone_id'], seen_percent=float(seen.any(axis=0).mean()*100),
                         stereo_supported_percent=float(support.mean()*100),
                         side_supported_percent=float(support[side].mean()*100) if side.any() else None,
                         supported_samples_evaluated=int(retained.sum()),
                         supported_surface_within_3mm_percent=float((nearest<=3).mean()*100) if len(nearest) else None))
        layers[f'Ground truth stone {record["stone_id"]}'] = base
    render_clouds(output/'comparison', layers, {}, frame='base_link',
                  title='Synthesized stack: RGB reconstruction and ground-truth surfaces (no alignment)')
    write_xyz_pcd(output/'rgb_stack.pcd', rgb)
    write_xyz_pcd(output/'ground_truth.pcd', np.concatenate(list(layers.values())[1:]))
    result = dict(rgb_stack_points=len(rgb), distance_to_triangle_mesh_mm=dict(
        median=float(np.median(distances)), p95=float(np.percentile(distances,95)),
        within_3mm_percent=float((distances<=3).mean()*100)), stones=rows,
        registration_applied=False, truth_used_for_depth_inference=False)
    atomic_json(output/'metrics.json', result)
    return result


def report_html(output, report):
    metrics = report['metrics']
    rows = ''
    for row in metrics['stones']:
        coverage = row['supported_surface_within_3mm_percent']
        coverage = '—' if coverage is None else f'{coverage:.1f}%'
        rows += (f'<tr><td>{row["stone_id"]}</td><td>{row["seen_percent"]:.1f}%</td>'
                 f'<td>{row["stereo_supported_percent"]:.1f}%</td><td>{coverage}</td></tr>')
    gallery = ''.join(f'<figure><img src="views/{html.escape(v["label"])}/rgb.png">'
                      f'<figcaption>{html.escape(v["label"])}</figcaption></figure>' for v in report['views'])
    errors = metrics['distance_to_triangle_mesh_mm']
    (output/'overview.html').write_text('''<!doctype html><meta charset="utf-8"><title>Stacked-scene capture simulation</title>
<style>body{font:16px system-ui;max-width:1300px;margin:30px auto;background:#f5f6f8;padding:20px}
a{color:#155db0}iframe{width:100%;height:680px;border:0}td,th{padding:10px;text-align:left;border-bottom:1px solid #bbb}
.gallery{display:flex;flex-wrap:wrap}figure{margin:8px}img{width:390px}</style>
<h1>Synthesized stacked-scene wrist capture</h1>'''+
        f'<p>{len(report["stones"])} stones in a saved synthetic stack; {len(report["views"])} reachable wrist views. '
        f'Central view tilt: {report["central_tilt_deg"]:g}°. No hardware commands were sent.</p>'
        '<p><a href="rgb_dense/rgb_only/overview.html">RGB reconstructed PCD</a> · '
        '<a href="comparison/overview.html">RGB vs ground truth</a> · '
        '<a href="preview.html">Capture route preview</a> · <a href="metrics.json">Metrics</a></p>'
        f'<p>{metrics["rgb_stack_points"]:,} reconstructed stack points. Distance to ground-truth triangle surfaces: '
        f'median {errors["median"]:.2f} mm, 95th percentile {errors["p95"]:.2f} mm. No ICP alignment.</p>'
        '<iframe src="scene.html"></iframe><h2>Surface visibility and reconstructed coverage</h2>'
        '<table><tr><th>Stone</th><th>Seen in any view</th><th>Two-view support</th><th>Supported surface within 3 mm (%)</th></tr>'+
        rows+'</table><p>Visibility includes undersides and occluded surfaces in its denominator. '
        'Reconstruction coverage evaluates supported surface samples above the table margin.</p>'
        '<h2>Rendered RGB views</h2><div class="gallery">'+gallery+'</div><h2>Scope</h2><ul>'+
        ''.join(f'<li>{html.escape(note)}</li>' for note in report['limitations'])+'</ul>')


def simulate(run, output, *, repository=ROOT, step=None):
    started=time.monotonic()
    run,output,repository=[Path(p).resolve() for p in (run,output,repository)]
    if output.exists():
        raise FileExistsError('choose a new simulation output directory')
    app=TabletopApplication.from_repository(repository)
    snapshot_step,records=load_stack(run,app,step)
    calibration=HandEyeCalibration.load(repository/'apps/tabletop/config/hand_eye.yml')
    kin=UR5eKinematics(repository/app.planning_profile.robot.urdf)
    objects=[NS(object_id=r['stone_id'],pose=np.array(r['world_pose'])) for r in records]
    saved=NS(planning=app.planning_profile,assets={f'object/{o.object_id}':repository/
        app.inventory_profile.model_directory/f'model_{o.object_id}.obj' for o in objects})
    plan_input=NS(problem=NS(place_scene_objects=objects[:-1],action=NS(object_id=objects[-1].object_id)))
    region=stack_region(saved,plan_input,objects[-1].pose)
    plan=build_stack_route(kin,calibration,app.planning_profile.robot.home_positions,region)
    output.mkdir(parents=True)
    calibration.snapshot(output/'calibration')
    (output/'robot.urdf').write_bytes(kin.urdf_path.read_bytes())
    atomic_json(output/'plan.json',plan);atomic_json(output/'region.json',region)
    render_preview(plan,kin,calibration,output/'preview.html')
    report=dict(kind='stacked_scene_capture_simulation',status='running',data_origin='synthetic',
        motion_sent=False,collision_checked=False,accuracy_1mm_verified=False,scene_frame='world_ur_base',
        source_run=str(run),source_run_sha256=digest(run),source_snapshot_step=snapshot_step,
        stones=records,central_tilt_deg=plan['central_tilt_deg'],omitted_views=plan['omitted_views'],views=[],
        limitations=['Procedural surface texture, ideal pinhole images and no camera noise; this does not predict real-camera accuracy.',
          'Scene poses come from an existing synthesized stacking-search snapshot; physical stability is not revalidated.',
          'The raycaster includes stone occlusion, tables and simplified Operations robot/gripper geometry.',
          'Camera housing, cables and unmodeled objects are absent. This is not collision validation.',
          'Ideal depth is saved for reference only. Fast-FoundationStereo receives RGB, intrinsics, camera transforms and the capture region.',
          'Camera poses are exact in the simulation. Real hand-eye and robot calibration errors are absent.',
          'Existing stack capture region, 60-mm padding and reachable-view selection are reused without ground-truth pose alignment.'])
    atomic_json(output/'summary.json',report)
    try:
        meshes=[];static=[]
        for record in records:
            mesh=o3d.io.read_triangle_mesh(record['mesh'])
            vertices=transformed(mesh.vertices,matrix(record['world_pose']));faces=np.asarray(mesh.triangles)
            meshes.append((vertices,faces));static.append((f'stone {record["stone_id"]}',vertices,faces,'#999999'))
        for box in planning_scene_boxes(app.planning_profile):
            low,high=np.array(box.minimum),np.array(box.maximum)
            mesh=o3d.geometry.TriangleMesh.create_box(*(high-low)).translate(low)
            static.append((box.component_id,np.asarray(mesh.vertices),np.asarray(mesh.triangles),'#ab895e'))
        robot=RobotMeshes(app,kin)
        samples=[sample_surface(*mesh,seed=r['stone_id']) for mesh,r in zip(meshes,records)]
        truth_points=np.concatenate([p for p,_ in samples]);visibility=[];cameras=[];robots=[]
        exporter=SimulatedCapture(output/'captures',calibration,plan)
        for view in plan['views']:
            print('render_stack_'+view['label'],flush=True)
            camera=BASE_LINK_T_UR_BASE @ np.asarray(view['base_link_T_color'])
            parts=robot.at(view['path'][-1]);robots.append(parts);cameras.append(camera)
            scene=TriangleScene([(v,f) for _,v,f,_ in static]+[(v,f) for _,v,f,_ in parts])
            hit=scene.image(calibration.color_matrix,camera,(calibration.width,calibration.height))
            valid=np.isfinite(hit['depth']);rgb=render_rgb(hit,len(records))
            directory=output/'views'/view['label'];directory.mkdir(parents=True)
            Image.fromarray(rgb).save(directory/'rgb.png')
            depth=np.where(valid,hit['depth'],0.)
            exporter.add(view,rgb,depth)
            np.savez_compressed(directory/'ideal_depth.npz',depth_m=depth,geometry_ids=hit['geometry_ids'])
            write_xyz_pcd(directory/'ideal_cloud.pcd',transformed(hit['points'][valid],BASE_LINK_T_UR_BASE))
            visibility.append(scene.visible(truth_points,calibration.color_matrix,camera,(calibration.width,calibration.height)))
            report['views'].append(dict(label=view['label'],stone_pixels=int(np.sum(valid & (hit['geometry_ids']<len(records))))))
        exporter.finish()
        report['render_capture_s']=time.monotonic()-started
        render_scene(output,static,robots,plan,cameras)
        vendor=repository/'.data/vendor/Fast-FoundationStereo'
        dense=reconstruct_with_runtime(output/'captures/session',None,None,output/'rgb_dense',repository=vendor,
            checkpoint=vendor/'weights/23-36-37/model_best_bp2_serialize.pth',region_file=output/'region.json')
        report['reconstruction_timings_s']=dense['timings_s']
        report['metrics']=evaluate(output,meshes,samples,np.asarray(visibility),cameras,records,region)
        report.update(status='complete',elapsed_s=time.monotonic()-started)
        report_html(output,report)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        atomic_json(output/'summary.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--step',type=int,help='search snapshot step; default is the last nonempty scene')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repository',type=Path,default=ROOT)
    args=parser.parse_args();simulate(**vars(args))
    print(args.output/'overview.html',flush=True)


if __name__=='__main__':main()
