"""Offline route visibility experiment with Operations assets and native posegen."""

import argparse
from dataclasses import replace
import html
import json
import shutil
from pathlib import Path
import time

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation

from stacking_common.actions import CorePlacementPoseSolver, InitializedPlacement
from stacking_common.domain import DirectionalTaper, ObjectInventory, State, TargetGeometry
from stacking_common.multiview.synthetic import TriangleScene, sample_surface, supported_samples, surface_texture
from stacking_tabletop.composition import TabletopApplication
from stacking_tabletop.execution.kinematics import UR5eKinematics
from stacking_tabletop.gui.operations.environment import planning_scene_boxes
from stacking_tabletop.planning import TabletopRobotTrajectoryProjector
from .calibration import BASE_LINK_T_UR_BASE
from .hand_eye import HandEyeCalibration
from .hand_eye_capture import atomic_json, digest
from .scene_views import validate_views
from .storage import write_xyz_pcd
from .hand_eye_synthetic import geometry, look_at
from .simulation_capture import SimulatedCapture

ROOT = Path(__file__).resolve().parents[5]


def matrix(pose):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(np.asarray(pose)[3:]).as_matrix()
    result[:3, 3] = np.asarray(pose)[:3]
    return result


def transformed(vertices, pose):
    return np.asarray(vertices) @ pose[:3, :3].T + pose[:3, 3]


def synthesize_stones(app, ids, target):
    """Solve separated pick placements above the configured table in local SI units."""
    table = next(box for box in app.planning_profile.obstacles if box.obstacle_id == 'pick_table')
    low, high = np.asarray(table.minimum), np.asarray(table.maximum)
    # Keep table-local XY but world Z: ground_height is explicitly the tabletop.
    origin = np.r_[(low[:2]+high[:2])/2, 0.]
    ground = float(high[2])
    initial = {key: np.array([0., 0., .08, 0., 0., 0., 1.]) for key in ids}
    loaded = app.load_inventory(ids, initial)
    inventory = ObjectInventory(loaded.objects, initial,
        TargetGeometry(high[0]-low[0], high[1]-low[1], .25, np.array([0., 0., ground]), DirectionalTaper()))
    # The capture fixture needs tighter contact-force precision for the pose optimizer.
    profile = replace(app.posegen_profile,
        objective={**app.posegen_profile.objective, "ground_height": ground},
        force_solver={**app.posegen_profile.force_solver, "tol_abs": 1e-7, "tol_rel": 1e-7})
    solver = CorePlacementPoseSolver(inventory, profile)
    state = State(list(ids), [], {}, [], [])
    meshes, records = [], []
    for index, key in enumerate(ids):
        pose = initial[key].copy()
        pose[:2] = target[:2] - origin[:2] + [(index-(len(ids)-1)/2)*.15, 0.]
        pose[2] = ground + .01 - inventory[index].mesh_arrays()[0][:, 2].min()
        solved = solver.solve(inventory, state, index, InitializedPlacement(pose, ground))
        pose = solved.pose.copy()
        pose[:3] += origin
        source = app.repository_root / app.inventory_profile.model_directory / f'model_{key}_mesh.ply'
        mesh = o3d.io.read_triangle_mesh(str(source))
        solved_world_pose = pose.copy()
        vertices = transformed(mesh.vertices, matrix(pose))
        # Posegen uses DSF geometry; seat the rendered triangle mesh explicitly.
        seating = float(high[2] - vertices[:, 2].min())
        pose[2] += seating
        vertices[:, 2] += seating
        faces = np.asarray(mesh.triangles)
        if not len(faces):
            raise ValueError(f'empty stone mesh: {source}')
        if np.any(vertices[:, :2].min(axis=0) < low[:2]) or np.any(vertices[:, :2].max(axis=0) > high[:2]):
            raise ValueError(f'synthesized stone {key} is outside the pick table; use a seed aimed at its center')
        for previous, _ in meshes:
            if np.all(np.minimum(vertices.max(axis=0), previous.max(axis=0)) >
                      np.maximum(vertices.min(axis=0), previous.min(axis=0))):
                raise ValueError('synthetic stone bounds overlap; reduce the stone count')
        meshes.append((vertices, faces))
        records.append(dict(stone_id=key, world_pose=pose.tolist(),
            posegen_world_pose=solved_world_pose.tolist(), mesh_seating_z_m=seating, mesh=str(source),
            mesh_sha256=digest(source), pose_solver_converged=solved.converged,
            c_gap_m=solved.c_gap, c_feq=solved.c_feq,
            posegen_ground_height_m=ground, force_tol_abs=1e-7, force_tol_rel=1e-7,
            solver_diagnostics={key: value for key, value in solved.diagnostics.items()
                                if not key.startswith("posegen_contact_")},
            mesh_min_z_m=float(vertices[:, 2].min()), table_height_m=float(high[2])))
    return meshes, records


class RobotMeshes:
    """Operations visual assets positioned using the saved capture FK."""
    def __init__(self, app, saved_kinematics):
        self.projector = TabletopRobotTrajectoryProjector(app.planning_profile, app.repository_root)
        self.nominal = UR5eKinematics(app.repository_root / app.planning_profile.robot.urdf)
        self.saved = saved_kinematics
        self.cache = {}

    def at(self, joints):
        result = []
        for part in self.projector.project(np.asarray(joints), None):
            key = (part.mesh_path, part.mesh_scale)
            if key not in self.cache:
                mesh = o3d.io.read_triangle_mesh(str(part.mesh_path))
                if len(mesh.triangles) > 1500:
                    mesh = mesh.simplify_quadric_decimation(1500)
                self.cache[key] = (np.asarray(mesh.vertices)*part.mesh_scale, np.asarray(mesh.triangles))
            vertices, faces = self.cache[key]
            link_name = part.part_id.split('/')[1] if part.part_id.startswith('robot/') else 'tool0'
            old = self.nominal.model.kinematics.find_link(link_name)
            new = self.saved.model.kinematics.find_link(link_name)
            if old is None or new is None:
                raise ValueError(f'saved robot description is missing {link_name}')
            correction = self.saved._pose(joints, new) @ np.linalg.inv(self.nominal._pose(joints, old))
            result.append((part.part_id, transformed(vertices, correction @ matrix(part.pose)), faces, part.color))
        return result


def render_rgb(hit, stone_count):
    """Fixed world texture and diffuse shading; not a photorealistic sensor model."""
    valid = np.isfinite(hit['depth'])
    rgb = np.full((*valid.shape, 3), 28., dtype=float)
    points = hit['points'][valid]
    texture = surface_texture(points)
    normal = hit['primitive_normals'][valid]
    light = np.array([.3, -.4, .85]); light /= np.linalg.norm(light)
    shade = .5 + .5*np.abs(normal @ light)
    colors = np.full((len(points), 3), [150., 145., 135.])
    ids = hit['geometry_ids'][valid]
    colors[ids >= stone_count] = [155., 115., 76.]
    rgb[valid] = colors * (texture*shade)[:, None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def render_scene(output, static, robots, plan, cameras):
    import plotly.graph_objects as go
    figure = go.Figure()
    def mesh_trace(vertices, triangles, name, color):
        return go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2], name=name,
            color=color, opacity=1., showlegend=False)
    for name, vertices, faces, color in static:
        figure.add_trace(mesh_trace(vertices, faces, name, color))
    centers = np.asarray(cameras)[:, :3, 3]
    figure.add_trace(go.Scatter3d(x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
        mode='markers+text', text=[v['label'] for v in plan['views']], name='Camera positions',
        marker=dict(size=4, color='#ff5533')))
    fixed = len(figure.data)
    groups = []
    for index, parts in enumerate(robots):
        group = []
        for name, vertices, faces, color in parts:
            group.append(len(figure.data))
            trace = mesh_trace(vertices, faces, name, color)
            trace.visible = index == 0
            figure.add_trace(trace)
        target = BASE_LINK_T_UR_BASE @ np.r_[plan['target'], 1.]
        figure.add_trace(go.Scatter3d(x=[centers[index, 0], target[0]],
            y=[centers[index, 1], target[1]], z=[centers[index, 2], target[2]],
            mode='lines', line=dict(color='#ff5533'), visible=index == 0, showlegend=False))
        group.append(len(figure.data)-1)
        groups.append(group)
    steps = []
    for index, group in enumerate(groups):
        visible = [i < fixed or i in group for i in range(len(figure.data))]
        steps.append(dict(method='update', label=plan['views'][index]['label'], args=[dict(visible=visible)]))
    figure.update_layout(title='Offline capture simulation — Operations cell and saved wrist route',
        scene=dict(aspectmode='data', xaxis_title='UR base X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)'),
        sliders=[dict(active=0, steps=steps)], height=700)
    figure.write_html(output/'scene.html', include_plotlyjs=True)


def render_report(output, static, robots, plan, cameras, report):
    render_scene(output, static, robots, plan, cameras)
    counts = report['view_counts']
    headings = ''.join(f'<th>Seen: {n} views</th>' for n in counts)
    headings += ''.join(f'<th>Side surface with 2+ views: {n}</th>' for n in counts)
    rows = ''.join('<tr><td>' + str(v['stone_id']) + '</td>' + ''.join(
        f'<td>{v[f"coverage{n}"][metric]:.1f}%</td>'
        for metric in ('seen_percent', 'side_supported_percent') for n in counts) + '</tr>'
        for v in report['coverage'])
    shots = ''.join(f'<figure><img width="480" src="views/{v["label"]}/rgb.png">'
        f'<figcaption>{html.escape(v["label"])}</figcaption></figure>' for v in plan['views'])
    notes = ''.join(f'<li>{html.escape(note)}</li>' for note in report['limitations'])
    convergence = ', '.join(f'stone {item["stone_id"]}: {item["pose_solver_converged"]}' for item in report['stones'])
    (output/'overview.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Capture simulation</title>
<style>body{{font:16px sans-serif;margin:2em;background:#eee}}td,th{{padding:10px;border:1px solid #aaa}}
.gallery{{display:flex;flex-wrap:wrap}}figure{{margin:8px}}iframe{{border:0;width:100%;height:740px}}</style>
<h1>Synthetic pick-scene capture</h1><p>Stones 9–11 placed by native posegen; ideal raycast images and PCD.
This evaluates geometric visibility, not RGB MVS accuracy.</p>
<p>Native pose solver converged: {convergence}. Meshes are vertically seated on the table;
this does not validate physical stability.</p><ul>{notes}</ul>
<table><tr><th>Stone</th>{headings}</tr>{rows}</table>
<iframe src="scene.html"></iframe><div class="gallery">{shots}</div>
<p>Operations GUI: choose <b>Identify existing wrist RGB + FEMTO capture</b>
and select the <code>captures</code> directory beside this page.</p>
<p><a href="summary.json">Detailed metrics and pose solver status</a></p>''')


def simulate(plan_directory, output, *, repository=ROOT):
    started = time.monotonic()
    plan_directory, output = Path(plan_directory).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'simulation output already exists: {output}')
    plan = json.loads((plan_directory/'plan.json').read_text())
    if plan.get('viewpoint_pattern') not in ('coverage13', 'coverage17'):
        raise ValueError('use a coverage13 or coverage17 plan for a nested viewpoint comparison')
    counts = [n for n in (9, 13, 17) if n <= len(plan['views'])]
    calibration = HandEyeCalibration.load(plan_directory/'calibration/profile.yml')
    kin = UR5eKinematics(plan_directory/'robot.urdf')
    if digest(plan_directory/'robot.urdf') != plan['robot_urdf_sha256']:
        raise ValueError('saved kinematics hash changed')
    if digest(plan_directory/'calibration/profile.yml') != plan['hand_eye_profile_sha256']:
        raise ValueError('saved calibration hash changed')
    validate_views(plan, kin, calibration)
    app = TabletopApplication.from_repository(repository)
    target = (BASE_LINK_T_UR_BASE @ np.r_[plan['target'], 1.])[:3]
    stones, records = synthesize_stones(app, (9, 10, 11), target)
    output.mkdir(parents=True)
    snapshot = output/'source_plan'; snapshot.mkdir()
    for name in ('plan.json', 'robot.urdf'):
        shutil.copy2(plan_directory/name, snapshot/name)
    shutil.copytree(plan_directory/'calibration', snapshot/'calibration')
    robot = RobotMeshes(app, kin)
    static = [(f'stone {record["stone_id"]}', v, f, '#999999') for (v, f), record in zip(stones, records)]
    boxes = planning_scene_boxes(app.planning_profile)
    for box in boxes:
        low, high = np.array(box.minimum), np.array(box.maximum)
        mesh = o3d.geometry.TriangleMesh.create_box(*(high-low)).translate(low)
        static.append((box.component_id, np.asarray(mesh.vertices), np.asarray(mesh.triangles), '#ab895e'))
    # Model the fixed camera at the top of its Operations obstruction box.
    camera_box = next(box for box in boxes if box.component_id == 'third_view_camera')
    eye = np.r_[(np.asarray(camera_box.minimum[:2])+camera_box.maximum[:2])/2, camera_box.maximum[2]]
    scene_center = np.mean([np.mean(vertices, axis=0) for vertices, _ in stones], axis=0)
    femto_world = look_at(eye, scene_center)
    femto_info = geometry(1024, 768)
    femto_meshes = [(v, f) for name, v, f, _ in static if name != 'third_view_camera']
    femto_meshes += [(v, f) for _, v, f, _ in robot.at(plan['seed_joints'])]
    femto_hit = TriangleScene(femto_meshes).image(np.asarray(femto_info['K']), femto_world, (1024, 768))
    femto_depth = np.where(np.isfinite(femto_hit['depth']), femto_hit['depth'], 0.)
    table = next(box for box in boxes if box.component_id == 'pick_table')
    corners = np.array([[x, y, z] for x in (table.minimum[0], table.maximum[0])
                        for y in (table.minimum[1], table.maximum[1])
                        for z in (table.maximum[2]-.02, table.maximum[2]+.20)])
    corners = transformed(corners, BASE_LINK_T_UR_BASE)
    exporter = SimulatedCapture(output/'captures', calibration, plan, femto_depth,
        BASE_LINK_T_UR_BASE @ femto_world, femto_info, (corners.min(axis=0), corners.max(axis=0)))
    sampled = [sample_surface(*mesh, seed=record['stone_id']) for mesh, record in zip(stones, records)]
    points = np.concatenate([v[0] for v in sampled])
    poses, visible, robots, frame_reports = [], [], [], []
    size = (calibration.width, calibration.height)
    matrix_k = calibration.color_matrix.copy()
    for view in plan['views']:
        print('simulate_' + view['label'], flush=True)
        camera = BASE_LINK_T_UR_BASE @ np.asarray(view['base_link_T_color'])
        parts = robot.at(view['path'][-1])
        scene = TriangleScene([(v, f) for _, v, f, _ in static] + [(v, f) for _, v, f, _ in parts])
        hit = scene.image(matrix_k, camera, size)
        directory = output/'views'/view['label']; directory.mkdir(parents=True)
        rgb = render_rgb(hit, len(stones))
        Image.fromarray(rgb).save(directory/'rgb.png')
        valid = np.isfinite(hit['depth'])
        exporter.add(view, rgb, np.where(valid, hit['depth'], 0.))
        base_points = transformed(hit['points'][valid], BASE_LINK_T_UR_BASE)
        write_xyz_pcd(directory/'ideal_cloud.pcd', base_points)
        np.savez_compressed(directory/'ideal_depth.npz', depth_m=np.where(valid, hit['depth'], 0),
            geometry_ids=hit['geometry_ids'], K=matrix_k, base_link_T_color=view['base_link_T_color'])
        visibility = scene.visible(points, matrix_k, camera, size)
        visible.append(visibility); poses.append(camera); robots.append(parts)
        frame_reports.append(dict(label=view['label'], stone_pixels=int(np.count_nonzero(valid & (hit['geometry_ids'] < 3))),
                                  world_T_color=camera.tolist()))
    capture_directory = exporter.finish()
    visible = np.asarray(visible)
    centers = np.asarray(poses)[:, :3, 3]
    coverage = []
    offset = 0
    for (samples, normals), record in zip(sampled, records):
        seen = visible[:, offset:offset+len(samples)]; offset += len(samples)
        item = dict(stone_id=record['stone_id'], sample_count=len(samples))
        side = np.abs(normals[:, 2]) < .5
        for count in counts:
            supported = supported_samples(seen[:count], samples, centers[:count])
            item[f'coverage{count}'] = dict(seen_percent=float(seen[:count].any(axis=0).mean()*100),
                supported_percent=float(supported.mean()*100),
                side_supported_percent=float(supported[side].mean()*100) if side.any() else 0.)
        coverage.append(item)
    report = dict(schema_version=1, kind='synthetic_scene_capture_visibility', status='complete',
        data_origin='synthetic', motion_sent=False, collision_checked=False, accuracy_1mm_verified=False,
        plan_sha256=digest(plan_directory/'plan.json'), source_plan=str(plan_directory),
        planning_profile_sha256=digest(Path(repository)/'apps/tabletop/config/planning.yml'),
        posegen_profile_sha256=digest(Path(repository)/'apps/tabletop/config/solvers/posegen.yml'),
        environment=[dict(id=box.component_id, minimum=list(box.minimum), maximum=list(box.maximum)) for box in boxes],
        capture_directory=str(capture_directory),
        femto_pose_source='top of configured Operations camera box, aimed at synthetic stones',
        image_model='ideal_undistorted_pinhole_full_resolution', cloud_frame='base_link',
        scene_frame='world_ur_base', stones=records, coverage=coverage, view_counts=counts, views=frame_reports,
        limitations=[
            'Ideal raycast depth is ground-truth geometry for this synthetic mesh scene, not an RGB reconstruction.',
            'RGB uses procedural world-fixed texture; it does not model real lighting, blur, exposure or sensor noise.',
            'Visibility includes tables, the modeled FEMTO obstruction, other stones and simplified Operations robot/gripper meshes.',
            'Camera housing, cables and unmodeled cell objects are absent. This is not collision validation or permission for robot motion.',
            'Two-view surface support is a geometric opportunity for MVS; texture matching and reconstruction accuracy remain untested.',
            'Surface percentages use 6000 area samples per stone, including its underside; invisible support-contact surfaces lower the total.',
            'Native posegen convergence is recorded per stone. Nonconverged results are synthetic candidates, not verified stable placements.',
            'Rendered meshes are vertically seated on the table after the DSF pose solve; both poses and the correction are retained.'])
    render_report(output, static, robots, plan, poses, report)
    report['elapsed_s'] = time.monotonic()-started
    atomic_json(output/'summary.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True, help='saved coverage13 or coverage17 plan; no hardware connection')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repository', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = simulate(args.plan, args.output, repository=args.repository)
    print(json.dumps(dict(status=report['status'], overview=str(args.output/'overview.html'),
                          coverage=report['coverage'], elapsed_s=report['elapsed_s']), indent=2))


if __name__ == '__main__':
    main()
