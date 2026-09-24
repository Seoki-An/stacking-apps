"""Dense RGB MVS with optional FEMTO-defined spatial crops and comparison regions."""
from pathlib import Path
import inspect
import time

import numpy as np
from scipy.spatial import cKDTree

from stacking_common.multiview.dense import (
    write_known_cameras, depth_bounds_from_roi, read_colmap_depth,
    backproject_colmap_depth, projected_roi_crop, write_pinhole_cameras,
    select_source_views,
)
from stacking_common.pointcloud.processing import fixed_grid_centroids
from stacking_common.pointcloud.types import PointCloudData
from stacking_common.pointcloud.plane_diagnostics import height_grid
from stacking_common.rendering.pointcloud import render_clouds
from .hand_eye import HandEyeCalibration
from .hand_eye_capture import atomic_json, digest
from .hand_eye_replay import inspect_session, inside, read_json
from .hand_eye_rgb import statistics
from .storage import write_xyz_pcd


PRODUCTION_REFERENCE_VIEWS = (0, 5, 6, 7, 8)


def _prepare_dense_workspace(workspace, names, boxes, *, crop_images, maximum_sources):
    """Crop undistorted pixels and rebuild their cameras before any depth maps exist."""
    import pycolmap
    from PIL import Image

    reconstruction = pycolmap.Reconstruction(workspace/'sparse')
    by_name = {image.name: image for image in reconstruction.images.values()}
    matrices, sizes, poses, crops = [], [], [], []
    for name in names:
        image = by_name[name]
        camera = reconstruction.cameras[image.camera_id]
        matrix = camera.calibration_matrix()
        inverse = np.eye(4)
        inverse[:3] = image.cam_from_world().matrix()
        pose = np.linalg.inv(inverse)
        original_size = (camera.width, camera.height)
        if crop_images:
            box, matrix = projected_roi_crop(matrix, pose, original_size, boxes, padding=16)
            with Image.open(workspace/'images'/name) as rgb:
                cropped = rgb.crop(tuple(box)).copy()
            cropped.save(workspace/'images'/name)
        else:
            box = np.array([0, 0, *original_size])
        size = tuple((box[2:] - box[:2]).tolist())
        matrices.append(matrix)
        sizes.append(size)
        poses.append(pose)
        crops.append(dict(name=name, crop_xyxy=box.tolist(),
                          uncropped_image_shape=[camera.height, camera.width],
                          image_shape=[size[1], size[0]],
                          retained_pixel_fraction=float(np.prod(size)/np.prod(original_size))))
    if crop_images:
        model = workspace/'cropped_model'
        write_pinhole_cameras(model, matrices, sizes, poses, names)
        pycolmap.Reconstruction(model).write(workspace/'sparse')
    sources = (select_source_views(matrices, sizes, poses, boxes, maximum_sources=maximum_sources)
               if maximum_sources is not None else
               [[j for j in range(len(names)) if j != i] for i in range(len(names))])
    return crops, sources


def dense_rgb_session(session, baseline, refinement, output, *,
                      mode_names=('original', 'refined'), reference_views=None,
                      crop_to_stones=False, maximum_sources=None, window_step=1, num_iterations=5):
    """Compare unchanged original and candidate camera poses using the same MVS settings."""
    started = time.monotonic()
    import pycolmap
    import open3d as o3d
    from PIL import Image
    session, baseline, refinement, output = [Path(p).resolve() for p in (session, baseline, refinement, output)]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (session, baseline, refinement)):
        raise ValueError('dense output must be separate from all input directories')
    manifest = inspect_session(session)
    base, refined = read_json(baseline/'summary.json'), read_json(refinement/'summary.json')
    session_hash = digest(session/'session.json')
    if (base.get('status') != 'complete' or refined.get('status') != 'complete'
            or base.get('session_sha256') != session_hash or refined.get('session_sha256') != session_hash
            or refined.get('baseline_hashes', {}).get('summary.json') != digest(baseline/'summary.json')
            or not refined.get('all_views_fit', {}).get('success')):
        raise ValueError('baseline/refinement must be complete and bound to this session')
    if not pycolmap.has_cuda:
        raise RuntimeError('dense RGB requires a CUDA-enabled pycolmap build')
    if type(window_step) is not int or window_step not in (1, 2):
        raise ValueError('COLMAP window_step must be 1 or 2')
    if type(num_iterations) is not int or not 1 <= num_iterations <= 5:
        raise ValueError('COLMAP num_iterations must be between 1 and 5')
    names, poses, sources = [], [], []
    for index, view in enumerate(manifest['views']):
        root = inside(session, view['path']); source = inside(session, base['views'][index]['source'])
        if not source.is_relative_to(root):
            raise ValueError('RGB source belongs to a different view')
        sources.append(source); poses.append(read_json(str(source)+'.json')['base_link_T_color']); names.append(f'view_{index:03d}.png')
    calibration = HandEyeCalibration.load(inside(session, manifest['views'][0]['path'])/'calibration/profile.yml')
    available_modes = dict(original=np.array(poses), refined=np.array(refined['all_views_fit']['base_link_T_color']))
    if not mode_names or any(name not in available_modes for name in mode_names):
        raise ValueError('dense RGB mode names must select original and/or refined poses')
    modes = {name: available_modes[name] for name in mode_names}
    if reference_views is None:
        reference_views = tuple(range(len(names)))
    else:
        reference_views = tuple(reference_views)
    if (not reference_views or len(set(reference_views)) != len(reference_views)
            or any(type(index) is not int or index not in range(len(names)) for index in reference_views)):
        raise ValueError('dense RGB reference views must be unique valid view indices')
    region = base['region']; low, high = np.array(region['minimum']), np.array(region['maximum'])
    # Keep the full recorded height interval: FEMTO does not constrain stone surface depths.
    if crop_to_stones:
        boxes = np.asarray([[[xy[0][0], xy[0][1], low[2]],
                             [xy[1][0], xy[1][1], high[2]]]
                            for xy in region['stone_boxes_xy']], dtype=float)
    else:
        boxes = np.asarray([[low, high]])
    bounds = depth_bounds_from_roi(np.concatenate(list(modes.values())), low, high)
    option_values = dict(gpu_index='0', max_image_size=1280, depth_min=bounds[0],
                         depth_max=bounds[1], num_threads=2, cache_size=2.,
                         write_consistency_graph=True, window_step=window_step,
                         num_iterations=num_iterations)
    photometric_options = pycolmap.PatchMatchOptions(
        dict(**option_values, geom_consistency=False, filter=False))
    geometric_options = pycolmap.PatchMatchOptions(
        dict(**option_values, geom_consistency=True, filter=True))
    output.mkdir(parents=True, exist_ok=False); (output/'images').mkdir()
    report = dict(status='running',kind='dense_rgb_mvs_experiment',schema_version=1,output_frame='base_link',
                  data_origin=manifest['data_origin'],session_sha256=session_hash,
                  baseline_sha256=digest(baseline/'summary.json'),refinement_sha256=digest(refinement/'summary.json'),
                  source_sha256={'hand_eye_dense.py':digest(__file__),'dense.py':digest(inspect.getfile(write_known_cameras))},
                  software=dict(pycolmap=pycolmap.__version__, numpy=np.__version__),
                  options=dict(photometric=photometric_options.todict(),
                               geometric=geometric_options.todict()), region=region,
                  reference_views=list(reference_views), crop_to_stones=crop_to_stones,
                  crop_world_boxes=boxes.tolist(), maximum_sources=maximum_sources,
                  accuracy_1mm_verified=False,modes={})
    atomic_json(output/'summary.json',report)
    try:
        for name,source in zip(names,sources):
            with np.load(str(source)+'.npz',allow_pickle=False) as arrays:
                rgb=arrays['color']
            if rgb.shape != (calibration.height,calibration.width,3) or rgb.dtype != np.uint8:
                raise ValueError('native RGB format differs from calibration')
            Image.fromarray(rgb).save(output/'images'/name)
        root=session/'femto_before'
        reference_path=inside(root,read_json(root/'field_scan.json')['fused_file'])
        reference=np.asarray(o3d.io.read_point_cloud(str(reference_path)).points)
        reference=reference[np.all((reference>=low)&(reference<=high),axis=1)]
        if not len(reference):raise ValueError('FEMTO comparison ROI is empty')
        coeff=np.array(region['support_plane'])
        def stone_parts(points):
            upper=points[:,2]>points[:,:2]@coeff[:2]+coeff[2]+.008
            return [points[upper & np.all((points[:,:2]>=lo)&(points[:,:2]<=hi),axis=1)] for lo,hi in region['stone_boxes_xy']]
        targets=stone_parts(reference)
        layers={'FEMTO reference':np.concatenate(targets)}
        rgb_layers = {}
        stone_ids = [item['stone_id'] for item in region.get('stones', [])]
        for mode,cameras in modes.items():
            prepare_started = time.monotonic()
            print(f'{mode}: prepare known camera poses',flush=True)
            folder=output/mode;folder.mkdir()
            write_known_cameras(folder/'model',calibration.color_matrix,calibration.color_distortion,
                                (calibration.width,calibration.height),cameras,names)
            workspace=folder/'workspace'
            pycolmap.undistort_images(workspace,folder/'model',output/'images',num_threads=2,
                                     undistort_options=pycolmap.UndistortCameraOptions(dict(max_image_size=1280)))
            crops, source_views = _prepare_dense_workspace(
                workspace, names, boxes, crop_images=crop_to_stones,
                maximum_sources=maximum_sources)
            # Geometric references and every source they consult need photometric maps.
            photometric_views = sorted(set(reference_views).union(
                *(set(source_views[i]) for i in reference_views)))
            def configuration(indices):
                return ''.join(names[i] + '\n' + ', '.join(names[j] for j in source_views[i]) + '\n'
                               for i in indices)
            mode_report = dict(crops=crops, source_views=source_views,
                               photometric_views=photometric_views, timings_s={})
            report['modes'][mode] = mode_report
            mode_report['timings_s']['prepare'] = time.monotonic() - prepare_started
            atomic_json(output/'summary.json', report)
            config_path = workspace/'stereo/patch-match.cfg'
            config_path.write_text(configuration(photometric_views))
            print(f'{mode}: CUDA photometric stereo for {len(photometric_views)} source views', flush=True)
            stage_started = time.monotonic()
            pycolmap.patch_match_stereo(workspace, options=photometric_options)
            mode_report['timings_s']['photometric'] = time.monotonic() - stage_started
            config_path.write_text(configuration(reference_views))
            print(f'{mode}: CUDA geometric stereo for {len(reference_views)} reference images', flush=True)
            stage_started = time.monotonic()
            pycolmap.patch_match_stereo(workspace, options=geometric_options)
            mode_report['timings_s']['geometric'] = time.monotonic() - stage_started
            export_started = time.monotonic()
            reconstruction=pycolmap.Reconstruction(workspace/'sparse')
            by_name={img.name:img for img in reconstruction.images.values()}
            views=[];clouds=[]
            for index in reference_views:
                name = names[index]
                depth_path=workspace/'stereo/depth_maps'/(name+'.geometric.bin')
                depth=read_colmap_depth(depth_path)
                image=by_name[name];cam=reconstruction.cameras[image.camera_id]
                if depth.shape != (cam.height,cam.width):raise ValueError('dense depth size differs from undistorted camera')
                world_to_camera=np.eye(4);world_to_camera[:3]=image.cam_from_world().matrix()
                points,pixels=backproject_colmap_depth(depth,cam.calibration_matrix(),np.linalg.inv(world_to_camera))
                valid=np.all((points>=low)&(points<=high),axis=1)
                points,pixels=points[valid],pixels[valid];clouds.append(points)
                np.savez_compressed(folder/f'view_{index:03d}.npz',points=points,pixel_index=pixels,
                                    image_shape=np.array(depth.shape),matrix=cam.calibration_matrix(),
                                    base_link_T_camera=np.linalg.inv(world_to_camera),
                                    crop_xyxy=np.array(crops[index]['crop_xyxy']),
                                    uncropped_image_shape=np.array(crops[index]['uncropped_image_shape']))
                write_xyz_pcd(folder/f'view_{index:03d}.pcd',points)
                views.append(dict(view=index,name=name,depth_sha256=digest(depth_path),
                                  crop_xyxy=crops[index]['crop_xyxy'],
                                  uncropped_image_shape=crops[index]['uncropped_image_shape'],
                                  image_sha256=digest(workspace/'images'/name),
                                  positive_depth_pixels=int(np.count_nonzero(np.isfinite(depth)&(depth>0))),roi_points=len(points),
                                  stone_points=[len(p) for p in stone_parts(points)]))
            all_points=np.concatenate(clouds)
            merged=fixed_grid_centroids(PointCloudData(all_points),.001).points
            write_xyz_pcd(folder/'rgb_dense_voxel_1mm.pcd',merged)
            groups=[];parts=stone_parts(merged)
            for stone,(points,target) in enumerate(zip(parts,targets)):
                if not len(target):raise ValueError('empty FEMTO stone reference')
                distances=cKDTree(target).query(points)[0]*1000 if len(points) else np.empty(0)
                reverse=cKDTree(points).query(target)[0]*1000 if len(points) else np.full(len(target),np.inf)
                per_view=[stone_parts(p)[stone] for p in clouds]
                # Requiring agreement in another reference depth map is an internal
                # consistency check; all maps share source images and calibration.
                support=[]
                for i,p in enumerate(per_view):
                    others=np.concatenate([v for j,v in enumerate(per_view) if i!=j])
                    d=cKDTree(others).query(p)[0] if len(others) else np.full(len(p),np.inf)
                    support.append(dict(view=views[i]['view'],points=len(p),fraction_within_2mm_of_another_map=float(np.mean(d<=.002)) if len(p) else None,
                                        distance_to_other_maps_mm=statistics(d[np.isfinite(d)]*1000)))
                grids=[height_grid(p,low[:2],high[:2],.005,minimum_samples=5)[1] for p in per_view]
                common=np.logical_and.reduce([np.isfinite(g) for g in grids])
                spread=np.ptp(np.stack(grids)[:,common],axis=0) if common.any() else np.empty(0)
                groups.append(dict(stone=stone,voxel_points=len(points),rgb_to_femto_mm=statistics(distances),
                              femto_fraction_within_3mm=float(np.mean(reverse<=3)),
                              rgb_fraction_farther_than_3mm=float(np.mean(distances>3)) if len(points) else None,
                              per_view_support=support,shared_5mm_cells=int(common.sum()),
                              across_view_height_spread_mm=statistics(spread*1000)))
                layers[f'{mode} RGB stone {stone}']=points
                stone_id = stone_ids[stone] if stone < len(stone_ids) and stone_ids[stone] is not None else f'cluster {stone}'
                rgb_layers[f'{mode} RGB stone {stone_id}'] = points
            mode_report['timings_s']['cloud_export'] = time.monotonic() - export_started
            mode_report.update(views=views,raw_roi_points=len(all_points),voxel_roi_points=len(merged),groups=groups)
            atomic_json(output/'summary.json',report)
        render_started = time.monotonic()
        render_clouds(output/'comparison',layers,{},title='Dense RGB multi-view stereo / FEMTO (diagnostic candidates)')
        render_clouds(output/'rgb_only',rgb_layers,{},title='Wrist RGB reconstructed stone surfaces (MVS)')
        report['render_s'] = time.monotonic() - render_started
        report.update(status='complete',elapsed_s=time.monotonic()-started,point_count=sum(m['voxel_roi_points'] for m in report['modes'].values()),
            limitations=['Depth is estimated from RGB alone; no wrist depth, FEMTO geometry, ICP or plane adjustment enters MVS.',
              'Optional image crops use padded stone XY regions and the full recorded Z interval; depth values are still estimated from RGB.',
              'Upper-surface comparison masks omit bases and can contain edge artifacts; cropped runs omit background outside the stone region.',
              'Saved pixel_index values address the cropped undistorted image; crop_xyxy and uncropped_image_shape preserve the offset to the full undistorted image.',
              'All reference depth maps share source images and camera poses. Their agreement is internal consistency, not independent accuracy.',
              'The saved combined cloud is a 1mm voxel centroid of already filtered depth maps, not an uncertainty-weighted or accepted production fusion.',
              'RGB points far from FEMTO may be additional visible surfaces or errors; counts do not distinguish those possibilities.',
              'NN coverage depends on alignment, visibility and sampling. Voxel size and point density are not accuracy guarantees.',
              'Every selected pose mode uses identical image, MVS and metric depth-range settings; CUDA PatchMatch may have nondeterministic numerical variation.'])
        atomic_json(output/'summary.json',report)
        return report
    except BaseException as error:
        report.update(status='failed',error=str(error));atomic_json(output/'summary.json',report)
        raise


def dense_rgb_production_session(session, baseline, refinement, output):
    """Reconstruct the refined route once using the five widest-baseline reference views."""
    return dense_rgb_session(session, baseline, refinement, output,
                             mode_names=('refined',),
                             reference_views=PRODUCTION_REFERENCE_VIEWS,
                             crop_to_stones=True, maximum_sources=4)


def dense_rgb_full_session(session, baseline, refinement, output):
    """Full-image, all-source retry with the same refined poses and reference views."""
    return dense_rgb_session(session, baseline, refinement, output,
                             mode_names=('refined',),
                             reference_views=PRODUCTION_REFERENCE_VIEWS)
