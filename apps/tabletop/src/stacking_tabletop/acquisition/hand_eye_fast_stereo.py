"""Offline Fast-FoundationStereo reconstruction of calibrated wrist RGB captures."""
import argparse
import inspect
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from stacking_common.multiview.learned_stereo import FastFoundationStereo, select_pairs, rectify_pair, disparity_points
from stacking_common.pointcloud.processing import fixed_grid_centroids
from stacking_common.pointcloud.types import PointCloudData
from stacking_common.rendering.pointcloud import render_clouds
from .hand_eye import HandEyeCalibration
from .hand_eye_capture import atomic_json, digest
from .hand_eye_replay import inspect_session, inside, read_json
from .storage import write_xyz_pcd


def reconstruct(session, baseline, refinement, output, *, repository, checkpoint, max_size=960, region_file=None):
    started = time.monotonic()
    session, output = Path(session).resolve(), Path(output).resolve()
    manifest = inspect_session(session)
    session_hash = digest(session/'session.json')
    if region_file is not None:
        if baseline is not None or refinement is not None:
            raise ValueError('choose recorded camera poses or refined camera inputs, not both')
        region_file = Path(region_file).resolve()
        region = read_json(region_file)
        if region.get('frame') != 'base_link':
            raise ValueError('recorded-pose reconstruction requires a base_link region')
        views, poses = [], []
        for view in manifest['views']:
            root = inside(session, view['path'])
            source = inside(root, read_json(root/'manifest.json')['frames'][0])
            views.append(dict(source=str(source.relative_to(session))))
            poses.append(read_json(str(source)+'.json')['base_link_T_color'])
        base = dict(region=region, views=views)
        poses = np.asarray(poses)
        input_hashes = dict(region_sha256=digest(region_file), camera_pose_source='recorded_robot_hand_eye')
        inputs = [session, region_file]
    else:
        if baseline is None or refinement is None:
            raise ValueError('provide baseline/refinement or a region for recorded camera poses')
        baseline, refinement = Path(baseline).resolve(), Path(refinement).resolve()
        base, refined = [read_json(p/'summary.json') for p in (baseline, refinement)]
        if (base.get('status') != 'complete' or refined.get('status') != 'complete'
                or base.get('session_sha256') != session_hash or refined.get('session_sha256') != session_hash
                or refined.get('baseline_hashes', {}).get('summary.json') != digest(baseline/'summary.json')
                or not refined.get('all_views_fit', {}).get('success')):
            raise ValueError('baseline/refinement must be complete and bound to this session')
        poses = np.array(refined['all_views_fit']['base_link_T_color'])
        input_hashes = dict(baseline_sha256=digest(baseline/'summary.json'),
                            refinement_sha256=digest(refinement/'summary.json'), camera_pose_source='rgb_refined')
        inputs = [session, baseline, refinement]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in inputs):
        raise ValueError('stereo output must be separate from inputs')
    if len(poses) != len(manifest['views']) or len(base['views']) != len(poses):
        raise ValueError('RGB camera/view count mismatch')
    if not 128 <= max_size <= 1280:
        raise ValueError('max_size must be between 128 and 1280')
    images, matrices, distortions = [], [], []
    for i, view in enumerate(manifest['views']):
        root = inside(session, view['path'])
        source = inside(session, base['views'][i]['source'])
        if not source.is_relative_to(root):
            raise ValueError('RGB source belongs to a different view')
        calibration = HandEyeCalibration.load(root/'calibration/profile.yml')
        with np.load(str(source)+'.npz', allow_pickle=False) as arrays:
            rgb = arrays['color'].copy()
        if rgb.shape != (calibration.height, calibration.width, 3):
            raise ValueError('RGB shape differs from calibration')
        images.append(rgb); matrices.append(calibration.color_matrix); distortions.append(calibration.color_distortion)
    region = base['region']; low, high = np.array(region['minimum']), np.array(region['maximum'])
    boxes = np.array([[[a[0], a[1], low[2]], [b[0], b[1], high[2]]]
                      for a, b in region['stone_boxes_xy']])
    from itertools import product
    corners = np.concatenate([np.array(list(product(*zip(a, b)))) for a, b in boxes])
    output.mkdir(parents=True, exist_ok=False); (output/'refined').mkdir()
    report = dict(status='running', kind='fast_foundation_stereo_experiment', schema_version=1,
                  experimental=True, output_frame='base_link', data_origin=manifest['data_origin'],
                  accuracy_1mm_verified=False, session_sha256=session_hash,
                  **input_hashes,
                  checkpoint_sha256=digest(checkpoint), repository_commit=subprocess.check_output(
                      ['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip(),
                  source_sha256={'hand_eye_fast_stereo.py': digest(__file__),
                                 'learned_stereo.py': digest(inspect.getfile(rectify_pair))},
                  region=region, max_size=max_size,
                  iterations=8, left_right_tolerance_px=1., pairs=[], timings_s={}, modes={'refined': {}})
    atomic_json(output/'summary.json', report)
    try:
        t = time.monotonic(); model = FastFoundationStereo(repository, checkpoint)
        report['timings_s']['model_load'] = time.monotonic()-t
        clouds = []
        for i, j in select_pairs(poses):
            print(f'fast_stereo: views {i} / {j}', flush=True)
            t = time.monotonic()
            rgb, masks, k, pose, distance = rectify_pair([images[i], images[j]],
                [matrices[i], matrices[j]], [distortions[i], distortions[j]], poses[[i, j]], max_size)
            depths = ((corners-pose[:3, 3]) @ pose[:3, :3])[:, 2]
            if depths.min() <= 0:
                raise ValueError('stone ROI crosses rectified camera plane')
            max_disp = int(np.ceil((k[0, 0]*distance/depths.min()+16)/32)*32)
            # The released checkpoint has a finite disparity positional embedding.
            # Reduce image scale (and intrinsics together), never truncate ROI depths.
            if max_disp > 416:
                pair_size = max(128, int(max_size * 384 / max_disp))
                rgb, masks, k, pose, distance = rectify_pair([images[i], images[j]],
                    [matrices[i], matrices[j]], [distortions[i], distortions[j]], poses[[i, j]], pair_size)
                max_disp = int(np.ceil((k[0, 0]*distance/depths.min()+16)/32)*32)
            if max_disp > 416:
                raise ValueError('pair requires disparities beyond checkpoint capacity')
            preparation = time.monotonic()-t
            t = time.monotonic(); disparity = model.predict(*rgb, max_disp)
            forward_s = time.monotonic()-t
            t = time.monotonic()
            reverse = np.fliplr(model.predict(np.fliplr(rgb[1]), np.fliplr(rgb[0]), max_disp)).copy()
            reverse_s = time.monotonic()-t
            t = time.monotonic()
            points, valid = disparity_points(disparity, reverse, masks, k, pose, distance)
            points = points[np.all((points >= low) & (points <= high), axis=1)]
            clouds.append(points)
            name = f'pair_{i:03d}_{j:03d}'
            np.savez_compressed(output/'refined'/f'{name}.npz', disparity=disparity,
                reverse_disparity=reverse, valid=valid, matrix=k, base_link_T_camera=pose,
                baseline_m=distance, points=points)
            from PIL import Image
            Image.fromarray(np.concatenate(rgb, axis=1)).save(output/'refined'/f'{name}.png')
            report['pairs'].append(dict(views=[i, j], baseline_m=distance, image_shape=list(disparity.shape),
                max_disparity=max_disp, consistent_pixels=int(valid.sum()), roi_points=len(points),
                timings_s=dict(prepare=preparation, forward=forward_s, reverse=reverse_s,
                               export=time.monotonic()-t)))
            atomic_json(output/'summary.json', report)
        t = time.monotonic()
        merged = fixed_grid_centroids(PointCloudData(np.concatenate(clouds)), .001).points
        if not len(merged):
            raise ValueError('no stereo points survived consistency and ROI checks')
        write_xyz_pcd(output/'refined/rgb_dense_voxel_1mm.pcd', merged)
        report['timings_s']['merge'] = time.monotonic()-t
        report['timings_s']['reconstruction_total'] = time.monotonic()-started
        plane = np.array(region['support_plane'])
        stone_points = merged[merged[:, 2] > merged[:, :2] @ plane[:2]+plane[2]+.008]
        keep = np.zeros(len(stone_points), bool)
        for a, b in region['stone_boxes_xy']:
            keep |= np.all((stone_points[:, :2] >= a) & (stone_points[:, :2] <= b), axis=1)
        stone_points = stone_points[keep]
        render_clouds(output/'rgb_only', {'Fast-FoundationStereo stones': stone_points}, {},
                      title='Experimental wrist RGB stereo — metric base_link')
        report['modes']['refined'].update(voxel_points=len(merged), stone_points=len(stone_points))
        report['status'] = 'complete'
        report['timings_s']['total_with_viewer'] = time.monotonic()-started
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        atomic_json(output/'summary.json', report)
    return report



def reconstruct_with_runtime(session, baseline, refinement, output, *, repository, checkpoint,
                             region_file=None, check=None):
    """Run optional model dependencies in their environment; keep the caller's ROS/GUI runtime."""
    repository = Path(repository).resolve()
    python = repository.parents[1] / 'venvs/fast-stereo/bin/python'
    if not python.is_file():
        raise RuntimeError('Fast-FoundationStereo environment is missing. Follow the setup in '
                           'docs/tabletop-rgb-scene.md or select --dense-backend colmap.')
    if check is None and Path(sys.executable).absolute() == python.absolute():
        return reconstruct(session, baseline, refinement, output,
                           repository=repository, checkpoint=checkpoint, region_file=region_file)
    command = [str(python), '-u', '-m', 'stacking_tabletop.acquisition.hand_eye_fast_stereo']
    for name, value in dict(session=session, baseline=baseline, refinement=refinement,
                            output=output, repository=repository, checkpoint=checkpoint, region_file=region_file).items():
        if value is None:
            continue
        command += ['--'+name.replace('_', '-'), str(Path(value).resolve())]
    print('Fast-FoundationStereo: starting isolated GPU reconstruction', flush=True)
    process = subprocess.Popen(command)
    try:
        if check is None:
            code = process.wait()
        else:
            while True:
                check()
                try:
                    code = process.wait(timeout=.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
        if code:
            raise RuntimeError(f'Fast-FoundationStereo exited with code {code}; see retained output')
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    report = read_json(Path(output)/'summary.json')
    if report.get('status') != 'complete':
        raise RuntimeError('Fast-FoundationStereo did not complete reconstruction')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('session', 'baseline', 'refinement', 'output'):
        parser.add_argument('--'+name, type=Path, required=name in ('session', 'output'))
    parser.add_argument('--region-file', type=Path, help='reconstruct using recorded camera poses without FEMTO')
    parser.add_argument('--repository', type=Path, default=Path('.data/vendor/Fast-FoundationStereo'))
    parser.add_argument('--checkpoint', type=Path,
        default=Path('.data/vendor/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth'))
    parser.add_argument('--max-size', type=int, default=960)
    args = parser.parse_args()
    report = reconstruct(**vars(args))
    print(report['timings_s'], flush=True)


if __name__ == '__main__':
    main()
