"""Offline, uncorrected table consistency measurements from portable sessions."""

from itertools import combinations
from pathlib import Path

import numpy as np

from stacking_common.pointcloud.depth import backproject_depth
from stacking_common.pointcloud.plane_diagnostics import (
    height_grid, fit_height_plane, difference_statistics)
from .hand_eye_capture import atomic_json, digest
from .hand_eye_replay import inspect_session, inside, native_depth_rays, read_json


def diagnose_session(session, region_file, output, *, visualize=True):
    """Compare identical observed XY cells; no registration or bias estimation."""
    session, output = Path(session).resolve(), Path(output).resolve()
    region_file = Path(region_file).resolve()
    manifest = inspect_session(session)
    if len(manifest['views']) < 2:
        raise ValueError('diagnostics require at least two views')
    if output.is_relative_to(session) or session.is_relative_to(output):
        raise ValueError('diagnostic output must be separate from session inputs')
    region_hash = digest(region_file)
    region = read_json(region_file)
    lower, upper = np.asarray(region['minimum'], float), np.asarray(region['maximum'], float)
    exclusions = np.asarray(region.get('exclude_xy', []), float).reshape(-1, 2, 2)
    cell_size = float(region['cell_size_m'])
    labels = region.get('view_labels', [view['path'] for view in manifest['views']])
    if (region.get('frame') != 'base_link' or lower.shape != (3,) or upper.shape != (3,)
            or not np.all(np.isfinite([lower, upper])) or np.any(lower >= upper)
            or not np.all(np.isfinite(exclusions))
            or np.any(exclusions[:, 0] >= exclusions[:, 1])
            or not isinstance(labels, list) or len(labels) != len(manifest['views'])
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError('invalid base_link region or unique view labels')
    xy, _, _ = height_grid(np.empty((0, 3)), lower[:2], upper[:2], cell_size)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, kind='hand_eye_table_consistency', status='running',
        data_origin=manifest['data_origin'], output_frame='base_link',
        session_sha256=digest(session/'session.json'), region_sha256=region_hash,
        source_hashes={'adapter': digest(__file__),
                      'numerics': digest(fit_height_plane.__code__.co_filename),
                      'projection': digest(backproject_depth.__code__.co_filename),
                      'native_geometry': digest(native_depth_rays.__code__.co_filename)},
        numpy_version=np.__version__, region=region, views=[], pairs=[],
        method='10+ frame bursts recommended; minimum 5 points per cell per frame; '
               'median Z per frame/cell, then median across frames; only cells observed '
               'in every frame of every view are compared; equal cell weights; '
               'plane z=ax+by+c with five 3-MAD trimming steps and 1 mm floor',
        limitations=['Fixed target/scene is assumed, not verified.',
                     'Cell medians can differ due to sampling on a sloped/nonflat surface.',
                     'Frame statistics measure burst repeatability, not independent trials.',
                     'Distance, orientation, hand-eye and robot errors are not separated by these metrics.',
                     'No independent base-frame reference; absolute 1 mm accuracy is unverified.'],
        registration_applied=False, depth_correction_applied=False, accuracy_1mm_verified=False)
    atomic_json(output/'diagnostics.json', report)
    try:
        grids, all_counts, poses = [], [], []
        for index, view in enumerate(manifest['views']):
            root = inside(session, view['path'])
            device, capture = read_json(root/'device.json'), read_json(root/'manifest.json')
            rays = native_depth_rays(device['streams']['depth'])
            frame_grids, counts, camera_poses, depths, frames = [], [], [], [], []
            for stem in capture['frames']:
                record = read_json(root/(stem+'.json'))
                pose = np.asarray(record['base_link_T_depth'])
                with np.load(root/(stem+'.npz'), allow_pickle=False) as raw:
                    cloud, _ = backproject_depth(raw['depth'], rays, device['depth_metres_per_unit'])
                world = cloud.transformed(pose).points
                keep = np.all((world >= lower) & (world <= upper), axis=1)
                for exclude_min, exclude_max in exclusions:
                    keep &= ~np.all((world[:, :2] >= exclude_min) & (world[:, :2] <= exclude_max), axis=1)
                if keep.sum() < 30:
                    raise ValueError(f'{view["path"]}/{stem}: insufficient table-region points')
                _, heights, count = height_grid(world[keep], lower[:2], upper[:2], cell_size)
                frame_grids.append(heights)
                counts.append(count)
                camera_poses.append(pose)
                depths.append(float(np.median(cloud.points[keep, 2])))
                frames.append(dict(source=view['path']+'/'+stem, timestamp_ns=record['depth']['timestamp_ns'],
                                   selected_points=int(keep.sum()), valid_cells=int(np.isfinite(heights).sum())))
            grids.append(np.array(frame_grids))
            all_counts.append(np.array(counts))
            poses.append(np.array(camera_poses))
            report['views'].append(dict(label=labels[index], source=view['path'], frames=frames,
                median_native_optical_depth_m=float(np.median(depths)),
                optical_depth_median_scope='selected table-region points, median of frame medians',
                first_base_link_T_depth=camera_poses[0].tolist()))

        # Intersection is shared by every frame and view. Missing cells are not filled.
        common = np.logical_and.reduce([np.all(np.isfinite(grid), axis=0) for grid in grids])
        if common.sum() < 6:
            raise ValueError('fewer than six table cells observed in every frame/view; inspect region/coverage')
        report['common_cell_count'] = int(common.sum())
        report['common_cell_fraction_of_bounding_grid'] = float(common.mean())
        anchor = xy[common].mean(axis=0)
        report['evaluation_xy_m'] = anchor.tolist()
        heights = []
        for view, grid, camera_poses in zip(report['views'], grids, poses):
            height = np.median(grid[:, common], axis=0)
            plane = fit_height_plane(xy[common], height)
            per_frame_planes = [fit_height_plane(xy[common], frame[common]) for frame in grid]
            frame_heights = [float(np.r_[anchor, 1.]@fit['parameters']) for fit in per_frame_planes]
            axis = camera_poses[0, :3, 2]
            view.update(plane=plane, fitted_height_at_common_anchor_mm=float(np.r_[anchor, 1.]@plane['parameters'])*1000,
                per_frame_plane_height_std_mm=float(np.std(frame_heights)*1000),
                median_cell_temporal_std_mm=float(np.median(np.std(grid[:, common], axis=0))*1000),
                optical_axis_to_fitted_plane_normal_deg=float(np.degrees(np.arccos(np.clip(abs(axis@plane['normal']), 0, 1)))),
                max_recorded_translation_from_first_mm=float(np.linalg.norm(camera_poses[:, :3, 3]-camera_poses[0, :3, 3], axis=1).max()*1000))
            heights.append(height)
        for first, second in combinations(range(len(grids)), 2):
            a, b = report['views'][first], report['views'][second]
            pose_a, pose_b = poses[first][0], poses[second][0]
            report['pairs'].append(dict(first=labels[first], second=labels[second],
                sign='second minus first', observed_height_difference=difference_statistics(heights[second]-heights[first]),
                fitted_height_at_anchor_difference_mm=b['fitted_height_at_common_anchor_mm']-a['fitted_height_at_common_anchor_mm'],
                plane_normal_difference_deg=float(np.degrees(np.arccos(np.clip(np.dot(a['plane']['normal'], b['plane']['normal']), -1, 1)))),
                optical_axis_change_deg=float(np.degrees(np.arccos(np.clip(pose_a[:3, 2]@pose_b[:3, 2], -1, 1)))),
                camera_translation_mm=float(np.linalg.norm(pose_b[:3, 3]-pose_a[:3, 3])*1000)))
        arrays = dict(xy=xy, common_mask=common, common_heights=np.array(heights))
        for index, (grid, counts) in enumerate(zip(grids, all_counts)):
            arrays[f'view_{index:03d}_frame_heights'] = grid
            arrays[f'view_{index:03d}_sample_counts'] = counts
        np.savez_compressed(output/'table_cells.npz', **arrays)
        if visualize:
            from stacking_common.rendering.height_differences import render_height_differences
            report['source_hashes']['renderer'] = digest(render_height_differences.__code__.co_filename)
            render_height_differences(output, xy, common, heights, labels, frame='base_link')
        if inspect_session(session) != manifest or digest(session/'session.json') != report['session_sha256']:
            raise ValueError('session changed during diagnostics')
        if digest(region_file) != region_hash:
            raise ValueError('region changed during diagnostics')
        report['output_hashes'] = {p.name: digest(p) for p in output.iterdir() if p.name != 'diagnostics.json'}
        report['status'] = 'complete'
        atomic_json(output/'diagnostics.json', report)
    except BaseException as error:
        report.update(status='incomplete', error=f'{type(error).__name__}: {error}')
        atomic_json(output/'diagnostics.json', report)
        raise
    return report
