"""Offline fixed-camera RGB triangulation experiment on a verified session."""

from itertools import combinations
import inspect
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from stacking_common.multiview.features import extract_features, match_features, build_tracks
from stacking_common.multiview.geometry import triangulate_track, epipolar_distances
from stacking_common.rendering.pointcloud import render_clouds
from stacking_common.pointcloud.types import validate_rigid_transform
from .hand_eye import HandEyeCalibration
from .hand_eye_capture import atomic_json, digest
from .hand_eye_replay import inspect_session, inside, read_json
from .storage import write_xyz_pcd


def statistics(values):
    values = np.asarray(values, float)
    return (dict(count=len(values), median=float(np.median(values)),
                 p95=float(np.percentile(values, 95)), maximum=float(np.max(values)))
            if len(values) else dict(count=0, median=None, p95=None, maximum=None))


def rgb_experiment(session, region_file, output):
    """Keep fixed calibration; fit two views and test other descriptor-linked views.

    FEMTO supplies comparison data and coarse projected stone masks only. Neither
    FEMTO nor wrist depth participates in correspondence or triangulation.
    """
    import cv2
    import open3d as o3d
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    session, output = Path(session).resolve(), Path(output).resolve()
    region_file = Path(region_file).resolve()
    if output.is_relative_to(session) or session.is_relative_to(output):
        raise ValueError('RGB output must be separate from session inputs')
    manifest = inspect_session(session)
    if len(manifest['views']) < 3 or 'femto_before' not in manifest['references']:
        raise ValueError('RGB experiment requires >=3 views and a FEMTO before reference')
    region = read_json(region_file)
    low, high = np.array(region['minimum']), np.array(region['maximum'])
    boxes = np.array(region['stone_boxes_xy'], float)
    plane = np.array(region['support_plane'], float)
    if (region.get('frame') != 'base_link' or low.shape != (3,) or high.shape != (3,)
            or not np.isfinite([low, high]).all() or np.any(high <= low)
            or boxes.ndim != 3 or boxes.shape[1:] != (2, 2) or not len(boxes)
            or not np.isfinite(boxes).all() or np.any(boxes[:, 1] <= boxes[:, 0])
            or plane.shape != (3,) or not np.isfinite(plane).all()):
        raise ValueError('invalid base_link region/stone boxes/support plane')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', kind='fixed_pose_rgb_feasibility', schema_version=1,
                  session_sha256=digest(session/'session.json'), region_sha256=digest(region_file),
                  output_frame='base_link', data_origin=manifest['data_origin'], accuracy_1mm_verified=False)
    report['source_sha256'] = {name: digest(path) for name, path in (
        ('hand_eye_rgb.py', __file__),
        ('geometry.py', inspect.getfile(triangulate_track)),
        ('features.py', inspect.getfile(extract_features)))}
    atomic_json(output/'summary.json', report)
    try:
        cv2.setNumThreads(2)
        cv2.setRNGSeed(0)
        reference_root = session/'femto_before'
        reference_path = inside(reference_root, read_json(reference_root/'field_scan.json')['fused_file'])
        reference = np.asarray(o3d.io.read_point_cloud(str(reference_path)).points)
        reference = reference[np.all((reference >= low) & (reference <= high), axis=1)]
        if not len(reference):
            raise ValueError('empty FEMTO comparison region')
        upper = reference[:, 2] > reference[:, :2] @ plane[:2] + plane[2] + .008
        stones = [reference[upper & np.all((reference[:, :2] >= lo) & (reference[:, :2] <= hi), axis=1)]
                  for lo, hi in boxes]
        if any(len(p) < 3 for p in stones):
            raise ValueError('each comparison stone region requires >=3 FEMTO points')
        images, features, poses, sources, feature_labels = [], [], [], [], []
        for i, view in enumerate(manifest['views']):
            root = inside(session, view['path'])
            calibration = HandEyeCalibration.load(root/'calibration/profile.yml')
            k, distortion = calibration.color_matrix, calibration.color_distortion
            stem = read_json(root/'manifest.json')['frames'][0]
            source = inside(root, stem)
            with np.load(str(source)+'.npz', allow_pickle=False) as frame:
                rgb = frame['color']
            if rgb.shape != (calibration.height, calibration.width, 3):
                raise ValueError('RGB shape differs from fitted calibration')
            pose = validate_rigid_transform(read_json(str(source)+'.json')['base_link_T_color'])
            feature = extract_features(rgb, k, distortion)
            mask = np.zeros(rgb.shape[:2], np.uint8)
            camera_T_base = np.linalg.inv(pose)
            for stone_index, points in enumerate(stones):
                camera = points @ camera_T_base[:3, :3].T + camera_T_base[:3, 3]
                camera = camera[camera[:, 2] > 0]
                if len(camera) < 3:
                    continue
                pixels, _ = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), k, distortion)
                hull = cv2.convexHull(pixels.astype(np.float32)).astype(np.int32)
                one = np.zeros_like(mask)
                cv2.fillConvexPoly(one, hull, 1)
                one = cv2.dilate(one, np.ones((11, 11), np.uint8))
                mask[one > 0] = stone_index + 1
            pixels = np.rint(feature['pixels']).astype(int)
            labels = mask[np.clip(pixels[:, 1], 0, len(mask)-1), np.clip(pixels[:, 0], 0, mask.shape[1]-1)]
            images.append(rgb); features.append(feature); poses.append(pose)
            feature_labels.append(labels); sources.append(str(source.relative_to(session)))
            np.savez_compressed(output/f'features_{i:03d}.npz', **{key: feature[key] for key in ('pixels', 'normalized', 'descriptors')},
                                projected_stone_label=labels)
        poses = np.array(poses)
        pairs, edges = [], []
        for i, j in combinations(range(len(images)), 2):
            matches = match_features(features[i]['descriptors'], features[j]['descriptors'])
            pair = np.array(matches, dtype=int).reshape(-1, 2)
            edges.extend([((i, a), (j, b)) for a, b in matches])
            error = epipolar_distances(features[i]['normalized'][pair[:, 0]], features[j]['normalized'][pair[:, 1]],
                                       poses[i], poses[j], k) if len(pair) else np.empty(0)
            stone_mask = (feature_labels[i][pair[:, 0]] > 0) & (feature_labels[i][pair[:, 0]] == feature_labels[j][pair[:, 1]])
            pairs.append(dict(views=[i, j], matches=len(matches), stone_matches=int(stone_mask.sum()),
                              epipolar_px=statistics(error), stone_epipolar_px=statistics(error[stone_mask]),
                              baseline_mm=float(np.linalg.norm(poses[i, :3, 3]-poses[j, :3, 3])*1000)))
        tracks, conflicts = build_tracks(edges)
        rows = []
        for index, track in enumerate(tracks):
            if len(track) < 3:
                continue
            views = np.array([v for v, _ in track])
            xy = np.array([features[v]['normalized'][f] for v, f in track])
            labels = np.array([feature_labels[v][f] for v, f in track])
            stone = max(set(labels.tolist()), key=lambda label: (np.count_nonzero(labels == label), -label))
            if np.count_nonzero(labels == stone) <= len(labels)//2:
                stone = 0
            row = dict(track=index, observations=track, stone=int(stone)-1)
            try:
                result = triangulate_track(xy, poses[views])
                errors = np.linalg.norm(result['residuals'] * np.array([k[0, 0], k[1, 1]]), axis=1)
                fit = result['fit_pair']; held = np.ones(len(track), bool); held[fit] = False
                reasons = []
                if np.any(result['depths'] <= 0): reasons.append('behind_camera')
                if result['parallax_deg'] < 2: reasons.append('parallax_below_2deg')
                if errors[fit].max() > 2: reasons.append('fit_reprojection_over_2px')
                if errors[held].max() > 2: reasons.append('unused_view_reprojection_over_2px')
                row.update(point=result['point'].tolist(), fit_views=views[fit].tolist(),
                           residual_px=errors.tolist(), unused_view_residual_px=errors[held].tolist(),
                           parallax_deg=result['parallax_deg'], baseline_mm=result['baseline_m']*1000,
                           reasons=reasons, geometry_pass=not reasons,
                           in_roi=bool(np.all((result['point'] >= low) & (result['point'] <= high))))
            except ValueError as error:
                row.update(geometry_pass=False, reasons=[str(error)], in_roi=False)
            rows.append(row)
        passed = [r for r in rows if r['geometry_pass'] and r['in_roi']]
        points = np.array([r['point'] for r in passed]).reshape(-1, 3)
        write_xyz_pcd(output/'rgb_sparse.pcd', points)
        np.savez_compressed(output/'rgb_sparse.npz', points=points,
                            track_index=[r['track'] for r in passed], stone=[r['stone'] for r in passed])
        groups = []
        layers = {'FEMTO comparison': reference, 'RGB sparse, geometry gates passed': points}
        for stone in range(-1, len(stones)):
            selected = [r for r in passed if r['stone'] == stone]
            p = np.array([r['point'] for r in selected]).reshape(-1, 3)
            target = reference if stone < 0 else stones[stone]
            distances = cKDTree(target).query(p)[0]*1000 if len(p) else np.empty(0)
            femto_near = cKDTree(p).query(target)[0] if len(p) else np.full(len(target), np.inf)
            groups.append(dict(stone=stone, passed_tracks=len(p), rgb_to_femto_mm=statistics(distances),
                               femto_fraction_within_3mm_of_sparse_rgb=float(np.mean(femto_near <= .003))))
            if stone >= 0:
                layers[f'RGB stone {stone}']=p
        for i, image in enumerate(images):
            fig, ax = plt.subplots(figsize=(12.8, 7.2)); ax.imshow(image)
            for stone in range(-1, len(stones)):
                dots = [features[i]['pixels'][f] for row in passed if row['stone'] == stone
                        for v, f in row['observations'] if v == i]
                if dots:
                    dots = np.array(dots)
                    ax.scatter(*dots.T, s=24, facecolors='none', edgecolors=('cyan' if stone < 0 else ['red', 'lime', 'yellow'][stone % 3]),
                               label='background/other' if stone < 0 else f'stone {stone}')
            ax.set_title(f'View {i}: RGB tracks passing geometry gates (coarse FEMTO masks)'); ax.axis('off'); ax.legend()
            fig.tight_layout(); fig.savefig(output/f'tracks_{i:03d}.png', dpi=120); plt.close(fig)
        atomic_json(output/'tracks.json', dict(tracks=rows, ambiguous_components=conflicts))
        render_clouds(output/'comparison', layers, {}, title='RGB sparse triangulation / FEMTO: fixed poses, no depth fitting')
        held_errors = [e for row in rows for e in row.get('unused_view_residual_px', [])]
        report.update(status='complete', point_count=len(points), views=[dict(source=s, keypoints=len(f['pixels']), sharpness=f['sharpness']) for s, f in zip(sources, features)],
                      pairs=pairs, descriptor_tracks=len(tracks), ambiguous_components=conflicts,
                      tracks_with_at_least_3_views=len(rows), passed_geometry=sum(r['geometry_pass'] for r in rows),
                      unused_view_residuals_before_gating_px=statistics(held_errors), groups=groups,
                      thresholds=dict(mutual_sift_ratio=.75, minimum_views=3, minimum_parallax_deg=2, maximum_reprojection_px=2),
                      region=region, software=dict(opencv=cv2.__version__, numpy=np.__version__),
                      method='First RGB frame per stop; fitted RGB intrinsics; recorded base_link_T_color; descriptor-only tracks; largest-baseline pair fit; other views checked without refitting.',
                      limitations=['Sparse features are not dense surface coverage.', 'FEMTO masks label tracks only; may include nearby table/edges.',
                                   'Accepted residuals are selection statistics; ungated residuals and every rejection are retained.',
                                   'Unused-view pixel checks do not verify metric accuracy or common camera/calibration bias.',
                                   'FEMTO reference and sparse NN distances include visibility and sampling error; no independent ground truth.',
                                   'Wrist depth, ICP corrections and FEMTO coordinates do not participate in point estimation.'])
        atomic_json(output/'summary.json', report)
        return report
    except BaseException as error:
        report.update(status='failed', error=str(error)); atomic_json(output/'summary.json', report)
        raise
