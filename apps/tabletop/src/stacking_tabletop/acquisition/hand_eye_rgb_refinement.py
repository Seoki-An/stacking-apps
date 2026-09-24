"""Offline camera-prior refinement; held-out image and stone-surface checks."""

from pathlib import Path
import inspect

import numpy as np
from scipy.spatial import cKDTree

from stacking_common.multiview.geometry import triangulate_track
from stacking_common.multiview.refinement import refine_cameras, project_normalized, align_support_plane, PosePrior
from stacking_common.rendering.pointcloud import render_clouds
from .hand_eye import HandEyeCalibration
from .hand_eye_capture import atomic_json, digest
from .hand_eye_replay import inspect_session, inside, read_json
from .hand_eye_rgb import statistics
from .storage import write_xyz_pcd


def _triangulate(row, features, poses, focal, excluded=None):
    observations = [(v, f) for v, f in row['observations'] if v != excluded]
    if len(observations) < 3:
        return None
    views = np.array([v for v, _ in observations])
    xy = np.array([features[v][f] for v, f in observations])
    try:
        result = triangulate_track(xy, poses[views])
    except ValueError:
        return None
    errors = np.linalg.norm(result['residuals']*focal, axis=1)
    fit = result['fit_pair']; held = np.ones(len(views), bool); held[fit] = False
    return dict(point=result['point'], observations=observations,
                views=views, xy=xy, errors=errors, unused_errors=errors[held],
                valid=bool(np.all(result['depths'] > 0) and result['parallax_deg'] >= 2 and errors.max() <= 2))


def _training(rows, features, poses, focal, region, excluded):
    """Select from training images alone; no unused-view residual gate."""
    low, high = np.array(region['minimum']), np.array(region['maximum'])
    boxes = np.array(region['stone_boxes_xy'])
    selected = []
    for row in rows:
        if row['stone'] >= 0:
            continue
        t = _triangulate(row, features, poses, focal, excluded)
        if t is None or not t['valid']:
            continue
        p = t['point']
        if not np.all((p >= low) & (p <= high)):
            continue
        if any(np.all((p[:2] >= lo-.015) & (p[:2] <= hi+.015)) for lo, hi in boxes):
            continue
        selected.append((row['track'], t))
    return selected


def _fit(selected, poses, focal, excluded, prior):
    active = [v for v in range(len(poses)) if v != excluded]
    points = np.array([t['point'] for _, t in selected])
    views, ids, xy = [], [], []
    for index, (_, t) in enumerate(selected):
        for v, obs in zip(t['views'], t['xy']):
            views.append(active.index(v)); ids.append(index); xy.append(obs)
    result = refine_cameras(poses[active], points, views, ids, xy, focal, prior=prior)
    updated = poses.copy(); updated[active] = result['poses']
    details = dict(success=result['success'], message=result['message'], evaluations=result['evaluations'],
                   optimality=result['optimality'], training_tracks=len(selected), anchor_view=active[0],
                   prior=result['prior'], camera_bound_hits=result['camera_bound_hits'],
                   point_bound_hits=result['point_bound_hits'],
                   fit_before_px=statistics(result['initial_pixel_errors']), fit_after_px=statistics(result['final_pixel_errors']),
                   corrections=[dict(view=v, local_rotation_vector_deg=np.degrees(delta[:3]).tolist(),
                                     translation_world_mm=(delta[3:]*1000).tolist())
                                for v, delta in zip(active, result['corrections'])],
                   base_link_T_color=updated.tolist(), training_track_ids=[i for i, _ in selected])
    return updated, details


def refine_rgb_session(session, baseline, output):
    """Run background-only BA, then five unused-camera checks for a five-view scan."""
    import open3d as o3d
    session, baseline, output = map(lambda p: Path(p).resolve(), (session, baseline, output))
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (session, baseline)):
        raise ValueError('refinement output must be separate from session and baseline')
    manifest = inspect_session(session)
    base = read_json(baseline/'summary.json')
    if (base.get('status') != 'complete' or base.get('kind') != 'fixed_pose_rgb_feasibility'
            or base.get('session_sha256') != digest(session/'session.json')):
        raise ValueError('RGB baseline is incomplete or belongs to another session')
    if len(manifest['views']) < 4:
        raise ValueError('unused-view refinement requires at least four views')
    poses, features = [], []
    for index, view in enumerate(manifest['views']):
        root = inside(session, view['path'])
        source = inside(session, base['views'][index]['source'])
        if not source.is_relative_to(root):
            raise ValueError('baseline source belongs to a different view')
        poses.append(read_json(str(source)+'.json')['base_link_T_color'])
        with np.load(baseline/f'features_{index:03d}.npz', allow_pickle=False) as arrays:
            features.append(arrays['normalized'])
    poses = np.array(poses)
    calibration = HandEyeCalibration.load(inside(session, manifest['views'][0]['path'])/'calibration/profile.yml')
    focal = np.diag(calibration.color_matrix)[:2]
    rows = read_json(baseline/'tracks.json')['tracks']
    region = base['region']; low, high = np.array(region['minimum']), np.array(region['maximum'])
    output.mkdir(parents=True, exist_ok=False)
    inputs = [baseline/'summary.json', baseline/'tracks.json'] + [baseline/f'features_{i:03d}.npz' for i in range(len(poses))]
    report = dict(status='running', kind='rgb_camera_prior_refinement', output_frame='base_link',
                  session_sha256=digest(session/'session.json'), baseline_hashes={p.name: digest(p) for p in inputs},
                  source_sha256={'hand_eye_rgb_refinement.py': digest(__file__), 'refinement.py': digest(inspect.getfile(refine_cameras)),
                                 'geometry.py': digest(inspect.getfile(triangulate_track))},
                  accuracy_1mm_verified=False, data_origin=manifest['data_origin'])
    atomic_json(output/'summary.json', report)
    try:
        prior = PosePrior()
        selected = _training(rows, features, poses, focal, region, None)
        print(f'Refining cameras from {len(selected)} background tracks', flush=True)
        updated, details = _fit(selected, poses, focal, None, prior)
        report['all_views_fit'] = details
        atomic_json(output/'summary.json', report)
        # Same pre-refinement cohort for both candidates, with post-fit failures retained.
        comparisons = []
        for row in rows:
            before = _triangulate(row, features, poses, focal)
            if before is None or not before['valid'] or not np.all((before['point'] >= low)&(before['point'] <= high)):
                continue
            after = _triangulate(row, features, updated, focal)
            comparisons.append(dict(track=row['track'], stone=row['stone'],
                before=before['point'].tolist(), after=after['point'].tolist() if after is not None else None,
                after_valid=bool(after is not None and after['valid']),
                before_unused_px=before['unused_errors'].tolist(), after_unused_px=after['unused_errors'].tolist() if after is not None else []))
        reference_root = session/'femto_before'
        reference_path = inside(reference_root, read_json(reference_root/'field_scan.json')['fused_file'])
        reference = np.asarray(o3d.io.read_point_cloud(str(reference_path)).points)
        reference = reference[np.all((reference >= low)&(reference <= high), axis=1)]
        coeff = np.array(region['support_plane'])
        above = reference[:, 2] > reference[:, :2]@coeff[:2]+coeff[2]+.008
        groups = []; layers = {'FEMTO stone surfaces': np.empty((0, 3))}
        targets = []
        for stone, (lo, hi) in enumerate(region['stone_boxes_xy']):
            target = reference[above & np.all((reference[:, :2] >= lo)&(reference[:, :2] <= hi), axis=1)]
            targets.append(target)
            cohort = [r for r in comparisons if r['stone'] == stone]
            a = np.array([r['before'] for r in cohort]).reshape(-1, 3)
            b = np.array([r['after'] for r in cohort if r['after'] is not None]).reshape(-1, 3)
            tree = cKDTree(target)
            groups.append(dict(stone=stone,cohort_count=len(cohort),after_failed_geometry=sum(not r['after_valid'] for r in cohort),
                               before_femto_mm=statistics(tree.query(a)[0]*1000), after_femto_mm=statistics(tree.query(b)[0]*1000),
                               before_unused_px=statistics([e for r in cohort for e in r['before_unused_px']]),
                               after_unused_px=statistics([e for r in cohort for e in r['after_unused_px']])))
            layers[f'stone {stone} fixed poses']=a; layers[f'stone {stone} refined poses (candidate)']=b
        layers['FEMTO stone surfaces']=np.concatenate(targets)
        # Separate diagnostic: only background support determines a rigid
        # reference adjustment. The same stone cohort remains untouched by fit.
        table_anchors = {}
        training_ids = set(details['training_track_ids'])
        for mode in ('before', 'after'):
            background = np.array([r[mode] for r in comparisons
                                   if r['track'] in training_ids and r[mode] is not None])
            anchor = align_support_plane(background, region['support_plane'])
            transform = anchor.pop('transform')
            anchor['transform'] = transform.tolist()
            anchor['stones'] = []
            for stone, target in enumerate(targets):
                points = np.array([r[mode] for r in comparisons if r['stone'] == stone and r[mode] is not None]).reshape(-1, 3)
                aligned = points@transform[:3, :3].T+transform[:3, 3]
                anchor['stones'].append(dict(stone=stone, count=len(points),
                                            femto_mm=statistics(cKDTree(target).query(aligned)[0]*1000)))
            table_anchors[mode] = anchor
        folds = []
        for excluded in range(len(poses)):
            selected = _training(rows, features, poses, focal, region, excluded)
            print(f'Unused camera {excluded}: {len(selected)} training tracks', flush=True)
            fold_poses, fold = _fit(selected, poses, focal, excluded, prior)
            validation = []
            # Stone tracks are never used to fit cameras. Fit their points from
            # remaining images, then project into the entirely excluded camera.
            for row in rows:
                unused = [(v, f) for v, f in row['observations'] if v == excluded]
                if row['stone'] < 0 or not unused:
                    continue
                before = _triangulate(row, features, poses, focal, excluded)
                if before is None or not before['valid']:
                    continue
                after = _triangulate(row, features, fold_poses, focal, excluded)
                target_xy = features[excluded][unused[0][1]]
                def error(t, cameras):
                    if t is None: return None
                    xy, depth = project_normalized(t['point'][None], cameras, [excluded], [0])
                    return dict(error_px=float(np.linalg.norm((xy[0]-target_xy)*focal)), positive_depth=bool(depth[0] > 0))
                validation.append(dict(track=row['track'], stone=row['stone'],before=error(before, poses),after=error(after,fold_poses),
                                       after_training_geometry_pass=bool(after is not None and after['valid'])))
            fold.update(excluded_view=excluded, validation_count=len(validation),
                        unused_view_before_px=statistics([r['before']['error_px'] for r in validation]),
                        unused_view_after_px=statistics([r['after']['error_px'] for r in validation if r['after'] is not None]),
                        failures_after=sum(r['after'] is None or not r['after']['positive_depth'] for r in validation))
            folds.append(fold)
            atomic_json(output/f'unused_view_{excluded:03d}.json', dict(fit=fold, validations=validation))
        report.update(status='complete', point_count=len(comparisons), groups=groups, unused_camera_folds=folds,
                      table_reference_diagnostic=table_anchors,
                      all_solvers_converged=bool(details['success'] and all(f['success'] for f in folds)),
                      limitations=['Pose priors are assumed (1mm translation, 0.2deg rotation sigma), not measured uncertainty.',
                      'Camera zero (first available in each fold) anchors the frame; all other pose components have 3-sigma bounds.',
                      'RGB intrinsics, distortion, wrist depth and FEMTO coordinates do not enter camera optimization.',
                      'The separate table-reference diagnostic uses the FEMTO support plane. It cannot observe in-plane translation or rotation about the normal and is not applied to saved RGB clouds.',
                      'FEMTO coarse masks/boxes exclude stones from background fitting; this is selection, not metric fitting.',
                      'Unused camera retains its recorded pose. Its error checks consistency with that prior, not independently known camera truth.',
                      'Descriptor association uses all images; held-out camera coordinates do not enter fitting or geometric training gates.',
                      'Both candidates use the same initial validation cohort; post-refinement geometric failures remain in reported distances.',
                      'Shared calibration bias and world-frame errors cannot be identified from image reprojection alone.',
                      'Candidate results are not installed calibration or accepted dense fusion.'])
        atomic_json(output/'stone_comparisons.json', dict(tracks=comparisons))
        for label in ('before', 'after'):
            p = np.array([r[label] for r in comparisons if r[label] is not None]).reshape(-1, 3)
            write_xyz_pcd(output/f'rgb_{label}.pcd', p)
        render_clouds(output/'comparison', layers, {}, title='RGB camera refinement: FEMTO / fixed / refined candidates')
        atomic_json(output/'summary.json', report)
        return report
    except BaseException as error:
        report.update(status='failed', error=str(error)); atomic_json(output/'summary.json', report)
        raise
