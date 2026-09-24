"""Compare recorded RGB/FEMTO stone regions to measured catalog PCDs."""
from pathlib import Path
import inspect
import shutil
import numpy as np
from stacking_common.pointcloud import PointCloudData
from stacking_common.pointcloud.processing import fixed_grid_centroids
from stacking_common.pointcloud.catalog_comparison import fit_catalog_candidate, assignment_ranking, observed_to_model, distance_summary
from stacking_common.rendering.pointcloud import render_clouds
from .hand_eye_scene import inspect_scene, load_npz
from .hand_eye_capture import atomic_json, digest
from ..perception.catalog import TabletopPcdCatalog


def compare_scene_catalog(scene, models, stone_ids, output):
    scene, models, output = [Path(p).resolve() for p in (scene, models, output)]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (scene, models)):
        raise ValueError('catalog experiment output must be separate from source scene and catalog')
    manifest = inspect_scene(scene)
    catalog = TabletopPcdCatalog(models, stone_ids)
    if len(stone_ids) != len(manifest['region']['stone_boxes_xy']) or len(stone_ids) > 6:
        raise ValueError('require one catalog ID per region, at most six regions')
    rgb = load_npz(scene/'sources/rgb_observations.npz')
    femto = load_npz(scene/'sources/femto_before.npz')
    bracket = load_npz(scene/'sources/femto_after.npz')
    clouds = {}
    for i in range(len(stone_ids)):
        for name, data, mask in [('rgb', rgb, rgb['keep'] & (rgb['stone'] == i)),
                                 ('femto', femto, femto['group_ids'] == i)]:
            clouds[name, i] = fixed_grid_centroids(PointCloudData(data['points'][mask]), .0015).points
    output.mkdir(parents=True, exist_ok=False)
    (output/'sources').mkdir()
    report = dict(schema_version=1, kind='catalog_surface_comparison', status='running',
                  scene_sha256=digest(scene/'scene.json'), data_origin=manifest['data_origin'],
                  output_frame='base_link', stone_ids=list(stone_ids), region=manifest['region'],
                  accuracy_1mm_verified=False, identity_verified=False, production_pose_accepted=False,
                  source_sha256={'adapter': digest(__file__), 'registration': digest(inspect.getfile(fit_catalog_candidate))},
                  model_hashes={}, candidates=[],
                  settings=dict(observation_voxel_m=.0015, spatial_split_cell_m=.004, rigid_only=True,
                                orientation_starts=144, score='mean of training median and p90 point-to-catalog distance, mm'))
    atomic_json(output/'summary.json', report)
    try:
        references = {i: catalog.load(i).points for i in stone_ids}
        for i in stone_ids:
            shutil.copyfile(catalog.path(i), output/'sources'/f'model_{i}.pcd')
            report['model_hashes'][str(i)] = digest(catalog.path(i))
        fitted = {}
        for region in range(len(stone_ids)):
            for sensor in ('femto', 'rgb'):
                points = clouds[sensor, region]
                np.save(output/'sources'/f'{sensor}_region_{region}.npy', points, allow_pickle=False)
                for stone in stone_ids:
                    print(f'Fitting region {region}, {sensor}, catalog stone {stone}', flush=True)
                    result, train, distances = fit_catalog_candidate(points, references[stone])
                    result.update(region=region, sensor=sensor, stone_id=stone)
                    report['candidates'].append(result)
                    fitted[sensor, region, stone] = result
                    np.savez_compressed(output/f'{sensor}_region_{region}_stone_{stone}.npz',
                                        train=train, distance_m=distances)
                    atomic_json(output/'summary.json', report)
        costs = {sensor: np.array([[fitted[sensor, region, stone]['score_mm'] for stone in stone_ids]
                                  for region in range(len(stone_ids))]) for sensor in ('femto', 'rgb')}
        report['assignments'] = {sensor: assignment_ranking(matrix, stone_ids) for sensor, matrix in costs.items()}
        report['assignments']['joint'] = assignment_ranking((costs['femto']+costs['rgb'])/2, stone_ids)
        report['training_cost_matrices_mm'] = {k: v.tolist() for k, v in costs.items()}
        assignment = report['assignments']['joint'][0]['stone_ids']
        report['selected'] = []
        for region, stone in enumerate(assignment):
            selected = dict(region=region, stone_id=stone)
            layers = {f'Catalog stone {stone} (full surface)': references[stone]}
            for sensor in ('femto', 'rgb'):
                fit = fitted[sensor, region, stone]
                t = np.asarray(fit['base_T_model'])
                points = clouds[sensor, region]
                local = (points-t[:3, 3]) @ t[:3, :3]
                layers[f'{sensor.upper()} independently aligned'] = local
                selected[sensor] = fit
            # Frozen FEMTO pose exposes the cross-sensor alignment mismatch separately.
            femto_pose = selected['femto']['base_T_model']
            selected['rgb_at_femto_pose'] = distance_summary(observed_to_model(clouds['rgb', region], references[stone], femto_pose))
            selected['femto_after_at_femto_pose'] = distance_summary(observed_to_model(
                bracket['points'][bracket['group_ids'] == region], references[stone], femto_pose))
            render_clouds(output/f'region_{region}', layers, {}, frame=f'catalog stone {stone}',
                          title=f'Region {region} → candidate stone {stone} · independent rigid fits · fixed scale')
            report['selected'].append(selected)
        report.update(status='complete', limitations=[
            'Catalog identity and poses are provisional: partial surface matching admits ambiguous or incorrect alignments.',
            'No scale fitting, shape warping, plane snapping or changes to the input scene/calibration.',
            'Observed-to-catalog distances do not penalize unseen catalog surfaces; no ray-based visibility proof is applied.',
            'Training patches alone select poses and one-to-one identity assignments; holdout is spatial interpolation, not independent sensor ground truth.',
            'RGB/FEMTO fits are independent, so their residuals compare local shape agreement after removing each fitted rigid pose.',
            'A smooth partial surface can fit an incorrect catalog patch. Alternative orientation basins and assignment scores are retained.',
            'Model PCDs are measured references with unverified error and potentially shared sensor bias, not certified ground truth.',
            'The 1.5mm evaluation sampling and catalog sampling contribute to nearest-point distances.'])
        report['files'] = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*'))
                           if p.is_file() and p.suffix in ('.npy', '.npz', '.pcd')}
        atomic_json(output/'summary.json', report)
        _write_overview(output, report)
        return report
    except BaseException as error:
        report.update(status='failed', error=str(error))
        atomic_json(output/'summary.json', report)
        raise


def _write_overview(output, report):
    rows = []
    for selected in report['selected']:
        for sensor in ('femto', 'rgb'):
            fit = selected[sensor]
            d = fit['holdout']
            rows.append(f'<tr><td>{selected["region"]}</td><td>{selected["stone_id"]}</td><td>{sensor.upper()}</td>'
                        f'<td>{d["median_mm"]:.2f}</td><td>{d["p95_mm"]:.2f}</td><td>{100*d["within_2mm"]:.1f}%</td></tr>')
    views = ''.join(f'<h2>Region {s["region"]} → candidate stone {s["stone_id"]}</h2>'
                    f'<p><a href="region_{s["region"]}/overview.html">Open full-size viewer</a></p>'
                    f'<iframe src="region_{s["region"]}/overview.html" title="Region {s["region"]}" loading="lazy"></iframe>'
                    for s in report['selected'])
    assignments = ''.join(f'<li>{name}: {ranking[0]["stone_ids"]}; total training score {ranking[0]["total_score_mm"]:.2f} mm, '
                          f'next assignment {ranking[1]["total_score_mm"]:.2f} mm</li>' for name, ranking in report['assignments'].items())
    (output/'overview.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Catalog comparison</title>'
        '<style>body{font:16px system-ui;max-width:1350px;margin:30px auto;padding:0 20px;color:#24334a}'
        'td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}iframe{width:100%;height:650px;border:1px solid #ddd}</style>'
        '</head><body><h1>RGB and FEMTO compared with stones 9, 10, 11</h1>'
        '<p>Independent rigid fits at fixed catalog scale. Provisional identities; accuracy is unverified. '
        'Each viewer uses the catalog stone coordinate frame, not the original scene frame.</p>'
        '<h2>Spatial holdout: observed-to-catalog distances</h2><table><tr><th>Region</th><th>Candidate stone</th><th>Sensor</th>'
        '<th>Median (mm)</th><th>p95 (mm)</th><th>Within 2 mm</th></tr>'+''.join(rows)+'</table>'
        '<h2>One-to-one assignment alternatives</h2><ul>'+assignments+'</ul>'
        '<p>Unseen catalog surfaces are shown for context and are not counted as capture errors. '
        'Holdout patches share the same captures and calibration; they do not establish absolute accuracy. '
        'Full scores, alternate pose basins and frozen-FEMTO-pose checks are in <a href="summary.json">summary.json</a>.</p>'
        +views+'</body></html>')
