"""Offline upper-surface shape check from a verified layered capture scene."""
from pathlib import Path
import inspect
import numpy as np
from stacking_common.pointcloud.surface_comparison import surface_grid, fit_shared_rigid, compare_region
from stacking_common.rendering.surface_comparison import render_surface_comparison
from .hand_eye_scene import inspect_scene, load_npz
from .hand_eye_capture import atomic_json, digest


def check_scene_surfaces(scene, output):
    scene, output = Path(scene).resolve(), Path(output).resolve()
    if output.is_relative_to(scene) or scene.is_relative_to(output):
        raise ValueError('surface-check output must be separate from scene input')
    manifest = inspect_scene(scene)
    rgb = load_npz(scene/'sources/rgb_observations.npz')
    before = load_npz(scene/'sources/femto_before.npz')
    after = load_npz(scene/'sources/femto_after.npz')
    boxes = manifest['region']['stone_boxes_xy']
    references = [before['points'][before['group_ids'] == i] for i in range(len(boxes))]
    checks = [after['points'][after['group_ids'] == i] for i in range(len(boxes))]
    candidates = [rgb['points'][rgb['keep'] & (rgb['stone'] == i)] for i in range(len(boxes))]
    if len(boxes) < 3:
        raise ValueError('surface check requires at least three regions for held-out-region checks')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, kind='stone_surface_comparison', status='running', output_frame='base_link',
                  data_origin=manifest['data_origin'], scene_sha256=digest(scene/'scene.json'),
                  source_sha256={'adapter': digest(__file__), 'numeric': digest(inspect.getfile(compare_region))},
                  accuracy_1mm_verified=False, calibration_update_accepted=False,
                  settings=dict(cell_size_m=.003, minimum_samples_per_cell=2, interior_erosion_cells=1,
                                correspondence_limit_m=.015, robust_scale_m=.001,
                                rotation_component_bound_deg=3, pivot_translation_component_bound_m=.015))
    atomic_json(output/'summary.json', report)
    try:
        ref_grids = [surface_grid(p, b) for p, b in zip(references, boxes)]
        rgb_grids = [surface_grid(p, b) for p, b in zip(candidates, boxes)]
        fit = fit_shared_rigid(ref_grids, rgb_grids)
        report['shared_fit'] = fit
        report['regions'] = []
        arrays = []
        for i, box in enumerate(boxes):
            metrics, data = compare_region(references[i], candidates[i], checks[i], box, fit['transform'])
            report['regions'].append(dict(region=i, **metrics))
            np.savez_compressed(output/f'region_{i:03d}.npz', **data)
            arrays.append(data)
        report['held_out_regions'] = []
        for held in range(len(boxes)):
            train = [i for i in range(len(boxes)) if i != held]
            fold_fit = fit_shared_rigid([ref_grids[i] for i in train], [rgb_grids[i] for i in train])
            metrics, data = compare_region(references[held], candidates[held], checks[held], boxes[held], fold_fit['transform'])
            np.savez_compressed(output/f'held_out_{held:03d}.npz', **data)
            report['held_out_regions'].append(dict(region=held, training_regions=train, fit=fold_fit, **metrics))
        render_surface_comparison(output, arrays)
        report.update(status='complete', limitations=[
            'The all-region shared fit uses the evaluated surfaces and is an optimistic diagnostic, not independent validation.',
            'Held-out-region fits use the other regions only; they share sensor calibration and are not ground truth.',
            'Each before/after comparison uses identical common interior cells; different folds may retain different cohorts.',
            'Median heights in 3mm cells approximate upper surfaces; lateral sampling and residual alignment affect height differences.',
            'One-cell erosion reduces edge contamination but does not certify physical interior or visibility.',
            'FEMTO after is used only for bracket comparison, not alignment. Repeatability does not establish accuracy.',
            'Residual disagreement does not identify which sensor is wrong or prove nonrigid sensor distortion.',
            'No transform is applied to the source scene, camera calibration, or production fusion.'])
        report['files'] = {p.name: digest(p) for p in sorted(output.glob('*.npz'))}
        atomic_json(output/'summary.json', report)
        return report
    except BaseException as error:
        report.update(status='failed', error=str(error))
        atomic_json(output/'summary.json', report)
        raise
