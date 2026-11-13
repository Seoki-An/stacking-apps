"""Region score inspection matches production grids and keeps target gaps masked."""

from dataclasses import asdict
import os
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from stacking_common.domain import CompositeTargetGeometry, TargetComponent
from stacking_common.visualization.heightmap import HeightMapRenderer
from stacking_common.visualization.sampling import initial_pose_score_view
from test_composite_target_sampling import box, inventory, pose, sampler, state, target


def score_view(inv, current, sample, initial=None):
    initial = pose(-3., z=.5) if initial is None else initial
    return initial_pose_score_view(inv, current, inv[0].object_id, initial,
        planar_config=asdict(sample.config), settings=sample.settings,
        height_map_resolution=(64, 64), height_map_margin=.2)


@pytest.mark.parametrize('kind', ['separate', 'fallback', 'l_shape', 'empty_mask'])
@pytest.mark.parametrize('soft', [False, True])
def test_region_scores_masks_terms_and_grids_match_production(kind, soft):
    geometry = (target(1000.) if kind == 'fallback' else
        CompositeTargetGeometry((TargetComponent('a', box(4., 1.)),
                                 TargetComponent('b', box(1., 4.), pose(-1.5, 1.5))))
        if kind == 'l_shape' else target())
    inv = inventory(geometry)
    current = state(inv)
    if soft:
        current.placed_object_indices = [1]
        current.object_poses[inv[1].object_id] = pose(-3., -.5, .125)
    sample = sampler(inv, yaw_angles=(0.,))
    sample.settings['connectivity'] = 'previous_action_soft' if soft else 'active_floor'
    sample.settings['score']['previous_action_distance'] = 2.5
    if kind == 'empty_mask':
        sample.settings['target_mask_margin'] = 100.
    production = sample.score_map(current, None, 0)
    before = sample.rng.get_state()
    view = score_view(inv, current, sample)
    after = sample.rng.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert len(view.regions) == (1 if kind == 'l_shape' else 2)
    assert len(view.target_outlines) == 2
    for region in view.regions:
        data = production['regions'][region.region_id]
        np.testing.assert_array_equal(region.x, data['x_coords'])
        np.testing.assert_array_equal(region.y, data['y_coords'])
        np.testing.assert_array_equal(region.valid, data['candidate_mask'])
        np.testing.assert_allclose(region.fields['score'], data['scores'], atol=1e-12)
        total = sum(region.fields[name] * weight for name, weight in region.weights.items())
        np.testing.assert_allclose(total[region.valid], data['scores'][region.valid], atol=1e-12)
        assert region.grid_source == data['grid_source']
        if kind == 'fallback':
            assert region.grid_source == 'region_fallback'
        if kind == 'empty_mask':
            assert not region.valid.any()
        xx, yy = np.meshgrid(region.x, region.y)
        matching = next(r for r in sample.floor_policy.target_regions.regions
                        if r.region_id == region.region_id)
        assert not (region.valid & ~matching.mask(xx, yy, 0.)).any()
    # Enabling term diagnostics never adds per-field grids to the search cache.
    assert all('terms' not in value for value in production['regions'].values())


def test_default_region_tracks_recorded_initial_xy_and_scene_height():
    inv = inventory()
    current = state(inv)
    current.placed_object_indices = [1]
    current.object_poses[inv[1].object_id] = pose(-3., z=.125)
    sample = sampler(inv, yaw_angles=(0.,))
    west = score_view(inv, current, sample, pose(-3., z=.5))
    east = score_view(inv, current, sample, pose(3., z=.5))
    assert west.region_id == ('west',) and east.region_id == ('east',)
    renderer = HeightMapRenderer(inv, (64, 64), .2)
    np.testing.assert_array_equal(east.scene_height, renderer.render(current))


def test_region_selector_keeps_native_grids_and_draws_component_outlines(monkeypatch):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('PySide6')
    pytest.importorskip('pyvistaqt')
    from PySide6.QtWidgets import QApplication, QWidget
    from stacking_excavator.gui.operations import score_view as gui
    app = QApplication.instance() or QApplication([])
    inv = inventory(target(1000.))
    data = score_view(inv, state(inv), sampler(inv, yaw_angles=(0.,)))
    plot = Mock(interactor=QWidget())
    monkeypatch.setattr(gui, 'QtInteractor', lambda *a, **kw: plot)
    dialog = gui.InitialPoseScoreDialog(data, step=1,
        candidate=SimpleNamespace(object_id=inv[0].object_id, rank=1))
    try:
        assert dialog.region.count() == 2
        dialog.region.setCurrentIndex(1 - dialog.region.currentIndex())
        assert dialog.data is data.regions[dialog.region.currentIndex()]
        assert 'local fallback grid' in dialog.summary.text()
        assert len([call for call in plot.add_mesh.call_args_list
                    if call.kwargs.get('color') == '#f4d35e']) == 8
        dialog.field.setCurrentIndex(1)
        dialog.weighted.setChecked(True)
        assert 'higher is better' in dialog.summary.text()
    finally:
        dialog.close()
