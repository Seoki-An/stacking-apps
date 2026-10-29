"""Floor tracking guides sampling without solved-pose rejection."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from stacking_common.actions import PosegenFeasibilityFilter, TargetContainmentFilter
from stacking_common.environment import VoxelOverlapEvaluator
from stacking_tabletop import TabletopApplication
from stacking_tabletop.config.action import load_action_generation_profile
import stacking_tabletop.composition as composition


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('value', [False, True])
def test_old_floor_flag_cannot_enable_rejection(tmp_path, value):
    values = yaml.safe_load((ROOT / 'apps/tabletop/config/action_generation.yml').read_text())
    assert 'strict_floor_first' not in values['filtering']
    values['filtering']['strict_floor_first'] = value
    path = tmp_path / 'action.yml'
    path.write_text(yaml.safe_dump(values))
    profile = load_action_generation_profile(path, max_poses_per_object=32, banned_object_ids=())
    assert not hasattr(profile, 'strict_floor_first')



def test_floor_tracking_is_sampling_only(monkeypatch):
    app = TabletopApplication.from_repository(ROOT)
    inventory = app.load_inventory([1], {1: np.array([0., 0., 0., 0., 0., 0., 1.])})
    containment = TargetContainmentFilter(VoxelOverlapEvaluator(inventory.target, .003), .5)
    generator = app.build_action_generator(inventory, SimpleNamespace(), filters=(containment,))
    assert len(generator.filters) == 2
    assert isinstance(generator.filters[0], PosegenFeasibilityFilter)
    assert generator.filters[1] is containment
    assert generator.planar_sampler.floor_policy is not None
    assert not hasattr(generator, 'floor_fill_fallback')
    generator.close()
    backend = SimpleNamespace(reset=lambda ids: None)
    monkeypatch.setattr(composition, 'CorePlacementBackend', SimpleNamespace(from_profile=lambda *a, **k: backend))
    monkeypatch.setattr(composition, 'TabletopNativePlanningBackend', lambda *a, **k: SimpleNamespace(solve=lambda problem: None))
    coordinator = app.build_planning_coordinator(inventory)
    assert not hasattr(coordinator, 'safety')
    assert coordinator.placement_backend is backend



@pytest.mark.parametrize('platform', ['excavator', 'tabletop'])
def test_unified_floor_profile_drives_sampling(tmp_path, platform):
    import shutil
    from stacking_excavator import ExcavatorApplication

    Application = ExcavatorApplication if platform == 'excavator' else TabletopApplication
    original = Application.from_repository(ROOT)
    inventory = original.load_inventory([1], {1: np.array([0., 0., 0., 0., 0., 0., 1.])})
    config_dir = tmp_path / 'apps' / platform / 'config'
    shutil.copytree(ROOT / 'apps' / platform / 'config', config_dir)
    path = config_dir / 'action_generation.yml'
    values = yaml.safe_load(path.read_text())
    assert 'floor_fill' not in values['planar']
    floor = values['planar']['floor']
    floor.update(bottom_tolerance=.04, fill_ratio=.8, grid_size=20,
                 footprint_scale=.85)
    path.write_text(yaml.safe_dump(values))
    app = Application.from_repository(tmp_path)
    generator = app.build_action_generator(inventory, SimpleNamespace())
    try:
        sampler = generator.planar_sampler
        policy = sampler.floor_policy
        assert policy.bottom_tolerance == floor['bottom_tolerance']
        assert policy.fill_ratio == floor['fill_ratio']
        assert policy.grid_size == floor['grid_size']
        assert 'minimum_contact_cells' not in floor
        assert sampler.settings['floor'] == floor
        assert 'floor_fill' not in sampler.settings
    finally:
        generator.close()


@pytest.mark.parametrize('old_key', ['floor_fill', 'lower_floor_fill_ratio',
                                     'connected_after', 'minimum_contact_cells', 'max_layers'])
def test_obsolete_floor_configuration_is_rejected(old_key):
    from stacking_common.profiles.action import resolve_scored_sampling_settings
    planar = yaml.safe_load((ROOT / 'apps/excavator/config/action_generation.yml').read_text())['planar']
    if old_key == 'floor_fill':
        planar['floor_fill'] = {'lower_floor_fill_ratio': .6}
    else:
        planar['floor'][old_key] = 1
    with pytest.raises(ValueError, match='floor'):
        resolve_scored_sampling_settings(planar)


@pytest.mark.parametrize('platform', ['excavator', 'tabletop'])
@pytest.mark.parametrize('removed', ['open_area', 'frontier'])
def test_removed_score_settings_are_rejected(platform, removed):
    from stacking_common.profiles.action import resolve_scored_sampling_settings
    planar = yaml.safe_load((ROOT / 'apps' / platform / 'config/action_generation.yml').read_text())['planar']
    assert removed not in planar['score']
    planar['score'][removed] = 0.
    with pytest.raises(ValueError, match='obsolete planar.score'):
        resolve_scored_sampling_settings(planar)


@pytest.mark.parametrize('complete_layers, next_layer', [
    (0, 'empty'), (9, 'empty'), (9, 'partial'), (0, 'gap'), (9, 'gap'),
])
@pytest.mark.parametrize('coverage_mode', ['bottom_slice', 'projected_footprint'])
def test_floor_detection_stops_at_first_incomplete_layer_without_cap(complete_layers, next_layer, coverage_mode):
    from stacking_common.actions.floor import ActiveFloorPolicy
    from stacking_common.actions import MappingOrientationProvider, PlanarSamplingConfig
    from stacking_common.actions.scored import ScoredPlanarPoseSampler
    from stacking_common.actions._scored_floor import active_floor_context
    from stacking_common.domain import ObjectInventory, StackObject, TargetGeometry, DirectionalTaper, State
    from test_action_generation import _BodyModel, _cube, _pose

    full = _cube(.2)
    objects = [StackObject(i, _BodyModel(), [full]) for i in range(complete_layers)]
    poses = {i: _pose(z=.2 * i + .1) for i in range(complete_layers)}
    expected_z = .2 * complete_layers
    if next_layer != 'empty':
        size = .1 if next_layer == 'partial' else .2
        objects.append(StackObject(complete_layers, _BodyModel(), [_cube(size)]))
        poses[complete_layers] = _pose(z=expected_z + size / 2 + (1. if next_layer == 'gap' else 0.))
    # Include an unplaced object so the empty-scene case has an inventory.
    objects.append(StackObject(len(objects), _BodyModel(), [full]))
    inventory = ObjectInventory(objects, {o.object_id: _pose() for o in objects},
        TargetGeometry(.2, .2, 4., np.zeros(3), DirectionalTaper(), 0.))
    state = State(list(inventory.object_ids), list(poses), poses, [], [])
    floor = dict(bottom_tolerance=.01, fill_ratio=.9, grid_size=16, coverage_mode=coverage_mode)
    policy = ActiveFloorPolicy(inventory, **floor)
    context = policy.context_details(state)
    assert len(context['completed_layers']) == complete_layers
    assert context['support_z'] == pytest.approx(expected_z)
    if next_layer == 'partial':
        assert 0. < context['occupancy'] < floor['fill_ratio']
    else:
        assert not context['occupied'].any()

    # Standalone sampler detection must also advance beyond eight layers and
    # stop on an empty layer even if unsupported stones remain above it.
    sampler = ScoredPlanarPoseSampler(inventory, PlanarSamplingConfig((16, 16), (0.,)),
        MappingOrientationProvider({o.object_id: _pose()[3:] for o in objects}),
        {'floor': {**floor, 'footprint_scale': .9}, 'contact_gap_tolerance': .01})
    observation = SimpleNamespace(height_map_xlim=(-.1, .1), height_map_ylim=(-.1, .1))
    standalone = active_floor_context(sampler.view(state, observation), sampler.state_view(state))
    assert standalone['support_z'] == pytest.approx(expected_z)
    np.testing.assert_array_equal(standalone['occupied'], context['occupied'])
