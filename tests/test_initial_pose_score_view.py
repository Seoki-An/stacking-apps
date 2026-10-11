"""Score inspection uses the restored production scorer and recorded context."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stacking_common.actions import MappingOrientationProvider, PlanarSamplingConfig
from stacking_common.actions.floor import ActiveFloorPolicy, floor_policy_settings
from stacking_common.actions.scored import ScoredPlanarPoseSampler
from stacking_common.domain import ConvexGeometry, DirectionalTaper, ObjectInventory, StackObject, State, TargetGeometry
from stacking_common.visualization.heightmap import HeightMapRenderer
from stacking_common.visualization.sampling import initial_pose_score_view
from test_action_generation import _BodyModel, _cube, _pose


def scene():
    cube = _cube(.1)
    slab = ConvexGeometry(cube.points * [2., 1., 1.], cube.faces)
    objects = [StackObject(10, _BodyModel(), [slab]),
               StackObject(20, _BodyModel(), [_cube(.02)])]
    inventory = ObjectInventory(objects, {o.object_id: _pose(x=1.) for o in objects},
        TargetGeometry(.2, .2, .3, np.zeros(3), DirectionalTaper(), 0.))
    state = State(list(inventory.object_ids), [0], {10: _pose(y=-.05, z=.05)}, [], [])
    return inventory, state


@pytest.fixture
def config():
    return dict(
        planar_config=asdict(PlanarSamplingConfig((16, 20), (0.,), 0., .6)),
        settings=dict(boltzmann_temperature=.3, boltzmann_min_probability=0.,
            target_mask_margin=0., contact_gap_tolerance=.01,
            score=dict(height=-3., support_likelihood=2., open_area=1., fill_area=2.,
                frontier=1., target_boundary=1.5, planar_bias=2.,
                planar_bias_axis='xy', planar_bias_xy=None, elite_fraction=.25),
            floor=dict(connected_after=1, footprint_scale=.9,
                minimum_contact_cells=1, fill_ratio=.75, grid_size=16,
                bottom_tolerance=.02, max_layers=8, coverage_mode='projected_footprint')),
        height_map_resolution=(64, 64), height_map_margin=.2)


@pytest.mark.parametrize('placed', [False, True])
def test_view_matches_fixed_orientation_production_maps_and_weighted_sum(config, placed):
    inventory, state = scene()
    if not placed:
        state = State(list(inventory.object_ids), [], {}, [], [])
    quaternion = Rotation.from_euler('xyz', [.2, -.1, .3]).as_quat()
    pose = np.r_[.01, .04, .3, quaternion]
    view = initial_pose_score_view(inventory, state, 20, pose, **config)
    sampler = ScoredPlanarPoseSampler(inventory, PlanarSamplingConfig(**config['planar_config']),
        MappingOrientationProvider({oid: quaternion for oid in inventory.object_ids}),
        config['settings'], floor_policy=ActiveFloorPolicy(inventory, **floor_policy_settings(config['settings']['floor'])))
    renderer = HeightMapRenderer(inventory, config['height_map_resolution'], config['height_map_margin'])
    observation = SimpleNamespace(scene_height_map=renderer.render(state),
        height_map_xlim=renderer.xlim, height_map_ylim=renderer.ylim)
    data = sampler.score_map(state, observation, 1)
    assert view.valid.any()
    for name, values in view.fields.items():
        key = {'score': 'scores', 'height': 'height_term'}.get(name, name)
        np.testing.assert_array_equal(values, data[key])
    np.testing.assert_array_equal(view.valid, data['candidate_mask'])
    np.testing.assert_allclose(sum(view.fields[k] * w for k, w in view.weights.items()), view.fields['score'])
    np.testing.assert_array_equal(view.scene_height, observation.scene_height_map)
    np.testing.assert_array_equal(view.initial_xy, pose[:2])
    # Inspection must not advance a production sampler's RNG.
    before = sampler.rng.get_state()
    initial_pose_score_view(inventory, state, 20, pose, **config)
    after = sampler.rng.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_view_handles_no_eligible_cells_and_rejects_placed_object(config):
    inventory, state = scene()
    config['settings']['target_mask_margin'] = 10.
    view = initial_pose_score_view(inventory, state, 20, _pose(), **config)
    assert not view.valid.any()
    assert all(np.isnan(values).all() for values in view.fields.values())
    with pytest.raises(ValueError, match='already in the scene'):
        initial_pose_score_view(inventory, state, 10, state.object_poses[10], **config)


def test_controller_uses_recorded_settings_pre_search_scene_and_caches(tmp_path, config):
    from stacking_common.visualization import (
        MCTSCandidateDiagnostic, MCTSDebugArtifact, MCTSDebugStep, SceneObjectPose, SceneSnapshot,
    )
    from stacking_excavator.gui.operations.controller import OperationsController

    inventory, state = scene()
    pose = _pose(y=.05, z=.3)
    candidate = MCTSCandidateDiagnostic('seed', 1, 1, 20, 'succeeded', True,
                                       pose, pose, pose, final_pose=pose)
    step = MCTSDebugStep(2, SceneSnapshot('stack', (
        SceneObjectPose(10, state.object_poses[10], object_index=0),)), (candidate,), 'seed', True)
    controller = OperationsController(tmp_path)  # No current YAML files available.
    controller.set_artifact(MCTSDebugArtifact('stack', (step,)))
    controller.contexts = {'recorded': SimpleNamespace(profiles={
        'sampling_settings': config['settings'],
        'action_generation_profile': {'planar': config['planar_config']},
        'environment_profile': {'observation': {'height_map_resolution': config['height_map_resolution']},
                                'height_map_margin': config['height_map_margin']},
    })}
    controller._score_inventory = ('recorded', inventory)
    view = controller.initial_pose_score_view()
    assert controller.initial_pose_score_view() is view
    assert view.valid.any()
    for settings in ({}, {'strategy': 'support_score_regions_v11', 'scoring': {}}):
        controller.contexts['recorded'].profiles['sampling_settings'] = settings
        with pytest.raises(ValueError, match='does not record'):
            controller.initial_pose_score_view()


@pytest.mark.parametrize('u_shape_filter', [False, True])
def test_score_view_reads_historical_split_floor_settings(config, u_shape_filter):
    inventory, state = scene()
    floor = config['settings']['floor']
    legacy = {key: value for key, value in config['settings'].items() if key != 'floor'}
    legacy['floor_fill'] = dict(connected_after=floor['connected_after'],
        u_shape_filter=u_shape_filter, u_shape_footprint_scale=floor['footprint_scale'],
        u_shape_min_frontier_contact_cells=floor['minimum_contact_cells'],
        lower_floor_fill_ratio=.6, lower_floor_occupancy_grid=32,
        lower_floor_bottom_z_tolerance=.2, lower_floor_max_layers=8)
    pose = _pose(y=.05, z=.3)
    current = initial_pose_score_view(inventory, state, 20, pose, **config)
    if u_shape_filter:
        with pytest.raises(ValueError, match='removed U-shape filter'):
            initial_pose_score_view(inventory, state, 20, pose,
                **{**config, 'settings': legacy, 'floor_settings': floor_policy_settings(floor)})
        return
    previous = initial_pose_score_view(inventory, state, 20, pose,
        **{**config, 'settings': legacy, 'floor_settings': floor_policy_settings(floor)})
    np.testing.assert_array_equal(current.valid, previous.valid)
    for name in current.fields:
        np.testing.assert_array_equal(current.fields[name], previous.fields[name])
