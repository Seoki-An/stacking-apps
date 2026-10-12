"""Rotation-aware proposal bounds agree with transformed geometry and grid masks."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stacking_common.actions import MappingOrientationProvider, PlanarSamplingConfig
from stacking_common.actions import _scored_floor as floor
from stacking_common.actions.scored import ScoredPlanarPoseSampler
from stacking_common.domain import ConvexGeometry, DirectionalTaper, ObjectInventory, StackObject, State, TargetGeometry
from test_action_generation import _BodyModel, _cube, _pose
from test_initial_pose_score_view import config


def make_scene(settings, *, offset=(0., 0., 0.), rotation=(0., 0., 0.), inset=1.):
    cube = _cube(1.)
    geometry = ConvexGeometry(cube.points * [.6, .18, .32] + offset, cube.faces)
    obj = StackObject(10, _BodyModel(), [geometry])
    inventory = ObjectInventory([obj], {10: _pose()},
        TargetGeometry(1., 1., 1., np.zeros(3), DirectionalTaper(), 0.))
    state = State([10], [], {}, [], [])
    observation = SimpleNamespace(scene_height_map=np.zeros((65, 65)),
        height_map_xlim=(-.8, .8), height_map_ylim=(-.8, .8))
    sampler = ScoredPlanarPoseSampler(inventory,
        PlanarSamplingConfig((33, 33), (0., np.pi / 2), target_anchor_inset_scale=inset),
        MappingOrientationProvider({10: Rotation.from_euler('xyz', rotation).as_quat()}), settings)
    return sampler, state, observation, geometry.points.copy()


@pytest.mark.parametrize('rotation', [(0., 0., 0.), (np.pi / 2, 0., 0.), (.4, -.3, .2)])
@pytest.mark.parametrize('offset', [(0., 0., 0.), (.12, -.06, .08)])
def test_target_masks_follow_rotated_vertices_and_origin(config, rotation, offset, monkeypatch):
    settings = config['settings']
    settings['target_mask_margin'] = .03
    sampler, state, obs, points = make_scene(settings, offset=offset, rotation=rotation)
    result = sampler.score_map(state, obs, 0)
    xy = np.stack(np.meshgrid(result['x_coords'], result['y_coords']), axis=-1)
    expected_union = np.zeros(xy.shape[:2], dtype=bool)
    rotations = sampler.quaternions(state, obs, 0)
    for index, quaternion in enumerate(rotations):
        vertices = Rotation.from_quat(quaternion).apply(points)[:, :2]
        expected = np.all((xy[..., None, :] + vertices >= -.47 - 1e-12)
                          & (xy[..., None, :] + vertices <= .47 + 1e-12), axis=(-1, -2))
        expected_union |= expected
        candidates = [c['xy'] for c in result['candidates'] if c['orientation_index'] == index]
        np.testing.assert_allclose(candidates, xy[expected], atol=1e-12)
    np.testing.assert_array_equal(result['valid'], expected_union)
    # A different state reuses geometry and bottom kernels, without recomputing either.
    saved_bounds = next(iter(sampler._footprint_bounds.values()))
    monkeypatch.setattr('stacking_common.actions.scored.bottom_height_kernel',
        lambda *args: pytest.fail('orientation kernel was rebuilt for a new state'))
    obs.scene_height_map[:] = .01
    sampler.score_map(state, obs, 0)
    assert next(iter(sampler._footprint_bounds.values())) is saved_bounds


@pytest.mark.parametrize('relaxed', [False, True])
def test_orientation_specific_connectivity_and_fill_match_explicit_rectangles(config, relaxed):
    settings = config['settings']
    settings['floor']['footprint_scale'] = .85
    sampler, state, obs, points = make_scene(settings, offset=(.07, -.03, 0.), inset=.6)
    axis = np.linspace(-.49, .49, 25)
    xx, yy = np.meshgrid(axis, axis)
    occupied = ((xx > .22) & (yy < .08)) | ((xx < -.3) & (yy > .15))
    layer = dict(xx=xx, yy=yy, occupied=occupied, support_z=0., bottom_tol=.02)
    sampler.floor_policy = SimpleNamespace(context_details=lambda state: layer)
    sampler.relax_floor_fill = relaxed
    result = sampler.score_map(state, obs, 0)
    xy = np.stack(np.meshgrid(result['x_coords'], result['y_coords']), axis=-1)
    neighbors = floor.dilated_mask(occupied)
    expected = {}
    connected_sets = []
    max_empty = 1
    for orientation, q in enumerate(sampler.quaternions(state, obs, 0)):
        vertices = Rotation.from_quat(q).apply(points)[:, :2]
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        connected_cells = set()
        for row, column in np.ndindex(xy.shape[:2]):
            anchor = xy[row, column]
            if not np.all((anchor >= -.5 - .6 * lower) & (anchor <= .5 - .6 * upper)):
                continue
            footprint = ((xx >= anchor[0] + .85 * lower[0]) & (xx <= anchor[0] + .85 * upper[0])
                         & (yy >= anchor[1] + .85 * lower[1]) & (yy <= anchor[1] + .85 * upper[1]))
            empty = np.count_nonzero(footprint & ~occupied)
            max_empty = max(max_empty, empty)
            if np.any(footprint & neighbors) and (relaxed or empty > 0):
                expected[(orientation, row, column)] = empty
                connected_cells.add((row, column))
        connected_sets.append(connected_cells)
    assert connected_sets[0] != connected_sets[1]
    actual = {}
    for candidate in result['candidates']:
        row = np.argmin(abs(result['y_coords'] - candidate['xy'][1]))
        column = np.argmin(abs(result['x_coords'] - candidate['xy'][0]))
        actual[(candidate['orientation_index'], row, column)] = candidate
    assert actual.keys() == expected.keys()
    for row, column in np.ndindex(xy.shape[:2]):
        choices = [(key, c) for key, c in actual.items() if key[1:] == (row, column)]
        assert result['candidate_mask'][row, column] == bool(choices)
        if choices:
            key, best = max(choices, key=lambda item: item[1]['score'])
            assert result['scores'][row, column] == best['score']
            assert result['orientation_index'][row, column] == key[0]
            assert result['fill_area'][row, column] == pytest.approx(0. if relaxed else expected[key] / max_empty)
