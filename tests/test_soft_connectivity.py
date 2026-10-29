"""Soft distance preferences never veto geometrically valid jump proposals."""
from types import SimpleNamespace

import numpy as np
import pytest

from stacking_common.actions import (InitializedPlacement, PlacementSolveResult,
    PlacementActionGeneratorConfig, PlanarPlacement)
from stacking_common.actions.connectivity import bounded_distance_penalty
from stacking_common.actions.scored import ScoredPlacementActionGenerator
from stacking_common.profiles.action import resolve_scored_sampling_settings
from stacking_common.workflows.search import StackingSearchApplication
from test_previous_action_connectivity import setup
from test_composite_target_sampling import sampler, pose, state


def test_bounded_penalty_uses_footprint_gap_and_is_scale_invariant():
    square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    shifts = np.array([[0., 0.], [1.05, 0.], [1.6, 0.], [101., 0.]])
    cost = bounded_distance_penalty(square, square, shifts, .1, .5)
    np.testing.assert_allclose(cost[:3], [0., 0., .5])
    assert .5 < cost[3] < 1.
    np.testing.assert_allclose(cost, bounded_distance_penalty(
        square * .1, square * .1, shifts * .1, .01, .05))


@pytest.mark.parametrize('composite', [False, True])
def test_soft_maps_keep_far_anchors_and_only_subtract_bounded_score(composite):
    inv, current, obs = setup(composite)
    sample = sampler(inv)
    sample.settings['connectivity'] = 'previous_action_soft'
    sample.settings['score']['previous_action_distance'] = 0.
    unweighted = list(sample.score_map(current, obs, 2)['candidates'])
    sample.settings['score']['previous_action_distance'] = 1.
    weighted = list(sample.score_map(current, obs, 2)['candidates'])
    assert len(weighted) == len(unweighted) > 0
    deltas = []
    for a, b in zip(unweighted, weighted):
        np.testing.assert_array_equal(a['xy'], b['xy'])
        deltas.append(a['score'] - b['score'])
    assert min(deltas) >= -1e-12 and max(deltas) < 1.
    far = [d for c, d in zip(weighted, deltas) if c['xy'][0] < 0]
    near = [d for c, d in zip(weighted, deltas) if c['xy'][0] > 0]
    assert far and near and np.mean(far) > np.mean(near)
    assert sample.allows_initial_pose(current, 2, pose(-3. if composite else -1.))
    # Reordering the preceding stones moves the preference, not the mask.
    current.placed_object_indices[:] = [1, 0]
    reversed_map = list(sample.score_map(current, obs, 2)['candidates'])
    assert len(reversed_map) == len(weighted)
    assert any(a['score'] != b['score'] for a, b in zip(weighted, reversed_map))


@pytest.mark.parametrize('composite', [False, True])
def test_empty_scene_has_no_distance_penalty(composite):
    inv, _, obs = setup(composite)
    sample = sampler(inv)
    sample.settings['connectivity'] = 'previous_action_soft'
    sample.settings['score']['previous_action_distance'] = 0.
    first = list(sample.score_map(state(inv), obs, 0)['candidates'])
    sample.settings['score']['previous_action_distance'] = 10.
    second = list(sample.score_map(state(inv), obs, 0)['candidates'])
    np.testing.assert_array_equal([c['score'] for c in first], [c['score'] for c in second])


def test_soft_policy_has_no_root_quota_or_post_simulation_connectivity_veto():
    app = SimpleNamespace(sampling_settings={'connectivity':'previous_action_soft',
                                             'root_global_fraction': .3})
    assert not StackingSearchApplication.root_jumps_enabled(app)
    assert StackingSearchApplication.build_connectivity_check(app, None) is None


def test_disconnected_mutation_and_solved_pose_are_accepted_by_soft_policy():
    inv, current, _ = setup(False)
    sample = sampler(inv)
    sample.settings['connectivity'] = 'previous_action_soft'
    generator = ScoredPlacementActionGenerator(inventory=inv,
        config=PlacementActionGeneratorConfig(max_poses_per_object=4,
            initial_height_clearance=.1, initial_height_xy_padding=.01,
            posegen_gap_threshold=.001), planar_sampler=sample,
        initializer=SimpleNamespace(initialize=lambda *args: InitializedPlacement(pose(-1.), 0.)),
        pose_solver=SimpleNamespace(solve=lambda *args: PlacementSolveResult(pose(-1.), 0., 0.)))
    candidate, reason = generator.build_candidates(current, [(2, PlanarPlacement(pose(-1.)))])[0]
    assert candidate is not None and reason is None
    assert generator.root_global_fraction == 0.


@pytest.mark.parametrize('weight', [-1., float('nan'), float('inf')])
def test_invalid_distance_weight_is_rejected(weight):
    inv, _, _ = setup(False)
    settings = sampler(inv).settings
    settings['score']['previous_action_distance'] = weight
    with pytest.raises(ValueError, match='previous_action_distance'):
        resolve_scored_sampling_settings(settings)
