"""Optional last-placement XY connectivity across fresh and mutated seeds."""
from types import SimpleNamespace

import numpy as np
import pytest

from stacking_common.actions import PlacementActionGeneratorConfig, PlanarPlacement
from stacking_common.actions.scored import ScoredPlacementActionGenerator
from stacking_common.domain import DirectionalTaper, State, TargetGeometry
from stacking_common.profiles.action import resolve_scored_sampling_settings
from stacking_common.visualization.heightmap import HeightMapRenderer
from test_composite_target_sampling import box, inventory, pose, sampler, state, target


def setup(composite):
    geometry = target() if composite else TargetGeometry(4., 2., 3., np.zeros(3), DirectionalTaper(), 0.)
    inv = inventory(geometry, [box(.4, .4, .3)] * 3)
    current = state(inv, [pose(-3. if composite else -1.), pose(3. if composite else 1.)])
    renderer = HeightMapRenderer(inv, (64, 64), .1)
    observation = SimpleNamespace(scene_height_map=renderer.render(current),
        height_map_xlim=renderer.xlim, height_map_ylim=renderer.ylim)
    return inv, current, observation


@pytest.mark.parametrize('composite', [False, True])
def test_only_previous_footprint_is_admitted_and_placement_order_matters(composite):
    inv, current, obs = setup(composite)
    sample = sampler(inv)
    baseline = list(sample.score_map(current, obs, 2)['candidates'])
    assert any(c['xy'][0] < 0 for c in baseline)
    assert any(c['xy'][0] > 0 for c in baseline)
    sample.settings['connectivity'] = 'previous_action'
    limited = list(sample.score_map(current, obs, 2)['candidates'])
    assert limited and all(c['xy'][0] > 0 for c in limited)
    proposals = sample.sample(inv, current, obs, 2, 16, 'lower_floor_first')
    assert proposals and all(sample.allows_initial_pose(current, 2, p.pose) for p in proposals)
    reversed_state = State(current.object_ids, [1, 0], current.object_poses, [], [])
    reversed_candidates = list(sample.score_map(reversed_state, obs, 2)['candidates'])
    assert reversed_candidates and all(c['xy'][0] < 0 for c in reversed_candidates)


@pytest.mark.parametrize('composite', [False, True])
def test_first_placement_is_unchanged(composite):
    inv, _, obs = setup(composite)
    empty = state(inv)
    first, second = sampler(inv), sampler(inv)
    second.settings['connectivity'] = 'previous_action'
    a = first.sample(inv, empty, obs, 0, 10, 'lower_floor_first')
    b = second.sample(inv, empty, obs, 0, 10, 'lower_floor_first')
    np.testing.assert_array_equal([p.pose for p in a], [p.pose for p in b])


@pytest.mark.parametrize('composite', [False, True])
def test_mutated_seed_is_rejected_before_initialization_and_solve(composite):
    inv, current, obs = setup(composite)
    sample = sampler(inv)
    sample.settings['connectivity'] = 'previous_action'
    def unexpected(*args, **kwargs):
        pytest.fail('disconnected seed reached native work')
    generator = ScoredPlacementActionGenerator(inventory=inv,
        config=PlacementActionGeneratorConfig(max_poses_per_object=4,
            initial_height_clearance=.1, initial_height_xy_padding=.01,
            posegen_gap_threshold=.001), planar_sampler=sample,
        initializer=SimpleNamespace(initialize=unexpected), pose_solver=SimpleNamespace(solve=unexpected))
    # A mutated seed near the earlier stone must not escape the fresh mask.
    proposal = PlanarPlacement(pose(-3. if composite else -1.))
    assert generator.build_candidates(current, [(2, proposal)]) == ((None, 'previous_action_disconnected'),)


def test_last_placement_above_active_floor_still_defines_connectivity():
    inv, current, obs = setup(False)
    # The last placement is well above the active floor; its XY neighborhood
    # is still the requested reference, not the occupied lower-floor stone.
    current.object_poses[2][2] = 1.5
    sample = sampler(inv)
    sample.settings['connectivity'] = 'previous_action'
    assert sample.allows_initial_pose(current, 2, pose(1.))
    assert not sample.allows_initial_pose(current, 2, pose(-1.))


def test_single_target_without_injected_floor_policy_uses_same_mask():
    inv, current, obs = setup(False)
    sample = sampler(inv)
    sample.floor_policy = None
    sample.settings['connectivity'] = 'previous_action'
    proposals = sample.sample(inv, current, obs, 2, 16, 'lower_floor_first')
    assert proposals and all(p.pose[0] > 0 for p in proposals)
    assert all(sample.allows_initial_pose(current, 2, p.pose) for p in proposals)


def test_profile_validates_connectivity_and_defaults_to_current_behavior():
    import yaml
    from pathlib import Path
    values = yaml.safe_load(Path('apps/excavator/config/action_generation.yml').read_text())['planar']
    values.pop('connectivity', None)
    assert resolve_scored_sampling_settings(values)['connectivity'] == 'active_floor'
    values['connectivity'] = 'previous_action'
    assert resolve_scored_sampling_settings(values)['connectivity'] == 'previous_action'
    values['connectivity'] = 'typo'
    with pytest.raises(ValueError, match='connectivity'):
        resolve_scored_sampling_settings(values)


def test_metric_gap_ignores_floor_grid_resolution_and_rectangle_corners():
    from stacking_common.actions.connectivity import footprint_distances
    # Diamonds have overlapping AABBs here, but their actual projections do not.
    diamond = np.array([[0., -1.], [1., 0.], [0., 1.], [-1., 0.]])
    distances = footprint_distances(diamond, diamond, [[0., 0.], [2., 0.], [1.5, 1.5]])
    np.testing.assert_allclose(distances, [0., 0., 1. / np.sqrt(2)], atol=1e-12)
    square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    np.testing.assert_allclose(footprint_distances(square, square,
        [[1.1, 0.], [1.2, 0.], [1.3, 1.4]]), [.1, .2, .5], atol=1e-12)


@pytest.mark.parametrize('composite', [False, True])
def test_pose_optimization_cannot_move_a_connected_seed_away(composite):
    from stacking_common.actions import InitializedPlacement, PlacementSolveResult
    inv, current, _ = setup(composite)
    sample = sampler(inv)
    sample.settings['connectivity'] = 'previous_action'
    near, far = pose(3. if composite else 1.), pose(-3. if composite else -1.)
    generator = ScoredPlacementActionGenerator(inventory=inv,
        config=PlacementActionGeneratorConfig(max_poses_per_object=4,
            initial_height_clearance=.1, initial_height_xy_padding=.01,
            posegen_gap_threshold=.001), planar_sampler=sample,
        initializer=SimpleNamespace(initialize=lambda *args: InitializedPlacement(near, 0.)),
        pose_solver=SimpleNamespace(solve=lambda *args: PlacementSolveResult(far, 0., 0.)))
    assert generator.build_candidates(current, [(2, PlanarPlacement(near))]) == (
        (None, 'previous_action_disconnected'),)


def test_simulated_gap_uses_both_final_poses_and_rejects_scalar_and_batch():
    from stacking_common.actions.connectivity import PreviousActionConnectivity
    from stacking_common.domain import Action
    from stacking_common.search import StackingEnvironmentEvaluator, EvaluationFidelity
    inv, parent, _ = setup(False)
    action = Action(2, pose(1.), pose(1.), object_id=3)
    poses = {**parent.object_poses, 3: pose(1.)}
    poses[2] = pose(-1.)  # The previous stone moves during simulation.
    child = State(parent.object_ids, [0, 1, 2], poses, [], [])
    result = SimpleNamespace(state=child, observation=None, reward=1., info={})
    env = SimpleNamespace(reset=lambda: None, restore=lambda state: None,
                          step=lambda a: result, step_batch=lambda states, actions: [result])
    evaluator = StackingEnvironmentEvaluator({EvaluationFidelity.LONG: env},
        connectivity_check=PreviousActionConnectivity(inv, .1))
    a = evaluator.evaluate(parent, action, EvaluationFidelity.LONG)
    b = evaluator.evaluate_batch([parent], [action], EvaluationFidelity.LONG)[0]
    for evaluated in (a, b):
        assert evaluated.failed
        assert evaluated.rejection_reason == 'previous_action_disconnected'
        assert evaluated.diagnostics['previous_action_xy_gap'] == pytest.approx(1.6)
    # If both move together, there is no separation to reject.
    child.object_poses[3] = pose(-1.)
    assert not evaluator.evaluate(parent, action, EvaluationFidelity.LONG).failed
