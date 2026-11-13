"""Floor fallback recomputes geometry and scores without changing scene history."""
from types import SimpleNamespace

import numpy as np
import pytest

from stacking_common.actions.filtering import ActiveLayerAdmissionFilter
from stacking_common.actions.floor import ActiveFloorPolicy
from stacking_common.actions import MappingOrientationProvider, PlanarSamplingConfig
from stacking_common.actions.scored import ScoredPlanarPoseSampler
from stacking_common.visualization.heightmap import HeightMapRenderer
from test_active_layer_admission import scene, decide
from test_action_generation import _pose


def test_next_floor_keeps_height_gate_and_does_not_mark_skipped_floor_complete():
    inv, policy, parent, _ = scene()
    parent.placed_object_indices.append(1)
    parent.object_poses[2] = _pose(x=.1, z=.3)
    sibling = parent.copy()
    gate = ActiveLayerAdmissionFilter(policy)
    original = policy.context_details(parent)
    with policy.advance_for_search((parent,), 1):
        advanced = policy.context_details(parent)
        assert advanced['support_z'] == pytest.approx(.2)
        assert advanced['completed_layers'] == []
        assert advanced['skipped_layers'] == [{'support_z': 0., 'coverage': .5}]
        assert not np.array_equal(original['occupied'], advanced['occupied'])
        assert decide(gate, inv, parent, .3).accepted
        assert not decide(gate, inv, parent, .5).accepted
        assert not decide(gate, inv, sibling, .3).accepted
        with policy.advance_for_search((parent,), 2):
            top = policy.context_details(parent)
            assert top['support_z'] == pytest.approx(.4)
            assert len(top['skipped_layers']) == 2
            assert top['completed_layers'] == []
            assert not top['occupied'].any()
            assert decide(gate, inv, parent, .5).accepted
        assert policy.context_details(parent)['support_z'] == pytest.approx(.2)
    assert policy.context_details(parent)['support_z'] == 0.
    np.testing.assert_array_equal(original['occupied'], policy.context_details(parent)['occupied'])


@pytest.mark.parametrize('composite', [False, True])
def test_score_cache_and_filling_follow_advanced_floor_and_restore(composite):
    if composite:
        from test_composite_target_sampling import inventory, policy as make_policy, state, pose
        inv = inventory()
        parent = state(inv, [pose(-3.5, -.5), pose(-3., .5, z=.25)])
        policy = make_policy(inv)
    else:
        inv, policy, parent, _ = scene()
        parent.placed_object_indices.append(1)
        parent.object_poses[2] = _pose(x=.1, z=.3)
    settings = {
        'floor': dict(bottom_tolerance=policy.bottom_tolerance, fill_ratio=policy.fill_ratio,
                      grid_size=policy.grid_size, coverage_mode=policy.coverage_mode,
                      footprint_scale=.9),
        'contact_gap_tolerance': .03, 'target_mask_margin': 0.,
        'score': dict(height=-1.5, support_likelihood=2., fill_area=4., target_boundary=.2),
    }
    sample = ScoredPlanarPoseSampler(inv, PlanarSamplingConfig((24, 24), (0.,), .5),
        MappingOrientationProvider({i: [0., 0., 0., 1.] for i in inv.object_ids}),
        settings, floor_policy=policy, connectivity_enabled=False)
    renderer = HeightMapRenderer(inv, (64, 64), .2)
    observation = SimpleNamespace(scene_height_map=renderer.render(parent),
        height_map_xlim=renderer.xlim, height_map_ylim=renderer.ylim)
    def scores(state):
        data = sample.score_map(state, observation, 2)
        return data['regions'][('west',)]['scores'] if composite else data['scores']
    baseline = scores(parent).copy()
    with policy.advance_for_search((parent,), 1):
        advanced = scores(parent).copy()
        assert not np.allclose(baseline, advanced, equal_nan=True)
        np.testing.assert_allclose(scores(parent.copy()), baseline, equal_nan=True)
        np.testing.assert_allclose(scores(parent), advanced, equal_nan=True)
        if composite:
            layers = {r['region_id']: r for r in policy.context_details(parent)['regions']}
            assert layers[('west',)]['support_z'] == pytest.approx(.25)
            assert layers[('east',)]['support_z'] == 0.  # Empty disconnected region cannot advance.
    np.testing.assert_allclose(scores(parent), baseline, equal_nan=True)


def test_search_advances_again_after_next_layer_pool_has_no_admissible_candidate():
    from dataclasses import replace
    from test_mcts_floor_fallback import setup
    from stacking_common.actions import PlanarPlacement
    from stacking_common.search import MCTSSearchRequest
    search, root, base, deeper, _ = setup(depth=1)
    root.state.placed_object_indices.append(1)
    root.state.object_poses[2] = _pose(x=.1, z=.3)
    search.config = replace(search.config, rejected_root_retry_rounds=2)
    base.planar_sampler.sample = lambda *args: [PlanarPlacement(_pose(z=.5))]
    try:
        result = search.search(root, MCTSSearchRequest(iterations=1))
        assert result.selected is not None
        assert result.floor_fill_fallback.pools == 2
        action = result.selected.action
        assert action.diagnostics['active_layer_fallback_level'] == 2
        assert action.diagnostics['fixed_scene_active_layer_support_z'] == pytest.approx(.4)
        assert not action.diagnostics['fixed_scene_active_layer_above']
        assert action.diagnostics['fixed_scene_completed_layers'] == []
        assert len(action.diagnostics['fixed_scene_skipped_layers']) == 2
        assert all(f.floor_policy.fallback_level(root.state) == 0 for g in (base, deeper)
                   for f in g.filters if isinstance(f, ActiveLayerAdmissionFilter))
    finally:
        search.close()


def test_exhausted_top_floor_does_not_create_another_fallback_pool():
    from test_mcts_floor_fallback import setup
    from stacking_common.search import MCTSSearchRequest
    search, root, base, _, _ = setup(fail=True, depth=1)
    try:
        result = search.search(root, MCTSSearchRequest(iterations=1))
        assert result.selected is None
        # One partially filled ground layer and an empty top: only one advance.
        assert result.floor_fill_fallback.pools == 1
        assert all(f.floor_policy.fallback_level(root.state) == 0 for f in base.filters
                   if isinstance(f, ActiveLayerAdmissionFilter))
    finally:
        search.close()


def test_timeout_in_advanced_floor_never_advances_again(monkeypatch):
    import stacking_common.search.mcts.search as module
    from test_mcts_floor_fallback import setup
    from stacking_common.search import MCTSSearchRequest
    search, root, base, _, evaluator = setup(fail=True, depth=1)
    clock = [0.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    evaluator.clock = clock  # The first physical check consumes the deadline.
    try:
        result = search.search(root, MCTSSearchRequest(iterations=1, max_time_seconds=10.))
        assert result.timed_out
        assert result.floor_fill_fallback.pools == 1
        assert evaluator.calls
        assert all(a.diagnostics['active_layer_fallback_level'] == 1 for a in evaluator.calls)
        assert all(f.floor_policy.fallback_level(root.state) == 0 for f in base.filters
                   if isinstance(f, ActiveLayerAdmissionFilter))
    finally:
        search.close()


def test_descendant_can_advance_two_incomplete_floors():
    from dataclasses import replace
    from stacking_common.actions import PlanarPlacement
    from stacking_common.search import MCTSSearchRequest
    from test_mcts_floor_fallback import setup
    search, root, base, deeper, _ = setup(depth=2)
    search.config = replace(search.config, rejected_root_retry_rounds=2)
    # Root puts a second stone above a partially filled ground floor.
    base.planar_sampler.sample = lambda *args: [PlanarPlacement(_pose(x=-.1, z=.3))]
    deeper.planar_sampler.sample = lambda *args: [PlanarPlacement(_pose(x=-.1, z=.5))]
    try:
        result = search.search(root, MCTSSearchRequest(iterations=100))
        assert result.selected is not None
        children = [c for n in root.children for c in n.evaluated_children()]
        assert children
        assert any(c.action.diagnostics.get('active_layer_fallback_level') == 2 for c in children)
        assert any(c.depth == 2 for c in children)
        assert all(f.floor_policy.fallback_level(n.state) == 0
                   for n in root.evaluated_children() for f in deeper.filters
                   if isinstance(f, ActiveLayerAdmissionFilter))
    finally:
        search.close()
