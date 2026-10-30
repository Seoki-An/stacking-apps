"""A single construction-order gate, independent of connectivity and repair."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from stacking_common.actions import PlacementSolveResult, InitializedPlacement
from stacking_common.actions.filtering import ActiveLayerAdmissionFilter
from stacking_common.actions.floor import ActiveFloorPolicy
from stacking_common.domain import ObjectInventory, StackObject, State, TargetGeometry, DirectionalTaper
from stacking_common.profiles.action import load_action_generation_profile
from test_action_generation import _BodyModel, _cube, _pose

ROOT = Path(__file__).resolve().parents[1]


def scene():
    objects = [StackObject(i, _BodyModel(), [_cube(.2)]) for i in (1, 2, 3)]
    inv = ObjectInventory(objects, {i: _pose() for i in (1, 2, 3)},
        TargetGeometry(.4, .2, 1., np.zeros(3), DirectionalTaper(), 0.))
    policy = ActiveFloorPolicy(inv, bottom_tolerance=.01, fill_ratio=.75,
                              grid_size=16, coverage_mode='projected_footprint')
    partial = State([1, 2, 3], [0], {1: _pose(x=-.1, z=.1)}, [], [])
    complete = State([1, 2, 3], [0, 1], {1: _pose(x=-.1, z=.1), 2: _pose(x=.1, z=.1)}, [], [])
    return inv, policy, partial, complete


def decide(gate, inv, state, z, x=0., context=None):
    return gate.evaluate(inv, state, 2, InitializedPlacement(_pose(z=5.), 0.),
        PlacementSolveResult(_pose(x=x, z=z), c_feq=0., c_gap=0.), scene_context=context)


def test_rejects_only_above_parent_layer_and_uses_solved_height():
    inv, policy, partial, complete = scene()
    gate = ActiveLayerAdmissionFilter(policy)
    assert decide(gate, inv, partial, .1).accepted
    assert decide(gate, inv, partial, .11 - 1e-10).accepted
    above = decide(gate, inv, partial, .12)
    assert not above.accepted and above.reason == 'above_active_layer'
    assert above.diagnostics['fixed_scene_active_layer_bottom'] == pytest.approx(.02)
    # No connectivity or empty-area overlap rejection is restored by this gate.
    assert decide(gate, inv, partial, .1, x=1.).accepted
    assert decide(gate, inv, partial, .1, x=-.1).accepted
    assert decide(gate, inv, complete, .3).accepted
    assert not decide(gate, inv, complete, .5).accepted
    # Reusing this filter for a sibling must not retain the completed layer.
    assert not decide(gate, inv, partial, .3).accepted


def test_prepared_context_and_empty_scene():
    inv, policy, partial, complete = scene()
    gate = ActiveLayerAdmissionFilter(policy)
    for state in (partial, complete, State([1, 2, 3], [], {}, [], [])):
        ctx = gate.prepare_scene(inv, state)
        for z in (.1, .3, .5):
            direct = decide(gate, inv, state, z)
            prepared = decide(gate, inv, state, z, context=ctx)
            assert direct.accepted == prepared.accepted
            assert direct.diagnostics == prepared.diagnostics


def test_batched_generation_filters_optimized_poses():
    from stacking_common.actions import PlacementActionGenerator, PlacementActionGeneratorConfig, PlanarPlacement
    inv, policy, partial, _ = scene()
    generator = PlacementActionGenerator(inv,
        PlacementActionGeneratorConfig(2, 0., 0., .001), SimpleNamespace(),
        SimpleNamespace(initialize=lambda inv, state, index, proposal:
                        InitializedPlacement(proposal.pose, 0.)),
        SimpleNamespace(solve=lambda inv, state, index, initial:
                        PlacementSolveResult(initial.pose, c_feq=0., c_gap=0.)),
        filters=(ActiveLayerAdmissionFilter(policy),))
    try:
        low, high = generator.build_candidates(partial, [
            (2, PlanarPlacement(_pose(z=.1))), (2, PlanarPlacement(_pose(z=.3)))])
        assert low[0] is not None and low[1] is None
        assert low[0].diagnostics['active_layer_admission_checked']
        assert high == (None, 'above_active_layer')
    finally:
        generator.close()


def test_composite_uses_candidate_region_layer():
    from test_composite_target_sampling import inventory, box, state, pose, policy
    inv = inventory(shapes=[box(2., 2., .5), box(.25, .25, .25)])
    floor = policy(inv)
    parent = state(inv, [pose(-3.)])
    gate = ActiveLayerAdmissionFilter(floor)
    def check(x, z):
        return gate.evaluate(inv, parent, 1, InitializedPlacement(pose(x, z=3.), 0.),
                             PlacementSolveResult(pose(x, z=z), c_feq=0., c_gap=0.))
    assert check(-3., .5).accepted  # Filled west region advances.
    assert not check(3., .5).accepted  # Empty east region stays at the ground.
    assert check(3., 0.).accepted  # No disconnected-region rejection.


@pytest.mark.parametrize('platform', ['tabletop', 'excavator'])
def test_profile_switch_and_root_descendant_wiring(platform, tmp_path):
    from stacking_common.artifacts.planning_context import _values
    from stacking_tabletop import TabletopApplication
    from stacking_excavator import ExcavatorApplication
    cls = TabletopApplication if platform == 'tabletop' else ExcavatorApplication
    app = cls.from_repository(ROOT)
    inv, _, _, _ = scene()
    assert app.action_generation_profile.generator.reject_above_active_layer
    for enabled in (False, True):
        changed = replace(app, action_generation_profile=replace(app.action_generation_profile,
            generator=replace(app.action_generation_profile.generator, reject_above_active_layer=enabled)))
        recorded = _values(changed.action_generation_profile)['generator']
        assert recorded.get('reject_above_active_layer', False) is enabled
        for root in (False, True):
            generator = changed.build_action_generator(inv, SimpleNamespace(), root=root)
            try:
                gates = [f for f in generator.filters if isinstance(f, ActiveLayerAdmissionFilter)]
                assert len(gates) == int(enabled)
                if gates:
                    assert gates[0].floor_policy is generator.planar_sampler.floor_policy
            finally:
                generator.close()
    values = yaml.safe_load((ROOT/f'apps/{platform}/config/action_generation.yml').read_text())
    values['filtering'].pop('reject_above_active_layer')
    path = tmp_path/'action.yml';path.write_text(yaml.safe_dump(values))
    assert not load_action_generation_profile(path,max_poses_per_object=32,banned_object_ids=()).generator.reject_above_active_layer
    values['filtering']['reject_above_active_layer'] = 'false'
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(TypeError,match='reject_above_active_layer'):
        load_action_generation_profile(path,max_poses_per_object=32,banned_object_ids=())
