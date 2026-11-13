"""Proposal-locality admission before physics and after simulated placement."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from stacking_common.actions import PlacementSolveResult, PlanarPlacement
from stacking_common.actions.filtering import ActiveLayerAdmissionFilter, InitialXYDisplacementFilter
from stacking_common.domain import Action, State
from stacking_common.profiles.action import load_action_generation_profile
from stacking_common.search import EvaluationFidelity, StackingEnvironmentEvaluator
from test_action_generation import _inventory, _state, _pose, _Initializer, _PlanarSampler
from stacking_common.actions import PlacementActionGenerator, PlacementActionGeneratorConfig

ROOT = Path(__file__).resolve().parents[1]


def test_recorded_step14_rejects_and_large_vertical_settling_does_not():
    length = .6421492938605162
    check = InitialXYDisplacementFilter([SimpleNamespace(volume=length**3)], .75)
    action = Action(0, _pose(.06381738930940628, -.43428343534469604, .6769),
                    _pose(.564516129032258, -.11290322580645151, 1.6018))
    decision = check.check(action)
    assert not decision.accepted
    assert decision.reason == 'initial_xy_displacement'
    assert decision.diagnostics['posegen_initial_xy_displacement'] == pytest.approx(.5949659377)
    assert decision.diagnostics['initial_xy_displacement_limit'] == pytest.approx(.4816119704)
    # Vertical clearance/settling is deliberately not part of this XY gate.
    assert check.check(Action(0, _pose(z=-10), _pose(z=10))).accepted


@pytest.mark.parametrize('scale', [.04, 1., 5.])
def test_limit_is_candidate_size_normalized_and_inclusive(scale):
    check = InitialXYDisplacementFilter([SimpleNamespace(volume=scale**3)], .75)
    limit = .75 * check.length_scales[0]
    assert check.check(Action(0, _pose(limit), _pose())).accepted
    assert not check.check(Action(0, _pose(limit * 1.001), _pose())).accepted


def test_fresh_and_mutated_proposals_keep_filter_during_floor_relaxation():
    inv = _inventory(2)
    parent = _state(inv, placed=(1,))
    config = PlacementActionGeneratorConfig(2, .01, 0., None,
        max_initial_xy_displacement_ratio=.75)
    from stacking_common.actions.floor import ActiveFloorPolicy
    floor = ActiveFloorPolicy(inv, bottom_tolerance=.01, fill_ratio=.75, grid_size=16)
    solver = SimpleNamespace(solve=lambda inv, state, i, initial:
                             PlacementSolveResult(initial.pose + np.array([.04, 0, 0, 0, 0, 0, 0]), 0., 0.))
    generator = PlacementActionGenerator(inv, config, _PlanarSampler(), _Initializer(), solver,
        filters=(ActiveLayerAdmissionFilter(floor),))
    from stacking_common.search import ActionGenerationRequest
    try:
        fresh = generator.generate(parent, None, ActionGenerationRequest(max_candidates=2))
        assert fresh.rejection_counts == {'initial_xy_displacement': 2}
        with generator.floor_fill_fallback((parent,)) as enabled:
            assert enabled
            # CEM mutation and seed construction share build_candidates.
            assert generator.build_candidates(parent, [(0, PlanarPlacement(_pose()))]) == (
                (None, 'initial_xy_displacement'),)
    finally:
        generator.close()


@pytest.mark.parametrize('solved_x,final_x,reason', [
    (.5, .7, None), (.5, .8, 'initial_xy_displacement'),
    (.8, .5, 'initial_xy_displacement'),
])
def test_scalar_batch_and_root_jump_cannot_bypass_final_check(solved_x, final_x, reason):
    check = InitialXYDisplacementFilter([SimpleNamespace(volume=1.)], .75)
    parent = State([10], [], {}, [], [])
    action = Action(0, _pose(solved_x), _pose(), object_id=10)
    child = State([10], [0], {10: _pose(final_x)}, [], [action.copy()])
    result = SimpleNamespace(state=child, observation=None, reward=2., info={})
    env = SimpleNamespace(reset=lambda: None, restore=lambda state: None,
        step=lambda action: result, step_batch=lambda states, actions: [result])
    evaluator = StackingEnvironmentEvaluator({EvaluationFidelity.LONG: env},
        initial_xy_displacement_filter=check, allow_root_jumps=True)
    evaluator.begin_search(parent)
    for actual in (evaluator.evaluate(parent, action, EvaluationFidelity.LONG),
                   evaluator.evaluate_batch([parent], [action], EvaluationFidelity.LONG)[0]):
        assert actual.failed == (reason is not None)
        assert actual.rejection_reason == reason
        assert actual.diagnostics['final_initial_xy_displacement'] == final_x
        assert actual.diagnostics['initial_xy_displacement_stage'] == 'simulation'
    result.info = {'rejection_reason': 'target_motion_limit'}
    assert evaluator.evaluate(parent, action, EvaluationFidelity.LONG).rejection_reason == 'target_motion_limit'


def test_preserved_solved_pose_is_checked_even_if_action_pose_was_replaced():
    check = InitialXYDisplacementFilter([SimpleNamespace(volume=1.)], .75)
    action = Action(0, _pose(.8), _pose())
    action.pose = _pose(.1)
    assert not check.check(action, _pose(.1)).accepted


@pytest.mark.parametrize('platform', ['excavator', 'tabletop'])
def test_profile_enabled_and_missing_or_null_key_preserves_old_behavior(tmp_path, platform):
    values = yaml.safe_load((ROOT / f'apps/{platform}/config/action_generation.yml').read_text())
    path = tmp_path / 'actions.yml'
    for mode in ('enabled', 'null', 'missing'):
        if mode == 'null':
            values['filtering']['max_initial_xy_displacement_ratio'] = None
        elif mode == 'missing':
            del values['filtering']['max_initial_xy_displacement_ratio']
        path.write_text(yaml.safe_dump(values))
        profile = load_action_generation_profile(path, max_poses_per_object=4, banned_object_ids=())
        assert profile.generator.max_initial_xy_displacement_ratio == (.5 if mode == 'enabled' else None)


@pytest.mark.parametrize('ratio', [0., -1., float('inf'), float('nan'), True])
def test_invalid_config_limit_rejected(ratio):
    with pytest.raises(ValueError, match='XY displacement'):
        PlacementActionGeneratorConfig(2, .01, 0., None, max_initial_xy_displacement_ratio=ratio)
