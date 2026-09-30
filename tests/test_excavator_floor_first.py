"""Recorded elevated placements must not bypass the excavator's empty floor."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stacking_common.actions import ActiveFloorFilter
from stacking_common.actions.types import InitializedPlacement, PlacementSolveResult
from stacking_common.domain import Action, State
from stacking_common.environment.transition import PlacementResult
from stacking_excavator import ExcavatorApplication
from stacking_excavator.planning.coordinator import ExcavatorPlanningCoordinator

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def recorded_scene():
    data = json.loads((ROOT / 'tests/fixtures/planning/excavator_floor_step12.json').read_text())
    app = ExcavatorApplication.from_repository(ROOT)
    ids = data['object_ids']
    inventory = app.load_inventory(ids, {i: np.array([0., 0., 0., 0., 0., 0., 1.]) for i in ids})
    state = State(ids, [ids.index(o['object_id']) for o in data['placed']],
        {o['object_id']: np.array(o['pose']) for o in data['placed']}, [], [])
    return app, inventory, state, data['candidates']


def test_default_excavator_rejects_recorded_upper_floor_candidates(recorded_scene):
    app, inventory, state, candidates = recorded_scene
    generator = app.build_action_generator(inventory, SimpleNamespace())
    floor_filter = next(f for f in generator.filters if isinstance(f, ActiveFloorFilter))
    support_z, occupied = floor_filter.policy.context(state)
    assert support_z == 0.
    assert .5 < occupied.mean() < .6  # Projected coverage remains below the fill target.
    for candidate in candidates:
        decision = floor_filter.evaluate(inventory, state,
            state.object_ids.index(candidate['object_id']),
            InitializedPlacement(np.array(candidate['init_pose']), 0.),
            PlacementSolveResult(np.array(candidate['pose']), 0., 0.))
        assert not decision.accepted
        assert decision.reason == 'active_floor_above_active_layer'


def test_floor_gate_can_be_explicitly_disabled(recorded_scene):
    app, inventory, _, _ = recorded_scene
    app = replace(app, action_generation_profile=replace(
        app.action_generation_profile, strict_floor_first=False))
    generator = app.build_action_generator(inventory, SimpleNamespace())
    assert not any(isinstance(f, ActiveFloorFilter) for f in generator.filters)


def test_fixed_scene_settling_cannot_admit_recorded_upper_floor_selection(recorded_scene):
    app, inventory, state, candidates = recorded_scene
    selected = next(c for c in candidates if c['selected'])
    object_id = selected['object_id']
    action = Action(state.object_ids.index(object_id), np.array(selected['pose']),
        np.array(selected['init_pose']), object_id=object_id)
    backend = SimpleNamespace(reset=lambda _: None,
        settle_fixed=lambda *_: PlacementResult(
            {**state.object_poses, object_id: action.pose.copy()}))
    calls = []
    coordinator = ExcavatorPlanningCoordinator(app, inventory,
        SimpleNamespace(plan=lambda problem: calls.append(problem)), placement_backend=backend)
    record = coordinator.attempt(state, action, step=12, attempt=1)
    assert not record.result.succeeded
    assert record.result.reason == 'active_floor_above_active_layer'
    assert not record.result.diagnostics['motion_planning_attempted']
    assert not calls


@pytest.fixture
def coverage_mismatch_scene():
    data = json.loads((ROOT / 'tests/fixtures/planning/excavator_floor_step11.json').read_text())
    app = ExcavatorApplication.from_repository(ROOT)
    ids = data['object_ids']
    inventory = app.load_inventory(ids, {i: np.array([0., 0., 0., 0., 0., 0., 1.]) for i in ids})
    state = State(ids, [ids.index(o['object_id']) for o in data['placed']],
        {o['object_id']: np.array(o['pose']) for o in data['placed']}, [], [])
    return app, inventory, state


def test_projected_coverage_advances_without_search_exhaustion(coverage_mismatch_scene):
    from stacking_common.actions import ActiveFloorPolicy
    app, inventory, state = coverage_mismatch_scene
    policy = app.build_floor_policy(inventory)
    context = policy.context_details(state)
    assert context['coverage_mode'] == 'projected_footprint'
    assert context['occupied'].shape == (16, 16)
    assert len(context['completed_layers']) == 1
    assert .6 <= context['completed_layers'][0]['coverage'] < .75
    assert context['support_z'] == pytest.approx(.9344364215187257)
    assert not context['occupied'].any()  # Previous-floor stones do not fill the next one.
    old = ActiveFloorPolicy(inventory, **{**app.floor_settings,
        'coverage_mode': 'bottom_slice', 'grid_size': 32})
    old_z, old_occupied = old.context(state)
    assert old_z == 0.
    assert old_occupied.mean() == pytest.approx(.5654296875)

    index = next(i for i in range(len(inventory)) if i not in state.placed_object_indices)
    pose = np.array([0., 0., 0., 0., 0., 0., 1.])
    bottom = policy._entry(index, pose)['bottom']
    pose[2] = context['support_z'] - bottom
    action = Action(index, pose, pose, object_id=state.object_ids[index])
    assert old.active_floor_rejection(state, action)[0] == 'above_active_layer'
    reason, diagnostics = policy.active_floor_rejection(state, action)
    assert reason is None
    assert diagnostics['fixed_scene_completed_layers'] == context['completed_layers']
    action.pose[2] += policy.bottom_tolerance + .01
    assert policy.active_floor_rejection(state, action)[0] == 'above_active_layer'


def test_projected_sampler_and_filter_share_floor_and_refresh_on_state_change(
    coverage_mismatch_scene, monkeypatch,
):
    from stacking_common.actions import scored
    app, inventory, state = coverage_mismatch_scene
    generator = app.build_action_generator(inventory, SimpleNamespace())
    sampler = generator.planar_sampler
    policy = next(f.policy for f in generator.filters if isinstance(f, ActiveFloorFilter))
    assert sampler.floor_policy is policy
    observation = SimpleNamespace(scene_height_map=np.zeros((32, 32)),
        height_map_xlim=np.array([-2., 2.]), height_map_ylim=np.array([-2.75, 2.75]))
    contexts = []
    original = scored.floor.score_xy_debug_map
    def capture(*args, **kwargs):
        contexts.append(kwargs['active_floor'])
        return original(*args, **kwargs)
    monkeypatch.setattr(scored.floor, 'score_xy_debug_map', capture)
    index = next(i for i in range(len(inventory)) if i not in state.placed_object_indices)
    sampler.score_map(state, observation, index)
    expected = policy.context_details(state)
    assert contexts[-1]['support_z'] == expected['support_z']
    np.testing.assert_array_equal(contexts[-1]['occupied'], expected['occupied'])
    # A new, less-filled state must not reuse the completed-floor decision.
    sparse = State(state.object_ids, state.placed_object_indices[:1],
        {state.object_ids[state.placed_object_indices[0]]:
         state.object_poses[state.object_ids[state.placed_object_indices[0]]]}, [], [])
    sampler.score_map(sparse, observation, index)
    assert contexts[-1]['support_z'] == 0.
    assert contexts[-1]['completed_layers'] == []
