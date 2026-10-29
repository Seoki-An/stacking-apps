"""Floor-guided proposals and physical validation on recorded scenes."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stacking_common.actions import PosegenFeasibilityFilter
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




def test_permissive_excavator_uses_floor_only_for_xy_sampling(recorded_scene, monkeypatch):
    from stacking_common.actions import PlanarPlacement
    from stacking_common.search import EvaluationFidelity

    _, inventory, state, candidates = recorded_scene
    app = ExcavatorApplication.from_repository(ROOT)
    generator = app.build_action_generator(inventory, SimpleNamespace())
    assert len(generator.filters) == 1
    assert isinstance(generator.filters[0], PosegenFeasibilityFilter)
    sampler = generator.planar_sampler
    assert sampler.floor_policy is not None
    assert all(sampler.settings['floor'][key] == value
               for key, value in app.floor_settings.items())
    assert app.default_mcts_request().root_sampling_priority == 'lower_floor_first'
    assert app.default_mcts_request().deeper_sampling_priority == 'lower_floor_first'
    selected = next(c for c in candidates if c['selected'])
    index = state.object_ids.index(selected['object_id'])
    initial = InitializedPlacement(np.array(selected['init_pose']), 0.)
    solved = PlacementSolveResult(np.array(selected['pose']), 0., 0.)
    assert sampler.floor_policy.pose_diagnostics(state,
        Action(index, solved.pose, initial.pose, object_id=selected['object_id']))['fixed_scene_active_layer_above']
    candidate, reason = generator._finish_candidate(state, index,
        PlanarPlacement(initial.pose), initial, solved)
    assert candidate is not None and reason is None
    assert generator._finish_candidate(state, index, PlanarPlacement(initial.pose),
        initial, replace(solved, c_gap=1.))[1] == 'posegen_gap'
    generator.close()

    # Exercise both evaluator composition paths without starting native physics.
    monkeypatch.setattr(type(app), 'build_search_environments',
        lambda *_: {EvaluationFidelity.LONG: SimpleNamespace(close=lambda: None)})
    search = app.build_mcts(inventory, environments=app.build_search_environments(inventory),
                            pose_solver=SimpleNamespace())
    assert not hasattr(search.transition_evaluator, 'settled_floor_policy')
    search.close()
    parallel = app.build_parallel_transition_evaluator(inventory, 2)
    assert all(not hasattr(e, 'settled_floor_policy') for e in parallel.evaluators)
    parallel.close()




@pytest.fixture
def coverage_mismatch_scene():
    data = json.loads((ROOT / 'tests/fixtures/planning/excavator_floor_step11.json').read_text())
    app = ExcavatorApplication.from_repository(ROOT)
    ids = data['object_ids']
    inventory = app.load_inventory(ids, {i: np.array([0., 0., 0., 0., 0., 0., 1.]) for i in ids})
    state = State(ids, [ids.index(o['object_id']) for o in data['placed']],
        {o['object_id']: np.array(o['pose']) for o in data['placed']}, [], [])
    return app, inventory, state




def test_projected_sampler_tracks_floor_and_refreshes_on_state_change(
    coverage_mismatch_scene, monkeypatch,
):
    from stacking_common.actions import scored
    app, inventory, state = coverage_mismatch_scene
    generator = app.build_action_generator(inventory, SimpleNamespace())
    sampler = generator.planar_sampler
    policy = sampler.floor_policy
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




@pytest.mark.parametrize('physical_failure', [False, True])
def test_fixed_scene_upper_pose_still_requires_physical_settling(recorded_scene, physical_failure):
    from stacking_common.planning import PlacementPlanningResult, PlanningStatus
    app, inventory, state, candidates = recorded_scene
    selected = next(c for c in candidates if c['selected'])
    object_id = selected['object_id']
    action = Action(state.object_ids.index(object_id), np.array(selected['pose']),
        np.array(selected['init_pose']), object_id=object_id)
    backend = SimpleNamespace(reset=lambda _: None,
        settle_fixed=lambda *_: PlacementResult(
            {**state.object_poses, object_id: action.pose.copy()}, failed=physical_failure,
            diagnostics={'rejection_reason': 'unsupported'} if physical_failure else {}))
    calls = []
    def plan(problem):
        calls.append(problem)
        return PlacementPlanningResult(PlanningStatus.INFEASIBLE, reason='motion-planner-reached')
    coordinator = ExcavatorPlanningCoordinator(app, inventory,
        SimpleNamespace(plan=plan), placement_backend=backend)
    record = coordinator.attempt(state, action, step=12, attempt=1)
    if physical_failure:
        assert record.result.reason == 'fixed_scene_resettle_unsupported'
        assert not calls
    else:
        assert record.result.reason == 'motion-planner-reached'
        assert record.result.diagnostics['motion_planning_attempted']
