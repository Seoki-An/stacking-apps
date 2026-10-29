"""Root jumps explore new areas; descendant continuations stay local."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from stacking_common.actions import (CEMRootActionGenerator, CEMRootConfig,
    InitializedPlacement, PlacementSolveResult, PlacementActionGeneratorConfig, PlanarPlacement)
from stacking_common.actions.root_sampling import RootMixturePlanarSampler
from stacking_common.actions.scored import ScoredPlacementActionGenerator
from stacking_common.actions.connectivity import PreviousActionConnectivity
from stacking_common.domain import Action, State
from stacking_common.search import (ActionCandidate, ActionGenerationRequest,
    ActionGenerationResult, EvaluationFidelity, StackingEnvironmentEvaluator,
    ParallelTransitionEvaluator)
from test_previous_action_connectivity import setup
from test_composite_target_sampling import sampler, pose
from test_cem_pool_selection import config as cem_config


@pytest.mark.parametrize('fraction', [-.1, 1.1, float('nan'), float('inf')])
def test_invalid_root_fraction_is_rejected(fraction):
    from stacking_common.profiles.action import resolve_scored_sampling_settings
    inv, _, _ = setup(False)
    settings = {**sampler(inv).settings, 'root_global_fraction': fraction}
    with pytest.raises(ValueError, match='root_global_fraction'):
        resolve_scored_sampling_settings(settings)


@pytest.mark.parametrize('mode,fraction,enabled', [
    ('previous_action', None, False), ('previous_action', 0., False),
    ('previous_action', .2, True), ('active_floor', .2, False),
])
def test_root_mixture_requires_explicit_previous_action_opt_in(mode, fraction, enabled):
    from stacking_common.workflows.search import StackingSearchApplication
    settings = {'connectivity': mode}
    if fraction is not None:
        settings['root_global_fraction'] = fraction
    assert StackingSearchApplication.root_jumps_enabled(
        SimpleNamespace(sampling_settings=settings)) is enabled


def mixed_sampler(inv):
    base = sampler(inv)
    settings = {**base.settings, 'connectivity': 'previous_action', 'root_global_fraction': .2}
    return RootMixturePlanarSampler(inv, base.config, base.orientation_provider, settings,
                                   seed=base.seed, floor_policy=base.floor_policy)


@pytest.mark.parametrize('composite', [False, True])
def test_small_object_budgets_preserve_global_share_and_allow_far_proposals(composite):
    inv, state, obs = setup(composite)
    mixed = mixed_sampler(inv)
    proposals = [p for _ in range(50) for p in mixed.sample(inv, state, obs, 2, 2, 'lower_floor_first')]
    assert len(proposals) == 100
    assert sum(p.diagnostics['root_sampling_group'] == 'global' for p in proposals) == 20
    assert any(p.pose[0] < 0 for p in proposals)
    assert all(p.pose[0] > 0 for p in proposals if p.diagnostics['root_sampling_group'] == 'connected')
    assert all(mixed.allows_initial_pose(state, 2, p.pose) for p in proposals)


def test_empty_connected_pool_falls_back_to_global(monkeypatch):
    inv, state, obs = setup(False)
    mixed = mixed_sampler(inv)
    monkeypatch.setattr(mixed.connected, 'sample', lambda *args: ())
    proposals = mixed.sample(inv, state, obs, 2, 10, 'lower_floor_first')
    assert len(proposals) == 10
    assert all(p.diagnostics['root_sampling_group'] == 'global' for p in proposals)


def test_root_solved_jump_is_classified_instead_of_rejected():
    inv, state, obs = setup(False)
    sample = mixed_sampler(inv)
    cfg = PlacementActionGeneratorConfig(max_poses_per_object=4, initial_height_clearance=.1,
        initial_height_xy_padding=.01, posegen_gap_threshold=.001)
    generator = ScoredPlacementActionGenerator(inventory=inv, config=cfg, planar_sampler=sample,
        initializer=SimpleNamespace(initialize=lambda *args: InitializedPlacement(pose(1.), 0.)),
        pose_solver=SimpleNamespace(solve=lambda *args: PlacementSolveResult(pose(-1.), 0., 0.)))
    candidate, reason = generator.build_candidates(state, [(2, PlanarPlacement(pose(1.)))])[0]
    assert reason is None and not candidate.diagnostics['root_connected']
    assert candidate.diagnostics['previous_action_xy_gap'] == pytest.approx(1.6)


class Scorer:
    def score(self, state, obs, candidates):
        return [c.diagnostics['score'] for c in candidates]


def pool():
    return [ActionCandidate(Action(i, pose(i), pose(i), object_id=i+1),
        diagnostics={'root_connected': i < 10, 'score': 100-i if i < 10 else -i}) for i in range(20)]


def test_cem_preserves_low_scoring_jumps_in_population_elites_and_returned_slice():
    candidates = pool()
    class Generator:
        root_global_fraction = .2
        def generate(self, *args):
            return ActionGenerationResult(tuple(candidates), len(candidates), True)
    generator = Generator()
    config = cem_config(population=10, elite=5, keep=10, iterations=1, max_per_object=1)
    cem = CEMRootActionGenerator(generator, Scorer(), config)
    state = State(list(range(1,21)), [], {}, [], [])
    ranked = cem._rank(state, None, candidates)
    assert sum(not c.diagnostics['root_connected'] for c in cem._retain(ranked, 10)) == 2
    assert sum(not c.diagnostics['root_connected'] for c in cem._elite_parents(ranked, state)) == 1
    first = cem.generate(state, None, ActionGenerationRequest(5))
    second = cem.generate(state, None, ActionGenerationRequest(5, attempt_offset=5))
    assert sum(not c.diagnostics['root_connected'] for c in first.candidates) == 1
    assert not ({c.action.object_id for c in first.candidates} & {c.action.object_id for c in second.candidates})
    assert all(c.diagnostics['cem_score'] == c.diagnostics['score'] for c in first.candidates)


def test_root_jump_reservation_obeys_shared_object_cap():
    candidates = pool()
    # The two highest-scoring connected actions share IDs with jump candidates.
    for i in (0,1):
        candidates[10+i] = ActionCandidate(candidates[i].action,
            diagnostics={'root_connected':False, 'score':-10-i})
    cem = CEMRootActionGenerator(SimpleNamespace(root_global_fraction=.2), Scorer(),
                                 cem_config(population=10, elite=5, keep=10))
    selected = cem._retain(candidates, 10, object_limit=1)
    assert len(selected) == 10
    assert len({c.action.object_id for c in selected}) == 10
    assert sum(not c.diagnostics['root_connected'] for c in selected) == 2


@pytest.mark.parametrize('parallel', [False, True])
def test_root_exemption_rebinds_on_promotion_and_never_applies_to_descendants(parallel):
    inv, parent, _ = setup(False)
    action = Action(2, pose(-1.), pose(-1.), object_id=3)
    child = State(parent.object_ids, [0,1,2], {**parent.object_poses,3:pose(-1.)}, [], [action])
    result = SimpleNamespace(state=child, observation=None, reward=1., info={})
    env = SimpleNamespace(reset=lambda:None, restore=lambda s:None, step=lambda a:result,
                          step_batch=lambda states, actions:[result])
    def evaluator():
        return StackingEnvironmentEvaluator({EvaluationFidelity.LONG:env},
            connectivity_check=PreviousActionConnectivity(inv,.1), allow_root_jumps=True)
    eval = ParallelTransitionEvaluator([evaluator(),evaluator()]) if parallel else evaluator()
    try:
        # An unbound evaluator must not infer that an arbitrary scene is a root.
        assert eval.evaluate(parent,action,EvaluationFidelity.LONG).failed
        eval.begin_search(parent)
        assert not eval.evaluate_batch([parent],[action],EvaluationFidelity.LONG)[0].failed
        earlier = State(parent.object_ids, [0], {1:pose(-1.)}, [], [])
        eval.begin_search(earlier)
        assert eval.evaluate(parent,action,EvaluationFidelity.LONG).failed
        eval.begin_search(parent)  # New search after advancing the root.
        assert not eval.evaluate(parent,action,EvaluationFidelity.LONG).failed
        result.state.failed = True  # The exemption must never waive physical failure.
        assert eval.evaluate(parent,action,EvaluationFidelity.LONG).failed
    finally:
        if parallel:eval.executor.shutdown()
