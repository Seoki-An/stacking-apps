"""Exhausted floors advance sampling and admission together without bypassing physics."""
from types import SimpleNamespace

import pytest

from stacking_common.actions import (
    PlacementActionGenerator, PlacementActionGeneratorConfig, PlanarPlacement,
    InitializedPlacement, PlacementSolveResult, CEMRootActionGenerator, CEMRootConfig,
)
from stacking_common.actions.filtering import ActiveLayerAdmissionFilter
from stacking_common.search import (
    MCTSNode, MCTSConfig, MCTSSearchRequest, MonteCarloTreeSearch, TransitionEvaluation,
)
from stacking_common.search.policies import ZeroValueEstimator
from test_active_layer_admission import scene
from test_action_generation import _pose


def generator(inv, policy, *, enabled=True, gap=0.):
    return PlacementActionGenerator(
        inv, PlacementActionGeneratorConfig(1, 0., 0., .001),
        SimpleNamespace(sample=lambda *args: [PlanarPlacement(_pose(z=.3))]),
        SimpleNamespace(initialize=lambda inv, state, index, proposal:
                        InitializedPlacement(proposal.pose, 0.)),
        SimpleNamespace(solve=lambda inv, state, index, initial:
                        PlacementSolveResult(initial.pose, c_feq=0., c_gap=gap)),
        filters=(ActiveLayerAdmissionFilter(policy),) if enabled else (),
    )


class Evaluator:
    def __init__(self, fail=False, clock=None):
        self.fail = fail
        self.clock = clock
        self.calls = []

    def evaluate(self, state, action, fidelity):
        self.calls.append(action.copy())
        if self.clock is not None:
            self.clock[0] = 20.
        next_state = state.copy()
        next_state.placed_object_indices.append(action.object_index)
        next_state.object_poses[action.object_id] = action.pose.copy()
        next_state.action_history.append(action.copy())
        return TransitionEvaluation(next_state, None, 1., fidelity,
            failed=self.fail, rejection_reason='physical_failure' if self.fail else None)


def setup(*, fail=False, depth=2):
    inv, policy, parent, _ = scene()
    deeper, base = generator(inv, policy), generator(inv, policy)
    scorer = SimpleNamespace(score=lambda state, obs, candidates: [0.] * len(candidates))
    cem = CEMRootActionGenerator(base, scorer,
        CEMRootConfig(population=2, elite=1, iterations=1, keep=2, max_per_object=2,
                      mutation_xy_std=0., mutation_yaw_std=0., fresh_fraction=1.,
                      duplicate_xy_threshold=.01, duplicate_yaw_threshold=.01))
    evaluator = Evaluator(fail)
    search = MonteCarloTreeSearch(
        MCTSConfig(max_depth=depth, max_children=2, root_max_children=2,
                   exploration_constant=1., discount=.9, rejected_root_retry_rounds=1),
        deeper, scorer, evaluator, ZeroValueEstimator(), root_action_generator=cem)
    return search, MCTSNode(state=parent), base, deeper, evaluator


@pytest.mark.parametrize('fail', [False, True])
def test_relaxation_retries_physics_and_restores_both_generators(fail):
    search, root, base, deeper, evaluator = setup(fail=fail)
    try:
        result = search.search(root, MCTSSearchRequest(iterations=4))
        assert result.floor_fill_fallback.attempted
        assert result.floor_fill_fallback.policy == 'floor_fill'
        assert result.floor_fill_fallback.selected is (not fail)
        assert (result.selected is None) is fail
        assert evaluator.calls
        assert all(a.diagnostics['floor_fill_fallback'] for a in evaluator.calls)
        assert all(not f.floor_policy.fallback_level(root.state) for g in (base, deeper) for f in g.filters
                   if isinstance(f, ActiveLayerAdmissionFilter))
        history = search.root_action_generator.statistics_history
        assert history[0].rejection_counts['above_active_layer'] > 0
        assert len(history) >= 2  # Strict and relaxed histories both retained.
        if not fail:
            assert any(c.children for c in root.children)
    finally:
        search.close()


def test_reused_relaxed_descendants_restart_with_strict_admission():
    search, root, _, _, _ = setup()
    try:
        first = search.search(root, MCTSSearchRequest(iterations=4))
        promoted = first.selected.detach()
        assert promoted.children
        old_children = set(promoted.children)
        search.search(promoted, MCTSSearchRequest(iterations=2))
        assert not old_children.intersection(promoted.children)
        assert search.root_action_generator.statistics_history[0].rejection_counts['above_active_layer'] > 0
    finally:
        search.close()


def test_fallback_context_restores_on_exception_and_keeps_posegen_gate():
    inv, policy, parent, _ = scene()
    gen = generator(inv, policy, gap=1.)
    original = gen.filters
    try:
        with pytest.raises(RuntimeError):
            with gen.floor_fill_fallback((parent,)) as enabled:
                assert enabled
                built = gen.build_candidates(parent, [(2, PlanarPlacement(_pose(z=.3)))])
                assert built[0][0] is None
                assert built[0][1] != 'above_active_layer'
                raise RuntimeError('cancel')
        assert gen.filters is original
    finally:
        gen.close()


def test_disabled_gate_does_not_enable_fallback():
    inv, policy, _, _ = scene()
    gen = generator(inv, policy, enabled=False)
    try:
        with gen.floor_fill_fallback() as enabled:
            assert not enabled
    finally:
        gen.close()


def test_expired_deadline_never_relaxes(monkeypatch):
    import stacking_common.search.mcts.search as module
    search, root, _, _, _ = setup()
    monkeypatch.setattr(module.time, 'monotonic', lambda: 20.)
    try:
        selected, _, result = search._run_proposal_fallback(root, MCTSSearchRequest(iterations=1), None, 10.)
        assert selected is None
        assert not result.attempted
    finally:
        search.close()


def test_strict_success_does_not_relax():
    search, root, _, _, _ = setup()
    root.state = scene()[3]  # Completed lower layer admits the next one.
    try:
        result = search.search(root, MCTSSearchRequest(iterations=2))
        assert result.selected is not None
        assert not result.floor_fill_fallback.attempted
        assert not result.selected.action.diagnostics.get('active_layer_fallback_level')
    finally:
        search.close()


def test_simulation_exception_restores_both_gates():
    search, root, base, deeper, evaluator = setup()
    originals = base.filters, deeper.filters

    def crash(*args):
        raise RuntimeError('simulation failed')

    evaluator.evaluate = crash
    try:
        with pytest.raises(RuntimeError, match='simulation failed'):
            search.search(root, MCTSSearchRequest(iterations=2))
        assert base.filters is originals[0]
        assert deeper.filters is originals[1]
    finally:
        search.close()


@pytest.mark.parametrize('fail_descendant', [False, True])
@pytest.mark.parametrize('batch_size', [1, 16])
def test_blocked_descendant_gets_one_pool_without_repeating_empty_backups(fail_descendant, batch_size):
    from dataclasses import replace
    from stacking_common.workflows.mcts_reporting import search_statistics
    search, root, base, deeper, evaluator = setup()
    search.config = replace(search.config, evaluation_batch_size=batch_size)
    base.planar_sampler.sample = lambda *args: [PlanarPlacement(_pose(x=-.1, z=.1))]
    deeper.planar_sampler.sample = lambda *args: [PlanarPlacement(_pose(x=-.1, z=.3))]
    original_evaluate = evaluator.evaluate

    def evaluate(state, action, fidelity):
        evaluator.fail = fail_descendant and len(state.placed_object_indices) > 1
        return original_evaluate(state, action, fidelity)

    evaluator.evaluate = evaluate
    try:
        result = search.search(root, MCTSSearchRequest(iterations=100))
        assert result.selected is not None
        assert not result.floor_fill_fallback.attempted  # Root itself is feasible.
        retries = [n for n in root.children if n.info.get('descendant_floor_retry_attempted')]
        assert retries
        assert all(n.info['descendant_floor_retry_attempts'] <= search.config.generation_batch_size
                   for n in retries)
        assert all(n.expansion_exhausted for n in retries)
        relaxed = [c for n in retries for c in n.children
                   if c.action.diagnostics.get('active_layer_fallback_level')]
        assert relaxed
        recorded = search_statistics(result, 0., search, step=2)['descendant_floor_retry']
        assert recorded['nodes'] == len(retries)
        assert recorded['evaluated'] == len(relaxed)
        assert recorded['accepted'] == (0 if fail_descendant else len(relaxed))
        assert all(c.failed == fail_descendant for c in relaxed)
        if fail_descendant:
            assert result.completed_iterations < 100
            # Only incoming reward and the newly observed child failure are
            # backed up; no repeated optimistic estimate of an empty branch.
            assert all(n.visits <= 2 for n in root.evaluated_children())
        else:
            assert any(c.selectable and c.depth == 2 for c in relaxed)
        assert not any(f.floor_policy.fallback_level(root.state) for f in deeper.filters if isinstance(f, ActiveLayerAdmissionFilter))
    finally:
        search.close()


def test_zero_work_generator_does_not_consume_all_iterations():
    from stacking_common.search import ActionGenerationResult
    search, root, base, deeper, _ = setup()
    base.planar_sampler.sample = lambda *args: [PlanarPlacement(_pose(x=-.1, z=.1))]
    deeper.generate_batch = lambda states, observations, requests: tuple(
        ActionGenerationResult((), attempts=0, exhausted=False) for _ in states)
    try:
        result = search.search(root, MCTSSearchRequest(iterations=100))
        assert result.selected is not None
        assert result.completed_iterations < 10
        assert all(n.visits == 1 for n in root.evaluated_children())
        assert all(n.expansion_exhausted for n in root.evaluated_children())
        assert not any(n.info.get('descendant_floor_retry_attempted') for n in root.children)
    finally:
        search.close()
