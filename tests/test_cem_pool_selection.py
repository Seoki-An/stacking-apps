"""CEM must keep the best eligible representative before limiting its pool."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from stacking_common.actions import CEMRootActionGenerator, CEMRootConfig
from stacking_common.domain import Action, State
from stacking_common.search import ActionCandidate, ActionGenerationRequest, ActionGenerationResult, ExecutionRejections


def candidate(oid, x, score):
    pose = np.array([x, 0., .1, 0., 0., 0., 1.])
    return ActionCandidate(Action(oid - 1, pose, pose, object_id=oid), diagnostics={"test_score": score})


class Scorer:
    def score(self, state, observation, candidates):
        return [c.diagnostics["test_score"] for c in candidates]


class Generator:
    planar_sampler = SimpleNamespace()

    def __init__(self, *batches):
        self.batches = iter(batches)
        self.requests = []
        self.parents = []

    def generate(self, state, observation, request):
        self.requests.append(request)
        batch = next(self.batches)
        return ActionGenerationResult(tuple(batch), len(batch), False)

    def select_proposals(self, state, observation, request):
        # Fresh proposals carry their prepared candidate through build_candidates.
        self.requests.append(request)
        return tuple((c.action.object_index, c) for c in next(self.batches)), False

    def build_candidates(self, state, proposals):
        built = []
        for i, (index, p) in enumerate(proposals):
            if isinstance(p, ActionCandidate):
                built.append((p, None))
                continue
            self.parents.append(index + 1)
            built.append((ActionCandidate(Action(index, p.pose, p.pose, object_id=index + 1),
                          diagnostics={"test_score": 10. + i}), None))
        return built


def config(**kwargs):
    return replace(CEMRootConfig(population=2, elite=1, iterations=2, keep=2,
        mutation_xy_std=.001, mutation_yaw_std=0., fresh_fraction=0.,
        duplicate_xy_threshold=.03, duplicate_yaw_threshold=.1), **kwargs)


def state():
    return State([1, 2, 3, 4], [], {}, [], [])


def test_better_mutation_replaces_nearby_parent_before_population_limit():
    engine = CEMRootActionGenerator(Generator([candidate(1, 0., 1.), candidate(2, 1., 0.)]), Scorer(), config())
    result = engine.generate(state(), None, ActionGenerationRequest(2))
    assert [c.diagnostics["cem_score"] for c in result.candidates] == [11., 0.]
    assert engine.last_statistics.duplicate_rejections == 2


@pytest.mark.parametrize("scores", [(1., 2.), (2., 1.), (2., 2.)])
def test_initial_duplicates_keep_best_score_and_stable_ties(scores):
    batch = [candidate(1, 0., scores[0]), candidate(1, .01, scores[1])]
    engine = CEMRootActionGenerator(Generator(batch), Scorer(), config(iterations=1))
    result = engine.generate(state(), None, ActionGenerationRequest(2))
    assert len(result.candidates) == 1
    assert result.candidates[0].diagnostics["cem_score"] == max(scores)
    assert result.candidates[0].action.pose[0] == (0. if scores[0] >= scores[1] else .01)


@pytest.mark.parametrize("exclusion", ["object", "action", "retry"])
def test_exclusions_precede_population_limit_and_do_not_starve_alternatives(exclusion):
    rejected = candidate(1, 0., 100.)
    base = Generator([rejected, candidate(2, 1., 1.)], [candidate(3, 2., 2.), candidate(4, 3., 3.)])
    engine = CEMRootActionGenerator(base, Scorer(), config(fresh_fraction=1.))
    exclusions = ExecutionRejections()
    if exclusion == "object":
        exclusions = ExecutionRejections(object_ids={1})
    elif exclusion == "action":
        exclusions = ExecutionRejections(actions=(rejected.action,))
    else:
        engine.begin_retry((rejected.action,))
    result = engine.generate(state(), None, ActionGenerationRequest(2, execution_rejections=exclusions))
    assert [c.action.object_id for c in result.candidates] == [4, 3]
    reason = "retry_excluded_duplicate" if exclusion == "retry" else "execution_rejected_" + exclusion
    assert engine.last_statistics.rejection_counts[reason] == 1
    # Build acceptance and attempt counts still describe physical generation.
    assert [r.accepted for r in engine.last_statistics.rounds] == [2, 2]
    assert engine.last_statistics.attempted == 4
    assert all(r.execution_rejections == exclusions for r in base.requests)


def test_execution_rejected_object_cannot_become_mutation_parent():
    base = Generator([candidate(1, 0., 100.), candidate(2, 1., 1.)])
    engine = CEMRootActionGenerator(base, Scorer(), config())
    result = engine.generate(state(), None, ActionGenerationRequest(
        2, execution_rejections=ExecutionRejections(object_ids={1})))
    assert base.parents == [2, 2]
    assert result.candidates and all(c.action.object_id == 2 for c in result.candidates)


def test_adaptive_cap_counts_only_available_objects():
    base = Generator([candidate(1, i, 10. - i) for i in range(4)])
    base.config = SimpleNamespace(banned_object_ids={4})
    engine = CEMRootActionGenerator(base, Scorer(), config(
        population=4, keep=4, elite=4, iterations=1, max_per_object=1))
    result = engine.generate(state(), None, ActionGenerationRequest(
        4, execution_rejections=ExecutionRejections(object_ids={2, 3})))
    assert len(result.candidates) == 4
    assert len(engine._elite_parents(result.candidates, state(), excluded_object_ids={2, 3})) == 4
