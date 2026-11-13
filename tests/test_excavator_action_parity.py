"""Legacy score-mode fixtures and explicit separation from tabletop policies."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import stacking_core as core
import yaml

from stacking_common.actions import MeshHeightInitializer, CEMRootActionGenerator
from stacking_common.actions.filtering import PosegenFeasibilityFilter, TargetContainmentFilter
from stacking_common.actions.ply_height import PlyHeightInitializer, read_ply_vertices
from stacking_common.actions.scored import ScoredPlanarPoseSampler, TargetXYOverlapFilter
from stacking_common.actions.types import PlanarPlacement, InitializedPlacement, PlacementSolveResult
from stacking_common.artifacts.planning_context import _values
from stacking_common.domain import State
from stacking_common.environment import VoxelOverlapEvaluator
from stacking_common.search.types import ActionGenerationRequest
from stacking_excavator import ExcavatorApplication
from stacking_tabletop import TabletopApplication

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/legacy_planner/scored_sampling.npz"
META = json.loads(FIXTURE.with_suffix(".json").read_text())


def inputs(placed=False):
    app = ExcavatorApplication.from_repository(ROOT)
    ids = META["object_ids"]
    inventory = app.load_inventory(ids, {i: np.array([0., 0., 0., 0., 0., 0., 1.]) for i in ids})
    state = State(ids, [0] if placed else [],
                  {55: np.array(META["placed_pose"])} if placed else {}, [], [])
    return app, inventory, state


def test_unified_planar_weights_change_excavator_scores(tmp_path):
    app, inventory, state = inputs(placed=True)
    config = tmp_path / "apps/excavator/config"
    shutil.copytree(ROOT / "apps/excavator/config", config)
    path = config / "action_generation.yml"
    values = yaml.safe_load(path.read_text())
    values["planar"]["score"].update(fill_area=.25, planar_bias=1.)
    path.write_text(yaml.safe_dump(values))
    changed = ExcavatorApplication.from_repository(tmp_path)
    assert "score_weights" not in changed.floor_settings
    with np.load(FIXTURE) as recorded:
        observation = SimpleNamespace(scene_height_map=recorded["placed_height"],
            height_map_xlim=np.array(META["xlim"]), height_map_ylim=np.array(META["ylim"]))
    maps = []
    for application in (app, changed):
        sampler = ScoredPlanarPoseSampler(inventory, application.action_generation_profile.planar,
            application.build_orientation_provider(), application.sampling_settings,
            floor_policy=application.build_floor_policy(inventory))
        maps.append(sampler.score_map(state, observation, META["candidate_index"]))
    baseline, actual = maps
    np.testing.assert_array_equal(actual["candidate_mask"], baseline["candidate_mask"])
    valid = baseline["candidate_mask"]
    # Fill coverage now depends on rotation, so changing its weight can also
    # change the winning orientation shown at an aggregate map cell.
    for data, application in zip(maps, (app, changed)):
        expected = sum(application.sampling_settings["score"][key] * data[key][valid]
                       for key in ("support_likelihood", "fill_area", "target_boundary", "planar_bias"))
        expected += application.sampling_settings["score"]["height"] * data["height_term"][valid]
        expected -= (application.sampling_settings["score"]["previous_action_distance"]
                     * data["previous_action_distance"][valid])
        np.testing.assert_allclose(data["scores"][valid], expected, rtol=0., atol=1e-12)
    valid = valid & (baseline["orientation_index"] == actual["orientation_index"])
    assert valid.any()
    delta = sum((changed.sampling_settings["score"][key] - app.sampling_settings["score"][key])
                * baseline[key][valid] for key in ("fill_area", "planar_bias"))
    assert np.any(delta != 0.)
    np.testing.assert_allclose(actual["scores"][valid], baseline["scores"][valid] + delta,
                               rtol=0., atol=1e-12)


@pytest.mark.parametrize("obsolete", ["floor", "scored_sampling", "score_weights"])
def test_obsolete_score_blocks_are_rejected_instead_of_ignored(tmp_path, obsolete):
    config = tmp_path / "apps/excavator/config"
    shutil.copytree(ROOT / "apps/excavator/config", config)
    path = config / "action_generation.yml"
    values = yaml.safe_load(path.read_text())
    if obsolete == "score_weights":
        values["planar"]["floor"][obsolete] = [-3., 2., 0., 2., 0., 1.5]
    else:
        values[obsolete] = {}
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="planar.score"):
        ExcavatorApplication.from_repository(tmp_path)


@pytest.mark.parametrize("placed", [False, True])
def test_score_map_reuse_preserves_fresh_draws_and_mutation_scores(placed, monkeypatch):
    from stacking_common.actions import scored
    app, inventory, state = inputs(placed)
    with np.load(FIXTURE) as recorded:
        observation = SimpleNamespace(scene_height_map=recorded["placed_height" if placed else "empty_height"],
            height_map_xlim=np.array(META["xlim"]), height_map_ylim=np.array(META["ylim"]))
    cached, uncached = [ScoredPlanarPoseSampler(inventory, app.action_generation_profile.planar,
        app.build_orientation_provider(), app.sampling_settings, seed=META["seed"]) for _ in range(2)]
    original = scored.floor.score_xy_debug_map
    calls = []
    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(scored.floor, 'score_xy_debug_map', record)
    draws = []
    index = META['candidate_index']
    for _ in range(3):
        before = len(calls)
        actual = cached.sample(inventory, state, observation, index, 32, 'lower_floor_first')
        assert len(calls) - before == (1 if not draws else 0)
        uncached._score_context = None
        expected = uncached.sample(inventory, state, observation, index, 32, 'lower_floor_first')
        np.testing.assert_array_equal([p.pose for p in actual], [p.pose for p in expected])
        assert [p.diagnostics for p in actual] == [p.diagnostics for p in expected]
        mutation = actual[0].pose.copy()
        mutation[0] += .031
        assert cached.score_at_object_pose(observation, mutation, index) == uncached.score_at_object_pose(observation, mutation, index)
        draws.append(np.array([p.pose for p in actual]))
    assert not np.array_equal(draws[0], draws[1])
    # Equivalent reconstructed inputs can reuse maps, with current lookup identity.
    duplicate = SimpleNamespace(**{k: v.copy() for k, v in vars(observation).items()})
    before = len(calls)
    cached.score_map(state.copy(), duplicate, index)
    assert len(calls) == before
    assert cached.score_at_object_pose(duplicate, mutation, index) == uncached.score_at_object_pose(observation, mutation, index)


@pytest.mark.parametrize('change', ['height', 'bounds', 'placed', 'pose', 'settings', 'orientation'])
def test_score_map_cache_invalidates_changed_inputs(change, monkeypatch):
    from stacking_common.actions import scored
    app, inventory, state = inputs(True)
    with np.load(FIXTURE) as recorded:
        observation = SimpleNamespace(scene_height_map=recorded['placed_height'],
            height_map_xlim=np.array(META['xlim']), height_map_ylim=np.array(META['ylim']))
    sampler = ScoredPlanarPoseSampler(inventory, app.action_generation_profile.planar,
        app.build_orientation_provider(), app.sampling_settings, seed=META['seed'])
    original = scored.floor.score_xy_debug_map
    calls = []
    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(scored.floor, 'score_xy_debug_map', record)
    index = META['candidate_index']
    sampler.score_map(state, observation, index)
    sampler.score_map(state, observation, index)
    assert len(calls) == 1
    if change == 'height':
        observation.scene_height_map[0, 0] += .1
    elif change == 'bounds':
        observation.height_map_xlim[0] -= 1
    elif change == 'placed':
        state.placed_object_indices.clear()
    elif change == 'pose':
        state.object_poses[55][0] += .1
    elif change == 'settings':
        sampler.settings['score']['height'] -= 1.
    else:
        quaternions = sampler.quaternions(state, observation, index)
        monkeypatch.setattr(sampler, 'quaternions', lambda *args: quaternions[:1])
    actual = sampler.score_map(state, observation, index)
    assert len(calls) == 2
    sampler._score_context = None
    expected = sampler.score_map(state, observation, index)
    np.testing.assert_array_equal(actual['scores'], expected['scores'])
    np.testing.assert_array_equal(actual['candidate_mask'], expected['candidate_mask'])


@pytest.mark.parametrize("feq,gap,converged,legacy,tabletop", [
    (.5, .001, False, True, False),
    (1., .001, True, False, True),
    (1.01, .001, True, False, False),
    (.5, .005, True, True, True),
    (.5, .0051, True, False, False),
])
def test_feasibility_acceptance_preserves_platform_semantics(feq, gap, converged, legacy, tabletop):
    solved = PlacementSolveResult(np.array([0., 0., 0., 0., 0., 0., 1.]), feq, gap, converged=converged)
    new = PosegenFeasibilityFilter(.005, 1., require_solver_convergence=False,
                                   reject_equilibrium_at_threshold=True)
    old = PosegenFeasibilityFilter(.005, 1.)
    assert new.evaluate(None, None, 0, None, solved).accepted is legacy
    assert old.evaluate(None, None, 0, None, solved).accepted is tabletop


def test_xy_overlap_accepts_overhang_before_volumetric_validation():
    app, inventory, state = inputs()
    # Center at the target's right edge: XY overlap exists but most of the
    # stone is outside. The legacy build gate retains it for downstream scoring.
    pose = np.array([1.2, 0., .5, 0., 0., 0., 1.])
    solved = PlacementSolveResult(pose, 0., 0.)
    initial = InitializedPlacement(pose, 0.)
    assert TargetXYOverlapFilter().evaluate(inventory, state, 2, initial, solved).accepted
    containment = TargetContainmentFilter(VoxelOverlapEvaluator(inventory.target, resolution=.03), .7)
    assert not containment.evaluate(inventory, state, 2, initial, solved).accepted
    solved = PlacementSolveResult(pose + np.array([10., 0., 0., 0., 0., 0., 0.]), 0., 0.)
    assert not TargetXYOverlapFilter().evaluate(inventory, state, 2, initial, solved).accepted


def test_ply_height_uses_surface_vertices_and_pads_both_footprints(tmp_path):
    # Two unit boxes separated by .18 m in XY. Legacy's two .12 m margins
    # overlap; tabletop's existing single .12 m margin does not.
    from test_action_generation import _inventory
    inventory = _inventory()
    points = np.array([[-.5, -.5, -.5], [.5, .5, .5]])
    for obj in inventory:
        (tmp_path / f"model_{obj.object_id}_mesh.ply").write_text(
            "ply\nformat ascii 1.0\nelement vertex 2\nproperty double x\nproperty double y\nproperty double z\nend_header\n"
            "-.5 -.5 -.5\n.5 .5 .5\n")
    state = State(list(inventory.object_ids), [0], {10: np.array([1.18, 0., 2., 0., 0., 0., 1.])}, [], [])
    proposal = PlanarPlacement(np.array([0., 0., 0., 0., 0., 0., 1.]))
    initializer = PlyHeightInitializer(tmp_path, clearance=.15, xy_padding=.12)
    actual = initializer.initialize(inventory, state, 1, proposal)
    assert actual.pose[2] == pytest.approx(3.15)
    assert actual.support_object_indices == (0,)
    np.testing.assert_array_equal(initializer.vertices(inventory[1]), points)
    assert MeshHeightInitializer(.15, .12).initialize(inventory, state, 1, proposal).pose[2] == pytest.approx(.17)
    assert read_ply_vertices(ROOT / "apps/excavator/assets/objects/model_55_mesh.ply").shape == (2562, 3)


@pytest.mark.parametrize("radius,target_only,expected", [(None, False, []), (.5, False, [1]), (.5, True, [0, 1])])
def test_boundary_selection_reaches_native_problem(radius, target_only, expected):
    app, inventory, _ = inputs()
    solver = app.build_pose_solver(inventory)
    solver.profile = replace(solver.profile, boundary_radius=radius, placed_as_boundaries=target_only)
    state = State(list(inventory.object_ids), [0, 1], {
        55: np.array([1.5, 0., .5, 0., 0., 0., 1.]),  # Exactly at the radius stays dynamic.
        50: np.array([1.6, 0., .5, 0., 0., 0., 1.]),
    }, [], [])
    class Captured(Exception):
        pass
    class CoreProxy:
        def __getattr__(self, name):
            return getattr(core, name)
        def PosegenProblem(self, *args, **kwargs):
            assert kwargs["boundaries"] == [solver.scene_builder.entity_id(i) for i in expected]
            raise Captured
    solver.core = CoreProxy()
    with pytest.raises(Captured):
        solver.solve(inventory, state, 2, InitializedPlacement(np.array([0., 0., .5, 0., 0., 0., 1.]), 0.))


def test_tabletop_scored_composition_records_aligned_filtering_policy():
    from test_action_generation import _inventory
    tabletop = TabletopApplication.from_repository(ROOT)
    inventory = _inventory()
    generator = tabletop.build_action_generator(inventory, SimpleNamespace(),
        filters=tabletop.build_target_filters(inventory, SimpleNamespace()))
    assert type(generator.planar_sampler) is ScoredPlanarPoseSampler
    assert type(generator.initializer) is MeshHeightInitializer
    assert any(isinstance(f, TargetContainmentFilter) for f in generator.filters)
    assert generator.config.require_solver_convergence is False
    assert generator.config.reject_equilibrium_at_threshold is True
    assert tabletop.posegen_profile.boundary_radius is None
    # Changed filter semantics must be explicit in recorded planning contexts.
    assert _values(tabletop.action_generation_profile.generator)["require_solver_convergence"] is False
    assert _values(tabletop.action_generation_profile.generator)["reject_equilibrium_at_threshold"] is True
    assert "boundary_radius" not in _values(tabletop.posegen_profile)
    excavator = ExcavatorApplication.from_repository(ROOT)
    assert not excavator.environment_profile.observation.banned_object_ids
    assert _values(excavator.posegen_profile)["boundary_radius"] == .5
    generator.close()


def test_fresh_population_allocation_retries_and_deadline(monkeypatch):
    app, inventory, state = inputs()
    with np.load(FIXTURE) as recorded:
        observation = SimpleNamespace(scene_height_map=recorded["empty_height"],
            height_map_xlim=np.array(META["xlim"]), height_map_ylim=np.array(META["ylim"]))
    solver = SimpleNamespace(solve=lambda inventory, state, index, initial:
                             PlacementSolveResult(initial.pose, 0., 0.))
    first = app.build_action_generator(inventory, solver)
    second = app.build_action_generator(inventory, solver)
    request = ActionGenerationRequest(max_candidates=6)
    try:
        initial = first.generate(state, observation, request)
        repeated = second.generate(state, observation, request)
        assert initial.attempts == len(initial.candidates) == 6
        np.testing.assert_array_equal([c.action.pose for c in initial.candidates],
                                      [c.action.pose for c in repeated.candidates])
        # This policy draws a fresh population rather than paging through the
        # old deterministic stream. A large retry offset must not starve it.
        retry = first.generate(state, observation, replace(request, attempt_offset=10000))
        assert retry.attempts == len(retry.candidates) == 6
        assert not retry.exhausted
        assert not np.array_equal([c.action.pose for c in initial.candidates],
                                  [c.action.pose for c in retry.candidates])
        monkeypatch.setattr(first.planar_sampler, "sample", lambda *a: pytest.fail("sampling past deadline"))
        assert first.generate(state, observation, replace(request, deadline=0.)).attempts == 0
    finally:
        first.close()
        second.close()


def test_cem_mutations_recompute_the_parent_objects_score(monkeypatch):
    app, inventory, state = inputs()
    with np.load(FIXTURE) as recorded:
        observation = SimpleNamespace(scene_height_map=recorded["empty_height"],
            height_map_xlim=np.array(META["xlim"]), height_map_ylim=np.array(META["ylim"]))
    solver = SimpleNamespace(solve=lambda inventory, state, index, initial:
                             PlacementSolveResult(initial.pose, 0., 0.))
    generator = app.build_action_generator(inventory, solver)
    scored = []
    score_at_object_pose = generator.planar_sampler.score_at_object_pose
    def record(observation, pose, index):
        result = score_at_object_pose(observation, pose, index)
        scored.append((index, result))
        return result
    monkeypatch.setattr(generator.planar_sampler, "score_at_object_pose", record)
    scorer = SimpleNamespace(score=lambda state, observation, candidates: [0.] * len(candidates))
    cem = CEMRootActionGenerator(generator, scorer,
        replace(app.mcts_profile.root_cem, population=12, elite=4, keep=6, iterations=2))
    try:
        result = cem.generate(state, observation, ActionGenerationRequest(max_candidates=6))
        assert result.candidates
        assert scored and all(0 <= i < len(inventory) and np.all(np.isfinite(values)) for i, values in scored)
        assert sum(round.mutation_attempts for round in cem.last_statistics.rounds) > 0
    finally:
        generator.close()


@pytest.mark.parametrize("support_z,heights,count,expected_lower", [
    (0., [0., .1, .8, .9], 2, 2),
    (0., [0., .1, .8, .9], 4, 2),
    (0., [.8, .9], 2, 0),
    (.8, [.8, 1., 1.5, 1.6], 2, 2),
])
def test_scored_priority_spends_budget_on_active_floor_before_higher_scores(
    monkeypatch, support_z, heights, count, expected_lower,
):
    from stacking_common.actions import scored
    app, inventory, state = inputs(placed=True)
    sampler = ScoredPlanarPoseSampler(inventory, app.action_generation_profile.planar,
        app.build_orientation_provider(), app.sampling_settings, seed=17)
    ceiling = support_z + .2
    zeros = np.zeros(len(heights))
    candidates = scored.floor.CandidateColumns(dict(
        xy=np.column_stack((np.arange(len(heights), dtype=float), zeros)),
        height=np.array(heights), score=np.where(np.array(heights) > ceiling, 100., 0.),
        orientation_index=np.zeros(len(heights), dtype=int), support_likelihood=zeros,
        placement_height=zeros, target_boundary=zeros, normalized_score=zeros))
    monkeypatch.setattr(sampler, "score_map", lambda *args: {"candidates": candidates})
    monkeypatch.setattr(scored.floor, "active_floor_context",
        lambda *args: {"support_z": support_z, "bottom_tol": .2})
    observation = SimpleNamespace(height_map_xlim=[-2., 2.], height_map_ylim=[-2., 2.])
    proposals = sampler.sample(inventory, state, observation, 2, count, "lower_floor_first")
    indices = [int(proposal.pose[0]) for proposal in proposals]
    assert len(indices) == count
    assert len(set(indices)) == count
    assert [heights[i] <= ceiling for i in indices] == (
        [True] * expected_lower + [False] * (count - expected_lower))


@pytest.mark.parametrize("change", ["height", "bounds", "placed", "pose", "settings", "grid"])
def test_state_height_maps_shared_between_stones_and_invalidated(change, monkeypatch):
    from stacking_common.actions import scored
    app, inventory, state = inputs(True)
    with np.load(FIXTURE) as recorded:
        observation = SimpleNamespace(scene_height_map=recorded["placed_height"].copy(),
            height_map_xlim=np.array(META["xlim"]), height_map_ylim=np.array(META["ylim"]))
    sampler = ScoredPlanarPoseSampler(inventory, app.action_generation_profile.planar,
        app.build_orientation_provider(), app.sampling_settings, seed=META["seed"])
    original = scored.floor.build_state_map_bundle
    bundles = []
    def record(*args, **kwargs):
        bundle = original(*args, **kwargs)
        bundles.append(bundle)
        return bundle
    monkeypatch.setattr(scored.floor, "build_state_map_bundle", record)
    sampler.score_map(state, observation, 1)
    scores = sampler.score_map(state, observation, 2)
    assert len(bundles) == 1
    # A standalone map must be identical to the map using another stone's bundle.
    expected = scored.floor.score_xy_debug_map(
        sampler.view(state, observation), sampler.state_view(state), 2,
        scene_height_map=observation.scene_height_map,
        support_kernels=sampler._kernels[sampler._score_keys[2]],
        footprint_bounds=sampler._footprint_bounds[sampler._score_keys[2]],
        footprint_polygons=sampler._footprint_polygons[sampler._score_keys[2]],
    )
    for key in ("scores", "height", "placement_height", "orientation_index"):
        np.testing.assert_array_equal(scores[key], expected[key])
    if change == "height":
        observation.scene_height_map[5:10, 5:10] += .2
    elif change == "bounds":
        observation.height_map_xlim[0] -= 1
    elif change == "placed":
        state.placed_object_indices.clear()
    elif change == "pose":
        state.object_poses[55][0] += .1
    elif change == "settings":
        sampler.settings["target_mask_margin"] = .1
    else:
        sampler.config = replace(sampler.config, grid_shape=(12, 16))
    before = len(bundles)
    actual = sampler.score_map(state, observation, 2)
    assert len(bundles) == before + 1
    assert sampler._state_map_bundle is not bundles[0]
    sampler._score_context = None
    rebuilt = sampler.score_map(state, observation, 2)
    for key in ("scores", "height", "placement_height", "orientation_index"):
        np.testing.assert_array_equal(actual[key], rebuilt[key])
