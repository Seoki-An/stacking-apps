"""Tabletop shares excavator's proposal strategy with tabletop physical inputs."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from stacking_common.actions import PlacementSolveResult, ScanOrientationFilter
from stacking_common.domain import State
from stacking_common.search import ActionGenerationRequest
from stacking_common.visualization.heightmap import HeightMapRenderer
from stacking_excavator import ExcavatorApplication
from stacking_tabletop import TabletopApplication

ROOT = Path(__file__).resolve().parents[1]
POSE = np.array([0., 0., .1, 0., 0., 0., 1.])


def scene(app, placed):
    inventory = app.load_inventory([1, 2], {1: POSE, 2: POSE})
    pose = POSE.copy()
    pose[2] = -inventory[0].mesh_arrays()[0][:, 2].min()
    state = State([1, 2], [0] if placed else [], {1: pose} if placed else {}, [], [])
    renderer = HeightMapRenderer(inventory, (128, 128), app.environment_profile.height_map_margin)
    observation = SimpleNamespace(scene_height_map=renderer.render(state),
        height_map_xlim=renderer.xlim, height_map_ylim=renderer.ylim)
    return inventory, state, observation


@pytest.mark.parametrize("placed", [False, True])
@pytest.mark.parametrize("scan", [False, True])
def test_same_physical_inputs_match_excavator_maps_draws_and_mutation_lookup(placed, scan):
    tabletop = TabletopApplication.from_repository(ROOT)
    tabletop = replace(tabletop, action_generation_profile=replace(
        tabletop.action_generation_profile, pose_from_scan=scan))
    inventory, state, observation = scene(tabletop, placed)
    policy = tabletop.build_floor_policy(inventory)
    excavator = ExcavatorApplication.from_repository(ROOT)
    # Resolve excavator with the same physical profile and orientation bank,
    # while retaining its independently configured dimensionless score weights.
    excavator = replace(excavator,
        action_generation_profile=tabletop.action_generation_profile,
        mcts_profile=replace(excavator.mcts_profile, root_cem=tabletop.mcts_profile.root_cem),
        floor_settings={"coverage_mode": policy.coverage_mode,
            "bottom_tolerance": policy.bottom_tolerance, "fill_ratio": policy.fill_ratio,
            "grid_size": policy.grid_size, "max_layers": policy.max_layers,
            "minimum_contact_cells": policy.minimum_contact_cells},
        sampling_settings={**excavator.sampling_settings,
            "contact_gap_tolerance": tabletop.sampling_settings["contact_gap_tolerance"],
            "floor": tabletop.sampling_settings["floor"]})
    generators = [app.build_action_generator(inventory, SimpleNamespace(),
        orientation_provider=tabletop.build_orientation_provider()) for app in (tabletop, excavator)]
    try:
        samplers = [generator.planar_sampler for generator in generators]
        maps = [sampler.score_map(state, observation, 1) for sampler in samplers]
        assert maps[0]["candidate_mask"].any()
        for key, value in maps[0].items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, maps[1][key])
        populations = []
        for _ in range(2):
            draws = [sampler.sample(inventory, state, observation, 1, 8, "lower_floor_first")
                     for sampler in samplers]
            assert draws[0]
            np.testing.assert_array_equal([p.pose for p in draws[0]], [p.pose for p in draws[1]])
            np.testing.assert_array_equal([p.score for p in draws[0]], [p.score for p in draws[1]])
            populations.append(np.array([p.pose for p in draws[0]]))
            for proposal in draws[0]:
                mutation = proposal.pose.copy()
                mutation[:2] += [.001, -.001]
                assert (samplers[0].score_at_object_pose(observation, mutation, 1)
                        == samplers[1].score_at_object_pose(observation, mutation, 1))
        assert not np.array_equal(*populations)
    finally:
        for generator in generators:
            generator.close()


def test_tabletop_retry_draws_fresh_population_and_respects_deadline(monkeypatch):
    app = TabletopApplication.from_repository(ROOT)
    inventory, state, observation = scene(app, False)
    solver = SimpleNamespace(solve=lambda inventory, state, index, initial:
        PlacementSolveResult(initial.pose, 0., 0.))
    generators = [app.build_action_generator(inventory, solver) for _ in range(2)]
    # Isolate proposal/allocation behavior from solved scan-orientation acceptance.
    for generator in generators:
        generator.filters = tuple(f for f in generator.filters if not isinstance(f, ScanOrientationFilter))
    request = ActionGenerationRequest(max_candidates=6, sampling_priority="lower_floor_first")
    try:
        first, repeated = [generator.generate(state, observation, request) for generator in generators]
        assert first.attempts == len(first.candidates) == 6
        ids = [candidate.action.object_id for candidate in first.candidates]
        assert ids.count(1) == ids.count(2) == 3
        poses = [candidate.action.init_pose for candidate in first.candidates]
        np.testing.assert_array_equal(poses, [c.action.init_pose for c in repeated.candidates])
        retry = generators[0].generate(state, observation, replace(request, attempt_offset=10000))
        assert retry.attempts == len(retry.candidates) == 6
        assert not retry.exhausted
        assert not np.array_equal(poses, [c.action.init_pose for c in retry.candidates])
        assert all(c.diagnostics["source"] == "scored_floor" for c in retry.candidates)
        monkeypatch.setattr(generators[0].planar_sampler, "sample",
            lambda *a: pytest.fail("sampled after deadline"))
        assert generators[0].generate(state, observation, replace(request, deadline=0.)).attempts == 0
    finally:
        for generator in generators:
            generator.close()


def test_tabletop_yaml_weights_drive_scores_and_are_recorded(tmp_path):
    from stacking_common.artifacts.planning_context import _values
    from stacking_tabletop.config.action import load_action_generation_profile

    app = TabletopApplication.from_repository(ROOT)
    inventory, state, observation = scene(app, True)
    values = yaml.safe_load((ROOT / "apps/tabletop/config/action_generation.yml").read_text())
    values["planar"]["score"].update(open_area=.25, frontier=.5, planar_bias=1.)
    path = tmp_path / "action_generation.yml"
    path.write_text(yaml.safe_dump(values))
    changed = replace(app, action_generation_profile=load_action_generation_profile(path,
        max_poses_per_object=app.action_generation_profile.generator.max_poses_per_object,
        banned_object_ids=()))
    generators = [application.build_action_generator(inventory, SimpleNamespace())
                  for application in (app, changed)]
    try:
        baseline, actual = [generator.planar_sampler.score_map(state, observation, 1)
                            for generator in generators]
        valid = baseline["candidate_mask"]
        assert valid.any()
        np.testing.assert_array_equal(actual["candidate_mask"], valid)
        delta = sum((changed.sampling_settings["score"][key] - app.sampling_settings["score"][key])
                    * baseline[key][valid] for key in ("open_area", "frontier", "planar_bias"))
        assert np.any(delta != 0.)
        np.testing.assert_allclose(actual["scores"][valid], baseline["scores"][valid] + delta,
                                   rtol=0., atol=1e-12)
        assert _values(app.action_generation_profile) != _values(changed.action_generation_profile)
    finally:
        for generator in generators:
            generator.close()
