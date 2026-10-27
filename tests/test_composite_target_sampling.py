"""Composite proposals, per-region floors, and fresh/optimized membership."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stacking_common.actions import (
    ActiveFloorFilter, CEMRootActionGenerator, CEMRootConfig, MappingOrientationProvider,
    MeshHeightInitializer, PlacementActionGeneratorConfig, PlacementSolveResult,
    PlanarPlacement, PlanarSamplingConfig,
)
from stacking_common.actions.floor import ActiveFloorPolicy
from stacking_common.actions.ply_height import PlyHeightInitializer
from stacking_common.actions.scored import ScoredPlanarPoseSampler, ScoredPlacementActionGenerator, TargetXYOverlapFilter
from stacking_common.actions.target_regions import TargetRegions, anchor_components
from stacking_common.domain import (
    Action, CompositeTargetGeometry, ConvexGeometry, DirectionalTaper, ObjectInventory,
    StackObject, State, TargetComponent, TargetGeometry,
)
from stacking_common.search import ActionCandidate, ActionGenerationRequest


def pose(x=0., y=0., z=0., yaw=0.):
    return np.r_[x, y, z, Rotation.from_euler("z", yaw).as_quat()]


def box(w=2., l=2., h=3.):
    value = TargetGeometry(w, l, h, np.zeros(3), DirectionalTaper(), 0.)
    return ConvexGeometry(value.points, value.faces)


def target(separation=3.):
    return CompositeTargetGeometry((TargetComponent("west", box(), pose(-separation)),
                                    TargetComponent("east", box(), pose(separation, yaw=.25))))


def inventory(geometry=None, shapes=None):
    geometry = target() if geometry is None else geometry
    shapes = [box(.25, .25, .25)] * 3 if shapes is None else shapes
    objects = [StackObject(index + 1, SimpleNamespace(mass=1., inertia=np.eye(3)), [shape])
               for index, shape in enumerate(shapes)]
    return ObjectInventory(objects, {obj.object_id: pose(20.) for obj in objects}, geometry)


def state(inv, poses=()):
    return State(list(inv.object_ids), list(range(len(poses))),
                 {inv[i].object_id: value for i, value in enumerate(poses)}, [], [])


def policy(inv, **kwargs):
    return ActiveFloorPolicy(inv, bottom_tolerance=.05, fill_ratio=.8,
                             grid_size=32, coverage_mode="projected_footprint", **kwargs)


def sampler(inv, *, yaw_angles=(0., np.pi / 2), seed=3, floor_policy=None):
    settings = {
        "floor": dict(bottom_tolerance=.05, fill_ratio=.8, grid_size=32,
                      coverage_mode="projected_footprint", footprint_scale=.9),
        "contact_gap_tolerance": .03, "target_mask_margin": 0.,
        "score": dict(height=-1.5, support_likelihood=2., fill_area=1., target_boundary=.2),
    }
    return ScoredPlanarPoseSampler(inv, PlanarSamplingConfig((24, 24), yaw_angles, 1.),
        MappingOrientationProvider({oid: [0., 0., 0., 1.] for oid in inv.object_ids}), settings,
        seed=seed, floor_policy=policy(inv) if floor_policy is None else floor_policy)


def action(inv, index, value):
    return Action(index, value, value, object_id=inv[index].object_id)


def test_separated_regions_seed_independently_and_reject_gap():
    inv = inventory()
    floor = policy(inv)
    current = state(inv, [pose(-3.7, -.7)])
    reason, info = floor.active_floor_rejection(current, action(inv, 1, pose(3., 0.)))
    assert reason is None
    assert info["target_region_id"] == ("east",)
    assert info["fixed_scene_active_layer_occupancy"] == 0.
    assert floor.active_floor_rejection(current, action(inv, 1, pose(-2.3, .6)))[0] == "disconnected_active_layer"
    assert floor.active_floor_rejection(current, action(inv, 1, pose(0.)))[0] == "outside_target_region"


def test_regions_advance_their_own_layers():
    inv = inventory(shapes=[box(2., 2., .5), box(.25, .25, .25)])
    floor = policy(inv)
    current = state(inv, [pose(-3.)])
    layers = {item["region_id"]: item for item in floor.context_details(current)["regions"]}
    assert layers[("west",)]["support_z"] == .5
    assert layers[("east",)]["support_z"] == 0.
    assert floor.active_floor_rejection(current, action(inv, 1, pose(3.)))[0] is None
    assert floor.active_floor_rejection(current, action(inv, 1, pose(3., z=.5)))[0] == "above_active_layer"


def test_l_shape_coverage_excludes_notch_and_shared_pieces_do_not_make_extra_floors():
    horizontal, vertical = box(4., 1., 2.), box(1., 4., 2.)
    geometry = CompositeTargetGeometry((TargetComponent("a", horizontal, pose(0., -1.5)),
                                        TargetComponent("b", vertical, pose(-1.5, 0.))))
    inv = inventory(geometry, [box(4., 1., .5), box(1., 4., .5)])
    floor = policy(inv)
    assert len(floor.target_regions.regions) == 1
    layers = floor.context_details(state(inv, [pose(0., -1.5), pose(-1.5, 0.)]))["regions"]
    assert layers[0]["completed_layers"][0]["coverage"] == 1.
    assert layers[0]["support_z"] == .5
    assert 0. < layers[0]["valid"].mean() < .5


def test_touching_subdivision_preserves_floor_coverage_but_small_real_gaps_stay_separate():
    whole = CompositeTargetGeometry((TargetComponent("whole", box(4., 2.)),))
    split = CompositeTargetGeometry((TargetComponent("a", box(), pose(-1.)), TargetComponent("b", box(), pose(1.))))
    values = []
    for geometry in (whole, split):
        inv = inventory(geometry)
        layer, = policy(inv).context_details(state(inv, [pose(-1.)]))["regions"]
        values.append(layer)
    assert values[0]["occupancy"] == values[1]["occupancy"]
    np.testing.assert_array_equal(values[0]["valid"], values[1]["valid"])
    gap = CompositeTargetGeometry((TargetComponent("a", box(), pose(-1.001)), TargetComponent("b", box(), pose(1.001))))
    assert len(TargetRegions(gap).regions) == 2


def test_elevated_target_rejected_until_support_policy_exists():
    geometry = CompositeTargetGeometry((TargetComponent("floating", box(), pose(z=1.)),))
    with pytest.raises(ValueError, match="common ground plane"):
        policy(inventory(geometry))
    assert len(policy(inventory(geometry), ground_z=1.).target_regions.regions) == 1


def test_sampling_rotates_regions_even_with_one_draw_per_object_and_large_gaps():
    inv = inventory(target(1000.))
    sample = sampler(inv)
    current = state(inv)
    proposals = [sample.sample(inv, current, None, index, 1, None)[0] for index in (0, 1, 2, 0)]
    assert [p.diagnostics["target_region_id"] for p in proposals] == [("east",), ("west",)] * 2
    assert all(p.diagnostics["sampling_policy"] == "corner_seed" for p in proposals)
    assert all(abs(p.pose[0]) > 998. for p in proposals)
    assert sample.score_map(current, None, 0)["regions"][("east",)]["scores"].shape == (24, 24)
    assert all(item["grid_source"] == "region_fallback"
               for item in sample.score_map(current, None, 0)["regions"].values())


def test_shared_grid_computes_support_once_per_orientation_and_keeps_region_pools(monkeypatch):
    from stacking_common.actions import composite_sampling

    inv = inventory()
    sample = sampler(inv)
    calls = []
    original = composite_sampling.support_likelihood_maps_by_kernel

    def record(surface, kernels, tolerance):
        calls.append(surface.copy())
        return original(surface, kernels, tolerance)

    monkeypatch.setattr(composite_sampling, "support_likelihood_maps_by_kernel", record)
    current = state(inv)
    scores = sample.score_map(current, None, 0)
    assert len(calls) == len(sample.quaternions(current, None, 0))
    east, west = scores["regions"].values()
    assert east["grid_source"] == west["grid_source"] == "shared"
    np.testing.assert_array_equal(east["x_coords"], west["x_coords"])
    np.testing.assert_array_equal(east["y_coords"], west["y_coords"])
    assert east["scores"].shape == west["scores"].shape == (24, 24)
    assert len(sample._composite._surfaces) == 1
    assert all(np.isfinite(surface).all() for surface in calls)
    assert {item["target_region_id"] for item in scores["candidates"]} == {("east",), ("west",)}
    assert sample.score_map(current, None, 0) is scores
    assert len(calls) == 2  # Repeated requests reuse this state's score map.
    np.testing.assert_allclose(
        [item["normalized_score"] for item in scores["candidates"]],
        (np.array([item["score"] for item in scores["candidates"]]) -
         min(item["score"] for item in scores["candidates"])) /
        max(np.ptp([item["score"] for item in scores["candidates"]]), 1e-12))


def test_shared_surface_keeps_ground_and_placed_geometry_in_target_gap(monkeypatch):
    from stacking_common.actions import composite_sampling

    inv = inventory(shapes=[box(1., 1., .5), box(.25, .25, .25)])
    sample = sampler(inv, yaw_angles=(0.,))
    surfaces = []
    original = composite_sampling.support_likelihood_maps_by_kernel

    def record(surface, kernels, tolerance):
        surfaces.append(surface.copy())
        return original(surface, kernels, tolerance)

    monkeypatch.setattr(composite_sampling, "support_likelihood_maps_by_kernel", record)
    scores = sample.score_map(state(inv, [pose(z=.2)]), None, 1)
    assert len(surfaces) == 1
    grid = next(iter(scores["regions"].values()))
    xx, yy = np.meshgrid(grid["x_coords"], grid["y_coords"])
    gap = ~inv.target.horizontal_mask(np.c_[xx.ravel(), yy.ravel()], 0.).reshape(xx.shape)
    assert np.isfinite(surfaces[0]).all()
    assert np.any(surfaces[0][gap] > .6)
    assert np.any(surfaces[0][gap] == 0.)
    assert not any((value["candidate_mask"] & gap).any() for value in scores["regions"].values())


def test_local_fallback_checks_stone_footprint_not_only_region_grid_coverage():
    geometry = CompositeTargetGeometry((TargetComponent("west", box(.4, 2.), pose(-3.)),
                                        TargetComponent("east", box(.4, 2.), pose(3.))))
    inv = inventory(geometry, [box(.38, .25, .25)])
    sample = sampler(inv, yaw_angles=(0.,))
    scores = sample.score_map(state(inv), None, 0)
    # Both walls have global-grid cell centers, but none fit the wide stone.
    from stacking_common.actions.target_regions import SamplingRegion
    _, _, xx, yy = SamplingRegion(geometry).grid((24, 24))
    assert all(region.mask(xx, yy, 0.).any() for region in sample.floor_policy.target_regions.regions)
    assert all(value["grid_source"] == "region_fallback" and value["candidates"]
               for value in scores["regions"].values())
    assert {item["target_region_id"] for item in scores["candidates"]} == {("east",), ("west",)}


def test_sampler_keeps_empty_second_region_after_first_placement():
    inv = inventory()
    sample = sampler(inv)
    current = state(inv, [pose(-3.7, -.7)])
    proposals = sample.sample(inv, current, None, 1, 12, "lower_floor_first")
    assert {p.diagnostics["target_region_id"] for p in proposals} == {("east",), ("west",)}
    assert any(p.diagnostics["sampling_policy"] == "corner_seed" and p.pose[0] > 0 for p in proposals)
    for p in proposals:
        assert abs(p.pose[0]) > 1.5


def test_l_shape_samples_exclude_notch_and_match_rotated_full_footprints():
    geometry = CompositeTargetGeometry((TargetComponent("a", box(4., 1.)),
                                        TargetComponent("b", box(1., 4.), pose(-1.5, 1.5))))
    inv = inventory(geometry)
    sample = sampler(inv)
    scores = sample.score_map(state(inv), None, 0)
    assert scores["candidates"]
    region, = sample.floor_policy.target_regions.regions
    for item in scores["candidates"]:
        assert geometry.contains([[*item["xy"], item["height"]]])[0]
        q = sample.quaternions(state(inv), None, 0)[item["orientation_index"]]
        rotated = Rotation.from_quat(q).apply(inv[0].mesh_arrays()[0])
        assert anchor_components(region, [item["xy"]], rotated, item["placement_height"]).any()


def test_oriented_anchors_can_fit_narrow_region_without_axis_aligned_box_false_rejection():
    angle = .7
    geometry = CompositeTargetGeometry((TargetComponent("strip", box(2., .45), pose(yaw=angle)),))
    inv = inventory(geometry, [box(.8, .2, .25)])
    sample = sampler(inv, yaw_angles=(0., angle))
    items = sample.score_map(state(inv), None, 0)["candidates"]
    assert items and {item["orientation_index"] for item in items} == {1}


def test_taper_mask_uses_actual_placement_height():
    shape = TargetGeometry(2., 2., 2., np.zeros(3), DirectionalTaper.from_degrees(20., 20., 0., 0.), 0.)
    region, = TargetRegions(CompositeTargetGeometry((TargetComponent("taper", ConvexGeometry(shape.points, shape.faces)),))).regions
    points = box(.25, .25, .25).points
    assert anchor_components(region, [[.7, 0.]], points, 0.).any()
    assert not anchor_components(region, [[.7, 0.]], points, 1.).any()


@pytest.mark.parametrize("ply", [False, True])
def test_initialization_uses_common_physical_ground_and_global_scene(ply, tmp_path):
    inv = inventory()
    current = state(inv, [pose(3., z=.5)])
    initial = (PlyHeightInitializer(tmp_path, clearance=.01, xy_padding=0.) if ply else
               MeshHeightInitializer(.01, 0.))
    assert initial.initialize(inv, current, 1, PlanarPlacement(pose(-3.))).support_height == 0.
    assert initial.initialize(inv, current, 1, PlanarPlacement(pose(3.))).support_height == .75


class Solver:
    def solve(self, inventory, state, index, initial):
        return PlacementSolveResult(initial.pose, 0., 0.)


def generator(inv):
    sample = sampler(inv)
    return ScoredPlacementActionGenerator(inventory=inv,
        config=PlacementActionGeneratorConfig(8, 0., 0., None),
        planar_sampler=sample, initializer=MeshHeightInitializer(0., 0.), pose_solver=Solver(),
        filters=(ActiveFloorFilter(sample.floor_policy),))


def test_generation_budget_does_not_discard_region_balanced_prefix():
    inv = inventory()
    gen = generator(inv)
    result = gen.generate(state(inv), None, ActionGenerationRequest(3))
    assert result.attempts == 3 and len(result.candidates) == 3
    assert {item.diagnostics["target_region_id"] for item in result.candidates} == {("east",), ("west",)}


def test_solved_pose_metadata_replaces_parent_region_and_gap_is_rejected():
    inv = inventory()
    gen = generator(inv)
    parent = PlanarPlacement(pose(-3.), diagnostics={"target_region_id": ("west",)})
    for x, expected in ((3., ("east",)), (0., None)):
        gen.pose_solver = SimpleNamespace(solve=lambda *args: PlacementSolveResult(pose(x), 0., 0.))
        candidate, reason = gen.build_candidate(state(inv), 0, parent)
        if expected is None:
            assert candidate is None and reason
        else:
            assert reason is None and candidate.diagnostics["target_region_id"] == expected


def test_floor_fallback_cannot_bypass_actual_target_overlap():
    inv = inventory()
    gen = generator(inv)
    parent = PlanarPlacement(pose(-3.))
    gen.pose_solver = SimpleNamespace(solve=lambda *args: PlacementSolveResult(pose(0.), 0., 0.))
    with gen.floor_fill_fallback():
        candidate, reason = gen.build_candidate(state(inv), 0, parent)
    assert candidate is None and reason == "final_pose_outside_target"
    assert not gen.planar_sampler.relax_floor_fill


def cem(gen, **kwargs):
    config = CEMRootConfig(6, 2, 2, 4, 0., 0., .5, 0., 0.)
    return CEMRootActionGenerator(gen, SimpleNamespace(score=lambda s, o, candidates: [0.] * len(candidates)),
                                  replace(config, **kwargs))


def test_cem_mutation_refreshes_region_labels_and_elites_cover_both_regions():
    inv = inventory()
    gen = generator(inv)
    candidates = []
    for x, index, region in ((-3., 0, "west"), (-3.1, 1, "west"), (3., 2, "east")):
        value = action(inv, index, pose(x))
        candidates.append(ActionCandidate(value, diagnostics={"target_region_id": (region,)}))
    search = cem(gen, mutation_xy_std=1.)
    assert [item.diagnostics["target_region_id"] for item in search._elite_parents(candidates)] == [("west",), ("east",)]
    rng = SimpleNamespace(normal=lambda *args, **kwargs: np.array([6., 0.]) if kwargs.get("size") == 2 else 0.)
    _, proposal = search._mutation_proposal(rng, candidates[0], None)
    assert proposal.diagnostics["target_region_id"] == ("east",)
    result = cem(gen).generate(state(inv), None, ActionGenerationRequest(4))
    assert result.candidates


def test_regional_elites_keep_per_object_limit_and_legacy_order_without_labels():
    inv = inventory()
    search = cem(generator(inv), max_per_object=1)
    ranked = [ActionCandidate(action(inv, index, pose(x)), diagnostics={"target_region_id": (region,)})
              for x, index, region in ((-3., 0, "west"), (3., 0, "east"), (3., 1, "east"))]
    assert search._elite_parents(ranked) == [ranked[0], ranked[2]]
    plain = [ActionCandidate(item.action) for item in ranked]
    assert search._elite_parents(plain) == search._limit_per_object(plain)[:2]


def test_xy_overlap_filter_rejects_notch_and_detects_edge_crossing():
    geometry = CompositeTargetGeometry((TargetComponent("a", box(4., 1.)),
                                        TargetComponent("b", box(1., 4.), pose(-1.5, 1.5))))
    inv = inventory(geometry)
    gate = TargetXYOverlapFilter()
    assert not gate.evaluate(inv, None, 0, None, PlacementSolveResult(pose(1., 2.), 0., 0.)).accepted
    assert gate.evaluate(inv, None, 0, None, PlacementSolveResult(pose(0., .55), 0., 0.)).accepted


def test_seeded_sampling_is_repeatable_and_state_cache_is_invalidated():
    inv = inventory()
    first, second = sampler(inv), sampler(inv)
    current = state(inv)
    a = first.sample(inv, current, None, 0, 8, None)
    b = second.sample(inv, current, None, 0, 8, None)
    np.testing.assert_array_equal([p.pose for p in a], [p.pose for p in b])
    old = first.score_map(current, None, 0)
    current.placed_object_indices = [1]
    current.object_poses[inv[1].object_id] = pose(-3.)
    assert first.score_map(current, None, 0) is not old


def test_component_order_does_not_change_sampling_or_region_identity():
    geometry = target()
    first, second = inventory(geometry), inventory(CompositeTargetGeometry(tuple(reversed(geometry.components))))
    a = sampler(first).sample(first, state(first), None, 0, 8, None)
    b = sampler(second).sample(second, state(second), None, 0, 8, None)
    np.testing.assert_array_equal([item.pose for item in a], [item.pose for item in b])
    assert [item.diagnostics["target_region_id"] for item in a] == [item.diagnostics["target_region_id"] for item in b]


def test_mutation_score_uses_latest_orientation_bank_not_stale_cached_map():
    geometry = CompositeTargetGeometry((TargetComponent("strip", box(2., .45)),))
    inv = inventory(geometry, [box(.8, .2, .25)])
    sample = sampler(inv, yaw_angles=(0.,))
    current = state(inv)
    assert sample.score_map(current, None, 0)["candidates"]
    sample.orientation_provider = MappingOrientationProvider({1: pose(yaw=np.pi / 2)[3:]})
    assert not sample.score_map(current, None, 0)["candidates"]
    # The initial bank still has a feasible pose, but it is not the active bank.
    assert sample.score_at_object_pose(None, pose(), 0) == (0., 0.)


def test_platform_can_build_composite_generator():
    from pathlib import Path
    from stacking_tabletop import TabletopApplication
    inv = inventory()
    app = TabletopApplication.from_repository(Path(__file__).resolve().parents[1])
    gen = app.build_action_generator(inv, Solver())
    assert gen.planar_sampler._composite is not None


@pytest.mark.parametrize("ground_z", [0., .7])
def test_raster_support_does_not_reject_a_tilted_stone_below_ground(ground_z):
    from stacking_common.actions._support_maps import bottom_height_kernel

    target = CompositeTargetGeometry((TargetComponent("floor", box(), pose(z=ground_z)),))
    inv = inventory(target, [box(.4, .3, .2)])
    sample = sampler(inv, yaw_angles=(0.,), floor_policy=policy(inv, ground_z=ground_z))
    quaternion = Rotation.from_euler("xyz", [17., 23., 11.], degrees=True).as_quat()
    sample.orientation_provider = MappingOrientationProvider({1: quaternion})
    vertices = sample.stones[0].points
    rotated = Rotation.from_quat(quaternion).apply(vertices)
    kernel = bottom_height_kernel(vertices, quaternion, 2. / 24., 2. / 24.)
    # The cell centers miss the lowest corner. The uncorrected estimate would
    # make the full vertex containment check reject every otherwise valid XY.
    assert kernel.bottom_height.min() > rotated[:, 2].min() + 1e-5
    scores = sample.score_map(state(inv), None, 0)
    assert scores["candidates"]
    for candidate in scores["candidates"]:
        assert candidate["placement_height"] + rotated[:, 2].min() == pytest.approx(ground_z)
