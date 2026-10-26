"""Union rewards and component-aware observations, without a full MCTS run."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import pickle

import numpy as np
import pytest

from stacking_common.artifacts import decode_observation, encode_observation
from stacking_common.domain import Action, CompositeTargetGeometry, StackObject, TargetComponent
from stacking_common.domain.geometry import inverse_transform_points
from stacking_common.environment import (
    ArrayObservationBuilder, InventoryObservationAdapter, ObservationConfig,
    RewardConfig, RewardWeights, StackingReward, VoxelOverlapEvaluator,
)
from stacking_common.environment.surface import SurfaceFlatness
from stacking_common.learning import encode_heightmap_observation
from stacking_common.visualization.heightmap import HeightMapRenderer
from test_composite_target_sampling import box, inventory, pose, state, target


def composite(*pieces):
    return CompositeTargetGeometry(tuple(TargetComponent(str(i), shape, transform)
                                         for i, (shape, transform) in enumerate(pieces)))


def reward(inv, resolution=.1, **weights):
    return StackingReward(inv, VoxelOverlapEvaluator(inv.target, resolution),
        RewardConfig(1., .5, replace(RewardWeights(0., 0., 1., 0., 0., 0.), **weights)))


def placed(inv, poses):
    current = state(inv, poses)
    index = len(poses) - 1
    current.action_history = [Action(index, poses[-1], poses[-1], object_id=inv[index].object_id,
                                    place_robustness_displacement=0.)]
    return current


@pytest.mark.parametrize("offset,volume", [(0., 8.), (1., 12.), (2., 16.), (10000., 16.)])
def test_union_volume_counts_overlap_once_without_allocating_the_gap(offset, volume):
    geometry = composite((box(2., 2., 2.), pose()), (box(2., 2., 2.), pose(offset)))
    evaluator = VoxelOverlapEvaluator(geometry, .25)
    assert evaluator.target_volume == pytest.approx(volume)
    assert len(evaluator._grid_points) == round(volume / .25**3)
    assert not hasattr(evaluator, "_lattice_indices")
    assert len(np.unique(evaluator._grid_points, axis=0)) == len(evaluator._grid_points)


def test_decomposition_order_and_duplicate_components_preserve_union_lattice():
    pieces = [(box(2., 2., 2.), pose(-1.)), (box(2., 2., 2.), pose(1.))]
    shapes = [composite((box(4., 2., 2.), pose())), composite(*pieces),
              composite(*reversed(pieces)), composite(*pieces, pieces[0])]
    grids = [VoxelOverlapEvaluator(shape, .2) for shape in shapes]
    for grid in grids[1:]:
        np.testing.assert_array_equal(grid._grid_points, grids[0]._grid_points)
        assert grid.target_volume == grids[0].target_volume


def test_rotated_components_match_dense_union_and_cropped_intersections():
    geometry = composite((box(2., 1., 2.), pose(yaw=.4)), (box(1., 2., 1.), pose(.8, yaw=-.6)))
    evaluator = VoxelOverlapEvaluator(geometry, .1)
    lower, upper = geometry.bounds
    axes = [np.arange(low + .05, high, .1) for low, high in zip(lower, upper)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    np.testing.assert_allclose(evaluator._grid_points, points[geometry.contains(points)], atol=1e-12)
    inv = inventory(geometry)
    for x in np.linspace(-1., 2., 10):
        value = pose(x, .2, .3, yaw=.7)
        inside = inv[0].contains_local_points(inverse_transform_points(evaluator._grid_points, value), 1e-7)
        assert evaluator._intersection_volume(inv[0], value) == pytest.approx(inside.sum() * .1**3)


def test_resolution_cannot_silently_omit_a_small_component():
    geometry = composite((box(), pose()), (box(.01, .01, .01), pose(10.)))
    with pytest.raises(ValueError, match="component dimension"):
        VoxelOverlapEvaluator(geometry, .1)


def test_containment_and_scene_coverage_use_unions_not_component_sums():
    geometry = composite((box(2., 2., 2.), pose(-1.)), (box(2., 2., 2.), pose(1.)))
    inv = inventory(geometry, [box(2., 1., 1.)] * 2)
    evaluator = VoxelOverlapEvaluator(geometry, .1)
    # This object spans a DSF seam. Reward geometry measures the true union;
    # native conservative posegen containment remains a separate policy.
    assert evaluator.object_containment(inv[0], pose()) == pytest.approx(1.)
    assert evaluator.target_coverage(inv[0], pose()) == pytest.approx(2. / 16.)
    pairs = [(inv[0], pose()), (inv[1], pose())]
    assert evaluator.scene_coverage(pairs) == pytest.approx(2. / 16.)
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda _: evaluator.scene_coverage(pairs), range(12)))
    assert values == pytest.approx([2. / 16.] * 12)


def test_reward_only_counts_new_union_occupancy_and_uses_fixed_inventory_scale():
    inv = inventory(composite((box(2., 2., 2.), pose()), (box(2., 2., 2.), pose(1.))),
                    [box(.4, .4, .4)] * 3)
    evaluate = reward(inv)
    first, first_info = evaluate.compute(placed(inv, [pose()]))
    duplicate, duplicate_info = evaluate.compute(placed(inv, [pose(), pose()]))
    added, added_info = evaluate.compute(placed(inv, [pose(), pose(), pose(1.)]))
    assert first == pytest.approx(1.)
    assert duplicate == 0.
    assert added == pytest.approx(1.)
    assert duplicate_info["target_IoU"] == first_info["target_IoU"]
    assert added_info["target_IoU"] == pytest.approx(2. * .4**3 / 12.)
    assert added_info["target_volume_method"] == "voxel_union"
    assert added_info["target_volume"] == pytest.approx(12.)


def test_reward_normalization_is_independent_of_extra_remote_target_volume():
    values = []
    for geometry in (composite((box(), pose())), target(100.)):
        inv = inventory(geometry, [box(.4, .4, .4)])
        values.append(reward(inv).compute(placed(inv, [pose(0. if len(geometry.components) == 1 else -100.)]))[0])
    assert values == pytest.approx([1., 1.])


def test_orientation_uses_local_component_not_center_in_gap(monkeypatch):
    inv = inventory(target())
    evaluate = reward(inv)
    monkeypatch.setattr(StackObject, "representative_face_normals",
        lambda self: (np.array([[.4, 0., np.sqrt(.84)]]), np.ones(1)))
    assert evaluate._inward_orientation_score(0, pose(-2.7)) == 0.
    assert evaluate._inward_orientation_score(0, pose(-3.3)) > 0.


def test_surface_uses_union_footprint_and_area_weights_not_disconnected_gap():
    geometry = composite((box(1., 1., 3.), pose(-10.)), (box(2., 1., 3.), pose(10.)))
    inv = inventory(geometry, [box(1., 1., 1.), box(2., 1., 2.)])
    surface = SurfaceFlatness(inv)
    assert surface.measure(state(inv)) == (0., 0.)
    assert surface.measure(state(inv, [pose(-10.)])) == pytest.approx((1., 1. / 3.))
    current = state(inv, [pose(-10.), pose(10.)])
    assert surface.measure(current) == pytest.approx((2., 5. / 3.))
    moved = state(inv, [pose(-10.), pose(10., z=1.)])
    assert surface.measure(moved) == pytest.approx(SurfaceFlatness(inv).measure(moved))
    assert surface.measure(current) == pytest.approx((2., 5. / 3.))


def test_surface_excludes_l_notch_and_overlapping_component_area():
    geometry = composite((box(4., 1., 3.), pose(0., -1.5)), (box(1., 4., 3.), pose(-1.5)))
    inv = inventory(geometry, [box(4., 1., 1.), box(1., 4., 1.)])
    assert SurfaceFlatness(inv).measure(state(inv, [pose(0., -1.5), pose(-1.5)])) == pytest.approx((1., 1.))


def observation_builder(inv):
    renderer = HeightMapRenderer(inv, (41, 81), .2)
    source = InventoryObservationAdapter(inv, 8, renderer)
    return ArrayObservationBuilder(source, ObservationConfig(2, renderer.resolution, True)), renderer


def test_observation_preserves_mesh_components_mask_and_envelope():
    inv = inventory(composite((box(2., 2., 1.), pose(-3.)), (box(2., 2., 2.), pose(3.))))
    builder, renderer = observation_builder(inv)
    obs = builder.build(state(inv, [pose(-3.)]))
    assert obs.target_kind == "composite"
    np.testing.assert_array_equal(obs.target_bounds, inv.target.bounds)
    assert obs.target_width == 8.
    assert obs.target_length == 2.
    assert obs.target_height == 2.
    assert obs.target_component_ids == ("0", "1")
    assert obs.target_component_point_offsets.tolist() == [0, 8, 16]
    assert obs.target_component_face_offsets.tolist() == [0, 12, 24]
    x = np.linspace(*renderer.xlim, 81)
    y = np.linspace(*renderer.ylim, 41)
    row = np.argmin(abs(y))
    gap, left, right = [np.argmin(abs(x - value)) for value in (0., -3., 3.)]
    assert not obs.target_footprint_mask[row, gap]
    assert obs.target_height_map[row, gap] == 0.
    assert obs.target_height_map[row, left] == pytest.approx(1.)
    assert obs.target_height_map[row, right] == pytest.approx(2.)
    assert not obs.target_bounds.flags.writeable
    assert not obs.target_footprint_mask.flags.writeable
    assert obs.scene_height_map[row, left] > 0.


def test_observation_deferred_and_record_roundtrips_preserve_components():
    inv = inventory()
    builder, _ = observation_builder(inv)
    current = state(inv, [pose(-3.)])
    lazy = builder.defer(current)
    current.object_poses[1] = pose(100.)
    obs = pickle.loads(pickle.dumps(lazy))
    assert obs.object_poses[0, 0] == -3.
    decoded = decode_observation(encode_observation(obs))
    assert decoded.target_component_ids == obs.target_component_ids
    np.testing.assert_array_equal(decoded.target_component_face_offsets, obs.target_component_face_offsets)
    decoded.target_bounds[0, 0] -= 1.
    assert not np.array_equal(decoded.target_bounds, obs.target_bounds)
    learning = encode_heightmap_observation(obs)
    assert learning["target_is_composite"]
    np.testing.assert_array_equal(learning["target_footprint_mask"], obs.target_footprint_mask)
    np.testing.assert_array_equal(learning["target_points"], obs.target_points.astype(np.float32))


def test_pre_composite_observation_records_still_decode_with_single_target_defaults():
    from test_object_geometry import target_geometry
    inv = inventory(target_geometry())
    builder, _ = observation_builder(inv)
    original = builder.build(state(inv))
    legacy = encode_observation(original)
    for name in ("target_kind", "target_bounds", "target_component_ids",
                 "target_component_point_offsets", "target_component_face_offsets",
                 "target_footprint_mask"):
        del legacy[name]
    restored = decode_observation(legacy)
    assert restored.target_kind == "single"
    assert restored.target_component_ids == ()
    assert restored.target_bounds is None
    np.testing.assert_array_equal(restored.target_points, original.target_points)
    np.testing.assert_array_equal(restored.target_height_map, original.target_height_map)
    assert set(encode_heightmap_observation(restored)) == set(encode_heightmap_observation(original))


@pytest.mark.parametrize("platform", ["tabletop", "excavator"])
def test_platform_can_observe_and_score_composite_without_native_allocation(platform):
    from stacking_tabletop import TabletopApplication
    from stacking_excavator import ExcavatorApplication
    application_type = TabletopApplication if platform == "tabletop" else ExcavatorApplication
    app = application_type.from_repository(Path(__file__).resolve().parents[1])
    app = replace(app, environment_profile=replace(app.environment_profile, overlap_resolution=.1))
    inv = inventory(composite((box(), pose(-3.)), (box(), pose(3.))), [box(.4, .4, .4)])
    evaluator = app.build_committed_state_evaluator(inv)
    observation, value, info = evaluator(placed(inv, [pose(-3.)]))
    assert observation.target_kind == "composite" and np.isfinite(value)
    assert 0. < info["target_IoU"] < 1.
    # Still fail before native solver allocation in an end-to-end run.
    with pytest.raises(NotImplementedError, match="full MCTS viewer and saved-run"):
        app.build_mcts(inv)
