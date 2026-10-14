"""Geometry and integration contracts for the optional wall construction mode."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stacking_common.actions import MappingOrientationProvider, PlanarSamplingConfig
from stacking_common.actions.wall import WallPlanarPoseSampler, WallPlacementFilter
from stacking_common.actions.types import InitializedPlacement, PlacementSolveResult
from stacking_common.domain import Action, DirectionalTaper, ObjectInventory, StackObject, State, TargetGeometry
from stacking_common.environment.wall import WallReward
from stacking_common.environment import RewardConfig, RewardWeights, VoxelOverlapEvaluator
from stacking_common.visualization.heightmap import HeightMapRenderer
from stacking_common.wall import ConstructionConfig, WallSurfaces
from test_action_generation import _cube, _BodyModel, _pose


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ConstructionConfig("two_sided_wall", .05)


def inventory(*, taper=None, origin=None, width=1., count=3, duplicate_components=False):
    geometry = _cube(.3)
    objects = [StackObject(i + 1, _BodyModel(), [geometry] * (2 if duplicate_components else 1))
               for i in range(count)]
    target = TargetGeometry(width, 2., 1., np.zeros(3) if origin is None else origin,
                            taper or DirectionalTaper(), 0.)
    return ObjectInventory(objects, {obj.object_id: _pose(x=4.) for obj in objects}, target)


def state(inv, poses=()):
    return State(list(inv.object_ids), list(range(len(poses))),
                 {inv.object_ids[i]: pose for i, pose in enumerate(poses)}, [], [])


def sampler(inv):
    settings = dict(boltzmann_temperature=.3, boltzmann_min_probability=0.,
        target_mask_margin=0., contact_gap_tolerance=.02,
        score=dict(height=-3., support_likelihood=2., elite_fraction=.25),
        floor=dict(bottom_tolerance=.02, coverage_mode="projected_footprint", grid_size=16,
                   fill_ratio=.75, footprint_scale=.9))
    return WallPlanarPoseSampler(inv, PlanarSamplingConfig((32, 32), (0.,)),
        MappingOrientationProvider({oid: _pose()[3:] for oid in inv.object_ids}),
        settings, construction=CONFIG)


def observation(inv, current):
    renderer = HeightMapRenderer(inv, (64, 64), .5)
    return SimpleNamespace(scene_height_map=renderer.render(current),
                           height_map_xlim=renderer.xlim, height_map_ylim=renderer.ylim)


def test_configuration_requires_explicit_physical_tolerance():
    assert not ConstructionConfig().enabled
    with pytest.raises(ValueError, match="surface_tolerance"):
        ConstructionConfig("two_sided_wall")
    with pytest.raises(ValueError, match="mode"):
        ConstructionConfig("two_rows")
    with pytest.raises(ValueError, match="unknown"):
        ConstructionConfig.from_mapping({"active_side": 0})


def test_asymmetric_taper_and_offset_use_rotated_stone_support():
    inv = inventory(taper=DirectionalTaper(width_negative=.1, width_positive=.2),
                    origin=np.array([2., 3., 4.]))
    wall = WallSurfaces(inv, CONFIG)
    q = Rotation.from_euler("xyz", [.1, -.2, .4]).as_quat()
    for side in (0, 1):
        z = 4.5
        x = wall.tangent_x(0, q, side, z)
        points = Rotation.from_quat(q).apply(wall.points[0]) + [x, 3., z]
        distance = points @ wall.normals[side] - wall.plane_offsets[side]
        assert distance.max() == pytest.approx(0., abs=1e-12)
        assert np.all(distance <= 1e-12)


def test_external_union_does_not_double_count_components_or_occluded_stones():
    inv = inventory()
    wall = WallSurfaces(inv, CONFIG)
    pose = _pose(x=-.35, z=.15)
    current = state(inv, [pose])
    one = wall.envelope(current)
    duplicate = WallSurfaces(inventory(duplicate_components=True), CONFIG)
    np.testing.assert_array_equal(one, duplicate.envelope(current))
    assert wall.progress(one) > 0
    assert wall.gain(one, 1, pose) == pytest.approx(0.)
    assert wall.gain(one, 1, _pose(x=-.05, z=.15)) == pytest.approx(0.)
    # An outer protrusion hides the previous face; union of quality would miss this.
    assert wall.gain(one, 1, _pose(x=-.45, z=.15)) < 0


def test_mirror_symmetry_and_two_face_stone():
    inv = inventory()
    wall = WallSurfaces(inv, CONFIG)
    left = wall.quality(wall.object_envelope(0, _pose(x=-.35, z=.15)))
    right = wall.quality(wall.object_envelope(0, _pose(x=.35, z=.15)))
    np.testing.assert_array_equal(left, right[::-1])
    narrow = inventory(width=.3)
    through = WallSurfaces(narrow, CONFIG)
    diagnostics = through.diagnostics(state(narrow, [_pose(z=.15)]))
    assert diagnostics["wall_negative_x_coverage"] > 0
    assert diagnostics["wall_positive_x_coverage"] > 0
    assert diagnostics["wall_cross_row_connected"]


def test_lower_surface_completion_is_more_valuable_and_exact_pose_invalidates_cache():
    inv = inventory()
    wall = WallSurfaces(inv, CONFIG)
    current = state(inv, [_pose(x=-.35, z=.15)])
    low = wall.progress(wall.envelope(current))
    current.object_poses[1][2] = .75
    high = wall.progress(wall.envelope(current))
    assert 0 < high < low


def test_both_faces_and_empty_opposite_ground_seed_remain_available():
    inv = inventory()
    source = sampler(inv)
    empty = state(inv)
    proposals = source.sample(inv, empty, observation(inv, empty), 0, 16, None)
    assert len(proposals) == 4  # two ends on each face for this single orientation
    assert {p.diagnostics["wall_proposal_side"] for p in proposals} == {"negative_x", "positive_x"}
    current = state(inv, [_pose(x=-.35, y=-.7, z=.15)])
    data = source.score_map(current, observation(inv, current), 1)
    opposite = [p for p in data["candidates"] if p["side"] == 1]
    assert opposite
    assert all(p["pose"][2] == pytest.approx(.15) for p in opposite)
    # Upper placements are possible despite unfilled interior ground.
    assert any(p["pose"][2] > .3 for p in data["candidates"])


def test_corner_retry_broadens_wall_seeds_without_switching_modes():
    inv = inventory()
    source = sampler(inv)
    empty = state(inv)
    obs = observation(inv, empty)
    corners = source.score_map(empty, obs, 0)["candidates"]
    source.relax_corner_seed = True
    broader = source.score_map(empty, obs, 0)["candidates"]
    assert len(broader) > len(corners)
    assert {c["side"] for c in broader} == {0, 1}
    source.relax_corner_seed = False
    assert len(source.score_map(empty, obs, 0)["candidates"]) == len(corners)


def test_filter_and_cem_use_actual_orientation_and_pose():
    inv = inventory()
    source = sampler(inv)
    empty = state(inv)
    obs = observation(inv, empty)
    proposal = source.sample(inv, empty, obs, 0, 1, None)[0]
    check = WallPlacementFilter(source.wall)
    initial = InitializedPlacement(proposal.pose, 0.)
    # Changing sides is legal; seed-side metadata cannot decide the result.
    assert check.evaluate(inv, empty, 0, initial, PlacementSolveResult(_pose(x=.35, z=.15), 0., 0.)).accepted
    assert not check.evaluate(inv, empty, 0, initial, PlacementSolveResult(_pose(z=.15), 0., 0.)).accepted
    mutated = proposal.pose.copy()
    mutated[3:] = Rotation.from_euler("z", np.pi / 4).as_quat()
    assert source.score_at_object_pose(obs, mutated, 0) == (0., 0.)  # now protrudes outside target


def test_orientation_bank_survives_generic_cache_eviction(monkeypatch):
    import stacking_common.actions.wall as module

    # More than 256 fixed orientations must stay warm across scene-map rebuilds.
    inv = inventory(count=34)
    source = sampler(inv)
    source.config = replace(source.config, yaw_angles=tuple(np.arange(8) * np.pi / 4))
    current = state(inv)
    obs = observation(inv, current)
    calls = []
    original = module.bottom_height_kernel

    def build(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(module, "bottom_height_kernel", build)
    for index in range(len(inv)):
        source.score_map(current, obs, index)
    assert len(calls) == 34 * 8
    # Generic surface samples can be evicted by unrelated CEM orientations.
    source.wall._samples.clear()
    source._wall_context = None
    calls.clear()
    monkeypatch.setattr(source.wall, "_ray_bounds", lambda *a: pytest.fail("rebuilt fixed geometry"))
    for index in range(len(inv)):
        source.score_map(current, obs, index)
    assert not calls


def test_geometry_refreshes_for_grid_spacing_and_orientation_changes():
    inv = inventory()
    source = sampler(inv)
    current = state(inv)
    obs = observation(inv, current)
    original = source._geometry(current, obs, 0)
    assert source._geometry(current, obs, 0) is original
    # A scene-height change does not change the stone's geometry.
    obs.scene_height_map[:] += .1
    assert source._geometry(current, obs, 0) is original
    obs.height_map_xlim = (obs.height_map_xlim[0] - .5, obs.height_map_xlim[1] + .5)
    resized = source._geometry(current, obs, 0)
    assert resized is not original
    source.orientation_provider.quaternions[inv.object_ids[0]] = Rotation.from_euler("z", .3).as_quat()
    rotated = source._geometry(current, obs, 0)
    assert rotated is not resized
    np.testing.assert_allclose(rotated[0].quaternion, Rotation.from_euler("z", .3).as_quat())
    # Replacing banks does not retain obsolete grids/orientations for an object.
    assert len(source._orientation_geometry) == 1


def test_batched_face_support_matches_separate_evaluation():
    inv = inventory()
    source = sampler(inv)
    current = state(inv, [_pose(x=-.35, y=-.5, z=.15)])
    obs = observation(inv, current)
    q = Rotation.from_euler("xyz", [.1, .2, .3]).as_quat()
    xy = np.array([[[-.35, -.5], [-.35, .5]], [[.35, -.5], [.35, .5]]])
    batched = source._support(obs, 1, q, xy.reshape(-1, 2))
    separate = [source._support(obs, 1, q, points) for points in xy]
    for field, values in enumerate(batched):
        np.testing.assert_array_equal(values.reshape(2, 2), [row[field] for row in separate])


def test_wall_reward_includes_motion_of_previously_placed_stones():
    inv = inventory()
    base = RewardConfig(1., .5, RewardWeights(0., 0., 0., 0., 0., 0.))
    reward = WallReward(inv, VoxelOverlapEvaluator(inv.target, resolution=.05), base,
                        construction=CONFIG, grid_shape=(32, 32))
    parent = state(inv, [_pose(x=-.35, y=-.6, z=.15)])
    child = state(inv, [_pose(y=-.6, z=.15), _pose(x=.35, y=.6, z=.75)])
    child.action_history = [Action(1, child.object_poses[2], child.object_poses[2], object_id=2)]
    value, info = reward.compute_transition(parent, child)
    assert value < 0  # losing lower face area outweighs adding an upper patch
    assert info["wall_surface_increment"] < 0


def test_tabletop_fixed_scene_preparation_rejects_surface_drift():
    from stacking_tabletop.planning.coordinator import TabletopPlanningCoordinator
    inv = inventory()
    coordinator = TabletopPlanningCoordinator(inv, lambda *args: None, SimpleNamespace(plan=lambda p: None),
        construction_filter=WallPlacementFilter(WallSurfaces(inv, CONFIG)))
    action = Action(0, _pose(z=.15), _pose(x=-.35, z=.15), object_id=1)
    _, failure, diagnostics = coordinator._prepare_fixed_scene(state(inv), action)
    assert failure.reason == "wall_surface_miss"
    assert not diagnostics["wall_solved_sides"]
    action.pose = _pose(x=.35, z=.15)
    assert coordinator._prepare_fixed_scene(state(inv), action)[1] is None


def test_committed_wall_reward_uses_original_parent():
    from stacking_excavator import ExcavatorApplication
    app = ExcavatorApplication.from_repository(ROOT)
    app = replace(app, action_generation_profile=replace(app.action_generation_profile, construction=CONFIG))
    inv = inventory()
    parent = state(inv, [_pose(x=-.35, y=-.6, z=.15)])
    child = state(inv, [_pose(y=-.6, z=.15), _pose(x=.35, y=.6, z=.75)])
    child.action_history = [Action(1, child.object_poses[2], child.object_poses[2], object_id=2)]
    evaluate = app.build_committed_state_evaluator(inv)
    _, _, diagnostics = evaluate.transition(parent, child)
    assert diagnostics["wall_surface_increment"] < 0


@pytest.mark.parametrize("platform", ["excavator", "tabletop"])
def test_mode_composition_is_opt_in(platform):
    from stacking_excavator import ExcavatorApplication
    from stacking_tabletop import TabletopApplication
    from stacking_common.actions import ActiveFloorFilter
    from stacking_common.actions.scored import ScoredPlanarPoseSampler
    app = (ExcavatorApplication if platform == "excavator" else TabletopApplication).from_repository(ROOT)
    # Test both policies independently of the operator's selected YAML mode.
    app = replace(app, action_generation_profile=replace(
        app.action_generation_profile, construction=ConstructionConfig()))
    inv = inventory()
    normal = app.build_action_generator(inv, SimpleNamespace())
    from stacking_common.artifacts.planning_context import _values
    assert "construction" not in _values(app.action_generation_profile)
    assert type(normal.planar_sampler) is ScoredPlanarPoseSampler
    assert any(isinstance(f, ActiveFloorFilter) for f in normal.filters)
    wall_app = replace(app, action_generation_profile=replace(app.action_generation_profile, construction=CONFIG))
    generator = wall_app.build_action_generator(inv, SimpleNamespace())
    assert _values(wall_app.action_generation_profile)["construction"]["mode"] == "two_sided_wall"
    assert isinstance(generator.planar_sampler, WallPlanarPoseSampler)
    assert not any(isinstance(f, ActiveFloorFilter) for f in generator.filters)
    assert isinstance(wall_app._build_state_evaluators(inv)[1], WallReward)
    with generator.floor_fill_fallback() as active:
        assert not active


@pytest.mark.parametrize("workers", [1, 3])
def test_transition_reward_keeps_parents_aligned_through_batched_pipeline(workers):
    from stacking_common.environment import StackingEnvironment
    from test_environment import FakeTransition, state as empty_state, action

    class Evaluator:
        def compute_transition(self, parent, child):
            return parent.tag * 100 + child.tag, {}

        def build(self, current):
            return current.tag

    class Transition(FakeTransition):
        backend = SimpleNamespace(place_batch=lambda: None, placement_batch_size=2)

        def iter_step_batches(self, parents, actions):
            for start in range(0, len(parents), 2):
                children = []
                for parent in parents[start:start + 2]:
                    child = empty_state()
                    child.tag = parent.tag + 10
                    children.append(child)
                yield children

        def step_batch(self, parents, actions):
            return [s for batch in self.iter_step_batches(parents, actions) for s in batch]

        def step(self, action):
            child = empty_state()
            child.tag = self.current.tag + 10
            self.current = child
            return child

    evaluators = [Evaluator() for _ in range(workers)]
    env = StackingEnvironment(Transition(), evaluators[0], evaluators[0],
        worker_evaluators=[(item, item) for item in evaluators[1:]])
    env.result_batch_size = 2
    parents = [empty_state() for _ in range(9)]
    for i, parent in enumerate(parents):
        parent.tag = i
    try:
        results = env.step_batch(parents, [action()] * 9)
        assert [result.reward for result in results] == [i * 101 + 10 for i in range(9)]
        env.restore(parents[4])
        assert env.step(action()).reward == 414
    finally:
        env.close()


def test_wall_inspection_uses_fixed_orientation_and_current_surface_maps():
    from dataclasses import asdict
    from stacking_common.visualization.sampling import wall_pose_score_view
    inv = inventory()
    current = state(inv, [_pose(x=-.35, z=.15)])
    source = sampler(inv)
    data = wall_pose_score_view(inv, current, 2, _pose(x=.35, z=.15),
        planar_config=asdict(source.config), construction=asdict(CONFIG), settings=source.settings,
        height_map_resolution=(64, 64), height_map_margin=.5)
    assert data.proposal_xy.shape[1] == 2
    assert len(data.proposal_scores) == len(data.proposal_xy) > 0
    assert data.quality[0].sum() > 0 and data.quality[1].sum() == 0


def test_wall_diagnostics_roundtrip_without_changing_legacy_candidate_payload():
    from stacking_common.visualization import MCTSCandidateDiagnostic
    from stacking_common.artifacts.mcts_debug import _encode_candidate, _decode_candidate
    candidate = MCTSCandidateDiagnostic('wall', 1, 0, 1, 'succeeded', False,
        _pose(), _pose(), _pose(), construction_diagnostics={
            'wall_proposal_side': 'negative_x', 'wall_surface_increment': .03})
    decoded = _decode_candidate(_encode_candidate(candidate))
    assert decoded.construction_diagnostics == candidate.construction_diagnostics
    legacy = _encode_candidate(replace(candidate, construction_diagnostics={}))
    assert 'construction_diagnostics' not in legacy
    assert not _decode_candidate(legacy).construction_diagnostics


def test_native_cpu_wall_search_selects_a_validated_candidate():
    pytest.importorskip("stacking_core")
    import json
    import random
    from stacking_excavator import ExcavatorApplication
    from stacking_common.search import EvaluationFidelity, MCTSNode

    app = ExcavatorApplication.from_repository(ROOT)
    p = app.mcts_profile
    app = replace(app,
        action_generation_profile=replace(app.action_generation_profile,
            construction=ConstructionConfig("two_sided_wall", .1)),
        mcts_profile=replace(p,
            parallel=replace(p.parallel, workers=1, posegen_backend="cpu", simulation_backend="cpu"),
            tree=replace(p.tree, root_max_children=4, final_validation_candidates=4),
            root_cem=replace(p.root_cem, population=16, elite=4, iterations=1, keep=4)))
    reference = json.loads((ROOT / "tests/fixtures/legacy_planner/reference.json").read_text())
    ids = [56, 58, 8]
    inv = app.load_inventory(ids, {i: np.concatenate(reference["pickup_poses"][str(i)]) for i in ids})
    environment = app.build_environment(inv, app.build_core_backend(inv, EvaluationFidelity.SHORT))
    search = app.build_mcts(inv, rng=random.Random(0))
    try:
        reset = environment.reset()
        result = search.search(MCTSNode(state=reset.state, observation=reset.observation),
            replace(app.default_mcts_request(), iterations=2, max_time_seconds=10.,
                    final_validation_reserve_seconds=3.))
        assert result.generated_candidates > 0
        assert result.selected is not None and result.selected.selectable
        assert result.selected.info["wall_surface_increment"] > 0
        assert result.selected.info["wall_settled_sides"]
        assert environment.snapshot().placed_object_indices == []
    finally:
        search.close()
        environment.close()


def test_wall_dialog_builds_three_panels_without_requiring_gl(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtWidgets import QApplication, QWidget
    from stacking_excavator.gui.operations import wall_score_view
    from stacking_common.visualization.sampling import wall_pose_score_view
    from dataclasses import asdict

    calls = []

    class Plot(QWidget):
        def __init__(self, parent, shape):
            super().__init__(parent)
            self.interactor = self
            assert shape == (1, 3)

        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            return lambda *args, **kwargs: calls.append((name, args, kwargs))

    monkeypatch.setattr(wall_score_view, "QtInteractor", Plot)
    app = QApplication.instance() or QApplication([])
    inv = inventory()
    source = sampler(inv)
    data = wall_pose_score_view(inv, state(inv), 1, _pose(), planar_config=asdict(source.config),
        construction=asdict(CONFIG), settings=source.settings, height_map_resolution=(64, 64), height_map_margin=.5)
    dialog = wall_score_view.WallScoreDialog(data, step=1, candidate=SimpleNamespace(object_id=1, rank=1))
    try:
        assert [args for name, args, _ in calls if name == "subplot"] == [(0, 0), (0, 1), (0, 2)]
        assert len([1 for name, _, _ in calls if name == "add_mesh"]) == 4
    finally:
        dialog.close()
        app.processEvents()
