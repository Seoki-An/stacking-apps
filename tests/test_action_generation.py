from dataclasses import dataclass
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from stacking_common.actions import (
    ActiveFloorFilter,
    CEMRootActionGenerator,
    CEMRootConfig,
    InitializedPlacement,
    MeshHeightInitializer,
    PlacementActionGenerator,
    PlacementActionGeneratorConfig,
    PlacementSolveResult,
    PlanarPlacement,
    PlanarSamplingConfig,
    MappingOrientationProvider,
    ScanAlignedOrientationProvider,
    ScanOrientationFilter,
    TargetContainmentFilter,
)
from stacking_common.domain import (
    ConvexGeometry,
    DirectionalTaper,
    ObjectInventory,
    StackObject,
    State,
    TargetGeometry,
)
from stacking_common.search import ActionGenerationRequest, ExecutionRejections
from stacking_common.search import ActionCandidate, ActionGenerationResult
from stacking_common.domain.geometry import quaternion_matrix
from stacking_tabletop.planning.safety import TabletopPlacementSafety
from stacking_common.actions.scored import ScoredPlanarPoseSampler


def _tabletop_floor_policy(inventory):
    return TabletopPlacementSafety(inventory, bottom_tolerance=.02, fill_ratio=.9,
        grid_size=32, max_layers=8, minimum_contact_cells=1, coverage_mode='bottom_slice')


class _RecoveryGenerator:
    def __init__(self, total=6):
        self.total = total
        self.offsets = []
        self.clock = 0.0

    def generate(self, state, observation, request):
        from stacking_common.domain import Action
        self.offsets.append(request.attempt_offset)
        start = min(request.attempt_offset, self.total)
        end = min(start + request.max_candidates, self.total)
        candidates = tuple(ActionCandidate(Action(0, _pose(x=i), _pose(x=i), object_id=10))
                           for i in range(start, end) if i >= 4)
        rejected = end - start - len(candidates)
        self.clock += 1.0
        return ActionGenerationResult(candidates, end - start, end == self.total,
            {"active_floor_above_active_layer": rejected} if rejected else {})

    def build_candidates(self, state, proposals):
        assert not proposals
        return ()


def _recovery_cem(base, iterations=3):
    return CEMRootActionGenerator(base, _ConstantScore(), CEMRootConfig(
        population=2, elite=1, iterations=iterations, keep=1,
        mutation_xy_std=0.0, mutation_yaw_std=0.0, fresh_fraction=0.0,
        duplicate_xy_threshold=0.0, duplicate_yaw_threshold=0.0))


def test_cem_empty_pool_consumes_fresh_proposals_without_repeating_prefix():
    base = _RecoveryGenerator()
    cem = _recovery_cem(base)
    result = cem.generate(None, None, ActionGenerationRequest(max_candidates=1))
    assert len(result.candidates) == 1
    assert base.offsets == [0, 2, 4]
    assert cem.last_statistics.attempted == 6
    assert cem.last_statistics.rejection_counts == {"active_floor_above_active_layer": 4}
    assert cem.last_statistics.fresh_exhausted
    assert [r.mutation_attempts for r in cem.last_statistics.rounds] == [0, 0, 0]


def test_cem_execution_filter_precedes_keep_and_preserves_alternative():
    from stacking_common.domain import Action

    cem = _recovery_cem(_RecoveryGenerator())
    rejection = Action(0, _pose(x=4), _pose(x=4), object_id=10)
    result = cem.generate(None, None, ActionGenerationRequest(
        max_candidates=1,
        execution_rejections=ExecutionRejections(actions=(rejection,)),
    ))
    assert len(result.candidates) == 1
    assert result.candidates[0].action.pose[0] == 5
    assert cem.last_statistics.final_candidates == 1
    assert cem.last_statistics.rejection_counts["execution_rejected_action"] == 1


def test_cem_execution_object_filter_reports_empty_pool():
    cem = _recovery_cem(_RecoveryGenerator())
    result = cem.generate(None, None, ActionGenerationRequest(
        max_candidates=1,
        execution_rejections=ExecutionRejections(object_ids={10}),
    ))
    assert not result.candidates
    assert result.attempts == 0
    assert cem.last_statistics.rejection_counts["execution_rejected_object"] == 2


def test_cem_empty_pool_stops_when_fresh_stream_is_exhausted():
    base = _RecoveryGenerator(total=3)
    cem = _recovery_cem(base, iterations=10)
    result = cem.generate(None, None, ActionGenerationRequest(max_candidates=1))
    assert not result.candidates
    assert base.offsets == [0, 2]
    assert cem.last_statistics.attempted == 3
    assert cem.last_statistics.fresh_exhausted


def test_cem_recovery_obeys_deadline_and_retry_continues_untried_tail(monkeypatch):
    base = _RecoveryGenerator()
    cem = _recovery_cem(base)
    monkeypatch.setattr("stacking_common.actions.cem.time.monotonic", lambda: base.clock)
    cem.generate(None, None, ActionGenerationRequest(max_candidates=1, deadline=1.0))
    assert base.offsets == [0]
    assert cem.last_statistics.timed_out
    assert not cem.last_statistics.fresh_exhausted
    cem.begin_retry(())
    result = cem.generate(None, None, ActionGenerationRequest(max_candidates=1, deadline=3.0))
    assert result.candidates
    assert base.offsets == [0, 2, 4]
    assert len(cem.statistics_history) == 2
    cem.reset_statistics()
    assert not cem.statistics_history


def test_cem_expired_deadline_does_not_start_native_work(monkeypatch):
    base = _RecoveryGenerator()
    cem = _recovery_cem(base)
    monkeypatch.setattr("stacking_common.actions.cem.time.monotonic", lambda: 10.0)
    result = cem.generate(None, None, ActionGenerationRequest(max_candidates=1, deadline=9.0))
    assert not result.candidates
    assert not base.offsets
    assert cem.last_statistics.timed_out


def test_early_floor_filter_matches_final_validation_for_floating_and_disconnected_poses():
    inventory = _inventory()
    state = _state(inventory, placed=(0,))
    policy = _tabletop_floor_policy(inventory)
    early = ActiveFloorFilter(policy)
    from stacking_common.domain import Action
    for pose, expected in [(_pose(z=0.08), "above_active_layer"),
                           (_pose(x=0.1, z=0.02), "disconnected_active_layer")]:
        initial = InitializedPlacement(pose, 0.0, ())
        solved = PlacementSolveResult(pose, c_feq=0.0, c_gap=0.0)
        decision = early.evaluate(inventory, state, 1, initial, solved)
        reason, _ = policy.active_floor_rejection(state, Action(1, pose, pose, object_id=20))
        assert reason == expected
        assert not decision.accepted
        assert decision.reason == f"active_floor_{expected}"


def test_filled_floor_advances_to_next_layer_without_lowering_threshold():
    from stacking_common.domain import Action
    inventory = _inventory()
    target = TargetGeometry(width=0.04, length=0.04, height=0.2,
        origin=np.zeros(3), taper=DirectionalTaper(), support_overlap=0.0)
    inventory = ObjectInventory([inventory[0], inventory[1]],
        {10: _pose(), 20: _pose()}, target)
    state = _state(inventory, placed=(0,))
    policy = _tabletop_floor_policy(inventory)
    support_z, occupied = policy.context(state)
    assert support_z == 0.04
    assert not np.any(occupied)
    assert policy.fill_ratio == 0.90
    pose = _pose(z=0.06)
    reason, _ = policy.active_floor_rejection(state, Action(1, pose, pose, object_id=20))
    assert reason is None
    assert state.placed_object_indices == [0]


@pytest.mark.parametrize("strict_floor_first", [False, True])
def test_tabletop_generator_captured_step4_floor_gate_is_optional(strict_floor_first):
    import json
    from dataclasses import replace
    from pathlib import Path
    from stacking_tabletop import TabletopApplication

    root = Path(__file__).resolve().parents[1]
    fixture = json.loads((root / "tests/fixtures/planning/active_floor_step4.json").read_text())
    app = TabletopApplication.from_repository(root)
    assert app.action_generation_profile.strict_floor_first
    app = replace(app, action_generation_profile=replace(
        app.action_generation_profile, strict_floor_first=strict_floor_first))
    ids = fixture["object_ids"]
    inventory = app.load_inventory(ids, {object_id: _pose() for object_id in ids})
    state = State(ids, [ids.index(o["object_id"]) for o in fixture["placed"]],
        {o["object_id"]: np.array(o["pose"]) for o in fixture["placed"]}, [], [])

    class RecordedSolver:
        pose = None

        def solve(self, *_args):
            return PlacementSolveResult(self.pose, c_feq=0.0, c_gap=0.0)

    solver = RecordedSolver()
    generator = app.build_action_generator(inventory, solver)
    # This floor-only fixture has no measured pick orientations. Its synthetic
    # identity poses cannot stand in for the scan-relative orientation check.
    generator.filters = tuple(f for f in generator.filters if not isinstance(f, ScanOrientationFilter))
    assert len(fixture["candidates"]) == 10
    for recorded in fixture["candidates"]:
        solver.pose = np.array(recorded["pose"])
        candidate, reason = generator.build_candidate(state,
            ids.index(recorded["object_id"]), PlanarPlacement(np.array(recorded["init_pose"])))
        if strict_floor_first:
            assert candidate is None
            assert reason == recorded["reason"] == "active_floor_above_active_layer"
        else:
            assert candidate is not None
            assert reason is None
    # Floor-aware proposal scoring stays enabled in either mode.
    assert isinstance(generator.planar_sampler.floor_policy, TabletopPlacementSafety)
    generator.close()


@dataclass
class _BodyModel:
    mass: float = 1.0
    inertia: np.ndarray = None

    def __post_init__(self) -> None:
        if self.inertia is None:
            self.inertia = np.eye(3)


def _pose(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    return np.array([x, y, z, 0.0, 0.0, 0.0, 1.0])


def _cube(size: float) -> ConvexGeometry:
    half = size / 2.0
    points = np.array(
        [
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
        ]
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
    )
    return ConvexGeometry(points, faces)


def _inventory(count: int = 2) -> ObjectInventory:
    objects = [
        StackObject(10 * (index + 1), _BodyModel(), [_cube(0.04)])
        for index in range(count)
    ]
    target = TargetGeometry(
        width=0.30,
        length=0.30,
        height=0.20,
        origin=np.zeros(3),
        taper=DirectionalTaper(),
        support_overlap=0.0,
    )
    return ObjectInventory(
        objects,
        {stack_object.object_id: _pose(1.0) for stack_object in objects},
        target,
    )


def _state(inventory: ObjectInventory, placed=()) -> State:
    poses = {
        inventory[index].object_id: _pose(z=0.02)
        for index in placed
    }
    return State(
        object_ids=list(inventory.object_ids),
        placed_object_indices=list(placed),
        object_poses=poses,
        trajectories=[],
        action_history=[],
    )


def test_mesh_height_initializer_places_above_highest_xy_support() -> None:
    inventory = _inventory()
    initializer = MeshHeightInitializer(clearance=0.01, xy_padding=0.0)

    result = initializer.initialize(
        inventory,
        _state(inventory, placed=(0,)),
        object_index=1,
        proposal=PlanarPlacement(_pose()),
    )

    assert np.isclose(result.support_height, 0.04)
    assert result.support_object_indices == (0,)
    assert np.isclose(result.pose[2], 0.07)


class _PlanarSampler:
    def sample(
        self,
        inventory,
        state,
        observation,
        object_index,
        count,
        sampling_priority,
    ):
        del inventory, state, observation, sampling_priority
        return tuple(
            PlanarPlacement(
                _pose(x=0.01 * object_index, y=0.01 * pose_index),
                score=float(pose_index),
            )
            for pose_index in range(count)
        )


class _Initializer:
    def initialize(self, inventory, state, object_index, proposal):
        del inventory, state, object_index
        pose = proposal.pose.copy()
        pose[2] = 0.05
        return InitializedPlacement(pose, support_height=0.0)


class _PoseSolver:
    def solve(self, inventory, state, object_index, initial):
        del inventory, state
        return PlacementSolveResult(
            pose=initial.pose,
            c_feq=0.25,
            c_gap=0.002 if object_index == 1 else 0.0,
            diagnostics={"solver": "fake"},
        )


class _ConcurrentPoseSolver:
    def __init__(self, worker: int, barrier: threading.Barrier) -> None:
        self.worker = worker
        self.barrier = barrier

    def solve(self, inventory, state, object_index, initial):
        del inventory, state, object_index
        self.barrier.wait(timeout=2.0)
        return PlacementSolveResult(
            pose=initial.pose,
            c_feq=0.0,
            c_gap=0.0,
            diagnostics={"pose_worker": self.worker},
        )


def test_generator_round_robins_filters_and_resumes_from_attempt_offset() -> None:
    inventory = _inventory()
    config = PlacementActionGeneratorConfig(
        max_poses_per_object=2,
        initial_height_clearance=0.01,
        initial_height_xy_padding=0.0,
        posegen_gap_threshold=0.001,
    )
    generator = PlacementActionGenerator(
        inventory,
        config,
        _PlanarSampler(),
        _Initializer(),
        _PoseSolver(),
    )
    state = _state(inventory)

    first = generator.generate(
        state,
        None,
        ActionGenerationRequest(max_candidates=2),
    )
    second = generator.generate(
        state,
        None,
        ActionGenerationRequest(max_candidates=2, attempt_offset=2),
    )

    assert first.attempts == second.attempts == 2
    assert first.rejection_counts == second.rejection_counts == {
        "posegen_gap": 1
    }
    assert not first.exhausted
    assert second.exhausted
    assert [value.action.object_id for value in first.candidates] == [10]
    assert [value.action.object_id for value in second.candidates] == [10]
    assert first.candidates[0].action.diagnostics["solver"] == "fake"
    assert second.candidates[0].action.pose[1] == 0.01


def test_target_containment_filter_rejects_before_simulation() -> None:
    inventory = _inventory()
    containment = {10: 0.25, 20: 0.75}
    overlap = SimpleNamespace(
        object_containment=lambda stack_object, pose: containment[
            stack_object.object_id
        ]
    )
    placement_filter = TargetContainmentFilter(overlap, threshold=0.5)
    initial = InitializedPlacement(_pose(z=0.05), support_height=0.0)

    rejected = placement_filter.evaluate(
        inventory,
        _state(inventory),
        0,
        initial,
        PlacementSolveResult(_pose(z=0.05), c_feq=0.0, c_gap=0.0),
    )
    accepted = placement_filter.evaluate(
        inventory,
        _state(inventory),
        1,
        initial,
        PlacementSolveResult(_pose(z=0.05), c_feq=0.0, c_gap=0.0),
    )

    assert not rejected.accepted
    assert rejected.reason == "stone_outside_target"
    assert rejected.diagnostics[
        "presimulation_target_containment_fraction"
    ] == 0.25
    assert accepted.accepted


def test_generator_parallelizes_pose_solves_and_preserves_order() -> None:
    inventory = _inventory()
    barrier = threading.Barrier(2)
    solvers = [_ConcurrentPoseSolver(index, barrier) for index in range(2)]
    generator = PlacementActionGenerator(
        inventory,
        PlacementActionGeneratorConfig(
            max_poses_per_object=1,
            initial_height_clearance=0.01,
            initial_height_xy_padding=0.0,
            posegen_gap_threshold=None,
        ),
        _PlanarSampler(),
        _Initializer(),
        solvers[0],
        worker_pose_solvers=solvers[1:],
    )

    result = generator.generate(
        _state(inventory), None, ActionGenerationRequest(max_candidates=2)
    )

    assert [candidate.action.object_id for candidate in result.candidates] == [10, 20]
    assert [
        candidate.diagnostics["pose_worker"] for candidate in result.candidates
    ] == [0, 1]
    generator.close()
    assert generator._executor is None


def test_generator_parallelizes_pose_solves_across_search_nodes() -> None:
    inventory = _inventory()
    barrier = threading.Barrier(2)
    solvers = [_ConcurrentPoseSolver(index, barrier) for index in range(2)]
    generator = PlacementActionGenerator(
        inventory,
        PlacementActionGeneratorConfig(
            max_poses_per_object=1,
            initial_height_clearance=0.01,
            initial_height_xy_padding=0.0,
            posegen_gap_threshold=None,
        ),
        _PlanarSampler(),
        _Initializer(),
        solvers[0],
        worker_pose_solvers=solvers[1:],
    )

    results = generator.generate_batch(
        (_state(inventory), _state(inventory)),
        (None, None),
        (
            ActionGenerationRequest(max_candidates=1),
            ActionGenerationRequest(max_candidates=1),
        ),
    )

    assert [result.attempts for result in results] == [1, 1]
    assert [
        result.candidates[0].diagnostics["pose_worker"]
        for result in results
    ] == [0, 1]
    generator.close()


def test_planar_sampler_uses_measured_orientation_provider() -> None:
    from pathlib import Path
    from stacking_tabletop import TabletopApplication

    inventory = _inventory(1)
    measured = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    app = TabletopApplication.from_repository(Path(__file__).resolve().parents[1])
    sampler = ScoredPlanarPoseSampler(
        inventory,
        PlanarSamplingConfig((3, 3), (0.0,), 0.0, 0.6),
        orientation_provider=MappingOrientationProvider({10: measured}),
        settings=app.sampling_settings,
        floor_policy=_tabletop_floor_policy(inventory),
    )
    observation = SimpleNamespace(
        scene_height_map=np.zeros((3, 3)),
        height_map_xlim=np.array([-0.15, 0.15]),
        height_map_ylim=np.array([-0.15, 0.15]),
    )

    proposal = sampler.sample(
        inventory, _state(inventory), observation, 0, 1, None
    )[0]

    np.testing.assert_allclose(proposal.pose[3:], measured)


def test_scan_orientation_yaw_aligns_longest_axis() -> None:
    geometry = _cube(0.04)
    points = geometry.points.copy()
    points[:, 0] *= 3.0
    elongated = ConvexGeometry(points, geometry.faces)
    stack_object = StackObject(10, _BodyModel(), [elongated])
    yaw = np.deg2rad(60.0) / 2.0
    measured = np.array([0.0, 0.0, np.sin(yaw), np.cos(yaw)])
    inventory = ObjectInventory(
        [stack_object],
        {10: np.array([0.0, 0.0, 0.0, *measured])},
        _inventory(1).target,
    )

    quaternion = ScanAlignedOrientationProvider().quaternion(
        inventory,
        _state(inventory),
        None,
        0,
    )
    longest_world = quaternion_matrix(quaternion) @ np.array([1.0, 0.0, 0.0])

    assert np.isclose(longest_world[1], 0.0, atol=1e-9)


def test_scan_orientation_matches_legacy_global_hull_pca() -> None:
    # Extra interior/component vertices must not bias the outer hull's PCA.
    rng = np.random.default_rng(17)
    outer = rng.normal(size=(40, 3)) * [3.0, 1.0, 0.5]
    interior = rng.normal(scale=0.03, size=(100, 3)) + [0.5, 0.5, 0.0]
    points = np.concatenate((outer, interior))
    scan = Rotation.from_euler("xyz", [25, -15, 63], degrees=True)
    stack_object = SimpleNamespace(object_id=10, mesh_arrays=lambda: (points, None))
    # Use a minimal inventory implementing indexing and measured-pose lookup.
    class Inventory:
        def __getitem__(self, index):
            return stack_object

        def initial_pose(self, object_id):
            return np.r_[np.zeros(3), scan.as_quat()]

    inventory = Inventory()
    hull = points[ConvexHull(points).vertices]
    centered = hull - hull.mean(axis=0)
    _, axes = np.linalg.eigh(centered.T @ centered / len(hull))
    longest = axes[:, -1]
    if longest[np.argmax(np.abs(longest))] < 0:
        longest = -longest
    world = scan.apply(longest)
    heading = np.arctan2(world[1], world[0])
    yaw = -((heading + np.pi / 2) % np.pi - np.pi / 2)
    expected = Rotation.from_euler("z", yaw) * scan
    actual = ScanAlignedOrientationProvider().quaternion(inventory, None, None, 0)
    np.testing.assert_allclose(Rotation.from_quat(actual).as_matrix(), expected.as_matrix(), atol=1e-12)

    # Removing all interior vertices gives exactly the same initialized pose.
    stack_object.mesh_arrays = lambda: (hull, None)
    repeated = ScanAlignedOrientationProvider().quaternion(inventory, None, None, 0)
    np.testing.assert_allclose(Rotation.from_quat(repeated).as_matrix(), expected.as_matrix(), atol=1e-12)


@pytest.mark.parametrize("rotation", [(0, 0, 180), (44.9, 0, 0), (45.1, 0, 0), (0, 180, 0), (35, 25, 130)])
@pytest.mark.parametrize("sign", [1, -1])
def test_scan_orientation_filter_matches_legacy_relative_rotation(rotation, sign):
    inventory = _inventory(1)
    scan = Rotation.from_euler("xyz", [20, -30, 50], degrees=True)
    inventory = ObjectInventory(list(inventory), {10: np.r_[np.zeros(3), scan.as_quat()]}, inventory.target)
    relative = Rotation.from_euler("xyz", rotation, degrees=True)
    pose = np.r_[0., 0., .2, sign * (relative * scan).as_quat()]
    decision = ScanOrientationFilter(_tabletop_floor_policy(inventory)).evaluate(
        inventory, _state(inventory), 0, InitializedPlacement(_pose(), 0.),
        PlacementSolveResult(pose, 0., 0.),
    )
    # Legacy uses the XY rotation-vector norm, not an Euler-angle cutoff.
    norm = float(np.linalg.norm(relative.as_rotvec()[:2]))
    zz = float(relative.as_matrix()[2, 2])
    expected = not (norm > np.pi / 4 or zz < 0)
    assert decision.accepted == expected
    assert decision.reason == (None if expected else "scan_orientation")
    assert decision.diagnostics["scan_orientation_rotvec_xy_norm"] == pytest.approx(norm)
    assert not decision.diagnostics["lower_fill_candidate"]


@pytest.mark.parametrize("angle", [90, 180])
def test_scan_orientation_lower_fill_exception_uses_solved_pose(angle):
    inventory = _inventory(1)
    state = _state(inventory)
    policy = _tabletop_floor_policy(inventory)
    orientation_filter = ScanOrientationFilter(policy)
    quaternion = Rotation.from_euler("x", angle, degrees=True).as_quat()
    points, _ = inventory[0].mesh_arrays()
    bottom = np.min(Rotation.from_quat(quaternion).apply(points)[:, 2])
    pose = np.r_[0., 0., -bottom, quaternion]
    initial = InitializedPlacement(_pose(z=.2), 0.)
    decision = orientation_filter.evaluate(inventory, state, 0, initial, PlacementSolveResult(pose, 0., 0.))
    assert decision.accepted
    assert not decision.diagnostics["scan_orientation_ok"]
    assert decision.diagnostics["lower_fill_candidate"]
    assert decision.diagnostics["scan_orientation_soft_lower_fill"]

    pose[2] += .2
    decision = orientation_filter.evaluate(inventory, state, 0, initial, PlacementSolveResult(pose, 0., 0.))
    assert not decision.accepted
    assert decision.reason == "scan_orientation"


def test_scan_orientation_filters_fresh_and_explicit_mutation_builds():
    inventory = _inventory(1)
    pose = np.r_[0., 0., .2, Rotation.from_euler("x", 90, degrees=True).as_quat()]
    solver = SimpleNamespace(solve=lambda *args: PlacementSolveResult(pose, 0., 0.))
    generator = PlacementActionGenerator(
        inventory, PlacementActionGeneratorConfig(
            max_poses_per_object=2, initial_height_clearance=.015,
            initial_height_xy_padding=.012, posegen_gap_threshold=.003,
        ),
        _PlanarSampler(), _Initializer(), solver,
        filters=(ScanOrientationFilter(_tabletop_floor_policy(inventory)),),
        worker_pose_solvers=(solver,),
    )
    try:
        state = _state(inventory)
        fresh = generator.generate(state, None, ActionGenerationRequest(max_candidates=2))
        assert not fresh.candidates
        assert fresh.rejection_counts == {"scan_orientation": 2}
        # CEM mutations go through this same explicit-proposal batch API.
        mutations = generator.build_candidates(state, ((0, PlanarPlacement(_pose())),) * 2)
        assert mutations == ((None, "scan_orientation"),) * 2
    finally:
        generator.close()


class _PlanarScore:
    def score(self, state, observation, candidates):
        del state, observation
        return [candidate.diagnostics["planar_score"] for candidate in candidates]


class _ConstantScore:
    def score(self, state, observation, candidates):
        del state, observation
        return [0.0] * len(candidates)


class _RescoringPlanarSampler(_PlanarSampler):
    def __init__(self) -> None:
        self.rescored_poses = []

    def score_at_pose(self, observation, pose):
        del observation
        self.rescored_poses.append(pose.copy())
        return 10.0, 1.0


def test_cem_root_generator_is_deterministic_diverse_and_bounded() -> None:
    inventory = _inventory()
    base = PlacementActionGenerator(
        inventory,
        PlacementActionGeneratorConfig(
            max_poses_per_object=3,
            initial_height_clearance=0.01,
            initial_height_xy_padding=0.0,
            posegen_gap_threshold=None,
        ),
        _PlanarSampler(),
        _Initializer(),
        _PoseSolver(),
    )
    config = CEMRootConfig(
        population=6,
        elite=2,
        iterations=2,
        keep=4,
        mutation_xy_std=0.002,
        mutation_yaw_std=0.01,
        fresh_fraction=0.25,
        duplicate_xy_threshold=0.0001,
        duplicate_yaw_threshold=0.0001,
        max_per_object=2,
        seed=7,
    )

    first_generator = CEMRootActionGenerator(base, _PlanarScore(), config)
    first = first_generator.generate(
        _state(inventory), None, ActionGenerationRequest(max_candidates=4)
    )
    second = CEMRootActionGenerator(base, _PlanarScore(), config).generate(
        _state(inventory), None, ActionGenerationRequest(max_candidates=4)
    )

    assert len(first.candidates) <= 4
    assert all(
        sum(
            value.action.object_id == candidate.action.object_id
            for value in first.candidates
        )
        <= 2
        for candidate in first.candidates
    )
    for left, right in zip(first.candidates, second.candidates):
        np.testing.assert_allclose(left.action.init_pose, right.action.init_pose)
    statistics = first_generator.last_statistics
    assert len(statistics.rounds) == 2
    assert statistics.attempted > config.population
    assert statistics.final_candidates == len(first.candidates)
    assert all(
        "cem_score" in candidate.diagnostics for candidate in first.candidates
    )


def test_cem_adds_normalized_height_map_bonus_to_root_ranking() -> None:
    inventory = _inventory()
    base = PlacementActionGenerator(
        inventory,
        PlacementActionGeneratorConfig(
            max_poses_per_object=3,
            initial_height_clearance=0.01,
            initial_height_xy_padding=0.0,
            posegen_gap_threshold=None,
        ),
        _PlanarSampler(),
        _Initializer(),
        _PoseSolver(),
    )
    generator = CEMRootActionGenerator(
        base,
        _ConstantScore(),
        CEMRootConfig(
            population=6,
            elite=2,
            iterations=1,
            keep=1,
            mutation_xy_std=0.0,
            mutation_yaw_std=0.0,
            fresh_fraction=0.0,
            duplicate_xy_threshold=0.0,
            duplicate_yaw_threshold=0.0,
            height_map_score_weight=1.0,
        ),
    )

    result = generator.generate(
        _state(inventory), None, ActionGenerationRequest(max_candidates=1)
    )

    candidate = result.candidates[0]
    assert candidate.diagnostics["planar_score"] == 2.0
    assert candidate.diagnostics["cem_height_map_score_normalized"] == 1.0
    assert candidate.diagnostics["cem_height_map_score_bonus"] == 1.0


def test_cem_recomputes_planar_score_after_mutating_pose() -> None:
    inventory = _inventory(1)
    sampler = _RescoringPlanarSampler()
    base = PlacementActionGenerator(
        inventory,
        PlacementActionGeneratorConfig(
            max_poses_per_object=2,
            initial_height_clearance=0.01,
            initial_height_xy_padding=0.0,
            posegen_gap_threshold=None,
        ),
        sampler,
        _Initializer(),
        _PoseSolver(),
    )
    generator = CEMRootActionGenerator(
        base,
        _ConstantScore(),
        CEMRootConfig(
            population=2,
            elite=1,
            iterations=2,
            keep=1,
            mutation_xy_std=0.01,
            mutation_yaw_std=0.01,
            fresh_fraction=0.0,
            duplicate_xy_threshold=0.0,
            duplicate_yaw_threshold=0.0,
            height_map_score_weight=1.0,
            seed=4,
        ),
    )

    result = generator.generate(
        _state(inventory), None, ActionGenerationRequest(max_candidates=1)
    )

    candidate = result.candidates[0]
    assert sampler.rescored_poses
    assert candidate.diagnostics["source"] == "cem_mutation"
    assert candidate.diagnostics["planar_score"] == 10.0
    assert candidate.diagnostics["cem_height_map_score_normalized"] == 1.0


def test_cem_fills_exhausted_fresh_quota_with_parent_mutations() -> None:
    inventory = _inventory(1)
    base = PlacementActionGenerator(
        inventory,
        PlacementActionGeneratorConfig(
            max_poses_per_object=1,
            initial_height_clearance=0.01,
            initial_height_xy_padding=0.0,
            posegen_gap_threshold=None,
        ),
        _PlanarSampler(),
        _Initializer(),
        _PoseSolver(),
    )
    generator = CEMRootActionGenerator(
        base,
        _ConstantScore(),
        CEMRootConfig(
            population=4,
            elite=1,
            iterations=2,
            keep=4,
            mutation_xy_std=0.01,
            mutation_yaw_std=0.01,
            fresh_fraction=0.5,
            duplicate_xy_threshold=0.0,
            duplicate_yaw_threshold=0.0,
            seed=5,
        ),
    )

    result = generator.generate(
        _state(inventory), None, ActionGenerationRequest(max_candidates=4)
    )

    second_round = generator.last_statistics.rounds[1]
    assert second_round.mutation_attempts == 2
    assert second_round.fresh_attempts == 0
    assert second_round.fresh_fallback_attempts == 2
    assert second_round.attempted == 4
    assert any(
        candidate.diagnostics.get("source") == "cem_fresh_fallback"
        for candidate in result.candidates
    )


def test_cem_fresh_round_starts_after_initial_population() -> None:
    inventory = _inventory(1)
    base = PlacementActionGenerator(
        inventory,
        PlacementActionGeneratorConfig(
            max_poses_per_object=4,
            initial_height_clearance=0.01,
            initial_height_xy_padding=0.0,
            posegen_gap_threshold=None,
        ),
        _PlanarSampler(),
        _Initializer(),
        _PoseSolver(),
    )
    generator = CEMRootActionGenerator(
        base,
        _PlanarScore(),
        CEMRootConfig(
            population=2,
            elite=1,
            iterations=2,
            keep=2,
            mutation_xy_std=0.0,
            mutation_yaw_std=0.01,
            fresh_fraction=0.5,
            duplicate_xy_threshold=0.0,
            duplicate_yaw_threshold=0.0,
            seed=6,
        ),
    )

    result = generator.generate(
        _state(inventory), None, ActionGenerationRequest(max_candidates=2)
    )

    assert generator.last_statistics.rounds[1].fresh_attempts == 1
    assert any(
        candidate.diagnostics["planar_score"] == 2.0
        for candidate in result.candidates
    )


def test_cem_retry_changes_seed_and_excludes_previous_roots() -> None:
    inventory = _inventory(1)
    base = PlacementActionGenerator(
        inventory,
        PlacementActionGeneratorConfig(
            max_poses_per_object=2,
            initial_height_clearance=0.01,
            initial_height_xy_padding=0.0,
            posegen_gap_threshold=None,
        ),
        _PlanarSampler(),
        _Initializer(),
        _PoseSolver(),
    )
    generator = CEMRootActionGenerator(
        base,
        _PlanarScore(),
        CEMRootConfig(
            population=2,
            elite=1,
            iterations=2,
            keep=2,
            mutation_xy_std=0.01,
            mutation_yaw_std=0.01,
            fresh_fraction=0.0,
            duplicate_xy_threshold=0.0001,
            duplicate_yaw_threshold=0.0001,
            seed=6,
        ),
    )
    state = _state(inventory)
    request = ActionGenerationRequest(max_candidates=2)
    initial = generator.generate(state, None, request)

    generator.begin_retry(
        candidate.action for candidate in initial.candidates
    )
    retry = generator.generate(state, None, request)

    assert retry.candidates
    assert generator.last_statistics.retry_round == 1
    assert (
        generator.last_statistics.rejection_counts[
            "retry_excluded_duplicate"
        ]
        >= 1
    )
    assert all(
        not any(
            np.allclose(
                candidate.action.init_pose,
                previous.action.init_pose,
                atol=1e-6,
            )
            for previous in initial.candidates
        )
        for candidate in retry.candidates
    )


def test_batched_solver_keeps_initial_rejections_order_and_filters():
    inventory = _inventory(3)
    calls = []

    class BatchSolver(_PoseSolver):
        def solve_batch(self, inventory, requests):
            calls.append(tuple(requests))
            return [_PoseSolver.solve(self, inventory, *job) for job in requests]

    generator = PlacementActionGenerator(inventory, PlacementActionGeneratorConfig(
        max_poses_per_object=1, initial_height_clearance=.01, initial_height_xy_padding=0.,
        posegen_gap_threshold=.001, banned_object_ids=(30,),
    ), _PlanarSampler(), _Initializer(), BatchSolver())
    state = _state(inventory)
    result = generator.build_candidates(state, [(2, PlanarPlacement(_pose())),
        (1, PlanarPlacement(_pose())), (0, PlanarPlacement(_pose()))])
    assert [job[1] for job in calls[0]] == [1, 0]
    assert result[0] == (None, 'banned_object')
    assert result[1] == (None, 'posegen_gap')
    assert result[2][0].action.object_id == 10
    assert generator._executor is None
    assert generator.build_candidates(state, []) == ()
    assert len(calls) == 1
    assert generator.build_candidates(_state(inventory, (0,)), [(0, PlanarPlacement(_pose()))]) == (
        (None, 'object_already_placed'),)
    assert len(calls) == 1


def test_batched_solver_combines_multiple_states():
    inventory = _inventory()
    calls = []

    class BatchSolver(_PoseSolver):
        def solve_batch(self, inventory, requests):
            calls.append(tuple(requests))
            return [_PoseSolver.solve(self, inventory, *job) for job in requests]

    generator = PlacementActionGenerator(inventory, PlacementActionGeneratorConfig(
        max_poses_per_object=1, initial_height_clearance=.01, initial_height_xy_padding=0.,
        posegen_gap_threshold=None,
    ), _PlanarSampler(), _Initializer(), BatchSolver())
    states = (_state(inventory), _state(inventory, (0,)))
    results = generator.generate_batch(states, (None, None), (
        ActionGenerationRequest(max_candidates=1), ActionGenerationRequest(max_candidates=1)))
    assert len(calls) == 1
    assert [job[0] for job in calls[0]] == list(states)
    assert [result.candidates[0].action.object_id for result in results] == [10, 20]


@pytest.mark.parametrize('require_convergence', [False, True])
def test_invalid_solver_result_cannot_become_an_action(require_convergence):
    inventory = _inventory()
    generator = PlacementActionGenerator(inventory, PlacementActionGeneratorConfig(
        max_poses_per_object=1, initial_height_clearance=.01, initial_height_xy_padding=0.,
        posegen_gap_threshold=None,
    ), _PlanarSampler(), _Initializer(), _PoseSolver())
    generator.filters = ()  # Unconditional rejection must not rely on filters.
    failed = PlacementSolveResult(_pose(), 0., 0., converged=False,
                                  rejection_reason='posegen_geometry_failure', valid=False)
    assert generator._finish_candidate(_state(inventory), 0, PlanarPlacement(_pose()),
                                       InitializedPlacement(_pose(), 0.), failed) == (
        None, 'posegen_geometry_failure')
    from stacking_common.actions.filtering import PosegenFeasibilityFilter
    assert not PosegenFeasibilityFilter(None, None,
        require_solver_convergence=require_convergence).evaluate(None, None, None, None, failed).accepted


def test_invalid_solver_result_requires_explicit_failure():
    with pytest.raises(ValueError, match='invalid placement solve'):
        PlacementSolveResult(_pose(), 0., 0., valid=False)
    with pytest.raises(ValueError, match='invalid placement solve'):
        PlacementSolveResult(_pose(), 0., 0., converged=False, valid=False)


def test_batched_ply_bounds_match_scalar_and_refresh_after_state_changes(tmp_path, monkeypatch):
    from stacking_common.actions.ply_height import PlyHeightInitializer

    inventory = _inventory(3)
    initializer = PlyHeightInitializer(tmp_path, clearance=.01, xy_padding=.005)
    first = _state(inventory, (0, 1))  # Equal-height supports must retain their order.
    second = first.copy()
    second.object_poses[10][0] = 1.
    second.object_poses[20][2] = .08
    empty = _state(inventory)
    rotated = _pose(x=.01)
    rotated[3:] = Rotation.from_euler('xyz', [.1, .2, .3]).as_quat()
    jobs = [(0, first, 2, PlanarPlacement(rotated)),
            (1, second, 2, PlanarPlacement(_pose())),
            (2, empty, 2, PlanarPlacement(_pose())),
            (0, first, 2, PlanarPlacement(_pose())),
            (1, second, 2, PlanarPlacement(rotated))]

    class BatchSolver:
        fail = False
        requests = ()

        def solve_batch(self, inventory, requests):
            self.requests = tuple(requests)
            if self.fail:
                raise RuntimeError('test solver failure')
            return [PlacementSolveResult(initial.pose, 0., 0.) for _, _, initial in requests]

    solver = BatchSolver()
    generator = PlacementActionGenerator(inventory, PlacementActionGeneratorConfig(
        max_poses_per_object=1, initial_height_clearance=.01, initial_height_xy_padding=.005,
        posegen_gap_threshold=None,
    ), _PlanarSampler(), initializer, solver)
    calls = []
    prepare = initializer.prepare_scene

    def counted(inventory, state):
        calls.append(state)
        return prepare(inventory, state)

    monkeypatch.setattr(initializer, 'prepare_scene', counted)

    def check():
        expected = [initializer.initialize(inventory, state, index, proposal)
                    for _, state, index, proposal in jobs]
        calls.clear()
        actual = generator._build_jobs(jobs)
        assert [id(state) for state in calls] == [id(first), id(second), id(empty)]
        for (candidate, reason), (_, _, initial), ref in zip(actual, solver.requests, expected):
            assert reason is None
            np.testing.assert_array_equal(candidate.action.pose, ref.pose)
            np.testing.assert_array_equal(initial.pose, ref.pose)
            assert initial.support_height == ref.support_height
            assert initial.support_object_indices == ref.support_object_indices
            assert initial.diagnostics == ref.diagnostics
        return actual

    before = check()
    assert solver.requests[3][2].support_object_indices == (0, 1)
    first.object_poses[10][2] += .1  # Same State instance, new contents next batch.
    after = check()
    assert after[0][0].action.pose[2] > before[0][0].action.pose[2]
    solver.fail = True
    with pytest.raises(RuntimeError, match='test solver failure'):
        generator._build_jobs(jobs)
    first.object_poses[10][2] += .1
    solver.fail = False
    check()  # A failed batch must not retain prepared bounds either.


@pytest.mark.parametrize('coverage_mode', ['bottom_slice', 'projected_footprint'])
def test_batched_floor_context_matches_scalar_for_mixed_states_and_refreshes(coverage_mode, monkeypatch):
    inventory = _inventory()
    policy = _tabletop_floor_policy(inventory)
    policy.coverage_mode = coverage_mode
    first = _state(inventory, (0,))
    second = first.copy()
    second.object_poses[10][0] = -.1
    empty = _state(inventory)

    class Initializer:
        def initialize(self, inventory, state, index, proposal):
            return InitializedPlacement(proposal.pose, 0.)

    class BatchSolver(_PoseSolver):
        def solve_batch(self, inventory, requests):
            return [self.solve(inventory, *request) for request in requests]

    solver = BatchSolver()
    generator = PlacementActionGenerator(inventory, PlacementActionGeneratorConfig(
        max_poses_per_object=1, initial_height_clearance=.01, initial_height_xy_padding=0.,
        posegen_gap_threshold=None,
    ), _PlanarSampler(), Initializer(), solver, filters=(ActiveFloorFilter(policy),))
    jobs = [(0, first, 1, PlanarPlacement(_pose(x=.04, z=.02))),
            (1, second, 1, PlanarPlacement(_pose(x=.04, z=.02))),
            (2, empty, 1, PlanarPlacement(_pose(x=.04, z=.02))),
            (0, first, 1, PlanarPlacement(_pose(z=.08))),
            (1, second, 1, PlanarPlacement(_pose(x=-.06, z=.02))),
            (0, first, 1, PlanarPlacement(_pose(x=-.10, z=.02)))]
    calls = []
    context_details = policy.context_details

    def counted(state):
        calls.append(state)
        return context_details(state)

    monkeypatch.setattr(policy, 'context_details', counted)

    def check():
        expected = [generator._build_candidate_with_solver(solver, state, index, proposal)
                    for _, state, index, proposal in jobs]
        calls.clear()
        actual = generator._build_jobs(jobs)
        assert [id(state) for state in calls] == [id(first), id(second), id(empty)]
        for (candidate, reason), (ref, ref_reason) in zip(actual, expected):
            assert reason == ref_reason
            assert (candidate is None) == (ref is None)
            if candidate is not None:
                np.testing.assert_array_equal(candidate.action.pose, ref.action.pose)
                assert candidate.diagnostics == ref.diagnostics
        return actual

    before = check()
    assert before[0][1] is None
    assert before[1][1] == 'active_floor_disconnected_active_layer'
    assert before[3][1] == 'active_floor_above_active_layer'
    first.object_poses[10][0] = .12
    after = check()
    assert after[0][1] == 'active_floor_disconnected_active_layer'
    # Returning to an old pose still builds a fresh per-batch context.
    first.object_poses[10][0] = 0.
    check()
