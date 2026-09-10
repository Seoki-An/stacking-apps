import numpy as np
import pytest
from types import MethodType, SimpleNamespace
from dataclasses import replace
from pathlib import Path

try:
    import stacking_core
except ImportError:
    stacking_core = None

from stacking_common.domain import (
    Action,
    ContactPoint,
    StackObjectTrajectory,
    State,
)
from stacking_common.environment import (
    CorePlacementBackend,
    CorePlacementConfig,
    SimulationPhaseConfig,
    SimulationTransition,
)
from stacking_common.environment.core_backend import (
    _StructuralSimulation,
    _contact_validation,
)


def pose(z: float) -> np.ndarray:
    return np.array([0.0, 0.0, z, 0.0, 0.0, 0.0, 1.0])


def cube_nodes() -> np.ndarray:
    return np.array(
        [
            [-0.05, -0.05, -0.05],
            [-0.05, -0.05, 0.05],
            [-0.05, 0.05, -0.05],
            [-0.05, 0.05, 0.05],
            [0.05, -0.05, -0.05],
            [0.05, -0.05, 0.05],
            [0.05, 0.05, -0.05],
            [0.05, 0.05, 0.05],
        ]
    )


@pytest.mark.skipif(
    stacking_core is None,
    reason="stacking-core Python bindings are not installed",
)
def test_core_backend_drives_simulation_transition() -> None:
    model = stacking_core.BodyModel.dsf(
        10,
        cube_nodes(),
        mass=1.0,
        inertia=np.eye(3),
        friction=0.6,
        sharpness=100,
    )
    backend = CorePlacementBackend(
        {10: model},
        CorePlacementConfig(
            structural=SimulationPhaseConfig(
                dt=0.01,
                energy_threshold=1.0,
                velocity_integral_threshold=1.0,
                min_time=0.0,
                max_time=0.1,
                extra_steps=1,
            ),
            settle_dt=0.01,
            settle_steps=2,
            contact_gap_tolerance=0.01,
            minimum_support_count=1,
        ),
    )
    transition = SimulationTransition([10], backend)

    transition.reset()
    result = transition.step(Action(0, pose(0.2), pose(0.3)))

    assert result.terminated is True
    assert result.failed is False
    assert result.simulation_settled is True
    assert result.object_poses[10][2] < 0.2
    assert len(result.trajectories[-1][10]) >= 2
    assert result.action_history[-1].object_id == 10

    fixed = backend.settle_fixed(
        State([10], [], {}, [], []),
        Action(0, pose(0.2), pose(0.3), object_id=10),
    )

    assert not fixed.failed
    assert fixed.object_poses[10][2] < 0.2
    assert fixed.diagnostics["fixed_scene_resettle_steps"] == 2


@pytest.mark.skipif(stacking_core is None, reason="native bindings unavailable")
def test_distinct_physics_preserves_settled_pose_but_resets_motion():
    from stacking_tabletop.config.simulation import load_simulation_profile

    profile = load_simulation_profile(
        Path(__file__).resolve().parents[1] / "apps/tabletop/config/solvers/simulation.yml"
    )
    profile = replace(profile, placement=replace(
        profile.placement, structural=_phase(0.1, 1), short_gate=None,
        settle_steps=2,
    ))
    model = stacking_core.BodyModel.dsf(
        10, cube_nodes(), mass=1.0, inertia=np.eye(3), friction=0.6, sharpness=100,
    )
    backend = CorePlacementBackend.from_profile({10: model}, profile)
    backend.reset([10])

    class RecordingSimulator:
        def __init__(self, native):
            self.native = native
            self.initial_scenes = []
            self.final_scene = None

        def clear_warm_start(self):
            self.native.clear_warm_start()

        def step_n(self, scene, dt, steps):
            self.initial_scenes.append(scene)
            result = self.native.step_n(scene, dt, steps)
            self.final_scene = result.snapshot
            return result

        def step(self, scene, dt):
            return self.step_n(scene, dt, 1)

    settling = RecordingSimulator(backend.settling_simulator)
    structural = RecordingSimulator(backend.simulator)
    backend.settling_simulator = settling
    backend.simulator = structural
    state = State([10], [], {}, [], [])
    action = Action(0, pose(0.2), pose(0.3), object_id=10)
    backend.place(state, action)
    assert settling.initial_scenes and structural.initial_scenes
    settled = settling.final_scene
    start = structural.initial_scenes[0]
    np.testing.assert_array_equal(start.body_pose(1), settled.body_pose(1))
    assert np.linalg.norm(settled.body_linear_velocity(1)) > 0
    np.testing.assert_array_equal(start.body_linear_velocity(1), np.zeros(3))
    np.testing.assert_array_equal(start.body_angular_velocity(1), np.zeros(3))

    structural_calls = len(structural.initial_scenes)
    settling_calls = len(settling.initial_scenes)
    backend.settle_fixed(state, action)
    assert len(structural.initial_scenes) == structural_calls
    assert len(settling.initial_scenes) > settling_calls
    backend.restore(state)
    backend.reset([10])
    assert backend.simulator is not structural
    assert backend.settling_simulator is not settling
    backend.close()
    assert backend.simulator is None and backend.settling_simulator is None


@pytest.mark.parametrize("final", [False, True])
def test_final_inventory_placement_uses_profile_quiet_step_requirement(final):
    backend = object.__new__(CorePlacementBackend)
    backend.object_ids = [10] if final else [10, 11]
    backend.config = CorePlacementConfig(
        structural=replace(_phase(4.0, 3), final_extra_steps=6),
        short_gate=replace(_phase(2.0, 1), final_extra_steps=2),
        settle_dt=0.01, settle_steps=0,
        contact_gap_tolerance=0.01, minimum_support_count=1,
    )
    required = []

    def simulate(self, scene, indices, phase, trajectories, integrals, elapsed, quiet):
        required.append(phase.extra_steps)
        return scene, elapsed + 0.1, phase.extra_steps, (), True, False

    backend._simulate_phase = MethodType(simulate, backend)
    backend._simulate_structural(
        "scene", [0], Action(0, pose(0.1), pose(0.2), object_id=10),
        StackObjectTrajectory(10),
    )
    assert required == ([2, 6] if final else [1, 3])


def _phase(max_time: float, extra_steps: int) -> SimulationPhaseConfig:
    return SimulationPhaseConfig(
        dt=0.01,
        energy_threshold=1.0,
        velocity_integral_threshold=1.0,
        min_time=0.0,
        max_time=max_time,
        extra_steps=extra_steps,
    )


def test_long_structural_phase_continues_short_snapshot_and_statistics() -> None:
    backend = object.__new__(CorePlacementBackend)
    backend.object_ids = [10]
    backend.config = CorePlacementConfig(
        structural=_phase(4.0, 3),
        short_gate=_phase(2.0, 1),
        settle_dt=0.01,
        settle_steps=0,
        contact_gap_tolerance=0.01,
        minimum_support_count=1,
    )
    calls = []

    def simulate_phase(
        self,
        scene,
        indices,
        phase,
        trajectories,
        velocity_integrals,
        elapsed,
        quiet_steps,
    ):
        del self, indices, trajectories
        calls.append((scene, phase, elapsed, quiet_steps))
        velocity_integrals[10] += 0.1
        if len(calls) == 1:
            return "short_snapshot", 0.5, 1, (), True, False
        assert scene == "short_snapshot"
        assert elapsed == 0.5
        assert quiet_steps == 1
        return "long_snapshot", 2.0, 3, (), True, False

    backend._simulate_phase = MethodType(simulate_phase, backend)
    action = Action(0, pose(0.1), pose(0.2), object_id=10)
    result = backend._simulate_structural(
        "initial_snapshot",
        [0],
        action,
        StackObjectTrajectory(10),
    )

    assert len(calls) == 2
    assert result.snapshot == "long_snapshot"
    assert result.velocity_integrals == {10: 0.2}
    assert result.stage_times == {
        "short": 0.5,
        "long_extension": 1.5,
        "total": 2.0,
    }


def test_simulation_records_elapsed_time_across_different_phase_dt():
    backend = object.__new__(CorePlacementBackend)
    backend.object_ids = [10]
    backend.scene_builder = SimpleNamespace(entity_id=lambda index: index)
    scene = SimpleNamespace(body_pose=lambda index: pose(0.1),
                            body_linear_velocity=lambda index: np.zeros(3),
                            body_angular_velocity=lambda index: np.zeros(3))
    backend.simulator = SimpleNamespace(step=lambda scene, dt: SimpleNamespace(snapshot=scene, contacts=()))
    backend._normalized_energy = lambda *args: 0.0
    trajectory = StackObjectTrajectory(10)
    trajectory.add_pose(pose(0.1), time=0)
    phase = replace(_phase(1, 1), min_time=0, dt=0.01)
    result = backend._simulate_phase(scene, [0], phase, {10: trajectory}, {10: 0}, 0, 0)
    backend._simulate_phase(scene, [0], replace(phase, dt=0.02, extra_steps=2),
                            {10: trajectory}, {10: 0}, result[1], result[2])
    np.testing.assert_allclose(trajectory.times, [0, 0.01, 0.03])


def test_failed_short_gate_skips_long_extension() -> None:
    backend = object.__new__(CorePlacementBackend)
    backend.object_ids = [10]
    backend.config = CorePlacementConfig(
        structural=_phase(4.0, 3),
        short_gate=_phase(2.0, 1),
        settle_dt=0.01,
        settle_steps=0,
        contact_gap_tolerance=0.01,
        minimum_support_count=1,
    )
    calls = []

    def simulate_phase(
        self,
        scene,
        indices,
        phase,
        trajectories,
        velocity_integrals,
        elapsed,
        quiet_steps,
    ):
        del self, indices, phase, trajectories, elapsed, quiet_steps
        calls.append(scene)
        velocity_integrals[10] = 2.0
        return "failed_short", 0.2, 0, (), False, True

    backend._simulate_phase = MethodType(simulate_phase, backend)
    result = backend._simulate_structural(
        "initial_snapshot",
        [0],
        Action(0, pose(0.1), pose(0.2), object_id=10),
        StackObjectTrajectory(10),
    )

    assert calls == ["initial_snapshot"]
    assert result.unstable
    assert result.stage_times["long_extension"] == 0.0


def test_short_freezing_preserves_the_descending_support_chain() -> None:
    backend = object.__new__(CorePlacementBackend)
    backend.object_ids = [10, 20, 30]
    backend.config = CorePlacementConfig(
        structural=SimulationPhaseConfig(
            dt=0.01,
            energy_threshold=1.0,
            velocity_integral_threshold=1.0,
            min_time=0.0,
            max_time=1.0,
            extra_steps=1,
            freeze_radius=0.1,
        ),
        settle_dt=0.01,
        settle_steps=0,
        contact_gap_tolerance=0.01,
        minimum_support_count=1,
        support_chain_z_tolerance=0.005,
    )
    backend.scene_builder = SimpleNamespace(entity_id=lambda index: index + 1)
    poses = {
        1: np.array([0.30, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0]),
        2: np.array([0.0, 0.0, 0.10, 0.0, 0.0, 0.0, 1.0]),
        3: np.array([0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 1.0]),
    }
    snapshot = SimpleNamespace(body_pose=lambda entity: poses[entity])

    frozen = backend._short_frozen_indices(
        snapshot,
        [0, 1, 2],
        2,
        ({"entity_first": 3, "entity_second": 2},),
    )

    assert frozen == {0}


def _contact(
    *,
    first: int | None,
    second: int | None,
    ground_first: bool = False,
    ground_second: bool = False,
    gap: float = 0.0,
) -> ContactPoint:
    return ContactPoint(
        entity_id_first=0 if ground_first else (first or 0) + 1,
        entity_id_second=0 if ground_second else (second or 0) + 1,
        object_index_first=first,
        object_index_second=second,
        ground_first=ground_first,
        ground_second=ground_second,
        point_first=np.zeros(3),
        point_second=np.zeros(3),
        normal=np.array([0.0, 0.0, 1.0]),
        gap=gap,
    )


def test_contact_validation_rejects_penetration_and_missing_support() -> None:
    ground_penetration = _contact(
        first=1,
        second=None,
        ground_second=True,
        gap=-0.02,
    )
    scene_penetration = _contact(first=1, second=0, gap=-0.02)

    assert _contact_validation(
        1,
        [ground_penetration],
        placed_count=1,
        gap_tolerance=0.01,
        minimum_support_count=1,
    )[0] == "place_plane_gap"
    assert _contact_validation(
        1,
        [scene_penetration],
        placed_count=1,
        gap_tolerance=0.01,
        minimum_support_count=1,
    )[0] == "place_scene_gap"
    reason, diagnostics = _contact_validation(
        1,
        [],
        placed_count=1,
        gap_tolerance=0.01,
        minimum_support_count=1,
    )
    assert reason == "final_support"
    assert diagnostics["support_count"] == 0


def test_contact_validation_accepts_deduplicated_support_sources() -> None:
    contacts = [
        _contact(first=1, second=0, gap=0.001),
        _contact(first=1, second=0, gap=0.002),
    ]

    reason, diagnostics = _contact_validation(
        1,
        contacts,
        placed_count=1,
        gap_tolerance=0.01,
        minimum_support_count=1,
    )

    assert reason is None
    assert diagnostics["support_count"] == 1
    assert diagnostics["support_sources"] == (0,)


@pytest.mark.parametrize(
    ("integrals", "expected"),
    [
        ({10: 0.0, 20: 0.2}, "target_motion_limit"),
        ({10: 0.2, 20: 0.0}, "scene_motion_limit"),
    ],
)
def test_physical_validation_attributes_target_and_scene_motion(
    integrals,
    expected,
) -> None:
    backend = object.__new__(CorePlacementBackend)
    backend.object_ids = [10, 20]
    backend.config = CorePlacementConfig(
        structural=SimulationPhaseConfig(
            dt=0.01,
            energy_threshold=1.0,
            velocity_integral_threshold=0.05,
            min_time=0.0,
            max_time=1.0,
            extra_steps=1,
        ),
        settle_dt=0.01,
        settle_steps=0,
        contact_gap_tolerance=0.01,
        minimum_support_count=1,
    )
    state = State(
        object_ids=[10, 20],
        placed_object_indices=[0],
        object_poses={10: pose(0.05)},
        trajectories=[],
        action_history=[],
    )
    simulation = _StructuralSimulation(
        snapshot=None,
        trajectories={},
        settled=False,
        contacts=(),
        unstable=True,
        velocity_integrals=integrals,
        elapsed=0.1,
        quiet_steps=0,
        stage_times={"structural": 0.1, "total": 0.1},
        velocity_threshold=0.05,
    )

    reason = backend._physical_rejection_reason(
        state,
        Action(1, pose(0.1), pose(0.2), object_id=20),
        simulation,
        (),
        True,
        {},
    )

    assert reason == expected
