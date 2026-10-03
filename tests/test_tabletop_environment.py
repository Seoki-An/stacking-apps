from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from stacking_common.domain import (
    Action,
    ConvexGeometry,
    DirectionalTaper,
    ObjectInventory,
    StackObject,
    State,
    TargetGeometry,
)
from stacking_common.environment import PlacementResult
from stacking_tabletop import (
    TabletopApplication,
    TabletopHeightMapRenderer,
    TabletopObjectAssetLoader,
)
from stacking_tabletop.config import (
    load_environment_profile,
    load_inventory_profile,
    load_simulation_profile,
)
from stacking_tabletop.environment.heightmap_reference import (
    render_open3d_height_map,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BodyModel:
    mass: float = 1.0
    inertia: np.ndarray = None

    def __post_init__(self) -> None:
        if self.inertia is None:
            self.inertia = np.eye(3)


class Backend:
    def __init__(self) -> None:
        self.object_ids: list[int] = []

    def reset(self, object_ids) -> None:
        self.object_ids = list(object_ids)

    def place(self, state: State, action: Action) -> PlacementResult:
        del state
        return PlacementResult(object_poses={action.object_id: action.pose})

    def restore(self, state: State) -> None:
        del state

    def close(self) -> None:
        pass


def pose(z: float = 0.0) -> np.ndarray:
    return np.array([0.0, 0.0, z, 0.0, 0.0, 0.0, 1.0])


def cube(size: float) -> ConvexGeometry:
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


def inventory() -> ObjectInventory:
    target = TargetGeometry(
        width=0.08,
        length=0.17,
        height=0.15,
        origin=np.zeros(3),
        taper=DirectionalTaper(),
        support_overlap=0.0,
    )
    stack_object = StackObject(10, BodyModel(), [cube(0.04)])
    return ObjectInventory([stack_object], {10: pose(0.02)}, target)


def square_inventory() -> ObjectInventory:
    target = TargetGeometry(
        width=0.08,
        length=0.08,
        height=0.10,
        origin=np.zeros(3),
        taper=DirectionalTaper(),
        support_overlap=0.0,
    )
    stack_object = StackObject(10, BodyModel(), [cube(0.04)])
    return ObjectInventory([stack_object], {10: pose(0.02)}, target)


def test_tabletop_height_map_rasterizes_objects_and_target() -> None:
    object_inventory = inventory()
    renderer = TabletopHeightMapRenderer(
        object_inventory,
        resolution=(21, 21),
        margin=0.02,
    )
    state = State(
        object_ids=[10],
        placed_object_indices=[0],
        object_poses={10: pose(0.02)},
        trajectories=[],
        action_history=[],
    )

    scene = renderer.render(state)
    target = renderer.render_target()

    assert scene.shape == target.shape == (21, 21)
    assert np.isclose(scene[10, 10], 0.04)
    assert np.isclose(target[10, 10], 0.15)


def test_tabletop_height_map_matches_grid_golden() -> None:
    object_inventory = square_inventory()
    renderer = TabletopHeightMapRenderer(
        object_inventory,
        resolution=(9, 9),
        margin=0.0,
    )
    state = State(
        object_ids=[10],
        placed_object_indices=[0],
        object_poses={10: pose(0.02)},
        trajectories=[],
        action_history=[],
    )
    expected = np.zeros((9, 9), dtype=np.float32)
    expected[2:7, 2:7] = 0.04

    assert np.allclose(renderer.render(state), expected, atol=1e-7)
    assert np.allclose(renderer.render_target(), 0.10, atol=1e-7)


def test_tabletop_height_map_matches_open3d_reference() -> None:
    pytest.importorskip("open3d")
    object_inventory = square_inventory()
    renderer = TabletopHeightMapRenderer(
        object_inventory,
        resolution=(41, 41),
        margin=0.0,
    )
    state = State(
        object_ids=[10],
        placed_object_indices=[0],
        object_poses={10: pose(0.02)},
        trajectories=[],
        action_history=[],
    )
    points, faces = object_inventory[0].mesh_arrays()
    world_points = points + np.array([0.0, 0.0, 0.02])
    reference = render_open3d_height_map(
        [(world_points, faces)],
        renderer.resolution,
        renderer.xlim,
        renderer.ylim,
    )

    assert np.allclose(renderer.render(state), reference, atol=2e-6)


def test_tabletop_profile_loads_and_composes_environment() -> None:
    profile = load_environment_profile(
        ROOT / "apps" / "tabletop" / "config" / "environment.yml"
    )
    assert profile.reward.weights.object_containment == 2.0

    backend = Backend()
    application = TabletopApplication.from_repository(ROOT)
    environment = application.build_environment(inventory(), backend)

    reset = environment.reset()
    assert reset.observation.scene_height_map.shape == (128, 128)
    result = environment.step(Action(0, pose(0.02), pose(0.1)))
    assert result.done is True
    assert backend.object_ids == [10]


def test_simulation_profile_builds_native_config_and_backend() -> None:
    profile = load_simulation_profile(
        ROOT / "apps" / "tabletop" / "config" / "solvers" / "simulation.yml"
    )
    # Exercise explicit solver values independently of operator profile tuning.
    profile = replace(profile, solver_tol_abs=1e-5, solver_tol_rel=1e-6,
                      solver_beta_init=0.1, solver_beta_update_interval=50)
    native = profile.native_config()

    assert np.allclose(native.gravity, [0.0, 0.0, -9.81])
    assert native.damping == profile.damping
    assert native.contact.detection_margin == 0.01
    assert native.solver.tol_abs == 1e-5
    assert native.solver.tol_rel == 1e-6
    assert native.contact.error_reduction_ratio == profile.error_reduction_ratio
    assert native.solver.constraint_scale_reference == profile.solver_constraint_scale_reference
    assert native.solver.beta_init == 0.1
    assert native.solver.beta_update_interval == 50
    assert native.solver.max_iters == 3000
    assert native.solver.stagnation_window == 0
    assert profile.placement.ground_friction == 5.0
    settling = profile.settling_native_config()
    assert settling.damping == profile.damping
    assert settling.contact.model.name.lower() == "single_point"
    assert native.contact.model.name.lower() == "limit_surface_4d"
    assert settling.contact.error_reduction_ratio == profile.settling_physics.error_reduction_ratio
    assert settling.contact.detection_margin == profile.settling_physics.detection_margin
    assert settling.solver.constraint_scale_reference == profile.settling_physics.solver_constraint_scale_reference
    assert settling.solver.beta_init == profile.settling_physics.solver_beta_init
    assert settling.solver.beta_update_interval == profile.settling_physics.solver_beta_update_interval
    assert settling.solver.max_iters == 2000
    assert settling.solver.stagnation_window == 0
    assert settling.solver.tol_abs == native.solver.tol_abs == 1e-5
    assert settling.solver.tol_rel == native.solver.tol_rel == 1e-6
    assert profile.placement.short_gate is not None
    assert profile.placement.short_gate.max_time == 4.0
    assert profile.placement.structural.max_time == 15.0
    short = load_simulation_profile(
        ROOT / "apps" / "tabletop" / "config" / "solvers" / "simulation.yml",
        fidelity="short",
    )
    assert short.placement.short_gate is None
    assert short.placement.structural.max_time == 4.0
    short_native = short.native_config()
    assert short_native.damping == profile.damping
    assert short_native.solver.tol_abs == short.solver_tol_abs
    assert short_native.solver.tol_rel == short.solver_tol_rel
    assert short_native.solver.stagnation_window == 0
    assert short.settling_native_config().solver.stagnation_window == 0
    assert short.settling_native_config().solver.tol_abs == 1e-5
    assert short.settling_native_config().solver.tol_rel == 1e-6
    assert short.settling_native_config().contact.model == settling.contact.model
    application = TabletopApplication.from_repository(ROOT)
    backend = application.build_core_backend(inventory())
    assert backend.config.settle_steps == 400
    assert np.allclose(backend.simulation_config.gravity, [0.0, 0.0, -9.81])
    assert backend.simulation_config.solver.tol_abs == application.simulation_profile.solver_tol_abs
    assert backend.simulation_config.solver.tol_rel == application.simulation_profile.solver_tol_rel
    assert backend.settling_simulation_config.solver.tol_abs == 1e-5
    assert backend.settling_simulation_config.solver.tol_rel == 1e-6
    assert backend.simulator is not backend.settling_simulator


@pytest.mark.parametrize("field", ["solver_tol_abs", "solver_tol_rel"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_simulation_profile_rejects_invalid_tolerances(field, value) -> None:
    profile = load_simulation_profile(
        ROOT / "apps" / "tabletop" / "config" / "solvers" / "simulation.yml"
    )
    with pytest.raises(ValueError, match="solver tolerances"):
        replace(profile, **{field: value})


def test_simulation_profile_passes_custom_tolerances_to_native_config() -> None:
    profile = load_simulation_profile(
        ROOT / "apps" / "tabletop" / "config" / "solvers" / "simulation.yml"
    )
    custom = replace(profile, solver_tol_abs=2e-8, solver_tol_rel=3e-9)
    native = custom.native_config()
    assert native.solver.tol_abs == 2e-8
    assert native.solver.tol_rel == 3e-9
    assert profile.native_config().solver.tol_abs == profile.solver_tol_abs
    assert custom.settling_native_config().solver.tol_abs == 1e-5


def flat_simulation_values():
    """Reconstruct the pre-grouping schema for compatibility tests."""
    import yaml

    values = yaml.safe_load((ROOT / "apps/tabletop/config/solvers/simulation.yml").read_text())
    physics = values.pop("physics")
    values.update(values.pop("phases"))
    values.update(physics["structural"])
    values["settling"].update(physics["settling"])
    return values


@pytest.mark.parametrize("fidelity", ["short", "long"])
def test_grouped_simulation_matches_flat_phase_and_physics_settings(tmp_path, fidelity):
    import yaml

    source = ROOT / "apps/tabletop/config/solvers/simulation.yml"
    flat = tmp_path / "simulation.yml"
    values = flat_simulation_values()
    flat.write_text(yaml.safe_dump(values))
    grouped = load_simulation_profile(source, fidelity=fidelity)
    legacy = load_simulation_profile(flat, fidelity=fidelity)
    assert legacy.placement == grouped.placement
    for left, right in ((legacy, grouped), (legacy.settling_physics, grouped.settling_physics)):
        for name in left.__dataclass_fields__:
            if name not in {"placement", "settling_physics"}:
                np.testing.assert_equal(getattr(left, name), getattr(right, name))

    # Old profiles keep their explicit tolerances, not the new YAML defaults.
    values["settling"]["solver"]["tol_abs"] = 0.001
    values["settling"]["solver"]["tol_rel"] = 0.0001
    flat.write_text(yaml.safe_dump(values))
    old = load_simulation_profile(flat, fidelity=fidelity).settling_native_config()
    assert old.solver.tol_abs == 0.001
    assert old.solver.tol_rel == 0.0001


@pytest.mark.parametrize("key", ["solver", "contact", "settling", "structural", "short_structural"])
def test_simulation_profile_rejects_mixed_layouts(key):
    import yaml
    from stacking_common.environment import CoreSimulationProfile

    values = yaml.safe_load((ROOT / "apps/tabletop/config/solvers/simulation.yml").read_text())
    values[key] = {}
    with pytest.raises(ValueError, match="cannot mix grouped and flat"):
        CoreSimulationProfile.from_mapping(values)


@pytest.mark.parametrize("section", ["physics", "phases"])
def test_grouped_simulation_requires_both_sections(section):
    import yaml
    from stacking_common.environment import CoreSimulationProfile

    values = yaml.safe_load((ROOT / "apps/tabletop/config/solvers/simulation.yml").read_text())
    del values[section]
    with pytest.raises(ValueError, match=f"missing '{section}'"):
        CoreSimulationProfile.from_mapping(values)


def test_legacy_flat_simulation_profile_still_uses_one_physics_profile():
    from stacking_common.environment import CoreSimulationProfile

    values = flat_simulation_values()
    del values["settling"]["contact"]
    del values["settling"]["solver"]
    profile = CoreSimulationProfile.from_mapping(values)
    assert profile.settling_physics is None
    structural = profile.native_config()
    settling = profile.settling_native_config()
    assert settling.contact.model == structural.contact.model
    assert settling.solver.tol_abs == structural.solver.tol_abs
    solver = settling.solver
    solver.tol_abs = 0.001
    settling.solver = solver
    assert profile.native_config().solver.tol_abs == structural.solver.tol_abs


@pytest.mark.parametrize("field,value", [
    ("solver_constraint_scale_reference", -0.1),
    ("solver_constraint_scale_reference", 0.0),
    ("solver_constraint_scale_reference", float("nan")),
    ("solver_beta_init", 0.0),
    ("solver_max_iters", 0), ("solver_max_iters", 1.5),
    ("solver_beta_update_interval", -1),
    ("solver_stagnation_window", -1), ("solver_stagnation_window", 1.5),
    ("solver_stagnation_window", True),
    ("error_reduction_ratio", 1.1), ("error_reduction_ratio", float("nan")),
])
def test_settling_physics_rejects_invalid_native_settings(field, value):
    profile = load_simulation_profile(ROOT / "apps/tabletop/config/solvers/simulation.yml")
    with pytest.raises(ValueError, match=field):
        replace(profile.settling_physics, **{field: value})


def test_stagnation_setting_can_be_enabled_or_left_to_native_default():
    import stacking_core

    profile = load_simulation_profile(ROOT / "apps/tabletop/config/solvers/simulation.yml")
    assert replace(profile, solver_stagnation_window=25).native_config().solver.stagnation_window == 25
    assert replace(profile, solver_stagnation_window=None).native_config().solver.stagnation_window == stacking_core.SolverConfig().stagnation_window


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_simulation_rejects_invalid_damping(value):
    profile = load_simulation_profile(ROOT / "apps/tabletop/config/solvers/simulation.yml")
    with pytest.raises(ValueError, match="damping"):
        replace(profile, damping=value)


def test_simulation_damping_defaults_to_disabled():
    from stacking_common.environment import CoreSimulationProfile

    values = flat_simulation_values()
    del values["integration"]["damping"]
    profile = CoreSimulationProfile.from_mapping(values)
    assert profile.native_config().damping == 0.0
    assert profile.settling_native_config().damping == 0.0


@pytest.mark.parametrize("phase", ["structural", "settling"])
def test_old_solver_damping_key_is_not_silently_ignored(phase):
    import yaml
    from stacking_common.environment import CoreSimulationProfile

    values = yaml.safe_load((ROOT / "apps/tabletop/config/solvers/simulation.yml").read_text())
    for settings in values["physics"].values():
        settings["solver"]["constraint_scale_reference"] = 0.01
    solver = values["physics"][phase]["solver"]
    solver["damping"] = solver.pop("constraint_scale_reference")
    with pytest.raises(ValueError, match="renamed to solver.constraint_scale_reference"):
        CoreSimulationProfile.from_mapping(values)


@pytest.mark.parametrize("fidelity", ["short", "long"])
def test_constraint_scaling_round_trip_is_independent_of_physical_damping(tmp_path, fidelity):
    import yaml

    values = yaml.safe_load((ROOT / "apps/tabletop/config/solvers/simulation.yml").read_text())
    values["integration"]["damping"] = 0.03
    values["physics"]["settling"]["solver"]["constraint_scale_reference"] = 0.04
    values["physics"]["structural"]["solver"]["constraint_scale_reference"] = 0.02
    path = tmp_path / "simulation.yml"
    path.write_text(yaml.safe_dump(values))
    profile = load_simulation_profile(path, fidelity=fidelity)
    native = profile.native_config()
    settling = profile.settling_native_config()
    assert native.damping == settling.damping == 0.03
    assert native.solver.constraint_scale_reference == 0.02
    assert settling.solver.constraint_scale_reference == 0.04
    assert not hasattr(native.solver, "damping")
    for value in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="solver_constraint_scale_reference"):
            replace(profile, solver_constraint_scale_reference=value)


def test_simulation_rejects_old_backend_location(tmp_path):
    path = tmp_path / "simulation.yml"
    source = ROOT / "apps/tabletop/config/solvers/simulation.yml"
    path.write_text(source.read_text() + "\nbackend: cuda\n")
    with pytest.raises(ValueError, match="parallel.simulation_backend"):
        load_simulation_profile(path)


def test_inventory_profile_and_component_asset_loading(tmp_path: Path) -> None:
    profile = load_inventory_profile(
        ROOT / "apps" / "tabletop" / "config" / "inventory.yml"
    )
    assert profile.density_overrides == ()
    assert profile.density_for(7) == profile.density
    model = tmp_path / "model_7.obj"
    model.write_text(
        """# mass: 0.001
# inertia: [[0.0001, 0.0, 0.0],
#   [0.0, 0.0001, 0.0],
#   [0.0, 0.0, 0.0001]]
# dsf_count: 2

o convex.0
# sharpness: 8
# mu: 0.5
v 0.0 0.0 0.0
v 0.02 0.0 0.0
v 0.0 0.02 0.0
v 0.0 0.0 0.02
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4

o convex.1
# sharpness: 12
# mu: 0.7
v 0.03 0.0 0.0
v 0.05 0.0 0.0
v 0.03 0.02 0.0
v 0.03 0.0 0.02
f 5 7 6
f 5 6 8
f 5 8 7
f 6 7 8
"""
    )
    loader = TabletopObjectAssetLoader(
        tmp_path,
        density=profile.density,
        density_overrides={7: 2000.0},
        target=profile.target.geometry(),
    )

    loaded = loader.load([7], {7: pose(0.1)})

    assert loaded.object_ids == (7,)
    assert len(loaded[0].geometries) == 2
    assert loaded[0].body_model.geometry_count == 2
    assert loaded[0].mass == 2.0
    assert loaded[0].geometries[0].sharpness == 8.0


def test_inventory_profile_parses_object_density_override(tmp_path: Path) -> None:
    source = ROOT / "apps" / "tabletop" / "config" / "inventory.yml"
    configured = source.read_text().replace(
        "density_overrides: {}",
        "density_overrides:\n    7: 1250.0",
    )
    path = tmp_path / "inventory.yml"
    path.write_text(configured)

    profile = load_inventory_profile(path)

    assert profile.density_for(1) == 1000.0
    assert profile.density_for(7) == 1250.0


def test_application_loads_complete_tabletop_object_catalog() -> None:
    application = TabletopApplication.from_repository(ROOT)
    object_ids = application.inventory_profile.object_ids
    initial_poses = {object_id: pose(0.1) for object_id in object_ids}

    loaded = application.load_inventory(object_ids, initial_poses)

    assert loaded.object_ids == object_ids
    assert all(stack_object.body_model.geometry_count > 0 for stack_object in loaded)


def test_application_rejects_object_outside_tabletop_catalog() -> None:
    application = TabletopApplication.from_repository(ROOT)

    with pytest.raises(ValueError, match="not in the tabletop catalog"):
        application.load_inventory([999], {999: pose(0.1)})
