import ast
from pathlib import Path

import numpy as np

from stacking_common.planning import SolverProfileFiles
from stacking_tabletop import TabletopApplication


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "packages" / "stacking-common" / "src" / "stacking_common"
TABLETOP = ROOT / "apps" / "tabletop"
FORBIDDEN_COMMON_IMPORTS = {
    "builtin_interfaces",
    "control_msgs",
    "geometry_msgs",
    "rclpy",
    "sensor_msgs",
    "stacking_tabletop",
    "stacking_excavator",
    "std_msgs",
    "tf2_ros",
    "trajectory_msgs",
}
FORBIDDEN_DOMAIN_IMPORTS = FORBIDDEN_COMMON_IMPORTS | {
    "diffsimpy",
    "gymnasium",
    "h5py",
    "open3d",
    "pyvista",
    "ray",
    "torch",
}
FORBIDDEN_ENVIRONMENT_IMPORTS = FORBIDDEN_DOMAIN_IMPORTS | {
    "matplotlib",
    "omegaconf",
    "scipy",
}
FORBIDDEN_SEARCH_IMPORTS = FORBIDDEN_ENVIRONMENT_IMPORTS | {
    "ray",
    "torch",
}
FORBIDDEN_VISUALIZATION_IMPORTS = FORBIDDEN_COMMON_IMPORTS | {
    "PyQt6",
    "PySide6",
    "open3d",
    "pyvista",
    "vtk",
}


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.partition(".")[0])
    return roots


def test_common_does_not_depend_on_platform_or_ros() -> None:
    violations = {
        str(path.relative_to(ROOT)): sorted(
            imported_roots(path) & FORBIDDEN_COMMON_IMPORTS
        )
        for path in COMMON.rglob("*.py")
        if imported_roots(path) & FORBIDDEN_COMMON_IMPORTS
    }
    assert violations == {}


def test_tabletop_device_logic_does_not_depend_on_ros_sdk_or_legacy() -> None:
    execution = TABLETOP / "src" / "stacking_tabletop" / "execution"
    forbidden = (FORBIDDEN_COMMON_IMPORTS - {"stacking_tabletop"}) | {
        "std_srvs", "dynamixel_sdk", "serial", "diffsimpy", "agent", "planning",
    }
    violations = {
        str(path.relative_to(ROOT)): sorted(imported_roots(path) & forbidden)
        for path in execution.rglob("*.py")
        if imported_roots(path) & forbidden
    }
    assert violations == {}


def test_tabletop_acquisition_keeps_ros_in_transport():
    acquisition = TABLETOP / "src" / "stacking_tabletop" / "acquisition"
    forbidden = (FORBIDDEN_COMMON_IMPORTS - {"stacking_tabletop"}) | {
        "orbbec_camera_msgs", "pyorbbecsdk", "cv_bridge", "sensor_msgs_py",
        "message_filters", "diffsimpy",
    }
    assert not {
        str(path.relative_to(ROOT)): sorted(imported_roots(path) & forbidden)
        for path in acquisition.glob("*.py") if imported_roots(path) & forbidden
    }


def test_domain_does_not_depend_on_runtime_frameworks() -> None:
    domain = COMMON / "domain"
    violations = {
        str(path.relative_to(ROOT)): sorted(
            imported_roots(path) & FORBIDDEN_DOMAIN_IMPORTS
        )
        for path in domain.rglob("*.py")
        if imported_roots(path) & FORBIDDEN_DOMAIN_IMPORTS
    }
    assert violations == {}


def test_environment_does_not_depend_on_runtime_frameworks() -> None:
    environment = COMMON / "environment"
    violations = {
        str(path.relative_to(ROOT)): sorted(
            imported_roots(path) & FORBIDDEN_ENVIRONMENT_IMPORTS
        )
        for path in environment.rglob("*.py")
        if imported_roots(path) & FORBIDDEN_ENVIRONMENT_IMPORTS
    }
    assert violations == {}


def test_search_does_not_depend_on_runtime_frameworks() -> None:
    search = COMMON / "search"
    violations = {
        str(path.relative_to(ROOT)): sorted(
            imported_roots(path) & FORBIDDEN_SEARCH_IMPORTS
        )
        for path in search.rglob("*.py")
        if imported_roots(path) & FORBIDDEN_SEARCH_IMPORTS
    }
    assert violations == {}


def test_common_visualization_contracts_do_not_depend_on_gui_backends() -> None:
    visualization = COMMON / "visualization"
    violations = {
        str(path.relative_to(ROOT)): sorted(
            imported_roots(path) & FORBIDDEN_VISUALIZATION_IMPORTS
        )
        for path in visualization.rglob("*.py")
        if imported_roots(path) & FORBIDDEN_VISUALIZATION_IMPORTS
    }
    assert violations == {}


def test_common_visualization_contracts_do_not_import_rendering_adapters() -> None:
    violations = []
    for path in (COMMON / "visualization").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            imports = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any(name == "rendering" or name.startswith(("stacking_common.rendering", "rendering."))
                   for name in imports):
                violations.append(str(path.relative_to(ROOT)))
    assert violations == []


def test_common_pointcloud_does_not_eagerly_import_open3d_or_ros() -> None:
    pointcloud = COMMON / "pointcloud"
    forbidden = FORBIDDEN_COMMON_IMPORTS | {"open3d"}
    violations = {
        str(path.relative_to(ROOT)): sorted(
            {
                alias.name.partition(".")[0]
                for node in ast.parse(path.read_text(), filename=str(path)).body
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            & forbidden
        )
        for path in pointcloud.rglob("*.py")
        if {
            alias.name.partition(".")[0]
            for node in ast.parse(path.read_text(), filename=str(path)).body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        & forbidden
    }
    assert violations == {}


def test_tabletop_perception_does_not_import_ros_messages() -> None:
    perception = TABLETOP / "src" / "stacking_tabletop" / "perception"
    forbidden = FORBIDDEN_COMMON_IMPORTS - {"stacking_tabletop"}
    violations = {
        str(path.relative_to(ROOT)): sorted(imported_roots(path) & forbidden)
        for path in perception.rglob("*.py")
        if imported_roots(path) & forbidden
    }
    assert violations == {}


def test_tabletop_search_does_not_import_legacy_agent_or_frameworks() -> None:
    search = TABLETOP / "src" / "stacking_tabletop" / "search"
    forbidden = FORBIDDEN_DOMAIN_IMPORTS | {
        "agent",
        "omegaconf",
        "torch",
    }
    violations = {
        str(path.relative_to(ROOT)): sorted(imported_roots(path) & forbidden)
        for path in search.rglob("*.py")
        if imported_roots(path) & forbidden
    }
    assert violations == {}


def test_operations_console_controller_is_headless() -> None:
    operations = TABLETOP / "src" / "stacking_tabletop" / "gui" / "operations"
    paths = (operations / "__init__.py", operations / "controller.py")
    forbidden = FORBIDDEN_COMMON_IMPORTS | {
        "PyQt6",
        "PySide6",
        "open3d",
        "pyvista",
        "pyvistaqt",
        "vtk",
    }
    violations = {
        str(path.relative_to(ROOT)): sorted(imported_roots(path) & forbidden)
        for path in paths
        if imported_roots(path) & forbidden
    }
    assert violations == {}


def test_tabletop_owns_one_profile_per_solver() -> None:
    profiles = SolverProfileFiles.from_directory(TABLETOP / "config" / "solvers")
    assert {path.stem for path in profiles.__dict__.values()} == {
        "simulation",
        "posegen",
        "grasp",
        "motion",
    }


def test_tabletop_application_is_composed_from_platform_profiles() -> None:
    application = TabletopApplication.from_repository(ROOT)
    assert application.solver_profiles.motion.is_file()
    assert application.environment_profile.observation.height_map_resolution == (
        128,
        128,
    )
    assert application.environment_profile.surface_sample_count == 128
    assert (
        application.action_generation_profile.generator.max_poses_per_object
        == 32
    )
    assert np.isclose(
        application.action_generation_profile.generator.initial_height_clearance,
        0.015,
    )
    assert application.action_generation_profile.planar.grid_shape == (32, 32)
    assert application.environment_profile.overlap_resolution == 0.003
    assert application.simulation_profile.placement.settle_steps == 400
    assert application.inventory_profile.object_ids == tuple(range(1, 21))
    assert application.perception_profile.field_registration.voxel_sizes[0] == 0.006
    assert application.inventory_profile.target.height == 0.15


def test_tabletop_catalog_owns_complete_asset_triples() -> None:
    application = TabletopApplication.from_repository(ROOT)
    directory = ROOT / application.inventory_profile.model_directory
    for object_id in application.inventory_profile.object_ids:
        assert (directory / f"model_{object_id}.obj").is_file()
        assert (directory / f"model_{object_id}.pcd").is_file()
        assert (directory / f"model_{object_id}_mesh.ply").is_file()


def test_platforms_do_not_import_each_other_or_legacy_runtime() -> None:
    for platform, other in (("tabletop", "stacking_excavator"),
                            ("excavator", "stacking_tabletop")):
        source = ROOT / "apps" / platform / "src"
        forbidden = {other, "diffsimpy", "agent"}
        assert not {
            str(path.relative_to(ROOT)): sorted(imported_roots(path) & forbidden)
            for path in source.rglob("*.py")
            if imported_roots(path) & forbidden
        }


def test_excavator_offline_app_has_no_device_dependencies() -> None:
    source = ROOT / "apps" / "excavator" / "src"
    forbidden = (FORBIDDEN_COMMON_IMPORTS - {"stacking_excavator"}) | {
        "ros2", "dynamixel_sdk", "serial", "torch", "ray",
    }
    assert not {
        str(path.relative_to(ROOT)): sorted(imported_roots(path) & forbidden)
        for path in source.rglob("*.py")
        if imported_roots(path) & forbidden
    }
