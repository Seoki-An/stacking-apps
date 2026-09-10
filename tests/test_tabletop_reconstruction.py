from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from stacking_common.pointcloud import PointCloudData, VoxelMergeStats
from stacking_common.reconstruction import (
    TriangleMeshData,
    estimate_mesh_inertial,
    is_watertight_mesh,
    load_dsf_components,
    repair_triangle_mesh,
    validate_triangle_mesh,
)
from stacking_tabletop.config import load_perception_profile
from stacking_tabletop.environment.assets import _parse_object_asset
from stacking_tabletop.perception import (
    DsfComponent,
    ReconstructionSession,
    ReconstructionSessionStore,
    ReconstructionViewRecord,
    ReconstructionOperatorOptions,
    TabletopDsfFitter,
    dsf_support_points,
    distribute_vertex_budget,
    split_points_by_seeds,
    TabletopDsfExporter,
    TabletopPoseGraphRefiner,
    estimate_hull_inertial,
)
from stacking_tabletop.perception.reconstruct_cli import (
    build_parser,
    default_session_directory,
    discover_capture_views,
    run_reconstruction,
)
from stacking_tabletop.perception.reconstruction import (
    _principal_axis_aligned_points,
)


ROOT = Path(__file__).resolve().parents[1]


def test_reconstruction_cli_uses_tabletop_data_session_layout() -> None:
    path = default_session_directory(ROOT, 7, "scan-a")

    assert path == (
        ROOT
        / "apps"
        / "tabletop"
        / "data"
        / "reconstruction"
        / "7"
        / "scan-a"
    )
    args = build_parser().parse_args([str(ROOT), "--object-id", "7"])
    assert args.session_directory is None

    with pytest.raises(ValueError, match="path-segment"):
        default_session_directory(ROOT, 7, "../outside")


def test_reconstruction_cli_discovers_capture_views_in_timestamp_order(
    tmp_path: Path,
) -> None:
    view_directory = tmp_path / "capture" / "manual_stone_views_tool0"
    view_directory.mkdir(parents=True)
    for name in ("100.pcd", "20.pcd", "3.pcd"):
        (view_directory / name).touch()

    views = discover_capture_views(tmp_path / "capture")

    assert tuple(path.name for path in views) == (
        "3.pcd",
        "20.pcd",
        "100.pcd",
    )
    args = build_parser().parse_args(
        [
            str(ROOT),
            "--object-id",
            "7",
            "--capture-directory",
            str(tmp_path / "capture"),
        ]
    )
    assert args.capture_directory == tmp_path / "capture"


def test_reconstruction_cli_rejects_conflicting_view_sources() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                str(ROOT),
                "--view",
                "view.pcd",
                "--capture-directory",
                "capture",
            ]
        )


def test_reconstruction_profile_is_explicit_and_tabletop_specific() -> None:
    profile = load_perception_profile(
        ROOT / "apps" / "tabletop" / "config" / "perception.yml"
    ).reconstruction

    assert profile.merge_voxel_size == 0.002
    assert profile.minimum_view_support == 2
    assert profile.pose_graph.registration.voxel_sizes == (0.006, 0.003, 0.002)
    assert profile.mesh.alpha == 0.005
    assert profile.mesh.method == "radial"
    assert profile.mesh.radial_subdivisions == 4
    assert profile.mesh.radial_neighbors == 120
    assert profile.mesh.bpa_radius_scales == (1.5, 2.5, 4.0)
    assert profile.mesh.repair_invalid_mesh is True
    assert profile.cleanup.manual_remove_distance == 0.005
    assert profile.cleanup.detached_voxel_size == 0.001
    assert profile.cleanup.fringe_margin == 0.003
    assert profile.convex_decomposition.enabled is True
    assert profile.convex_decomposition.maximum_components == 3
    assert profile.dsf_fitting.enabled is True
    assert profile.dsf_fitting.fit_source == "split_cloud"
    assert profile.dsf_fitting.total_vertices == 60
    assert profile.dsf.maximum_nodes == 128


def test_reconstruction_operator_options_validate_visualization_opacity() -> None:
    options = ReconstructionOperatorOptions(visualization_opacity=0.5)
    assert options.show_input is False
    assert options.remove_statistical_outliers is True
    assert options.remove_detached_clusters is True
    assert options.remove_radial_fringe is True
    assert options.keep_largest_cluster is True
    with pytest.raises(ValueError, match="opacity"):
        ReconstructionOperatorOptions(visualization_opacity=0.0)


def test_manual_removal_alignment_centers_and_orients_principal_axes() -> None:
    rng = np.random.default_rng(17)
    points = rng.normal(size=(500, 3)) * np.array([0.04, 0.02, 0.008])
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = points @ rotation.T + np.array([0.2, -0.1, 0.3])

    aligned = _principal_axis_aligned_points(transformed)
    covariance = np.cov(aligned, rowvar=False, bias=True)

    assert np.allclose(np.mean(aligned, axis=0), 0.0, atol=1e-12)
    assert np.all(np.diag(covariance)[:-1] >= np.diag(covariance)[1:])
    assert np.allclose(
        covariance - np.diag(np.diag(covariance)),
        0.0,
        atol=1e-12,
    )
    assert np.allclose(
        np.linalg.norm(aligned[1:] - aligned[0], axis=1),
        np.linalg.norm(transformed[1:] - transformed[0], axis=1),
    )


def test_dsf_vertex_budget_is_distributed_across_components() -> None:
    assert distribute_vertex_budget(14, 3) == (5, 5, 4)
    with pytest.raises(ValueError, match="too small"):
        distribute_vertex_budget(11, 3)


def test_manual_seed_partition_assigns_each_point_once() -> None:
    first = np.array(
        [[-1.0, value, 0.0] for value in (-0.03, -0.01, 0.01, 0.03)]
    )
    second = np.array(
        [[1.0, value, 0.0] for value in (-0.03, -0.01, 0.01, 0.03)]
    )
    points = np.vstack((first, second))

    partitions = split_points_by_seeds(
        points,
        np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        minimum_points=4,
    )

    assert np.allclose(partitions[0], first)
    assert np.allclose(partitions[1], second)


def test_dsf_fitter_returns_requested_vertex_count() -> None:
    pytest.importorskip("scipy")
    profile = load_perception_profile(
        ROOT / "apps/tabletop/config/perception.yml"
    ).reconstruction.dsf_fitting
    profile = replace(
        profile,
        total_vertices=8,
        direction_subdivisions=1,
        maximum_iterations=12,
        ransac=replace(profile.ransac, enabled=False),
    )
    rng = np.random.default_rng(9)
    points = rng.normal(size=(500, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points *= np.array([0.04, 0.03, 0.02])

    progress = []
    result = TabletopDsfFitter(profile).fit(
        (points,),
        progress=lambda completed, total, message: progress.append(
            (completed, total, message)
        ),
    )[0]

    assert result.vertices.shape == (8, 3)
    assert result.sharpness >= 2.0
    assert result.cost >= 0.0
    assert any(
        total == profile.maximum_iterations
        and message.endswith("optimizing smooth support")
        for _completed, total, message in progress
    )
    assert progress[-1] == (1, 1, "Fitted dsf component 1/1")


def test_hull_inertial_metadata_matches_unit_cube() -> None:
    pytest.importorskip("scipy")
    corners = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    )

    inertial = estimate_hull_inertial(corners)

    assert inertial.mass == pytest.approx(8.0)
    assert np.allclose(inertial.center_of_mass, np.zeros(3), atol=1e-12)
    assert np.allclose(inertial.inertia, np.eye(3) * (16.0 / 3.0))


def test_watertight_mesh_inertial_metadata_matches_translated_cube() -> None:
    corners = np.array(
        [
            [x, y, z]
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ]
    ) + np.array([3.0, -2.0, 1.0])
    faces = np.array(
        [
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
            [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
            [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ]
    )
    mesh = TriangleMeshData(corners, faces)

    inertial = estimate_mesh_inertial(mesh)

    assert is_watertight_mesh(mesh)
    assert inertial.mass == pytest.approx(8.0)
    assert np.allclose(inertial.center_of_mass, [3.0, -2.0, 1.0])
    assert np.allclose(inertial.inertia, np.eye(3) * (16.0 / 3.0))


def test_mesh_inertial_rejects_open_surface() -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    mesh = TriangleMeshData(
        vertices,
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2]]),
    )

    assert not is_watertight_mesh(mesh)
    with pytest.raises(ValueError, match="watertight"):
        estimate_mesh_inertial(mesh)


def test_mesh_validation_detects_intersecting_closed_components() -> None:
    vertices = np.array(
        [
            [0.942809, 0.0, -0.333333],
            [-0.471405, 0.816497, -0.333333],
            [-0.471405, -0.816497, -0.333333],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]])
    mesh = TriangleMeshData(
        np.vstack((vertices, vertices + np.array([0.15, 0.0, 0.0]))),
        np.vstack((faces, faces + 4)),
    )

    validation = validate_triangle_mesh(mesh)

    assert validation.watertight
    assert validation.self_intersecting
    assert not validation.valid
    with pytest.raises(ValueError, match="self-intersecting"):
        estimate_mesh_inertial(mesh)


def test_conservative_mesh_repair_removes_duplicate_faces() -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    invalid = TriangleMeshData(vertices, np.vstack((faces, faces[0])))

    repaired = repair_triangle_mesh(invalid)
    validation = validate_triangle_mesh(
        repaired,
        repair_attempted=True,
        repaired=True,
    )

    assert len(repaired.triangles) == 4
    assert validation.valid
    assert validation.repair_attempted
    assert validation.repaired


def test_dsf_export_preserves_legacy_mass_and_inertia_metadata(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    profile = load_perception_profile(
        ROOT / "apps" / "tabletop" / "config" / "perception.yml"
    ).reconstruction.dsf
    corners = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    )
    triangles = np.array(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5]]
    )
    component = DsfComponent(corners, triangles, profile.sharpness, profile.friction)
    inertial = estimate_hull_inertial(corners)
    path = TabletopDsfExporter(profile).write(
        tmp_path / "model_21.obj",
        (component,),
        inertial,
    )

    asset = _parse_object_asset(path)
    loaded_components = load_dsf_components(path)

    assert asset.mass == pytest.approx(8.0)
    assert np.allclose(asset.inertia, np.eye(3) * (16.0 / 3.0))
    assert len(asset.components) == 1
    assert len(loaded_components) == 1
    assert np.allclose(loaded_components[0].nodes, component.nodes)
    assert loaded_components[0].sharpness == component.sharpness
    assert loaded_components[0].friction == pytest.approx(component.friction)


def test_dsf_export_preserves_nodes_not_referenced_by_hull_faces(
    tmp_path: Path,
) -> None:
    pytest.importorskip("scipy")
    profile = load_perception_profile(
        ROOT / "apps/tabletop/config/perception.yml"
    ).reconstruction.dsf
    corners = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    )
    nodes = np.vstack((corners, np.zeros((1, 3))))
    exporter = TabletopDsfExporter(profile)
    component = exporter.component_from_vertices(nodes, sharpness=4)
    path = exporter.write(
        tmp_path / "model.obj",
        (component,),
        estimate_hull_inertial(corners),
    )

    loaded = _parse_object_asset(path).components[0].geometry

    assert len(component.nodes) == 9
    assert len(loaded.points) == 9
    assert np.any(np.all(np.isclose(loaded.points, 0.0), axis=1))


def test_dsf_surface_uses_sharpness_instead_of_vertex_hull() -> None:
    nodes = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    )
    direction = np.array([[1.0, 0.0, 0.0]])

    rounded = dsf_support_points(nodes, 2, direction)[0]
    sharp = dsf_support_points(nodes, 60, direction)[0]

    assert rounded[0] == pytest.approx(2.0)
    assert sharp[0] < 1.03
    assert rounded[0] > sharp[0]


def test_reconstruction_session_round_trip_keeps_inertial_state(tmp_path: Path) -> None:
    store = ReconstructionSessionStore()
    session = ReconstructionSession(
        object_id=21,
        views=(
            ReconstructionViewRecord("view_000", "view_000.pcd"),
            ReconstructionViewRecord("view_001", "view_001.pcd", accepted=False),
        ),
    ).with_outputs(
        VoxelMergeStats(1, 100, 80, 70, 10),
        {"dsf": "output/dsf.obj"},
        unit_density_mass=1.0,
        inertia=np.eye(3),
        center_of_mass=np.array([0.1, 0.2, 0.3]),
    )

    loaded = store.load(store.save(session, tmp_path / "session.json"))

    assert loaded.object_id == session.object_id
    assert loaded.status == "exported"
    assert loaded.views[1].accepted is False
    assert dict(loaded.outputs) == {"dsf": "output/dsf.obj"}
    assert np.allclose(loaded.inertia, np.eye(3))
    assert np.allclose(loaded.center_of_mass, [0.1, 0.2, 0.3])


def test_pose_graph_recovers_view_to_reference_transform() -> None:
    pytest.importorskip("open3d")
    profile = load_perception_profile(
        ROOT / "apps" / "tabletop" / "config" / "perception.yml"
    ).reconstruction.pose_graph
    rng = np.random.default_rng(31)
    reference = PointCloudData(
        rng.uniform([-0.03, -0.02, -0.01], [0.04, 0.025, 0.035], (1600, 3))
    )
    offset = np.eye(4)
    offset[:3, 3] = [0.005, -0.003, 0.002]

    result = TabletopPoseGraphRefiner(profile).refine(
        (reference, reference.transformed(offset))
    )

    assert result.optimized is True
    assert result.reason is None
    assert np.allclose(result.transforms[0], np.eye(4), atol=1e-8)
    assert np.allclose(result.transforms[1], np.linalg.inv(offset), atol=2e-4)


def test_headless_reconstruction_writes_reloadable_session_and_assets(
    tmp_path: Path,
) -> None:
    o3d = pytest.importorskip("open3d")
    pytest.importorskip("scipy")
    rng = np.random.default_rng(43)
    directions = rng.normal(size=(5000, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    points = directions * 0.03
    paths = []
    for index, translation in enumerate(([0.0, 0.0, 0.0], [0.003, -0.002, 0.001])):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points + translation)
        path = tmp_path / f"view_{index}.pcd"
        assert o3d.io.write_point_cloud(str(path), cloud)
        paths.append(path)

    session = run_reconstruction(
        ROOT,
        tmp_path / "reconstruction",
        object_id=21,
        view_paths=tuple(paths),
    )

    output = tmp_path / "reconstruction" / "output"
    assert session.status == "exported"
    assert session.pose_graph_optimized is True
    assert session.unit_density_mass > 0.0
    assert (output / "merged.pcd").is_file()
    assert (output / "mesh.ply").is_file()
    asset = _parse_object_asset(output / "dsf.obj")
    assert asset.mass == pytest.approx(session.unit_density_mass)
    assert np.allclose(asset.inertia, session.inertia)
