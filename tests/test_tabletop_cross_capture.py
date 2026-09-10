import json
from pathlib import Path

import numpy as np
import pytest

from stacking_common.pointcloud import (
    CrossCaptureMerger,
    PointCloudData,
    RegistrationResult,
)
from stacking_tabletop.config import load_perception_profile
from stacking_tabletop.perception import (
    TabletopCrossCaptureMerger,
    resolve_capture_cloud,
    write_point_cloud,
)
from stacking_tabletop.perception.merge_captures_cli import (
    build_parser,
    default_merge_directory,
    discover_merge_captures,
    run_cross_capture_merge,
)
from stacking_tabletop.perception.object_reconstruct_cli import (
    build_parser as build_reconstruct_parser,
    run_object_reconstruction,
)
from stacking_tabletop.perception.prepare_capture_cli import (
    build_parser as build_prepare_parser,
)


ROOT = Path(__file__).resolve().parents[1]


def test_cross_capture_cli_accepts_two_prepared_captures() -> None:
    args = build_parser().parse_args(
        [".", "stone1_1", "stone1_2", "--object-id", "1"]
    )

    assert args.object_id == 1
    assert args.capture_directory == [Path("stone1_1"), Path("stone1_2")]
    prepare = build_prepare_parser().parse_args([".", "stone1_1"])
    reconstruct = build_reconstruct_parser().parse_args(
        [".", "merged.pcd", "--object-id", "1"]
    )

    assert prepare.capture_directory == Path("stone1_1")
    assert reconstruct.merged_cloud == Path("merged.pcd")
    assert default_merge_directory(ROOT, 1, "scan") == (
        ROOT / "apps/tabletop/data/merged/1/scan"
    )


def test_tabletop_merger_is_the_common_algorithm() -> None:
    assert TabletopCrossCaptureMerger is CrossCaptureMerger


def test_object_reconstruction_cli_exposes_operator_controls() -> None:
    args = build_reconstruct_parser().parse_args(
        [
            ".",
            "merged.pcd",
            "--object-id",
            "1",
            "--visualize",
            "--manual-remove",
            "--manual-remove-distance",
            "0.004",
            "--no-statistical-outlier-removal",
            "--no-detached-cluster-removal",
            "--no-radial-fringe-removal",
            "--keep-all-clusters",
            "--mesh-method",
            "poisson",
            "--mesh-poisson-depth",
            "10",
            "--dsf-maximum-nodes",
            "64",
            "--convex-maximum-components",
            "4",
            "--dsf-fit-vertices",
            "48",
            "--prompt-convex-maximum-components",
            "--review-result",
        ]
    )

    assert args.visualize is True
    assert args.manual_remove is True
    assert args.manual_remove_distance == 0.004
    assert args.remove_statistical_outliers is False
    assert args.remove_detached_clusters is False
    assert args.remove_radial_fringe is False
    assert args.keep_largest_cluster is False
    assert args.mesh_method == "poisson"
    assert args.mesh_poisson_depth == 10
    assert args.dsf_maximum_nodes == 64
    assert args.convex_maximum_components == 4
    assert args.dsf_fit_vertices == 48
    assert args.prompt_convex_maximum_components is True
    assert args.review_result is True

    defaults = build_reconstruct_parser().parse_args(
        [".", "merged.pcd", "--object-id", "1"]
    )
    assert defaults.remove_statistical_outliers is True
    assert defaults.remove_detached_clusters is True
    assert defaults.remove_radial_fringe is True
    assert defaults.keep_largest_cluster is True


def test_capture_cloud_resolution_prefers_manual_object_merge(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "stone1_1"
    capture.mkdir()
    expected = capture / "stone_manual_tool0.pcd"
    expected.touch()

    assert resolve_capture_cloud(capture) == expected
    assert resolve_capture_cloud(expected) == expected


def test_merge_capture_discovery_expands_parent_in_name_order(
    tmp_path: Path,
) -> None:
    second = _prepared_capture(tmp_path / "stone1_2")
    first = _prepared_capture(tmp_path / "stone1_1")
    (tmp_path / "unprepared").mkdir()

    assert discover_merge_captures([tmp_path]) == (
        first.resolve(),
        second.resolve(),
    )


def test_merge_capture_discovery_deduplicates_explicit_and_parent_inputs(
    tmp_path: Path,
) -> None:
    first = _prepared_capture(tmp_path / "stone1_1")
    second = _prepared_capture(tmp_path / "stone1_2")

    assert discover_merge_captures([first, tmp_path]) == (
        first.resolve(),
        second.resolve(),
    )


def test_cross_capture_gicp_recovers_regrasp_transform() -> None:
    pytest.importorskip("open3d")
    profile = load_perception_profile(
        ROOT / "apps/tabletop/config/perception.yml"
    ).cross_capture_merge
    target, source, expected = _cloud_pair()
    initial = expected.copy()
    initial[:3, 3] += [0.001, -0.001, 0.0005]
    progress = []

    result = TabletopCrossCaptureMerger(profile).merge(
        (target, source),
        (initial,),
        progress=lambda completed, total, message: progress.append(
            (completed, total, message)
        ),
    )

    assert result.alignments[0].decision.accepted
    assert np.allclose(result.alignments[0].transform, expected, atol=5e-4)
    assert len(result.cloud) > 1500
    assert progress == [
        (0, 1, "Running ICP for capture 2"),
        (1, 1, "ICP refined capture 2"),
    ]


def test_initializer_and_icp_alternate_against_accumulated_target() -> None:
    profile = load_perception_profile(
        ROOT / "apps/tabletop/config/perception.yml"
    ).cross_capture_merge
    merger = CrossCaptureMerger(profile)

    class IdentityBackend:
        def register(self, source, target, initial):
            return RegistrationResult(initial, 1.0, 1.0, 0.0, len(source), ())

    class RecordingInitializer:
        def __init__(self):
            self.target_sizes = []

        def initialize(self, source, target, source_index):
            self.target_sizes.append(len(target))
            return np.eye(4)

    merger.backend = IdentityBackend()
    initializer = RecordingInitializer()
    base = np.array(
        [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.001, 0.0]]
    )
    clouds = tuple(PointCloudData(base + 0.02 * index) for index in range(3))

    merger.merge(clouds, initializer=initializer)

    assert initializer.target_sizes[1] > initializer.target_sizes[0]


def test_merge_and_reconstruction_commands_have_separate_artifacts(
    tmp_path: Path,
) -> None:
    pytest.importorskip("open3d")
    pytest.importorskip("scipy")
    target, source, expected = _cloud_pair()
    captures = []
    for index, cloud in enumerate((target, source), start=1):
        capture = tmp_path / f"stone1_{index}"
        capture.mkdir()
        write_point_cloud(capture / "stone_manual_tool0.pcd", cloud)
        captures.append(capture)

    class FixedInitializer:
        def initialize(self, source, target, source_index):
            return expected

    report_path = run_cross_capture_merge(
        ROOT,
        tmp_path / "result",
        object_id=1,
        capture_directories=tuple(captures),
        initializer=FixedInitializer(),
        show_result=False,
    )
    report = json.loads(report_path.read_text())

    assert report["status"] == "merged"
    assert report["alignments"][0]["decision"]["accepted"] is True
    assert (tmp_path / "result/merged.pcd").is_file()
    assert not (tmp_path / "result/output/mesh.ply").exists()

    reconstruction_progress = []
    reconstruction_report = run_object_reconstruction(
        ROOT,
        tmp_path / "result",
        tmp_path / "reconstruction",
        object_id=1,
        density_kg_m3=1000.0,
        progress=lambda completed, total, message: reconstruction_progress.append(
            (completed, total, message)
        ),
    )
    reconstruction = json.loads(reconstruction_report.read_text())

    assert reconstruction["status"] == "exported"
    assert reconstruction["schema_version"] == 6
    assert reconstruction["cleaned_point_count"] > 0
    assert reconstruction["mesh_profile"]["method"] == "radial"
    assert reconstruction["mesh_validation"]["valid"] is True
    assert reconstruction["mesh_validation"]["self_intersecting"] is False
    assert reconstruction["convex_decomposition"]["component_count"] >= 1
    assert reconstruction["inertial"]["mass_source"] == "configured_density"
    assert reconstruction["inertial"]["volume_method"] == "watertight_mesh"
    assert reconstruction["inertial"]["mass_kg"] == pytest.approx(
        1000.0 * reconstruction["inertial"]["unit_density_mass"]
    )
    assert reconstruction["inertial"]["mass_kg"] > 0.0
    assert (tmp_path / "reconstruction/output/merged.pcd").is_file()
    assert (tmp_path / "reconstruction/output/mesh.ply").is_file()
    assert (tmp_path / "reconstruction/output/dsf.obj").is_file()
    assert (tmp_path / "reconstruction/output/object_metadata.json").is_file()
    assert any(
        total == 0 and message.startswith("Fitting ")
        for _completed, total, message in reconstruction_progress
    )
    assert reconstruction_progress[-1] == (
        1,
        1,
        "Reconstruction artifacts written",
    )


def _cloud_pair() -> tuple[PointCloudData, PointCloudData, np.ndarray]:
    rng = np.random.default_rng(82)
    directions = rng.normal(size=(4000, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = np.column_stack(
        (
            np.full(len(directions), 0.035),
            np.full(len(directions), 0.023),
            np.full(len(directions), 0.029),
        )
    )
    target = PointCloudData(directions * radii)
    regrasp = np.eye(4)
    angle = np.deg2rad(14.0)
    regrasp[:3, :3] = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    regrasp[:3, 3] = [0.018, -0.012, 0.009]
    source = target.transformed(regrasp)
    return target, source, np.linalg.inv(regrasp)


def _prepared_capture(path: Path) -> Path:
    prepared = path / "prepared"
    prepared.mkdir(parents=True)
    (prepared / "object_merged_tool0.pcd").touch()
    return path
