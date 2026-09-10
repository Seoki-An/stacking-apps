"""Reconstruct geometry and DSF artifacts from one merged object cloud."""

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from stacking_common.pointcloud import PointCloudData
from stacking_common.reconstruction import (
    ReconstructionPipeline,
    ReconstructionResult,
)
from stacking_tabletop.config import (
    ConvexDecompositionProfile,
    DsfExportProfile,
    DsfFittingProfile,
    MeshReconstructionProfile,
    ReconstructionCleanupProfile,
    load_inventory_profile,
    load_perception_profile,
)

from .convex_decomposition import (
    DsfPartitionOptions,
    load_or_select_split_seeds,
    select_maximum_components,
    show_convex_decomposition,
    show_point_partitions,
)
from .reconstruction import (
    InteractivePointRemover,
    ReconstructionOperatorOptions,
    load_point_cloud,
    show_point_cloud,
    show_reconstruction,
    write_point_cloud,
    write_triangle_mesh,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct mesh, inertia, and DSF from a merged object PCD."
    )
    parser.add_argument("repository", type=Path)
    parser.add_argument(
        "merged_cloud",
        type=Path,
        help="Merged PCD/PLY file or merge artifact directory.",
    )
    parser.add_argument("--object-id", type=int, required=True)
    parser.add_argument(
        "--mass-kg",
        type=float,
        help="Optional measured-mass override for density and inertia.",
    )
    parser.add_argument("--session-id")
    visualization = parser.add_argument_group("visualization")
    visualization.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "Show input, cleanup, mesh, decomposition, point partitions, "
            "and final DSFs."
        ),
    )
    visualization.add_argument("--show-input", action="store_true")
    visualization.add_argument("--show-cleaned", action="store_true")
    visualization.add_argument("--show-mesh", action="store_true")
    visualization.add_argument("--show-result", action="store_true")
    visualization.add_argument(
        "--review-result",
        action="store_true",
        help=(
            "Review the smooth DSF result and press R to reselect the "
            "CoACD component count or manual split seeds."
        ),
    )
    visualization.add_argument(
        "--visualization-opacity",
        type=float,
        default=0.35,
        help="Mesh opacity in overlay windows (default: 0.35).",
    )

    cleanup = parser.add_argument_group("point-cloud cleanup")
    cleanup.add_argument(
        "--no-statistical-outlier-removal",
        dest="remove_statistical_outliers",
        action="store_false",
        help="Disable the default statistical outlier removal.",
    )
    cleanup.add_argument(
        "--stat-neighbors",
        type=int,
        help="Override statistical outlier neighbor count.",
    )
    cleanup.add_argument(
        "--stat-std-ratio",
        type=float,
        help="Override statistical outlier standard-deviation ratio.",
    )
    cleanup.add_argument(
        "--no-detached-cluster-removal",
        dest="remove_detached_clusters",
        action="store_false",
        help="Disable the default detached-island cleanup.",
    )
    cleanup.add_argument("--detached-voxel-size", type=float)
    cleanup.add_argument("--detached-eps", type=float)
    cleanup.add_argument("--detached-minimum-points", type=int)
    cleanup.add_argument("--detached-attach-distance", type=float)
    cleanup.add_argument("--detached-minimum-attached-size", type=int)
    cleanup.add_argument("--detached-keep-ratio", type=float)
    cleanup.add_argument(
        "--no-radial-fringe-removal",
        dest="remove_radial_fringe",
        action="store_false",
        help="Disable the default outward radial-fringe cleanup.",
    )
    cleanup.add_argument("--fringe-neighbors", type=int)
    cleanup.add_argument("--fringe-margin", type=float)
    cleanup.add_argument("--fringe-reference-quantile", type=float)
    cleanup.add_argument(
        "--keep-all-clusters",
        dest="keep_largest_cluster",
        action="store_false",
        help="Disable the default largest-cluster filter.",
    )
    cleanup.add_argument("--cluster-voxel-size", type=float)
    cleanup.add_argument("--cluster-eps", type=float)
    cleanup.add_argument("--cluster-min-points", type=int)
    cleanup.add_argument(
        "--manual-remove",
        action="store_true",
        help="Interactively pick unwanted point neighborhoods before meshing.",
    )
    cleanup.add_argument("--manual-remove-distance", type=float)
    cleanup.add_argument("--manual-max-remove-ratio", type=float)
    cleanup.add_argument(
        "--manual-left-click",
        action="store_true",
        help="Use left click instead of P for manual removal picks.",
    )
    cleanup.set_defaults(
        remove_statistical_outliers=True,
        remove_detached_clusters=True,
        remove_radial_fringe=True,
        keep_largest_cluster=True,
    )

    mesh = parser.add_argument_group("mesh reconstruction")
    mesh.add_argument(
        "--mesh-method",
        choices=("radial", "alpha", "bpa", "poisson"),
    )
    mesh.add_argument("--mesh-voxel-size", type=float)
    mesh.add_argument("--mesh-alpha", type=float)
    mesh.add_argument(
        "--mesh-bpa-radius-scales",
        type=_positive_float_tuple,
        metavar="SCALE,...",
    )
    mesh.add_argument("--mesh-radial-subdivisions", type=int)
    mesh.add_argument("--mesh-radial-neighbors", type=int)
    mesh.add_argument("--mesh-radial-radius-quantile", type=float)
    mesh.add_argument("--mesh-radial-radius-padding", type=float)
    mesh.add_argument("--mesh-radial-center-quantile", type=float)
    mesh.add_argument("--mesh-radial-empty-angle-degrees", type=float)
    mesh.add_argument("--mesh-radial-empty-radius-quantile", type=float)
    mesh.add_argument("--mesh-radial-support-angle-degrees", type=float)
    mesh.add_argument("--mesh-radial-minimum-support-points", type=int)
    mesh.add_argument("--mesh-radial-tighten-angle-degrees", type=float)
    mesh.add_argument("--mesh-radial-tighten-quantile", type=float)
    mesh.add_argument("--mesh-radial-tighten-blend", type=float)
    mesh.add_argument("--mesh-radial-maximum-radius-quantile", type=float)
    mesh.add_argument("--mesh-radial-smoothing-iterations", type=int)
    mesh.add_argument("--mesh-radial-smoothing-weight", type=float)
    mesh.add_argument("--mesh-normal-radius", type=float)
    mesh.add_argument("--mesh-normal-max-neighbors", type=int)
    mesh.add_argument("--mesh-poisson-depth", type=int)
    mesh.add_argument("--mesh-poisson-density-quantile", type=float)
    mesh.add_argument("--mesh-smoothing-iterations", type=int)
    mesh.add_argument("--mesh-simplification-voxel-size", type=float)
    mesh.add_argument(
        "--keep-mesh-components",
        action="store_true",
        help="Keep every connected mesh component instead of only the largest.",
    )

    convex = parser.add_argument_group("convex decomposition")
    convex.add_argument(
        "--no-convex-decomposition",
        action="store_true",
        help="Fit/export one DSF without running the default CoACD stage.",
    )
    convex.add_argument("--convex-threshold", type=float)
    convex.add_argument("--convex-maximum-components", type=int)
    convex.add_argument(
        "--convex-preprocess-mode",
        choices=("auto", "on", "off"),
    )
    convex.add_argument("--convex-preprocess-resolution", type=int)
    convex.add_argument("--convex-resolution", type=int)
    convex.add_argument("--convex-mcts-nodes", type=int)
    convex.add_argument("--convex-mcts-iterations", type=int)
    convex.add_argument("--convex-mcts-max-depth", type=int)
    convex.add_argument("--convex-pca", action="store_true")
    convex.add_argument("--no-convex-merge", action="store_true")
    convex.add_argument("--convex-decimate", action="store_true")
    convex.add_argument("--convex-maximum-hull-vertices", type=int)
    convex.add_argument("--convex-seed", type=int)
    convex.add_argument("--convex-real-metric", action="store_true")
    convex.add_argument("--show-convex", action="store_true")
    convex.add_argument(
        "--prompt-convex-maximum-components",
        action="store_true",
        help="Select the CoACD component cap interactively for this object.",
    )

    fitting = parser.add_argument_group("DSF fitting and partitioning")
    fitting.add_argument(
        "--no-dsf-fitting",
        action="store_true",
        help="Export convex vertices directly without optimized DSF fitting.",
    )
    fitting.add_argument(
        "--dsf-fit-source",
        choices=("convex_vertices", "split_cloud", "both"),
    )
    fitting.add_argument("--dsf-fit-vertices", type=int)
    fitting.add_argument("--dsf-direction-subdivisions", type=int)
    fitting.add_argument("--dsf-fit-maximum-iterations", type=int)
    fitting.add_argument("--dsf-support-quantile", type=float)
    fitting.add_argument(
        "--dsf-loss",
        choices=("linear", "soft_l1", "huber", "cauchy", "arctan"),
    )
    fitting.add_argument("--dsf-loss-scale", type=float)
    fitting.add_argument("--dsf-center-interior-upper-bound", type=float)
    fitting.add_argument("--dsf-maximum-sharpness", type=int)
    fitting.add_argument("--dsf-split-inside-tolerance", type=float)
    fitting.add_argument("--dsf-split-minimum-points", type=int)
    fitting.add_argument("--no-dsf-ransac", action="store_true")
    fitting.add_argument("--dsf-ransac-iterations", type=int)
    fitting.add_argument("--dsf-ransac-sample-ratio", type=float)
    fitting.add_argument("--dsf-ransac-minimum-sample-points", type=int)
    fitting.add_argument("--dsf-ransac-maximum-sample-points", type=int)
    fitting.add_argument("--dsf-ransac-fit-maximum-iterations", type=int)
    fitting.add_argument("--dsf-ransac-inlier-threshold", type=float)
    fitting.add_argument("--dsf-ransac-minimum-inlier-ratio", type=float)
    fitting.add_argument("--dsf-ransac-score-direction-subdivisions", type=int)
    fitting.add_argument("--dsf-ransac-seed", type=int)
    fitting.add_argument(
        "--manual-split",
        action="store_true",
        help="Pick one seed per desired DSF component instead of using CoACD.",
    )
    fitting.add_argument("--manual-split-seed-file", type=Path)
    fitting.add_argument("--force-manual-split-selection", action="store_true")
    fitting.add_argument("--manual-split-left-click", action="store_true")
    fitting.add_argument("--manual-split-marker-radius", type=float, default=0.003)
    fitting.add_argument("--show-dsf-partitions", action="store_true")

    dsf = parser.add_argument_group("DSF export")
    dsf.add_argument(
        "--dsf-maximum-nodes",
        type=int,
        help="Maximum direct-export nodes when DSF fitting is disabled.",
    )
    dsf.add_argument(
        "--dsf-sharpness",
        type=int,
        help="Fixed direct-export sharpness when DSF fitting is disabled.",
    )
    dsf.add_argument("--dsf-friction", type=float)
    return parser


def default_reconstruction_directory(
    repository: str | Path,
    object_id: int,
    session_id: str | None = None,
) -> Path:
    if object_id <= 0:
        raise ValueError("object ID must be positive")
    name = session_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("session ID must be one path-segment name")
    return (
        Path(repository).resolve()
        / "apps"
        / "tabletop"
        / "data"
        / "reconstruction"
        / str(object_id)
        / name
    )


def resolve_merged_cloud(path: str | Path) -> Path:
    value = Path(path).expanduser().resolve()
    if value.is_file():
        return value
    candidate = value / "merged.pcd"
    if value.is_dir() and candidate.is_file():
        return candidate
    raise FileNotFoundError(f"merged object cloud does not exist: {value}")


def run_object_reconstruction(
    repository: str | Path,
    merged_cloud: str | Path,
    session_directory: str | Path,
    *,
    object_id: int,
    cleanup_profile: ReconstructionCleanupProfile | None = None,
    mesh_profile: MeshReconstructionProfile | None = None,
    convex_decomposition_profile: ConvexDecompositionProfile | None = None,
    dsf_fitting_profile: DsfFittingProfile | None = None,
    dsf_profile: DsfExportProfile | None = None,
    operator_options: ReconstructionOperatorOptions | None = None,
    partition_options: DsfPartitionOptions | None = None,
    measured_mass_kg: float | None = None,
    density_kg_m3: float | None = None,
    manual_remover: Callable[[PointCloudData, bool], PointCloudData] | None = None,
    manual_seed_provider: Callable[[PointCloudData], object] | None = None,
    component_selector: Callable[[PointCloudData, int], int] | None = None,
    result_reviewer: Callable[[ReconstructionResult], bool] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> Path:
    if object_id <= 0:
        raise ValueError("object ID must be positive")
    if measured_mass_kg is not None and measured_mass_kg <= 0.0:
        raise ValueError("measured mass must be positive")
    if density_kg_m3 is not None and density_kg_m3 <= 0.0:
        raise ValueError("configured density must be positive")
    root = Path(repository).resolve()
    session = Path(session_directory).resolve()
    source_path = resolve_merged_cloud(merged_cloud)
    source = load_point_cloud(source_path)
    profile = load_perception_profile(
        root / "apps" / "tabletop" / "config" / "perception.yml"
    )
    cleanup_profile = cleanup_profile or profile.reconstruction.cleanup
    mesh_profile = mesh_profile or profile.reconstruction.mesh
    convex_decomposition_profile = (
        convex_decomposition_profile
        or profile.reconstruction.convex_decomposition
    )
    dsf_fitting_profile = (
        dsf_fitting_profile or profile.reconstruction.dsf_fitting
    )
    dsf_profile = dsf_profile or profile.reconstruction.dsf
    options = operator_options or ReconstructionOperatorOptions()
    partition = partition_options or DsfPartitionOptions()
    reconstruction_profile = replace(
        profile.reconstruction,
        cleanup=cleanup_profile,
        mesh=mesh_profile,
        convex_decomposition=convex_decomposition_profile,
        dsf_fitting=dsf_fitting_profile,
        dsf=dsf_profile,
    )
    pipeline = ReconstructionPipeline(reconstruction_profile)
    force_seed_selection = partition.force_seed_selection
    if options.show_input:
        show_point_cloud(source, "Merged object cloud: input")
    while True:
        seed_provider = manual_seed_provider
        if partition.manual_split and seed_provider is None:
            seed_provider = lambda cloud: load_or_select_split_seeds(
                cloud,
                path=partition.seed_path,
                force_selection=force_seed_selection,
                left_clicking=partition.left_clicking,
                marker_radius=partition.marker_radius,
            )
        selected_component_selector = component_selector
        if (
            selected_component_selector is None
            and partition.prompt_maximum_components
            and not partition.manual_split
        ):
            selected_component_selector = select_maximum_components
        selected_remover = manual_remover
        if selected_remover is None:
            remover = InteractivePointRemover(cleanup_profile)
            selected_remover = lambda cloud, left: remover.remove(
                cloud,
                left_clicking=left,
            )
        result = pipeline.reconstruct(
            source,
            operator_options=options,
            partition_options=partition,
            manual_remover=selected_remover,
            manual_seed_provider=seed_provider,
            component_selector=selected_component_selector,
            progress=progress,
        )
        if options.show_cleaned:
            show_point_cloud(result.cleaned, "Merged object cloud: cleaned")
        if options.show_mesh:
            show_reconstruction(
                result.centered,
                result.mesh,
                mesh_opacity=options.visualization_opacity,
                title="Centered cloud and reconstructed mesh",
            )
        if partition.show_convex and result.convex_parts:
            show_convex_decomposition(result.centered, result.convex_parts)
        if partition.show_partitions:
            show_point_partitions(result.point_partitions)
        if options.show_result or result_reviewer is not None:
            retry_label = None
            if partition.review_result and partition.manual_split:
                retry_label = "re-pick manual split seeds"
            elif partition.review_result and partition.prompt_maximum_components:
                retry_label = "re-select CoACD maximum components"
            accepted = (
                result_reviewer(result)
                if result_reviewer is not None
                else show_reconstruction(
                    result.centered,
                    result.mesh,
                    components=result.components,
                    mesh_opacity=options.visualization_opacity,
                    title="Final mesh and smooth DSF support surface",
                    retry_label=retry_label,
                )
            )
            if not accepted:
                if partition.manual_split or partition.prompt_maximum_components:
                    force_seed_selection = True
                    continue
                raise ValueError("operator rejected the reconstruction result")
        break
    cleaned = result.cleaned
    centered = result.centered
    mesh = result.mesh
    parts = result.convex_parts
    fit_results = result.fit_results
    components = result.components
    inertial = result.inertial
    split_method = result.split_method
    selected_convex_profile = replace(
        convex_decomposition_profile,
        maximum_components=result.convex_maximum_components,
    )
    output = session / "output"
    if progress is not None:
        progress(0, 0, "Writing reconstruction artifacts")
    merged_path = write_point_cloud(output / "merged.pcd", centered)
    mesh_path = write_triangle_mesh(output / "mesh.ply", mesh)
    dsf_path = pipeline.exporter.write(output / "dsf.obj", components, inertial)
    metadata_path = None
    effective_density = None
    mass_kg = None
    mass_source = None
    commissioned_inertia = None
    if measured_mass_kg is not None or density_kg_m3 is not None:
        if measured_mass_kg is not None:
            mass_kg = measured_mass_kg
            effective_density = measured_mass_kg / inertial.mass
            mass_source = "measured_override"
        else:
            effective_density = density_kg_m3
            mass_kg = effective_density * inertial.mass
            mass_source = "configured_density"
        commissioned_inertia = effective_density * inertial.inertia
        metadata_path = output / "object_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "object_id": object_id,
                    "frame": "object_center_of_mass",
                    "mass_kg": mass_kg,
                    "mass_source": mass_source,
                    "measured_mass_kg": measured_mass_kg,
                    "configured_density_kg_m3": density_kg_m3,
                    "effective_density_kg_m3": effective_density,
                    "volume_method": result.inertial_method,
                    "mesh_validation": asdict(result.mesh_validation),
                    "center_of_mass_in_source": inertial.center_of_mass.tolist(),
                    "inertia_kg_m2": commissioned_inertia.tolist(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    report = {
        "schema_version": 6,
        "status": "exported",
        "object_id": object_id,
        "source_merged_cloud": str(source_path),
        "source_coordinate_frame": "first_capture_tool0",
        "output_coordinate_frame": "object_center_of_mass",
        "point_count": len(source),
        "cleaned_point_count": len(cleaned),
        "cleanup": {
            "profile": asdict(cleanup_profile),
            "remove_statistical_outliers": options.remove_statistical_outliers,
            "remove_detached_clusters": options.remove_detached_clusters,
            "remove_radial_fringe": options.remove_radial_fringe,
            "keep_largest_cluster": options.keep_largest_cluster,
            "manual_remove": options.manual_remove,
        },
        "mesh_profile": asdict(mesh_profile),
        "mesh_validation": {
            **asdict(result.mesh_validation),
            "valid": result.mesh_validation.valid,
        },
        "convex_decomposition": {
            "profile": asdict(selected_convex_profile),
            "method": split_method,
            "decomposition_part_count": len(parts),
            "component_count": len(components),
        },
        "dsf_fitting": {
            "profile": asdict(dsf_fitting_profile),
            "manual_split": partition.manual_split,
            "seed_path": (
                None if partition.seed_path is None else str(partition.seed_path)
            ),
            "results": [
                {
                    "vertices": len(result.vertices),
                    "sharpness": result.sharpness,
                    "cost": result.cost,
                    "evaluations": result.evaluations,
                }
                for result in fit_results
            ],
        },
        "dsf_profile": asdict(dsf_profile),
        "inertial": {
            "unit_density_mass": inertial.mass,
            "center_of_mass_in_source": inertial.center_of_mass.tolist(),
            "inertia": inertial.inertia.tolist(),
            "mass_kg": mass_kg,
            "mass_source": mass_source,
            "measured_mass_kg": measured_mass_kg,
            "configured_density_kg_m3": density_kg_m3,
            "effective_density_kg_m3": effective_density,
            "volume_method": result.inertial_method,
            "commissioned_inertia_kg_m2": (
                None
                if commissioned_inertia is None
                else commissioned_inertia.tolist()
            ),
        },
        "outputs": {
            "merged_point_cloud": str(merged_path.relative_to(session)),
            "mesh": str(mesh_path.relative_to(session)),
            "dsf": str(dsf_path.relative_to(session)),
            "metadata": (
                None
                if metadata_path is None
                else str(metadata_path.relative_to(session))
            ),
        },
    }
    report_path = session / "reconstruction.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if progress is not None:
        progress(1, 1, "Reconstruction artifacts written")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        repository = Path(args.repository).resolve()
        profile = load_perception_profile(
            repository
            / "apps"
            / "tabletop"
            / "config"
            / "perception.yml"
        ).reconstruction
        inventory_profile = load_inventory_profile(
            repository / "apps" / "tabletop" / "config" / "inventory.yml"
        )
        if args.object_id not in inventory_profile.object_ids:
            raise ValueError(
                f"object {args.object_id} is not in the tabletop catalog"
            )
        cleanup_profile = replace(
            profile.cleanup,
            statistical_neighbors=_value(
                args.stat_neighbors,
                profile.cleanup.statistical_neighbors,
            ),
            statistical_std_ratio=_value(
                args.stat_std_ratio,
                profile.cleanup.statistical_std_ratio,
            ),
            detached_voxel_size=_value(
                args.detached_voxel_size,
                profile.cleanup.detached_voxel_size,
            ),
            detached_eps=_value(
                args.detached_eps,
                profile.cleanup.detached_eps,
            ),
            detached_minimum_points=_value(
                args.detached_minimum_points,
                profile.cleanup.detached_minimum_points,
            ),
            detached_attach_distance=_value(
                args.detached_attach_distance,
                profile.cleanup.detached_attach_distance,
            ),
            detached_minimum_attached_size=_value(
                args.detached_minimum_attached_size,
                profile.cleanup.detached_minimum_attached_size,
            ),
            detached_keep_ratio=_value(
                args.detached_keep_ratio,
                profile.cleanup.detached_keep_ratio,
            ),
            fringe_neighbors=_value(
                args.fringe_neighbors,
                profile.cleanup.fringe_neighbors,
            ),
            fringe_margin=_value(
                args.fringe_margin,
                profile.cleanup.fringe_margin,
            ),
            fringe_reference_quantile=_value(
                args.fringe_reference_quantile,
                profile.cleanup.fringe_reference_quantile,
            ),
            cluster_voxel_size=_value(
                args.cluster_voxel_size,
                profile.cleanup.cluster_voxel_size,
            ),
            cluster_eps=_value(args.cluster_eps, profile.cleanup.cluster_eps),
            cluster_min_points=_value(
                args.cluster_min_points,
                profile.cleanup.cluster_min_points,
            ),
            manual_remove_distance=_value(
                args.manual_remove_distance,
                profile.cleanup.manual_remove_distance,
            ),
            manual_max_remove_ratio=_value(
                args.manual_max_remove_ratio,
                profile.cleanup.manual_max_remove_ratio,
            ),
        )
        mesh_profile = replace(
            profile.mesh,
            method=_value(args.mesh_method, profile.mesh.method),
            voxel_size=_value(args.mesh_voxel_size, profile.mesh.voxel_size),
            alpha=_value(args.mesh_alpha, profile.mesh.alpha),
            bpa_radius_scales=_value(
                args.mesh_bpa_radius_scales,
                profile.mesh.bpa_radius_scales,
            ),
            radial_subdivisions=_value(
                args.mesh_radial_subdivisions,
                profile.mesh.radial_subdivisions,
            ),
            radial_neighbors=_value(
                args.mesh_radial_neighbors,
                profile.mesh.radial_neighbors,
            ),
            radial_radius_quantile=_value(
                args.mesh_radial_radius_quantile,
                profile.mesh.radial_radius_quantile,
            ),
            radial_radius_padding=_value(
                args.mesh_radial_radius_padding,
                profile.mesh.radial_radius_padding,
            ),
            radial_center_quantile=_value(
                args.mesh_radial_center_quantile,
                profile.mesh.radial_center_quantile,
            ),
            radial_empty_angle_degrees=_value(
                args.mesh_radial_empty_angle_degrees,
                profile.mesh.radial_empty_angle_degrees,
            ),
            radial_empty_radius_quantile=_value(
                args.mesh_radial_empty_radius_quantile,
                profile.mesh.radial_empty_radius_quantile,
            ),
            radial_support_angle_degrees=_value(
                args.mesh_radial_support_angle_degrees,
                profile.mesh.radial_support_angle_degrees,
            ),
            radial_minimum_support_points=_value(
                args.mesh_radial_minimum_support_points,
                profile.mesh.radial_minimum_support_points,
            ),
            radial_tighten_angle_degrees=_value(
                args.mesh_radial_tighten_angle_degrees,
                profile.mesh.radial_tighten_angle_degrees,
            ),
            radial_tighten_quantile=_value(
                args.mesh_radial_tighten_quantile,
                profile.mesh.radial_tighten_quantile,
            ),
            radial_tighten_blend=_value(
                args.mesh_radial_tighten_blend,
                profile.mesh.radial_tighten_blend,
            ),
            radial_maximum_radius_quantile=_value(
                args.mesh_radial_maximum_radius_quantile,
                profile.mesh.radial_maximum_radius_quantile,
            ),
            radial_smoothing_iterations=_value(
                args.mesh_radial_smoothing_iterations,
                profile.mesh.radial_smoothing_iterations,
            ),
            radial_smoothing_weight=_value(
                args.mesh_radial_smoothing_weight,
                profile.mesh.radial_smoothing_weight,
            ),
            normal_radius=_value(
                args.mesh_normal_radius,
                profile.mesh.normal_radius,
            ),
            normal_max_neighbors=_value(
                args.mesh_normal_max_neighbors,
                profile.mesh.normal_max_neighbors,
            ),
            poisson_depth=_value(
                args.mesh_poisson_depth,
                profile.mesh.poisson_depth,
            ),
            poisson_density_quantile=_value(
                args.mesh_poisson_density_quantile,
                profile.mesh.poisson_density_quantile,
            ),
            smoothing_iterations=_value(
                args.mesh_smoothing_iterations,
                profile.mesh.smoothing_iterations,
            ),
            simplification_voxel_size=_value(
                args.mesh_simplification_voxel_size,
                profile.mesh.simplification_voxel_size,
            ),
            keep_largest_component=(
                False
                if args.keep_mesh_components
                else profile.mesh.keep_largest_component
            ),
        )
        convex_profile = replace(
            profile.convex_decomposition,
            enabled=(
                False
                if args.no_convex_decomposition
                else profile.convex_decomposition.enabled
            ),
            threshold=_value(
                args.convex_threshold,
                profile.convex_decomposition.threshold,
            ),
            maximum_components=_value(
                args.convex_maximum_components,
                profile.convex_decomposition.maximum_components,
            ),
            preprocess_mode=_value(
                args.convex_preprocess_mode,
                profile.convex_decomposition.preprocess_mode,
            ),
            preprocess_resolution=_value(
                args.convex_preprocess_resolution,
                profile.convex_decomposition.preprocess_resolution,
            ),
            resolution=_value(
                args.convex_resolution,
                profile.convex_decomposition.resolution,
            ),
            mcts_nodes=_value(
                args.convex_mcts_nodes,
                profile.convex_decomposition.mcts_nodes,
            ),
            mcts_iterations=_value(
                args.convex_mcts_iterations,
                profile.convex_decomposition.mcts_iterations,
            ),
            mcts_max_depth=_value(
                args.convex_mcts_max_depth,
                profile.convex_decomposition.mcts_max_depth,
            ),
            pca=args.convex_pca or profile.convex_decomposition.pca,
            merge=(
                False
                if args.no_convex_merge
                else profile.convex_decomposition.merge
            ),
            decimate=(
                args.convex_decimate
                or profile.convex_decomposition.decimate
            ),
            maximum_hull_vertices=_value(
                args.convex_maximum_hull_vertices,
                profile.convex_decomposition.maximum_hull_vertices,
            ),
            seed=_value(
                args.convex_seed,
                profile.convex_decomposition.seed,
            ),
            real_metric=(
                args.convex_real_metric
                or profile.convex_decomposition.real_metric
            ),
        )
        ransac_profile = replace(
            profile.dsf_fitting.ransac,
            enabled=(
                False
                if args.no_dsf_ransac
                else profile.dsf_fitting.ransac.enabled
            ),
            iterations=_value(
                args.dsf_ransac_iterations,
                profile.dsf_fitting.ransac.iterations,
            ),
            sample_ratio=_value(
                args.dsf_ransac_sample_ratio,
                profile.dsf_fitting.ransac.sample_ratio,
            ),
            minimum_sample_points=_value(
                args.dsf_ransac_minimum_sample_points,
                profile.dsf_fitting.ransac.minimum_sample_points,
            ),
            maximum_sample_points=_value(
                args.dsf_ransac_maximum_sample_points,
                profile.dsf_fitting.ransac.maximum_sample_points,
            ),
            fit_maximum_iterations=_value(
                args.dsf_ransac_fit_maximum_iterations,
                profile.dsf_fitting.ransac.fit_maximum_iterations,
            ),
            inlier_threshold=_value(
                args.dsf_ransac_inlier_threshold,
                profile.dsf_fitting.ransac.inlier_threshold,
            ),
            minimum_inlier_ratio=_value(
                args.dsf_ransac_minimum_inlier_ratio,
                profile.dsf_fitting.ransac.minimum_inlier_ratio,
            ),
            score_direction_subdivisions=_value(
                args.dsf_ransac_score_direction_subdivisions,
                profile.dsf_fitting.ransac.score_direction_subdivisions,
            ),
            seed=_value(
                args.dsf_ransac_seed,
                profile.dsf_fitting.ransac.seed,
            ),
        )
        fitting_profile = replace(
            profile.dsf_fitting,
            enabled=(
                False
                if args.no_dsf_fitting
                else profile.dsf_fitting.enabled
            ),
            fit_source=_value(
                args.dsf_fit_source,
                profile.dsf_fitting.fit_source,
            ),
            total_vertices=_value(
                args.dsf_fit_vertices,
                profile.dsf_fitting.total_vertices,
            ),
            direction_subdivisions=_value(
                args.dsf_direction_subdivisions,
                profile.dsf_fitting.direction_subdivisions,
            ),
            maximum_iterations=_value(
                args.dsf_fit_maximum_iterations,
                profile.dsf_fitting.maximum_iterations,
            ),
            support_quantile=_value(
                args.dsf_support_quantile,
                profile.dsf_fitting.support_quantile,
            ),
            loss=_value(args.dsf_loss, profile.dsf_fitting.loss),
            loss_scale=_value(
                args.dsf_loss_scale,
                profile.dsf_fitting.loss_scale,
            ),
            center_interior_upper_bound=_value(
                args.dsf_center_interior_upper_bound,
                profile.dsf_fitting.center_interior_upper_bound,
            ),
            maximum_sharpness=_value(
                args.dsf_maximum_sharpness,
                profile.dsf_fitting.maximum_sharpness,
            ),
            split_inside_tolerance=_value(
                args.dsf_split_inside_tolerance,
                profile.dsf_fitting.split_inside_tolerance,
            ),
            split_minimum_points=_value(
                args.dsf_split_minimum_points,
                profile.dsf_fitting.split_minimum_points,
            ),
            ransac=ransac_profile,
        )
        dsf_profile = replace(
            profile.dsf,
            maximum_nodes=_value(
                args.dsf_maximum_nodes,
                profile.dsf.maximum_nodes,
            ),
            sharpness=_value(args.dsf_sharpness, profile.dsf.sharpness),
            friction=_value(args.dsf_friction, profile.dsf.friction),
        )
        operator = ReconstructionOperatorOptions(
            remove_statistical_outliers=args.remove_statistical_outliers,
            remove_detached_clusters=args.remove_detached_clusters,
            remove_radial_fringe=args.remove_radial_fringe,
            keep_largest_cluster=args.keep_largest_cluster,
            manual_remove=args.manual_remove,
            manual_left_click=args.manual_left_click,
            show_input=args.visualize or args.show_input,
            show_cleaned=args.visualize or args.show_cleaned,
            show_mesh=args.visualize or args.show_mesh,
            show_result=(
                args.visualize or args.show_result or args.review_result
            ),
            visualization_opacity=args.visualization_opacity,
        )
        partition = DsfPartitionOptions(
            manual_split=args.manual_split,
            seed_path=args.manual_split_seed_file,
            force_seed_selection=args.force_manual_split_selection,
            left_clicking=args.manual_split_left_click,
            marker_radius=args.manual_split_marker_radius,
            show_convex=args.visualize or args.show_convex,
            show_partitions=args.visualize or args.show_dsf_partitions,
            prompt_maximum_components=args.prompt_convex_maximum_components,
            review_result=args.review_result,
        )
        session = default_reconstruction_directory(
            args.repository,
            args.object_id,
            args.session_id,
        )
        report = run_object_reconstruction(
            args.repository,
            args.merged_cloud,
            session,
            object_id=args.object_id,
            cleanup_profile=cleanup_profile,
            mesh_profile=mesh_profile,
            convex_decomposition_profile=convex_profile,
            dsf_fitting_profile=fitting_profile,
            dsf_profile=dsf_profile,
            operator_options=operator,
            partition_options=partition,
            measured_mass_kg=args.mass_kg,
            density_kg_m3=inventory_profile.density_for(args.object_id),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"object reconstruction exported: {report}")
    return 0


def _positive_float_tuple(value: str) -> tuple[float, ...]:
    try:
        result = tuple(
            float(part.strip())
            for part in value.split(",")
            if part.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not result or any(number <= 0.0 for number in result):
        raise argparse.ArgumentTypeError("all radius scales must be positive")
    return result


def _value(value, default):
    return default if value is None else value


if __name__ == "__main__":
    raise SystemExit(main())
