"""Headless object reconstruction composed from shared implementations."""

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from stacking_common.pointcloud import PointCloudData

from .cleanup import PointCloudCleaner
from .config import ReconstructionProfile
from .convex_decomposition import (
    ConvexDecomposer,
    build_dsf_point_sets,
    split_points_by_seeds,
)
from .dsf_fitting import DsfFitter
from .geometry import (
    DsfExporter,
    estimate_hull_inertial,
    estimate_mesh_inertial,
    validate_triangle_mesh,
)
from .meshing import MeshReconstructor, repair_triangle_mesh
from .types import (
    DsfPartitionOptions,
    InertialMetadata,
    ReconstructionOperatorOptions,
    ReconstructionResult,
    TriangleMeshData,
)


class ReconstructionPipeline:
    """Clean, center, mesh, partition, fit, and commission one object cloud."""

    def __init__(self, profile: ReconstructionProfile) -> None:
        self.profile = profile
        self.cleaner = PointCloudCleaner(profile.cleanup)
        self.mesher = MeshReconstructor(profile.mesh)
        self.fitter = DsfFitter(profile.dsf_fitting)
        self.exporter = DsfExporter(profile.dsf)

    def reconstruct(
        self,
        source: PointCloudData,
        *,
        operator_options: ReconstructionOperatorOptions | None = None,
        partition_options: DsfPartitionOptions | None = None,
        manual_split_seeds: np.ndarray | None = None,
        manual_remover: Callable[[PointCloudData, bool], PointCloudData] | None = None,
        manual_seed_provider: Callable[[PointCloudData], np.ndarray] | None = None,
        component_selector: Callable[[PointCloudData, int], int] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> ReconstructionResult:
        options = operator_options or ReconstructionOperatorOptions()
        partition = partition_options or DsfPartitionOptions()
        _report_operation(progress, "Cleaning the merged point cloud")
        cleaned = self.cleaner.clean(
            source,
            options,
            manual_remover=manual_remover,
        )
        hull_inertial = estimate_hull_inertial(cleaned.points)
        centered = PointCloudData(cleaned.points - hull_inertial.center_of_mass)
        _report_operation(progress, "Reconstructing the object surface mesh")
        mesh = self.mesher.reconstruct(centered)
        mesh_validation = validate_triangle_mesh(mesh)
        if not mesh_validation.valid and self.profile.mesh.repair_invalid_mesh:
            _report_operation(progress, "Repairing the reconstructed mesh")
            mesh = repair_triangle_mesh(mesh)
            repaired_validation = validate_triangle_mesh(
                mesh,
                repair_attempted=True,
            )
            mesh_validation = replace(
                repaired_validation,
                repaired=repaired_validation.valid,
            )
        try:
            mesh_inertial = estimate_mesh_inertial(mesh)
        except ValueError:
            inertial = hull_inertial
            inertial_method = "convex_hull"
        else:
            local_center = mesh_inertial.center_of_mass
            centered = PointCloudData(centered.points - local_center)
            mesh = TriangleMeshData(
                mesh.vertices - local_center,
                mesh.triangles,
            )
            inertial = InertialMetadata(
                mesh_inertial.mass,
                mesh_inertial.inertia,
                hull_inertial.center_of_mass + local_center,
            )
            inertial_method = "watertight_mesh"
        split_sets = None
        selected_components = self.profile.convex_decomposition.maximum_components
        if partition.manual_split:
            if manual_split_seeds is None and manual_seed_provider is not None:
                manual_split_seeds = manual_seed_provider(centered)
            if manual_split_seeds is None:
                raise ValueError(
                    "manual DSF partitioning requires explicit split seeds"
                )
            point_sets = split_points_by_seeds(
                centered.points,
                manual_split_seeds,
                minimum_points=self.profile.dsf_fitting.split_minimum_points,
            )
            parts = ()
            split_method = "manual_seeds"
        else:
            convex_profile = self.profile.convex_decomposition
            if component_selector is not None:
                selected_components = component_selector(
                    centered,
                    convex_profile.maximum_components,
                )
                convex_profile = replace(
                    convex_profile,
                    maximum_components=selected_components,
                )
            if convex_profile.enabled and convex_profile.maximum_components > 1:
                _report_operation(
                    progress,
                    "Running CoACD convex decomposition "
                    f"({convex_profile.mcts_iterations} MCTS iterations, "
                    f"up to {convex_profile.maximum_components} components)",
                )
            parts = ConvexDecomposer(convex_profile).decompose(mesh)
            if (
                progress is not None
                and convex_profile.enabled
                and convex_profile.maximum_components > 1
            ):
                progress(1, 1, f"CoACD produced {len(parts)} component(s)")
            split_method = (
                "coacd"
                if convex_profile.enabled
                and convex_profile.maximum_components > 1
                else "single"
            )
            if split_method == "single":
                point_sets = (np.asarray(centered.points),)
                split_sets = point_sets
            else:
                point_sets, split_sets = build_dsf_point_sets(
                    centered.points,
                    parts,
                    fit_source=self.profile.dsf_fitting.fit_source,
                    inside_tolerance=(
                        self.profile.dsf_fitting.split_inside_tolerance
                    ),
                    minimum_points=(
                        self.profile.dsf_fitting.split_minimum_points
                    ),
                )
        fit_results = ()
        if self.profile.dsf_fitting.enabled:
            _report_operation(
                progress,
                f"Fitting {len(point_sets)} smooth DSF component(s)",
            )
            fit_results = self.fitter.fit(point_sets, progress=progress)
            components = tuple(
                self.exporter.component_from_vertices(
                    result.vertices,
                    sharpness=max(2, int(result.sharpness)),
                )
                for result in fit_results
            )
        elif partition.manual_split:
            components = tuple(
                self.exporter.component_from_points(
                    points,
                    sharpness=self.profile.dsf.sharpness,
                )
                for points in point_sets
            )
        else:
            components = tuple(
                self.exporter.component_from_points(
                    part.vertices,
                    sharpness=self.profile.dsf.sharpness,
                )
                for part in parts
            )
        if progress is not None:
            progress(1, 1, "Reconstruction computation complete")
        return ReconstructionResult(
            source=source,
            cleaned=cleaned,
            centered=centered,
            mesh=mesh,
            convex_parts=parts,
            point_partitions=tuple(split_sets or point_sets),
            fit_results=fit_results,
            components=components,
            inertial=inertial,
            inertial_method=inertial_method,
            mesh_validation=mesh_validation,
            split_method=split_method,
            convex_maximum_components=selected_components,
        )


def _report_operation(
    progress: Callable[[int, int, str], None] | None,
    message: str,
) -> None:
    if progress is not None:
        progress(0, 0, message)
