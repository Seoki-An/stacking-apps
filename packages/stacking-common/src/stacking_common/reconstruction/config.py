"""Validated, platform-neutral reconstruction configuration schemas."""

from dataclasses import dataclass

import numpy as np

from stacking_common.pointcloud import (
    MultiscaleRegistrationConfig,
    RegistrationQualityConfig,
)


@dataclass(frozen=True)
class PoseGraphProfile:
    registration: MultiscaleRegistrationConfig
    sequential_quality: RegistrationQualityConfig
    loop_quality: RegistrationQualityConfig
    maximum_final_translation: float
    maximum_final_rotation_degrees: float
    edge_prune_threshold: float
    loop_closure_preference: float
    loop_minimum_capture_separation: int
    loop_maximum_position_delta: float
    loop_maximum_view_angle_degrees: float
    maximum_loops_per_node: int

    def __post_init__(self) -> None:
        limits = (
            self.maximum_final_translation,
            self.maximum_final_rotation_degrees,
        )
        if not np.all(np.isfinite(limits)) or any(value <= 0 for value in limits):
            raise ValueError("pose-graph correction limits must be positive")
        if not 0.0 <= self.edge_prune_threshold <= 1.0:
            raise ValueError("pose-graph edge prune threshold must be in [0, 1]")
        if not np.isfinite(self.loop_closure_preference) or self.loop_closure_preference <= 0.0:
            raise ValueError("loop-closure preference must be positive")
        if self.loop_minimum_capture_separation <= 0 or self.maximum_loops_per_node <= 0:
            raise ValueError("pose-graph loop counts must be positive")
        if self.loop_maximum_position_delta <= 0.0 or not 0.0 < self.loop_maximum_view_angle_degrees < 180.0:
            raise ValueError("pose-graph loop geometry limits are invalid")


@dataclass(frozen=True)
class ReconstructionCleanupProfile:
    statistical_neighbors: int
    statistical_std_ratio: float
    detached_voxel_size: float
    detached_eps: float
    detached_minimum_points: int
    detached_attach_distance: float
    detached_minimum_attached_size: int
    detached_keep_ratio: float
    fringe_neighbors: int
    fringe_margin: float
    fringe_reference_quantile: float
    cluster_voxel_size: float
    cluster_eps: float
    cluster_min_points: int
    manual_remove_distance: float
    manual_max_remove_ratio: float

    def __post_init__(self) -> None:
        metric = (
            self.statistical_std_ratio,
            self.detached_voxel_size,
            self.detached_eps,
            self.detached_attach_distance,
            self.fringe_margin,
            self.cluster_voxel_size,
            self.cluster_eps,
            self.manual_remove_distance,
        )
        if not np.all(np.isfinite(metric)) or any(
            value <= 0.0 for value in metric
        ):
            raise ValueError("reconstruction cleanup scales must be positive")
        counts = (
            self.statistical_neighbors,
            self.detached_minimum_points,
            self.detached_minimum_attached_size,
            self.fringe_neighbors,
            self.cluster_min_points,
        )
        if any(value < 2 for value in counts):
            raise ValueError("reconstruction cleanup neighbor counts are invalid")
        if not 0.0 < self.detached_keep_ratio <= 1.0:
            raise ValueError("detached-cluster keep ratio must lie in (0, 1]")
        if not 0.0 <= self.fringe_reference_quantile <= 1.0:
            raise ValueError("fringe reference quantile must lie in [0, 1]")
        if not 0.0 < self.manual_max_remove_ratio < 1.0:
            raise ValueError("manual removal ratio must lie in (0, 1)")


@dataclass(frozen=True)
class MeshReconstructionProfile:
    method: str
    voxel_size: float
    alpha: float
    bpa_radius_scales: tuple[float, ...]
    radial_subdivisions: int
    radial_neighbors: int
    radial_radius_quantile: float
    radial_radius_padding: float
    radial_center_quantile: float
    radial_empty_angle_degrees: float
    radial_empty_radius_quantile: float
    radial_support_angle_degrees: float
    radial_minimum_support_points: int
    radial_tighten_angle_degrees: float
    radial_tighten_quantile: float
    radial_tighten_blend: float
    radial_maximum_radius_quantile: float
    radial_smoothing_iterations: int
    radial_smoothing_weight: float
    normal_radius: float
    normal_max_neighbors: int
    poisson_depth: int
    poisson_density_quantile: float
    smoothing_iterations: int
    simplification_voxel_size: float
    keep_largest_component: bool
    repair_invalid_mesh: bool = True

    def __post_init__(self) -> None:
        metric = (self.voxel_size, self.alpha, self.normal_radius)
        if not np.all(np.isfinite(metric)) or any(
            value <= 0 for value in metric
        ):
            raise ValueError("mesh reconstruction scales must be positive")
        if self.method not in {"radial", "alpha", "bpa", "poisson"}:
            raise ValueError("mesh method must be radial, alpha, bpa, or poisson")
        if not self.bpa_radius_scales or any(
            not np.isfinite(value) or value <= 0.0
            for value in self.bpa_radius_scales
        ):
            raise ValueError("BPA radius scales must be positive")
        if self.normal_max_neighbors < 3 or self.poisson_depth < 1:
            raise ValueError("mesh reconstruction counts are invalid")
        if self.radial_subdivisions < 0 or self.radial_neighbors < 1:
            raise ValueError("radial mesh sampling counts are invalid")
        if self.radial_minimum_support_points < 0:
            raise ValueError("radial minimum support cannot be negative")
        quantiles = (
            self.radial_radius_quantile,
            self.radial_center_quantile,
            self.radial_empty_radius_quantile,
            self.radial_tighten_quantile,
            self.radial_maximum_radius_quantile,
        )
        if any(
            not np.isfinite(value) or not 0.0 <= value <= 1.0
            for value in quantiles
        ):
            raise ValueError("radial mesh quantiles must lie in [0, 1]")
        radial_nonnegative = (
            self.radial_radius_padding,
            self.radial_empty_angle_degrees,
            self.radial_support_angle_degrees,
            self.radial_tighten_angle_degrees,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in radial_nonnegative):
            raise ValueError("radial mesh scales and angles cannot be negative")
        if self.radial_smoothing_iterations < 0:
            raise ValueError("radial smoothing iterations cannot be negative")
        if not 0.0 <= self.radial_smoothing_weight <= 1.0:
            raise ValueError("radial smoothing weight must lie in [0, 1]")
        if not 0.0 <= self.radial_tighten_blend <= 1.0:
            raise ValueError("radial tightening blend must lie in [0, 1]")
        if not 0.0 <= self.poisson_density_quantile < 1.0:
            raise ValueError("Poisson density quantile must lie in [0, 1)")
        if self.smoothing_iterations < 0:
            raise ValueError("mesh smoothing iterations cannot be negative")
        if (
            not np.isfinite(self.simplification_voxel_size)
            or self.simplification_voxel_size < 0.0
        ):
            raise ValueError("mesh simplification voxel size cannot be negative")


@dataclass(frozen=True)
class ConvexDecompositionProfile:
    enabled: bool
    threshold: float
    maximum_components: int
    preprocess_mode: str
    preprocess_resolution: int
    resolution: int
    mcts_nodes: int
    mcts_iterations: int
    mcts_max_depth: int
    pca: bool
    merge: bool
    decimate: bool
    maximum_hull_vertices: int
    seed: int
    real_metric: bool

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold) or self.threshold <= 0.0:
            raise ValueError("convex decomposition threshold must be positive")
        if self.maximum_components < 1 or self.maximum_hull_vertices < 4:
            raise ValueError("convex decomposition component limits are invalid")
        if self.preprocess_mode not in {"auto", "on", "off"}:
            raise ValueError("convex preprocessing mode must be auto, on, or off")
        counts = (
            self.preprocess_resolution,
            self.resolution,
            self.mcts_nodes,
            self.mcts_iterations,
            self.mcts_max_depth,
        )
        if any(value < 1 for value in counts):
            raise ValueError("convex decomposition counts must be positive")
        if self.seed < 0:
            raise ValueError("convex decomposition seed cannot be negative")


@dataclass(frozen=True)
class DsfRansacProfile:
    enabled: bool
    iterations: int
    sample_ratio: float
    minimum_sample_points: int
    maximum_sample_points: int
    fit_maximum_iterations: int
    inlier_threshold: float
    minimum_inlier_ratio: float
    score_direction_subdivisions: int
    seed: int | None

    def __post_init__(self) -> None:
        if self.iterations < 1 or self.fit_maximum_iterations < 1:
            raise ValueError("DSF RANSAC iteration counts must be positive")
        if self.minimum_sample_points < 4:
            raise ValueError("DSF RANSAC needs at least four sample points")
        if self.maximum_sample_points < self.minimum_sample_points:
            raise ValueError("DSF RANSAC sample limits are invalid")
        if not 0.0 < self.sample_ratio <= 1.0:
            raise ValueError("DSF RANSAC sample ratio must lie in (0, 1]")
        if not np.isfinite(self.inlier_threshold) or self.inlier_threshold <= 0.0:
            raise ValueError("DSF RANSAC inlier threshold must be positive")
        if not 0.0 < self.minimum_inlier_ratio <= 1.0:
            raise ValueError("DSF RANSAC inlier ratio must lie in (0, 1]")
        if self.score_direction_subdivisions < 0:
            raise ValueError("DSF RANSAC direction subdivisions cannot be negative")
        if self.seed is not None and self.seed < 0:
            raise ValueError("DSF RANSAC seed cannot be negative")


@dataclass(frozen=True)
class DsfFittingProfile:
    enabled: bool
    fit_source: str
    total_vertices: int
    direction_subdivisions: int
    maximum_iterations: int
    support_quantile: float
    loss: str
    loss_scale: float
    center_interior_upper_bound: float
    maximum_sharpness: int
    split_inside_tolerance: float
    split_minimum_points: int
    ransac: DsfRansacProfile

    def __post_init__(self) -> None:
        if self.fit_source not in {"convex_vertices", "split_cloud", "both"}:
            raise ValueError("DSF fit source is invalid")
        if self.total_vertices < 4 or self.maximum_iterations < 1:
            raise ValueError("DSF fitting vertex and iteration counts are invalid")
        if self.direction_subdivisions < 0 or self.maximum_sharpness < 2:
            raise ValueError("DSF fitting direction or sharpness limits are invalid")
        if not 0.0 < self.support_quantile <= 1.0:
            raise ValueError("DSF support quantile must lie in (0, 1]")
        if self.loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
            raise ValueError("DSF fitting loss is unsupported")
        if not np.isfinite(self.loss_scale) or self.loss_scale <= 0.0:
            raise ValueError("DSF fitting loss scale must be positive")
        if not 0.25 < self.center_interior_upper_bound < 1.0:
            raise ValueError(
                "DSF center interior upper bound must lie in (0.25, 1)"
            )
        if (
            not np.isfinite(self.split_inside_tolerance)
            or self.split_inside_tolerance < 0.0
        ):
            raise ValueError("DSF split tolerance cannot be negative")
        if self.split_minimum_points < 4:
            raise ValueError("DSF split needs at least four points per component")


@dataclass(frozen=True)
class DsfExportProfile:
    maximum_nodes: int
    sharpness: int
    friction: float

    def __post_init__(self) -> None:
        if self.maximum_nodes < 4:
            raise ValueError("DSF export requires at least four nodes")
        if self.sharpness < 2:
            raise ValueError("DSF sharpness must be at least two")
        if not np.isfinite(self.friction) or self.friction < 0.0:
            raise ValueError("DSF friction must be non-negative")


@dataclass(frozen=True)
class ReconstructionProfile:
    merge_voxel_size: float
    minimum_view_support: int
    pose_graph: PoseGraphProfile
    cleanup: ReconstructionCleanupProfile
    mesh: MeshReconstructionProfile
    convex_decomposition: ConvexDecompositionProfile
    dsf_fitting: DsfFittingProfile
    dsf: DsfExportProfile

    def __post_init__(self) -> None:
        if not np.isfinite(self.merge_voxel_size) or self.merge_voxel_size <= 0.0:
            raise ValueError("reconstruction merge voxel size must be positive")
        if self.minimum_view_support <= 0:
            raise ValueError("reconstruction view support must be positive")
