"""Loading of explicit tabletop perception profiles."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

import numpy as np

from stacking_common.pointcloud import (
    CrossCaptureMergeProfile,
    MultiscaleRegistrationConfig,
    RegistrationQualityConfig,
)
from stacking_common.reconstruction import (
    ConvexDecompositionProfile,
    DsfExportProfile,
    DsfFittingProfile,
    DsfRansacProfile,
    MeshReconstructionProfile,
    PoseGraphProfile,
    ReconstructionCleanupProfile,
    ReconstructionProfile,
)


@dataclass(frozen=True)
class FieldSegmentationProfile:
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    voxel_size: float
    table_distance: float
    designated_table_height: float
    maximum_object_height: float
    cluster_distance: float
    minimum_cluster_points: int
    random_seed: int
    maximum_table_tilt_degrees: float
    maximum_plane_candidates: int
    ransac_iterations: int

    def __post_init__(self) -> None:
        lower = np.asarray(self.workspace_min, dtype=np.float64)
        upper = np.asarray(self.workspace_max, dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("segmentation workspace bounds must be XYZ")
        if not np.all(np.isfinite([*lower, *upper])) or np.any(lower >= upper):
            raise ValueError("segmentation workspace bounds are invalid")
        metric = (
            self.voxel_size,
            self.table_distance,
            self.maximum_object_height,
            self.cluster_distance,
        )
        if not np.all(np.isfinite(metric)) or any(value <= 0 for value in metric):
            raise ValueError("segmentation metric values must be positive")
        if not np.isfinite(self.designated_table_height):
            raise ValueError("designated table height must be finite")
        if self.minimum_cluster_points <= 0:
            raise ValueError("minimum cluster points must be positive")
        if not 0.0 <= self.maximum_table_tilt_degrees < 90.0:
            raise ValueError("maximum table tilt must be in [0, 90)")
        if self.maximum_plane_candidates <= 0 or self.ransac_iterations <= 0:
            raise ValueError("plane fitting counts must be positive")
        lower = lower.copy()
        upper = upper.copy()
        lower.setflags(write=False)
        upper.setflags(write=False)
        object.__setattr__(self, "workspace_min", lower)
        object.__setattr__(self, "workspace_max", upper)


@dataclass(frozen=True)
class DualFemtoFusionProfile:
    registration: MultiscaleRegistrationConfig
    merge_voxel_size: float
    table_voxel_size: float
    table_distance: float
    maximum_object_height: float
    maximum_translation: float
    maximum_rotation_degrees: float
    minimum_symmetric_overlap: float
    maximum_symmetric_rmse: float

    def __post_init__(self) -> None:
        metric = (
            self.merge_voxel_size,
            self.table_voxel_size,
            self.table_distance,
            self.maximum_object_height,
            self.maximum_translation,
            self.maximum_rotation_degrees,
            self.maximum_symmetric_rmse,
        )
        if not np.all(np.isfinite(metric)) or any(value <= 0 for value in metric):
            raise ValueError("dual-Femto fusion limits must be positive")
        if not 0.0 <= self.minimum_symmetric_overlap <= 1.0:
            raise ValueError("fusion overlap must be in [0, 1]")


@dataclass(frozen=True)
class FieldAssignmentProfile:
    target_coverage_weight: float
    source_coverage_weight: float
    rmse_weight: float
    extent_weight: float
    bottom_weight: float
    below_table_weight: float
    bottom_scale: float
    below_table_tolerance: float

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if not np.all(np.isfinite(values)) or any(value < 0 for value in values):
            raise ValueError("field assignment values must be non-negative")
        if self.bottom_scale <= 0.0 or self.below_table_tolerance <= 0.0:
            raise ValueError("field assignment metric scales must be positive")


@dataclass(frozen=True)
class CameraProfile:
    serial: str
    pointcloud_topic: str
    depth_mode: str

    def __post_init__(self) -> None:
        if not self.serial or not self.pointcloud_topic or not self.depth_mode:
            raise ValueError("camera serial, topic, and depth mode are required")


@dataclass(frozen=True)
class CaptureProfile:
    frame_count: int
    warmup_frames: int
    period: float
    timeout: float

    def __post_init__(self) -> None:
        if self.frame_count <= 0 or self.warmup_frames < 0:
            raise ValueError("capture frame counts are invalid")
        if not np.all(np.isfinite([self.period, self.timeout])):
            raise ValueError("capture timing values must be finite")
        if self.period < 0.0 or self.timeout <= 0.0:
            raise ValueError("capture timing values are invalid")


@dataclass(frozen=True)
class InHandCaptureProfile:
    crop_min: np.ndarray
    crop_max: np.ndarray
    minimum_temporal_support: int
    temporal_tolerance: float
    enable_boundary_filter: bool
    discontinuity_threshold: float
    discontinuity_radius_pixels: int
    boundary_dilation_pixels: int
    reference_voxel_size: float
    manual_remove_distance: float
    manual_max_remove_ratio: float

    def __post_init__(self) -> None:
        lower = np.asarray(self.crop_min, dtype=np.float64)
        upper = np.asarray(self.crop_max, dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
            raise ValueError("in-hand crop bounds are invalid")
        metric = (
            self.temporal_tolerance,
            self.discontinuity_threshold,
            self.reference_voxel_size,
            self.manual_remove_distance,
        )
        if not np.all(np.isfinite(metric)) or any(
            value <= 0.0 for value in metric
        ):
            raise ValueError("in-hand capture metric values must be positive")
        if self.minimum_temporal_support <= 0 or self.discontinuity_radius_pixels <= 0:
            raise ValueError("in-hand capture support and radius must be positive")
        if self.boundary_dilation_pixels < 0:
            raise ValueError("in-hand boundary dilation must be non-negative")
        if not 0.0 < self.manual_max_remove_ratio < 1.0:
            raise ValueError("manual removal ratio must lie in (0, 1)")
        lower.setflags(write=False)
        upper.setflags(write=False)
        object.__setattr__(self, "crop_min", lower)
        object.__setattr__(self, "crop_max", upper)


@dataclass(frozen=True)
class TabletopPerceptionProfile:
    field_registration: MultiscaleRegistrationConfig
    field_quality: RegistrationQualityConfig
    in_hand_registration: MultiscaleRegistrationConfig
    in_hand_quality: RegistrationQualityConfig
    segmentation: FieldSegmentationProfile
    assignment: FieldAssignmentProfile
    dual_femto_fusion: DualFemtoFusionProfile
    primary_camera: CameraProfile
    secondary_camera: CameraProfile
    capture: CaptureProfile
    in_hand_capture: InHandCaptureProfile
    cross_capture_merge: CrossCaptureMergeProfile
    reconstruction: ReconstructionProfile


def load_perception_profile(path: str | Path) -> TabletopPerceptionProfile:
    """Load independently resolved field and in-hand registration values."""

    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(f"missing perception profile: {profile_path}")
    values = yaml.safe_load(profile_path.read_text())
    if not isinstance(values, Mapping):
        raise TypeError("perception profile root must be a mapping")
    if values.get("units") != "si":
        raise ValueError("perception profile units must be 'si'")
    registration = _mapping(values, "registration")
    field = _mapping(registration, "field")
    in_hand = _mapping(registration, "in_hand")
    fusion = _mapping(values, "dual_femto_fusion")
    reconstruction = _mapping(values, "reconstruction")
    in_hand_capture = _mapping(values, "in_hand_capture")
    cross_capture = _mapping(values, "cross_capture_merge")
    pose_graph = _mapping(reconstruction, "pose_graph")
    dsf_fitting = _mapping(reconstruction, "dsf_fitting")
    cameras = _mapping(values, "cameras")
    return TabletopPerceptionProfile(
        field_registration=_registration(field),
        # Field pose seeds are coarse search hypotheses, not calibrated poses.
        field_quality=_quality(_mapping(field, "quality"), bound_correction=False),
        in_hand_registration=_registration(in_hand),
        in_hand_quality=_quality(_mapping(in_hand, "quality")),
        segmentation=_segmentation(_mapping(values, "segmentation")),
        assignment=_assignment(_mapping(values, "assignment_score")),
        dual_femto_fusion=DualFemtoFusionProfile(
            registration=_registration(_mapping(fusion, "registration")),
            merge_voxel_size=float(_required(fusion, "merge_voxel_size")),
            table_voxel_size=float(_required(fusion, "table_voxel_size")),
            table_distance=float(_required(fusion, "table_distance")),
            maximum_object_height=float(
                _required(fusion, "maximum_object_height")
            ),
            maximum_translation=float(
                _required(fusion, "maximum_translation")
            ),
            maximum_rotation_degrees=float(
                _required(fusion, "maximum_rotation_degrees")
            ),
            minimum_symmetric_overlap=float(
                _required(fusion, "minimum_symmetric_overlap")
            ),
            maximum_symmetric_rmse=float(
                _required(fusion, "maximum_symmetric_rmse")
            ),
        ),
        primary_camera=_camera(_mapping(cameras, "primary")),
        secondary_camera=_camera(_mapping(cameras, "secondary")),
        capture=_capture(_mapping(values, "capture")),
        in_hand_capture=InHandCaptureProfile(
            crop_min=np.asarray(_sequence(in_hand_capture, "crop_min")),
            crop_max=np.asarray(_sequence(in_hand_capture, "crop_max")),
            minimum_temporal_support=int(_required(in_hand_capture, "minimum_temporal_support")),
            temporal_tolerance=float(_required(in_hand_capture, "temporal_tolerance")),
            enable_boundary_filter=bool(_required(in_hand_capture, "enable_boundary_filter")),
            discontinuity_threshold=float(_required(in_hand_capture, "discontinuity_threshold")),
            discontinuity_radius_pixels=int(_required(in_hand_capture, "discontinuity_radius_pixels")),
            boundary_dilation_pixels=int(_required(in_hand_capture, "boundary_dilation_pixels")),
            reference_voxel_size=float(_required(in_hand_capture, "reference_voxel_size")),
            manual_remove_distance=float(_required(in_hand_capture, "manual_remove_distance")),
            manual_max_remove_ratio=float(_required(in_hand_capture, "manual_max_remove_ratio")),
        ),
        cross_capture_merge=CrossCaptureMergeProfile(
            registration=_registration(_mapping(cross_capture, "registration")),
            quality=_quality(_mapping(cross_capture, "quality")),
            preview_voxel_size=float(_required(cross_capture, "preview_voxel_size")),
            output_voxel_size=float(_required(cross_capture, "output_voxel_size")),
            initial_translation_step=float(_required(cross_capture, "initial_translation_step")),
            initial_rotation_step_degrees=float(_required(cross_capture, "initial_rotation_step_degrees")),
        ),
        reconstruction=ReconstructionProfile(
            merge_voxel_size=float(
                _required(reconstruction, "merge_voxel_size")
            ),
            minimum_view_support=int(
                _required(reconstruction, "minimum_view_support")
            ),
            pose_graph=PoseGraphProfile(
                registration=_registration(
                    _mapping(pose_graph, "registration")
                ),
                sequential_quality=_quality(
                    _mapping(pose_graph, "sequential_quality")
                ),
                loop_quality=_quality(
                    _mapping(pose_graph, "loop_quality")
                ),
                maximum_final_translation=float(
                    _required(pose_graph, "maximum_final_translation")
                ),
                maximum_final_rotation_degrees=float(
                    _required(
                        pose_graph,
                        "maximum_final_rotation_degrees",
                    )
                ),
                edge_prune_threshold=float(
                    _required(pose_graph, "edge_prune_threshold")
                ),
                loop_closure_preference=float(
                    _required(pose_graph, "loop_closure_preference")
                ),
                loop_minimum_capture_separation=int(
                    _required(pose_graph, "loop_minimum_capture_separation")
                ),
                loop_maximum_position_delta=float(
                    _required(pose_graph, "loop_maximum_position_delta")
                ),
                loop_maximum_view_angle_degrees=float(
                    _required(pose_graph, "loop_maximum_view_angle_degrees")
                ),
                maximum_loops_per_node=int(
                    _required(pose_graph, "maximum_loops_per_node")
                ),
            ),
            cleanup=_cleanup(_mapping(reconstruction, "cleanup")),
            mesh=_mesh(_mapping(reconstruction, "mesh")),
            convex_decomposition=_convex_decomposition(
                _mapping(reconstruction, "convex_decomposition")
            ),
            dsf_fitting=_dsf_fitting(dsf_fitting),
            dsf=_dsf(_mapping(reconstruction, "dsf")),
        ),
    )


def _quality(
    values: Mapping[str, object], *, bound_correction: bool = True,
) -> RegistrationQualityConfig:
    return RegistrationQualityConfig(
        minimum_target_coverage=float(
            _required(values, "minimum_target_coverage")
        ),
        minimum_source_coverage=float(
            _required(values, "minimum_source_coverage")
        ),
        maximum_rmse=float(_required(values, "maximum_rmse")),
        maximum_correction_translation=float(
            _required(values, "maximum_correction_translation")
        ) if bound_correction else None,
        maximum_correction_rotation_degrees=float(
            _required(values, "maximum_correction_rotation_degrees")
        ) if bound_correction else None,
    )


def _assignment(values: Mapping[str, object]) -> FieldAssignmentProfile:
    return FieldAssignmentProfile(
        target_coverage_weight=float(
            _required(values, "target_coverage_weight")
        ),
        source_coverage_weight=float(
            _required(values, "source_coverage_weight")
        ),
        rmse_weight=float(_required(values, "rmse_weight")),
        extent_weight=float(_required(values, "extent_weight")),
        bottom_weight=float(_required(values, "bottom_weight")),
        below_table_weight=float(_required(values, "below_table_weight")),
        bottom_scale=float(_required(values, "bottom_scale")),
        below_table_tolerance=float(
            _required(values, "below_table_tolerance")
        ),
    )


def _segmentation(values: Mapping[str, object]) -> FieldSegmentationProfile:
    return FieldSegmentationProfile(
        workspace_min=np.asarray(_sequence(values, "workspace_min")),
        workspace_max=np.asarray(_sequence(values, "workspace_max")),
        voxel_size=float(_required(values, "voxel_size")),
        table_distance=float(_required(values, "table_distance")),
        designated_table_height=float(
            _required(values, "designated_table_height")
        ),
        maximum_object_height=float(
            _required(values, "maximum_object_height")
        ),
        cluster_distance=float(_required(values, "cluster_distance")),
        minimum_cluster_points=int(
            _required(values, "minimum_cluster_points")
        ),
        random_seed=int(_required(values, "random_seed")),
        maximum_table_tilt_degrees=float(
            _required(values, "maximum_table_tilt_degrees")
        ),
        maximum_plane_candidates=int(
            _required(values, "maximum_plane_candidates")
        ),
        ransac_iterations=int(_required(values, "ransac_iterations")),
    )


def _camera(values: Mapping[str, object]) -> CameraProfile:
    return CameraProfile(
        serial=str(_required(values, "serial")),
        pointcloud_topic=str(_required(values, "pointcloud_topic")),
        depth_mode=str(_required(values, "depth_mode")),
    )


def _capture(values: Mapping[str, object]) -> CaptureProfile:
    return CaptureProfile(
        frame_count=int(_required(values, "frame_count")),
        warmup_frames=int(_required(values, "warmup_frames")),
        period=float(_required(values, "period")),
        timeout=float(_required(values, "timeout")),
    )


def _mesh(values: Mapping[str, object]) -> MeshReconstructionProfile:
    return MeshReconstructionProfile(
        method=str(_required(values, "method")),
        voxel_size=float(_required(values, "voxel_size")),
        alpha=float(_required(values, "alpha")),
        bpa_radius_scales=tuple(
            float(value) for value in _sequence(values, "bpa_radius_scales")
        ),
        radial_subdivisions=int(_required(values, "radial_subdivisions")),
        radial_neighbors=int(_required(values, "radial_neighbors")),
        radial_radius_quantile=float(_required(values, "radial_radius_quantile")),
        radial_radius_padding=float(_required(values, "radial_radius_padding")),
        radial_center_quantile=float(_required(values, "radial_center_quantile")),
        radial_empty_angle_degrees=float(
            _required(values, "radial_empty_angle_degrees")
        ),
        radial_empty_radius_quantile=float(
            _required(values, "radial_empty_radius_quantile")
        ),
        radial_support_angle_degrees=float(
            _required(values, "radial_support_angle_degrees")
        ),
        radial_minimum_support_points=int(
            _required(values, "radial_minimum_support_points")
        ),
        radial_tighten_angle_degrees=float(
            _required(values, "radial_tighten_angle_degrees")
        ),
        radial_tighten_quantile=float(_required(values, "radial_tighten_quantile")),
        radial_tighten_blend=float(_required(values, "radial_tighten_blend")),
        radial_maximum_radius_quantile=float(
            _required(values, "radial_maximum_radius_quantile")
        ),
        radial_smoothing_iterations=int(
            _required(values, "radial_smoothing_iterations")
        ),
        radial_smoothing_weight=float(_required(values, "radial_smoothing_weight")),
        normal_radius=float(_required(values, "normal_radius")),
        normal_max_neighbors=int(_required(values, "normal_max_neighbors")),
        poisson_depth=int(_required(values, "poisson_depth")),
        poisson_density_quantile=float(
            _required(values, "poisson_density_quantile")
        ),
        smoothing_iterations=int(_required(values, "smoothing_iterations")),
        simplification_voxel_size=float(
            _required(values, "simplification_voxel_size")
        ),
        keep_largest_component=bool(
            _required(values, "keep_largest_component")
        ),
        repair_invalid_mesh=bool(_required(values, "repair_invalid_mesh")),
    )


def _cleanup(values: Mapping[str, object]) -> ReconstructionCleanupProfile:
    return ReconstructionCleanupProfile(
        statistical_neighbors=int(_required(values, "statistical_neighbors")),
        statistical_std_ratio=float(_required(values, "statistical_std_ratio")),
        detached_voxel_size=float(_required(values, "detached_voxel_size")),
        detached_eps=float(_required(values, "detached_eps")),
        detached_minimum_points=int(_required(values, "detached_minimum_points")),
        detached_attach_distance=float(_required(values, "detached_attach_distance")),
        detached_minimum_attached_size=int(
            _required(values, "detached_minimum_attached_size")
        ),
        detached_keep_ratio=float(_required(values, "detached_keep_ratio")),
        fringe_neighbors=int(_required(values, "fringe_neighbors")),
        fringe_margin=float(_required(values, "fringe_margin")),
        fringe_reference_quantile=float(_required(values, "fringe_reference_quantile")),
        cluster_voxel_size=float(_required(values, "cluster_voxel_size")),
        cluster_eps=float(_required(values, "cluster_eps")),
        cluster_min_points=int(_required(values, "cluster_min_points")),
        manual_remove_distance=float(_required(values, "manual_remove_distance")),
        manual_max_remove_ratio=float(_required(values, "manual_max_remove_ratio")),
    )


def _convex_decomposition(
    values: Mapping[str, object],
) -> ConvexDecompositionProfile:
    return ConvexDecompositionProfile(
        enabled=bool(_required(values, "enabled")),
        threshold=float(_required(values, "threshold")),
        maximum_components=int(_required(values, "maximum_components")),
        preprocess_mode=str(_required(values, "preprocess_mode")),
        preprocess_resolution=int(_required(values, "preprocess_resolution")),
        resolution=int(_required(values, "resolution")),
        mcts_nodes=int(_required(values, "mcts_nodes")),
        mcts_iterations=int(_required(values, "mcts_iterations")),
        mcts_max_depth=int(_required(values, "mcts_max_depth")),
        pca=bool(_required(values, "pca")),
        merge=bool(_required(values, "merge")),
        decimate=bool(_required(values, "decimate")),
        maximum_hull_vertices=int(_required(values, "maximum_hull_vertices")),
        seed=int(_required(values, "seed")),
        real_metric=bool(_required(values, "real_metric")),
    )


def _dsf_fitting(values: Mapping[str, object]) -> DsfFittingProfile:
    ransac = _mapping(values, "ransac")
    raw_seed = _required(ransac, "seed")
    return DsfFittingProfile(
        enabled=bool(_required(values, "enabled")),
        fit_source=str(_required(values, "fit_source")),
        total_vertices=int(_required(values, "total_vertices")),
        direction_subdivisions=int(
            _required(values, "direction_subdivisions")
        ),
        maximum_iterations=int(_required(values, "maximum_iterations")),
        support_quantile=float(_required(values, "support_quantile")),
        loss=str(_required(values, "loss")),
        loss_scale=float(_required(values, "loss_scale")),
        center_interior_upper_bound=float(
            _required(values, "center_interior_upper_bound")
        ),
        maximum_sharpness=int(_required(values, "maximum_sharpness")),
        split_inside_tolerance=float(
            _required(values, "split_inside_tolerance")
        ),
        split_minimum_points=int(
            _required(values, "split_minimum_points")
        ),
        ransac=DsfRansacProfile(
            enabled=bool(_required(ransac, "enabled")),
            iterations=int(_required(ransac, "iterations")),
            sample_ratio=float(_required(ransac, "sample_ratio")),
            minimum_sample_points=int(
                _required(ransac, "minimum_sample_points")
            ),
            maximum_sample_points=int(
                _required(ransac, "maximum_sample_points")
            ),
            fit_maximum_iterations=int(
                _required(ransac, "fit_maximum_iterations")
            ),
            inlier_threshold=float(_required(ransac, "inlier_threshold")),
            minimum_inlier_ratio=float(
                _required(ransac, "minimum_inlier_ratio")
            ),
            score_direction_subdivisions=int(
                _required(ransac, "score_direction_subdivisions")
            ),
            seed=None if raw_seed is None else int(raw_seed),
        ),
    )


def _dsf(values: Mapping[str, object]) -> DsfExportProfile:
    return DsfExportProfile(
        maximum_nodes=int(_required(values, "maximum_nodes")),
        sharpness=int(_required(values, "sharpness")),
        friction=float(_required(values, "friction")),
    )


def _registration(
    values: Mapping[str, object],
) -> MultiscaleRegistrationConfig:
    voxel_sizes = _sequence(values, "voxel_sizes")
    max_iterations = _sequence(values, "max_iterations")
    return MultiscaleRegistrationConfig(
        voxel_sizes=tuple(float(value) for value in voxel_sizes),
        max_iterations=tuple(int(value) for value in max_iterations),
        correspondence_distance_scale=float(
            _required(values, "correspondence_distance_scale")
        ),
        evaluation_distance=float(
            _required(values, "evaluation_distance")
        ),
        minimum_points=int(_required(values, "minimum_points")),
    )


def _mapping(
    values: Mapping[str, object], key: str
) -> Mapping[str, object]:
    result = _required(values, key)
    if not isinstance(result, Mapping):
        raise TypeError(f"perception profile '{key}' must be a mapping")
    return result


def _sequence(
    values: Mapping[str, object], key: str
) -> Sequence[object]:
    result = _required(values, key)
    if not isinstance(result, Sequence) or isinstance(result, str):
        raise TypeError(f"perception profile '{key}' must be a sequence")
    return result


def _required(values: Mapping[str, object], key: str):
    if key not in values:
        raise ValueError(f"perception profile is missing '{key}'")
    return values[key]
