"""Sensor-independent point-cloud values and registration contracts."""

from .acquisition import CapturedPointCloud, PointCloudSource
from .incremental import IncrementalAlignment, IncrementalRegistration
from .cross_capture import (
    CrossCaptureAlignment,
    CrossCaptureInitializer,
    CrossCaptureMerger,
    CrossCaptureMergeProfile,
    CrossCaptureMergeResult,
)
from .extraction import ObjectViewExtractor
from .registration import (
    MultiscaleRegistrationConfig,
    PointCloudRegistration,
    RegistrationDecision,
    RegistrationQualityConfig,
    RegistrationResult,
    RegistrationScaleResult,
    evaluate_registration,
)
from .open3d_registration import Open3dRegistrationBackend
from .processing import voxel_downsample
from .reconstruction import (
    MultiViewVoxelMerge,
    VoxelMergeConfig,
    VoxelMergeResult,
    VoxelMergeStats,
)
from .segmentation import (
    SupportPlaneFit,
    SupportSurfaceProfile,
    SupportSurfaceSegmentation,
    SupportSurfaceSegmenter,
)
from .types import PointCloudData, validate_rigid_transform

__all__ = [
    "CapturedPointCloud",
    "IncrementalAlignment",
    "IncrementalRegistration",
    "CrossCaptureAlignment",
    "CrossCaptureInitializer",
    "CrossCaptureMerger",
    "CrossCaptureMergeProfile",
    "CrossCaptureMergeResult",
    "MultiscaleRegistrationConfig",
    "MultiViewVoxelMerge",
    "ObjectViewExtractor",
    "Open3dRegistrationBackend",
    "PointCloudData",
    "PointCloudSource",
    "PointCloudRegistration",
    "RegistrationDecision",
    "RegistrationQualityConfig",
    "RegistrationResult",
    "RegistrationScaleResult",
    "SupportPlaneFit",
    "SupportSurfaceProfile",
    "SupportSurfaceSegmentation",
    "SupportSurfaceSegmenter",
    "VoxelMergeConfig",
    "VoxelMergeResult",
    "VoxelMergeStats",
    "evaluate_registration",
    "validate_rigid_transform",
    "voxel_downsample",
]
