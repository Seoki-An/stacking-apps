"""Platform-neutral alignment and fusion of independently captured clouds."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .open3d_registration import Open3dRegistrationBackend
from .registration import (
    MultiscaleRegistrationConfig,
    RegistrationDecision,
    RegistrationQualityConfig,
    RegistrationResult,
    evaluate_registration,
)
from .processing import voxel_downsample
from .types import PointCloudData, validate_rigid_transform


@dataclass(frozen=True)
class CrossCaptureMergeProfile:
    registration: MultiscaleRegistrationConfig
    quality: RegistrationQualityConfig
    preview_voxel_size: float
    output_voxel_size: float
    initial_translation_step: float
    initial_rotation_step_degrees: float

    def __post_init__(self) -> None:
        metric = (
            self.preview_voxel_size,
            self.output_voxel_size,
            self.initial_translation_step,
            self.initial_rotation_step_degrees,
        )
        if not np.all(np.isfinite(metric)) or any(value <= 0.0 for value in metric):
            raise ValueError("cross-capture merge scales must be positive")


@dataclass(frozen=True)
class CrossCaptureAlignment:
    source_index: int
    transform: np.ndarray
    registration: RegistrationResult
    decision: RegistrationDecision

    def __post_init__(self) -> None:
        if self.source_index <= 0:
            raise ValueError("cross-capture source index must be positive")
        value = validate_rigid_transform(self.transform, "cross-capture transform")
        value.setflags(write=False)
        object.__setattr__(self, "transform", value)


@dataclass(frozen=True)
class CrossCaptureMergeResult:
    cloud: PointCloudData
    aligned_clouds: tuple[PointCloudData, ...]
    alignments: tuple[CrossCaptureAlignment, ...]


class CrossCaptureInitializer(Protocol):
    """Platform/operator adapter that supplies a source-to-target seed."""

    def initialize(
        self,
        source: PointCloudData,
        target: PointCloudData,
        source_index: int,
    ) -> np.ndarray: ...


class CrossCaptureMerger:
    """Refine supplied alignment seeds and merge scans into capture zero."""

    def __init__(self, profile: CrossCaptureMergeProfile) -> None:
        self.profile = profile
        self.backend = Open3dRegistrationBackend(
            profile.registration,
            generalized=True,
            robust_kernel_scale=1.5,
        )

    def merge(
        self,
        clouds: tuple[PointCloudData, ...],
        initial_transforms: tuple[np.ndarray, ...] | None = None,
        *,
        initializer: CrossCaptureInitializer | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> CrossCaptureMergeResult:
        if len(clouds) < 2:
            raise ValueError("cross-capture merge requires at least two clouds")
        if (initial_transforms is None) == (initializer is None):
            raise ValueError(
                "provide either initial transforms or one initializer"
            )
        if (
            initial_transforms is not None
            and len(initial_transforms) != len(clouds) - 1
        ):
            raise ValueError("one initial transform is required per moving capture")
        aligned = [clouds[0]]
        reference = voxel_downsample(
            clouds[0], self.profile.preview_voxel_size
        )
        alignments = []
        total = len(clouds) - 1
        for source_index, source in enumerate(clouds[1:], start=1):
            initial = (
                initializer.initialize(source, reference, source_index)
                if initializer is not None
                else initial_transforms[source_index - 1]
            )
            if progress is not None:
                progress(
                    source_index - 1,
                    total,
                    f"Running ICP for capture {source_index + 1}",
                )
            registration = self.backend.register(source, reference, initial)
            decision = evaluate_registration(
                registration,
                initial,
                self.profile.quality,
            )
            if not decision.accepted:
                raise ValueError(
                    f"capture {source_index + 1} ICP rejected: "
                    + ", ".join(decision.reasons)
                )
            aligned_source = source.transformed(registration.transform)
            aligned.append(aligned_source)
            reference = voxel_downsample(
                PointCloudData(
                    np.vstack((reference.points, aligned_source.points))
                ),
                self.profile.preview_voxel_size,
            )
            alignments.append(
                CrossCaptureAlignment(
                    source_index,
                    registration.transform,
                    registration,
                    decision,
                )
            )
            if progress is not None:
                progress(
                    source_index,
                    total,
                    f"ICP refined capture {source_index + 1}",
                )
        merged = voxel_downsample(
            PointCloudData(np.vstack([cloud.points for cloud in aligned])),
            self.profile.output_voxel_size,
        )
        return CrossCaptureMergeResult(
            merged,
            tuple(aligned),
            tuple(alignments),
        )
