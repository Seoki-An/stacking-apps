"""Typed contracts for point-cloud registration backends."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .types import PointCloudData, validate_rigid_transform


@dataclass(frozen=True)
class MultiscaleRegistrationConfig:
    """Explicit metric resolutions for one multiscale registration task."""

    voxel_sizes: tuple[float, ...]
    max_iterations: tuple[int, ...]
    correspondence_distance_scale: float
    evaluation_distance: float
    minimum_points: int

    def __post_init__(self) -> None:
        voxel_sizes = tuple(float(value) for value in self.voxel_sizes)
        iterations = tuple(int(value) for value in self.max_iterations)
        if not voxel_sizes or len(voxel_sizes) != len(iterations):
            raise ValueError(
                "registration voxel sizes and iteration counts must align"
            )
        if not np.all(np.isfinite(voxel_sizes)) or any(
            value <= 0.0 for value in voxel_sizes
        ):
            raise ValueError("registration voxel sizes must be positive")
        if any(value <= 0 for value in iterations):
            raise ValueError("registration iteration counts must be positive")
        if any(
            voxel_sizes[index] <= voxel_sizes[index + 1]
            for index in range(len(voxel_sizes) - 1)
        ):
            raise ValueError(
                "registration voxel sizes must decrease from coarse to fine"
            )
        if (
            not np.isfinite(self.correspondence_distance_scale)
            or self.correspondence_distance_scale <= 0.0
        ):
            raise ValueError("correspondence distance scale must be positive")
        if (
            not np.isfinite(self.evaluation_distance)
            or self.evaluation_distance <= 0.0
        ):
            raise ValueError("registration evaluation distance must be positive")
        if not isinstance(self.minimum_points, int) or self.minimum_points < 3:
            raise ValueError("registration minimum point count must be at least 3")
        object.__setattr__(self, "voxel_sizes", voxel_sizes)
        object.__setattr__(self, "max_iterations", iterations)


@dataclass(frozen=True)
class RegistrationScaleResult:
    voxel_size: float
    fitness: float
    rmse: float
    correspondences: int


@dataclass(frozen=True)
class RegistrationResult:
    transform: np.ndarray
    target_coverage: float
    source_coverage: float
    rmse: float
    correspondences: int
    scales: tuple[RegistrationScaleResult, ...]

    def __post_init__(self) -> None:
        transform = validate_rigid_transform(
            self.transform,
            "registration transform",
        )
        transform.setflags(write=False)
        object.__setattr__(self, "transform", transform)
        for name, value in (
            ("target coverage", self.target_coverage),
            ("source coverage", self.source_coverage),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"registration {name} must be in [0, 1]")
        if self.rmse < 0.0 or np.isnan(self.rmse):
            raise ValueError("registration RMSE must be non-negative")
        if self.correspondences < 0:
            raise ValueError("registration correspondence count is invalid")


@dataclass(frozen=True)
class RegistrationQualityConfig:
    minimum_target_coverage: float
    minimum_source_coverage: float
    maximum_rmse: float
    maximum_correction_translation: float | None
    maximum_correction_rotation_degrees: float | None

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum target coverage", self.minimum_target_coverage),
            ("minimum source coverage", self.minimum_source_coverage),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        limits = (self.maximum_rmse,) + tuple(value for value in (
            self.maximum_correction_translation,
            self.maximum_correction_rotation_degrees,
        ) if value is not None)
        if not np.all(np.isfinite(limits)) or any(
            value <= 0.0 for value in limits
        ):
            raise ValueError("registration quality limits must be positive")


@dataclass(frozen=True)
class RegistrationDecision:
    accepted: bool
    reasons: tuple[str, ...]
    correction_translation: float
    correction_rotation_degrees: float


def evaluate_registration(
    result: RegistrationResult,
    initial_transform: np.ndarray,
    config: RegistrationQualityConfig,
) -> RegistrationDecision:
    """Apply workflow-specific quality gates to one registration result."""

    initial = validate_rigid_transform(
        initial_transform,
        "initial registration transform",
    )
    relative = np.linalg.inv(initial) @ result.transform
    translation = float(np.linalg.norm(relative[:3, 3]))
    cosine = np.clip(
        (np.trace(relative[:3, :3]) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    rotation = float(np.rad2deg(np.arccos(cosine)))
    reasons = []
    if result.target_coverage < config.minimum_target_coverage:
        reasons.append("low_target_coverage")
    if result.source_coverage < config.minimum_source_coverage:
        reasons.append("low_source_coverage")
    if not np.isfinite(result.rmse) or result.rmse > config.maximum_rmse:
        reasons.append("high_rmse")
    if (config.maximum_correction_translation is not None
            and translation > config.maximum_correction_translation):
        reasons.append("large_translation_correction")
    if (config.maximum_correction_rotation_degrees is not None
            and rotation > config.maximum_correction_rotation_degrees):
        reasons.append("large_rotation_correction")
    return RegistrationDecision(
        accepted=not reasons,
        reasons=tuple(reasons),
        correction_translation=translation,
        correction_rotation_degrees=rotation,
    )


class PointCloudRegistration(Protocol):
    """Backend that aligns a source model to an observed target cloud."""

    def register(
        self,
        source: PointCloudData,
        target: PointCloudData,
        initial_transform: np.ndarray,
    ) -> RegistrationResult: ...
