"""Typed configuration for platform-neutral placement generation."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PlacementActionGeneratorConfig:
    """Resolved physical limits used while constructing placement actions."""

    max_poses_per_object: int
    initial_height_clearance: float
    initial_height_xy_padding: float
    posegen_gap_threshold: float | None
    posegen_equilibrium_threshold: float | None = None
    banned_object_ids: frozenset[int] = field(default_factory=frozenset)
    require_solver_convergence: bool = field(default=True, metadata={"omit_default_from_context": True})
    reject_equilibrium_at_threshold: bool = field(default=False, metadata={"omit_default_from_context": True})
    stone_volume_exponent: float = field(default=0.0, metadata={"omit_default_from_context": True})
    reject_above_active_layer: bool = field(default=False, metadata={"omit_default_from_context": True})

    def __post_init__(self) -> None:
        for name in ("require_solver_convergence", "reject_equilibrium_at_threshold",
                     "reject_above_active_layer"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.max_poses_per_object <= 0:
            raise ValueError("max poses per object must be positive")
        if not np.isfinite(self.stone_volume_exponent) or self.stone_volume_exponent < 0:
            raise ValueError("stone volume exponent must be finite and non-negative")
        for name, value in (
            ("initial height clearance", self.initial_height_clearance),
            ("initial height XY padding", self.initial_height_xy_padding),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.posegen_gap_threshold is not None and (
            not np.isfinite(self.posegen_gap_threshold)
            or self.posegen_gap_threshold < 0.0
        ):
            raise ValueError(
                "pose-generator gap threshold must be finite and non-negative"
            )
        if self.posegen_equilibrium_threshold is not None and (
            not np.isfinite(self.posegen_equilibrium_threshold)
            or self.posegen_equilibrium_threshold < 0.0
        ):
            raise ValueError(
                "pose-generator equilibrium threshold must be finite and non-negative"
            )
        banned = frozenset(int(value) for value in self.banned_object_ids)
        if any(value < 0 for value in banned):
            raise ValueError("banned object IDs must be non-negative")
        object.__setattr__(self, "banned_object_ids", banned)


@dataclass(frozen=True)
class PlanarSamplingConfig:
    """Resolved grid and orientation settings for planar proposals."""

    grid_shape: tuple[int, int]
    yaw_angles: tuple[float, ...]
    target_anchor_inset_scale: float | None = None

    def __post_init__(self) -> None:
        if len(self.grid_shape) != 2 or any(
            int(value) <= 0 for value in self.grid_shape
        ):
            raise ValueError("planar grid shape must contain two positive values")
        yaws = tuple(float(value) for value in self.yaw_angles)
        if not yaws or not np.all(np.isfinite(yaws)):
            raise ValueError("planar yaw angles must be non-empty and finite")
        if self.target_anchor_inset_scale is not None and (
            not np.isfinite(self.target_anchor_inset_scale)
            or self.target_anchor_inset_scale < 0.0
        ):
            raise ValueError(
                "target anchor inset scale must be finite and non-negative"
            )
        object.__setattr__(
            self, "grid_shape", tuple(int(value) for value in self.grid_shape)
        )
        object.__setattr__(self, "yaw_angles", yaws)
        if self.target_anchor_inset_scale is not None:
            object.__setattr__(
                self,
                "target_anchor_inset_scale",
                float(self.target_anchor_inset_scale),
            )


@dataclass(frozen=True)
class CudaPosegenCapacityConfig:
    """CUDA workspace sizes; bodies and geometries are inventory-sized minima."""

    batch: int = 512
    bodies: int = 64
    geometries: int = 256
    pairs: int = 4096
    contacts: int = 512
    incidences: int = 1024

    def __post_init__(self):
        for name in ("batch", "bodies", "geometries", "pairs", "contacts", "incidences"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"posegen_capacity.{name} must be a positive integer")
        if self.batch > 65535:
            raise ValueError("posegen_capacity.batch must not exceed 65535")
