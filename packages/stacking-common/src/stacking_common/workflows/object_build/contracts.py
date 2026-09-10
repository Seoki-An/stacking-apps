"""Typed operator interactions emitted by object-build workflows."""

from dataclasses import dataclass

import numpy as np

from stacking_common.pointcloud import PointCloudData, validate_rigid_transform


@dataclass(frozen=True)
class CropSelection:
    minimum: np.ndarray
    maximum: np.ndarray
    removal_centers: np.ndarray


@dataclass(frozen=True)
class AlignmentSeed:
    source: PointCloudData
    target: PointCloudData
    transform: np.ndarray

    def __post_init__(self) -> None:
        value = validate_rigid_transform(self.transform, "alignment seed")
        value.setflags(write=False)
        object.__setattr__(self, "transform", value)


@dataclass(frozen=True)
class PartitionSelection:
    seeds: np.ndarray | None = None
    maximum_components: int | None = None


@dataclass(frozen=True)
class ReviewDecision:
    accepted: bool
    retry_stage: str | None = None
    note: str | None = None
