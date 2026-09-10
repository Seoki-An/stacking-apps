"""Platform-neutral values and voxel consensus for object reconstruction."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .types import PointCloudData


@dataclass(frozen=True)
class VoxelMergeConfig:
    voxel_size: float
    minimum_view_support: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.voxel_size) or self.voxel_size <= 0.0:
            raise ValueError("merge voxel size must be positive")
        if self.minimum_view_support <= 0:
            raise ValueError("minimum view support must be positive")


@dataclass(frozen=True)
class VoxelMergeStats:
    view_count: int
    input_point_count: int
    occupied_voxel_count: int
    supported_voxel_count: int
    rejected_voxel_count: int


@dataclass(frozen=True)
class VoxelMergeResult:
    cloud: PointCloudData
    stats: VoxelMergeStats


class MultiViewVoxelMerge:
    """Merge view centroids only where enough distinct views agree."""

    def __init__(self, config: VoxelMergeConfig) -> None:
        self.config = config

    def merge(
        self,
        views: tuple[PointCloudData, ...],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> VoxelMergeResult:
        if not views:
            raise ValueError("at least one reconstruction view is required")
        if self.config.minimum_view_support > len(views):
            raise ValueError("minimum view support exceeds the view count")

        keys_by_view = []
        centroids_by_view = []
        input_count = 0
        total = len(views)
        for index, view in enumerate(views, start=1):
            input_count += len(view)
            keys, centroids = self._view_centroids(view.points)
            keys_by_view.append(keys)
            centroids_by_view.append(centroids)
            if progress is not None:
                progress(index, total)

        keys = np.concatenate(keys_by_view, axis=0)
        centroids = np.concatenate(centroids_by_view, axis=0)
        unique_keys, inverse, support = np.unique(
            keys,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        sums = np.zeros((len(unique_keys), 3), dtype=np.float64)
        np.add.at(sums, inverse, centroids)
        merged = sums / support[:, None]
        accepted = support >= self.config.minimum_view_support
        return VoxelMergeResult(
            cloud=PointCloudData(merged[accepted]),
            stats=VoxelMergeStats(
                view_count=len(views),
                input_point_count=input_count,
                occupied_voxel_count=len(unique_keys),
                supported_voxel_count=int(np.count_nonzero(accepted)),
                rejected_voxel_count=int(np.count_nonzero(~accepted)),
            ),
        )

    def _view_centroids(
        self,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not len(points):
            return (
                np.empty((0, 3), dtype=np.int64),
                np.empty((0, 3), dtype=np.float64),
            )
        keys = np.floor(points / self.config.voxel_size + 0.5).astype(
            np.int64
        )
        unique_keys, inverse, counts = np.unique(
            keys,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        sums = np.zeros((len(unique_keys), 3), dtype=np.float64)
        np.add.at(sums, inverse, points)
        return unique_keys, sums / counts[:, None]
