"""Tabletop names for the shared support-surface segmentation algorithm."""

from stacking_common.pointcloud import (
    SupportPlaneFit,
    SupportSurfaceProfile,
    SupportSurfaceSegmentation,
    SupportSurfaceSegmenter,
)
from stacking_tabletop.config import FieldSegmentationProfile

TablePlaneFit = SupportPlaneFit
SegmentedField = SupportSurfaceSegmentation
class TabletopFieldSegmenter(SupportSurfaceSegmenter):
    def __init__(self, profile: FieldSegmentationProfile) -> None:
        super().__init__(
            SupportSurfaceProfile(
                workspace_min=profile.workspace_min,
                workspace_max=profile.workspace_max,
                voxel_size=profile.voxel_size,
                plane_distance=profile.table_distance,
                ransac_iterations=profile.ransac_iterations,
                maximum_plane_candidates=profile.maximum_plane_candidates,
                maximum_tilt_degrees=profile.maximum_table_tilt_degrees,
                designated_height=profile.designated_table_height,
                maximum_object_height=profile.maximum_object_height,
                minimum_cluster_points=profile.minimum_cluster_points,
                cluster_distance=profile.cluster_distance,
                random_seed=profile.random_seed,
            )
        )

__all__ = ["SegmentedField", "TablePlaneFit", "TabletopFieldSegmenter"]
