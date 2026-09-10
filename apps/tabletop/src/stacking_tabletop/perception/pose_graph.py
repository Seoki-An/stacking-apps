"""Tabletop compatibility names for shared pose-graph refinement."""

from stacking_common.reconstruction import (
    PoseGraphEdgeResult,
    PoseGraphRefinementResult,
    PoseGraphRefiner,
    build_pose_graph_edge_pairs,
)


TabletopPoseGraphRefiner = PoseGraphRefiner

__all__ = [
    "PoseGraphEdgeResult",
    "PoseGraphRefinementResult",
    "TabletopPoseGraphRefiner",
    "build_pose_graph_edge_pairs",
]
