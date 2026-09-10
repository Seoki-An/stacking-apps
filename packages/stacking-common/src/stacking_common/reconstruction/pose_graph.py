"""Bounded pose-graph refinement for approximately aligned object views."""

from dataclasses import dataclass

import numpy as np

from stacking_common.pointcloud import (
    Open3dRegistrationBackend,
    PointCloudData,
    RegistrationDecision,
    RegistrationResult,
    evaluate_registration,
    validate_rigid_transform,
)

from .config import PoseGraphProfile


@dataclass(frozen=True)
class PoseGraphEdgeResult:
    source_index: int
    target_index: int
    loop_closure: bool
    registration: RegistrationResult
    decision: RegistrationDecision


@dataclass(frozen=True)
class PoseGraphRefinementResult:
    transforms: tuple[np.ndarray, ...]
    edges: tuple[PoseGraphEdgeResult, ...]
    optimized: bool
    reason: str | None

    def __post_init__(self) -> None:
        values = []
        for index, transform in enumerate(self.transforms):
            value = validate_rigid_transform(
                transform,
                f"pose-graph transform {index}",
            )
            value.setflags(write=False)
            values.append(value)
        object.__setattr__(self, "transforms", tuple(values))


class PoseGraphRefiner:
    """Refine neighboring views and explicitly requested loop closures."""

    def __init__(self, profile: PoseGraphProfile) -> None:
        self.profile = profile
        self.backend = Open3dRegistrationBackend(profile.registration)

    def refine(
        self,
        views: tuple[PointCloudData, ...],
        loop_edges: tuple[tuple[int, int], ...] = (),
        *,
        capture_indices: tuple[int, ...] | None = None,
        viewpoints: tuple[np.ndarray, ...] | None = None,
    ) -> PoseGraphRefinementResult:
        identities = tuple(np.eye(4) for _ in views)
        if len(views) < 2:
            return PoseGraphRefinementResult(
                identities,
                (),
                False,
                "too_few_views",
            )
        self._validate_loop_edges(len(views), loop_edges)
        edge_pairs = build_pose_graph_edge_pairs(
            len(views),
            capture_indices=capture_indices,
            viewpoints=viewpoints,
            explicit_loops=loop_edges,
            loop_minimum_capture_separation=(
                self.profile.loop_minimum_capture_separation
            ),
            loop_maximum_position_delta=(
                self.profile.loop_maximum_position_delta
            ),
            loop_maximum_view_angle_degrees=(
                self.profile.loop_maximum_view_angle_degrees
            ),
            maximum_loops_per_node=self.profile.maximum_loops_per_node,
        )

        edge_results = []
        for source, target, loop_closure in edge_pairs:
            edge_results.append(
                self._register_edge(views, source, target, loop_closure)
            )

        accepted_pairs = tuple(
            (edge.source_index, edge.target_index)
            for edge in edge_results
            if edge.decision.accepted
        )
        if not _connected(len(views), accepted_pairs):
            return PoseGraphRefinementResult(
                identities,
                tuple(edge_results),
                False,
                "disconnected_graph",
            )

        o3d = _open3d()
        graph = o3d.pipelines.registration.PoseGraph()
        for _ in views:
            graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))

        for edge in edge_results:
            if not edge.decision.accepted:
                continue
            source = _cloud(o3d, views[edge.source_index])
            target = _cloud(o3d, views[edge.target_index])
            information = (
                o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    source,
                    target,
                    self.profile.registration.evaluation_distance,
                    edge.registration.transform,
                )
            )
            quality = min(
                edge.registration.target_coverage,
                edge.registration.source_coverage,
            )
            if edge.loop_closure:
                quality *= self.profile.loop_closure_preference
            graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    edge.source_index,
                    edge.target_index,
                    edge.registration.transform,
                    information * quality,
                    uncertain=edge.loop_closure,
                )
            )

        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=(
                self.profile.registration.evaluation_distance
            ),
            edge_prune_threshold=self.profile.edge_prune_threshold,
            preference_loop_closure=self.profile.loop_closure_preference,
            reference_node=0,
        )
        o3d.pipelines.registration.global_optimization(
            graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )
        transforms = tuple(np.asarray(node.pose).copy() for node in graph.nodes)
        if any(not self._bounded(transform) for transform in transforms):
            return PoseGraphRefinementResult(
                identities,
                tuple(edge_results),
                False,
                "final_correction_exceeds_limit",
            )
        return PoseGraphRefinementResult(
            transforms,
            tuple(edge_results),
            True,
            None,
        )

    def _register_edge(
        self,
        views: tuple[PointCloudData, ...],
        source: int,
        target: int,
        loop_closure: bool,
    ) -> PoseGraphEdgeResult:
        result = self.backend.register(
            views[source],
            views[target],
            np.eye(4),
        )
        quality = (
            self.profile.loop_quality
            if loop_closure
            else self.profile.sequential_quality
        )
        return PoseGraphEdgeResult(
            source_index=source,
            target_index=target,
            loop_closure=loop_closure,
            registration=result,
            decision=evaluate_registration(result, np.eye(4), quality),
        )

    def _bounded(self, transform: np.ndarray) -> bool:
        translation = np.linalg.norm(transform[:3, 3])
        cosine = np.clip(
            (np.trace(transform[:3, :3]) - 1.0) * 0.5,
            -1.0,
            1.0,
        )
        rotation = np.rad2deg(np.arccos(cosine))
        return bool(
            translation <= self.profile.maximum_final_translation
            and rotation <= self.profile.maximum_final_rotation_degrees
        )

    @staticmethod
    def _validate_loop_edges(
        view_count: int,
        loop_edges: tuple[tuple[int, int], ...],
    ) -> None:
        for source, target in loop_edges:
            if (
                source < 0
                or target < 0
                or source >= view_count
                or target >= view_count
                or source == target
            ):
                raise ValueError(
                    f"invalid pose-graph loop edge ({source}, {target})"
                )


def _cloud(o3d, cloud: PointCloudData):
    value = o3d.geometry.PointCloud()
    value.points = o3d.utility.Vector3dVector(
        np.array(cloud.points, copy=True)
    )
    return value


def build_pose_graph_edge_pairs(
    view_count: int,
    *,
    capture_indices: tuple[int, ...] | None = None,
    viewpoints: tuple[np.ndarray, ...] | None = None,
    explicit_loops: tuple[tuple[int, int], ...] = (),
    loop_minimum_capture_separation: int = 5,
    loop_maximum_position_delta: float = 0.015,
    loop_maximum_view_angle_degrees: float = 7.5,
    maximum_loops_per_node: int = 2,
) -> tuple[tuple[int, int, bool], ...]:
    """Use true capture neighbors and conservative repeated-view loop edges."""

    indices = tuple(range(view_count)) if capture_indices is None else tuple(capture_indices)
    if len(indices) != view_count or len(set(indices)) != view_count:
        raise ValueError("pose-graph capture indices must be unique and match views")
    if any(second <= first for first, second in zip(indices, indices[1:])):
        raise ValueError("pose-graph capture indices must be strictly increasing")
    sequential = {
        (node + 1, node, False)
        for node in range(view_count - 1)
        if indices[node + 1] - indices[node] == 1
    }
    loops = {
        (max(source, target), min(source, target), True)
        for source, target in explicit_loops
    }
    if viewpoints is not None:
        if len(viewpoints) != view_count:
            raise ValueError("pose-graph viewpoints must match views")
        values = tuple(np.asarray(value, dtype=np.float64) for value in viewpoints)
        for value in values:
            if value.shape != (4, 4) or not np.all(np.isfinite(value)):
                raise ValueError("every pose-graph viewpoint must be a finite 4x4 matrix")
        centers = np.asarray([value[:3, 3] for value in values])
        directions = np.asarray([value[:3, 2] for value in values])
        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError("pose-graph viewpoint directions must be nonzero")
        directions /= norms[:, None]
        for source in range(view_count):
            candidates = []
            for target in range(view_count):
                if source == target or abs(indices[source] - indices[target]) < loop_minimum_capture_separation:
                    continue
                distance = float(np.linalg.norm(centers[source] - centers[target]))
                angle = float(np.rad2deg(np.arccos(np.clip(directions[source] @ directions[target], -1.0, 1.0))))
                if distance <= loop_maximum_position_delta and angle <= loop_maximum_view_angle_degrees:
                    candidates.append((distance / loop_maximum_position_delta + angle / loop_maximum_view_angle_degrees, target))
            for _, target in sorted(candidates)[:maximum_loops_per_node]:
                loops.add((max(source, target), min(source, target), True))
    sequential_nodes = {(source, target) for source, target, _ in sequential}
    loops = {
        edge for edge in loops if (edge[0], edge[1]) not in sequential_nodes
    }
    return tuple(sorted(sequential)) + tuple(sorted(loops))


def _connected(view_count: int, edges: tuple[tuple[int, int], ...]) -> bool:
    adjacency = [set() for _ in range(view_count)]
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    reached = {0}
    pending = [0]
    while pending:
        for neighbor in adjacency[pending.pop()]:
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    return len(reached) == view_count


def _open3d():
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError(
            "Open3D is required for pose-graph refinement"
        ) from error
    return o3d

