"""CoACD decomposition and DSF point-partition algorithms."""

import inspect

import numpy as np

from .config import ConvexDecompositionProfile
from .types import ConvexPart, TriangleMeshData


class ConvexDecomposer:
    """Decompose a reconstructed surface with optional CoACD."""

    def __init__(self, profile: ConvexDecompositionProfile) -> None:
        self.profile = profile

    def decompose(self, mesh: TriangleMeshData) -> tuple[ConvexPart, ...]:
        if not self.profile.enabled or self.profile.maximum_components == 1:
            return (ConvexPart(mesh.vertices, mesh.triangles),)
        try:
            import coacd
        except ImportError as error:
            raise RuntimeError(
                "CoACD is required for convex decomposition; install the "
                "common perception extras"
            ) from error
        value = coacd.Mesh(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.triangles, dtype=np.int32),
        )
        arguments = {
            "threshold": self.profile.threshold,
            "max_convex_hull": self.profile.maximum_components,
            "preprocess_mode": self.profile.preprocess_mode,
            "preprocess_resolution": self.profile.preprocess_resolution,
            "resolution": self.profile.resolution,
            "mcts_nodes": self.profile.mcts_nodes,
            "mcts_iterations": self.profile.mcts_iterations,
            "mcts_max_depth": self.profile.mcts_max_depth,
            "pca": self.profile.pca,
            "merge": self.profile.merge,
            "decimate": self.profile.decimate,
            "max_ch_vertex": self.profile.maximum_hull_vertices,
            "seed": self.profile.seed,
            "real_metric": self.profile.real_metric,
        }
        try:
            accepted = set(inspect.signature(coacd.run_coacd).parameters)
            arguments = {
                name: argument
                for name, argument in arguments.items()
                if name in accepted
            }
        except (TypeError, ValueError):
            pass
        result = coacd.run_coacd(value, **arguments)
        parts = tuple(ConvexPart(vertices, triangles) for vertices, triangles in result)
        if not parts:
            raise ValueError("CoACD produced no convex components")
        return parts


def build_dsf_point_sets(
    points: np.ndarray,
    parts: tuple[ConvexPart, ...],
    *,
    fit_source: str,
    inside_tolerance: float,
    minimum_points: int,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...] | None]:
    vertices = tuple(np.asarray(part.vertices) for part in parts)
    if fit_source == "convex_vertices":
        return vertices, None
    split = split_points_by_convex_parts(
        points,
        parts,
        inside_tolerance=inside_tolerance,
        minimum_points=minimum_points,
    )
    if fit_source == "split_cloud":
        return split, split
    if fit_source == "both":
        return (
            tuple(
                np.vstack((part_points, part_vertices))
                for part_points, part_vertices in zip(split, vertices)
            ),
            split,
        )
    raise ValueError(f"unsupported DSF fit source: {fit_source}")


def split_points_by_convex_parts(
    points: np.ndarray,
    parts: tuple[ConvexPart, ...],
    *,
    inside_tolerance: float,
    minimum_points: int,
) -> tuple[np.ndarray, ...]:
    try:
        from scipy.spatial import ConvexHull
    except ImportError as error:
        raise RuntimeError("SciPy is required for convex partitioning") from error
    values = np.asarray(points, dtype=np.float64)
    if not len(parts) or not len(values):
        raise ValueError("point partitioning requires points and convex parts")
    halfspaces = []
    centers = []
    for part in parts:
        hull = ConvexHull(part.vertices)
        halfspaces.append((hull.equations[:, :3], hull.equations[:, 3]))
        centers.append(np.mean(part.vertices, axis=0))
    signed = np.empty((len(values), len(parts)), dtype=np.float64)
    for index, (normal, offset) in enumerate(halfspaces):
        signed[:, index] = np.max(
            values @ normal.T + offset[None, :],
            axis=1,
        )
    inside = signed <= inside_tolerance
    assignment = np.argmin(signed, axis=1)
    any_inside = np.any(inside, axis=1)
    if np.any(any_inside):
        masked = np.where(inside, signed, np.inf)
        assignment[any_inside] = np.argmin(masked[any_inside], axis=1)
    if np.any(~any_inside):
        distance = np.linalg.norm(
            values[~any_inside, None, :] - np.asarray(centers)[None, :, :],
            axis=2,
        )
        assignment[~any_inside] = np.argmin(distance, axis=1)
    return _ensure_minimum_partition_points(
        values,
        assignment,
        np.asarray(centers),
        minimum_points,
    )


def split_points_by_seeds(
    points: np.ndarray,
    seeds: np.ndarray,
    *,
    minimum_points: int,
) -> tuple[np.ndarray, ...]:
    values = np.asarray(points, dtype=np.float64)
    centers = np.asarray(seeds, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or not len(centers):
        raise ValueError("manual split seeds must have shape (N, 3), N > 0")
    distance = np.linalg.norm(
        values[:, None, :] - centers[None, :, :],
        axis=2,
    )
    return _ensure_minimum_partition_points(
        values,
        np.argmin(distance, axis=1),
        centers,
        minimum_points,
    )


def _ensure_minimum_partition_points(
    points: np.ndarray,
    assignment: np.ndarray,
    centers: np.ndarray,
    minimum_points: int,
) -> tuple[np.ndarray, ...]:
    if minimum_points < 4:
        raise ValueError("DSF partitions require at least four points")
    results = []
    for index, center in enumerate(centers):
        selected = points[assignment == index]
        if len(selected) < minimum_points:
            count = min(max(minimum_points, 4), len(points))
            distance = np.linalg.norm(points - center, axis=1)
            selected = points[np.argsort(distance)[:count]]
        results.append(selected)
    return tuple(results)
