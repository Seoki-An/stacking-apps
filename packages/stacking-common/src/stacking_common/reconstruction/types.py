"""Canonical values exchanged by object-reconstruction stages."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stacking_common.pointcloud import PointCloudData


def _readonly(value: np.ndarray, dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TriangleMeshData:
    vertices: np.ndarray
    triangles: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        triangles = np.asarray(self.triangles, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
            raise ValueError("mesh vertices must have shape (N, 3)")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("mesh vertices must be finite")
        if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
            raise ValueError("mesh triangles must have shape (M, 3)")
        if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
            raise ValueError("mesh triangle index is out of range")
        object.__setattr__(self, "vertices", _readonly(vertices, np.float64))
        object.__setattr__(self, "triangles", _readonly(triangles, np.int64))


@dataclass(frozen=True)
class ConvexPart:
    vertices: np.ndarray
    triangles: np.ndarray

    def __post_init__(self) -> None:
        mesh = TriangleMeshData(self.vertices, self.triangles)
        if len(mesh.vertices) < 4:
            raise ValueError("convex part needs at least four vertices")
        object.__setattr__(self, "vertices", mesh.vertices)
        object.__setattr__(self, "triangles", mesh.triangles)


@dataclass(frozen=True)
class DsfFitResult:
    """Continuous optimizer result before runtime commissioning."""

    vertices: np.ndarray
    sharpness: float
    cost: float
    evaluations: int
    parameters: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        parameters = np.asarray(self.parameters, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
            raise ValueError("fitted DSF needs at least four XYZ vertices")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("fitted DSF vertices must be finite")
        if not np.isfinite(self.sharpness) or self.sharpness < 2.0:
            raise ValueError("fitted DSF sharpness must be at least two")
        if not np.isfinite(self.cost) or self.cost < 0.0:
            raise ValueError("fitted DSF cost must be finite and non-negative")
        if self.evaluations < 1:
            raise ValueError("fitted DSF evaluation count must be positive")
        if (
            parameters.ndim != 1
            or len(parameters) != 4 * len(vertices) + 1
            or not np.all(np.isfinite(parameters))
        ):
            raise ValueError("fitted DSF parameters must be a finite vector")
        object.__setattr__(self, "vertices", _readonly(vertices, np.float64))
        object.__setattr__(self, "parameters", _readonly(parameters, np.float64))


@dataclass(frozen=True)
class DsfComponent:
    """Commissioned runtime DSF; triangles are debug/export topology only."""

    nodes: np.ndarray
    triangles: np.ndarray
    sharpness: int
    friction: float

    def __post_init__(self) -> None:
        mesh = TriangleMeshData(self.nodes, self.triangles)
        if self.sharpness < 2:
            raise ValueError("DSF component sharpness must be at least two")
        if not np.isfinite(self.friction) or self.friction < 0.0:
            raise ValueError("DSF component friction must be non-negative")
        object.__setattr__(self, "nodes", mesh.vertices)
        object.__setattr__(self, "triangles", mesh.triangles)


@dataclass(frozen=True)
class InertialMetadata:
    """Unit-density inertial properties about the reported center of mass."""

    mass: float
    inertia: np.ndarray
    center_of_mass: np.ndarray

    def __post_init__(self) -> None:
        inertia = np.asarray(self.inertia, dtype=np.float64)
        center = np.asarray(self.center_of_mass, dtype=np.float64)
        if not np.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError("unit-density mass must be positive")
        if inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)):
            raise ValueError("inertia must be a finite 3x3 matrix")
        if np.min(np.linalg.eigvalsh(inertia)) <= 0.0:
            raise ValueError("inertia must be positive definite")
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("center of mass must be finite XYZ")
        object.__setattr__(self, "inertia", _readonly(inertia, np.float64))
        object.__setattr__(self, "center_of_mass", _readonly(center, np.float64))


@dataclass(frozen=True)
class MeshValidationResult:
    """Validity of the surface used for volume integration."""

    watertight: bool
    self_intersecting: bool
    repair_attempted: bool = False
    repaired: bool = False

    @property
    def valid(self) -> bool:
        return self.watertight and not self.self_intersecting


@dataclass(frozen=True)
class ReconstructionOperatorOptions:
    remove_statistical_outliers: bool = True
    remove_detached_clusters: bool = True
    remove_radial_fringe: bool = True
    keep_largest_cluster: bool = True
    manual_remove: bool = False
    manual_left_click: bool = False
    show_input: bool = False
    show_cleaned: bool = False
    show_mesh: bool = False
    show_result: bool = False
    visualization_opacity: float = 0.35

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.visualization_opacity)
            or not 0.0 < self.visualization_opacity <= 1.0
        ):
            raise ValueError("visualization opacity must lie in (0, 1]")


@dataclass(frozen=True)
class DsfPartitionOptions:
    manual_split: bool = False
    seed_path: Path | None = None
    force_seed_selection: bool = False
    left_clicking: bool = False
    marker_radius: float = 0.003
    show_convex: bool = False
    show_partitions: bool = False
    prompt_maximum_components: bool = False
    review_result: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.marker_radius) or self.marker_radius <= 0.0:
            raise ValueError("manual split marker radius must be positive")
        if self.seed_path is not None:
            object.__setattr__(
                self,
                "seed_path",
                Path(self.seed_path).expanduser().resolve(),
            )


@dataclass(frozen=True)
class ReconstructionResult:
    source: PointCloudData
    cleaned: PointCloudData
    centered: PointCloudData
    mesh: TriangleMeshData
    convex_parts: tuple[ConvexPart, ...]
    point_partitions: tuple[np.ndarray, ...]
    fit_results: tuple[DsfFitResult, ...]
    components: tuple[DsfComponent, ...]
    inertial: InertialMetadata
    inertial_method: str
    mesh_validation: MeshValidationResult
    split_method: str
    convex_maximum_components: int

    def __post_init__(self) -> None:
        if self.inertial_method not in {"watertight_mesh", "convex_hull"}:
            raise ValueError("unsupported reconstruction inertial method")
