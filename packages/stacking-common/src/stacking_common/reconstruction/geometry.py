"""Mass properties, DSF support evaluation, and runtime DSF export."""

from pathlib import Path

import numpy as np

from .config import DsfExportProfile
from .types import (
    DsfComponent,
    InertialMetadata,
    MeshValidationResult,
    TriangleMeshData,
)


class DsfExporter:
    """Commission fitted/support vertices and write the runtime DSF OBJ codec."""

    def __init__(self, profile: DsfExportProfile) -> None:
        self.profile = profile

    def component(self, mesh: TriangleMeshData) -> DsfComponent:
        return self.component_from_points(
            mesh.vertices,
            sharpness=self.profile.sharpness,
        )

    def component_from_points(
        self,
        points: np.ndarray,
        *,
        sharpness: int,
    ) -> DsfComponent:
        return self.component_from_vertices(
            farthest_points(points, self.profile.maximum_nodes),
            sharpness=sharpness,
        )

    def component_from_vertices(
        self,
        vertices: np.ndarray,
        *,
        sharpness: int,
    ) -> DsfComponent:
        nodes = np.array(vertices, dtype=np.float64, copy=True)
        if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) < 4:
            raise ValueError("DSF component needs at least four fitted vertices")
        return DsfComponent(
            nodes=nodes,
            triangles=convex_faces(nodes),
            sharpness=sharpness,
            friction=self.profile.friction,
        )

    def write(
        self,
        path: str | Path,
        components: tuple[DsfComponent, ...],
        inertial: InertialMetadata,
    ) -> Path:
        if not components:
            raise ValueError("at least one DSF component is required")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        inertia_rows = [
            ", ".join(f"{value:.17g}" for value in row)
            for row in inertial.inertia
        ]
        lines = [
            f"# mass: {inertial.mass:.17g}",
            f"# inertia: [[{inertia_rows[0]}],",
            f"#   [{inertia_rows[1]}],",
            f"#   [{inertia_rows[2]}]]",
            f"# dsf_count: {len(components)}",
            "",
        ]
        vertex_offset = 1
        for index, component in enumerate(components):
            lines.extend(
                (
                    f"o convex.{index}",
                    f"# sharpness: {component.sharpness}",
                    f"# mu: {component.friction:.17g}",
                )
            )
            lines.extend(
                f"v {x:.17g} {y:.17g} {z:.17g}"
                for x, y, z in component.nodes
            )
            lines.extend(
                "f "
                + " ".join(str(vertex_offset + int(value)) for value in face)
                for face in component.triangles
            )
            lines.append("")
            vertex_offset += len(component.nodes)
        destination.write_text("\n".join(lines), encoding="utf-8")
        return destination


def load_dsf_components(path: str | Path) -> tuple[DsfComponent, ...]:
    """Read commissioned DSF components from the runtime OBJ codec."""
    vertices: list[list[float]] = []
    components = []
    current = None

    def finish(end: int) -> None:
        nonlocal current
        if current is None:
            return
        start = current["start"]
        sharpness = current["sharpness"]
        friction = current["friction"]
        faces = current["faces"]
        if sharpness is None or friction is None or not faces:
            raise ValueError("DSF OBJ component metadata is incomplete")
        components.append(
            DsfComponent(
                np.asarray(vertices[start:end], dtype=np.float64),
                np.asarray(faces, dtype=np.int64) - start,
                int(round(sharpness)),
                float(friction),
            )
        )
        current = None

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("o "):
            finish(len(vertices))
            current = {
                "start": len(vertices),
                "sharpness": None,
                "friction": None,
                "faces": [],
            }
        elif line.startswith("v "):
            values = line.split()
            if len(values) != 4:
                raise ValueError("DSF OBJ vertices must contain XYZ values")
            vertices.append([float(value) for value in values[1:]])
        elif line.startswith("f "):
            if current is None:
                raise ValueError("DSF OBJ face appears before an object")
            indices = [int(value.split("/", 1)[0]) - 1 for value in line.split()[1:]]
            if len(indices) != 3:
                raise ValueError("DSF OBJ faces must be triangles")
            current["faces"].append(indices)
        elif current is not None and line.startswith("# sharpness:"):
            current["sharpness"] = float(line.partition(":")[2])
        elif current is not None and line.startswith("# mu:"):
            current["friction"] = float(line.partition(":")[2])
    finish(len(vertices))
    if not components:
        raise ValueError("DSF OBJ contains no components")
    return tuple(components)


def estimate_hull_inertial(points: np.ndarray) -> InertialMetadata:
    """Estimate legacy-compatible unit-density convex-hull mass properties."""

    try:
        from scipy.spatial import ConvexHull
    except ImportError as error:
        raise RuntimeError(
            "SciPy is required for reconstruction inertial metadata"
        ) from error
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 4:
        raise ValueError("at least four XYZ points are required for inertia")
    hull = ConvexHull(values)
    reference = values.mean(axis=0)
    tetrahedra = []
    total_volume = 0.0
    first_moment = np.zeros(3)
    for simplex in hull.simplices:
        tetrahedron = np.vstack((reference, values[simplex]))
        volume = abs(np.linalg.det(tetrahedron[1:] - tetrahedron[0])) / 6.0
        if volume <= 0.0:
            continue
        tetrahedra.append((tetrahedron, volume))
        total_volume += volume
        first_moment += volume * tetrahedron.mean(axis=0)
    if total_volume <= 0.0:
        raise ValueError("point-cloud convex hull has zero volume")
    center = first_moment / total_volume
    second_moment = np.zeros((3, 3))
    for tetrahedron, volume in tetrahedra:
        vertices = tetrahedron - center
        vertex_sum = vertices.sum(axis=0)
        vertex_outer_sum = np.einsum("ni,nj->ij", vertices, vertices)
        second_moment += volume / 20.0 * (
            np.outer(vertex_sum, vertex_sum) + vertex_outer_sum
        )
    inertia = np.trace(second_moment) * np.eye(3) - second_moment
    inertia = 0.5 * (inertia + inertia.T)
    return InertialMetadata(total_volume, inertia, center)


def is_watertight_mesh(mesh: TriangleMeshData) -> bool:
    """Return whether every non-degenerate edge has opposite face incidence."""

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    first = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    second = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    extent = max(float(np.ptp(vertices, axis=0).max()), 1.0)
    if np.any(np.linalg.norm(np.cross(first, second), axis=1) <= 1e-14 * extent**2):
        return False
    incidence: dict[tuple[int, int], list[int]] = {}
    for face in triangles:
        for source, target in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            if source == target:
                return False
            edge = tuple(sorted((source, target)))
            incidence.setdefault(edge, []).append(1 if source < target else -1)
    return all(len(values) == 2 and sum(values) == 0 for values in incidence.values())


def validate_triangle_mesh(
    mesh: TriangleMeshData,
    *,
    repair_attempted: bool = False,
    repaired: bool = False,
) -> MeshValidationResult:
    """Check volume topology and geometric self intersections."""

    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D is required for mesh validation") from error
    value = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.array(mesh.vertices, copy=True)),
        o3d.utility.Vector3iVector(np.array(mesh.triangles, copy=True)),
    )
    return MeshValidationResult(
        watertight=is_watertight_mesh(mesh),
        self_intersecting=bool(value.is_self_intersecting()),
        repair_attempted=repair_attempted,
        repaired=repaired,
    )


def estimate_mesh_inertial(mesh: TriangleMeshData) -> InertialMetadata:
    """Integrate unit-density properties of a validated watertight triangle mesh."""

    validation = validate_triangle_mesh(mesh)
    if not validation.valid:
        reason = (
            "self-intersecting"
            if validation.self_intersecting
            else "not consistently oriented and watertight"
        )
        raise ValueError(f"triangle mesh is {reason}")
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    total_volume = 0.0
    first_moment = np.zeros(3)
    second_moment = np.zeros((3, 3))
    for face_indices in _face_components(triangles):
        component = triangles[face_indices]
        first = vertices[component[:, 0]]
        second = vertices[component[:, 1]]
        third = vertices[component[:, 2]]
        signed_volumes = np.einsum(
            "ij,ij->i",
            first,
            np.cross(second, third),
        ) / 6.0
        signed_volume = float(np.sum(signed_volumes))
        extent = max(float(np.ptp(vertices, axis=0).max()), 1.0)
        if abs(signed_volume) <= 1e-14 * extent**3:
            raise ValueError("watertight mesh component has zero volume")
        volumes = signed_volumes * (1.0 if signed_volume > 0.0 else -1.0)
        points = np.stack((first, second, third), axis=1)
        sums = np.sum(points, axis=1)
        component_volume = float(np.sum(volumes))
        total_volume += component_volume
        first_moment += np.sum(volumes[:, None] * sums / 4.0, axis=0)
        vertex_outer = np.einsum("nki,nkj->nij", points, points)
        sum_outer = np.einsum("ni,nj->nij", sums, sums)
        second_moment += np.sum(
            volumes[:, None, None] * (vertex_outer + sum_outer) / 20.0,
            axis=0,
        )
    if total_volume <= 0.0:
        raise ValueError("watertight mesh has zero volume")
    center = first_moment / total_volume
    inertia_origin = np.trace(second_moment) * np.eye(3) - second_moment
    offset = total_volume * (
        np.dot(center, center) * np.eye(3) - np.outer(center, center)
    )
    inertia = inertia_origin - offset
    inertia = 0.5 * (inertia + inertia.T)
    return InertialMetadata(total_volume, inertia, center)


def _face_components(triangles: np.ndarray) -> tuple[np.ndarray, ...]:
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for index, face in enumerate(triangles):
        for first, second in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            edge_faces.setdefault(tuple(sorted((first, second))), []).append(index)
    neighbors = [set() for _ in triangles]
    for faces in edge_faces.values():
        for first in faces:
            neighbors[first].update(index for index in faces if index != first)
    remaining = set(range(len(triangles)))
    components = []
    while remaining:
        pending = [remaining.pop()]
        selected = []
        while pending:
            face = pending.pop()
            selected.append(face)
            adjacent = neighbors[face] & remaining
            remaining.difference_update(adjacent)
            pending.extend(adjacent)
        components.append(np.asarray(selected, dtype=np.int64))
    return tuple(components)


def dsf_support_points(
    nodes: np.ndarray,
    sharpness: int | float,
    directions: np.ndarray,
) -> np.ndarray:
    """Evaluate the smooth finite-sharpness DSF boundary."""

    values = np.asarray(nodes, dtype=np.float64)
    dirs = np.asarray(directions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 4:
        raise ValueError("DSF nodes must have shape (N, 3), N >= 4")
    if dirs.ndim != 2 or dirs.shape[1] != 3 or not len(dirs):
        raise ValueError("DSF directions must have shape (M, 3), M > 0")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(dirs)):
        raise ValueError("DSF nodes and directions must be finite")
    if not np.isfinite(sharpness) or sharpness < 2.0:
        raise ValueError("DSF sharpness must be at least two")
    norm = np.linalg.norm(dirs, axis=1)
    if np.any(norm <= 0.0):
        raise ValueError("DSF directions must be non-zero")
    dirs = dirs / norm[:, None]
    center = values.mean(axis=0)
    local = values - center
    projection = np.maximum(dirs @ local.T, 0.0)
    maximum = np.max(projection, axis=1)
    if np.any(maximum <= 0.0):
        raise ValueError("DSF nodes do not span every requested direction")
    normalized = projection / maximum[:, None]
    power_minus_one = normalized ** (sharpness - 1.0)
    power_sum = np.sum(power_minus_one * normalized, axis=1)
    factor = power_sum ** (1.0 / sharpness - 1.0)
    return center + (factor[:, None] * power_minus_one) @ local


def convex_faces(points: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import ConvexHull
    except ImportError as error:
        raise RuntimeError("SciPy is required for convex surface geometry") from error
    values = np.asarray(points, dtype=np.float64)
    faces = np.asarray(ConvexHull(values).simplices, dtype=np.int64).copy()
    center = values.mean(axis=0)
    for index, face in enumerate(faces):
        edge_1 = values[face[1]] - values[face[0]]
        edge_2 = values[face[2]] - values[face[0]]
        if np.dot(np.cross(edge_1, edge_2), values[face[0]] - center) < 0.0:
            faces[index] = face[::-1]
    return faces


def farthest_points(points: np.ndarray, maximum: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if len(values) <= maximum:
        return np.array(values, copy=True)
    selected = np.empty(maximum, dtype=np.int64)
    selected[0] = int(
        np.argmax(np.linalg.norm(values - values.mean(axis=0), axis=1))
    )
    distance = np.full(len(values), np.inf)
    for index in range(1, maximum):
        delta = values - values[selected[index - 1]]
        distance = np.minimum(distance, np.einsum("ij,ij->i", delta, delta))
        selected[index] = int(np.argmax(distance))
    return np.array(values[selected], copy=True)
