"""Validated convex geometry used by common stacking workflows."""

from dataclasses import dataclass, field

import numpy as np


def quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return a rotation matrix for a scalar-last ``[x, y, z, w]`` quaternion."""

    x, y, z, w = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 1e-12:
        raise ValueError("quaternion must have non-zero norm")
    x, y, z, w = np.array([x, y, z, w]) / norm
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    )


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Transform body-frame points by ``[x, y, z, qx, qy, qz, qw]``."""

    pose = _validated_pose(pose)
    return np.asarray(points, dtype=float) @ quaternion_matrix(pose[3:]).T + pose[:3]


def inverse_transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Transform world-frame points into a posed body's local frame."""

    pose = _validated_pose(pose)
    return (np.asarray(points, dtype=float) - pose[:3]) @ quaternion_matrix(
        pose[3:]
    )


@dataclass(frozen=True)
class ConvexGeometry:
    """One convex geometry component represented by a triangular surface."""

    points: np.ndarray
    faces: np.ndarray
    sharpness: float = 20.0
    _face_centers: np.ndarray = field(init=False, repr=False)
    _face_normals: np.ndarray = field(init=False, repr=False)
    _face_areas: np.ndarray = field(init=False, repr=False)
    _plane_offsets: np.ndarray = field(init=False, repr=False)
    _volume: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        faces = np.asarray(self.faces, dtype=int)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
            raise ValueError("geometry points must have shape (N, 3), N >= 4")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4:
            raise ValueError("geometry faces must have shape (M, 3), M >= 4")
        if not np.all(np.isfinite(points)):
            raise ValueError("geometry points must contain only finite values")
        if np.any(faces < 0) or np.any(faces >= len(points)):
            raise ValueError("geometry faces contain an invalid vertex index")
        if not np.isfinite(self.sharpness) or self.sharpness <= 0.0:
            raise ValueError("geometry sharpness must be positive")

        centers, normals, areas = _face_data(points, faces)
        if len(centers) != len(faces):
            raise ValueError("geometry contains a degenerate face")
        offsets = np.einsum("ij,ij->i", normals, centers)
        tolerance = max(float(np.linalg.norm(np.ptp(points, axis=0))) * 1e-9, 1e-12)
        if np.any(points @ normals.T - offsets > tolerance):
            raise ValueError("geometry surface must define a convex polytope")

        center = np.mean(points, axis=0)
        triangles = points[faces] - center
        volumes = np.abs(
            np.einsum(
                "ij,ij->i",
                triangles[:, 0],
                np.cross(triangles[:, 1], triangles[:, 2]),
            )
        ) / 6.0
        volume = float(np.sum(volumes))
        if volume <= 1e-12:
            raise ValueError("geometry must have positive volume")

        object.__setattr__(self, "points", _readonly_copy(points))
        object.__setattr__(self, "faces", _readonly_copy(faces))
        object.__setattr__(self, "sharpness", float(self.sharpness))
        object.__setattr__(self, "_face_centers", _readonly_copy(centers))
        object.__setattr__(self, "_face_normals", _readonly_copy(normals))
        object.__setattr__(self, "_face_areas", _readonly_copy(areas))
        object.__setattr__(self, "_plane_offsets", _readonly_copy(offsets))
        object.__setattr__(self, "_volume", volume)

    @property
    def volume(self) -> float:
        return self._volume

    @property
    def extent(self) -> np.ndarray:
        return np.ptp(self.points, axis=0)

    def face_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return body-frame face centers, outward normals, and areas."""

        return (
            self._face_centers.copy(),
            self._face_normals.copy(),
            self._face_areas.copy(),
        )

    def contains(self, points: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
        """Test local-frame points against this convex component."""

        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("query points must have shape (N, 3)")
        # Put query points on the contiguous axis. Voxel queries greatly
        # outnumber faces, so the reduction combines whole plane rows into the
        # output mask instead of separately reducing each point's short row.
        return np.all(
            self._face_normals @ values.T
            <= (self._plane_offsets + float(tolerance))[:, None],
            axis=0,
        )

    def vertex_surface_samples(
        self, count: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return deterministic padded samples for learning observations."""

        if count <= 0:
            raise ValueError("surface sample count must be positive")
        normals = np.zeros_like(self.points)
        for face_index, face in enumerate(self.faces):
            normals[face] += self._face_normals[face_index] * self._face_areas[
                face_index
            ]
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > 1e-12
        normals[valid] /= norms[valid, None]

        indices = _farthest_point_indices(self.points, min(count, len(self.points)))
        out_points = np.zeros((count, 3), dtype=np.float32)
        out_normals = np.zeros((count, 3), dtype=np.float32)
        out_mask = np.zeros(count, dtype=bool)
        out_points[: len(indices)] = self.points[indices]
        out_normals[: len(indices)] = normals[indices]
        out_mask[: len(indices)] = True
        return out_points, out_normals, out_mask


def _validated_pose(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=float)
    if value.shape != (7,):
        raise ValueError("pose must have shape (7,)")
    if not np.all(np.isfinite(value)):
        raise ValueError("pose must contain only finite values")
    if np.linalg.norm(value[3:]) <= 1e-12:
        raise ValueError("pose quaternion must have non-zero norm")
    return value


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    result = value.copy()
    result.setflags(write=False)
    return result


def _face_data(
    points: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = points[faces]
    centers = np.mean(triangles, axis=1)
    raw_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    norms = np.linalg.norm(raw_normals, axis=1)
    valid = norms > 1e-12
    centers = centers[valid]
    raw_normals = raw_normals[valid]
    norms = norms[valid]
    normals = raw_normals / norms[:, None]
    outward = np.einsum(
        "ij,ij->i", normals, centers - np.mean(points, axis=0)
    ) >= 0.0
    normals[~outward] *= -1.0
    return centers, normals, 0.5 * norms


def _farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=int)
    center = np.mean(points, axis=0)
    first = int(np.argmax(np.sum((points - center) ** 2, axis=1)))
    selected = np.empty(count, dtype=int)
    selected[0] = first
    min_distance = np.sum((points - points[first]) ** 2, axis=1)
    for index in range(1, count):
        selected[index] = int(np.argmax(min_distance))
        distance = np.sum((points - points[selected[index]]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
    return selected
