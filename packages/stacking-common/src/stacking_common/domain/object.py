"""An immutable stack-object model shared by common workflows."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .geometry import ConvexGeometry, _farthest_point_indices


@dataclass(frozen=True)
class StackObject:
    """Application geometry associated with one stacking-core body model."""

    object_id: int
    body_model: object
    geometries: tuple[ConvexGeometry, ...]

    def __init__(
        self,
        object_id: int,
        body_model: object,
        geometries: Sequence[ConvexGeometry],
    ) -> None:
        values = tuple(geometries)
        if int(object_id) < 0:
            raise ValueError("stack object ID must be non-negative")
        if not values:
            raise ValueError("stack object requires at least one geometry")
        if getattr(body_model, "mass", None) is None:
            raise ValueError("stack object body model must contain inertial data")
        inertia = np.asarray(getattr(body_model, "inertia", None), dtype=float)
        if inertia.shape != (3, 3):
            raise ValueError("stack object inertia must have shape (3, 3)")
        object.__setattr__(self, "object_id", int(object_id))
        object.__setattr__(self, "body_model", body_model)
        object.__setattr__(self, "geometries", values)

    @property
    def mass(self) -> float:
        return float(self.body_model.mass)

    @property
    def inertia(self) -> np.ndarray:
        return np.asarray(self.body_model.inertia, dtype=float).copy()

    @property
    def volume(self) -> float:
        return float(sum(geometry.volume for geometry in self.geometries))

    @property
    def bound_extent(self) -> np.ndarray:
        points, _ = self.mesh_arrays()
        return np.ptp(points, axis=0)

    @property
    def sharpness(self) -> float:
        volumes = np.asarray(
            [geometry.volume for geometry in self.geometries], dtype=float
        )
        values = np.asarray(
            [geometry.sharpness for geometry in self.geometries], dtype=float
        )
        return float(np.average(values, weights=volumes))

    def mesh_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the combined body-frame triangular mesh."""

        points = []
        faces = []
        offset = 0
        for geometry in self.geometries:
            points.append(geometry.points)
            faces.append(geometry.faces + offset)
            offset += len(geometry.points)
        return np.concatenate(points).copy(), np.concatenate(faces).copy()

    def contains_local_points(
        self, points: np.ndarray, tolerance: float = 1e-9
    ) -> np.ndarray:
        """Test body-frame points against the union of convex components."""

        values = np.asarray(points, dtype=float)
        inside = np.zeros(len(values), dtype=bool)
        for geometry in self.geometries:
            inside |= geometry.contains(values, tolerance)
        return inside

    def representative_face_normals(self) -> tuple[np.ndarray, np.ndarray]:
        """Return outward component-face normals and their surface areas."""

        normals = []
        areas = []
        for geometry in self.geometries:
            _, component_normals, component_areas = geometry.face_data()
            normals.append(component_normals)
            areas.append(component_areas)
        return np.concatenate(normals), np.concatenate(areas)

    def surface_samples(
        self, count: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return deterministic padded samples across every component."""

        if count <= 0:
            raise ValueError("surface sample count must be positive")
        points = []
        normals = []
        for geometry in self.geometries:
            component_points, component_normals, mask = (
                geometry.vertex_surface_samples(len(geometry.points))
            )
            points.append(component_points[mask])
            normals.append(component_normals[mask])
        points_array = np.concatenate(points)
        normals_array = np.concatenate(normals)
        indices = _farthest_point_indices(
            points_array, min(count, len(points_array))
        )
        out_points = np.zeros((count, 3), dtype=np.float32)
        out_normals = np.zeros((count, 3), dtype=np.float32)
        out_mask = np.zeros(count, dtype=bool)
        out_points[: len(indices)] = points_array[indices]
        out_normals[: len(indices)] = normals_array[indices]
        out_mask[: len(indices)] = True
        return out_points, out_normals, out_mask

    def physical_features(self) -> np.ndarray:
        """Return mass, volume, inertia, and extent for learning observations."""

        return np.concatenate(
            [
                np.asarray([self.mass, self.volume], dtype=np.float32),
                self.inertia.astype(np.float32).reshape(-1),
                self.bound_extent.astype(np.float32),
            ]
        )
