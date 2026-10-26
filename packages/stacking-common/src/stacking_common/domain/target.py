"""Platform-neutral target-region geometry."""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull

from ._validation import pose_array, vector3_array
from .geometry import ConvexGeometry, transform_points


@dataclass(frozen=True)
class TargetComponent:
    """One convex target piece, posed in the canonical world frame.

    Component IDs describe the decomposition, not separate construction goals.
    Mesh containment uses the convex polytope, as for the original target;
    native posegen uses its smooth DSF representation of these vertices.
    """

    component_id: str
    geometry: ConvexGeometry
    pose: np.ndarray = field(
        default_factory=lambda: np.array([0., 0., 0., 0., 0., 0., 1.]))
    _world_geometry: ConvexGeometry = field(init=False, repr=False)
    _footprint: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.component_id, str) or not self.component_id.strip():
            raise ValueError("target component ID must be a non-empty string")
        if not isinstance(self.geometry, ConvexGeometry):
            raise TypeError("target component geometry must be ConvexGeometry")
        pose = pose_array(self.pose, "target component pose")
        pose[3:] /= np.linalg.norm(pose[3:])
        pose.setflags(write=False)
        geometry = ConvexGeometry(
            transform_points(self.geometry.points, pose), self.geometry.faces,
            sharpness=self.geometry.sharpness,
        )
        xy = geometry.points[:, :2]
        footprint = xy[ConvexHull(xy).vertices].copy()
        footprint.setflags(write=False)
        object.__setattr__(self, "component_id", self.component_id.strip())
        object.__setattr__(self, "pose", pose)
        object.__setattr__(self, "_world_geometry", geometry)
        object.__setattr__(self, "_footprint", footprint)

    @property
    def world_geometry(self) -> ConvexGeometry:
        """Immutable component mesh with its rigid transform applied once."""
        return self._world_geometry

    @property
    def bounds(self) -> np.ndarray:
        points = self._world_geometry.points
        return np.stack((points.min(axis=0), points.max(axis=0)))

    def contains(self, points: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
        return self._world_geometry.contains(points, tolerance)

    def footprint_distance(self, points: np.ndarray) -> np.ndarray:
        """XY distance to the component projection; zero inside it."""
        values = _xy_points(points)
        edges = np.roll(self._footprint, -1, axis=0) - self._footprint
        delta = values[:, None, :] - self._footprint[None, :, :]
        fraction = np.clip(
            np.einsum("nej,ej->ne", delta, edges)
            / np.einsum("ej,ej->e", edges, edges), 0., 1.)
        distance = np.linalg.norm(delta - fraction[:, :, None] * edges, axis=2).min(axis=1)
        # ConvexHull orders the projected vertices counterclockwise.
        cross = edges[None, :, 0] * delta[:, :, 1] - edges[None, :, 1] * delta[:, :, 0]
        distance[np.all(cross >= 0., axis=1)] = 0.
        return distance


@dataclass(frozen=True)
class CompositeTargetGeometry:
    """Union of convex target components, without filling gaps or notches.

    This is a geometry/native-posegen contract, not yet an end-to-end search
    target. It deliberately has no synthetic width/origin/taper or volume:
    rectangular policies and summed component volumes are incorrect for a
    general union. Consumers must use component or union queries, not box fields.
    """

    components: tuple[TargetComponent, ...]

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if not components:
            raise ValueError("composite target requires at least one component")
        if not all(isinstance(value, TargetComponent) for value in components):
            raise TypeError("composite target components must be TargetComponent values")
        ids = [component.component_id for component in components]
        if len(set(ids)) != len(ids):
            raise ValueError("target component IDs must be unique")
        object.__setattr__(self, "components", components)

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(component.component_id for component in self.components)

    @property
    def meshes(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Separate world-frame meshes, preserving every convex piece."""
        return tuple((component.world_geometry.points.copy(),
                      component.world_geometry.faces.copy())
                     for component in self.components)

    @property
    def bounds(self) -> np.ndarray:
        bounds = np.stack([component.bounds for component in self.components])
        return np.stack((bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)))

    @property
    def points(self) -> np.ndarray:
        """Concatenated vertices, never the convex hull of the union."""
        return np.concatenate([component.world_geometry.points for component in self.components])

    @property
    def faces(self) -> np.ndarray:
        faces, offset = [], 0
        for component in self.components:
            geometry = component.world_geometry
            faces.append(geometry.faces + offset)
            offset += len(geometry.points)
        return np.concatenate(faces)

    def component_membership(self, points: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
        """Return an (N, K) mask; overlapping pieces may both contain a point."""
        return np.column_stack([component.contains(points, tolerance) for component in self.components])

    def contains(self, points: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
        return np.any(self.component_membership(points, tolerance), axis=1)

    def horizontal_mask(self, points: np.ndarray, z: float, tolerance: float = 1e-9) -> np.ndarray:
        """Union membership of XY samples at one world-frame height."""
        return self.contains(_horizontal_points(points, z), tolerance)

    def footprint_distance(self, points: np.ndarray) -> np.ndarray:
        """Distance to the union of XY projections, not to its bounding box."""
        return np.min([component.footprint_distance(points) for component in self.components], axis=0)


@dataclass(frozen=True)
class DirectionalTaper:
    """Inward target-face angles in radians."""

    width_positive: float = 0.0
    width_negative: float = 0.0
    length_positive: float = 0.0
    length_negative: float = 0.0

    @classmethod
    def from_degrees(
        cls,
        width_positive: float = 0.0,
        width_negative: float = 0.0,
        length_positive: float = 0.0,
        length_negative: float = 0.0,
    ) -> "DirectionalTaper":
        return cls(
            width_positive=float(np.deg2rad(width_positive)),
            width_negative=float(np.deg2rad(width_negative)),
            length_positive=float(np.deg2rad(length_positive)),
            length_negative=float(np.deg2rad(length_negative)),
        )

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.width_positive,
                self.width_negative,
                self.length_positive,
                self.length_negative,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("target taper angles must be finite")
        if np.any(np.abs(values) >= np.pi / 2.0):
            raise ValueError("target taper angles must be between -pi/2 and pi/2")


@dataclass(frozen=True)
class TargetGeometry:
    """A tapered rectangular target volume in the canonical world frame."""

    width: float
    length: float
    height: float
    origin: np.ndarray
    taper: DirectionalTaper
    support_overlap: float = 0.05
    _geometry: ConvexGeometry = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.taper, DirectionalTaper):
            raise TypeError("target taper must be a DirectionalTaper")
        dimensions = np.asarray(
            [self.width, self.length, self.height, self.support_overlap],
            dtype=float,
        )
        if not np.all(np.isfinite(dimensions)):
            raise ValueError("target dimensions must be finite")
        if self.width <= 0.0 or self.length <= 0.0 or self.height <= 0.0:
            raise ValueError("target dimensions must be positive")
        if self.support_overlap < 0.0:
            raise ValueError("target support overlap must be non-negative")
        origin = vector3_array(self.origin, "target origin")

        points = self._vertices(origin)
        faces = np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ],
            dtype=int,
        )
        geometry = ConvexGeometry(points, faces)
        object.__setattr__(self, "width", float(self.width))
        object.__setattr__(self, "length", float(self.length))
        object.__setattr__(self, "height", float(self.height))
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "support_overlap", float(self.support_overlap))
        object.__setattr__(self, "_geometry", geometry)

    @property
    def points(self) -> np.ndarray:
        return self._geometry.points.copy()

    @property
    def faces(self) -> np.ndarray:
        return self._geometry.faces.copy()

    @property
    def volume(self) -> float:
        return self._geometry.volume

    def contains(self, points: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
        return self._geometry.contains(points, tolerance)

    @property
    def components(self) -> tuple[TargetComponent, ...]:
        """Expose the legacy tapered box as one canonical component."""
        return (TargetComponent("target", self._geometry),)

    @property
    def component_ids(self) -> tuple[str, ...]:
        return ("target",)

    @property
    def meshes(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        return ((self.points, self.faces),)

    @property
    def bounds(self) -> np.ndarray:
        return np.stack((self._geometry.points.min(axis=0), self._geometry.points.max(axis=0)))

    def component_membership(self, points: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
        return self.contains(points, tolerance)[:, None]

    def horizontal_mask(self, points: np.ndarray, z: float, tolerance: float = 1e-9) -> np.ndarray:
        return self.contains(_horizontal_points(points, z), tolerance)

    def footprint_distance(self, points: np.ndarray) -> np.ndarray:
        return self.components[0].footprint_distance(points)

    def _vertices(self, origin: np.ndarray) -> np.ndarray:
        bottom_z = -self.support_overlap * self.height
        top_z = self.height

        def bounds(z: float) -> tuple[float, float, float, float]:
            x_min = -self.width / 2.0 + np.tan(
                self.taper.width_negative
            ) * z
            x_max = self.width / 2.0 - np.tan(
                self.taper.width_positive
            ) * z
            y_min = -self.length / 2.0 + np.tan(
                self.taper.length_negative
            ) * z
            y_max = self.length / 2.0 - np.tan(
                self.taper.length_positive
            ) * z
            if x_min >= x_max or y_min >= y_max:
                raise ValueError("target taper closes the target before its top")
            return x_min, x_max, y_min, y_max

        bottom = bounds(bottom_z)
        top = bounds(top_z)
        return origin + np.array(
            [
                [bottom[0], bottom[2], bottom_z],
                [bottom[1], bottom[2], bottom_z],
                [bottom[1], bottom[3], bottom_z],
                [bottom[0], bottom[3], bottom_z],
                [top[0], top[2], top_z],
                [top[1], top[2], top_z],
                [top[1], top[3], top_z],
                [top[0], top[3], top_z],
            ]
        )


TargetRegion = TargetGeometry | CompositeTargetGeometry


def _xy_points(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("target XY points must be finite with shape (N, 2)")
    return values


def _horizontal_points(points: np.ndarray, z: float) -> np.ndarray:
    values = _xy_points(points)
    if not np.isfinite(z):
        raise ValueError("target section height must be finite")
    return np.column_stack((values, np.full(len(values), z)))
