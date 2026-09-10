"""Platform-neutral target-region geometry."""

from dataclasses import dataclass, field

import numpy as np

from ._validation import vector3_array
from .geometry import ConvexGeometry


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
