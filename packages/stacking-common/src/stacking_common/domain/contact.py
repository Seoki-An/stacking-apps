"""Typed contact data produced by a placement transition."""

from dataclasses import dataclass

import numpy as np

from ._validation import vector3_array


@dataclass(frozen=True)
class ContactPoint:
    entity_id_first: int
    entity_id_second: int
    object_index_first: int | None
    object_index_second: int | None
    ground_first: bool
    ground_second: bool
    point_first: np.ndarray
    point_second: np.ndarray
    normal: np.ndarray
    gap: float

    def __post_init__(self) -> None:
        entity_id_first = int(self.entity_id_first)
        entity_id_second = int(self.entity_id_second)
        if entity_id_first < 0 or entity_id_second < 0:
            raise ValueError("contact entity IDs must be non-negative")
        for name in ("object_index_first", "object_index_second"):
            value = getattr(self, name)
            value = None if value is None else int(value)
            if value is not None and value < 0:
                raise ValueError("contact object indices must be non-negative")
            object.__setattr__(self, name, value)
        gap = float(self.gap)
        if not np.isfinite(gap):
            raise ValueError("contact gap must be finite")
        object.__setattr__(self, "entity_id_first", entity_id_first)
        object.__setattr__(self, "entity_id_second", entity_id_second)
        object.__setattr__(self, "ground_first", bool(self.ground_first))
        object.__setattr__(self, "ground_second", bool(self.ground_second))
        object.__setattr__(
            self, "point_first", vector3_array(self.point_first, "point_first")
        )
        object.__setattr__(
            self, "point_second", vector3_array(self.point_second, "point_second")
        )
        normal = vector3_array(self.normal, "normal")
        if np.linalg.norm(normal) <= 1e-12:
            raise ValueError("contact normal must have non-zero norm")
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "gap", gap)
