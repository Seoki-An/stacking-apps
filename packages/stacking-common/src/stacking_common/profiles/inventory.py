"""Typed platform object-asset and target configuration."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from stacking_common.domain import DirectionalTaper, TargetGeometry


@dataclass(frozen=True)
class TargetProfile:
    width: float
    length: float
    height: float
    origin: np.ndarray
    support_overlap: float
    taper: DirectionalTaper

    def __post_init__(self) -> None:
        geometry = self.geometry()
        object.__setattr__(self, "width", geometry.width)
        object.__setattr__(self, "length", geometry.length)
        object.__setattr__(self, "height", geometry.height)
        object.__setattr__(self, "origin", geometry.origin)
        object.__setattr__(self, "support_overlap", geometry.support_overlap)

    def geometry(self) -> TargetGeometry:
        return TargetGeometry(
            width=self.width,
            length=self.length,
            height=self.height,
            origin=self.origin,
            taper=self.taper,
            support_overlap=self.support_overlap,
        )


@dataclass(frozen=True)
class InventoryProfile:
    model_directory: Path
    object_ids: tuple[int, ...]
    density: float
    density_overrides: tuple[tuple[int, float], ...]
    target: TargetProfile

    def __post_init__(self) -> None:
        if not self.object_ids:
            raise ValueError("platform object catalog must not be empty")
        if any(object_id <= 0 for object_id in self.object_ids):
            raise ValueError("platform object IDs must be positive")
        if len(set(self.object_ids)) != len(self.object_ids):
            raise ValueError("platform object IDs must be unique")
        if not np.isfinite(self.density) or self.density <= 0.0:
            raise ValueError("object density must be positive and finite")
        override_ids = [object_id for object_id, _ in self.density_overrides]
        if len(override_ids) != len(set(override_ids)):
            raise ValueError("object density overrides must have unique IDs")
        if not set(override_ids).issubset(self.object_ids):
            raise ValueError("density override refers to an unknown object")
        if any(
            not np.isfinite(density) or density <= 0.0
            for _, density in self.density_overrides
        ):
            raise ValueError("density overrides must be positive and finite")
    def density_for(self, object_id: int) -> float:
        value = int(object_id)
        if value not in self.object_ids:
            raise ValueError(f"object {value} is not in the platform catalog")
        return dict(self.density_overrides).get(value, self.density)


def load_inventory_profile(path: str | Path) -> InventoryProfile:
    """Load the explicitly resolved platform inventory profile."""

    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(f"missing inventory profile: {profile_path}")
    values = yaml.safe_load(profile_path.read_text())
    if not isinstance(values, Mapping):
        raise TypeError("inventory profile root must be a mapping")
    if values.get("units") != "si":
        raise ValueError("inventory profile units must be 'si'")

    objects = _mapping(values, "objects")
    target = _mapping(values, "target")
    taper = _mapping(target, "taper_degrees")
    object_ids = _required(objects, "ids")
    if not isinstance(object_ids, Sequence) or isinstance(object_ids, str):
        raise TypeError("inventory profile object IDs must be a sequence")
    raw_overrides = objects.get("density_overrides", {})
    if not isinstance(raw_overrides, Mapping):
        raise TypeError("object density overrides must be a mapping")
    return InventoryProfile(
        model_directory=Path(str(_required(objects, "model_directory"))),
        object_ids=tuple(int(object_id) for object_id in object_ids),
        density=float(_required(objects, "density")),
        density_overrides=tuple(
            sorted(
                (int(object_id), float(density))
                for object_id, density in raw_overrides.items()
            )
        ),
        target=TargetProfile(
            width=float(_required(target, "width")),
            length=float(_required(target, "length")),
            height=float(_required(target, "height")),
            origin=np.asarray(_required(target, "origin"), dtype=float),
            support_overlap=float(_required(target, "support_overlap")),
            taper=DirectionalTaper.from_degrees(
                width_positive=float(
                    _required(taper, "width_positive")
                ),
                width_negative=float(
                    _required(taper, "width_negative")
                ),
                length_positive=float(
                    _required(taper, "length_positive")
                ),
                length_negative=float(
                    _required(taper, "length_negative")
                ),
            ),
        ),
    )


def _mapping(values: Mapping[str, object], key: str) -> Mapping[str, object]:
    result = _required(values, key)
    if not isinstance(result, Mapping):
        raise TypeError(f"inventory profile '{key}' must be a mapping")
    return result


def _required(values: Mapping[str, object], key: str):
    if key not in values:
        raise ValueError(f"inventory profile is missing '{key}'")
    return values[key]
