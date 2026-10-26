"""Typed platform object-asset and target configuration."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from stacking_common.domain import (
    CompositeTargetGeometry,
    ConvexGeometry,
    DirectionalTaper,
    TargetComponent,
    TargetGeometry,
)


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
class TargetComponentProfile:
    """One local convex target shape and its world-frame rigid pose.

    Exactly one of ``shape`` (the legacy tapered volume) or ``nodes`` is set.
    Nodes define one convex hull, never a hull over several components.
    """

    component_id: str
    shape: TargetProfile | None = None
    nodes: np.ndarray | None = None
    pose: np.ndarray = field(
        default_factory=lambda: np.array([0., 0., 0., 0., 0., 0., 1.])
    )

    def __post_init__(self) -> None:
        if (self.shape is None) == (self.nodes is None):
            raise ValueError("target component requires exactly one of shape or nodes")
        if self.shape is not None and not isinstance(self.shape, TargetProfile):
            raise TypeError("target component shape must be a TargetProfile")
        if self.nodes is not None:
            nodes = np.array(self.nodes, dtype=float, copy=True)
            if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) < 4:
                raise ValueError("target component nodes must have shape (N, 3), N >= 4")
            if not np.all(np.isfinite(nodes)):
                raise ValueError("target component nodes must be finite")
            nodes.setflags(write=False)
            object.__setattr__(self, "nodes", nodes)
        component = self.geometry()
        object.__setattr__(self, "component_id", component.component_id)
        object.__setattr__(self, "pose", component.pose)

    def geometry(self) -> TargetComponent:
        if self.shape is not None:
            shape = self.shape.geometry()
            geometry = ConvexGeometry(shape.points, shape.faces)
        else:
            from scipy.spatial import ConvexHull, QhullError

            try:
                hull = ConvexHull(self.nodes)
            except QhullError as error:
                raise ValueError(
                    "target component nodes must define a nondegenerate 3D convex hull"
                ) from error
            faces = hull.simplices.copy()
            triangles = self.nodes[faces]
            normals = np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            )
            reverse = np.einsum("ij,ij->i", normals, hull.equations[:, :3]) < 0
            faces[reverse] = faces[reverse, ::-1]
            geometry = ConvexGeometry(self.nodes, faces)
        return TargetComponent(self.component_id, geometry, self.pose)


@dataclass(frozen=True)
class CompositeTargetProfile:
    """A target defined by the union of independently posed convex components."""

    components: tuple[TargetComponentProfile, ...]

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if any(not isinstance(item, TargetComponentProfile) for item in components):
            raise TypeError("target components must be TargetComponentProfile values")
        object.__setattr__(self, "components", components)
        self.geometry()

    def geometry(self) -> CompositeTargetGeometry:
        return CompositeTargetGeometry(tuple(item.geometry() for item in self.components))


@dataclass(frozen=True)
class InventoryProfile:
    model_directory: Path
    object_ids: tuple[int, ...]
    density: float
    density_overrides: tuple[tuple[int, float], ...]
    target: TargetProfile | CompositeTargetProfile
    target_name: str | None = field(default=None, metadata={"omit_default_from_context": True})

    def __post_init__(self) -> None:
        if self.target_name is not None and (
            not isinstance(self.target_name, str) or not self.target_name.strip()
        ):
            raise ValueError("inventory target name must be a non-empty string")
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


def load_inventory_profile(path: str | Path, *, target_shape=None, recorded_target=None) -> InventoryProfile:
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
    from .targets import resolve_inventory_target
    target, target_name = resolve_inventory_target(profile_path, values,
        target_shape=target_shape, recorded_target=recorded_target)
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
        target=target,
        target_name=target_name,
    )


def _target_profile(
    values: Mapping[str, object],
) -> TargetProfile | CompositeTargetProfile:
    if "components" not in values:
        return _shape_profile(values)
    if set(values) != {"components"}:
        raise ValueError("target components cannot be mixed with single-target fields")
    components = values["components"]
    if not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
        raise TypeError("target components must be a sequence")
    profiles = []
    for component in components:
        if not isinstance(component, Mapping):
            raise TypeError("target component must be a mapping")
        unknown = set(component) - {"id", "pose", "shape", "nodes"}
        if unknown:
            raise ValueError(f"unknown target component fields: {sorted(unknown)}")
        if ("shape" in component) == ("nodes" in component):
            raise ValueError("target component requires exactly one of shape or nodes")
        kwargs = {}
        if "pose" in component:
            kwargs["pose"] = np.asarray(component["pose"], dtype=float)
        profiles.append(TargetComponentProfile(
            component_id=_required(component, "id"),
            shape=(
                _shape_profile(_mapping(component, "shape"))
                if "shape" in component else None
            ),
            nodes=(np.asarray(component["nodes"], dtype=float)
                   if "nodes" in component else None),
            **kwargs,
        ))
    return CompositeTargetProfile(tuple(profiles))


def target_profile_from_record(values: Mapping[str, object]) -> TargetProfile | CompositeTargetProfile:
    """Rehydrate resolved context values (radians), not the YAML input schema."""
    if "components" not in values:
        return TargetProfile(**{**values, "support_overlap": values.get("support_overlap", 0.),
                                "taper": DirectionalTaper(**values.get("taper", {}))})
    if set(values) != {"components"}:
        raise ValueError("target components cannot be mixed with single-target fields")
    return CompositeTargetProfile(tuple(TargetComponentProfile(
        component_id=item["component_id"],
        shape=None if item.get("shape") is None else target_profile_from_record(item["shape"]),
        nodes=item.get("nodes"), pose=item["pose"],
    ) for item in values["components"]))


def _shape_profile(values: Mapping[str, object]) -> TargetProfile:
    taper = _mapping(values, "taper_degrees")
    return TargetProfile(
        width=float(_required(values, "width")),
        length=float(_required(values, "length")),
        height=float(_required(values, "height")),
        origin=np.asarray(_required(values, "origin"), dtype=float),
        support_overlap=float(_required(values, "support_overlap")),
        taper=DirectionalTaper.from_degrees(
            width_positive=float(_required(taper, "width_positive")),
            width_negative=float(_required(taper, "width_negative")),
            length_positive=float(_required(taper, "length_positive")),
            length_negative=float(_required(taper, "length_negative")),
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
