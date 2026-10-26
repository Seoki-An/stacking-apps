"""Platform-owned named targets, resolved independently of object assets."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from .inventory import _target_profile, target_profile_from_record


@dataclass(frozen=True)
class TargetRegistry:
    default: str
    shapes: Mapping

    def select(self, name=None):
        name = self.default if name is None else name
        if name not in self.shapes:
            raise ValueError(f"unknown target shape {name!r}; choose from {', '.join(self.shapes)}")
        return self.shapes[name], name


def load_target_registry(path):
    values = yaml.safe_load(Path(path).read_text())
    if not isinstance(values, Mapping) or set(values) != {"units", "default", "shapes"}:
        raise ValueError("target registry requires units, default and shapes")
    if values["units"] != "si":
        raise ValueError("target registry units must be 'si'")
    if not isinstance(values["default"], str) or not values["default"].strip():
        raise ValueError("target registry default must name a registered shape")
    shapes = values["shapes"]
    if not isinstance(shapes, Mapping) or not shapes:
        raise ValueError("target registry shapes must be a non-empty mapping")
    if any(not isinstance(name, str) or not name.strip() or name != name.strip() for name in shapes):
        raise ValueError("target shape names must be non-empty trimmed strings")
    if any(not isinstance(shape, Mapping) for shape in shapes.values()):
        raise ValueError("each registered target must be a shape mapping")
    registry = TargetRegistry(values["default"], {name: _target_profile(shape) for name, shape in shapes.items()})
    registry.select()
    return registry


def resolve_inventory_target(path, values, *, target_shape=None, recorded_target=None):
    """Recorded inputs bypass the registry; old inline targets remain valid."""
    if recorded_target is not None:
        name = recorded_target.get("target_name")
        if target_shape is not None and target_shape != name:
            raise ValueError("resume preserves the recorded target; omit --target-shape")
        return target_profile_from_record(recorded_target["target"]), name
    if target_shape is None and "target" in values:
        if not isinstance(values["target"], Mapping):
            raise TypeError("inventory target must be a mapping")
        return _target_profile(values["target"]), None
    return load_target_registry(Path(path).with_name("target_shapes.yml")).select(target_shape)


def target_shape_choices(inventory_path):
    """Names/default for an app selector; legacy inline inputs need no registry."""
    path = Path(inventory_path)
    registry_path = path.with_name("target_shapes.yml")
    if not registry_path.is_file():
        return None, ()
    registry = load_target_registry(registry_path)
    values = yaml.safe_load(path.read_text())
    return (None if "target" in values else registry.default), tuple(registry.shapes)
