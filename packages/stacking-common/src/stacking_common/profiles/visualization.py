"""Loading of platform-owned Operations Console reference geometry."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


ENVIRONMENT_LAYERS = ("ground", "support", "obstacle")


@dataclass(frozen=True)
class SceneBox:
    """One axis-aligned platform reference or planning volume."""

    component_id: str
    layer: str
    minimum: np.ndarray
    maximum: np.ndarray
    color: tuple[float, float, float]
    opacity: float

    def __post_init__(self) -> None:
        component_id = str(self.component_id).strip()
        layer = str(self.layer).strip()
        minimum = np.asarray(self.minimum, dtype=float)
        maximum = np.asarray(self.maximum, dtype=float)
        color = tuple(float(value) for value in self.color)
        opacity = float(self.opacity)
        if not component_id:
            raise ValueError("scene-box ID is required")
        if layer not in ENVIRONMENT_LAYERS:
            raise ValueError(f"unsupported scene-box layer: {layer}")
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("scene-box bounds must be three-vectors")
        if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
            raise ValueError("scene-box bounds must be finite")
        if np.any(maximum <= minimum):
            raise ValueError("scene-box maximum must exceed its minimum")
        if len(color) != 3 or any(not 0.0 <= value <= 1.0 for value in color):
            raise ValueError("scene-box color must contain three values in [0, 1]")
        if not np.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
            raise ValueError("scene-box opacity must be in [0, 1]")
        minimum = minimum.copy()
        maximum = maximum.copy()
        minimum.setflags(write=False)
        maximum.setflags(write=False)
        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "opacity", opacity)

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        return (
            float(self.minimum[0]),
            float(self.maximum[0]),
            float(self.minimum[1]),
            float(self.maximum[1]),
            float(self.minimum[2]),
            float(self.maximum[2]),
        )


@dataclass(frozen=True)
class VisualizationProfile:
    """Resolved platform scene components and default layer visibility."""

    boxes: tuple[SceneBox, ...]
    default_layers: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        boxes = tuple(self.boxes)
        defaults = dict(self.default_layers)
        if not all(isinstance(value, SceneBox) for value in boxes):
            raise TypeError("visualization boxes must be SceneBox values")
        component_ids = [value.component_id for value in boxes]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("visualization scene-box IDs must be unique")
        if set(defaults) != set(ENVIRONMENT_LAYERS):
            raise ValueError("every environment layer requires a visibility default")
        if any(not isinstance(value, bool) for value in defaults.values()):
            raise TypeError("environment-layer visibility must be boolean")
        object.__setattr__(self, "boxes", boxes)
        object.__setattr__(
            self,
            "default_layers",
            tuple((layer, defaults[layer]) for layer in ENVIRONMENT_LAYERS),
        )

    @property
    def layer_visibility(self) -> dict[str, bool]:
        return dict(self.default_layers)


def load_visualization_profile(path: str | Path) -> VisualizationProfile:
    """Load the SI-unit platform visualization profile."""

    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(f"missing visualization profile: {profile_path}")
    values = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise TypeError("visualization profile root must be a mapping")
    if values.get("units") != "si":
        raise ValueError("visualization profile units must be 'si'")
    environment = _mapping(values, "environment")
    defaults = _mapping(environment, "default_layers")
    raw_boxes = _required(environment, "boxes")
    if not isinstance(raw_boxes, Sequence) or isinstance(raw_boxes, str):
        raise TypeError("visualization boxes must be a sequence")
    return VisualizationProfile(
        boxes=tuple(
            _scene_box(_mapping_value(value, "scene box"))
            for value in raw_boxes
        ),
        default_layers=tuple(
            (layer, _boolean(defaults, layer))
            for layer in ENVIRONMENT_LAYERS
        ),
    )


def _scene_box(values: Mapping[str, object]) -> SceneBox:
    return SceneBox(
        component_id=str(_required(values, "id")),
        layer=str(_required(values, "layer")),
        minimum=np.asarray(_required(values, "minimum"), dtype=float),
        maximum=np.asarray(_required(values, "maximum"), dtype=float),
        color=tuple(_required(values, "color")),
        opacity=float(_required(values, "opacity")),
    )


def _mapping(values: Mapping[str, object], key: str) -> Mapping[str, object]:
    return _mapping_value(_required(values, key), key)


def _mapping_value(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"visualization profile '{name}' must be a mapping")
    return value


def _required(values: Mapping[str, object], key: str):
    if key not in values:
        raise ValueError(f"visualization profile is missing '{key}'")
    return values[key]


def _boolean(values: Mapping[str, object], key: str) -> bool:
    value = _required(values, key)
    if not isinstance(value, bool):
        raise TypeError(f"visualization profile '{key}' must be boolean")
    return value
