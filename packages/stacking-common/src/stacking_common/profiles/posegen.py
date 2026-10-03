"""Loading of the resolved platform native pose-generation profile."""

from collections.abc import Mapping
from pathlib import Path

import yaml

from stacking_common.actions import CorePosegenProfile


def load_posegen_profile(path: str | Path) -> CorePosegenProfile:
    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(f"missing posegen profile: {profile_path}")
    values = yaml.safe_load(profile_path.read_text())
    if not isinstance(values, Mapping):
        raise TypeError("posegen profile root must be a mapping")
    if values.get("units") != "si":
        raise ValueError("posegen profile units must be 'si'")
    if "backend" in values:
        raise ValueError("move posegen backend to parallel.posegen_backend in the MCTS profile")
    scene = _mapping(values, "scene")
    return CorePosegenProfile(
        frame_id=int(_required(scene, "frame_id")),
        target_model_id=int(_required(scene, "target_model_id")),
        target_entity_id=int(_required(scene, "target_entity_id")),
        target_friction=float(_required(scene, "target_friction")),
        target_sharpness=int(_required(scene, "target_sharpness")),
        target_epsilon=float(_required(scene, "target_epsilon")),
        placed_as_boundaries=bool(_required(scene, "placed_as_boundaries")),
        boundary_radius=scene.get("boundary_radius"),
        trust_region=_mapping(values, "trust_region"),
        hausdorff=_mapping(values, "hausdorff"),
        objective=_mapping(values, "objective"),
        force_solver=_mapping(values, "force_solver"),
        schedule=_mapping(values, "schedule"),
    )


def _mapping(values: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = _required(values, key)
    if not isinstance(value, Mapping):
        raise TypeError(f"posegen profile '{key}' must be a mapping")
    return value


def _required(values: Mapping[str, object], key: str):
    if key not in values:
        raise ValueError(f"posegen profile is missing '{key}'")
    return values[key]
