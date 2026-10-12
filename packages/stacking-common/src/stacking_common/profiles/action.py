"""Loading of the resolved platform action-generation profile."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from stacking_common.actions import (
    PlacementActionGeneratorConfig,
    PlanarSamplingConfig,
)


@dataclass(frozen=True)
class ActionGenerationProfile:
    """Common action settings resolved for the platform platform."""

    generator: PlacementActionGeneratorConfig
    planar: PlanarSamplingConfig
    strict_floor_first: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.strict_floor_first, bool):
            raise TypeError("strict_floor_first must be boolean")


def load_action_generation_profile(
    path: str | Path,
    *,
    max_poses_per_object: int,
    banned_object_ids: Iterable[int],
) -> ActionGenerationProfile:
    """Load explicit SI-unit action-generation values."""

    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"missing action-generation profile: {profile_path}"
        )
    values = yaml.safe_load(profile_path.read_text())
    if not isinstance(values, Mapping):
        raise TypeError("action-generation profile root must be a mapping")
    if values.get("units") != "si":
        raise ValueError("action-generation profile units must be 'si'")
    initialization = _mapping(values, "initialization")
    filtering = _mapping(values, "filtering")
    planar = _mapping(values, "planar")
    orientation = _mapping(values, "orientation")
    if "yaw_angles_degrees" in planar:
        raise ValueError("move planar.yaw_angles_degrees to orientation.yaw_angles_degrees")
    return ActionGenerationProfile(
        strict_floor_first=filtering.get("strict_floor_first", False),
        generator=PlacementActionGeneratorConfig(
            max_poses_per_object=max_poses_per_object,
            banned_object_ids=frozenset(int(value) for value in banned_object_ids),
            require_solver_convergence=filtering.get("require_solver_convergence", True),
            reject_equilibrium_at_threshold=filtering.get("reject_equilibrium_at_threshold", False),
            initial_height_clearance=float(
                _required(initialization, "height_clearance")
            ),
            initial_height_xy_padding=float(_required(initialization, "xy_padding")),
            posegen_gap_threshold=float(
                _required(filtering, "posegen_gap_threshold")
            ),
            posegen_equilibrium_threshold=float(
                _required(filtering, "posegen_equilibrium_threshold")
            ),
        ),
        planar=PlanarSamplingConfig(
            grid_shape=tuple(int(value) for value in _required(planar, "grid_shape")),
            yaw_angles=tuple(
                float(np.deg2rad(value))
                for value in _required(orientation, "yaw_angles_degrees")
            ),
            target_anchor_inset_scale=(
                None
                if planar.get("target_anchor_inset_scale") is None
                else float(planar["target_anchor_inset_scale"])
            ),
        ),
    )


def _mapping(values: Mapping[str, object], key: str) -> Mapping[str, object]:
    result = _required(values, key)
    if not isinstance(result, Mapping):
        raise TypeError(f"action-generation profile '{key}' must be a mapping")
    return result


def resolve_scored_sampling_settings(planar: Mapping[str, object]) -> dict:
    """Extract the shared scored sampler's explicit platform settings."""
    if "floor_fill" in planar:
        raise ValueError("merge floor_fill into planar.floor; use one floor configuration")
    floor = dict(_mapping(planar, "floor"))
    for key in ("coverage_mode", "bottom_tolerance", "fill_ratio", "grid_size",
                "footprint_scale"):
        _required(floor, key)
    obsolete = {key for key in floor if key in {"connected_after", "minimum_contact_cells", "max_layers"}
                or key.startswith(("lower_floor_", "u_shape_"))}
    if obsolete:
        raise ValueError(f"obsolete planar.floor settings: {sorted(obsolete)}")
    score = dict(_mapping(planar, "score"))
    obsolete_scores = {"open_area", "frontier"} & score.keys()
    if obsolete_scores:
        raise ValueError(f"obsolete planar.score settings: {sorted(obsolete_scores)}")
    return {
        **{key: _required(planar, key) for key in (
            "boltzmann_temperature", "boltzmann_min_probability",
            "target_mask_margin", "contact_gap_tolerance")},
        "score": score,
        "floor": floor,
    }


def _required(values: Mapping[str, object], key: str):
    if key not in values:
        raise ValueError(f"action-generation profile is missing '{key}'")
    return values[key]
