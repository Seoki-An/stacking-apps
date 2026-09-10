"""Loading of the resolved platform simulation profile."""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import yaml

from stacking_common.environment import CoreSimulationProfile


def load_simulation_profile(
    path: str | Path,
    *,
    fidelity: str = "long",
) -> CoreSimulationProfile:
    """Load stacking-core simulation settings from one SI-unit profile."""

    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"missing simulation profile: {profile_path}"
        )
    values = yaml.safe_load(profile_path.read_text())
    if not isinstance(values, Mapping):
        raise TypeError("simulation profile root must be a mapping")
    if fidelity not in {"short", "long"}:
        raise ValueError("simulation fidelity must be 'short' or 'long'")
    profile = CoreSimulationProfile.from_mapping(values)
    if fidelity == "short":
        short = profile.placement.short_gate
        if short is None:
            raise ValueError("simulation profile is missing 'short_structural'")
        profile = replace(
            profile,
            placement=replace(profile.placement, structural=short, short_gate=None),
        )
    return profile
