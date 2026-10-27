"""Loading of the resolved platform simulation profile."""

from collections.abc import Mapping
from pathlib import Path

import yaml

from stacking_common.environment import CoreSimulationProfile


def load_simulation_profile(
    path: str | Path,
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
    profile = CoreSimulationProfile.from_mapping(values)
    if profile.placement.short_gate is not None:
        raise ValueError("short_structural was removed from runtime profiles; "
                         "configure one phases.structural after phases.settling")
    return profile
