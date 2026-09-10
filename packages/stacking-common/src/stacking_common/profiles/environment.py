"""Loading of the resolved platform environment profile."""

from collections.abc import Mapping
from pathlib import Path

import yaml

from stacking_common.environment import EnvironmentProfile


def load_environment_profile(path: str | Path) -> EnvironmentProfile:
    """Load one explicitly resolved SI-unit environment profile."""

    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"missing environment profile: {profile_path}"
        )
    values = yaml.safe_load(profile_path.read_text())
    if not isinstance(values, Mapping):
        raise TypeError("environment profile root must be a mapping")
    return EnvironmentProfile.from_mapping(values)
