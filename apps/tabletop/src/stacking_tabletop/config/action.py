"""Tabletop orientation and scored-sampling settings around common actions."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from stacking_common.profiles.action import (
    ActionGenerationProfile,
    load_action_generation_profile as _load,
    resolve_scored_sampling_settings,
)


@dataclass(frozen=True)
class TabletopActionGenerationProfile(ActionGenerationProfile):
    # Scan alignment is the historical default; record explicit departures.
    pose_from_scan: bool = field(default=True, metadata={"omit_default_from_context": True})
    # Record every scored-sampling setting so resume detects algorithm changes.
    sampling_settings: dict = field(kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.pose_from_scan, bool):
            raise TypeError("orientation.pose_from_scan must be boolean")


def load_action_generation_profile(path, *, max_poses_per_object, banned_object_ids):
    profile = _load(path, max_poses_per_object=max_poses_per_object, banned_object_ids=banned_object_ids)
    values = yaml.safe_load(Path(path).read_text())
    orientation = values.get("orientation", {})
    if not isinstance(orientation, dict):
        raise TypeError("orientation must be a mapping")
    return TabletopActionGenerationProfile(profile.generator, profile.planar, profile.strict_floor_first,
        orientation.get("pose_from_scan", True),
        construction=profile.construction,
        sampling_settings=resolve_scored_sampling_settings(values["planar"]))


def resolve_pose_from_scan(configured, override, recorded=None):
    if recorded is not None:
        if override is not None and override != recorded:
            raise ValueError(f"resume preserves recorded pose_from_scan={recorded}; start a new run to change it")
        return recorded
    return configured if override is None else override
