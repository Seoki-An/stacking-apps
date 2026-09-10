"""Common action settings plus tabletop's scan-orientation selection."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from stacking_common.profiles.action import ActionGenerationProfile, load_action_generation_profile as _load


@dataclass(frozen=True)
class TabletopActionGenerationProfile(ActionGenerationProfile):
    # Older tabletop runs always used scan alignment. Omit that default from
    # contexts so their saved hashes remain valid when resuming.
    pose_from_scan: bool = field(default=True, metadata={"omit_default_from_context": True})

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.pose_from_scan, bool):
            raise TypeError("orientation.pose_from_scan must be boolean")


def load_action_generation_profile(path, *, max_poses_per_object, banned_object_ids):
    profile = _load(path, max_poses_per_object=max_poses_per_object, banned_object_ids=banned_object_ids)
    orientation = yaml.safe_load(Path(path).read_text()).get("orientation", {})
    if not isinstance(orientation, dict):
        raise TypeError("orientation must be a mapping")
    return TabletopActionGenerationProfile(profile.generator, profile.planar, profile.strict_floor_first,
                                          orientation.get("pose_from_scan", True))


def resolve_pose_from_scan(configured, override, recorded=None):
    if recorded is not None:
        if override is not None and override != recorded:
            raise ValueError(f"resume preserves recorded pose_from_scan={recorded}; start a new run to change it")
        return recorded
    return configured if override is None else override
