"""Typed common contract for a resolved environment profile."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from .observation import ObservationConfig
from .reward import RewardConfig, RewardWeights
from .robustness import PlaceStabilityConfig


@dataclass(frozen=True)
class EnvironmentProfile:
    """Resolved observation and reward settings for one platform."""

    observation: ObservationConfig
    surface_sample_count: int
    height_map_margin: float
    overlap_resolution: float
    reward: RewardConfig
    place_stability: PlaceStabilityConfig = field(default_factory=PlaceStabilityConfig)

    def __post_init__(self) -> None:
        if self.surface_sample_count <= 0:
            raise ValueError("surface sample count must be positive")
        if not np.isfinite(self.height_map_margin) or self.height_map_margin < 0.0:
            raise ValueError("height-map margin must be finite and non-negative")
        if (
            not np.isfinite(self.overlap_resolution)
            or self.overlap_resolution <= 0.0
        ):
            raise ValueError("overlap resolution must be positive and finite")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "EnvironmentProfile":
        """Parse a resolved mapping without applying implicit scale rules."""

        units = values.get("units")
        if units != "si":
            raise ValueError("environment profile units must be 'si'")
        observation = _mapping(values, "observation")
        reward = _mapping(values, "reward")
        weights = _mapping(reward, "weights")

        observation_config = ObservationConfig(
            max_poses_per_object=int(
                _required(observation, "max_poses_per_object")
            ),
            height_map_resolution=tuple(
                int(value)
                for value in _required(observation, "height_map_resolution")
            ),
            render_height_map=_boolean(
                observation, "render_height_map"
            ),
            action_sample_count=int(
                _required(observation, "action_sample_count")
            ),
            banned_object_ids=frozenset(
                int(value)
                for value in observation.get("banned_object_ids", [])
            ),
        )
        reward_config = RewardConfig(
            posegen_threshold=float(_required(reward, "posegen_threshold")),
            containment_threshold=float(
                _required(reward, "containment_threshold")
            ),
            nonfinite_displacement=float(
                _required(reward, "nonfinite_displacement")
            ),
            max_displacement=_optional_float(reward, "max_displacement"),
            weights=RewardWeights(
                surface_flatness=float(weights.get("surface_flatness", 0.0)),
                stability=float(_required(weights, "stability")),
                object_containment=float(
                    _required(weights, "object_containment")
                ),
                target_overlap_increment=float(
                    _required(weights, "target_overlap_increment")
                ),
                place_stability=float(
                    _required(weights, "place_stability")
                ),
                large_object_lower=float(
                    _required(weights, "large_object_lower")
                ),
                inward_orientation=float(
                    _required(weights, "inward_orientation")
                ),
            ),
        )
        return cls(
            observation=observation_config,
            surface_sample_count=int(
                _required(observation, "surface_sample_count")
            ),
            height_map_margin=float(
                _required(observation, "height_map_margin")
            ),
            overlap_resolution=float(
                _required(reward, "overlap_resolution")
            ),
            reward=reward_config,
            place_stability=PlaceStabilityConfig.from_mapping(reward.get("place_stability", {})),
        )


def _mapping(values: Mapping[str, object], key: str) -> Mapping[str, object]:
    result = _required(values, key)
    if not isinstance(result, Mapping):
        raise TypeError(f"environment profile '{key}' must be a mapping")
    return result


def _required(values: Mapping[str, object], key: str):
    if key not in values:
        raise ValueError(f"environment profile is missing '{key}'")
    return values[key]


def _boolean(values: Mapping[str, object], key: str) -> bool:
    value = _required(values, key)
    if not isinstance(value, bool):
        raise TypeError(f"environment profile '{key}' must be boolean")
    return value


def _optional_float(values: Mapping[str, object], key: str) -> float | None:
    value = values.get(key)
    return None if value is None else float(value)
