"""Sensor-independent point-cloud acquisition values."""

from dataclasses import dataclass
from typing import Protocol

from .types import PointCloudData


@dataclass(frozen=True)
class CapturedPointCloud:
    cloud: PointCloudData
    frame_id: str
    timestamp_ns: int
    sensor_id: str

    def __post_init__(self) -> None:
        if not self.frame_id or not self.sensor_id:
            raise ValueError("captured cloud frame and sensor IDs are required")
        if self.timestamp_ns <= 0:
            raise ValueError("captured cloud timestamp must be positive")


class PointCloudSource(Protocol):
    def capture(self) -> CapturedPointCloud: ...
