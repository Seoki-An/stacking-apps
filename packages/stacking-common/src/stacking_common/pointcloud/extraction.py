"""Platform-neutral contracts for extracting an object from scene clouds."""

from typing import Protocol

from .types import PointCloudData


class ObjectViewExtractor(Protocol):
    """Extract object points from one cloud using platform-owned context."""

    def extract(self, cloud: PointCloudData, context: object) -> PointCloudData: ...

