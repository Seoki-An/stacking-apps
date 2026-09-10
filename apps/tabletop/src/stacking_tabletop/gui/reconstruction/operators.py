"""Thread-safe requests connecting numerical stages to the embedded viewport."""

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

import numpy as np
from PySide6.QtCore import QObject, Signal

from stacking_common.pointcloud import PointCloudData
from stacking_common.reconstruction import ReconstructionResult
from stacking_tabletop.config import TabletopPerceptionProfile
from stacking_tabletop.perception.extraction import ManualObjectSelection
from stacking_tabletop.perception.in_hand import InHandIcpInspection
from stacking_tabletop.perception.reconstruction import (
    _principal_axis_aligned_points,
)


@dataclass
class ViewportRequest:
    kind: str
    payload: object
    completed: Event = field(default_factory=Event)
    result: object = None
    error: BaseException | None = None

    def resolve(self, value: object) -> None:
        self.result = value
        self.completed.set()

    def reject(self, error: BaseException) -> None:
        self.error = error
        self.completed.set()


class WorkbenchOperators(QObject):
    """Implement platform operator contracts through one Qt viewport."""

    interaction_requested = Signal(object)

    def __init__(self, profile: TabletopPerceptionProfile) -> None:
        super().__init__()
        self.profile = profile

    def select(self, merged: PointCloudData) -> ManualObjectSelection:
        return self._request("object_selection", merged)

    def initialize(
        self,
        source: PointCloudData,
        target: PointCloudData,
        source_index: int,
    ) -> np.ndarray:
        return self._request(
            "alignment",
            (source, target, source_index),
        )

    def inspect_icp(self, case: InHandIcpInspection) -> str:
        return str(self._request("icp_inspection", case))

    def remove(
        self,
        cloud: PointCloudData,
        _left_clicking: bool = False,
    ) -> PointCloudData:
        aligned = _principal_axis_aligned_points(cloud.points)
        centers = np.asarray(
            self._request("cleanup", PointCloudData(aligned)),
            dtype=np.float64,
        ).reshape(-1, 3)
        if not len(centers):
            return cloud
        cleanup = self.profile.reconstruction.cleanup
        squared = np.sum(
            (aligned[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )
        remove = np.min(squared, axis=1) <= cleanup.manual_remove_distance**2
        ratio = float(np.mean(remove))
        if ratio > cleanup.manual_max_remove_ratio:
            raise ValueError(
                f"manual selection would remove {ratio:.1%} of the cloud; "
                f"maximum is {cleanup.manual_max_remove_ratio:.1%}"
            )
        return PointCloudData(cloud.points[~remove])

    def split_seeds(self, cloud: PointCloudData) -> np.ndarray:
        return np.asarray(
            self._request("partition", cloud),
            dtype=np.float64,
        ).reshape(-1, 3)

    def review(self, result: ReconstructionResult) -> bool:
        return bool(self._request("review", result))

    def continue_prepare(
        self,
        completed: int,
        total: int,
        next_capture: Path,
    ) -> bool:
        return bool(
            self._request(
                "prepare_continue",
                (completed, total, next_capture),
            )
        )

    def _request(self, kind: str, payload: object):
        request = ViewportRequest(kind, payload)
        self.interaction_requested.emit(request)
        request.completed.wait()
        if request.error is not None:
            raise request.error
        return request.result
