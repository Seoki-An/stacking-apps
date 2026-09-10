"""Incremental registration against accepted views in one common frame."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .processing import voxel_downsample
from .registration import (
    PointCloudRegistration,
    RegistrationDecision,
    RegistrationQualityConfig,
    RegistrationResult,
    evaluate_registration,
)
from .types import PointCloudData


@dataclass(frozen=True)
class IncrementalAlignment:
    source: PointCloudData
    cloud: PointCloudData
    correction: np.ndarray
    decision: RegistrationDecision
    registration: RegistrationResult | None


class IncrementalRegistration:
    """Propose, accept/reject, and undo views without sensor or operator code.

    Inputs already share a nominal coordinate frame. Rejected views never enter
    the reference. Undo reconstructs it in the original voxel-merge order.
    """

    def __init__(
        self,
        backend: PointCloudRegistration,
        quality: RegistrationQualityConfig,
        reference_voxel_size: float,
        *,
        downsample: Callable[[PointCloudData, float], PointCloudData] = voxel_downsample,
    ) -> None:
        if not np.isfinite(reference_voxel_size) or reference_voxel_size <= 0:
            raise ValueError("reference voxel size must be positive")
        self.backend = backend
        self.quality = quality
        self.reference_voxel_size = reference_voxel_size
        self.downsample = downsample
        self.reference: PointCloudData | None = None
        self._views: list[PointCloudData | None] = []
        self._pending: IncrementalAlignment | None = None

    def propose(self, source: PointCloudData) -> IncrementalAlignment:
        identity = np.eye(4)
        correction = identity
        registration = None
        if not len(source):
            decision = RegistrationDecision(False, ("empty_cloud",), 0.0, 0.0)
        elif self.reference is None:
            decision = RegistrationDecision(True, (), 0.0, 0.0)
        else:
            try:
                registration = self.backend.register(source, self.reference, identity)
                decision = evaluate_registration(registration, identity, self.quality)
                if decision.accepted:
                    correction = registration.transform
            except (RuntimeError, ValueError):
                decision = RegistrationDecision(False, ("registration_error",), 0.0, 0.0)
        correction = correction.copy()
        correction.setflags(write=False)
        self._pending = IncrementalAlignment(
            source, source.transformed(correction), correction, decision, registration,
        )
        return self._pending

    def commit(self, proposal: IncrementalAlignment, *, selected: bool) -> None:
        if proposal is not self._pending:
            raise ValueError("alignment proposal is stale or already committed")
        if selected and not proposal.decision.accepted:
            raise ValueError("cannot accept a view that failed registration gates")
        reference = self._merge(self.reference, proposal.cloud) if selected else self.reference
        self._views.append(proposal.cloud if selected else None)
        self.reference = reference
        self._pending = None

    def undo(self) -> None:
        if not self._views:
            raise ValueError("no committed view to undo")
        reference = None
        for cloud in self._views[:-1]:
            if cloud is not None:
                reference = self._merge(reference, cloud)
        self._views.pop()
        self.reference = reference
        self._pending = None

    def _merge(self, reference, cloud):
        sampled = self.downsample(cloud, self.reference_voxel_size)
        if reference is None:
            return sampled
        return self.downsample(
            PointCloudData(np.vstack((reference.points, sampled.points))),
            self.reference_voxel_size,
        )
