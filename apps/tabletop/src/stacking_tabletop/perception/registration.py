"""Open3D implementation of common point-cloud registration contracts."""

import numpy as np

from stacking_common.pointcloud import (
    Open3dRegistrationBackend,
    PointCloudData,
    RegistrationDecision,
    RegistrationQualityConfig,
    RegistrationResult,
    evaluate_registration,
)

from .catalog import TabletopPcdCatalog

class TabletopObjectRegistrar:
    """Register a configured tabletop object model to observed XYZ points."""

    def __init__(
        self,
        catalog: TabletopPcdCatalog,
        backend: Open3dRegistrationBackend,
        quality: RegistrationQualityConfig | None = None,
    ) -> None:
        self.catalog = catalog
        self.backend = backend
        self.quality = quality

    def register(
        self,
        object_id: int,
        observation: PointCloudData,
        initial_transform: np.ndarray,
    ) -> RegistrationResult:
        return self.backend.register(
            self.catalog.load(object_id),
            observation,
            initial_transform,
        )

    def register_with_decision(
        self,
        object_id: int,
        observation: PointCloudData,
        initial_transform: np.ndarray,
    ) -> tuple[RegistrationResult, RegistrationDecision]:
        if self.quality is None:
            raise RuntimeError("registrar has no quality-gate profile")
        result = self.register(object_id, observation, initial_transform)
        return result, evaluate_registration(
            result,
            initial_transform,
            self.quality,
        )
