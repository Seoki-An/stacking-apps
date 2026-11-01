"""Platform-neutral stacking action generation."""

from .cem import (
    CEMRootActionGenerator,
    CEMRootConfig,
    CEMRoundStatistics,
    CEMStatistics,
)
from .config import PlacementActionGeneratorConfig, PlanarSamplingConfig
from .core_solver import CorePlacementPoseSolver, CorePosegenProfile
from .contracts import (
    PlacementFilter,
    PlacementInitializer,
    PlacementPoseSolver,
    PlanarPoseSampler,
    ObjectOrientationProvider,
)
from .filtering import (
    ActiveLayerAdmissionFilter,
    InitialXYDisplacementFilter,
    PosegenFeasibilityFilter,
    ScanOrientationFilter,
    TargetContainmentFilter,
)
from .floor import ActiveFloorPolicy
from .generator import PlacementActionGenerator
from .initialization import MeshHeightInitializer
from .orientation import (
    InventoryOrientationProvider,
    MappingOrientationProvider,
    ScanAlignedOrientationProvider,
)
from .types import (
    InitializedPlacement,
    PlacementFilterDecision,
    PlacementSolveResult,
    PlanarPlacement,
)

__all__ = [
    "ActiveFloorPolicy",
    "ActiveLayerAdmissionFilter",
    "CEMRootActionGenerator",
    "CEMRootConfig",
    "CEMRoundStatistics",
    "CEMStatistics",
    "CorePlacementPoseSolver",
    "CorePosegenProfile",
    "InitializedPlacement",
    "InitialXYDisplacementFilter",
    "InventoryOrientationProvider",
    "MeshHeightInitializer",
    "MappingOrientationProvider",
    "ScanAlignedOrientationProvider",
    "ScanOrientationFilter",
    "ObjectOrientationProvider",
    "PlacementActionGenerator",
    "PlacementActionGeneratorConfig",
    "PlacementFilter",
    "PlacementFilterDecision",
    "PlacementInitializer",
    "PlacementPoseSolver",
    "PlacementSolveResult",
    "PlanarPlacement",
    "PlanarPoseSampler",
    "PlanarSamplingConfig",
    "PosegenFeasibilityFilter",
    "TargetContainmentFilter",
]
