"""Platform-neutral orchestration of a stacking episode."""

from .contracts import (
    HeightMapRenderer,
    ObservationBuilder,
    RewardEvaluator,
    TransitionModel,
)
from .core_backend import (
    CorePlacementBackend,
    CorePlacementConfig,
    CorePhysicsProfile,
    CoreSimulationProfile,
    SimulationPhaseConfig,
)
from .environment import StackingEnvironment
from .observation import (
    ArrayObservationBuilder,
    InventoryObservationAdapter,
    Observation,
    ObservationConfig,
    ObservationGeometryBatch,
)
from .overlap import OverlapEvaluator, VoxelOverlapEvaluator
from .profile import EnvironmentProfile
from .reward import RewardConfig, RewardWeights, StackingReward
from .result import ResetResult, StepResult
from .transition import PlacementBackend, PlacementResult, SimulationTransition

__all__ = [
    "ArrayObservationBuilder",
    "CorePlacementBackend",
    "CorePlacementConfig",
    "CorePhysicsProfile",
    "CoreSimulationProfile",
    "EnvironmentProfile",
    "HeightMapRenderer",
    "InventoryObservationAdapter",
    "Observation",
    "ObservationBuilder",
    "ObservationConfig",
    "ObservationGeometryBatch",
    "OverlapEvaluator",
    "PlacementBackend",
    "PlacementResult",
    "ResetResult",
    "RewardConfig",
    "RewardEvaluator",
    "RewardWeights",
    "SimulationTransition",
    "SimulationPhaseConfig",
    "StackingEnvironment",
    "StackingReward",
    "StepResult",
    "TransitionModel",
    "VoxelOverlapEvaluator",
]
