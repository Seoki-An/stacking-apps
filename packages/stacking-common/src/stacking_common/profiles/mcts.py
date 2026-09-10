"""Loading of resolved platform MCTS and root-proposal settings."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from stacking_common.actions import CEMRootConfig
from stacking_common.search import (
    EvaluationFidelity,
    FinalSelectionConfig,
    MCTSConfig,
    MCTSSearchRequest,
    ProgressiveWideningConfig,
)


@dataclass(frozen=True)
class MCTSParallelConfig:
    """Platform CPU allocation for independent native search workers."""

    enabled: bool
    workers: int

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("MCTS parallel enabled flag must be boolean")
        if not isinstance(self.workers, int) or self.workers <= 0:
            raise ValueError("MCTS parallel worker count must be positive")

    @property
    def effective_workers(self) -> int:
        return self.workers if self.enabled else 1


@dataclass(frozen=True)
class MCTSProfile:
    tree: MCTSConfig
    request: MCTSSearchRequest
    root_cem: CEMRootConfig
    parallel: MCTSParallelConfig
    bootstrap_mean_reward: float
    bootstrap_horizon: int


def load_mcts_profile(path: str | Path) -> MCTSProfile:
    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(f"missing MCTS profile: {profile_path}")
    values = yaml.safe_load(profile_path.read_text())
    if not isinstance(values, Mapping):
        raise TypeError("MCTS profile root must be a mapping")
    if values.get("units") != "si":
        raise ValueError("MCTS profile units must be 'si'")
    tree = _mapping(values, "tree")
    widening = _mapping(values, "widening")
    generation = _mapping(values, "generation")
    validation = _mapping(values, "validation")
    final_selection = _mapping(values, "final_selection")
    parallel = _mapping(values, "parallel")
    cem = _mapping(values, "root_cem")
    bootstrap = _mapping(values, "bootstrap")
    discount = float(_required(tree, "discount"))
    config = MCTSConfig(
        max_depth=int(_required(tree, "max_depth")),
        max_children=int(_required(tree, "max_children")),
        root_max_children=int(_required(tree, "root_max_children")),
        exploration_constant=float(_required(tree, "exploration_constant")),
        discount=discount,
        failure_reward=float(_required(tree, "failure_reward")),
        expansion_attempt_multiplier=float(
            _required(generation, "attempt_multiplier")
        ),
        generation_batch_size=int(_required(generation, "batch_size")),
        expansion_retry_batches=int(
            _required(generation, "expansion_retry_batches")
        ),
        min_feasible_children=int(
            _required(generation, "min_feasible_children")
        ),
        evaluation_batch_size=int(
            _required(generation, "evaluation_batch_size")
        ),
        final_validation_candidates=int(
            _required(validation, "max_candidates")
        ),
        initial_depth1_steps=int(_required(tree, "initial_depth1_steps")),
        validate_root_before_search=bool(
            _required(validation, "root_before_search")
        ),
        root_lookahead=bool(_required(tree, "root_lookahead")),
        rejected_root_retry_rounds=int(
            _required(validation, "rejected_root_retry_rounds")
        ),
        final_selection=FinalSelectionConfig(
            mode=str(_required(final_selection, "mode")),
            max_weight=float(_required(final_selection, "max_weight")),
            validate_descendant_path=bool(
                _required(final_selection, "validate_descendant_path")
            ),
            validate_descendant_depth=int(
                _required(final_selection, "validate_descendant_depth")
            ),
            require_validated_descendant=bool(
                _required(final_selection, "require_validated_descendant")
            ),
        ),
        widening=ProgressiveWideningConfig(
            enabled=bool(_required(widening, "enabled")),
            coefficient=float(_required(widening, "coefficient")),
            exponent=float(_required(widening, "exponent")),
        ),
    )
    request = MCTSSearchRequest(
        iterations=int(_required(tree, "iterations")),
        evaluation_fidelity=EvaluationFidelity.SHORT,
        max_time_seconds=float(_required(tree, "max_search_time")),
        simulation_depth=int(_required(tree, "simulation_depth")),
        beyond_depth_fidelity=EvaluationFidelity.LONG,
        final_validation_fidelity=EvaluationFidelity.LONG,
        final_validation_reserve_seconds=float(
            _required(validation, "time_reserve")
        ),
        root_sampling_priority=str(
            _required(generation, "root_sampling_priority")
        ),
        deeper_sampling_priority=str(
            _required(generation, "deeper_sampling_priority")
        ),
        max_revalidation_drift=float(
            _required(validation, "max_state_drift")
        ),
    )
    return MCTSProfile(
        tree=config,
        request=request,
        root_cem=CEMRootConfig(
            population=int(_required(cem, "population")),
            elite=int(_required(cem, "elite")),
            iterations=int(_required(cem, "iterations")),
            keep=int(_required(cem, "keep")),
            mutation_xy_std=float(_required(cem, "mutation_xy_std")),
            mutation_yaw_std=float(
                np.deg2rad(_required(cem, "mutation_yaw_std_degrees"))
            ),
            fresh_fraction=float(_required(cem, "fresh_fraction")),
            duplicate_xy_threshold=float(
                _required(cem, "duplicate_xy_threshold")
            ),
            duplicate_yaw_threshold=float(
                np.deg2rad(_required(cem, "duplicate_yaw_threshold_degrees"))
            ),
            max_per_object=int(_required(cem, "max_per_object")),
            height_map_score_weight=float(
                _required(cem, "height_map_score_weight")
            ),
            seed=int(_required(cem, "seed")),
        ),
        parallel=MCTSParallelConfig(
            enabled=_required(parallel, "enabled"),
            workers=int(_required(parallel, "workers")),
        ),
        bootstrap_mean_reward=float(_required(bootstrap, "mean_reward")),
        bootstrap_horizon=int(_required(bootstrap, "horizon")),
    )


def _mapping(values: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = _required(values, key)
    if not isinstance(value, Mapping):
        raise TypeError(f"MCTS profile '{key}' must be a mapping")
    return value


def _required(values: Mapping[str, object], key: str):
    if key not in values:
        raise ValueError(f"MCTS profile is missing '{key}'")
    return values[key]
