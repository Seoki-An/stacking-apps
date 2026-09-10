"""Typed configuration for platform-neutral Monte Carlo tree search."""

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FinalSelectionConfig:
    """Choose a root from mean evidence and validated descendant returns."""

    mode: str = "mean"
    max_weight: float = 0.5
    validate_descendant_path: bool = False
    validate_descendant_depth: int = 1
    require_validated_descendant: bool = False

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower().replace("-", "_")
        aliases = {
            "max": "max_descendant",
            "descendant_max": "max_descendant",
            "mix": "mixed",
            "max_mean_mix": "mixed",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"mean", "max_descendant", "mixed"}:
            raise ValueError("unknown MCTS final-selection mode")
        if not math.isfinite(self.max_weight) or not 0.0 <= self.max_weight <= 1.0:
            raise ValueError(
                "MCTS final-selection maximum weight must be between zero and one"
            )
        if (
            not isinstance(self.validate_descendant_depth, int)
            or self.validate_descendant_depth < 0
        ):
            raise ValueError(
                "MCTS descendant-validation depth must be a non-negative integer"
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self, "validate_descendant_path", bool(self.validate_descendant_path)
        )
        object.__setattr__(
            self,
            "require_validated_descendant",
            bool(self.require_validated_descendant),
        )


@dataclass(frozen=True)
class ProgressiveWideningConfig:
    """Limit descendant branching as a function of node visits."""

    enabled: bool = True
    coefficient: float = 2.0
    exponent: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.coefficient) or self.coefficient <= 0.0:
            raise ValueError("widening coefficient must be finite and positive")
        if not math.isfinite(self.exponent) or not 0.0 <= self.exponent <= 1.0:
            raise ValueError("widening exponent must be between zero and one")


@dataclass(frozen=True)
class MCTSConfig:
    """Tree-search mechanics without action, model, or simulator settings."""

    max_depth: int
    max_children: int
    exploration_constant: float
    discount: float
    root_max_children: int | None = None
    expansion_attempt_multiplier: float = 1.0
    failure_reward: float = -1.0
    generation_batch_size: int = 1
    expansion_retry_batches: int = 1
    min_feasible_children: int = 1
    evaluation_batch_size: int = 1
    final_validation_candidates: int = 0
    initial_depth1_steps: int = 0
    validate_root_before_search: bool = False
    root_lookahead: bool = False
    rejected_root_retry_rounds: int = 0
    final_selection: FinalSelectionConfig = field(
        default_factory=FinalSelectionConfig
    )
    widening: ProgressiveWideningConfig = field(
        default_factory=ProgressiveWideningConfig
    )

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("MCTS maximum depth must be non-negative")
        if self.max_children <= 0:
            raise ValueError("MCTS maximum children must be positive")
        if self.root_max_children is not None and self.root_max_children <= 0:
            raise ValueError("MCTS root maximum children must be positive")
        if (
            not math.isfinite(self.exploration_constant)
            or self.exploration_constant < 0.0
        ):
            raise ValueError(
                "MCTS exploration constant must be finite and non-negative"
            )
        if not math.isfinite(self.discount) or not 0.0 <= self.discount <= 1.0:
            raise ValueError("MCTS discount must be between zero and one")
        if (
            not math.isfinite(self.expansion_attempt_multiplier)
            or self.expansion_attempt_multiplier < 1.0
        ):
            raise ValueError(
                "MCTS expansion attempt multiplier must be at least one"
            )
        if not math.isfinite(self.failure_reward):
            raise ValueError("MCTS failure reward must be finite")
        for name, value in (
            ("generation batch size", self.generation_batch_size),
            ("expansion retry batches", self.expansion_retry_batches),
            ("minimum feasible children", self.min_feasible_children),
            ("evaluation batch size", self.evaluation_batch_size),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"MCTS {name} must be a positive integer")
        if (
            not isinstance(self.final_validation_candidates, int)
            or self.final_validation_candidates < 0
        ):
            raise ValueError(
                "MCTS final validation candidate count must be non-negative"
            )
        if (
            not isinstance(self.initial_depth1_steps, int)
            or self.initial_depth1_steps < 0
        ):
            raise ValueError("MCTS initial depth-1 step count must be non-negative")
        if (
            not isinstance(self.rejected_root_retry_rounds, int)
            or self.rejected_root_retry_rounds < 0
        ):
            raise ValueError("MCTS rejected-root retry rounds must be non-negative")
        if not isinstance(self.final_selection, FinalSelectionConfig):
            raise TypeError("MCTS final selection must be a FinalSelectionConfig")
        object.__setattr__(
            self, "validate_root_before_search", bool(self.validate_root_before_search)
        )
        object.__setattr__(self, "root_lookahead", bool(self.root_lookahead))

    def hard_child_limit(self, depth: int) -> int:
        """Return the configured branch limit at a node depth."""

        if depth < 0:
            raise ValueError("MCTS node depth must be non-negative")
        if depth == 0 and self.root_max_children is not None:
            return self.root_max_children
        return self.max_children

    def effective_max_depth(self, placed_count: int) -> int:
        """Return the legacy early-step depth cap for one root state."""

        if not isinstance(placed_count, int) or placed_count < 0:
            raise ValueError("placed object count must be a non-negative integer")
        if self.initial_depth1_steps and placed_count < self.initial_depth1_steps:
            return min(self.max_depth, 1)
        return self.max_depth
