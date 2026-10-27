"""Cross-entropy root proposals built on the common placement generator."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Generic, Mapping, TypeVar

import numpy as np

from stacking_common.domain import Action, State
from stacking_common.search.contracts import PriorScorer
from stacking_common.search.types import (
    ActionCandidate,
    ActionGenerationRequest,
    ActionGenerationResult,
)

from .generator import PlacementActionGenerator
from .types import PlanarPlacement


ObservationT = TypeVar("ObservationT")


@dataclass(frozen=True)
class CEMRootConfig:
    """Resolved controls for root-only cross-entropy proposals."""

    population: int
    elite: int
    iterations: int
    keep: int
    mutation_xy_std: float
    mutation_yaw_std: float
    fresh_fraction: float
    duplicate_xy_threshold: float
    duplicate_yaw_threshold: float
    max_per_object: int | None = None
    height_map_score_weight: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("population", self.population),
            ("elite count", self.elite),
            ("iteration count", self.iterations),
            ("keep count", self.keep),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"CEM {name} must be a positive integer")
        if self.elite > self.population:
            raise ValueError("CEM elite count cannot exceed its population")
        if self.keep > self.population:
            raise ValueError("CEM keep count cannot exceed its population")
        for name, value in (
            ("XY mutation standard deviation", self.mutation_xy_std),
            ("yaw mutation standard deviation", self.mutation_yaw_std),
            ("duplicate XY threshold", self.duplicate_xy_threshold),
            ("duplicate yaw threshold", self.duplicate_yaw_threshold),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"CEM {name} must be finite and non-negative")
        if (
            not math.isfinite(self.fresh_fraction)
            or not 0.0 <= self.fresh_fraction <= 1.0
        ):
            raise ValueError("CEM fresh fraction must be between zero and one")
        if self.max_per_object is not None and self.max_per_object <= 0:
            raise ValueError("CEM maximum per object must be positive")
        if (
            not math.isfinite(self.height_map_score_weight)
            or self.height_map_score_weight < 0.0
        ):
            raise ValueError(
                "CEM height-map score weight must be finite and non-negative"
            )


@dataclass(frozen=True)
class CEMRoundStatistics:
    """Candidate accounting for one initial or mutation CEM round."""

    round_index: int
    attempted: int
    accepted: int
    rejected: int
    duplicate_rejections: int
    retained: int
    mutation_attempts: int = 0
    fresh_attempts: int = 0
    fresh_fallback_attempts: int = 0


@dataclass(frozen=True)
class CEMStatistics:
    """Compact diagnostics for the most recent root CEM proposal."""

    rounds: tuple[CEMRoundStatistics, ...] = ()
    rejection_counts: Mapping[str, int] = MappingProxyType({})
    final_candidates: int = 0
    retry_round: int = 0
    fresh_exhausted: bool = False
    next_fresh_offset: int = 0
    timed_out: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "rounds": [asdict(value) for value in self.rounds],
            "rejection_counts": dict(self.rejection_counts),
            "final_candidates": self.final_candidates,
            "retry_round": self.retry_round,
            "fresh_exhausted": self.fresh_exhausted,
            "next_fresh_offset": self.next_fresh_offset,
            "timed_out": self.timed_out,
        }

    def __post_init__(self) -> None:
        object.__setattr__(self, "rounds", tuple(self.rounds))
        object.__setattr__(
            self,
            "rejection_counts",
            MappingProxyType(dict(self.rejection_counts)),
        )

    @property
    def attempted(self) -> int:
        return sum(round_stats.attempted for round_stats in self.rounds)

    @property
    def rejected(self) -> int:
        return sum(round_stats.rejected for round_stats in self.rounds)

    @property
    def duplicate_rejections(self) -> int:
        return sum(
            round_stats.duplicate_rejections for round_stats in self.rounds
        )


class CEMRootActionGenerator(Generic[ObservationT]):
    """Generate a scored, diverse root pool through CEM mutation."""

    def __init__(
        self,
        generator: PlacementActionGenerator[ObservationT],
        scorer: PriorScorer[ObservationT],
        config: CEMRootConfig,
    ) -> None:
        self.generator = generator
        self.scorer = scorer
        self.config = config
        self.last_statistics = CEMStatistics()
        self.statistics_history: list[CEMStatistics] = []
        self._fresh_start = 0
        self._retry_round = 0
        self._excluded_actions: tuple[Action, ...] = ()

    def reset_statistics(self) -> None:
        """Clear diagnostics before a new root search."""

        self.last_statistics = CEMStatistics()
        self.statistics_history.clear()
        self._fresh_start = 0
        self._retry_round = 0
        self._excluded_actions = ()

    def begin_retry(self, excluded_actions) -> None:
        """Start a deterministic diversified pool excluding earlier roots."""

        self._retry_round += 1
        self._fresh_start = (
            0 if self.last_statistics.fresh_exhausted
            else self.last_statistics.next_fresh_offset
        )
        self._excluded_actions = tuple(
            action.copy()
            for action in excluded_actions
            if isinstance(action, Action)
        )

    def generate(
        self,
        state: State,
        observation: ObservationT,
        request: ActionGenerationRequest,
    ) -> ActionGenerationResult:
        rejection_counts: Counter[str] = Counter()
        round_statistics: list[CEMRoundStatistics] = []
        def expired() -> bool:
            return request.deadline is not None and time.monotonic() >= request.deadline
        initial = (
            ActionGenerationResult((), attempts=0, exhausted=False)
            if expired() else self._initial_pool(state, observation, request)
        )
        fresh_offset = self._fresh_start + initial.attempts
        fresh_exhausted = initial.exhausted
        rejection_counts.update(initial.rejection_counts)
        pool, duplicates = self._deduplicate(initial.candidates)
        pool = self._rank(state, observation, pool)[: self.config.population]
        initial_rejected = sum(initial.rejection_counts.values())
        if initial.attempts > len(initial.candidates) + initial_rejected:
            unclassified = (
                initial.attempts - len(initial.candidates) - initial_rejected
            )
            rejection_counts["unclassified_generation"] += unclassified
            initial_rejected += unclassified
        round_statistics.append(
            CEMRoundStatistics(
                round_index=0,
                attempted=initial.attempts,
                accepted=len(initial.candidates),
                rejected=initial_rejected,
                duplicate_rejections=duplicates,
                retained=len(pool),
                fresh_attempts=initial.attempts,
            )
        )
        rng = np.random.default_rng(self.config.seed + self._retry_round)
        for iteration in range(1, self.config.iterations):
            if expired():
                break
            ranked = self._rank(state, observation, pool)
            parents = self._elite_parents(ranked, state)
            if not parents and fresh_exhausted:
                break
            fresh_count = self.config.population if not parents else int(
                round(self.config.population * self.config.fresh_fraction)
            )
            mutate_count = max(self.config.population - fresh_count, 0) if parents else 0
            mutations = [
                self._mutation_proposal(
                    rng,
                    parents[int(rng.integers(len(parents)))],
                    observation,
                )
                for _ in range(mutate_count)
            ]
            built = self.generator.build_candidates(state, mutations)
            candidates = [
                candidate for candidate, _ in built if candidate is not None
            ]
            round_rejections: Counter[str] = Counter(
                reason for _, reason in built if reason is not None
            )
            mutation_accounted = len(candidates) + sum(
                round_rejections.values()
            )
            if len(mutations) > mutation_accounted:
                round_rejections["unclassified_generation"] += (
                    len(mutations) - mutation_accounted
                )
            fresh_attempts = 0
            fresh_fallback_attempts = 0
            if fresh_count and not expired():
                if fresh_exhausted:
                    fresh = ActionGenerationResult((), attempts=0, exhausted=True)
                else:
                    fresh = self.generator.generate(
                        state, observation,
                        ActionGenerationRequest(
                            max_candidates=fresh_count,
                            sampling_priority=request.sampling_priority,
                            attempt_offset=fresh_offset,
                            deadline=request.deadline,
                        ),
                    )
                candidates.extend(fresh.candidates)
                fresh_attempts = fresh.attempts
                fresh_offset += fresh.attempts
                fresh_exhausted = fresh.exhausted
                round_rejections.update(fresh.rejection_counts)
                accounted = len(fresh.candidates) + sum(
                    fresh.rejection_counts.values()
                )
                if fresh.attempts > accounted:
                    round_rejections["unclassified_generation"] += (
                        fresh.attempts - accounted
                    )
                if parents and not expired():
                    fresh_fallback_attempts = max(fresh_count - fresh.attempts, 0)
                if fresh_fallback_attempts:
                    fallback_proposals = [
                        self._mutation_proposal(
                            rng,
                            parents[int(rng.integers(len(parents)))],
                            observation,
                            source="cem_fresh_fallback",
                        )
                        for _ in range(fresh_fallback_attempts)
                    ]
                    fallback = self.generator.build_candidates(
                        state, fallback_proposals
                    )
                    candidates.extend(
                        candidate
                        for candidate, _ in fallback
                        if candidate is not None
                    )
                    round_rejections.update(
                        reason
                        for _, reason in fallback
                        if reason is not None
                    )
            combined = [*ranked, *candidates]
            pool, duplicates = self._deduplicate(combined)
            pool = self._rank(state, observation, pool)[: self.config.population]
            rejection_counts.update(round_rejections)
            round_statistics.append(
                CEMRoundStatistics(
                    round_index=iteration,
                    attempted=(
                        len(mutations)
                        + fresh_attempts
                        + fresh_fallback_attempts
                    ),
                    accepted=len(candidates),
                    rejected=sum(round_rejections.values()),
                    duplicate_rejections=duplicates,
                    retained=len(pool),
                    mutation_attempts=len(mutations),
                    fresh_attempts=fresh_attempts,
                    fresh_fallback_attempts=fresh_fallback_attempts,
                )
            )

        ranked = self._rank(state, observation, pool)
        # Exclude downstream failures before the keep/per-object caps, so they
        # cannot crowd out eligible alternatives. Keep round acceptance counts
        # separate from this final eligibility filter.
        eligible = []
        for candidate in ranked:
            reason = request.execution_rejections.reason(candidate.action)
            if reason is None:
                eligible.append(candidate)
            else:
                rejection_counts[reason] += 1
        ranked = eligible
        excluded = sum(
            any(
                _duplicate(
                    candidate.action,
                    action,
                    self.config.duplicate_xy_threshold,
                    self.config.duplicate_yaw_threshold,
                )
                for action in self._excluded_actions
            )
            for candidate in ranked
        )
        if excluded:
            rejection_counts["retry_excluded_duplicate"] += excluded
            ranked = [
                candidate
                for candidate in ranked
                if not any(
                    _duplicate(
                        candidate.action,
                        action,
                        self.config.duplicate_xy_threshold,
                        self.config.duplicate_yaw_threshold,
                    )
                    for action in self._excluded_actions
                )
            ]
        ranked = self._limit_per_object(
            ranked, self._object_limit(state, self.config.keep)
        )[: self.config.keep]
        self.last_statistics = CEMStatistics(
            rounds=tuple(round_statistics),
            rejection_counts=rejection_counts,
            final_candidates=len(ranked),
            retry_round=self._retry_round,
            fresh_exhausted=fresh_exhausted,
            next_fresh_offset=fresh_offset,
            timed_out=expired(),
        )
        self.statistics_history.append(self.last_statistics)
        start = min(request.attempt_offset, len(ranked))
        selected = ranked[start : start + request.max_candidates]
        return ActionGenerationResult(
            candidates=tuple(selected),
            attempts=len(selected),
            exhausted=start + len(selected) >= len(ranked),
        )

    def _initial_pool(self, state, observation, request):
        return self.generator.generate(
            state,
            observation,
            ActionGenerationRequest(
                max_candidates=self.config.population,
                sampling_priority=request.sampling_priority,
                attempt_offset=self._fresh_start,
                deadline=request.deadline,
            ),
        )

    def _mutation_proposal(
        self,
        rng,
        parent,
        observation,
        *,
        source="cem_mutation",
    ):
        action = parent.action
        pose = action.init_pose.copy()
        if self.config.mutation_xy_std > 0.0:
            pose[:2] += rng.normal(0.0, self.config.mutation_xy_std, size=2)
        yaw = float(rng.normal(0.0, self.config.mutation_yaw_std))
        pose[3:] = _quaternion_multiply(_yaw_quaternion(yaw), pose[3:])
        score = float(parent.diagnostics.get("planar_score", 0.0))
        diagnostics = {
            **parent.diagnostics,
            "source": source,
        }
        region_metadata = getattr(self.generator.planar_sampler, "region_metadata", None)
        if callable(region_metadata):
            diagnostics.update(region_metadata(action.object_index, pose))
        score_at_pose = getattr(
            self.generator.planar_sampler, "score_at_pose", None
        )
        score_at_object_pose = getattr(self.generator.planar_sampler, "score_at_object_pose", None)
        if callable(score_at_object_pose):
            resolved_score = score_at_object_pose(
                observation, pose, action.object_index)
        elif callable(score_at_pose):
            resolved_score = score_at_pose(observation, pose)
        else:
            resolved_score = None
        if resolved_score is not None:
            score, normalized = resolved_score
            diagnostics.update(
                {
                    "height_map_score": float(score),
                    "height_map_score_normalized": float(normalized),
                    "height_map_value": -float(score),
                }
            )
        proposal = PlanarPlacement(
            pose=pose,
            score=score,
            diagnostics=diagnostics,
        )
        return action.object_index, proposal

    def _rank(self, state, observation, candidates):
        if not candidates:
            return []
        base_scores = [
            float(value)
            for value in self.scorer.score(state, observation, candidates)
        ]
        if len(base_scores) != len(candidates):
            raise ValueError("CEM scorer returned the wrong number of scores")
        planar_scores = np.asarray(
            [
                float(candidate.diagnostics.get("planar_score", 0.0))
                for candidate in candidates
            ],
            dtype=float,
        )
        normalized = np.full(len(candidates), np.nan, dtype=float)
        for index, candidate in enumerate(candidates):
            explicit = candidate.diagnostics.get(
                "height_map_score_normalized"
            )
            if explicit is not None and math.isfinite(float(explicit)):
                normalized[index] = float(np.clip(float(explicit), 0.0, 1.0))
        unresolved = ~np.isfinite(normalized)
        finite = np.isfinite(planar_scores)
        normalizable = finite & unresolved
        if np.any(normalizable):
            low = float(np.min(planar_scores[finite]))
            high = float(np.max(planar_scores[finite]))
            if high > low:
                normalized[normalizable] = (
                    planar_scores[normalizable] - low
                ) / (high - low)
            else:
                normalized[normalizable] = 0.0
        normalized[~np.isfinite(normalized)] = 0.0
        scores = np.asarray(base_scores, dtype=float) + (
            self.config.height_map_score_weight * normalized
        )
        return [
            ActionCandidate(
                candidate.action,
                diagnostics={
                    **candidate.diagnostics,
                    "cem_base_score": float(base_score),
                    "cem_height_map_score_normalized": float(planar_score),
                    "cem_height_map_score_bonus": float(
                        self.config.height_map_score_weight * planar_score
                    ),
                    "cem_score": float(score),
                },
            )
            for candidate, base_score, planar_score, score in sorted(
                zip(candidates, base_scores, normalized, scores),
                key=lambda values: values[3],
                reverse=True,
            )
            if math.isfinite(score)
        ]

    def _deduplicate(self, candidates):
        kept = []
        duplicate_count = 0
        for candidate in candidates:
            if any(
                _duplicate(
                    candidate.action,
                    value.action,
                    self.config.duplicate_xy_threshold,
                    self.config.duplicate_yaw_threshold,
                )
                for value in kept
            ):
                duplicate_count += 1
                continue
            kept.append(candidate)
        return kept, duplicate_count

    def _object_limit(self, state, target_count):
        """Keep the configured diversity cap until too few stones remain."""
        if self.config.max_per_object is None:
            return None
        remaining = len(state.object_ids) - len(state.placed_object_indices)
        if remaining == 0:
            return 0
        return max(self.config.max_per_object, math.ceil(target_count / remaining))

    def _limit_per_object(self, candidates, limit):
        if limit is None:
            return candidates
        counts = {}
        kept = []
        for candidate in candidates:
            object_id = candidate.action.object_id
            if counts.get(object_id, 0) >= limit:
                continue
            counts[object_id] = counts.get(object_id, 0) + 1
            kept.append(candidate)
        return kept

    def _elite_parents(self, ranked, state):
        """Keep region diversity in exploration; final ranking is unchanged."""
        limit = self._object_limit(state, self.config.elite)
        if not any(item.diagnostics.get("target_region_id") is not None for item in ranked):
            return self._limit_per_object(ranked, limit)[:self.config.elite]
        selected, seen, counts = [], set(), {}
        for diversify in (True, False):
            for item in ranked:
                if any(item is previous for previous in selected):
                    continue
                region = item.diagnostics.get("target_region_id")
                region = None if region is None else tuple(region)
                if diversify and (region is None or region in seen):
                    continue
                oid = item.action.object_id
                if limit is not None and counts.get(oid, 0) >= limit:
                    continue
                selected.append(item)
                seen.add(region)
                counts[oid] = counts.get(oid, 0) + 1
                if len(selected) == self.config.elite:
                    return selected
        return selected


def _duplicate(first: Action, second: Action, xy_threshold, yaw_threshold) -> bool:
    if first.object_id != second.object_id:
        return False
    if np.linalg.norm(first.init_pose[:2] - second.init_pose[:2]) > xy_threshold:
        return False
    first_quaternion = first.init_pose[3:] / np.linalg.norm(first.init_pose[3:])
    second_quaternion = second.init_pose[3:] / np.linalg.norm(second.init_pose[3:])
    dot = float(abs(np.dot(first_quaternion, second_quaternion)))
    angle = 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))
    return angle <= yaw_threshold


def _yaw_quaternion(angle: float) -> np.ndarray:
    half = angle / 2.0
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)])


def _quaternion_multiply(first, second) -> np.ndarray:
    x1, y1, z1, w1 = np.asarray(first, dtype=float)
    x2, y2, z2, w2 = np.asarray(second, dtype=float)
    result = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )
    return result / np.linalg.norm(result)
