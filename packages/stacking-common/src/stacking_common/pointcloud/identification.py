"""Registration over explicit pose hypotheses with quality and ambiguity gates."""

from dataclasses import dataclass
import numpy as np

from .registration import RegistrationResult, evaluate_registration


@dataclass(frozen=True)
class PoseIdentification:
    result: RegistrationResult | None
    valid: bool
    reasons: tuple[str, ...]
    score: float | None
    candidate_count: int


def identify_pose(source, target, seeds, backend, quality, *, minimum_points,
                  ambiguity_score_gap, distinct_translation, distinct_rotation_degrees):
    """Rank as legacy fitness-minus-RMSE; reject distinct near-equal solutions.

    Several seeds converging to the same pose do not constitute ambiguity.
    The caller supplies platform-specific metric limits and orientation seeds.
    """
    if len(target) < minimum_points:
        return PoseIdentification(None, False, ("insufficient_evidence",), None, 0)
    candidates = []
    for seed in seeds:
        try:
            result = backend.register(source, target, seed)
        except ValueError:
            # A crop can become too sparse at a coarse registration scale.
            continue
        decision = evaluate_registration(result, seed, quality)
        scale = result.scales[-1]
        score = scale.fitness - scale.rmse
        candidates.append((score, result, decision))
    if not candidates:
        return PoseIdentification(None, False, ("insufficient_evidence",), None, 0)
    candidates.sort(key=lambda item: item[0], reverse=True)
    valid = [item for item in candidates if item[2].accepted]
    score, best, decision = valid[0] if valid else candidates[0]
    reasons = list(decision.reasons)
    for alternative_score, alternative, _ in valid[1:]:
        if score - alternative_score > ambiguity_score_gap:
            break
        delta = np.linalg.inv(best.transform) @ alternative.transform
        angle = np.rad2deg(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))
        if np.linalg.norm(delta[:3, 3]) > distinct_translation or angle > distinct_rotation_degrees:
            reasons.append("ambiguous_pose")
            break
    return PoseIdentification(best, not reasons, tuple(reasons), score, len(candidates))
