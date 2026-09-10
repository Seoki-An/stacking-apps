"""Orientation-bank registration and one-to-one field assignment."""

from dataclasses import dataclass

import numpy as np

from stacking_common.pointcloud import (
    PointCloudData,
    RegistrationDecision,
    RegistrationQualityConfig,
    RegistrationResult,
    evaluate_registration,
)
from stacking_tabletop.config import FieldAssignmentProfile

from .catalog import TabletopPcdCatalog
from .registration import Open3dRegistrationBackend


@dataclass(frozen=True)
class CandidateRegistration:
    registration: RegistrationResult
    decision: RegistrationDecision
    initial_rotation_index: int
    initial_transform: np.ndarray
    score: float
    bottom_height: float
    below_table_fraction: float


@dataclass(frozen=True)
class FieldAssignmentResult:
    assignments: dict[int, int]
    candidates: dict[tuple[int, int], CandidateRegistration]


class TabletopFieldMatcher:
    """Match known catalog objects to segmented field clusters."""

    def __init__(
        self,
        catalog: TabletopPcdCatalog,
        backend: Open3dRegistrationBackend,
        quality: RegistrationQualityConfig,
        score_profile: FieldAssignmentProfile,
    ) -> None:
        self.catalog = catalog
        self.backend = backend
        self.quality = quality
        self.score_profile = score_profile

    def register_best(
        self,
        object_id: int,
        observation: PointCloudData,
        table_plane: np.ndarray,
        rotations: tuple[np.ndarray, ...] = None,
    ) -> CandidateRegistration:
        model = self.catalog.load(object_id)
        plane = np.asarray(table_plane, dtype=np.float64)
        if plane.shape != (4,) or np.linalg.norm(plane[:3]) <= 1e-12:
            raise ValueError("table plane must contain a nonzero normal")
        rotations = ORIENTATION_BANK if rotations is None else rotations
        if not rotations:
            raise ValueError("registration orientation bank must not be empty")
        best = None
        for index, rotation in enumerate(rotations):
            initial = initial_table_transform(
                model.points,
                observation.points,
                rotation,
                plane,
            )
            registration = self.backend.register(model, observation, initial)
            snapped, bottom, below = _snap_and_evaluate(
                self.backend,
                model,
                observation,
                registration,
                plane,
                self.score_profile.below_table_tolerance,
            )
            decision = evaluate_registration(snapped, initial, self.quality)
            score = _score(
                model,
                observation,
                snapped,
                bottom,
                below,
                self.backend.config.evaluation_distance,
                self.score_profile,
            )
            candidate = CandidateRegistration(
                registration=snapped,
                decision=decision,
                initial_rotation_index=index,
                initial_transform=initial,
                score=score,
                bottom_height=bottom,
                below_table_fraction=below,
            )
            if best is None or candidate.score > best.score:
                best = candidate
        return best

    def assign(
        self,
        object_ids: tuple[int, ...],
        clusters: tuple[PointCloudData, ...],
        table_plane: np.ndarray,
        rotations: tuple[np.ndarray, ...] = None,
    ) -> FieldAssignmentResult:
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as error:
            raise RuntimeError(
                "SciPy is required for one-to-one field assignment"
            ) from error
        scores = np.full((len(object_ids), len(clusters)), -1e6)
        candidates = {}
        for object_index, object_id in enumerate(object_ids):
            for cluster_index, cluster in enumerate(clusters):
                candidate = self.register_best(
                    object_id,
                    cluster,
                    table_plane,
                    rotations,
                )
                candidates[(object_index, cluster_index)] = candidate
                if candidate.decision.accepted:
                    scores[object_index, cluster_index] = candidate.score
        rows, columns = linear_sum_assignment(-scores)
        assignments = {
            int(cluster_index): int(object_ids[object_index])
            for object_index, cluster_index in zip(rows, columns)
            if scores[object_index, cluster_index] > -1e6
        }
        return FieldAssignmentResult(assignments, candidates)


def initial_table_transform(
    model_points: np.ndarray,
    observed_points: np.ndarray,
    rotation: np.ndarray,
    table_plane: np.ndarray,
) -> np.ndarray:
    """Center one orientation and place its first percentile on the table."""

    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("initial rotation must have shape (3, 3)")
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = (
        np.median(observed_points, axis=0)
        - rotation @ np.median(model_points, axis=0)
    )
    aligned = model_points @ rotation.T + transform[:3, 3]
    normal = table_plane[:3]
    bottom = float(np.quantile(aligned @ normal + table_plane[3], 0.01))
    transform[:3, 3] -= bottom * normal / float(normal @ normal)
    return transform


def _snap_and_evaluate(
    backend: Open3dRegistrationBackend,
    model: PointCloudData,
    observation: PointCloudData,
    registration: RegistrationResult,
    plane: np.ndarray,
    below_tolerance: float,
) -> tuple[RegistrationResult, float, float]:
    transform = registration.transform.copy()
    aligned = model.points @ transform[:3, :3].T + transform[:3, 3]
    normal = plane[:3]
    normal_squared = float(normal @ normal)
    bottom = float(np.quantile(aligned @ normal + plane[3], 0.01))
    transform[:3, 3] -= bottom * normal / normal_squared
    aligned = model.points @ transform[:3, :3].T + transform[:3, 3]
    heights = aligned @ normal + plane[3]
    below = float(np.mean(heights < -below_tolerance))
    evaluated = backend.evaluate(
        model,
        observation,
        transform,
        scales=registration.scales,
        correspondences=registration.correspondences,
    )
    snapped_bottom = float(np.quantile(heights, 0.01))
    return evaluated, snapped_bottom, below


def _score(
    model: PointCloudData,
    observation: PointCloudData,
    registration: RegistrationResult,
    bottom: float,
    below: float,
    evaluation_distance: float,
    profile: FieldAssignmentProfile,
) -> float:
    model_extent = np.linalg.norm(np.ptp(model.points, axis=0))
    observed_extent = np.linalg.norm(np.ptp(observation.points, axis=0))
    extent_penalty = abs(
        np.log(max(observed_extent, 1e-9) / max(model_extent, 1e-9))
    )
    return float(
        profile.target_coverage_weight * registration.target_coverage
        + profile.source_coverage_weight * registration.source_coverage
        - profile.rmse_weight
        * min(registration.rmse / evaluation_distance, 2.0)
        - profile.extent_weight * extent_penalty
        - profile.bottom_weight
        * min(abs(bottom) / profile.bottom_scale, 2.0)
        - profile.below_table_weight * below
    )


def _rotation_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    def axis_x(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    def axis_y(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    def axis_z(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    return axis_z(yaw) @ axis_y(pitch) @ axis_x(roll)


ORIENTATION_BANK_DEGREES = tuple(
    (roll, pitch, yaw)
    for roll, pitch in (
        (0.0, 0.0),
        (180.0, 0.0),
        (90.0, 0.0),
        (-90.0, 0.0),
        (0.0, 90.0),
        (0.0, -90.0),
    )
    for yaw in (0.0, 90.0, 180.0, 270.0)
)
ORIENTATION_BANK = tuple(
    _rotation_xyz(*np.deg2rad(angles)) for angles in ORIENTATION_BANK_DEGREES
)
