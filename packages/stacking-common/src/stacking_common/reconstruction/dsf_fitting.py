"""Platform-neutral differentiable-support-function fitting."""

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from .config import DsfFittingProfile
from .types import DsfFitResult


class DsfFitter:
    """Fit one smooth convex DSF to each point-cloud partition."""

    def __init__(self, profile: DsfFittingProfile) -> None:
        self.profile = profile

    def fit(
        self,
        point_sets: tuple[np.ndarray, ...],
        *,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> tuple[DsfFitResult, ...]:
        budgets = distribute_vertex_budget(
            self.profile.total_vertices,
            len(point_sets),
        )
        results = []
        for index, (points, vertices) in enumerate(
            zip(point_sets, budgets),
            start=1,
        ):
            print(
                f"fitting DSF [{index}/{len(point_sets)}]: "
                f"{len(points)} points, {vertices} vertices",
                flush=True,
            )
            label = f"DSF component {index}/{len(point_sets)}"
            results.append(
                self._fit_one(
                    np.asarray(points),
                    vertices,
                    progress=progress,
                    label=label,
                )
            )
            if progress is not None:
                progress(index, len(point_sets), f"Fitted {label.lower()}")
        return tuple(results)

    def _fit_one(
        self,
        points: np.ndarray,
        vertices: int,
        *,
        progress: Callable[[int, int, str], None] | None = None,
        label: str = "DSF component",
    ) -> DsfFitResult:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or len(values) < 4:
            raise ValueError("DSF fitting requires at least four XYZ points")
        selected = values
        if self.profile.ransac.enabled:
            selected = self._ransac_inliers(
                values,
                vertices,
                progress=progress,
                label=label,
            )
        return _fit_once(
            self.profile,
            selected,
            vertices,
            progress=(
                None
                if progress is None
                else lambda completed, total: progress(
                    completed,
                    total,
                    f"{label}: optimizing smooth support",
                )
            ),
        )

    def _ransac_inliers(
        self,
        points: np.ndarray,
        vertices: int,
        *,
        progress: Callable[[int, int, str], None] | None = None,
        label: str = "DSF component",
    ) -> np.ndarray:
        config = self.profile.ransac
        rng = np.random.default_rng(config.seed)
        count = len(points)
        sample_count = int(np.ceil(count * config.sample_ratio))
        sample_count = min(sample_count, config.maximum_sample_points, count)
        sample_count = max(
            min(sample_count, count),
            min(config.minimum_sample_points, count),
        )
        best_inliers = None
        best_score = None
        quick_profile = replace(
            self.profile,
            maximum_iterations=config.fit_maximum_iterations,
            ransac=replace(config, enabled=False),
        )
        for iteration in range(1, config.iterations + 1):
            if progress is not None:
                progress(
                    iteration,
                    config.iterations,
                    f"{label}: RANSAC inlier search",
                )
            sample = points[rng.choice(count, size=sample_count, replace=False)]
            try:
                candidate = _fit_once(quick_profile, sample, vertices)
                inliers, violations = _support_inliers(
                    self.profile,
                    candidate,
                    points,
                )
            except (FloatingPointError, RuntimeError, ValueError):
                continue
            inlier_count = int(np.sum(inliers))
            if not inlier_count:
                continue
            score = (
                inlier_count,
                -float(np.median(np.maximum(violations[inliers], 0.0))),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_inliers = inliers
        if best_inliers is None:
            return points
        ratio = float(np.mean(best_inliers))
        if ratio < config.minimum_inlier_ratio:
            return points
        return points[best_inliers]


def distribute_vertex_budget(total: int, components: int) -> tuple[int, ...]:
    if components < 1:
        raise ValueError("DSF fitting requires at least one component")
    minimum = 4 * components
    if total < minimum:
        raise ValueError(
            f"DSF vertex budget {total} is too small for {components} "
            f"components; at least {minimum} is required"
        )
    base, remainder = divmod(total, components)
    return tuple(base + (index < remainder) for index in range(components))


def _fit_once(
    profile: DsfFittingProfile,
    points: np.ndarray,
    vertex_count: int,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> DsfFitResult:
    try:
        from scipy.optimize import least_squares
    except ImportError as error:
        raise RuntimeError("SciPy is required for DSF fitting") from error
    model = _DsfModel(profile, vertices=_initial_vertices(vertex_count))
    directions = _directions(profile.direction_subdivisions)
    projections = points @ directions
    support = np.quantile(
        projections,
        profile.support_quantile,
        axis=0,
    )
    cost = _ShapeCost(profile, directions, support, progress=progress)
    result = least_squares(
        fun=cost.value,
        x0=model.phi,
        jac=cost.jacobian,
        method="trf",
        max_nfev=profile.maximum_iterations,
        loss=profile.loss,
        f_scale=profile.loss_scale,
    )
    fitted = _DsfModel(profile, phi=result.x)
    return DsfFitResult(
        fitted.vertices.T,
        fitted.sharpness,
        float(2.0 * result.cost),
        int(result.nfev),
        result.x,
    )


def _support_inliers(
    profile: DsfFittingProfile,
    fitted: DsfFitResult,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model = _DsfModel(profile, phi=fitted.parameters)
    directions = _directions(profile.ransac.score_direction_subdivisions)
    support, _ = model.support(directions)
    envelope = support + directions.T @ model.center
    violation = np.max(points @ directions - envelope[None, :], axis=1)
    return violation <= profile.ransac.inlier_threshold, violation


class _ShapeCost:
    def __init__(
        self,
        profile: DsfFittingProfile,
        directions: np.ndarray,
        target: np.ndarray,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        self.profile = profile
        self.directions = directions
        self.target = target
        self.last_phi = None
        self.last_value = None
        self.last_jacobian = None
        self.evaluations = 0
        self.progress = progress

    def evaluate(self, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.last_phi is not None and np.array_equal(phi, self.last_phi):
            return self.last_value, self.last_jacobian
        model = _DsfModel(self.profile, phi=phi)
        support, derivative = model.support(self.directions, derivative=True)
        value = support + self.directions.T @ model.center - self.target
        jacobian = derivative + self.directions.T @ model.center_derivative
        self.last_phi = np.array(phi, copy=True)
        self.last_value = value
        self.last_jacobian = jacobian
        self.evaluations += 1
        if self.progress is not None:
            self.progress(
                min(self.evaluations, self.profile.maximum_iterations),
                self.profile.maximum_iterations,
            )
        return value, jacobian

    def value(self, phi: np.ndarray) -> np.ndarray:
        return self.evaluate(phi)[0]

    def jacobian(self, phi: np.ndarray) -> np.ndarray:
        return self.evaluate(phi)[1]


class _DsfModel:
    def __init__(
        self,
        profile: DsfFittingProfile,
        *,
        vertices: np.ndarray | None = None,
        phi: np.ndarray | None = None,
    ) -> None:
        self.profile = profile
        if phi is None:
            self.vertices = np.asarray(vertices, dtype=np.float64)
            self.vertex_count = self.vertices.shape[1]
            self.alpha = np.zeros(self.vertex_count)
            self.edge_parameter = 0.0
            self.phi = np.hstack(
                (
                    self.vertices.flatten("F"),
                    self.alpha,
                    self.edge_parameter,
                )
            )
        else:
            self.phi = np.asarray(phi, dtype=np.float64)
            self.vertex_count = (len(self.phi) - 1) // 4
            self.vertices = self.phi[: 3 * self.vertex_count].reshape(
                (self.vertex_count, 3)
            ).T
            self.alpha = self.phi[3 * self.vertex_count : -1]
            self.edge_parameter = float(self.phi[-1])
        scale = (self.vertex_count - 1) / (
            profile.center_interior_upper_bound * self.vertex_count - 1
        )
        shifted = self.alpha - np.max(self.alpha)
        weight = np.exp(shifted)
        weight /= np.sum(weight)
        beta = weight / scale + (scale - 1) / (scale * self.vertex_count)
        self.center = self.vertices @ beta
        self.local_vertices = self.vertices - self.center[:, None]
        center_vertices = np.hstack(
            [value * np.eye(3) for value in beta]
        )
        weight_derivative = (
            np.diag(weight) - np.outer(weight, weight)
        ) / scale
        center_alpha = self.vertices @ weight_derivative
        self.center_derivative = np.hstack(
            (
                center_vertices,
                center_alpha,
                np.zeros((3, 1)),
            )
        )
        span = profile.maximum_sharpness / 2.0 - 1.0
        self.sharpness = span * (np.tanh(self.edge_parameter) + 1.0) + 2.0
        self.sharpness_derivative = span * (
            1.0 - np.tanh(self.edge_parameter) ** 2
        )

    def support(
        self,
        directions: np.ndarray,
        *,
        derivative: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        x = np.asarray(directions, dtype=np.float64)
        projection = np.clip(x.T @ self.local_vertices, 0.0, None)
        scale = np.maximum(np.max(projection, axis=1), 1e-15)
        normalized = projection / scale[:, None]
        power_minus_one = normalized ** (self.sharpness - 1.0)
        powered = power_minus_one * normalized
        total = np.maximum(np.sum(powered, axis=1), 1e-15)
        h_factor = total ** (1.0 / self.sharpness - 1.0)
        support = h_factor * total * scale
        if not derivative:
            return support, None
        dh_dz = power_minus_one * h_factor[:, None]
        dh_dvertices_tensor = dh_dz[..., None] * x.T[:, None, :]
        dh_dvertices = dh_dvertices_tensor.reshape(
            (len(x.T), -1)
        )
        dh_dcenter = -np.sum(dh_dvertices_tensor, axis=1)
        log_normalized = np.log(np.where(normalized > 0.0, normalized, 1.0))
        dh_dsharpness = (
            scale
            * h_factor
            / self.sharpness
            * (
                np.sum(powered * log_normalized, axis=1)
                - total * np.log(total) / self.sharpness
            )
        )
        dh_dedge = dh_dsharpness[:, None] * self.sharpness_derivative
        jacobian = (
            np.hstack(
                (
                    dh_dvertices,
                    np.zeros((len(x.T), self.vertex_count)),
                    dh_dedge,
                )
            )
            + dh_dcenter @ self.center_derivative
        )
        return support, jacobian


def _initial_vertices(count: int) -> np.ndarray:
    if count < 4:
        raise ValueError("DSF fitting requires at least four vertices")
    if count == 20:
        ratio = (1.0 + np.sqrt(5.0)) * 0.5
        inverse = 1.0 / ratio
        vertices = [
            (x, y, z)
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ]
        vertices.extend(
            (0.0, y * inverse, z * ratio)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        )
        vertices.extend(
            (x * inverse, y * ratio, 0.0)
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
        )
        vertices.extend(
            (x * ratio, 0.0, z * inverse)
            for x in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        )
        values = np.asarray(vertices, dtype=np.float64)
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        return values.T
    index = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    angle = np.pi * (3.0 - np.sqrt(5.0)) * index
    return np.vstack((radius * np.cos(angle), radius * np.sin(angle), z))


def _directions(subdivisions: int) -> np.ndarray:
    if subdivisions < 0:
        raise ValueError("direction subdivisions cannot be negative")
    ratio = (1.0 + np.sqrt(5.0)) * 0.5
    vertices = np.asarray(
        [
            (-1.0, ratio, 0.0),
            (1.0, ratio, 0.0),
            (-1.0, -ratio, 0.0),
            (1.0, -ratio, 0.0),
            (0.0, -1.0, ratio),
            (0.0, 1.0, ratio),
            (0.0, -1.0, -ratio),
            (0.0, 1.0, -ratio),
            (ratio, 0.0, -1.0),
            (ratio, 0.0, 1.0),
            (-ratio, 0.0, -1.0),
            (-ratio, 0.0, 1.0),
        ],
        dtype=np.float64,
    )
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = np.asarray(
        [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ],
        dtype=np.int64,
    )
    values = [vertex.copy() for vertex in vertices]
    for _ in range(subdivisions):
        midpoint_indices: dict[tuple[int, int], int] = {}

        def midpoint(first: int, second: int) -> int:
            key = tuple(sorted((first, second)))
            if key not in midpoint_indices:
                value = values[first] + values[second]
                value /= np.linalg.norm(value)
                midpoint_indices[key] = len(values)
                values.append(value)
            return midpoint_indices[key]

        refined = []
        for first, second, third in faces:
            ab = midpoint(int(first), int(second))
            bc = midpoint(int(second), int(third))
            ca = midpoint(int(third), int(first))
            refined.extend(
                (
                    (first, ab, ca),
                    (second, bc, ab),
                    (third, ca, bc),
                    (ab, bc, ca),
                )
            )
        faces = np.asarray(refined, dtype=np.int64)
    return np.asarray(values, dtype=np.float64).T
