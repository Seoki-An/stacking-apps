"""Pure NumPy planar proposal sampling."""

from __future__ import annotations

from typing import Protocol
from collections import deque

import numpy as np

from stacking_common.domain import ObjectInventory, State
from stacking_common.domain.geometry import quaternion_matrix

from .config import PlanarSamplingConfig
from .orientation import InventoryOrientationProvider
from .types import PlanarPlacement
from .floor import FloorProposalRanker


class HeightMapObservation(Protocol):
    """Observation fields required by height-map planar sampling."""

    scene_height_map: np.ndarray | None
    height_map_xlim: np.ndarray
    height_map_ylim: np.ndarray


class HeightMapPlanarPoseSampler:
    """Rank target-fitting grid poses by the current scene height."""

    SUPPORTED_PRIORITIES = frozenset({None, "lower_floor_first"})

    def __init__(self, config: PlanarSamplingConfig, orientation_provider=None, *, floor_policy=None) -> None:
        self.config = config
        self.floor_policy = floor_policy
        self.orientation_provider = (
            InventoryOrientationProvider()
            if orientation_provider is None
            else orientation_provider
        )

    def sample(
        self,
        inventory: ObjectInventory,
        state: State,
        observation: HeightMapObservation,
        object_index: int,
        count: int,
        sampling_priority: str | None,
    ) -> tuple[PlanarPlacement, ...]:
        if count <= 0:
            raise ValueError("planar proposal count must be positive")
        if sampling_priority not in self.SUPPORTED_PRIORITIES:
            raise ValueError(
                f"unsupported planar sampling priority: {sampling_priority!r}"
            )
        if object_index < 0 or object_index >= len(inventory):
            raise ValueError("planar proposal object index is outside the inventory")

        height_map = np.asarray(observation.scene_height_map, dtype=float)
        if height_map.ndim != 2 or height_map.size == 0:
            raise ValueError("planar sampling requires a two-dimensional height map")
        if not np.all(np.isfinite(height_map)):
            raise ValueError("scene height map must contain only finite values")
        xlim = _limits(observation.height_map_xlim, "x")
        ylim = _limits(observation.height_map_ylim, "y")
        sampled_height = _resample_nearest(height_map, self.config.grid_shape)
        height_min = float(np.min(sampled_height))
        height_max = float(np.max(sampled_height))
        rows, columns = self.config.grid_shape
        x_values = np.linspace(xlim[0], xlim[1], columns)
        y_values = np.linspace(ylim[0], ylim[1], rows)
        floor_ranker = (
            FloorProposalRanker(self.floor_policy, state, x_values, y_values, sampled_height)
            if self.floor_policy is not None and sampling_priority == "lower_floor_first"
            else None
        )

        stack_object = inventory[object_index]
        points, _ = stack_object.mesh_arrays()
        variants = getattr(self.orientation_provider, "quaternions", None)
        if callable(variants):
            base_quaternions = variants(inventory, state, observation, object_index)
        else:
            base_quaternions = (self.orientation_provider.quaternion(
                inventory, state, observation, object_index),)
        target = inventory.target
        target_min = target.origin[:2] - np.array(
            [target.width, target.length], dtype=float
        ) / 2.0
        target_max = target.origin[:2] + np.array(
            [target.width, target.length], dtype=float
        ) / 2.0
        anchor_inset_scale = self.config.target_anchor_inset_scale
        anchor_inset = (
            None
            if anchor_inset_scale is None
            else 0.5
            * anchor_inset_scale
            * np.ptp(points[:, :2], axis=0)
        )

        ranked: list[tuple[float, int, int, int, np.ndarray]] = []
        floor_metrics = {}
        orientations = tuple(_quaternion_multiply(_yaw_quaternion(yaw), base)
                             for base in base_quaternions for yaw in self.config.yaw_angles)
        for orientation_index, quaternion in enumerate(orientations):
            rotated_xy = points @ quaternion_matrix(quaternion).T
            local_min = np.min(rotated_xy[:, :2], axis=0)
            local_max = np.max(rotated_xy[:, :2], axis=0)
            kernel = None if floor_ranker is None else floor_ranker.kernel(object_index, quaternion)
            if anchor_inset is None:
                center_min = target_min - local_min
                center_max = target_max - local_max
            else:
                center_min = target_min + anchor_inset
                center_max = target_max - anchor_inset
            valid_x = (x_values >= center_min[0]) & (x_values <= center_max[0])
            valid_y = (y_values >= center_min[1]) & (y_values <= center_max[1])
            for row in np.flatnonzero(valid_y):
                for column in np.flatnonzero(valid_x):
                    if floor_ranker is not None:
                        metrics = floor_ranker.metrics(kernel, row, column)
                        if metrics is None:
                            continue
                        half_extent = (center_max - center_min) / 2.0
                        metrics["target_boundary_proximity"] = float(np.max(np.abs(
                            (np.array([x_values[column], y_values[row]]) - (center_min + center_max) / 2.0)
                            / np.maximum(half_extent, 1e-12)
                        )))
                        floor_metrics[(int(row), int(column), orientation_index)] = metrics
                    pose = np.array(
                        [
                            x_values[column],
                            y_values[row],
                            0.0,
                            *quaternion,
                        ],
                        dtype=float,
                    )
                    ranked.append(
                        (
                            float(sampled_height[row, column]),
                            int(row),
                            int(column),
                            int(orientation_index),
                            pose,
                        )
                    )

        if floor_ranker is not None and ranked:
            heights = [m["footprint_support_height"] for m in floor_metrics.values()]
            low, high = min(heights), max(heights)
            max_fill = max(m["active_floor_unfilled_cells"] for m in floor_metrics.values())
            rescored = []
            for _, row, column, orientation_index, pose in ranked:
                metrics = floor_metrics[(row, column, orientation_index)]
                height_term = (metrics["footprint_support_height"] - low) / max(high - low, 1e-6)
                score = float(np.dot(self.floor_policy.score_weights, (
                    height_term, metrics["footprint_support_likelihood"],
                    metrics["active_floor_open_fraction"],
                    metrics["active_floor_unfilled_cells"] / max(max_fill, 1),
                    metrics["active_floor_frontier_fraction"],
                    metrics["target_boundary_proximity"],
                )))
                metrics["floor_sampling_score"] = score
                rescored.append((-score, row, column, orientation_index, pose))
            ranked = rescored
        ranked.sort(key=lambda value: value[:4])
        if floor_ranker is not None:
            # Keep the best seeds from each gap/orientation before taking
            # neighboring variants of one locally attractive placement.
            buckets = {}
            for value in ranked:
                _, row, column, orientation, _ = value
                region = floor_metrics[(row, column, orientation)]["floor_gap_region"]
                buckets.setdefault((region, orientation), deque()).append(value)
            queues = list(buckets.values())
            ranked = []
            while queues:
                ranked.extend(queue.popleft() for queue in queues)
                queues = [queue for queue in queues if queue]
        selected: list[PlanarPlacement] = []
        selected_xy: dict[int, list[np.ndarray]] = {}
        for height, row, column, orientation_index, pose in ranked:
            metrics = floor_metrics.get((row, column, orientation_index), {})
            support_height = metrics.get("footprint_support_height", height)
            if any(
                np.linalg.norm(pose[:2] - xy)
                < self.config.minimum_xy_separation
                for xy in selected_xy.get(orientation_index, ())
            ):
                continue
            selected.append(
                PlanarPlacement(
                    pose=pose,
                    score=-height,
                    diagnostics={
                        "source": "height_map",
                        "height_map_score": -support_height,
                        "height_map_score_normalized": _normalized_height_score(
                            support_height, height_min, height_max
                        ),
                        "height_map_value": support_height,
                        "grid_row": row,
                        "grid_column": column,
                        "orientation_index": orientation_index,
                        **floor_metrics.get((row, column, orientation_index), {}),
                    },
                )
            )
            selected_xy.setdefault(orientation_index, []).append(
                pose[:2].copy()
            )
            if len(selected) == count:
                break
        return tuple(selected)

    def score_at_pose(
        self,
        observation: HeightMapObservation,
        pose: np.ndarray,
    ) -> tuple[float, float]:
        """Return raw and normalized low-floor scores at an arbitrary XY pose."""

        height_map = np.asarray(observation.scene_height_map, dtype=float)
        if height_map.ndim != 2 or height_map.size == 0:
            raise ValueError("planar scoring requires a two-dimensional height map")
        if not np.all(np.isfinite(height_map)):
            raise ValueError("scene height map must contain only finite values")
        xlim = _limits(observation.height_map_xlim, "x")
        ylim = _limits(observation.height_map_ylim, "y")
        xy = np.asarray(pose, dtype=float)[:2]
        column = _nearest_index(xy[0], xlim, height_map.shape[1])
        row = _nearest_index(xy[1], ylim, height_map.shape[0])
        height = float(height_map[row, column])
        return (
            -height,
            _normalized_height_score(
                height,
                float(np.min(height_map)),
                float(np.max(height_map)),
            ),
        )


def _limits(values: np.ndarray, axis: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"height-map {axis} limits must have shape (2,)")
    if result[0] >= result[1]:
        raise ValueError(f"height-map {axis} limits must be increasing")
    return result


def _resample_nearest(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows = np.rint(np.linspace(0, values.shape[0] - 1, shape[0])).astype(int)
    columns = np.rint(np.linspace(0, values.shape[1] - 1, shape[1])).astype(int)
    return values[np.ix_(rows, columns)]


def _nearest_index(value: float, limits: np.ndarray, size: int) -> int:
    fraction = (float(value) - float(limits[0])) / float(limits[1] - limits[0])
    return int(np.clip(np.rint(fraction * (size - 1)), 0, size - 1))


def _normalized_height_score(height: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((high - height) / (high - low), 0.0, 1.0))


def _yaw_quaternion(angle: float) -> np.ndarray:
    half = float(angle) / 2.0
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)])


def _quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
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
