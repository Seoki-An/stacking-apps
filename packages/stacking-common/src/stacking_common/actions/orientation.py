"""Orientation providers for planar placement sampling."""

from collections.abc import Mapping

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from stacking_common.domain import ObjectInventory, State
from stacking_common.domain.geometry import quaternion_matrix


class InventoryOrientationProvider:
    """Use the initial inventory pose, normally supplied by perception."""

    def quaternion(self, inventory, state, observation, object_index):
        del state, observation
        object_id = inventory[object_index].object_id
        return inventory.initial_pose(object_id)[3:]


class ScanAlignedOrientationProvider:
    """Yaw-align a measured pose's longest object axis as legacy sampling did."""

    def quaternion(self, inventory, state, observation, object_index):
        del state, observation
        stack_object = inventory[object_index]
        measured = _quaternion(inventory.initial_pose(stack_object.object_id)[3:])
        points, _ = stack_object.mesh_arrays()
        longest_local = _longest_axis(points)
        longest_world = quaternion_matrix(measured) @ longest_local
        if np.linalg.norm(longest_world[:2]) < 1e-9:
            return measured
        heading = float(np.arctan2(longest_world[1], longest_world[0]))
        correction = -((heading + np.pi / 2.0) % np.pi - np.pi / 2.0)
        return _quaternion_multiply(_yaw_quaternion(correction), measured)


class MappingOrientationProvider:
    """Use explicit measured quaternions keyed by stable object ID."""

    def __init__(self, quaternions: Mapping[int, np.ndarray]) -> None:
        self.quaternions = {
            int(object_id): _quaternion(value)
            for object_id, value in quaternions.items()
        }

    def quaternion(
        self,
        inventory: ObjectInventory,
        state: State,
        observation,
        object_index: int,
    ) -> np.ndarray:
        del state, observation
        object_id = inventory[object_index].object_id
        try:
            return self.quaternions[object_id].copy()
        except KeyError as error:
            raise ValueError(
                f"missing measured orientation for object {object_id}"
            ) from error


def _quaternion(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("measured orientation must be a finite XYZW quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("measured orientation quaternion must be non-zero")
    return quaternion.copy() / norm


def _longest_axis(points: np.ndarray) -> np.ndarray:
    """Use the overall hull, not the density of decomposed component vertices."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        return np.array([1.0, 0.0, 0.0])
    values = values[np.all(np.isfinite(values), axis=1)]
    if len(values) < 3:
        return np.array([1.0, 0.0, 0.0])
    try:
        values = values[ConvexHull(values).vertices]
    except QhullError:
        # Retain the existing PCA fallback for planar/degenerate test geometry.
        pass
    centered = values - np.mean(values, axis=0)
    covariance = centered.T @ centered / float(len(centered))
    if not np.all(np.isfinite(covariance)) or np.linalg.norm(covariance) < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    _, axes = np.linalg.eigh(covariance)
    axis = axes[:, -1]
    return -axis if axis[np.argmax(np.abs(axis))] < 0.0 else axis


def _yaw_quaternion(angle: float) -> np.ndarray:
    half = float(angle) / 2.0
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)])


def _quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.asarray(first, dtype=float)
    x2, y2, z2, w2 = np.asarray(second, dtype=float)
    return _quaternion(
        np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ]
        )
    )


# Characterized against stacking-planner canonical asset orientation.
def _positive_dominant_component(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    dominant = int(np.argmax(np.abs(axis)))
    return -axis if axis[dominant] < 0.0 else axis


def canonical_principal_axis_rotation(vertices: np.ndarray) -> Rotation:
    """Return the stone-local rotation with longest X and shortest Z axes."""
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")

    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        return Rotation.identity()

    centered = points - np.mean(points, axis=0)
    covariance = centered.T @ centered / float(len(centered))
    if not np.all(np.isfinite(covariance)) or np.linalg.norm(covariance) < 1e-12:
        return Rotation.identity()

    _, axes = np.linalg.eigh(covariance)
    x_axis = _positive_dominant_component(axes[:, -1])
    z_axis = _positive_dominant_component(axes[:, 0])
    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-9:
        return Rotation.identity()
    y_axis = y_axis / y_norm
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    principal_axes = np.column_stack((x_axis, y_axis, z_axis))
    return Rotation.from_matrix(principal_axes.T)


class PrincipalAxisOrientationProvider:
    """Asset-local longest-X/shortest-Z orientations with explicit roll variants."""

    def __init__(self, roll_angles):
        self.roll_angles = tuple(float(angle) for angle in roll_angles)
        if not self.roll_angles or not np.all(np.isfinite(self.roll_angles)):
            raise ValueError("roll angles must be nonempty and finite")
        self._cache = {}

    def quaternions(self, inventory, state, observation, object_index):
        obj = inventory[object_index]
        key = id(obj)
        if key not in self._cache:
            points, _ = obj.mesh_arrays()
            try:
                points = points[ConvexHull(points).vertices]
            except QhullError:
                pass
            base = canonical_principal_axis_rotation(points)
            orientations = tuple(
                (Rotation.from_rotvec([angle, 0., 0.]) * base).as_quat()
                for angle in self.roll_angles)
            # Retain the object so another inventory cannot reuse its ID.
            self._cache[key] = (obj, orientations)
        return tuple(value.copy() for value in self._cache[key][1])

    def quaternion(self, inventory, state, observation, object_index):
        return self.quaternions(inventory, state, observation, object_index)[0]
