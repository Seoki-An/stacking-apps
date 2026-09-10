"""Platform height-map rendering without a GUI or ROS dependency."""

import numpy as np

from stacking_common.domain import ObjectInventory, State
from stacking_common.domain.geometry import transform_points


class HeightMapRenderer:
    """Rasterize platform meshes onto a fixed XY grid."""

    def __init__(
        self,
        inventory: ObjectInventory,
        resolution: tuple[int, int],
        margin: float,
    ) -> None:
        if len(resolution) != 2 or any(int(value) <= 0 for value in resolution):
            raise ValueError("height-map resolution must contain two positive values")
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("height-map margin must be finite and non-negative")

        self.inventory = inventory
        self.resolution = tuple(int(value) for value in resolution)
        self.margin = float(margin)
        center = inventory.target.origin[:2]
        span = max(inventory.target.width, inventory.target.length) + self.margin
        self.xlim = _readonly(
            np.array([center[0] - span / 2.0, center[0] + span / 2.0])
        )
        self.ylim = _readonly(
            np.array([center[1] - span / 2.0, center[1] + span / 2.0])
        )
        self._target_height_map = _readonly(
            self._rasterize(
                [(inventory.target.points, inventory.target.faces)]
            )
        )

    def render(self, state: State) -> np.ndarray:
        if tuple(state.object_ids) != self.inventory.object_ids:
            raise ValueError("state inventory does not match height-map inventory")

        meshes = []
        for index in state.placed_object_indices:
            stack_object = self.inventory[index]
            pose = state.object_poses.get(
                stack_object.object_id,
                self.inventory.initial_pose(stack_object.object_id),
            )
            points, faces = stack_object.mesh_arrays()
            meshes.append((transform_points(points, pose), faces))
        return self._rasterize(meshes)

    def render_target(self) -> np.ndarray:
        return self._target_height_map.copy()

    def _rasterize(
        self, meshes: list[tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        height, width = self.resolution
        output = np.zeros((height, width), dtype=np.float32)
        x = np.linspace(self.xlim[0], self.xlim[1], width)
        y = np.linspace(self.ylim[0], self.ylim[1], height)

        for points, faces in meshes:
            triangles = np.asarray(points, dtype=float)[
                np.asarray(faces, dtype=int)
            ]
            for triangle in triangles:
                _rasterize_triangle(output, x, y, triangle)
        return output


def _rasterize_triangle(
    output: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    triangle: np.ndarray,
) -> None:
    x0, y0, z0 = triangle[0]
    x1, y1, z1 = triangle[1]
    x2, y2, z2 = triangle[2]
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(float(denominator)) <= 1e-12:
        return

    column_start = max(int(np.searchsorted(x, np.min(triangle[:, 0]))), 0)
    column_stop = min(
        int(np.searchsorted(x, np.max(triangle[:, 0]), side="right")),
        len(x),
    )
    row_start = max(int(np.searchsorted(y, np.min(triangle[:, 1]))), 0)
    row_stop = min(
        int(np.searchsorted(y, np.max(triangle[:, 1]), side="right")),
        len(y),
    )
    if column_start >= column_stop or row_start >= row_stop:
        return

    xx, yy = np.meshgrid(
        x[column_start:column_stop],
        y[row_start:row_stop],
        indexing="xy",
    )
    weight0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denominator
    weight1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denominator
    weight2 = 1.0 - weight0 - weight1
    tolerance = 1e-7
    inside = (
        (weight0 >= -tolerance)
        & (weight1 >= -tolerance)
        & (weight2 >= -tolerance)
    )
    values = weight0 * z0 + weight1 * z1 + weight2 * z2
    region = output[row_start:row_stop, column_start:column_stop]
    np.maximum(region, np.where(inside, values, -np.inf), out=region)


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values).copy()
    result.setflags(write=False)
    return result
