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
        x = np.linspace(self.xlim[0], self.xlim[1], width)
        y = np.linspace(self.ylim[0], self.ylim[1], height)
        output = np.zeros(height * width)
        for points, faces in meshes:
            triangles = np.asarray(points, dtype=float)[
                np.asarray(faces, dtype=int)
            ]
            _rasterize_triangles(output, x, y, triangles.reshape(-1, 3, 3))
        # Rounding is monotonic, so rounding the maximum equals the maximum of
        # values rounded one triangle at a time into a float32 map.
        return output.reshape(height, width).astype(np.float32)


def _rasterize_triangles(
    output: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    triangles: np.ndarray,
) -> None:
    """Max each triangle's interpolated height into its covered grid cells."""
    (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = (
        triangles[:, corner].T for corner in range(3)
    )
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    column_start = np.maximum(np.searchsorted(x, triangles[:, :, 0].min(axis=1)), 0)
    column_stop = np.minimum(
        np.searchsorted(x, triangles[:, :, 0].max(axis=1), side="right"), len(x)
    )
    row_start = np.maximum(np.searchsorted(y, triangles[:, :, 1].min(axis=1)), 0)
    row_stop = np.minimum(
        np.searchsorted(y, triangles[:, :, 1].max(axis=1), side="right"), len(y)
    )
    columns = np.maximum(column_stop - column_start, 0)
    rows = np.maximum(row_stop - row_start, 0)
    keep = np.flatnonzero((np.abs(denominator) > 1e-12) & (columns > 0) & (rows > 0))
    if not len(keep):
        return
    # One entry per (triangle, covered cell), row-major within each triangle.
    cells = rows[keep] * columns[keep]
    triangle = np.repeat(keep, cells)
    offset = np.arange(len(triangle)) - np.repeat(np.cumsum(cells) - cells, cells)
    column = column_start[triangle] + offset % columns[triangle]
    row = row_start[triangle] + offset // columns[triangle]
    xx, yy = x[column], y[row]
    x0, y0, z0, x1, y1, z1, x2, y2, z2, denominator = (
        values[triangle] for values in (x0, y0, z0, x1, y1, z1, x2, y2, z2, denominator)
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
    np.maximum.at(output, (row * len(x) + column)[inside], values[inside])


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values).copy()
    result.setflags(write=False)
    return result
