"""Height initialization from the commissioned PLY surface vertices."""

from pathlib import Path

import numpy as np

from stacking_common.domain.geometry import transform_points
from .types import InitializedPlacement
from .target_regions import ground_height


def read_ply_vertices(path):
    """Read scalar PLY vertex tables without loading a visualization library.

Supports ASCII and binary PLY with vertices as the first element, including
the double XYZ/normal files exported by Open3D for the commissioned catalog.
Face data is not needed for bounding-box initialization.
"""
    scalar_types = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
                    "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
                    "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
                    "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8"}
    with Path(path).open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        encoding = None
        count = None
        element = None
        fields = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"incomplete PLY header: {path}")
            parts = line.decode("ascii").split()
            if not parts:
                continue
            if parts[0] == "end_header":
                break
            if parts[0] == "format":
                encoding = parts[1]
            elif parts[0] == "element":
                element = parts[1]
                if count is None:
                    if element != "vertex":
                        raise ValueError("PLY vertices must be the first element")
                    count = int(parts[2])
            elif parts[0] == "property" and element == "vertex":
                if parts[1] not in scalar_types:
                    raise ValueError("PLY vertex properties must be scalar")
                fields.append((parts[2], scalar_types[parts[1]]))
        if count is None or count <= 0 or not {"x", "y", "z"} <= {n for n, _ in fields}:
            raise ValueError("PLY must contain nonempty XYZ vertices")
        if encoding == "ascii":
            values = np.loadtxt(stream, max_rows=count, ndmin=2)
            points = values[:, [[n for n, _ in fields].index(axis) for axis in "xyz"]]
        elif encoding in ("binary_little_endian", "binary_big_endian"):
            endian = "<" if encoding == "binary_little_endian" else ">"
            values = np.fromfile(stream, dtype=np.dtype([(n, endian + t) for n, t in fields]), count=count)
            points = np.column_stack([values[axis] for axis in "xyz"])
        else:
            raise ValueError(f"unsupported PLY encoding: {encoding}")
        if points.shape != (count, 3) or not np.all(np.isfinite(points)):
            raise ValueError("PLY vertex table is incomplete or nonfinite")
        return points


class PlyHeightInitializer:
    def __init__(self, model_directory, *, clearance, xy_padding, ground_z=None):
        self.model_directory = Path(model_directory)
        self.clearance, self.xy_padding = float(clearance), float(xy_padding)
        self.ground_z = ground_z
        self._vertices = {}
        self._bounds = {}

    def vertices(self, obj):
        if obj.object_id not in self._vertices:
            path = self.model_directory / f"model_{obj.object_id}_mesh.ply"
            # Legacy falls back to the hull when a PLY is unavailable.
            self._vertices[obj.object_id] = (read_ply_vertices(path) if path.is_file()
                                             else obj.mesh_arrays()[0])
        return self._vertices[obj.object_id]

    def prepare_scene(self, inventory, state):
        """Compute placed-body bounds for one read-only batch input state."""
        bounds = []
        for index in state.placed_object_indices:
            obj = inventory[index]
            pose = np.asarray(state.object_poses[obj.object_id], dtype=float)
            # Child states repeat their parents' placed poses; reuse exact bounds.
            key = (obj.object_id, pose.tobytes())
            if key not in self._bounds:
                if len(self._bounds) >= 4096:
                    self._bounds.clear()
                placed = transform_points(self.vertices(obj), pose)
                low, high = placed.min(axis=0), placed.max(axis=0)
                low.setflags(write=False)
                high.setflags(write=False)
                self._bounds[key] = (low, high)
            bounds.append((index, *self._bounds[key]))
        return tuple(bounds)

    def initialize(self, inventory, state, object_index, proposal, *, scene_context=None):
        pose = proposal.pose.copy()
        pose[2] = 0.
        points = transform_points(self.vertices(inventory[object_index]), pose)
        low, high = points.min(axis=0), points.max(axis=0)
        height = ground_height(inventory, self.ground_z)
        supports = []
        bounds = self.prepare_scene(inventory, state) if scene_context is None else scene_context
        for index, p_low, p_high in bounds:
            # Each footprint expands by padding, exactly as legacy initialize_height.
            if np.all(high[:2] + 2*self.xy_padding >= p_low[:2]) and np.all(p_high[:2] + 2*self.xy_padding >= low[:2]):
                if p_high[2] > height:
                    height, supports = float(p_high[2]), [index]
                elif np.isclose(p_high[2], height):
                    supports.append(index)
        pose[2] = height - low[2] + self.clearance
        return InitializedPlacement(pose, height, tuple(supports), diagnostics={
            "initial_height_clearance": self.clearance, "initial_height_xy_padding": self.xy_padding,
            "initial_height_mesh_source": "ply"})
