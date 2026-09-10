"""File adapters for canonical reconstruction values."""

from pathlib import Path

import numpy as np

from stacking_common.pointcloud import PointCloudData

from .types import TriangleMeshData


def load_point_cloud(path: str | Path) -> PointCloudData:
    o3d = _open3d()
    source = Path(path)
    value = o3d.io.read_point_cloud(str(source))
    if not value.has_points():
        raise ValueError(f"point cloud is empty or unreadable: {source}")
    return PointCloudData(np.asarray(value.points))


def write_point_cloud(path: str | Path, cloud: PointCloudData) -> Path:
    o3d = _open3d()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = o3d.geometry.PointCloud()
    value.points = o3d.utility.Vector3dVector(np.array(cloud.points, copy=True))
    if not o3d.io.write_point_cloud(str(destination), value):
        raise OSError(f"failed to write point cloud: {destination}")
    return destination


def write_triangle_mesh(path: str | Path, mesh: TriangleMeshData) -> Path:
    o3d = _open3d()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = o3d.geometry.TriangleMesh()
    value.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices, copy=True))
    value.triangles = o3d.utility.Vector3iVector(
        np.array(mesh.triangles, dtype=np.int32, copy=True)
    )
    value.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(destination), value):
        raise OSError(f"failed to write triangle mesh: {destination}")
    return destination


def _open3d():
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D is required for reconstruction file I/O") from error
    return o3d
