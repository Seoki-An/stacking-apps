"""Small point-cloud processing primitives used by larger workflows."""

import numpy as np

from .types import PointCloudData


def sample_mesh_surface(vertices, faces, count=12000):
    """Deterministic area-weighted samples without changing a global RNG."""
    vertices = PointCloudData(vertices).points
    faces = np.asarray(faces)
    if (faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu" or
            np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("surface faces must be valid triangle vertex indices")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("surface sample count must be positive")
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1)
    if not len(areas) or areas.sum() <= 0:
        raise ValueError("model requires nondegenerate triangles")
    rng = np.random.default_rng(0)
    selected = triangles[rng.choice(len(triangles), count, p=areas / areas.sum())]
    u = np.sqrt(rng.random(count))
    v = rng.random(count)
    return PointCloudData((1-u[:, None]) * selected[:, 0] +
                         (u*(1-v))[:, None] * selected[:, 1] + (u*v)[:, None] * selected[:, 2])


def fixed_grid_centroids(cloud: PointCloudData, voxel_size: float) -> PointCloudData:
    """Centroids in origin-anchored floor-indexed cells, as in field capture."""
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel size must be finite and positive")
    points = cloud.points
    if not len(points):
        return PointCloudData(points.copy())
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((int(inverse.max()) + 1, 3))
    np.add.at(sums, inverse, points)
    return PointCloudData(sums / np.bincount(inverse)[:, None])


def voxel_downsample(cloud: PointCloudData, voxel_size: float) -> PointCloudData:
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel size must be positive")
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D is required for voxel downsampling") from error
    value = o3d.geometry.PointCloud()
    value.points = o3d.utility.Vector3dVector(np.array(cloud.points, copy=True))
    value = value.voxel_down_sample(voxel_size)
    return PointCloudData(np.asarray(value.points))
