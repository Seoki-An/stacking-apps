"""Open3D height-map reference used only for parity checks and benchmarks."""

import numpy as np


def render_open3d_height_map(
    meshes: list[tuple[np.ndarray, np.ndarray]],
    resolution: tuple[int, int],
    xlim: np.ndarray,
    ylim: np.ndarray,
) -> np.ndarray:
    """Render top-surface heights with the legacy Open3D ray-cast method."""

    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError(
            "Open3D is required for height-map reference rendering"
        ) from error

    height, width = resolution
    output = np.zeros((height, width), dtype=np.float32)
    valid_meshes = [
        (np.asarray(points, dtype=float), np.asarray(faces, dtype=int))
        for points, faces in meshes
        if len(faces) > 0
    ]
    if not valid_meshes:
        return output

    ray_origin_z = max(float(np.max(points[:, 2])) for points, _ in valid_meshes)
    ray_origin_z += 1.0
    x = np.linspace(xlim[0], xlim[1], width, dtype=np.float32)
    y = np.linspace(ylim[0], ylim[1], height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    rays = np.zeros((height * width, 6), dtype=np.float32)
    rays[:, 0] = xx.reshape(-1)
    rays[:, 1] = yy.reshape(-1)
    rays[:, 2] = ray_origin_z
    rays[:, 5] = -1.0

    scene = o3d.t.geometry.RaycastingScene()
    for points, faces in valid_meshes:
        mesh = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(points.astype(np.float32)),
            o3d.core.Tensor(faces.astype(np.int32)),
        )
        scene.add_triangles(mesh)
    distance = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    hit = np.isfinite(distance)
    output.reshape(-1)[hit] = np.maximum(
        ray_origin_z - distance[hit],
        0.0,
    )
    return output
