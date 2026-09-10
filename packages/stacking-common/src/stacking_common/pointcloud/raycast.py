"""Deterministic, headless mesh-to-LiDAR emulation with sensor-frame output."""

import numpy as np

from .types import validate_rigid_transform


def simulate_lidar(meshes, world_T_sensor, *, fov=(120., 70.)):
    """Cast nearest-surface rays using the legacy planner's angular convention.

    Meshes are Open3D legacy triangle meshes already in world coordinates.
    The +X sensor axis is forward, +Z is up. Resolution is the legacy 0.15 by
    0.36 degrees. A 30 m ground plane is included, as in the legacy emulator.
    Returns sensor XYZ and original ray indices (no measurement noise).
    """
    import open3d as o3d

    transform = validate_rigid_transform(world_T_sensor, "world_T_sensor")
    fov = np.asarray(fov, dtype=float)
    if fov.shape != (2,) or not np.all(np.isfinite(fov)) or np.any(fov <= 0) or np.any(fov > 180):
        raise ValueError("LiDAR field of view must be two angles in (0, 180] degrees")
    counts = (int(fov[0] / .15), int(fov[1] / .36))
    if min(counts) < 2:
        raise ValueError("LiDAR field of view needs at least two rays per axis")
    phi, theta = np.meshgrid(
        np.linspace(np.deg2rad(-fov[0] / 2), np.deg2rad(fov[0] / 2), counts[0]),
        np.linspace(np.deg2rad(90 - fov[1] / 2), np.deg2rad(90 + fov[1] / 2), counts[1]))
    phi, theta = phi.ravel(), theta.ravel()
    rays = np.zeros((len(phi), 6), dtype=np.float32)
    rays[:, 3:] = np.stack([np.sin(theta) * np.cos(phi),
                            np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
    rays[:, :3] = transform[:3, 3]
    rays[:, 3:] = np.einsum("ij,nj->ni", transform[:3, :3], rays[:, 3:])
    scene = o3d.t.geometry.RaycastingScene()
    ground = o3d.t.geometry.TriangleMesh.create_box(30., 30., .001).translate((-15, -15, -.001))
    scene.add_triangles(ground)
    for mesh in meshes:
        if not mesh.has_triangles():
            raise ValueError("LiDAR scene geometry must contain triangles")
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    distances = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    selected = np.isfinite(distances)
    world = rays[selected, :3] + distances[selected, None] * rays[selected, 3:]
    sensor_T_world = np.linalg.inv(transform)
    points = world.astype(np.float64) @ sensor_T_world[:3, :3].T + sensor_T_world[:3, 3]
    return points, np.flatnonzero(selected).astype(np.uint32)
