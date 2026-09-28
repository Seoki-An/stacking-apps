"""Ideal pinhole visibility tests on triangle scenes; no sensor noise model."""

import numpy as np
import open3d as o3d


class TriangleScene:
    def __init__(self, meshes):
        self.scene = o3d.t.geometry.RaycastingScene()
        for vertices, triangles in meshes:
            self.scene.add_triangles(
                o3d.core.Tensor(np.asarray(vertices, dtype=np.float32)),
                o3d.core.Tensor(np.asarray(triangles, dtype=np.uint32)))

    def cast(self, origins, directions):
        origins = np.broadcast_to(origins, np.shape(directions))
        rays = np.concatenate((origins, directions), axis=-1).astype(np.float32)
        result = self.scene.cast_rays(o3d.core.Tensor(rays))
        return {key: value.numpy() for key, value in result.items()}

    def image(self, matrix, world_T_camera, size):
        """Return ideal axial depth and first-hit geometry for integer pixel centers."""
        width, height = size
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(matrix).T
        rays = rays @ world_T_camera[:3, :3].T
        result = self.cast(world_T_camera[:3, 3], rays)
        result['depth'] = result['t_hit']  # local camera ray z is exactly one
        valid = np.isfinite(result['depth'])
        result['points'] = np.zeros((*valid.shape, 3), dtype=np.float32)
        result['points'][valid] = (world_T_camera[:3, 3]
                                  + rays[valid] * result['depth'][valid, None])
        return result

    def visible(self, points, matrix, world_T_camera, size, *, tolerance=.0005):
        """Frustum and first-hit visibility, including occluders in this scene."""
        points = np.asarray(points)
        camera = (points - world_T_camera[:3, 3]) @ world_T_camera[:3, :3]
        projected = camera @ matrix.T
        uv = projected[:, :2] / np.maximum(projected[:, 2, None], 1e-12)
        inside = ((camera[:, 2] > 0) & (uv[:, 0] >= -.5) & (uv[:, 0] < size[0]-.5)
                  & (uv[:, 1] >= -.5) & (uv[:, 1] < size[1]-.5))
        delta = points - world_T_camera[:3, 3]
        distance = np.linalg.norm(delta, axis=1)
        hit = self.cast(world_T_camera[:3, 3], delta / np.maximum(distance[:, None], 1e-12))
        return inside & (np.abs(hit['t_hit'] - distance) <= tolerance)


def sample_surface(vertices, triangles, count=6000, *, seed=0):
    """Deterministic samples uniform in triangle surface area, with face normals."""
    faces = np.asarray(vertices)[np.asarray(triangles)]
    cross = np.cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    if lengths.sum() <= 0:
        raise ValueError('surface mesh has no area')
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(faces), count, p=lengths/lengths.sum())
    a, b = rng.random((2, count))
    a = np.sqrt(a)
    weights = np.column_stack((1-a, a*(1-b), a*b))
    points = np.sum(faces[chosen] * weights[:, :, None], axis=1)
    return points, cross[chosen] / lengths[chosen, None]


def supported_samples(visible, points, centers, *, minimum_angle_deg=1.):
    """Samples seen from at least two positions with useful geometric parallax."""
    visible, points, centers = np.asarray(visible), np.asarray(points), np.asarray(centers)
    supported = np.zeros(len(points), dtype=bool)
    directions = centers[:, None, :] - points[None, :, :]
    directions /= np.maximum(np.linalg.norm(directions, axis=2, keepdims=True), 1e-12)
    limit = np.cos(np.deg2rad(minimum_angle_deg))
    for i in range(len(centers)):
        for j in range(i):
            cosine = np.sum(directions[i]*directions[j], axis=1)
            supported |= visible[i] & visible[j] & (cosine <= limit)
    return supported


def surface_texture(points):
    """Deterministic broadband 3D value noise anchored to the scene, not the camera."""
    points = np.asarray(points)
    texture = np.zeros(len(points))
    for frequency, weight in ((50., .35), (100., .30), (200., .25), (400., .10)):
        scaled = points * frequency
        cell = np.floor(scaled).astype(np.int64)
        blend = scaled-cell
        blend = blend*blend*(3-2*blend)
        value = np.zeros(len(points))
        for x in (0, 1):
            for y in (0, 1):
                for z in (0, 1):
                    offset = np.array([x, y, z])
                    index = (cell+offset).astype(np.uint32)
                    hashed = (index[:, 0]*np.uint32(73856093) ^ index[:, 1]*np.uint32(19349663)
                              ^ index[:, 2]*np.uint32(83492791))
                    hashed ^= hashed >> np.uint32(16)
                    hashed *= np.uint32(2246822519)
                    hashed ^= hashed >> np.uint32(13)
                    mix = np.prod(np.where(offset, blend, 1-blend), axis=1)
                    value += mix * hashed.astype(float)/np.iinfo(np.uint32).max
        texture += weight*value
    return np.clip(.10 + 1.8*texture, .1, 1.8)
