"""Shared point-cloud surface reconstruction implementations."""

import numpy as np

from stacking_common.pointcloud import PointCloudData

from .config import MeshReconstructionProfile
from .types import TriangleMeshData


class MeshReconstructor:
    """Build a surface mesh with explicit platform metric scales."""

    def __init__(self, profile: MeshReconstructionProfile) -> None:
        self.profile = profile

    def reconstruct(self, cloud: PointCloudData) -> TriangleMeshData:
        o3d = _open3d()
        points = _open3d_cloud(o3d, cloud)
        points = points.voxel_down_sample(self.profile.voxel_size)
        if len(points.points) < 4:
            raise ValueError("at least four downsampled points are required")
        if self.profile.method == "radial":
            mesh = _radial_mesh(o3d, points, self.profile)
        elif self.profile.method == "alpha":
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                points,
                self.profile.alpha,
            )
        else:
            points.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.profile.normal_radius,
                    max_nn=self.profile.normal_max_neighbors,
                )
            )
            neighbors = min(
                self.profile.normal_max_neighbors,
                len(points.points) - 1,
            )
            points.orient_normals_consistent_tangent_plane(neighbors)
            if self.profile.method == "bpa":
                distances = np.asarray(
                    points.compute_nearest_neighbor_distance()
                )
                radius = float(np.median(distances))
                radii = o3d.utility.DoubleVector(
                    [radius * scale for scale in self.profile.bpa_radius_scales]
                )
                mesh = (
                    o3d.geometry.TriangleMesh
                    .create_from_point_cloud_ball_pivoting(points, radii)
                )
            else:
                mesh, densities = (
                    o3d.geometry.TriangleMesh
                    .create_from_point_cloud_poisson(
                        points,
                        depth=self.profile.poisson_depth,
                    )
                )
                if (
                    len(densities)
                    and self.profile.poisson_density_quantile > 0.0
                ):
                    density = np.asarray(densities)
                    threshold = np.quantile(
                        density,
                        self.profile.poisson_density_quantile,
                    )
                    mesh.remove_vertices_by_mask(density < threshold)
                bounds = points.get_axis_aligned_bounding_box()
                mesh = mesh.crop(bounds.scale(1.03, bounds.get_center()))
        _clean_mesh(mesh, self.profile.keep_largest_component)
        if self.profile.simplification_voxel_size > 0.0:
            mesh = mesh.simplify_vertex_clustering(
                voxel_size=self.profile.simplification_voxel_size,
                contraction=(
                    o3d.geometry.SimplificationContraction.Average
                ),
            )
            _clean_mesh(mesh, self.profile.keep_largest_component)
        if self.profile.smoothing_iterations:
            mesh = mesh.filter_smooth_taubin(
                number_of_iterations=self.profile.smoothing_iterations
            )
            _clean_mesh(mesh, self.profile.keep_largest_component)
        if not len(mesh.vertices) or not len(mesh.triangles):
            raise ValueError(
                f"{self.profile.method} reconstruction produced an empty mesh"
            )
        mesh.compute_vertex_normals()
        return _mesh_data(mesh)


def repair_triangle_mesh(mesh: TriangleMeshData) -> TriangleMeshData:
    """Apply conservative topology cleanup without inventing surface patches."""

    o3d = _open3d()
    value = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.array(mesh.vertices, copy=True)),
        o3d.utility.Vector3iVector(np.array(mesh.triangles, copy=True)),
    )
    value.remove_duplicated_vertices()
    value.remove_duplicated_triangles()
    value.remove_degenerate_triangles()
    value.remove_non_manifold_edges()
    value.remove_unreferenced_vertices()
    value.orient_triangles()
    if not len(value.vertices) or not len(value.triangles):
        return mesh
    return _mesh_data(value)



def _radial_mesh(o3d, cloud, profile: MeshReconstructionProfile):
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:
        raise RuntimeError(
            "SciPy is required for radial mesh reconstruction"
        ) from error
    points = np.asarray(cloud.points)
    center = np.quantile(points, profile.radial_center_quantile, axis=0)
    vectors = points - center
    radii = np.linalg.norm(vectors, axis=1)
    valid = radii > 1e-9
    if np.count_nonzero(valid) < 4:
        raise ValueError("point cloud cannot define a radial mesh")
    point_directions = vectors[valid] / radii[valid, None]
    point_radii = radii[valid]
    sphere = o3d.geometry.TriangleMesh.create_icosahedron(radius=1.0)
    if profile.radial_subdivisions:
        sphere = sphere.subdivide_midpoint(
            number_of_iterations=profile.radial_subdivisions
        )
    sphere_vertices = np.asarray(sphere.vertices)
    sphere_directions = sphere_vertices / np.linalg.norm(
        sphere_vertices, axis=1
    )[:, None]
    tree = cKDTree(point_directions)
    count = min(profile.radial_neighbors, len(point_directions))
    distance, neighbors = tree.query(sphere_directions, k=count)
    if count == 1:
        distance = distance[:, None]
        neighbors = neighbors[:, None]
    surface_radii = np.quantile(
        point_radii[neighbors],
        profile.radial_radius_quantile,
        axis=1,
    )
    nearest_angle = 2.0 * np.arcsin(
        np.clip(distance[:, 0] * 0.5, 0.0, 1.0)
    )
    empty_angle = np.deg2rad(profile.radial_empty_angle_degrees)
    if empty_angle > 0.0:
        fill = min(
            float(np.quantile(point_radii, profile.radial_empty_radius_quantile)),
            float(np.median(surface_radii)),
        )
        weight = np.clip(nearest_angle / empty_angle, 0.0, 1.0)
        surface_radii = (1.0 - weight) * surface_radii + weight * fill
    support_angle = np.deg2rad(profile.radial_support_angle_degrees)
    if support_angle > 0.0 and profile.radial_minimum_support_points > 0:
        chord = 2.0 * np.sin(support_angle * 0.5)
        support = np.fromiter(
            (len(ids) for ids in tree.query_ball_point(sphere_directions, r=chord)),
            dtype=np.int32,
            count=len(sphere_directions),
        )
        fill = min(
            float(np.quantile(point_radii, profile.radial_empty_radius_quantile)),
            float(np.median(surface_radii)),
        )
        weight = np.clip(
            (profile.radial_minimum_support_points - support)
            / max(profile.radial_minimum_support_points, 1),
            0.0,
            1.0,
        )
        surface_radii = (1.0 - weight) * surface_radii + weight * fill
    tighten_angle = np.deg2rad(profile.radial_tighten_angle_degrees)
    if tighten_angle > 0.0 and profile.radial_tighten_blend > 0.0:
        chord = 2.0 * np.sin(tighten_angle * 0.5)
        tightened = surface_radii.copy()
        for index, ids in enumerate(
            tree.query_ball_point(sphere_directions, r=chord)
        ):
            if ids:
                tightened[index] = np.quantile(
                    point_radii[ids],
                    profile.radial_tighten_quantile,
                )
        blended = (
            (1.0 - profile.radial_tighten_blend) * surface_radii
            + profile.radial_tighten_blend * tightened
        )
        surface_radii = np.minimum(surface_radii, blended)
    maximum = np.quantile(
        point_radii,
        profile.radial_maximum_radius_quantile,
    )
    surface_radii = np.minimum(surface_radii, maximum)
    surface_radii = _smooth_vertex_values(
        surface_radii,
        np.asarray(sphere.triangles),
        profile.radial_smoothing_iterations,
        profile.radial_smoothing_weight,
    )
    surface_radii = np.maximum(
        surface_radii + profile.radial_radius_padding,
        1e-6,
    )
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        center + sphere_directions * surface_radii[:, None]
    )
    mesh.triangles = sphere.triangles
    return mesh


def _smooth_vertex_values(
    values: np.ndarray,
    triangles: np.ndarray,
    iterations: int,
    weight: float,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    if iterations <= 0 or weight <= 0.0:
        return result
    neighbors = [set() for _ in range(len(result))]
    for first, second, third in triangles:
        neighbors[first].update((second, third))
        neighbors[second].update((first, third))
        neighbors[third].update((first, second))
    for _ in range(iterations):
        smoothed = result.copy()
        for index, adjacent in enumerate(neighbors):
            if adjacent:
                smoothed[index] = np.mean(result[list(adjacent)])
        result = (1.0 - weight) * result + weight * smoothed
    return result


def _farthest_points(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return np.array(points, copy=True)
    selected = np.empty(maximum, dtype=np.int64)
    selected[0] = int(
        np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1))
    )
    distance = np.full(len(points), np.inf)
    for index in range(1, maximum):
        delta = points - points[selected[index - 1]]
        distance = np.minimum(distance, np.einsum("ij,ij->i", delta, delta))
        selected[index] = int(np.argmax(distance))
    return np.array(points[selected], copy=True)


def _open3d_cloud(o3d, cloud: PointCloudData):
    value = o3d.geometry.PointCloud()
    value.points = o3d.utility.Vector3dVector(
        np.array(cloud.points, copy=True)
    )
    return value


def _clean_mesh(mesh, keep_largest_component: bool) -> None:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if keep_largest_component and len(mesh.triangles):
        labels, counts, _ = mesh.cluster_connected_triangles()
        counts = np.asarray(counts)
        if len(counts) > 1:
            remove = np.asarray(labels) != int(np.argmax(counts))
            mesh.remove_triangles_by_mask(remove)
            mesh.remove_unreferenced_vertices()



def _mesh_data(mesh) -> TriangleMeshData:
    return TriangleMeshData(
        np.asarray(mesh.vertices),
        np.asarray(mesh.triangles),
    )


def _open3d():
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError(
            "Open3D is required for tabletop reconstruction"
        ) from error
    return o3d
