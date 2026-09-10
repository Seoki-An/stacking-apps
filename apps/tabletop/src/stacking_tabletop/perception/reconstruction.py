"""Tabletop interaction and visualization adapters for shared reconstruction."""

import numpy as np

from stacking_common.pointcloud import PointCloudData
from stacking_common.reconstruction import (
    DsfComponent,
    DsfExporter,
    InertialMetadata,
    MeshReconstructor,
    PointCloudCleaner,
    ReconstructionOperatorOptions,
    TriangleMeshData,
    dsf_support_points,
    estimate_hull_inertial,
    load_point_cloud,
    write_point_cloud,
    write_triangle_mesh,
)
from stacking_tabletop.config import ReconstructionCleanupProfile
from stacking_tabletop.visualization import add_scaled_pyvista_axes


class TabletopPointCloudCleaner(PointCloudCleaner):
    """Attach the tabletop manual-removal UI to shared headless cleanup."""

    def clean(
        self,
        cloud: PointCloudData,
        options: ReconstructionOperatorOptions,
    ) -> PointCloudData:
        remover = InteractivePointRemover(self.profile)
        return super().clean(
            cloud,
            options,
            manual_remover=lambda value, left: remover.remove(
                value,
                left_clicking=left,
            ),
        )


class InteractivePointRemover:
    """Remove operator-picked point neighborhoods with a safety limit."""

    def __init__(self, profile: ReconstructionCleanupProfile) -> None:
        self.profile = profile

    def remove(
        self,
        cloud: PointCloudData,
        *,
        left_clicking: bool = False,
    ) -> PointCloudData:
        pv = _pyvista()
        source_points = np.asarray(cloud.points)
        points = _principal_axis_aligned_points(source_points)
        aligned = PointCloudData(points)
        picked = []
        plotter = pv.Plotter(title="Remove reconstruction outliers")
        add_scaled_pyvista_axes(plotter, points)
        _add_cloud_to_plotter(pv, plotter, aligned, name="object")
        plotter.add_text(
            "The cloud is centered and principal-axis aligned.\n"
            "Press P over unwanted points to add a removal region.\n"
            "Press U to undo. Close the window to apply removal.",
            position="upper_left",
            font_size=10,
        )

        def pick(point) -> None:
            if point is None:
                return
            center = np.asarray(point, dtype=np.float64)
            picked.append(center)
            plotter.add_mesh(
                pv.Sphere(
                    radius=self.profile.manual_remove_distance,
                    center=center,
                ),
                color="red",
                opacity=0.25,
                name=f"remove_{len(picked)}",
                pickable=False,
            )

        def undo() -> None:
            if picked:
                plotter.remove_actor(f"remove_{len(picked)}")
                picked.pop()

        plotter.enable_point_picking(
            callback=pick,
            left_clicking=left_clicking,
            show_point=True,
            point_size=12,
            color="red",
        )
        plotter.add_key_event("u", undo)
        plotter.add_key_event("U", undo)
        plotter.show()
        plotter.close()
        if not picked:
            return cloud
        centers = np.asarray(picked)
        squared = np.sum(
            (points[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )
        remove = np.min(squared, axis=1) <= self.profile.manual_remove_distance**2
        ratio = float(np.mean(remove))
        if ratio > self.profile.manual_max_remove_ratio:
            raise ValueError(
                f"manual selection would remove {ratio:.1%} of the cloud; "
                f"maximum is {self.profile.manual_max_remove_ratio:.1%}"
            )
        return PointCloudData(source_points[~remove])


def _principal_axis_aligned_points(points: np.ndarray) -> np.ndarray:
    """Center points and express them in a deterministic principal-axis frame."""

    center = np.mean(points, axis=0)
    centered = points - center
    covariance = centered.T @ centered / len(centered)
    values, axes = np.linalg.eigh(covariance)
    axes = axes[:, np.argsort(values)[::-1]]
    for axis in range(3):
        projection = centered @ axes[:, axis]
        extreme = projection[np.argmax(np.abs(projection))]
        if extreme < 0.0:
            axes[:, axis] *= -1.0
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] *= -1.0
    return centered @ axes


TabletopMeshReconstructor = MeshReconstructor
TabletopDsfExporter = DsfExporter


def show_point_cloud(cloud: PointCloudData, title: str) -> None:
    pv = _pyvista()
    plotter = pv.Plotter(title=title)
    add_scaled_pyvista_axes(plotter, cloud.points)
    _add_cloud_to_plotter(pv, plotter, cloud, name="object")
    plotter.show()
    plotter.close()


def show_reconstruction(
    cloud: PointCloudData,
    mesh: TriangleMeshData,
    *,
    components: tuple[DsfComponent, ...] = (),
    mesh_opacity: float = 0.35,
    title: str = "Object reconstruction",
    retry_label: str | None = None,
) -> bool:
    if not 0.0 < mesh_opacity <= 1.0:
        raise ValueError("mesh opacity must lie in (0, 1]")
    pv = _pyvista()
    plotter = pv.Plotter(title=title)
    add_scaled_pyvista_axes(plotter, cloud.points)
    _add_cloud_to_plotter(pv, plotter, cloud, name="object")
    plotter.add_mesh(
        _pyvista_mesh(pv, mesh.vertices, mesh.triangles),
        color="lightsteelblue",
        opacity=mesh_opacity,
        show_edges=True,
        edge_color="black",
        name="mesh",
    )
    colors = ("orange", "limegreen", "magenta", "cyan", "yellow")
    for index, component in enumerate(components):
        plotter.add_mesh(
            _dsf_surface_mesh(pv, component),
            color=colors[index % len(colors)],
            opacity=min(1.0, mesh_opacity + 0.15),
            show_edges=False,
            name=f"dsf_{index}",
        )
    accepted = {"value": True}
    if retry_label is not None:
        plotter.add_text(
            f"Enter / Space: accept\nR: {retry_label}",
            position="upper_left",
            font_size=10,
        )

        def key_press(interactor, _event) -> None:
            key = interactor.GetKeySym()
            if key in ("Return", "KP_Enter", "space", "q", "Escape"):
                accepted["value"] = True
                interactor.TerminateApp()
            elif key in ("r", "R"):
                accepted["value"] = False
                interactor.TerminateApp()

        plotter.iren.interactor.AddObserver("KeyPressEvent", key_press)
    plotter.show()
    plotter.close()
    return accepted["value"]


def _dsf_surface_mesh(pv, component: DsfComponent, subdivisions: int = 4):
    sphere = pv.Icosphere(nsub=subdivisions)
    points = dsf_support_points(
        component.nodes,
        component.sharpness,
        np.asarray(sphere.points),
    )
    return _pyvista_mesh(pv, points, _convex_faces(points))


def _convex_faces(points: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import ConvexHull
    except ImportError as error:
        raise RuntimeError("SciPy is required for DSF surface geometry") from error
    values = np.asarray(points, dtype=np.float64)
    hull = ConvexHull(values)
    faces = np.asarray(hull.simplices, dtype=np.int64).copy()
    center = values.mean(axis=0)
    for index, face in enumerate(faces):
        edge_1 = values[face[1]] - values[face[0]]
        edge_2 = values[face[2]] - values[face[0]]
        normal = np.cross(edge_1, edge_2)
        if np.dot(normal, values[face[0]] - center) < 0.0:
            faces[index] = face[::-1]
    return faces


def _add_cloud_to_plotter(pv, plotter, cloud: PointCloudData, name: str) -> None:
    points = np.asarray(cloud.points)
    value = pv.PolyData(points)
    value["z"] = points[:, 2]
    plotter.add_points(
        value,
        scalars="z",
        point_size=4,
        render_points_as_spheres=False,
        name=name,
    )


def _pyvista_mesh(pv, vertices: np.ndarray, triangles: np.ndarray):
    faces = np.column_stack(
        (np.full(len(triangles), 3, dtype=np.int64), triangles)
    )
    return pv.PolyData(vertices, faces)


def _pyvista():
    try:
        import pyvista as pv
    except ImportError as error:
        raise RuntimeError(
            "PyVista is required for reconstruction visualization and manual cleanup"
        ) from error
    return pv

