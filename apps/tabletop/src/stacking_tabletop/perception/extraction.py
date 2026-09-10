"""Tabletop object extraction strategies and operator selection UI."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from stacking_common.pointcloud import PointCloudData

from .capture import TabletopCaptureView
from .reconstruction import TriangleMeshData


@dataclass(frozen=True)
class ManualObjectSelection:
    bounds: np.ndarray
    removal_centers: np.ndarray
    removal_distance: float
    maximum_remove_ratio: float

    def __post_init__(self) -> None:
        bounds = np.asarray(self.bounds, dtype=np.float64)
        centers = np.asarray(self.removal_centers, dtype=np.float64).reshape(-1, 3)
        if bounds.shape != (6,) or not np.all(np.isfinite(bounds)):
            raise ValueError("manual object bounds must contain six finite values")
        if np.any(bounds[[1, 3, 5]] <= bounds[[0, 2, 4]]):
            raise ValueError("manual object bounds must have positive extent")
        if not np.all(np.isfinite(centers)):
            raise ValueError("manual removal centers must be finite")
        if not np.isfinite(self.removal_distance) or self.removal_distance <= 0.0:
            raise ValueError("manual removal distance must be positive")
        if not 0.0 < self.maximum_remove_ratio < 1.0:
            raise ValueError("manual maximum removal ratio must lie in (0, 1)")
        bounds.setflags(write=False)
        centers.setflags(write=False)
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "removal_centers", centers)


class ManualObjectExtractor:
    def __init__(self, selection: ManualObjectSelection) -> None:
        self.selection = selection

    def extract(self, cloud: PointCloudData, context: object = None) -> PointCloudData:
        points = cloud.points
        bounds = self.selection.bounds
        lower, upper = bounds[[0, 2, 4]], bounds[[1, 3, 5]]
        selected = points[np.all((points >= lower) & (points <= upper), axis=1)]
        if not len(selected) or not len(self.selection.removal_centers):
            return PointCloudData(selected)
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(self.selection.removal_centers).query(selected, k=1)
        remove = distances <= self.selection.removal_distance
        if float(np.mean(remove)) > self.selection.maximum_remove_ratio:
            return PointCloudData(selected)
        return PointCloudData(selected[~remove])


class InteractiveObjectSelector:
    """Select one reusable tool-frame box and optional gripper regions."""

    def __init__(self, removal_distance: float, maximum_remove_ratio: float) -> None:
        self.removal_distance = removal_distance
        self.maximum_remove_ratio = maximum_remove_ratio

    def select(self, merged: PointCloudData) -> ManualObjectSelection:
        if not len(merged):
            raise ValueError("cannot select an object from an empty merge")
        try:
            import pyvista as pv
        except ImportError as error:
            raise RuntimeError(
                "PyVista is required for interactive object separation"
            ) from error
        points = np.asarray(merged.points)
        cloud = pv.PolyData(points)
        cloud["z"] = points[:, 2]
        box = {"surface": pv.Box(bounds=cloud.bounds)}
        plotter = pv.Plotter(title="Select the object region")
        plotter.add_points(cloud, scalars="z", point_size=4)
        plotter.add_text(
            "Move and resize the green box around points to KEEP.\n"
            "Close the window when finished.",
            position="upper_left",
            font_size=10,
        )
        plotter.add_box_widget(
            callback=lambda value: box.__setitem__(
                "surface", pv.wrap(value).copy(deep=True)
            ),
            bounds=cloud.bounds,
            rotation_enabled=False,
            color="green",
            use_planes=False,
        )
        plotter.show()
        plotter.close()
        bounds = np.asarray(box["surface"].bounds, dtype=np.float64)
        lower, upper = bounds[[0, 2, 4]], bounds[[1, 3, 5]]
        cropped = points[np.all((points >= lower) & (points <= upper), axis=1)]
        if not len(cropped):
            raise ValueError("interactive object box selected no points")
        centers = self._pick_removal_centers(pv, cropped)
        return ManualObjectSelection(
            bounds,
            centers,
            self.removal_distance,
            self.maximum_remove_ratio,
        )

    def _pick_removal_centers(self, pv, points: np.ndarray) -> np.ndarray:
        picked = []
        cloud = pv.PolyData(points)
        cloud["z"] = points[:, 2]
        plotter = pv.Plotter(title="Select gripper regions")
        plotter.add_points(cloud, scalars="z", point_size=4)
        plotter.add_text(
            "Press P over unwanted gripper points. Press U to undo.\n"
            "Close without picks when the box already contains only the object.",
            position="upper_left",
            font_size=10,
        )

        def pick(point) -> None:
            if point is None:
                return
            center = np.asarray(point, dtype=np.float64)
            picked.append(center)
            plotter.add_mesh(
                pv.Sphere(radius=self.removal_distance, center=center),
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
            left_clicking=False,
            show_point=True,
            point_size=12,
            color="red",
        )
        plotter.add_key_event("u", undo)
        plotter.add_key_event("U", undo)
        plotter.show()
        plotter.close()
        return np.asarray(picked, dtype=np.float64).reshape(-1, 3)


class ToolGeometryProvider(Protocol):
    def __call__(self, view: TabletopCaptureView) -> tuple[TriangleMeshData, ...]: ...


class ModeledGripperExtractor:
    """Remove points near platform-provided gripper meshes in tool coordinates."""

    def __init__(self, geometry_provider: ToolGeometryProvider, surface_margin: float) -> None:
        if not np.isfinite(surface_margin) or surface_margin <= 0.0:
            raise ValueError("gripper surface margin must be positive")
        self.geometry_provider = geometry_provider
        self.surface_margin = surface_margin

    def extract(
        self,
        cloud: PointCloudData,
        context: TabletopCaptureView,
    ) -> PointCloudData:
        if not len(cloud):
            return cloud
        try:
            import open3d as o3d
        except ImportError as error:
            raise RuntimeError("Open3D is required for modeled gripper masking") from error
        scene = o3d.t.geometry.RaycastingScene()
        for mesh in self.geometry_provider(context):
            legacy = o3d.geometry.TriangleMesh()
            legacy.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices, copy=True))
            legacy.triangles = o3d.utility.Vector3iVector(np.array(mesh.triangles, copy=True))
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
        distances = scene.compute_distance(
            o3d.core.Tensor(np.asarray(cloud.points, dtype=np.float32))
        ).numpy()
        return PointCloudData(cloud.points[distances > self.surface_margin])
