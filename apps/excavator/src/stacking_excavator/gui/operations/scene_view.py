"""PyVistaQt adapter for operations-console render frames."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from stacking_common.profiles.visualization import SceneBox, VisualizationProfile
from stacking_common.visualization.dsf import object_surface_path
from stacking_common.rendering.pyvista import open_target_shell, read_dsf_surface
from stacking_common.visualization.rendering import (
    configure_scene_passes, object_surface_culling, scene_is_translucent, stone_opacity,
)

from .controller import OperationsFrame, RenderObject


_COLORS = (
    "#5da5da",
    "#f17c67",
    "#60bd68",
    "#b291d2",
    "#f3ae4e",
    "#4dbea8",
    "#d77fb3",
)

# The legacy Open3D candidate viewer used ``defaultLit`` with the material's
# fully rough defaults.  VTK's Phong material is glossier by default, so keep
# its response diffuse while smoothing only the reconstructed stone normals.
_LEGACY_BACKGROUND = "#1f1f1f"
_LEGACY_SCENE_STONE = "#a8a8a8"
_LEGACY_KEY_LIGHT_ELEVATION = 45.0
_LEGACY_KEY_LIGHT_AZIMUTH = -35.0
_LEGACY_KEY_LIGHT_INTENSITY = 0.8
_LEGACY_OCCLUSION_RADIUS = 0.3
_LEGACY_OCCLUSION_BIAS = 0.015
_ROUGH_MESH_MATERIAL = {
    "lighting": True,
    "ambient": 0.42,
    "diffuse": 0.58,
    "specular": 0.0,
}
_ROBOT_VISUAL_MATERIAL = {
    "lighting": True,
    "ambient": 0.25,
    "diffuse": 0.65,
    "specular": 0.35,
    "specular_power": 40.0,
}


class OperationsSceneView(QtInteractor):
    """Render common values while resolving excavator mesh assets locally."""

    def __init__(
        self,
        parent,
        asset_directory: Path,
        visualization: VisualizationProfile,
    ) -> None:
        super().__init__(parent)
        self.asset_directory = Path(asset_directory)
        self.visualization = visualization
        self._mesh_cache: dict[tuple[Path, str], pv.DataSet] = {}
        self._object_geometry_mode = "dsf"
        self._transparent_stones = False
        self._white_background = False
        self._robot_mesh_cache: dict[tuple[Path, str], pv.DataSet] = {}
        self.show_empty()

    def show_empty(self) -> None:
        self.suppress_rendering = True
        try:
            self._prepare_scene()
            self.add_text(
                "Open a stacking run in Search Inspector",
                position="upper_left",
                color=self._text_color,
                font_size=11,
            )
        finally:
            self.suppress_rendering = False
        self.render()

    def render_perception(self, proposal, cloud_path, *, layers, reset_camera):
        """Observed points and review overlays in the proposal's explicit frame."""
        from scipy.spatial.transform import Rotation

        key = (None if proposal is None else proposal["sha256"], str(cloud_path),
               tuple(layers.items()), self._object_geometry_mode)
        if not reset_camera and key == getattr(self, "_perception_key", None) and getattr(self, "_showing_perception", False):
            return
        camera = None if reset_camera else self.camera_position
        self.suppress_rendering = True
        try:
            self._prepare_scene(translucent=True)
            if proposal is not None:
                if cloud_path and layers.get("Observed cloud", True):
                    cloud = pv.read(cloud_path)
                    if cloud.n_points:
                        self.add_points(cloud.points, name="perception/cloud", color="#aaaaaa", point_size=3,
                                        render_points_as_spheres=False)
                for obj in proposal["objects"]:
                    for field, label, color in (("initial_pose", "Initial poses", (.3, .6, 1.)),
                                                ("pose", "Proposed poses", (.3, .9, .4) if obj["valid"] else (1., .3, .2))):
                        if layers.get(label, True) and obj.get(field) is not None:
                            matrix = np.array(obj[field])
                            pose = np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()]
                            self._add_object(RenderObject(f"perception/{field}/{obj['object_id']}",
                                obj["object_id"], pose, "proposal", opacity=.3, color=color))
                crop = proposal.get("crop")
                if crop and layers.get("Crop boundary", True):
                    extent = np.array(crop["half_extents"]) + crop["margins"]
                    box = pv.Box(bounds=tuple(v for e in extent for v in (-e, e)))
                    actor = self.add_mesh(box, name="perception/crop", style="wireframe", color="#f0cb55")
                    actor.user_matrix = np.array(crop["reference_T_box"])
                self.add_text(f"{proposal['kind']} · {proposal['frame']} · revision {proposal['scene_revision']}\n"
                    "Gray: processed observation · Blue: initial · Green/red: proposed",
                    font_size=10, color=self._text_color)
            else:
                self.add_text("Open a crop and initial poses, or a saved proposal", font_size=11, color=self._text_color)
            if camera is None:
                self.reset_camera()
            else:
                self.camera_position = camera
        finally:
            self.suppress_rendering = False
        self._perception_key = key
        self._showing_perception = True
        self.render()

    def render_frame(
        self,
        frame: OperationsFrame,
        *,
        reset_camera: bool,
        environment_visibility: Mapping[str, bool],
    ) -> None:
        scene_key = _robot_scene_key(frame, environment_visibility)
        if (not reset_camera and scene_key is not None
                and scene_key == getattr(self, "_last_robot_scene_key", None)):
            # Playback changes transforms, not meshes or static scene actors.
            for part in frame.robot_parts:
                self.actors[f"plan/{part.part_id}"].user_matrix = _pose_matrix(part.pose)
            for value in frame.objects:
                self.actors[value.instance_id].user_matrix = _pose_matrix(value.pose)
            self.render()
            return
        previous_camera = None if reset_camera else self.camera_position
        self.suppress_rendering = True
        try:
            self._populate_frame(frame, previous_camera, environment_visibility)
        finally:
            self.suppress_rendering = False
        self._last_robot_scene_key = scene_key
        self.render()

    def _populate_frame(
        self,
        frame: OperationsFrame,
        previous_camera,
        environment_visibility: Mapping[str, bool],
    ) -> None:
        """Replace all actors while intermediate renders remain suppressed."""

        boxes = self.visualization.boxes if frame.environment_boxes is None else frame.environment_boxes
        self._prepare_scene(translucent=scene_is_translucent(
            frame, boxes, environment_visibility, transparent_stones=self._transparent_stones))
        if frame.environment_boxes is None:
            self._add_environment(environment_visibility)
        else:
            for component in frame.environment_boxes:
                if environment_visibility.get(component.layer, False):
                    self._add_scene_box(component)
        if frame.target is not None and environment_visibility.get("target", True):
            self._add_target(frame.target)
            if frame.target_transform is not None:
                self.actors["target"].user_matrix = _pose_matrix(frame.target_transform)
        collision_overlay = any(value.geometry_kind == "visual" for value in frame.robot_parts)
        for value in frame.robot_parts:
            self._add_robot_part(value, collision_overlay=collision_overlay)
        for value in frame.placed_objects:
            self._add_object(value, frame.object_mesh_paths)
        for value in frame.overlays:
            self._add_object(value, frame.object_mesh_paths)
        if frame.contacts:
            points = np.vstack([value.point for value in frame.contacts])
            self.add_points(
                points,
                color="#f4d35e",
                point_size=12,
                render_points_as_spheres=True,
            )
        if frame.forces:
            origins = np.vstack([value.point for value in frame.forces])
            forces = np.vstack([value.force for value in frame.forces])
            largest = max(float(np.linalg.norm(forces, axis=1).max()), 1e-12)
            self.add_arrows(
                origins,
                forces / largest,
                mag=0.35,
                color="#ff5964",
            )
        self.add_text(
            f"Step {frame.step}",
            position="upper_left",
            color=self._text_color,
            font_size=11,
        )
        if previous_camera is not None:
            self.camera_position = previous_camera
        elif frame.camera is not None:
            self.camera_position = (
                frame.camera.eye.tolist(),
                frame.camera.look_at.tolist(),
                frame.camera.up.tolist(),
            )
            self.camera.view_angle = frame.camera.vertical_fov_degrees
        else:
            self.reset_camera()

    def _prepare_scene(self, *, translucent: bool = False) -> None:
        self._last_robot_scene_key = None
        self._showing_perception = False
        self.clear()
        self.set_background("#ffffff" if self._white_background else _LEGACY_BACKGROUND)
        self._configure_legacy_lighting()
        configure_scene_passes(
            self, translucent=translucent,
            occlusion_radius=_LEGACY_OCCLUSION_RADIUS,
            occlusion_bias=_LEGACY_OCCLUSION_BIAS,
        )
        self.add_axes(color=self._text_color)

    @property
    def _text_color(self):
        return "#252525" if self._white_background else "#d5dbe3"

    def set_white_background(self, enabled: bool) -> None:
        self._white_background = bool(enabled)
        self._last_robot_scene_key = None
        self._perception_key = None

    def _configure_legacy_lighting(self) -> None:
        """Use one fixed rough-surface key light instead of camera lights."""

        key_light = pv.Light(
            light_type="scene light",
            color="#ffffff",
            intensity=_LEGACY_KEY_LIGHT_INTENSITY,
        )
        key_light.set_direction_angle(
            _LEGACY_KEY_LIGHT_ELEVATION,
            _LEGACY_KEY_LIGHT_AZIMUTH,
        )
        self.remove_all_lights(only_active=True)
        self.add_light(key_light, only_active=True)

    def _add_environment(self, visibility: Mapping[str, bool]) -> None:
        for component in self.visualization.boxes:
            if visibility.get(component.layer, False):
                self._add_scene_box(component)

    def _add_scene_box(self, component: SceneBox) -> None:
        transparent = component.opacity < 1.0
        self.add_mesh(
            pv.Box(bounds=component.bounds),
            name=f"environment/{component.component_id}",
            color=component.color,
            opacity=component.opacity,
            **_ROUGH_MESH_MATERIAL,
            smooth_shading=False,
            show_edges=transparent,
            edge_color="#ffffff",
            edge_opacity=0.35 if transparent else 1.0,
            pickable=False,
        )

    def _add_target(self, target) -> None:
        self.add_mesh(
            open_target_shell(target),
            color="#ffffff",
            opacity=0.12,
            show_edges=True,
            edge_color="#ffffff",
            edge_opacity=0.60,
            line_width=2,
            **_ROUGH_MESH_MATERIAL,
            smooth_shading=False,
            pickable=False,
            name="target",
        )

    def _add_object(self, value: RenderObject, mesh_paths=None) -> None:
        opacity = stone_opacity(value.opacity, self._transparent_stones)
        mesh = self._object_mesh(value.object_id) if mesh_paths is None else self._recorded_object_mesh(mesh_paths[value.object_id])
        color = value.color or _COLORS[value.object_id % len(_COLORS)]
        if value.layer in {"placed", "trajectory", "final"}:
            color = _LEGACY_SCENE_STONE
        actor = self.add_mesh(
            mesh,
            name=value.instance_id,
            color=color,
            opacity=opacity,
            culling=object_surface_culling(opacity),
            show_edges=False,
            **_ROUGH_MESH_MATERIAL,
            smooth_shading=True,
        )
        actor.user_matrix = _pose_matrix(value.pose)

    def set_transparent_stones(self, enabled: bool) -> None:
        self._transparent_stones = bool(enabled)
        self._last_robot_scene_key = None
        self._perception_key = None

    def set_object_geometry_mode(self, mode: str) -> None:
        if mode not in {"dsf", "ply"}:
            raise ValueError(f"Unknown object geometry mode: {mode}")
        if mode != self._object_geometry_mode:
            self._object_geometry_mode = mode
            self._last_robot_scene_key = None

    def _recorded_object_mesh(self, path):
        mode = self._object_geometry_mode
        path = object_surface_path(path, mode)
        key = (path, mode)
        if key not in self._mesh_cache:
            self._mesh_cache[key] = read_dsf_surface(path) if mode == "dsf" else pv.read(path)
        return self._mesh_cache[key]

    def _object_mesh(self, object_id: int) -> pv.DataSet:
        return self._recorded_object_mesh(self.asset_directory / f"model_{object_id}.obj")

    def _add_robot_part(self, value, *, collision_overlay: bool = False) -> None:
        collision = value.geometry_kind == "collision"
        wireframe = collision and collision_overlay
        preview = value.part_id.startswith('scan_preview/')
        mesh = self._robot_mesh(value.mesh_path, value.geometry_kind).copy()
        mesh.scale(value.mesh_scale, inplace=True)
        appearance = {"color": "#ffb347" if collision else value.color}
        if not collision and not preview and not value.part_id.startswith("gripper/"):
            for name in ("RGBA", "RGB"):
                if name in mesh.point_data:
                    appearance = {"scalars": name, "rgb": True, "preference": "point"}
                    break
        actor = self.add_mesh(
            mesh,
            name=f"plan/{value.part_id}",
            **appearance,
            opacity=.3 if preview else 1.0,
            style="wireframe" if wireframe else "surface",
            line_width=1.5 if wireframe else 1.0,
            show_edges=False,
            **(_ROUGH_MESH_MATERIAL if collision else _ROBOT_VISUAL_MATERIAL),
            smooth_shading=not collision or "dsf_surface" in mesh.field_data,
            show_scalar_bar=False,
            pickable=False,
        )
        actor.user_matrix = _pose_matrix(value.pose)

    def _robot_mesh(self, path: Path, geometry_kind: str = "visual") -> pv.DataSet:
        path = Path(path)
        key = (path, geometry_kind)
        cached = self._robot_mesh_cache.get(key)
        if cached is not None:
            return cached
        mesh = (read_dsf_surface(path) if geometry_kind == "collision" and path.suffix.lower() == ".obj"
                else _prepare_robot_mesh(pv.read(path), geometry_kind))
        self._robot_mesh_cache[key] = mesh
        return mesh


def _robot_scene_key(frame, environment_visibility):
    """Reuse actors when only object/robot transforms change."""
    if frame.contacts or frame.forces:
        return None
    return (
        frame.step, frame.active_candidate_id, repr(frame.target), repr(frame.camera), repr(frame.target_transform),
        tuple(sorted(environment_visibility.items())),
        repr(frame.environment_boxes), repr(frame.object_mesh_paths),
        tuple((o.instance_id, o.object_id, o.layer, o.selected,
               o.opacity, o.color) for o in frame.objects),
        tuple((p.part_id, str(p.mesh_path), p.geometry_kind, p.mesh_scale, p.color)
              for p in frame.robot_parts),
    )


def _prepare_robot_mesh(mesh: pv.PolyData, geometry_kind: str) -> pv.PolyData:
    """Repair collision winding and smooth visual normals without moving vertices.

    Polygon collision meshes retain their flat facets and outward normals.
    DSF assets are tessellated separately by read_dsf_surface before rendering;
    their stored node-hull triangles never pass through this fallback.
    """
    if geometry_kind == "collision":
        return mesh.triangulate().clean().compute_normals(
            cell_normals=True, point_normals=False, consistent_normals=True,
            auto_orient_normals=True, non_manifold_traversal=False,
        )
    # Both STL facets and the imported PLY normals can be flat per triangle.
    # Weld coincident vertices and recompute smooth normals, splitting only
    # genuine mechanical edges. Keep the original RGB/RGBA vertex materials.
    return mesh.clean().compute_normals(
        cell_normals=True, point_normals=True, consistent_normals=True,
        split_vertices=True, feature_angle=45.0,
    )


def _pose_matrix(pose: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(pose[3:], dtype=float)
    x, y, z, w = np.array([x, y, z, w]) / np.linalg.norm([x, y, z, w])
    rotation = np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = pose[:3]
    return transform
