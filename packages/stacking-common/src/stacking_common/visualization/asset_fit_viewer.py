"""Keyboard review of original stone surfaces and their actual curved DSF fits."""

import hashlib
import json
from pathlib import Path

import numpy as np

from stacking_common.rendering.pyvista import read_dsf_surfaces
from stacking_common.visualization.rendering import configure_scene_passes


class AssetFitViewer:
    """Three linked views: surface, individual fitted DSFs, and transparent overlay."""

    def __init__(self, directory, *, start_id=None, screenshot_directory=None, off_screen=False):
        import pyvista as pv

        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest["status"] != "complete":
            raise ValueError("the asset catalog is not complete")
        self.ids = sorted(int(value) for value in self.manifest["models"])
        if start_id is None:
            start_id = next((oid for oid in self.ids
                if self.manifest["models"][str(oid)]["kind"] == "synthetic"), self.ids[0])
        if start_id not in self.ids:
            raise ValueError(f"stone {start_id} is not in this catalog")
        self.index = self.ids.index(start_id)
        self.screenshot_directory = Path(screenshot_directory or Path.cwd())
        self.audit = {}
        if (self.directory / "validation.json").exists():
            audit = json.loads((self.directory / "validation.json").read_text())
            self.audit = {row["object_id"]: row for row in audit["models"]}
        self.plot = pv.Plotter(shape=(1, 3), off_screen=off_screen, window_size=(1800, 800),
            border=False, title="Stone surface / fitted DSF review")
        bindings = {"Right": self.next, "space": self.next, "n": self.next,
            "Left": self.previous, "p": self.previous, "Home": self.first, "End": self.last,
            "r": self.reset_view, "s": self.save_screenshot, "q": self.close, "Escape": self.close}
        for key, callback in bindings.items():
            self.plot.iren.clear_events_for_key(key)
            self.plot.add_key_event(key, callback)
        self.display(reset_view=True)

    @property
    def object_id(self):
        return self.ids[self.index]

    def next(self):
        self.index = (self.index+1) % len(self.ids)
        self.display()

    def previous(self):
        self.index = (self.index-1) % len(self.ids)
        self.display()

    def first(self):
        self.index = 0
        self.display()

    def last(self):
        self.index = len(self.ids)-1
        self.display()

    def display(self, *, reset_view=False):
        import pyvista as pv

        camera = None if reset_view else self.plot.camera_position
        record = self.manifest["models"][str(self.object_id)]
        for name, expected in record["files"].items():
            if hashlib.sha256((self.directory / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"catalog asset changed: {name}")
        surface = pv.read(self.directory / f"model_{self.object_id}_mesh.ply")
        self.components = read_dsf_surfaces(self.directory / f"model_{self.object_id}.obj")
        self.cloud = np.vstack([surface.points, *(mesh.points for mesh in self.components)])
        palette = ("#d77938", "#2c9da7", "#6386c7", "#ad6faa", "#7a9e43", "#c75464")
        material = dict(smooth_shading=True, show_edges=False, ambient=.30, diffuse=.70, specular=.05)
        self.plot.clear()
        self.plot.enable_lightkit()
        for column in range(3):
            self.plot.subplot(0, column)
            self.plot.set_background("white")
            # SSAO delegate passes do not compose correctly across viewports.
            configure_scene_passes(self.plot, translucent=True,
                                   occlusion_radius=.20, occlusion_bias=.01)
            if column in (0, 2):
                self.plot.add_mesh(surface, color="#969696", opacity=.45 if column == 2 else 1.,
                                   culling="back", **material)
            if column in (1, 2):
                for index, mesh in enumerate(self.components):
                    self.plot.add_mesh(mesh, color=palette[index % len(palette)],
                        opacity=.45 if column == 2 else 1., culling="back", **material)
            title = ("Original / deformed surface", "Fitted DSF components", "Surface + DSF overlay")[column]
            self.plot.add_text(title, position="upper_left", font_size=13, color="#252525")
        self.plot.subplot(0, 0)
        parent = f"; parent {record['parent_id']}" if "parent_id" in record else ""
        details = f"Stone {self.object_id}  [{self.index+1}/{len(self.ids)}]  {record['kind']}{parent}"
        details += "\nRight / Space / N: next   Left / P: previous"
        details += "\nHome / End: first / last   R: reset view"
        details += "\nS: screenshot   Q / Esc: close"
        self.plot.add_text(details, position="lower_left", font_size=10, color="#252525")
        self.plot.subplot(0, 1)
        self.plot.add_text(f"{len(self.components)} components; colours identify parts\n"
                           "Curved DSF surfaces, not OBJ node hulls",
                           position="lower_left", font_size=10, color="#252525")
        self.plot.subplot(0, 2)
        audit = self.audit.get(self.object_id)
        text = "Grey: original surface   Colours: fitted DSFs"
        if audit:
            text += f"\nSurface / DSF volume IoU: {audit['surface_dsf_voxel_iou']:.3f}"
            text += f"   Mass: {audit['mass_kg']:.1f} kg"
        self.plot.add_text(text, position="lower_left", font_size=10, color="#252525")
        self.plot.link_views()
        self._fit_camera(camera)
        self.plot.render()

    def _fit_camera(self, camera=None):
        direction = np.array([1.7, -2.5, 1.3]) if camera is None else np.subtract(camera[0], camera[1])
        direction /= np.linalg.norm(direction)
        up = np.array([0., 0., 1.]) if camera is None else np.array(camera[2])
        right = np.cross(up, direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        projected = self.cloud @ np.array([right, up]).T
        center = (self.cloud.min(axis=0)+self.cloud.max(axis=0))/2
        midpoint = (projected.min(axis=0)+projected.max(axis=0))/2
        center += (midpoint-center @ np.array([right, up]).T) @ np.array([right, up])
        span = float(np.ptp(self.cloud, axis=0).max())
        self.plot.camera_position = [center+direction*span*4, center, up]
        self.plot.enable_parallel_projection()
        extent = np.ptp(projected, axis=0)
        self.plot.camera.parallel_scale = max(extent[1], extent[0]/(600/800))*.72
        self.plot.camera.clipping_range = (.001, span*20)

    def reset_view(self):
        self._fit_camera()
        self.plot.render()

    def save_screenshot(self):
        self.screenshot_directory.mkdir(parents=True, exist_ok=True)
        path = self.screenshot_directory / f"stone_{self.object_id:03d}_fit.png"
        self.plot.screenshot(path)
        print(f"Saved {path}", flush=True)
        return path

    def show(self):
        self.plot.show()

    def close(self):
        self.plot.close()
