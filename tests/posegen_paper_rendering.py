"""Matched-camera figures for a pose solve and its final force graph."""

import hashlib
import json
from pathlib import Path

import numpy as np


FORCE_ARROW_STYLE = dict(min_length_m=.16, max_length_m=.55, exponent=.5,
                         shaft_radius_m=.006, tip_radius_m=.02, tip_length_m=.055)


def make_force_arrow(point, force, max_force):
    """Use fixed physical widths and a bounded square-root length mapping."""
    import pyvista as pv

    magnitude = np.linalg.norm(force)
    style = FORCE_ARROW_STYLE
    length = style["min_length_m"] + (style["max_length_m"] - style["min_length_m"]) * (
        magnitude / max_force) ** style["exponent"]
    # PyVista scales every glyph dimension by length; compensate the radii
    # and head length so only the total arrow length changes with force.
    return pv.Arrow(start=point, direction=force / magnitude, scale=length,
                    tip_length=style["tip_length_m"] / length,
                    tip_radius=style["tip_radius_m"] / length,
                    shaft_radius=style["shaft_radius_m"] / length)


def load_stone_meshes(directory, object_ids, mode, recorded_hashes):
    """Load the requested display surfaces and retain their exact provenance."""
    import pyvista as pv

    from stacking_common.rendering.pyvista import read_dsf_surface
    from stacking_common.visualization.dsf import object_surface_path

    meshes, sources = {}, {}
    for object_id in object_ids:
        path = object_surface_path(Path(directory) / f"model_{object_id}.obj", mode)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        role = "object" if mode == "dsf" else "object_surface"
        expected = recorded_hashes.get(f"{role}/{object_id}")
        if expected is not None and digest != expected:
            raise ValueError(f"recorded stone surface changed: {path}")
        meshes[object_id] = read_dsf_surface(path) if mode == "dsf" else pv.read(path)
        sources[str(object_id)] = dict(path=str(path.resolve()), sha256=digest)
    return meshes, sources


def render_pose_optimization(directory, meshes, placed_poses, candidate_id, trace, forces,
                             framing_poses=(), ground_margin=.8, ghost_count=7):
    """Render canonical meshes/poses and world-frame force incidences in N.

    ``forces`` contains (object_id, point, force) tuples. Both sides of a
    stone/stone contact are shown: each arrow is the force on its owning stone.
    Yellow/green mark the initial/final candidate; gray ghosts show updates.
    Original display meshes retain their geometry and have no edge outlines.
    Mesh transparency is independent of the opaque PNG background.
    ``framing_poses`` are extra candidate poses the camera must also frame,
    so traces compared side by side share one view. ``ground_margin`` (m)
    extends the ground plane beyond the framed objects. ``ghost_count``
    poses, evenly spaced along the path, are drawn between the endpoints.
    """
    import pyvista as pv
    from scipy.spatial.transform import Rotation

    from stacking_common.visualization.rendering import configure_scene_passes

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    candidate_mesh = meshes[candidate_id]
    poses = np.array([sample.pose for sample in trace.samples])
    final_poses = {**placed_poses, candidate_id: trace.result.pose}

    def transformed(mesh, pose):
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
        matrix[:3, 3] = pose[:3]
        return mesh.transform(matrix, inplace=False)

    displayed = {i: transformed(meshes[i], p) for i, p in final_poses.items()}
    cloud = np.vstack([m.points for m in displayed.values()] +
                      [transformed(candidate_mesh, p).points for p in [*poses, *framing_poses]])
    low, high = cloud.min(axis=0), cloud.max(axis=0)
    ground_center = (low[:2] + high[:2]) / 2
    ground_size = high[:2] - low[:2] + ground_margin
    ground_corners = np.array([[ground_center[0] + x * ground_size[0] / 2,
                                ground_center[1] + y * ground_size[1] / 2, 0.]
                               for x in (-1, 1) for y in (-1, 1)])
    cloud = np.vstack([cloud, ground_corners])
    center = (cloud.min(axis=0) + cloud.max(axis=0)) / 2
    span = max(high - low)
    direction = np.array([1.7, -2.5, 1.3])
    direction /= np.linalg.norm(direction)
    right = np.cross([0, 0, 1], direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    projected = cloud @ np.array([right, up]).T
    projected_center = (projected.min(axis=0) + projected.max(axis=0)) / 2
    center += (projected_center - center @ np.array([right, up]).T) @ np.array([right, up])
    extent = np.ptp(projected, axis=0)
    parallel_scale = max(extent[1], extent[0] / (2400 / 1800)) * .56
    camera = [center + span * direction * 4, center, (0, 0, 1)]
    magnitudes = np.array([np.linalg.norm(f) for _, _, f in forces])
    max_force = float(magnitudes.max()) if len(magnitudes) else 0.0
    radius = candidate_mesh.length / 2
    rotations = Rotation.from_quat(poses[:, 3:])
    angles = (rotations[:-1].inv() * rotations[1:]).magnitude()
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
                                 + radius * angles)]
    ghost_indices = np.unique(np.r_[0, np.searchsorted(distance, np.linspace(0, distance[-1], ghost_count)),
                                    len(poses)-1])
    colors = dict(ground="#5a5046", stone="#808080", initial="#c5bf22", final="#24b33a",
                  ghost="#c0c0c0", force="#ed2525")
    opacities = dict(stone=1., initial=1., final=1., ghost=.30, force_stone=.50)
    settings = dict(camera_eye=camera[0].tolist(), camera_look_at=center.tolist(), camera_up=[0, 0, 1],
                    parallel_scale=parallel_scale, window_size=[2400, 1800], dpi=300,
                    ghost_sample_indices=ghost_indices.tolist(),
                    force_arrow_style={**FORCE_ARROW_STYLE,
                        "length_formula": "min_length + (max_length - min_length) * (force_N / max_force_N)**exponent"},
                    max_force_N=max_force, colors=colors, opacities=opacities,
                    mesh_cells={str(i): mesh.n_cells for i, mesh in meshes.items()},
                    mesh_simplification=False, smooth_shading=True, show_edges=False,
                    pdf_content="raster")
    (directory / "pose_optimization_rendering.json").write_text(json.dumps(settings, indent=2) + "\n")
    paths = []
    for mode in ("trajectory", "forces"):
        plot = pv.Plotter(off_screen=True, window_size=(2400, 1800), lighting="light kit")
        try:
            plot.set_background("white")
            configure_scene_passes(plot, translucent=True, occlusion_radius=.3, occlusion_bias=.015)
            ground = pv.Plane(center=(*ground_center, 0), direction=(0, 0, 1),
                              i_size=ground_size[0], j_size=ground_size[1],
                              i_resolution=1, j_resolution=1)
            plot.add_mesh(ground, color=colors["ground"], lighting=False)
            material = dict(smooth_shading=True, show_edges=False,
                            specular=0, ambient=.32, diffuse=.68)
            for object_id, mesh in displayed.items():
                if mode == "trajectory" and object_id == candidate_id:
                    continue
                plot.add_mesh(mesh, color=colors["stone"],
                              opacity=opacities["force_stone"] if mode == "forces" else opacities["stone"],
                              culling="back", **material)
            if mode == "trajectory":
                # All samples are native iterates, including their rotations.
                # Space ghost meshes by actual translation/rotation distance,
                # avoiding a dense opaque pile of nearly identical final poses.
                for index in ghost_indices:
                    pose = poses[index]
                    role = "initial" if index == 0 else "final" if index == len(poses)-1 else "ghost"
                    plot.add_mesh(transformed(candidate_mesh, pose), color=colors[role],
                                  opacity=opacities[role], culling="back", **material)
            else:
                for (object_id, point, force), magnitude in zip(forces, magnitudes):
                    if magnitude <= 1e-12:
                        continue
                    arrow = make_force_arrow(point, force, max_force)
                    plot.add_mesh(arrow, color=colors["force"], lighting=False)
                    plot.add_mesh(pv.Sphere(radius=.009, center=point), color="white", lighting=False)
            plot.camera_position = camera
            plot.enable_parallel_projection()
            plot.camera.parallel_scale = parallel_scale
            plot.camera.clipping_range = (.01, span * 20)
            path = directory / f"pose_optimization_{mode}.png"
            plot.screenshot(path)
            # Embed print resolution and provide a PDF at the same physical size.
            # The 3D surface rendering remains a raster in the PDF.
            from PIL import Image
            with Image.open(path) as pixels:
                pixels.save(path, dpi=(300, 300))
                pixels.convert("RGB").save(path.with_suffix(".pdf"), resolution=300)
            paths.append(path)
        finally:
            plot.close()
    return tuple(paths)
