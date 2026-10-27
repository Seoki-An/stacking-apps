"""Physical-scale inventory sheets with principal-axis-aligned stone meshes."""

from pathlib import Path

import numpy as np


def principal_axis_transform(points):
    """Rigidly align largest/second/smallest vertex variance to X/Y/Z.

    Fix eigenvector signs deterministically, keep a proper rotation, center the
    aligned XY bounds, and put the bottom at Z=0. Never rescale the object.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("stone vertices must be finite Nx3 points")
    centroid = points.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov((points-centroid).T))
    axes = axes[:, ::-1]
    for column in range(3):
        if axes[np.argmax(np.abs(axes[:, column])), column] < 0:
            axes[:, column] *= -1
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    aligned = (points-centroid) @ axes
    lower, upper = aligned.min(axis=0), aligned.max(axis=0)
    center = np.r_[(lower[:2]+upper[:2])/2, lower[2]]
    transform = np.eye(4)
    transform[:3, :3] = axes.T
    transform[:3, 3] = -axes.T @ centroid - center
    return transform


def inventory_grid(meshes):
    """Return rigid placements on 4x4 pages, sorted by original AABB volume.

    Every page uses the same cell spacing and camera scale, including the last
    partial page. Meshes and original coordinates remain untouched.
    """
    if not meshes:
        raise ValueError("inventory cannot be empty")
    records = []
    for object_id, mesh in meshes.items():
        points = np.asarray(mesh.points)
        transform = principal_axis_transform(points)
        aligned = points @ transform[:3, :3].T + transform[:3, 3]
        extents = np.ptp(aligned, axis=0)
        records.append(dict(object_id=int(object_id), transform=transform,
                            original_bbox_volume_m3=float(np.prod(np.ptp(points, axis=0))),
                            aligned_extents_m=extents.tolist()))
    records.sort(key=lambda entry: (entry["original_bbox_volume_m3"], entry["object_id"]))
    cell = 1.5 * max(max(entry["aligned_extents_m"]) for entry in records)
    if cell <= 0:
        raise ValueError("inventory meshes must have nonzero extent")
    for index, entry in enumerate(records):
        page, slot = divmod(index, 16)
        row, column = divmod(slot, 4)
        entry.update(page=page+1, row=row+1, column=column+1)
        entry["transform"][:2, 3] += [(column-1.5)*cell, (1.5-row)*cell]
    return records, cell


def render_inventory_pages(directory, meshes, records, cell, *, show_labels=False):
    """Render opaque original meshes from +Z with an orthographic camera."""
    import pyvista as pv
    from PIL import Image

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    max_height = max(entry["aligned_extents_m"][2] for entry in records)
    for page in range(1, records[-1]["page"]+1):
        plot = pv.Plotter(off_screen=True, window_size=(2400, 2400), lighting="light kit")
        try:
            plot.set_background("white")
            for entry in records:
                if entry["page"] != page:
                    continue
                mesh = meshes[entry["object_id"]].transform(entry["transform"], inplace=False)
                plot.add_mesh(mesh, color=(.58, .58, .58), opacity=1., show_edges=False,
                              smooth_shading=True, ambient=.32, diffuse=.68, specular=0.)
                if show_labels:
                    position = [(entry["column"]-2.5)*cell, (2.5-entry["row"])*cell-.38*cell,
                                max_height+.01*cell]
                    plot.add_point_labels([position], [str(entry["object_id"])],
                        show_points=False, always_visible=True, shape=None, font_size=24,
                        text_color="#252525")
            plot.camera_position = [(0., 0., max_height+10*cell), (0., 0., 0.), (0., 1., 0.)]
            plot.enable_parallel_projection()
            plot.camera.parallel_scale = 2.05*cell
            plot.reset_camera_clipping_range()
            pixels = plot.screenshot(return_img=True)
            figure = Image.fromarray(pixels).convert("RGB")
            stem = f"inventory_top_{page:02d}"
            png, pdf = directory / f"{stem}.png", directory / f"{stem}.pdf"
            figure.save(png, dpi=(300, 300))
            figure.save(pdf, resolution=300.)
            paths.extend((png, pdf))
        finally:
            plot.close()
    return paths
