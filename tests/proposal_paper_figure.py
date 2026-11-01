"""Export a paper illustration of height map -> proposal score -> initial pose.

    MPLCONFIGDIR=/tmp/stacking-paper-mpl python tests/proposal_paper_figure.py \
        --run .stacks/excavator_plan_for_visual2.stack --step 23 \
        --output .analysis/proposal-paper

Uses the selected candidate's recorded initial pose and pre-placement scene.
Scores are reconstructed for that orientation with the recorded scoring profile;
they are not the multi-orientation search population or a new sampled placement.
The PDF retains vector maps and annotations; its original-mesh 3D view is raster.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from height_score_paper_figures import load_candidate_bottom_height
from posegen_paper_figures import load_recorded_placement
from posegen_paper_rendering import load_stone_meshes
from stacking_common.artifacts import StackRunStore
from stacking_common.execution.values import plain_json
from stacking_common.visualization.dsf import object_surface_path
from stacking_common.visualization.sampling import initial_pose_score_view


def render_initial_pose(directory, meshes, placed, candidate, view, bottom,
                        view_direction=(-1.7, 2.5, -1.5)):
    """Use original surfaces at recorded poses, with a fitted orthographic camera."""
    import pyvista as pv
    from scipy.spatial.transform import Rotation
    from stacking_common.visualization.rendering import configure_scene_passes

    displayed = {}
    for object_id, pose in {**placed, candidate.object_id: candidate.init_pose}.items():
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
        transform[:3, 3] = pose[:3]
        displayed[object_id] = meshes[object_id].transform(transform, inplace=False)
    cloud = np.vstack([mesh.points for mesh in displayed.values()])
    low, high = cloud.min(axis=0), cloud.max(axis=0)
    ground_center = (low[:2] + high[:2]) / 2
    ground_size = high[:2] - low[:2] + .65
    corners = np.array([[*(ground_center + np.array([x, y])*ground_size/2), 0.]
                        for x in (-1, 1) for y in (-1, 1)])
    cloud = np.vstack([cloud, corners])
    direction = -np.asarray(view_direction, dtype=float)
    if (direction.shape != (3,) or not np.isfinite(direction).all()
            or np.linalg.norm(direction[:2]) <= 1e-12):
        raise ValueError("view direction must be finite and not parallel to the Z up axis")
    direction /= np.linalg.norm(direction)
    right = np.cross([0, 0, 1], direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    basis = np.array([right, up])
    projected = cloud @ basis.T
    center = (cloud.min(axis=0) + cloud.max(axis=0)) / 2
    projected_center = (projected.min(axis=0) + projected.max(axis=0)) / 2
    center += (projected_center - center @ basis.T) @ basis
    window = (2400, 2100)
    extent = np.ptp(projected, axis=0)
    parallel_scale = max(extent[1], extent[0] / (window[0]/window[1])) * .55
    camera = [center + direction * max(high-low) * 5, center, (0, 0, 1)]

    plot = pv.Plotter(off_screen=True, window_size=window, lighting="light kit")
    try:
        plot.set_background("white")
        configure_scene_passes(plot, translucent=False, occlusion_radius=.3, occlusion_bias=.015)
        plot.add_mesh(pv.Plane(center=(*ground_center, 0.), direction=(0, 0, 1),
            i_size=ground_size[0], j_size=ground_size[1], i_resolution=1, j_resolution=1),
            color="#5a5046", lighting=False)
        for object_id, mesh in displayed.items():
            plot.add_mesh(mesh, color="#e000dc" if object_id == candidate.object_id else "#808080",
                smooth_shading=True, show_edges=False, specular=0., ambient=.32, diffuse=.68)

        # Actual vertical clearances at three underside samples; the pose is not lifted.
        rows, columns = np.nonzero(np.isfinite(bottom))
        bx = np.linspace(*view.scene_xlim, bottom.shape[1])
        by = np.linspace(*view.scene_ylim, bottom.shape[0])
        underside = np.c_[bx[columns], by[rows], bottom[rows, columns]]
        projected_bottom = underside @ basis.T
        front = (underside @ direction) > np.median(underside @ direction)
        candidates = np.flatnonzero(front)
        horizontal = projected_bottom[candidates, 0]
        selected = [candidates[np.argmin(abs(horizontal-value))]
                    for value in np.quantile(horizontal, [.15, .5, .85])]
        for index in np.unique(selected):
            point = underside[index]
            floor = float(view.scene_height[rows[index], columns[index]])
            for z in np.arange(floor, point[2]-.01, .10):
                end = min(z+.055, point[2])
                plot.add_mesh(pv.Line((*point[:2], z), (*point[:2], end)),
                    color="white", line_width=2, lighting=False)
        plot.camera_position = camera
        plot.enable_parallel_projection()
        plot.camera.parallel_scale = parallel_scale
        plot.camera.clipping_range = (.01, max(high-low)*20)
        path = Path(directory) / "sampled_initial_pose.png"
        pixels = plot.screenshot(path)
    finally:
        plot.close()

    candidate_points = displayed[candidate.object_id].points
    positions = (candidate_points - center) @ basis.T
    # A visible surface point near the candidate's upper-right silhouette.
    anchor_index = np.argmax(positions[:, 0]*.7 + positions[:, 1]
                             + (candidate_points @ direction)*.1)
    anchor = positions[anchor_index] / (2*parallel_scale*np.array([window[0]/window[1], 1])) + .5
    return pixels, anchor, dict(camera_eye=camera[0].tolist(), camera_look_at=center.tolist(),
        camera_up=[0, 0, 1], parallel_scale=float(parallel_scale), window_size=list(window),
        ground_color="#5a5046", stone_color="#808080", candidate_color="#e000dc",
        smooth_shading=True, show_edges=False, mesh_simplification=False,
        pose_display="recorded initial pose, without translation or rotation adjustments")


def render_figure(directory, view, bottom, scene_pixels, anchor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "font.size": 9, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.16, 2.5), facecolor="white")
    map_width = .255
    map_height = map_width * 7.16 / 2.5
    axes = [fig.add_axes((left, .225, map_width, map_height)) for left in (.04, .36)]
    # Caption centers are equally spaced at .1675, .4875, and .8075.
    scene_ax = fig.add_axes((.6525, .16, .31, .81))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(cmap(0.))
    lower, upper = view.target_bounds
    bx = np.linspace(*view.scene_xlim, view.scene_height.shape[1])
    by = np.linspace(*view.scene_ylim, view.scene_height.shape[0])
    score = np.ma.array(view.fields["score"], mask=~view.valid | ~np.isfinite(view.fields["score"]))
    if not score.count():
        raise ValueError("recorded orientation has no eligible score cells")
    maps = [(bx, by, view.scene_height), (view.x, view.y, score)]
    color_limits = []
    try:
        for index, (ax, (x, y, values)) in enumerate(zip(axes, maps)):
            limits = (float(values.min()), float(values.max()))
            color_limits.append(limits)
            ax.set_facecolor(cmap(0.))
            pixels = ax.pcolormesh(x, y, values, shading="nearest", cmap=cmap,
                vmin=limits[0], vmax=limits[1], edgecolors="face", linewidth=.1, rasterized=False)
            if index == 1:
                ax.contour(bx, by, np.isfinite(bottom).astype(float), levels=[.5],
                           colors=["#f000e8"], linewidths=.85)
                ax.plot(*view.initial_xy, "o", color="#f000e8", markersize=5,
                        markeredgecolor="white", markeredgewidth=.8, zorder=5)
                ax.annotate(r"Sampled $(x,y)$", xy=view.initial_xy, xycoords="data",
                    xytext=(.075, .92), textcoords="axes fraction", color="white", fontsize=8.5,
                    arrowprops=dict(arrowstyle="-", color="white", linewidth=.8,
                                    shrinkA=2, shrinkB=4), zorder=6)
            ax.add_patch(Rectangle(lower, *(upper-lower), fill=False,
                                   edgecolor="#f4e439", linewidth=.9))
            ax.set(xlim=view.scene_xlim, ylim=view.scene_ylim, aspect="equal", xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_linewidth(.45)
                spine.set_edgecolor("#777777")
            origin = (-.055, -.035)
            for endpoint in ((.035, -.035), (-.055, .10)):
                ax.annotate("", xy=endpoint, xytext=origin, xycoords="axes fraction",
                    annotation_clip=False, arrowprops=dict(arrowstyle="-|>", color="#252525",
                                                         linewidth=.7, mutation_scale=6))
            ax.text(.055, -.035, r"$x$", transform=ax.transAxes, va="center", fontsize=8)
            ax.text(-.055, .13, r"$y$", transform=ax.transAxes, ha="center", fontsize=8)
            position = ax.get_position()
            bar_ax = fig.add_axes((position.x0+.035, .143, position.width-.07, .028))
            bar = fig.colorbar(pixels, cax=bar_ax, orientation="horizontal")
            bar.set_ticks([])
            bar.outline.set_linewidth(.5)
            bar.solids.set_rasterized(False)

        scene_ax.imshow(scene_pixels, extent=(0, 1, 0, 1),
                        aspect=scene_pixels.shape[0]/scene_pixels.shape[1])
        scene_ax.set_axis_off()
        scene_ax.plot(*anchor, "o", color="black", markeredgecolor="white", markeredgewidth=.6,
                      markersize=3.5)
        label_position = np.minimum(anchor+np.array([.10, .05]), [.92, .95])
        scene_ax.annotate(r"$\hat{q}$", xy=anchor, xytext=label_position,
                          fontsize=12, ha="left", va="center",
                          arrowprops=dict(arrowstyle="-", linewidth=.85, color="black", shrinkB=3))
        for start, end in ((.305, .345), (.627, .67)):
            fig.add_artist(FancyArrowPatch((start, .59), (end, .59), transform=fig.transFigure,
                arrowstyle="->", mutation_scale=12, linewidth=1.2, color="#252525"))
        for ax, caption in zip([*axes, scene_ax],
                ("(a) Scene height", "(b) Proposal score", "(c) Sampled initial pose")):
            bounds = ax.get_position()
            fig.text(bounds.x0+bounds.width/2, .045, caption, ha="center", va="center", fontsize=10.5)
        paths = []
        for suffix in ("png", "pdf"):
            path = Path(directory) / f"proposal_pipeline.{suffix}"
            fig.savefig(path, dpi=600, facecolor="white")
            paths.append(path)
    finally:
        plt.close(fig)
    return paths, color_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, default=23)
    parser.add_argument("--candidate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view-direction", nargs=3, type=float, default=(-1.7, 2.5, -1.5),
                        metavar=("X", "Y", "Z"),
                        help="Camera looking direction with +Z up; 0 1 0 looks along +Y with +X right")
    args = parser.parse_args()
    print(f"Loading recorded placement at step {args.step}", flush=True)
    inventory, _, state, candidate, models, context_hash = load_recorded_placement(
        args.run, args.step, args.candidate)
    context = StackRunStore.load_run_context(args.run, context_hash)
    profiles = context.profiles
    environment = profiles["environment_profile"]
    print("Reconstructing the recorded orientation's proposal score map", flush=True)
    view = initial_pose_score_view(inventory, state, candidate.object_id, candidate.init_pose,
        planar_config=profiles["action_generation_profile"]["planar"],
        settings=profiles["sampling_settings"], floor_settings=profiles.get("floor_settings"),
        height_map_resolution=environment["observation"]["height_map_resolution"],
        height_map_margin=environment["height_map_margin"])
    placed = {state.object_ids[i]: state.object_poses[state.object_ids[i]]
              for i in state.placed_object_indices}
    meshes, mesh_sources = load_stone_meshes(models, [*placed, candidate.object_id],
                                           "ply", context.asset_hashes)
    bottom, bottom_metadata = load_candidate_bottom_height(
        object_surface_path(models / f"model_{candidate.object_id}.obj", "ply"),
        candidate.init_pose, view, context.asset_hashes.get(f"object_surface/{candidate.object_id}"))
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(placed)} placed stones and candidate {candidate.object_id}", flush=True)
    scene_pixels, anchor, camera = render_initial_pose(
        args.output, meshes, placed, candidate, view, bottom, args.view_direction)
    paths, limits = render_figure(args.output, view, bottom, scene_pixels, anchor)
    np.savez_compressed(args.output / "proposal_pipeline.npz", scene_height=view.scene_height,
        scene_xlim=view.scene_xlim, scene_ylim=view.scene_ylim, x=view.x, y=view.y,
        score=view.fields["score"], valid=view.valid, initial_pose=candidate.init_pose,
        candidate_bottom_height=bottom, target_bounds=np.array(view.target_bounds),
        **{f"raw_{key}": value for key, value in view.fields.items() if key != "score"})
    metadata = dict(source_run=str(args.run.resolve()),
        source_run_sha256=hashlib.sha256(args.run.read_bytes()).hexdigest(),
        step=args.step, candidate=candidate.candidate_id, candidate_object_id=candidate.object_id,
        context_sha256=context_hash, initial_pose=candidate.init_pose.tolist(),
        placed_poses={str(i): pose.tolist() for i, pose in placed.items()},
        meshes=mesh_sources, bottom_height=bottom_metadata, rendering=camera,
        score_reconstruction="fixed recorded initial orientation, pre-placement scene, normal floor gates",
        sampling_settings=plain_json(profiles["sampling_settings"]), score_weights=view.weights,
        planar_config=plain_json(profiles["action_generation_profile"]["planar"]),
        color_limits=dict(scene_height_m=limits[0], proposal_score=limits[1]),
        missing_score_display="viridis minimum purple; masked cells are not extrapolated scores",
        footprint="original PLY mesh bottom-map silhouette, no convex decomposition",
        sampled_marker="recorded initial XY", orientation_label="recorded initial pose",
        layout=dict(size_inches=[7.16, 2.5], dpi=600, background="white",
                    pdf_content="vector heatmaps and annotations, raster 3D mesh rendering"))
    (args.output / "proposal_pipeline.json").write_text(json.dumps(metadata, indent=2)+"\n")
    for path in paths:
        print(path, flush=True)


if __name__ == "__main__":
    main()
