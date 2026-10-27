"""Export scene height, total placement score, and four component paper maps.

    MPLCONFIGDIR=/tmp/stacking-paper-mpl python tests/height_score_paper_figures.py \
        --run .stacks/excavator_plan21.stack --step 20 \
        --output .analysis/height-score-paper

Scores use the recorded candidate's initial orientation and pre-search scene,
with the existing diagnostic scorer's normal floor gates. Height normalization
is over this one orientation, not the multi-orientation search population.
Each component panel shows its signed weighted contribution (higher is better).
Unscored or excluded regions inside the target are neutral gray; outside is
black. No score extrapolation is rendered. Masked comparisons are retained.
Raw terms, weighted terms, masks, and grid coordinates are retained in NPZ.
The translucent candidate overlay is the original PLY mesh's bottom-height
map in its initial pose; it has no convex-component outlines. Requires PyVista.
Separate candidate_bottom_height figures crop this surface to the stone and
show elevation above its lowest sampled point, with candidate-relative XY.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from posegen_paper_figures import load_recorded_placement
from stacking_common.artifacts import StackRunStore
from stacking_common.domain.geometry import transform_points
from stacking_common.execution.values import plain_json
from stacking_common.visualization.dsf import object_surface_path
from stacking_common.visualization.heightmap import _rasterize_triangles
from stacking_common.visualization.sampling import initial_pose_score_view


PANELS = (
    ("scene_height", "Scene height", "Height (m)"),
    ("height", "Height score", "Weighted contribution"),
    ("support_likelihood", "Support score", "Weighted contribution"),
    ("score", "Total placement score", "Score"),
    ("fill_area", "Filling score", "Weighted contribution"),
    ("target_boundary", "Target boundary score", "Weighted contribution"),
)

MAP_STYLE = dict(background="white", masked="black", target="#f4d35e", target_interior="#808080",
                 initial_xy="magenta", text="#252525", colormap="viridis",
                 bottom_colormap="Greys", bottom_opacity=.5,
                 individual_font_scaling="match overview font size relative to plot dimensions",
                 overview_size_inches=[3.5, 2.3], overview_axis_label_size=4.5,
                 overview_tick_label_size=3.5, overview_dpi=600,
                 overview_padding_inches=.003, overview_spacing=.005,
                 bottom_overlay_rendering="alpha composited into vector cell colors",
                 pdf_content="vector heatmap cells, overlays, axes, and text")


def rasterize_bottom_height(points, faces, pose, x, y):
    """Minimum triangle-surface world Z at each XY sample; outside is NaN.

    Negating Z lets the existing maximum-height rasterizer compute a lower
    envelope. Use the actual mesh triangles, preserving its concavities and
    silhouette rather than a convex hull or component outlines.
    """
    inverted = transform_points(points, pose).copy()
    inverted[:, 2] *= -1
    output = np.full(len(y) * len(x), -np.inf)
    _rasterize_triangles(output, x, y, inverted[np.asarray(faces, dtype=int)])
    bottom = -output.reshape(len(y), len(x))
    bottom[~np.isfinite(bottom)] = np.nan
    return bottom


def load_candidate_bottom_height(path, pose, view, expected_hash=None):
    """Rasterize the original reconstructed PLY mesh at scene-map resolution."""
    import pyvista as pv

    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"recorded candidate surface changed: {path}")
    mesh = pv.read(path).triangulate()
    x = np.linspace(*view.scene_xlim, view.scene_height.shape[1])
    y = np.linspace(*view.scene_ylim, view.scene_height.shape[0])
    bottom = rasterize_bottom_height(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:], pose, x, y)
    if not np.isfinite(bottom).any():
        raise ValueError("candidate bottom surface does not overlap the scene-map samples")
    return bottom, dict(path=str(path.resolve()), sha256=digest,
        definition="minimum original PLY triangle-surface world Z at each XY sample",
        units="m", colormap=MAP_STYLE["bottom_colormap"], opacity=MAP_STYLE["bottom_opacity"],
        color_limits_m=[float(np.nanmin(bottom)), float(np.nanmax(bottom))],
        color_direction="lower = white, higher = black", resolution=list(bottom.shape))


def weighted_fields(view):
    """Retain signed production weights and verify the total before plotting."""
    weighted = {key: view.fields[key] * weight for key, weight in view.weights.items()}
    if view.valid.any():
        total = sum(weighted.values())
        if not np.allclose(total[view.valid], view.fields["score"][view.valid], rtol=1e-12, atol=1e-12):
            raise ValueError("weighted score components do not reproduce the total score")
    return weighted


def candidate_bottom_patch(view, bottom_height):
    """Crop the saved lower surface, retaining XY handedness and metric spacing."""
    rows, columns = np.nonzero(np.isfinite(bottom_height))
    if not len(rows):
        raise ValueError("candidate bottom surface has no finite samples")
    row_slice = slice(rows.min(), rows.max()+1)
    column_slice = slice(columns.min(), columns.max()+1)
    x = np.linspace(*view.scene_xlim, bottom_height.shape[1])[column_slice] - view.initial_xy[0]
    y = np.linspace(*view.scene_ylim, bottom_height.shape[0])[row_slice] - view.initial_xy[1]
    minimum_z = float(np.nanmin(bottom_height))
    return x, y, bottom_height[row_slice, column_slice] - minimum_z, minimum_z


def match_overview_fonts(fig, ax, reference_plot_inches, *, colorbar=None):
    """Match overview text/plot proportions after accounting for label margins."""
    style = MAP_STYLE
    for _ in range(8):
        fig.canvas.draw()
        bounds = ax.get_window_extent()
        plot_inches = min(bounds.width, bounds.height) / fig.dpi
        scale = plot_inches / reference_plot_inches
        label_size = style["overview_axis_label_size"] * scale
        tick_size = style["overview_tick_label_size"] * scale
        if abs(ax.xaxis.label.get_fontsize() - label_size) < .01:
            break
        ax.xaxis.label.set_fontsize(label_size)
        ax.yaxis.label.set_fontsize(label_size)
        ax.tick_params(labelsize=tick_size)
        if colorbar is not None:
            ax.set_title(ax.get_title(loc="left"), loc="left", fontsize=5.5 * scale)
            colorbar.ax.xaxis.label.set_fontsize(label_size)
            colorbar.ax.tick_params(labelsize=tick_size)


def render_candidate_bottom_map(directory, view, bottom_height, reference_plot_inches):
    """Export clean and annotated contact-side maps from the original mesh."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y, heights, _ = candidate_bottom_patch(view, bottom_height)
    style = MAP_STYLE
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    cmap = plt.get_cmap(style["colormap"]).copy()
    cmap.set_bad((0, 0, 0, 0))
    dx = (view.scene_xlim[1]-view.scene_xlim[0]) / (bottom_height.shape[1]-1)
    dy = (view.scene_ylim[1]-view.scene_ylim[0]) / (bottom_height.shape[0]-1)
    span = max(x[-1]-x[0]+dx, y[-1]-y[0]+dy)
    center = np.array([(x[0]+x[-1])/2, (y[0]+y[-1])/2])
    half_width = .55 * span
    paths = []
    for annotated in (False, True):
        fig, ax = plt.subplots(figsize=(6, 6), layout="constrained", facecolor=style["background"])
        try:
            ax.set_facecolor(style["background"])
            pixels = ax.pcolormesh(x, y, np.ma.masked_invalid(heights), shading="nearest",
                cmap=cmap, vmin=0., vmax=max(float(np.nanmax(heights)), 1e-12),
                edgecolors="face", linewidth=.1, rasterized=False)
            ax.set(xlim=(center[0]-half_width, center[0]+half_width),
                   ylim=(center[1]-half_width, center[1]+half_width), aspect="equal",
                   xlabel="X relative to candidate (m)", ylabel="Y relative to candidate (m)")
            ax.tick_params(colors=style["text"], labelsize=style["overview_tick_label_size"])
            for label in (ax.xaxis.label, ax.yaxis.label):
                label.set_color(style["text"])
                label.set_fontsize(style["overview_axis_label_size"])
            for spine in ax.spines.values():
                spine.set_color(style["text"])
            bar = None
            if annotated:
                ax.set_title("Candidate bottom height", loc="left", fontsize=5.5, color=style["text"])
                bar = fig.colorbar(pixels, ax=ax, orientation="horizontal", fraction=.055,
                                   pad=.12, shrink=.85, aspect=26, format="%.3g")
                bar.set_label("Elevation above lowest sampled underside point (m)", fontsize=4.5)
                bar.ax.tick_params(labelsize=style["overview_tick_label_size"])
                if bar.solids is not None:
                    bar.solids.set_rasterized(False)
            match_overview_fonts(fig, ax, reference_plot_inches, colorbar=bar)
            name = "candidate_bottom_height" + ("_annotated" if annotated else "")
            for suffix in ("png", "pdf"):
                path = Path(directory) / f"{name}.{suffix}"
                fig.savefig(path, dpi=300, facecolor=style["background"], bbox_inches="tight", pad_inches=.15)
                paths.append(path)
        finally:
            plt.close(fig)
    return paths


def render_maps(directory, view, bottom_height):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize, to_rgb
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FormatStrFormatter

    directory = Path(directory)
    weighted = weighted_fields(view)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    style = MAP_STYLE
    cmap = plt.get_cmap(style["colormap"]).copy()
    cmap.set_bad((0, 0, 0, 0))
    bottom_cmap = plt.get_cmap(style["bottom_colormap"]).copy()
    bottom_cmap.set_bad((0, 0, 0, 0))
    bottom_x = np.linspace(*view.scene_xlim, view.scene_height.shape[1])
    bottom_y = np.linspace(*view.scene_ylim, view.scene_height.shape[0])
    bottom_values = np.ma.masked_invalid(bottom_height)
    lower, upper = view.target_bounds

    def cell_edges(coordinates):
        midpoints = (coordinates[:-1]+coordinates[1:])/2
        return np.r_[2*coordinates[0]-midpoints[0], midpoints, 2*coordinates[-1]-midpoints[-1]]

    def bottom_overlay(ax, x, y, values, limits, score_panel, neutral_target):
        """Composite alpha into vector cell colors, avoiding PDF stitching seams.

        Split at both raster grids and the target rectangle so every polygon
        retains the original constant field value and bottom-height sample.
        No interpolation, bitmap, or convex geometry is introduced.
        """
        field_x, field_y = cell_edges(x), cell_edges(y)
        bottom_edges_x, bottom_edges_y = cell_edges(bottom_x), cell_edges(bottom_y)
        edges_x = np.unique(np.r_[field_x, bottom_edges_x, lower[0], upper[0]])
        edges_y = np.unique(np.r_[field_y, bottom_edges_y, lower[1], upper[1]])
        xx, yy = np.meshgrid((edges_x[:-1]+edges_x[1:])/2, (edges_y[:-1]+edges_y[1:])/2)
        row = np.searchsorted(bottom_edges_y, yy, side="right")-1
        column = np.searchsorted(bottom_edges_x, xx, side="right")-1
        inside_bottom = (row >= 0) & (row < len(bottom_y)) & (column >= 0) & (column < len(bottom_x))
        bottom = bottom_height[np.clip(row, 0, len(bottom_y)-1), np.clip(column, 0, len(bottom_x)-1)]
        visible = inside_bottom & np.isfinite(bottom)
        in_target = (xx >= lower[0]) & (xx <= upper[0]) & (yy >= lower[1]) & (yy <= upper[1])
        background = np.empty((*xx.shape, 3))
        background[:] = to_rgb(style["masked"] if score_panel else style["background"])
        if neutral_target:
            background[in_target] = to_rgb(style["target_interior"])
        row = np.searchsorted(field_y, yy, side="right")-1
        column = np.searchsorted(field_x, xx, side="right")-1
        inside_field = (row >= 0) & (row < len(y)) & (column >= 0) & (column < len(x))
        field = np.ma.filled(values, np.nan)[np.clip(row, 0, len(y)-1), np.clip(column, 0, len(x)-1)]
        known = inside_field & np.isfinite(field)
        if score_panel:
            known &= in_target
        background[known] = cmap(Normalize(*limits)(field[known]))[:, :3]
        gray = bottom_cmap(Normalize(float(bottom_values.min()), float(bottom_values.max()))(bottom))[:, :, :3]
        blended = style["bottom_opacity"]*gray + (1-style["bottom_opacity"])*background
        row, column = np.nonzero(visible)
        left, right = edges_x[column], edges_x[column+1]
        bottom_edge, top = edges_y[row], edges_y[row+1]
        polygons = np.stack((np.column_stack((left, bottom_edge)), np.column_stack((right, bottom_edge)),
                             np.column_stack((right, top)), np.column_stack((left, top))), axis=1)
        ax.add_collection(PolyCollection(polygons, facecolors=blended[visible], edgecolors="face",
                                         linewidth=.1, antialiaseds=False, rasterized=False))

    def panel(ax, name, *, overview=False):
        score_panel = name != "scene_height"
        neutral_target = score_panel and not name.endswith("_masked")
        field_name = name.removesuffix("_masked")
        if name == "scene_height":
            x = np.linspace(*view.scene_xlim, view.scene_height.shape[1])
            y = np.linspace(*view.scene_ylim, view.scene_height.shape[0])
            values = view.scene_height
        else:
            x, y = view.x, view.y
            field = view.fields["score"] if field_name == "score" else weighted.get(
                field_name, np.full(view.valid.shape, np.nan))
            mask = ~np.isfinite(field) | ~view.valid
            values = np.ma.array(field, mask=mask)
        finite = np.asarray(values)[~np.ma.getmaskarray(values) & np.isfinite(values)]
        limits = (float(finite.min()), float(finite.max())) if len(finite) else (0., 1.)
        if limits[0] == limits[1]:
            limits = (limits[0]-.5, limits[1]+.5)
        limits = tuple(0. if value == 0. else value for value in limits)
        ax.set_facecolor(style["masked"] if score_panel else style["background"])
        if neutral_target:
            ax.add_patch(Rectangle(lower, *(upper-lower), facecolor=style["target_interior"],
                                   edgecolor="none", zorder=0))
        # Matching hairline strokes close PDF viewer seams between vector cells.
        pixels = ax.pcolormesh(x, y, values, shading="nearest", cmap=cmap,
                              vmin=limits[0], vmax=limits[1], edgecolors="face",
                              linewidth=.1, rasterized=False)
        if score_panel:
            pixels.set_clip_path(Rectangle(lower, *(upper-lower), transform=ax.transData))
        bottom_overlay(ax, x, y, values, limits, score_panel, neutral_target)
        ax.add_patch(Rectangle((lower[0], lower[1]), upper[0]-lower[0], upper[1]-lower[1],
                               fill=False, edgecolor=style["target"], linewidth=.65 if overview else 1.1))
        ax.plot(view.initial_xy[0], view.initial_xy[1], "o", markersize=2.5 if overview else 5,
                color=style["initial_xy"], markeredgewidth=0)
        ax.set(xlim=view.scene_xlim, ylim=view.scene_ylim, aspect="equal")
        if overview:
            ax.set_xticks(np.linspace(*view.scene_xlim, 3))
            ax.set_yticks(np.linspace(*view.scene_ylim, 3))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.tick_params(colors=style["text"], labelsize=style["overview_tick_label_size"],
                           width=.5, length=1, pad=.5)
            ax.label_outer()
        else:
            ax.set_xlabel("X (m)", fontsize=style["overview_axis_label_size"], color=style["text"])
            ax.set_ylabel("Y (m)", fontsize=style["overview_axis_label_size"], color=style["text"])
            ax.tick_params(colors=style["text"], labelsize=style["overview_tick_label_size"])
        for spine in ax.spines.values():
            spine.set_color(style["text"])
            spine.set_linewidth(.5 if overview else .8)

    overview, axes = plt.subplots(2, 3, figsize=style["overview_size_inches"], layout="compressed",
                                  sharex=True, sharey=True, facecolor=style["background"])
    overview.get_layout_engine().set(w_pad=style["overview_padding_inches"],
        h_pad=style["overview_padding_inches"],
        wspace=style["overview_spacing"], hspace=style["overview_spacing"])
    overview.supxlabel("X (m)", fontsize=style["overview_axis_label_size"], color=style["text"])
    overview.supylabel("Y (m)", fontsize=style["overview_axis_label_size"], color=style["text"])
    paths = []
    try:
        for ax, (name, title, units) in zip(axes.flat, PANELS):
            panel(ax, name, overview=True)
        for suffix in ("png", "pdf"):
            path = directory / f"height_score_overview.{suffix}"
            overview.savefig(path, dpi=style["overview_dpi"], facecolor=style["background"])
            paths.append(path)
        overview.canvas.draw()
        bounds = axes.flat[0].get_window_extent()
        reference_plot_inches = min(bounds.width, bounds.height) / overview.dpi
    finally:
        plt.close(overview)
    individual_panels = PANELS + tuple((name+"_masked", title, units)
                                       for name, title, units in PANELS if name != "scene_height")
    for name, title, units in individual_panels:
        fig, ax = plt.subplots(figsize=(6, 6), layout="constrained", facecolor=style["background"])
        try:
            panel(ax, name)
            match_overview_fonts(fig, ax, reference_plot_inches)
            for suffix in ("png", "pdf"):
                path = directory / f"{name}.{suffix}"
                fig.savefig(path, dpi=300, facecolor=style["background"], bbox_inches="tight", pad_inches=.15)
                paths.append(path)
        finally:
            plt.close(fig)
    paths.extend(render_candidate_bottom_map(directory, view, bottom_height, reference_plot_inches))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--candidate", help="recorded candidate ID; defaults to the selected candidate")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    print("Loading recorded scene and scoring settings", flush=True)
    inventory, _, state, candidate, models, context_hash = load_recorded_placement(
        args.run, args.step, args.candidate)
    context = StackRunStore.load_run_context(args.run, context_hash)
    profiles = context.profiles
    construction = profiles["action_generation_profile"].get("construction", {})
    if construction.get("mode", "volume_fill") != "volume_fill":
        raise ValueError("This run used a removed construction mode; its score maps are unavailable")
    environment = profiles["environment_profile"]
    settings = profiles.get("sampling_settings", {})
    view = initial_pose_score_view(inventory, state, candidate.object_id, candidate.init_pose,
        planar_config=profiles["action_generation_profile"]["planar"], settings=settings,
        floor_settings=profiles.get("floor_settings"),
        height_map_resolution=environment["observation"]["height_map_resolution"],
        height_map_margin=environment["height_map_margin"])
    weighted = weighted_fields(view)
    bottom_path = object_surface_path(models / f"model_{candidate.object_id}.obj", "ply")
    bottom, bottom_metadata = load_candidate_bottom_height(bottom_path, candidate.init_pose, view,
        context.asset_hashes.get(f"object_surface/{candidate.object_id}"))
    args.output.mkdir(parents=True, exist_ok=False)
    arrays = dict(scene_height=view.scene_height, scene_xlim=view.scene_xlim, scene_ylim=view.scene_ylim,
                  x=view.x, y=view.y, valid=view.valid, initial_xy=view.initial_xy,
                  target_bounds=np.array(view.target_bounds), initial_pose=candidate.init_pose,
                  candidate_bottom_height=bottom)
    arrays.update({f"raw_{key}": value for key, value in view.fields.items()})
    arrays.update({f"weighted_{key}": value for key, value in weighted.items()})
    bottom_x, bottom_y, bottom_relative, bottom_minimum = candidate_bottom_patch(view, bottom)
    arrays.update(candidate_bottom_x=bottom_x, candidate_bottom_y=bottom_y,
                  candidate_bottom_relative_height=bottom_relative,
                  candidate_bottom_minimum_z=bottom_minimum)
    np.savez_compressed(args.output / "height_score_maps.npz", **arrays)
    participating = [state.object_ids[i] for i in state.placed_object_indices] + [candidate.object_id]
    metadata = dict(source_run=str(args.run.resolve()), source_run_sha256=hashlib.sha256(args.run.read_bytes()).hexdigest(),
        context_sha256=context_hash, step=args.step, candidate=candidate.candidate_id,
        candidate_object_id=candidate.object_id, initial_pose=candidate.init_pose.tolist(),
        placed_object_ids=[state.object_ids[i] for i in state.placed_object_indices],
        placed_poses={str(state.object_ids[i]): state.object_poses[state.object_ids[i]].tolist()
                      for i in state.placed_object_indices},
        frame="stack", array_order="row Y, column X", display_axes="horizontal X, vertical Y",
        figure_style=MAP_STYLE,
        overview=dict(panel_order=[[panel[0] for panel in PANELS[:3]], [panel[0] for panel in PANELS[3:]]],
            titles=False, colorbars=False, shared_axes=True,
            size_inches=MAP_STYLE["overview_size_inches"]),
        candidate_bottom_map=bottom_metadata,
        candidate_bottom_figure=dict(files="candidate_bottom_height[_annotated].png/pdf",
            definition="original mesh lower surface minus its lowest finite scene-grid sample",
            z_reference_m=bottom_minimum, xy_origin_m=view.initial_xy.tolist(),
            axes="same X/Y directions as scene maps; candidate-relative, no mirroring",
            colormap=MAP_STYLE["colormap"], opacity=1., background=MAP_STYLE["background"],
            reference="https://vislab.isr.tecnico.ulisboa.pt/wp-content/uploads/paper_pdfs/2021_amenezes_from_ocks.pdf"),
        score_display=dict(extension="none; actual scores under the original eligibility mask",
            target_interior=MAP_STYLE["target_interior"], outside_target=MAP_STYLE["masked"],
            domain="clipped to physical target base", planner_data="unchanged; raw and weighted arrays retained",
            masked_figures="*_masked.png/pdf"),
        score_weights=view.weights, component_panels="signed weighted contributions; higher is better",
        valid_cells=int(view.valid.sum()), masked_cells=int(view.valid.size-view.valid.sum()),
        reconstruction="fixed initial orientation, pre-search scene, normal floor gates; no retry relaxation",
        height_normalization="over this orientation's eligible cells, not the multi-orientation search population",
        inventory_profile=plain_json(profiles["inventory_profile"]),
        sampling_settings=plain_json(settings), floor_settings=plain_json(profiles.get("floor_settings")),
        planar_config=plain_json(profiles["action_generation_profile"]["planar"]),
        height_map_resolution=list(environment["observation"]["height_map_resolution"]),
        height_map_margin=environment["height_map_margin"],
        model_sha256={str(i): context.asset_hashes[f"object/{i}"] for i in participating},
        overlays=dict(yellow="target base", translucent_grayscale="original candidate mesh bottom height",
                      magenta="initial XY", gray_cells="unscored or excluded cells inside target",
                      black_cells="outside target; masked cells in comparison figures"))
    (args.output / "height_score_maps.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Step {args.step}, stone {candidate.object_id}, {view.valid.sum()} admissible cells", flush=True)
    for path in render_maps(args.output, view, bottom):
        print(path, flush=True)


if __name__ == "__main__":
    main()
