"""Render seesaw pose-optimization trajectories for both equilibrium models.

A 0.25 m cube starts 0.3 m above the overhang of a plank balanced on a
narrow 0.15 m pillar (see tests/posegen_structure_toy.py). Structure
equilibrium moves it over the pillar; single-stone equilibrium leaves it on
the overhang. Both
traces are native accepted iterates, rendered like the paper's
pose-optimization figure with one shared camera. Each final pose is then
simulated to record whether the stack stands. A plot of the cube's x against
the outer (trust-region) iteration accompanies the renders.

    python tests/posegen_seesaw_figure.py --output .analysis/posegen-seesaw [--x0 0.475]
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from stacking_common.actions.core_solver import CorePlacementPoseSolver
from stacking_common.actions.types import InitializedPlacement

from posegen_paper_figures import force_incidences
from posegen_paper_rendering import render_pose_optimization
from posegen_paper_trace import trace_pose_optimization
from posegen_structure_toy import CUBE, MODES, PILLAR, PLANK, StructureToy, simulate, tilt_degrees


def box_mesh(size):
    """Display a toy box with flat faces.

    At sharpness 60 the DSF boxes are within millimeters of true boxes, but
    sampling their support map leaves long triangles across each flat face
    that shade as star patterns. Split vertices give each face its own normal.
    """
    import pyvista as pv

    sx, sy, sz = np.asarray(size) / 2
    return pv.Box((-sx, sx, -sy, sy, -sz, sz)).triangulate().compute_normals(
        split_vertices=True, feature_angle=30)


def outer_iterations(trace):
    """Return cumulative trust-region iterations and cube x for every sample.

    Samples are the accepted updates; rejected trials leave the pose unchanged.
    """
    offsets = np.r_[0, np.cumsum(trace.stage_iterations)]
    iterations = [0 if s.stage == 0 else offsets[s.stage - 1] + s.iteration for s in trace.samples]
    return np.array(iterations), np.array([s.pose[0] for s in trace.samples])


def plot_x_history(traces, pillar_width, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    color = {"structure": "#2a78d6", "single": "#eb6834"}
    label = {"structure": "Structure equilibrium", "single": "Single-stone equilibrium"}
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": ink2, "axes.labelcolor": ink,
                         "xtick.color": ink2, "ytick.color": ink2})
    fig, ax = plt.subplots(figsize=(3.4, 2.1), constrained_layout=True)
    ax.axhspan(-pillar_width / 2, pillar_width / 2, color="#e9e8e4", lw=0, zorder=0)
    ax.text(0.99, -pillar_width / 4, "over pillar", transform=ax.get_yaxis_transform(),
            ha="right", va="center", color=ink2, fontsize=7)
    for mode in ("structure", "single"):
        iterations, x = outer_iterations(traces[mode])
        total = sum(traces[mode].stage_iterations)
        ax.step(np.r_[iterations, total], np.r_[x, x[-1]], where="post",
                color=color[mode], lw=1.6, label=label[mode], zorder=3)
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Box position $x$ (m)")
    ax.set_xlim(0, max(sum(t.stage_iterations) for t in traces.values()))
    ax.grid(axis="y", color=grid, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="center right", frameon=False, fontsize=7)
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def stiff_collision_profile(profile):
    """Stiffen the homotopy's early penetration penalty, for this figure only.

    The excavator schedule's k_gap_c is 60 at the first stage (its ramp starts
    at zero), so large early steps cut into the 0.1 m plank or pass through it.
    A constant of 6000 keeps every traced iterate clear of the plank.
    """
    weights = dict(profile.schedule["objective_weights"])
    weights["k_gap_c"] = {**weights["k_gap_c"], "constant": 6000.0}
    return replace(profile, schedule={**profile.schedule, "objective_weights": weights})


def plank_overlap(toy, pose):
    """Depth (m) by which the axis-aligned cube box overlaps the plank."""
    half = np.asarray(toy.sizes[CUBE]) / 2
    plank_half = np.asarray(toy.sizes[PLANK]) / 2
    plank = toy.placed[PLANK][:3]
    low, high = pose[:3] - half, pose[:3] + half
    if high[0] < plank[0] - plank_half[0] or low[0] > plank[0] + plank_half[0]:
        return 0.0
    return max(0.0, min(high[2], plank[2] + plank_half[2]) - max(low[2], plank[2] - plank_half[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--x0", type=float, default=0.475,
                        help="cube x the single-stone settling solve starts from (m)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    # A narrow pillar leaves only a small stable band around the pivot.
    toy = StructureToy(sizes={PILLAR: (0.15, 0.4, 0.4), CUBE: (0.25, 0.25, 0.25)})
    profiles = {mode: stiff_collision_profile(toy.solvers[mode].profile) for mode in MODES}
    clearance_z = toy.placed[PLANK][2] + toy.sizes[PLANK][2] / 2 + toy.sizes[CUBE][2] / 2 + 0.30
    # The plank's DSF top is a shallow dome (0.5 mm lower at its ends), and
    # single-stone equilibrium leaves x free, so the gravity term slides the
    # cube toward the plank's end during the solve. Start both traces 0.3 m
    # above where that solve settles, so the single-stone cube only descends.
    probe = toy.initial_pose(args.x0)
    probe[2] = clearance_z
    start = CorePlacementPoseSolver(toy.inventory, profiles["single"]).solve(
        toy.inventory, toy.state, 2, InitializedPlacement(probe, 0.0)).pose
    start[2] = clearance_z
    initial = InitializedPlacement(start, 0.0)
    traces = {mode: trace_pose_optimization(toy.inventory, profiles[mode], toy.state, 2,
                                            initial, max_samples=400)
              for mode in MODES}
    meshes = {object_id: box_mesh(size) for object_id, size in toy.sizes.items()}
    placed = {object_id: toy.placed[object_id] for object_id in toy.placed}
    framing = [sample.pose for trace in traces.values() for sample in trace.samples]

    record = dict(x0=args.x0, start_pose=start.tolist(), modes={})
    for mode, trace in traces.items():
        forces = force_incidences(toy.inventory, trace.result.diagnostics)
        render_pose_optimization(args.output / mode, meshes, placed, CUBE, trace, forces,
                                 framing_poses=framing, ground_margin=0.2,
                                 ghost_count=18)
        after = simulate(toy.objects, {**toy.placed, CUBE: trace.result.pose})
        record["modes"][mode] = dict(
            final_pose=trace.result.pose.tolist(), c_feq=trace.result.c_feq,
            c_gap=trace.result.c_gap, stage_iterations=trace.stage_iterations,
            samples=[dict(stage=s.stage, iteration=s.iteration, pose=s.pose.tolist())
                     for s in trace.samples],
            simulated_plank_tilt_degrees=tilt_degrees(after[PLANK]),
            simulated_cube_drop_m=float(trace.result.pose[2] - after[CUBE][2]),
            max_plank_overlap_m=max(plank_overlap(toy, s.pose) for s in trace.samples),
        )
        print(f"{mode:9s} x {start[0]:+.3f} -> {trace.result.pose[0]:+.3f} m, "
              f"c_feq {trace.result.c_feq:.3f}, plank tilt after simulation "
              f"{tilt_degrees(after[PLANK]):.0f} deg, max plank overlap "
              f"{record['modes'][mode]['max_plank_overlap_m'] * 1000:.1f} mm", flush=True)
    (args.output / "seesaw.json").write_text(json.dumps(record, indent=2) + "\n")
    plot_x_history(traces, toy.sizes[PILLAR][0], args.output / "seesaw_x_history.png")
    print(args.output / "seesaw_x_history.png", flush=True)

    from PIL import Image
    panels = [Image.open(args.output / mode / "pose_optimization_trajectory.png").convert("RGB")
              for mode in ("structure", "single")]
    preview = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), "white")
    x = 0
    for panel in panels:
        preview.paste(panel, (x, 0))
        x += panel.width
    preview.save(args.output / "seesaw_trajectories.png")
    print(args.output / "seesaw_trajectories.png", flush=True)


if __name__ == "__main__":
    main()
