"""Toy posegen ablation: whole-structure versus single-stone equilibrium.

A plank rests centered on a narrow pillar, and a cube is placed on the plank:

            [cube]
    ======================   plank (dynamic, already placed)
             |    |          pillar (dynamic, already placed)
    ----------------------   ground

The excavator posegen profile solves the cube twice. "structure" is the
current posegen: placed bodies are force-graph nodes with their own wrench
balance. "single" sets placed_as_boundaries, so placed bodies become fixed
supports and only the candidate's equilibrium is checked. Every solved pose is
then simulated to see whether the stack stands.

Single-stone equilibrium accepts the cube anywhere on the plank, including
the overhang that tips it. Structure equilibrium moves the cube over the
pillar. Posegen models each body pair as one contact point, so it treats the
plank as pivoting on a point and centers the cube, while simulation keeps the
stack up anywhere within roughly |x| <= 0.25 m.

    python tests/posegen_structure_toy.py [--plot toy.png]
"""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in ("packages/stacking-common/src", "apps/excavator/src"):
    sys.path.insert(0, str(ROOT / directory))

import stacking_core as core

from stacking_common.actions.core_solver import CorePlacementPoseSolver
from stacking_common.actions.filtering import PosegenFeasibilityFilter
from stacking_common.actions.types import InitializedPlacement
from stacking_common.domain import (
    ConvexGeometry, DirectionalTaper, ObjectInventory, StackObject, State, TargetGeometry,
)
from stacking_common.profiles.action import load_action_generation_profile
from stacking_common.profiles.posegen import load_posegen_profile

CONFIG = ROOT / "apps/excavator/config"
PILLAR, PLANK, CUBE = 1, 2, 3
SIZES = {PILLAR: (0.3, 0.4, 0.4), PLANK: (1.2, 0.4, 0.1), CUBE: (0.2, 0.2, 0.2)}
MASSES = {PILLAR: 2.0, PLANK: 1.0, CUBE: 1.0}
# Sharper than the default DSF so box faces stay flat enough for the plank to balance.
SHARPNESS = 60
FRICTION = 0.8
PARKED = np.array([2.5, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])  # Cube pose while unplaced.
FACES = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
MODES = ("single", "structure")


def pose(x, z):
    return np.array([x, 0.0, z, 0.0, 0.0, 0.0, 1.0])


def box(object_id, mass_factor=1.0):
    sx, sy, sz = SIZES[object_id]
    mass = MASSES[object_id] * mass_factor
    points = 0.5 * np.array([[-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
                             [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz]])
    inertia = mass / 12 * np.diag([sy**2 + sz**2, sx**2 + sz**2, sx**2 + sy**2])
    model = core.BodyModel.dsf(object_id, points, mass=mass, inertia=inertia,
                               friction=FRICTION, sharpness=SHARPNESS)
    return StackObject(object_id, model, [ConvexGeometry(points, FACES, sharpness=SHARPNESS)])


def tilt_degrees(pose_value):
    """Angle between the body z-axis and world z."""
    x, y = pose_value[3:5]
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0))))


def simulate(objects, poses, seconds=3.0, dt=0.01):
    plane = core.BodyModel.plane(100, friction=FRICTION)
    bodies = [core.BodyInstance(100, plane, pose(0.0, 0.0), mobility=core.Mobility.STATIC)]
    bodies += [core.BodyInstance(o.object_id, o.body_model, poses[o.object_id]) for o in objects]
    scene = core.SceneSnapshot(1, bodies)
    simulator = core.Simulator()
    for _ in range(int(round(seconds / dt))):
        scene = simulator.step(scene, dt).snapshot
    return {o.object_id: scene.body_pose(o.object_id) for o in objects}


class StructureToy:
    def __init__(self, mass_factor=1.0):
        self.objects = [box(object_id, mass_factor) for object_id in (PILLAR, PLANK, CUBE)]
        nominal = {PILLAR: pose(0.0, 0.2), PLANK: pose(0.0, 0.45), CUBE: PARKED}
        # DSF boxes are slightly rounded; place the base where physics settles it.
        settled = simulate(self.objects, nominal)
        self.placed = {PILLAR: settled[PILLAR], PLANK: settled[PLANK]}
        target = TargetGeometry(width=3.0, length=3.0, height=2.0, origin=np.zeros(3),
                                taper=DirectionalTaper(), support_overlap=0.0)
        self.inventory = ObjectInventory(self.objects, {**self.placed, CUBE: PARKED}, target)
        self.state = State(object_ids=[PILLAR, PLANK, CUBE], placed_object_indices=[0, 1],
                           object_poses={**self.placed, CUBE: PARKED},
                           trajectories=[], action_history=[])
        profile = load_posegen_profile(CONFIG / "solvers/posegen.yml")
        self.solvers = {
            mode: CorePlacementPoseSolver(self.inventory, replace(
                profile, placed_as_boundaries=mode == "single", boundary_radius=None))
            for mode in MODES
        }
        generator = load_action_generation_profile(
            CONFIG / "action_generation.yml", max_poses_per_object=1, banned_object_ids=(),
        ).generator
        self.filter = PosegenFeasibilityFilter(
            generator.posegen_gap_threshold,
            generator.posegen_equilibrium_threshold,
            require_solver_convergence=generator.require_solver_convergence,
            reject_equilibrium_at_threshold=generator.reject_equilibrium_at_threshold,
        )

    def initial_pose(self, x0):
        return pose(x0, self.placed[PLANK][2] + 0.17)

    def place(self, mode, x0):
        initial = InitializedPlacement(self.initial_pose(x0), 0.0)
        solved = self.solvers[mode].solve(self.inventory, self.state, 2, initial)
        decision = self.filter.evaluate(self.inventory, self.state, 2, initial, solved)
        after = simulate(self.objects, {**self.placed, CUBE: solved.pose})
        plank_tilt = tilt_degrees(after[PLANK])
        cube_drop = float(solved.pose[2] - after[CUBE][2])
        return dict(mode=mode, x0=float(x0), solved=solved, accepted=decision.accepted,
                    after=after, plank_tilt=plank_tilt, cube_drop=cube_drop,
                    stands=plank_tilt < 15.0 and cube_drop < 0.2)


def plot(toy, rows, path, showcase_x0=0.35):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import Affine2D

    surface, ink, ink2, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
    color = {"structure": "#2a78d6", "single": "#eb6834"}
    label = {"structure": "Structure equilibrium (current posegen)",
             "single": "Single-stone equilibrium (placed = fixed)"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink2, "text.color": ink})

    def draw(ax, object_id, pose_value, **style):
        sx, _, sz = SIZES[object_id]
        x, y, z, w = pose_value[3:]
        pitch = np.arctan2(2 * (x * z - w * y), 1 - 2 * (y * y + z * z))
        rect = Rectangle((-sx / 2, -sz / 2), sx, sz, **style)
        rect.set_transform(Affine2D().rotate(pitch).translate(pose_value[0], pose_value[2])
                           + ax.transData)
        ax.add_patch(rect)

    fig = plt.figure(figsize=(12, 8.6), facecolor=surface)
    layout = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.42, wspace=0.16,
                              left=0.06, right=0.98, top=0.84, bottom=0.07)
    fig.suptitle("Toy: cube placed on a plank balanced on a pillar", x=0.06, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(0.06, 0.905, "Same scene, excavator posegen profile and schedule; only "
             "placed_as_boundaries differs. Each solved pose is then simulated for 3 s.",
             color=ink2)

    for column, mode in enumerate(MODES):
        ax = fig.add_subplot(layout[0, column], facecolor=surface)
        row = next(r for r in rows if r["mode"] == mode and np.isclose(r["x0"], showcase_x0))
        solved = row["solved"]
        start = toy.initial_pose(showcase_x0)
        ax.axhline(0, color=ink2, lw=1.2)
        for object_id in (PILLAR, PLANK):
            draw(ax, object_id, toy.placed[object_id], fc="none", ec=muted, lw=1, ls=(0, (3, 2)))
        draw(ax, CUBE, start, fc="none", ec=muted, lw=1, ls=(0, (1, 2)))
        draw(ax, CUBE, solved.pose, fc="none", ec=color[mode], lw=1.6, ls=(0, (3, 2)))
        draw(ax, PILLAR, row["after"][PILLAR], fc="#d9d8d3", ec=ink2, lw=1)
        draw(ax, PLANK, row["after"][PLANK], fc="#bfbdb6", ec=ink2, lw=1)
        draw(ax, CUBE, row["after"][CUBE], fc=color[mode], ec=surface, lw=2)
        if abs(solved.pose[0] - showcase_x0) > 0.02:
            ax.annotate("", xy=(solved.pose[0], start[2] + 0.13),
                        xytext=(showcase_x0, start[2] + 0.13),
                        arrowprops=dict(arrowstyle="->", color=color[mode], lw=1.6))
        ax.text(showcase_x0, start[2] + 0.15, "start", ha="center", va="bottom",
                color=muted, fontsize=9)
        ax.set_xlim(-0.75, 0.95)
        ax.set_ylim(-0.03, 0.95)
        ax.set_aspect("equal")
        ax.axis("off")
        verdict = "filter " + ("accepts" if row["accepted"] else "rejects") + " it, and the stack " \
            + ("stands" if row["stands"] else "topples")
        ax.set_title(f"{'ab'[column]})  {label[mode]}\n"
                     f"cube x {showcase_x0:+.2f} -> {solved.pose[0]:+.2f} m,  "
                     f"E {solved.c_feq:.2f},  plank tilt after sim {row['plank_tilt']:.0f}°\n"
                     f"{verdict}", loc="left", fontsize=10)
    fig.text(0.06, 0.52, "dashed = posegen output (scene before physics)     "
             "solid = after 3 s simulation     dotted = initial guess", color=ink2, fontsize=9)

    panels = (("c)  Where posegen puts the cube   (●  stack stands   ✕  stack topples)",
               "posegen output x (m)", lambda r: r["solved"].pose[0]),
              ("d)  What the feasibility filter sees",
               "E  (normalized force + torque residual)", lambda r: r["solved"].c_feq))
    for column, (title, ylabel, value) in enumerate(panels):
        ax = fig.add_subplot(layout[1, column], facecolor=surface)
        for mode in MODES:
            mode_rows = [r for r in rows if r["mode"] == mode]
            x0 = np.array([r["x0"] for r in mode_rows])
            y = np.array([value(r) for r in mode_rows])
            stands = np.array([r["stands"] for r in mode_rows])
            ax.plot(x0, y, color=color[mode], lw=2, zorder=2, label=label[mode])
            ax.scatter(x0[stands], y[stands], s=46, color=color[mode],
                       edgecolor=surface, lw=2, zorder=3)
            ax.scatter(x0[~stands], y[~stands], s=70, marker="X", color=color[mode],
                       edgecolor=surface, lw=1.5, zorder=3)
        if column == 1:
            threshold = toy.filter.equilibrium_threshold
            ax.axhline(threshold, color=ink2, lw=1, ls=(0, (4, 3)))
            ax.text(0.56, threshold + 0.03, "excavator filter threshold", ha="right",
                    color=ink2, fontsize=8.5)
        else:
            ax.legend(loc="lower right", frameon=False, fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("initial guess x (m)")
        ax.set_ylabel(ylabel)
        ax.grid(color=grid, lw=0.8)
        ax.set_axisbelow(True)
    fig.savefig(path, dpi=130, facecolor=surface)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", type=Path, help="write the sweep figure to this PNG")
    args = parser.parse_args()

    toy = StructureToy()
    rows = []
    print(" x0     mode       x_out   c_feq  accepted  plank_tilt  stands")
    for x0 in np.round(np.arange(-0.55, 0.5501, 0.05), 3):
        for mode in MODES:
            row = toy.place(mode, x0)
            rows.append(row)
            print(f"{x0:+.2f}  {mode:9s}  {row['solved'].pose[0]:+.3f}  "
                  f"{row['solved'].c_feq:6.3f}  {str(row['accepted']):8s}  "
                  f"{row['plank_tilt']:9.1f}°  {row['stands']}")
    for mode in MODES:
        mode_rows = [r for r in rows if r["mode"] == mode]
        print(f"{mode:9s}: accepted and stands {sum(r['accepted'] and r['stands'] for r in mode_rows)}, "
              f"accepted but topples {sum(r['accepted'] and not r['stands'] for r in mode_rows)}, "
              f"rejected {sum(not r['accepted'] for r in mode_rows)} / {len(mode_rows)}")
    if args.plot:
        plot(toy, rows, args.plot)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
