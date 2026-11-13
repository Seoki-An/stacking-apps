"""Replay optimized poses with settling-only damping and export matched videos.

Example (run from the repository root, with the CUDA native extension installed)::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/settling_damping_video.py \
        --run .stacks/RSS_visualization/tower-full-validation-20261113/tower_and_wall.stack \
        --step 73 --candidates 5 3 54 50 \
        --output .analysis/settling-damping-videos-20261113

Existing stones are fixed during this refinement phase, as in planning. This
does not run or certify structural stability. Recorded profiles are used except
for damping; structural/robustness rates are explicitly zero. MP4s play at half
speed, using actual poses without displacement exaggeration. Raw 100 Hz traces,
numeric summaries, and an HTML comparison page accompany the videos.
"""

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

RATES = ((0., 0.), (.25, .5), (.5, 1.), (1., 2.))
LABELS = ("Undamped", "Weak", "Selected default", "Strong")
COLORS = ("#777777", "#277da8", "#dc851b", "#a04ca3")


def pose_matrix(pose):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def capture(args):
    from stacking_common.artifacts import StackRunStore
    from stacking_common.domain import Action
    from stacking_excavator import ExcavatorApplication
    from posegen_paper_rendering import load_stone_meshes

    source = args.run.resolve()
    repository = args.repository or source.parent / "repository"
    checkpoint = StackRunStore.load_planning_checkpoint(source, step=args.step - 1)
    context = StackRunStore.load_run_context(source, checkpoint.context_sha256)
    app = ExcavatorApplication.from_repository(
        repository, mcts_config=repository / "apps/excavator/config/mcts_cuda.yml",
        recorded_target=context.profiles["inventory_profile"])
    inventory = app.load_inventory(checkpoint.state.object_ids,
        {i: checkpoint.pick_scene.pose(i) for i in checkpoint.state.object_ids})
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as db:
        record = json.loads(db.execute(
            "SELECT payload_json FROM records WHERE kind='mcts_step' AND record_id=?",
            (str(args.step),)).fetchone()[0])
    by_rank = {item["rank"]: item for item in record["candidates"]}
    candidates = [by_rank[rank] for rank in args.candidates]
    actions = [Action(object_index=c["object_index"], object_id=c["object_id"],
                      pose=c["solved_pose"], init_pose=c["init_pose"]) for c in candidates]
    placed = {checkpoint.state.object_ids[i]: checkpoint.state.object_poses[
        checkpoint.state.object_ids[i]] for i in checkpoint.state.placed_object_indices}
    model_dir = repository / context.profiles["inventory_profile"]["model_directory"]
    _, mesh_sources = load_stone_meshes(model_dir,
        list(dict.fromkeys([*placed, *(a.object_id for a in actions)])), "ply", context.asset_hashes)
    # Check the actual simulation geometry, too (display surfaces are not DSFs).
    for object_id in checkpoint.state.object_ids:
        path = model_dir / f"model_{object_id}.obj"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == context.asset_hashes[f"object/{object_id}"]

    profile = app.simulation_profile
    manifest = dict(source=str(source), context_sha256=context.sha256,
        checkpoint=checkpoint.step, step=args.step, rates=RATES, labels=LABELS,
        candidates=[dict(rank=c["rank"], stone=c["object_id"],
                         recorded_rejection=c["rejection_reason"]) for c in candidates],
        dt=profile.placement.settle_dt, steps=profile.placement.settle_steps,
        erp=profile.settling_physics.error_reduction_ratio,
        contact_model=profile.settling_physics.contact_model,
        structural_damping=dict(linear=0., angular=0.),
        existing_stones="fixed during settling", structural_validation="not performed by this script",
        starting_pose="recorded solved_pose (optimizer result)",
        display_meshes=mesh_sources, playback_speed=.5, displacement_exaggeration=1,
        summaries=[])
    traces = []
    for linear, angular in RATES:
        settling = replace(profile.settling_physics, damping=0.,
                           linear_damping=linear, angular_damping=angular)
        trial = replace(app, simulation_profile=replace(profile, damping=0.,
            linear_damping=0., angular_damping=0., settling_physics=settling))
        backend = trial.build_core_backend(inventory)
        try:
            backend.reset(checkpoint.state.object_ids)
            backend.config = replace(backend.config, settle_sample_count=backend.config.settle_steps)
            print(f"Simulating linear={linear}, angular={angular}", flush=True)
            outputs = backend._settling([checkpoint.state] * len(actions), actions)
            variant = []
            for candidate, action, out in zip(candidates, actions, outputs):
                entity = backend.scene_builder.entity_id(action.object_index)
                times = np.r_[0., np.asarray(out.sample_times)]
                poses = np.vstack([action.pose, np.asarray(out.sample_poses(entity))])
                keep = np.r_[True, np.diff(times) > 1e-8]
                times, poses = times[keep], poses[keep]
                assert np.isclose(times[-1], manifest["dt"] * manifest["steps"])
                variant.append(poses)
                angle = (Rotation.from_quat(poses[0, 3:]).inv()
                         * Rotation.from_quat(poses[-1, 3:])).magnitude()
                manifest["summaries"].append(dict(candidate=candidate["rank"],
                    linear=linear, angular=angular,
                    final_origin_displacement_m=float(np.linalg.norm(poses[-1, :3] - poses[0, :3])),
                    final_rotation_deg=float(np.degrees(angle)),
                    end_linear_speed_m_s=float(np.linalg.norm(out.snapshot.body_linear_velocity(entity))),
                    end_angular_speed_rad_s=float(np.linalg.norm(out.snapshot.body_angular_velocity(entity))),
                    unconverged_steps=sum(p.unconverged_steps for p in out.phases)))
            traces.append(variant)
        finally:
            backend.close()
    np.savez_compressed(args.output / "traces.npz", poses=np.asarray(traces), times=times,
                        placed_ids=list(placed), placed_poses=list(placed.values()))
    (args.output / "comparison.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest["summaries"][0]))
        writer.writeheader()
        writer.writerows(manifest["summaries"])


def render(directory):
    import imageio.v2 as imageio
    import pyvista as pv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = json.loads((directory / "comparison.json").read_text())
    data = np.load(directory / "traces.npz")
    times, traces = data["times"], data["poses"]
    meshes = {int(i): pv.read(s["path"]) for i, s in manifest["display_meshes"].items()}
    old_meshes = [meshes[int(i)].transform(pose_matrix(pose), inplace=False)
                  for i, pose in zip(data["placed_ids"], data["placed_poses"])]
    scene = pv.merge(old_meshes)
    duration, fps, playback = float(times[-1]), 20, .5
    frame_times = np.r_[np.zeros(fps), np.linspace(0., duration, round(duration / playback * fps) + 1),
                        np.full(fps, duration)]
    fig, axes = plt.subplots(len(manifest["candidates"]), 2, figsize=(10, 10), squeeze=False)
    html = ["<!doctype html><meta charset='utf-8'><title>Settling damping comparison</title>",
        "<style>body{font:18px sans-serif;max-width:1100px;margin:30px auto;background:white}"
        "video,img{width:100%}button{padding:10px;margin:5px}</style>",
        "<h1>Settling-only damping</h1><p>Step " + str(manifest["step"]) +
        "; same optimized pose in all four panels. Linear/angular rates are in s⁻¹. "
        "Gray stones are fixed during refinement. No motion exaggeration. "
        "Structural/robustness damping is zero; these clips do not certify stability.</p>",
        "<p>Playback: <button onclick=\"document.querySelectorAll('video').forEach(v=>v.playbackRate=.5)\">0.25× simulation</button>"
        "<button onclick=\"document.querySelectorAll('video').forEach(v=>v.playbackRate=1)\">0.5× simulation</button>"
        "<button onclick=\"document.querySelectorAll('video').forEach(v=>v.playbackRate=2)\">1× simulation</button></p>"]
    for ci, candidate in enumerate(manifest["candidates"]):
        rank, stone = candidate["rank"], candidate["stone"]
        poses = traces[:, ci]
        moving_mesh = meshes[stone]
        # One fixed camera frames all variants, including failed/falling paths.
        cloud = np.concatenate([moving_mesh.points @ Rotation.from_quat(p[3:]).as_matrix().T + p[:3]
                                for track in poses for p in track[::10]])
        low, high = cloud.min(axis=0), cloud.max(axis=0)
        center = (low + high) / 2
        direction = np.array([1.4, -2.4, 1.7]); direction /= np.linalg.norm(direction)
        right = np.cross([0., 0., 1.], direction); right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        projected = (cloud - center) @ np.array([right, up]).T
        scale = max(np.ptp(projected[:, 1]), np.ptp(projected[:, 0]) / (640 / 480), 1.3) * .72
        plot = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1280, 960), border=False)
        actors, labels, interpolators = [], [], []
        try:
            for vi, ((linear, angular), label) in enumerate(zip(RATES, LABELS)):
                plot.subplot(vi // 2, vi % 2)
                plot.set_background("white")
                material = dict(smooth_shading=True, show_edges=False, specular=0., ambient=.4)
                plot.add_mesh(scene, color="#a1a3a5", **material)
                bounds = scene.bounds
                plot.add_mesh(pv.Plane(center=((bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, 0.),
                    i_size=bounds[1]-bounds[0]+2., j_size=bounds[3]-bounds[2]+2.,
                    i_resolution=1, j_resolution=1), color="#786c5b")
                actor = plot.add_mesh(moving_mesh, color="#efab24", **material)
                actor.user_matrix = pose_matrix(poses[vi, 0])
                actors.append(actor)
                plot.add_text(f"{label}: linear {linear:g} / angular {angular:g} s^-1",
                              position=(16, 445), font_size=12, color="black")
                plot.add_text(f"Step {manifest['step']} | candidate #{rank} | stone {stone} | 0.5x speed",
                              position=(16, 421), font_size=10, color="#555555")
                text = plot.add_text(" ", position=(16, 12), font_size=10, color="black")
                text.GetTextProperty().SetBackgroundColor(1., 1., 1.)
                text.GetTextProperty().SetBackgroundOpacity(.9)
                labels.append(text)
                plot.camera_position = [center + direction * 30., center, [0, 0, 1]]
                plot.enable_parallel_projection(); plot.camera.parallel_scale = scale
                plot.camera.clipping_range = (.01, 100.)
                interpolators.append(Slerp(times, Rotation.from_quat(poses[vi, :, 3:])))
                displacement = np.linalg.norm(poses[vi, :, :3] - poses[vi, 0, :3], axis=1)
                angles = np.degrees((Rotation.from_quat(poses[vi, 0, 3:]).inv()
                                    * Rotation.from_quat(poses[vi, :, 3:])).magnitude())
                axes[ci, 0].plot(times, displacement * 1000., color=COLORS[vi], label=f"{linear:g} / {angular:g}")
                axes[ci, 1].plot(times, angles, color=COLORS[vi])
            plot.show(auto_close=False, interactive=False)
            name = f"candidate_{rank:02d}_stone_{stone}"
            print(f"Rendering {name}", flush=True)
            with imageio.get_writer(directory / f"{name}.mp4", fps=fps, codec="libx264",
                                    quality=8, macro_block_size=16, ffmpeg_params=["-movflags", "+faststart"]) as writer:
                for fi, t in enumerate(frame_times):
                    for vi, (actor, label, interp) in enumerate(zip(actors, labels, interpolators)):
                        pose = np.r_[[np.interp(t, times, poses[vi, :, k]) for k in range(3)], interp(t).as_quat()]
                        actor.user_matrix = pose_matrix(pose)
                        displacement = np.linalg.norm(pose[:3] - poses[vi, 0, :3]) * 1000.
                        angle = np.degrees((Rotation.from_quat(poses[vi, 0, 3:]).inv() * interp(t)).magnitude())
                        label.SetInput(f"t = {t:.2f} s    translation = {displacement:.1f} mm    rotation = {angle:.1f} deg")
                    plot.render()
                    pixels = plot.screenshot(return_img=True)
                    writer.append_data(pixels)
                    if fi in (0, len(frame_times) - 1):
                        imageio.imwrite(directory / f"{name}_{'initial' if fi == 0 else 'final'}.png", pixels)
                    if fi % 80 == 0:
                        print(f"  {fi}/{len(frame_times)} frames", flush=True)
            axes[ci, 0].set_ylabel(f"Candidate #{rank}\nTranslation (mm)")
            axes[ci, 1].set_ylabel("Rotation (deg)")
            for ax in axes[ci]:
                ax.set_xlabel("Simulation time (s)"); ax.grid(alpha=.2)
            html.append(f"<h2>Candidate #{rank} / stone {stone}</h2><video controls preload='metadata' "
                        f"poster='{name}_initial.png' src='{name}.mp4'></video>")
        finally:
            plot.close()
    axes[0, 0].legend(title="Linear / angular (s⁻¹)", fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "motion_comparison.png", dpi=160)
    fig.savefig(directory / "motion_comparison.pdf")
    plt.close(fig)
    html.append("<h2>Motion relative to the optimized pose</h2><img src='motion_comparison.png'>")
    (directory / "index.html").write_text("\n".join(html), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--candidates", type=int, nargs="+", default=[5, 3, 54, 50])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--trace-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.render_only:
        capture(args)
    if not args.trace_only:
        render(args.output)


if __name__ == "__main__":
    main()
