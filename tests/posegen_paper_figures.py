"""Export paired paper figures from one recorded excavator placement.

Run from the repository root (requires stacking-core, PyVista and Matplotlib):

    python tests/posegen_paper_figures.py \
        --run .stacks/excavator_plan21.stack --step 20 \
        --output .analysis/posegen-paper

The selected candidate is the default; --candidate selects another candidate ID.
Use --stone-mesh ply to render reconstructed stone meshes instead of DSF surfaces.
This is a CPU replay of the recorded posegen profile, not simulation or motion
planning, and need not reproduce a saved CUDA solve's endpoint. Output includes
two matched-camera PNGs and a JSON record of native samples and forces in N.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages/stacking-common/src"))

from stacking_common.actions.core_solver import CorePosegenProfile
from stacking_common.actions.types import InitializedPlacement
from stacking_common.artifacts import StackRunStore
from stacking_common.assets import ObjectAssetLoader
from stacking_common.domain import DirectionalTaper, State, TargetGeometry
from stacking_common.execution.values import plain_json

from posegen_paper_trace import trace_pose_optimization
from posegen_paper_rendering import load_stone_meshes, render_pose_optimization


def load_recorded_placement(run, step_number, candidate_id=None):
    artifact = StackRunStore.load_mcts(run)
    step = next(s for s in artifact.steps if s.step == step_number)
    candidate_id = candidate_id or step.selected_candidate_id
    candidate = next(c for c in step.candidates if c.candidate_id == candidate_id)
    checkpoint = StackRunStore.load_planning_checkpoint(run, step=step_number)
    context = StackRunStore.load_run_context(run, checkpoint.context_sha256)
    profile = CorePosegenProfile(**context.profiles["posegen_profile"])
    stored = context.profiles["inventory_profile"]
    target = dict(stored["target"])
    target["taper"] = DirectionalTaper(**target["taper"])
    ids = checkpoint.state.object_ids
    if ids[candidate.object_index] != candidate.object_id:
        raise ValueError("recorded candidate and checkpoint inventory orders differ")
    initial_poses = {i: np.array([0., 0., 0., 0., 0., 0., 1.]) for i in ids}
    placed = {o.object_id: o.pose for o in step.scene.objects}
    initial_poses.update(placed)
    initial_poses[candidate.object_id] = candidate.init_pose
    model_directory = ROOT / stored["model_directory"]
    inventory = ObjectAssetLoader(model_directory, stored["density"], TargetGeometry(**target),
                                  dict(stored["density_overrides"])).load(ids, initial_poses)
    participating = [*placed, candidate.object_id]
    for object_id in participating:
        path = model_directory / f"model_{object_id}.obj"
        expected = context.asset_hashes[f"object/{object_id}"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"recorded stone model changed: {path}")
    state = State(ids, [ids.index(i) for i in placed], initial_poses, [], [])
    return inventory, profile, state, candidate, model_directory, context.sha256


def force_incidences(inventory, diagnostics):
    """Map native entity IDs (inventory index + 1) to actual object IDs."""
    points = diagnostics["posegen_contact_points"]
    forces = diagnostics["posegen_contact_forces"]
    if set(points) != set(forces):
        raise ValueError("force and point entity IDs differ")
    rows = []
    for entity_id, contact_forces in forces.items():
        index = int(entity_id) - 1
        if not 0 <= index < len(inventory):
            raise ValueError(f"unknown force entity: {entity_id}")
        if len(points[entity_id]) != len(contact_forces):
            raise ValueError("force and point counts differ")
        for point, force in zip(points[entity_id], contact_forces):
            point, force = np.asarray(point), np.asarray(force)
            if point.shape != (3,) or force.shape != (3,) or not np.isfinite([point, force]).all():
                raise ValueError("contact incidence must contain finite world-frame three-vectors")
            rows.append((inventory[index].object_id, point, force))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--stone-mesh", choices=("dsf", "ply"), default="dsf",
                        help="stone display geometry: DSF solver surfaces (default) or reconstructed PLY meshes")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    print("Loading recorded scene and profiles", flush=True)
    inventory, profile, state, candidate, models, context_hash = load_recorded_placement(
        args.run, args.step, args.candidate)
    context = StackRunStore.load_run_context(args.run, context_hash)
    placed = {state.object_ids[i]: state.object_poses[state.object_ids[i]]
              for i in state.placed_object_indices}
    meshes, mesh_sources = load_stone_meshes(models, [*placed, candidate.object_id],
                                            args.stone_mesh, context.asset_hashes)
    print(f"Tracing stone {candidate.object_id} with {len(state.placed_object_indices)} placed stones", flush=True)
    trace = trace_pose_optimization(inventory, profile, state, candidate.object_index,
                                    InitializedPlacement(candidate.init_pose, 0.))
    forces = force_incidences(inventory, trace.result.diagnostics)
    record = dict(
        source_run=str(args.run.resolve()), step=args.step, candidate=candidate.candidate_id,
        context_sha256=context_hash, backend="cpu", frame="stack", force_units="N",
        trace_method="accepted native trust-region prefixes with preceding schedule replay",
        stage_iterations=trace.stage_iterations, stage_accepted_iterations=trace.stage_accepted_iterations,
        candidate_object_id=candidate.object_id, recorded_solved_pose=candidate.solved_pose.tolist(),
        final_pose=trace.result.pose.tolist(), c_gap=trace.result.c_gap, c_feq=trace.result.c_feq,
        pose_converged=trace.result.diagnostics["posegen_pose_converged"],
        force_converged=trace.result.diagnostics["posegen_force_converged"],
        placed_poses={str(i): p.tolist() for i, p in placed.items()},
        samples=[dict(stage=s.stage, iteration=s.iteration, pose=s.pose.tolist()) for s in trace.samples],
        force_incidences=[dict(object_id=i, point=p.tolist(), force=f.tolist()) for i, p, f in forces],
        stone_mesh_mode=args.stone_mesh, rendered_meshes=mesh_sources,
    )
    import stacking_core._native as native
    record["native_binary_sha256"] = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    record["source_run_sha256"] = hashlib.sha256(args.run.read_bytes()).hexdigest()
    record["posegen_profile"] = plain_json(context.profiles["posegen_profile"])
    record["inventory_profile"] = plain_json(context.profiles["inventory_profile"])
    record["recorded_model_sha256"] = {str(i): context.asset_hashes[f"object/{i}"]
                                         for i in [*placed, candidate.object_id]}
    (args.output / "pose_optimization.json").write_text(json.dumps(record, indent=2) + "\n")
    paths = render_pose_optimization(args.output, meshes, placed, candidate.object_id, trace, forces)
    print(f"Stages: {trace.stage_iterations}; accepted: {trace.stage_accepted_iterations}", flush=True)
    for path in paths:
        print(path, flush=True)


if __name__ == "__main__":
    main()
