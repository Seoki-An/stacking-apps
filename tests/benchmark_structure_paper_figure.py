"""Compare final full-inventory MCTS/greedy structures as vector PDF/SVG.

    MPLCONFIGDIR=/tmp/stacking-paper-mpl python tests/benchmark_structure_paper_figure.py \
        --benchmark .stacks/benchmarks/one-sided-wall-height23-20261101 \
        --seed 6 --output .analysis/seed6-wall-comparison

Only reads committed final checkpoints and existing benchmark summaries.
No planning, simulation, pose adjustment, or mesh simplification is performed.
"""

import argparse
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages/stacking-common/src"))

from stacking_common.artifacts import StackRunStore
from stacking_common.artifacts.object_assets import materialize_object_assets
from stacking_common.execution.values import plain_json
from stacking_common.visualization.vector_mesh import orthographic_basis, vector_mesh_faces

COVERAGE_DEFINITION = ("union occupancy of commissioned convex-component OBJ meshes on the recorded "
                       "0.05 m target voxel grid / analytic target volume, including support overlap")


def write_numerical_results(directory, records):
    """Verify the recorded filling and export volumes, metrics, and definitions."""
    from stacking_common.assets import ObjectAssetLoader
    from stacking_common.domain import DirectionalTaper, TargetGeometry
    from stacking_common.environment.overlap import VoxelOverlapEvaluator

    if any(record["target"] != records[0]["target"] for record in records):
        raise ValueError("numerical comparison target profiles differ")
    rows = []
    saved_models = TemporaryDirectory(prefix='stack-figure-models-')
    for record in records:
        source = Path(record["source_run"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != record["source_run_sha256"]:
            raise ValueError(f"benchmark source changed: {source}")
        context = StackRunStore.load_run_context(source, record["context_sha256"])
        profile = context.profiles["inventory_profile"]
        target_data = dict(profile["target"])
        target_data["taper"] = DirectionalTaper(**target_data["taper"])
        target = TargetGeometry(**target_data)
        paths = materialize_object_assets(source, context, ROOT, saved_models.name)
        models = next(iter(paths.values())).parent
        loader = ObjectAssetLoader(models, profile["density"], target, dict(profile["density_overrides"]))
        placements = []
        for object_id, pose in record["poses"].items():
            object_id = int(object_id)
            path = models / f"model_{object_id}.obj"
            if hashlib.sha256(path.read_bytes()).hexdigest() != context.asset_hashes[f"object/{object_id}"]:
                raise ValueError(f"recorded overlap geometry changed: {path}")
            placements.append((loader.load_object(object_id), np.asarray(pose)))
        overlap = VoxelOverlapEvaluator(target, .05)
        filling = overlap.scene_coverage(placements)
        metrics = record["metrics"]
        if not np.isclose(filling, metrics["target_coverage"], rtol=0., atol=1e-12):
            raise ValueError(f"recomputed filling differs from benchmark: {source}")
        rows.append(dict(method=record["method"], seed=record["seed"],
            available_stones=len(record["inventory_ids"]), placed_stones=record["placed"],
            target_filling_percent=100*filling, target_unfilled_percent=100*(1-filling),
            filled_target_volume_m3=filling*target.volume,
            unfilled_target_volume_m3=(1-filling)*target.volume,
            benchmark_target_volume_m3=target.volume,
            nominal_above_ground_target_volume_m3=target.width*target.length*target.height,
            max_sampled_surface_height_m=metrics["max_height"],
            mean_surface_height_m=metrics["mean_height"], height_variance_m2=metrics["height_variance"],
            total_search_time_s=record["search_seconds"], total_search_time_min=record["search_seconds"]/60,
            discounted_return=metrics["discounted_return"], outcome=record["outcome"]))
    methods = {row["method"]: row for row in rows}
    mcts, greedy = methods["mcts"], methods["greedy"]
    difference = dict(
        placed_stones=mcts["placed_stones"]-greedy["placed_stones"],
        target_filling_percentage_points=mcts["target_filling_percent"]-greedy["target_filling_percent"],
        filled_target_volume_m3=mcts["filled_target_volume_m3"]-greedy["filled_target_volume_m3"],
        relative_filling_increase_percent=100*(mcts["target_filling_percent"]/greedy["target_filling_percent"]-1)
            if greedy["target_filling_percent"] else None,
        total_search_time_s=mcts["total_search_time_s"]-greedy["total_search_time_s"])
    definitions = dict(filling=COVERAGE_DEFINITION, voxel_size_m=.05,
        support_overlap=(f"benchmark target extends {target.support_overlap:g} * height = "
            f"{target.support_overlap*target.height:.6f} m below ground; denominator {target.volume:.6f} m3"),
        nominal_target=(f"{target.width:g} x {target.length:g} x {target.height:g} m above ground; "
            f"volume {target.width*target.length*target.height:.6f} m3"),
        unfilled="entire target minus occupied union, including unbuilt space and gaps; not enclosed-cavity porosity",
        surface_height="recorded top-envelope statistics on a 32x32 target-footprint grid; empty cells have zero height",
        search_time="recorded sum of search wall times; excludes physical execution",
        relative_filling_increase="(MCTS filling / greedy filling - 1) * 100",
        validation="filling recomputed from recorded final poses and hash-verified commissioned OBJ geometry")
    directory = Path(directory)
    json_path = directory / "numerical_results.json"
    json_path.write_text(json.dumps(dict(results=rows, mcts_minus_greedy=difference,
        definitions=definitions, sources=[dict(method=r["method"], source_run=r["source_run"],
            source_run_sha256=r["source_run_sha256"], context_sha256=r["context_sha256"])
            for r in records]), indent=2)+"\n")
    csv_path = directory / "numerical_results.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"Seed {rows[0]['seed']}: full {rows[0]['available_stones']}-stone inventory, final structures",
             "", f"{'Metric':<36} {'MCTS':>15} {'Greedy':>15}", "-"*68]
    for label, key, precision in (
            ("Stones placed", "placed_stones", 0),
            ("Target filling (%)", "target_filling_percent", 4),
            ("Target unfilled (%)", "target_unfilled_percent", 4),
            ("Filled target volume (m3)", "filled_target_volume_m3", 6),
            ("Unfilled target volume (m3)", "unfilled_target_volume_m3", 6),
            ("Benchmark target volume (m3)", "benchmark_target_volume_m3", 6),
            ("Max sampled surface height (m)", "max_sampled_surface_height_m", 4),
            ("Mean surface height (m)", "mean_surface_height_m", 4),
            ("Height variance (m2)", "height_variance_m2", 4),
            ("Total search time (min)", "total_search_time_min", 3),
            ("Discounted return", "discounted_return", 6)):
        lines.append(f"{label:<36} {mcts[key]:>15.{precision}f} {greedy[key]:>15.{precision}f}")
    lines += ["", f"Outcomes: MCTS={mcts['outcome']}; greedy={greedy['outcome']}.", "", "MCTS minus greedy:",
        f"  {difference['placed_stones']:+d} stones",
        f"  {difference['target_filling_percentage_points']:+.4f} percentage points filling",
        f"  {difference['filled_target_volume_m3']:+.6f} m3 filled target volume",
        f"  {difference['total_search_time_s']:+.3f} s search time", "", "Definitions:"]
    lines += [f"  {key}: {value}" for key, value in definitions.items()]
    text_path = directory / "numerical_results.txt"
    text_path.write_text("\n".join(lines)+"\n")
    return [json_path, csv_path, text_path]


def load_result(directory, method, seed):
    import pyvista as pv
    from scipy.spatial.transform import Rotation
    from stacking_common.visualization.dsf import object_surface_path

    path = directory / f"{method}-seed{seed}.stack"
    summary_path = directory / (f"{method}-summary.json" if seed == 0 else f"{method}-seed{seed}-summary.json")
    summary = json.loads(summary_path.read_text())
    checkpoint = StackRunStore.load_planning_checkpoint(path)
    context = StackRunStore.load_run_context(path, checkpoint.context_sha256)
    state = checkpoint.state
    profile = context.profiles["inventory_profile"]
    if summary["seed"] != seed or context.profiles["runtime"]["seed"] != seed:
        raise ValueError("summary/checkpoint seed mismatch")
    if len(state.object_ids) != 49 or tuple(state.object_ids) != tuple(profile["object_ids"]):
        raise ValueError("expected the full 49-stone inventory")
    if summary["placed"] != len(state.placed_object_indices):
        raise ValueError("summary/checkpoint placed count mismatch")
    triangles, normals, sources, poses = [], [], {}, {}
    saved_models = TemporaryDirectory(prefix='stack-figure-models-')
    paths = materialize_object_assets(path, context, ROOT, saved_models.name)
    for index in state.placed_object_indices:
        object_id = state.object_ids[index]
        mesh_path = object_surface_path(paths[f"object/{object_id}"], "ply")
        digest = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
        expected = context.asset_hashes.get(f"object_surface/{object_id}")
        if expected is None or digest != expected:
            raise ValueError(f"original surface does not match recorded hash: {mesh_path}")
        mesh = pv.read(mesh_path).triangulate().compute_normals(
            point_normals=True, cell_normals=False, consistent_normals=True,
            auto_orient_normals=True, split_vertices=False)
        pose = state.object_poses[object_id]
        rotation = Rotation.from_quat(pose[3:])
        points = rotation.apply(mesh.points) + pose[:3]
        faces = mesh.faces.reshape(-1, 4)[:, 1:]
        triangles.append(points[faces])
        normals.append(rotation.apply(mesh.point_normals)[faces])
        sources[str(object_id)] = dict(asset_role=f"object_surface/{object_id}", sha256=digest,
                                      triangles=int(mesh.n_cells))
        poses[str(object_id)] = pose.tolist()
    return np.concatenate(triangles), np.concatenate(normals), dict(
        method=method, seed=seed, source_run=str(path.resolve()),
        source_run_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        summary_path=str(summary_path.resolve()),
        summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        checkpoint_step=checkpoint.step, context_sha256=context.sha256,
        inventory_ids=list(state.object_ids), target=plain_json(profile["target"]),
        placed=len(state.placed_object_indices), poses=poses, meshes=sources,
        outcome=summary["outcome"], search_seconds=summary["total_search_seconds"],
        metrics=summary["metrics"], runtime=plain_json(context.profiles["runtime"]))


def render_comparison(directory, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    if results[0][2]["target"] != results[1][2]["target"]:
        raise ValueError("comparison target profiles differ")
    if results[0][2]["inventory_ids"] != results[1][2]["inventory_ids"]:
        raise ValueError("comparison inventories differ")
    target = results[0][2]["target"]
    if any(target["taper"].values()):
        raise ValueError("this benchmark figure expects its recorded untapered target")
    # Nearly normal to the broad Y-Z wall face, exposing inter-stone cavities.
    basis = orthographic_basis([10., -1., 1.2])
    origin = np.array(target["origin"])
    corners = np.array(list(product((-.5*target["width"], .5*target["width"]),
                                    (-.5*target["length"], .5*target["length"]),
                                    (0., target["height"])))) + origin
    edges = np.array([[a, b] for i, a in enumerate(corners) for b in corners[i+1:]
                      if np.count_nonzero(abs(a-b) > 1e-10) == 1])
    cloud = np.vstack([result[0].reshape(-1, 3) for result in results] + [corners])
    low, high = cloud.min(axis=0), cloud.max(axis=0)
    ground = np.array([[low[0]-.35, low[1]-.35, origin[2]],
                       [high[0]+.35, low[1]-.35, origin[2]],
                       [high[0]+.35, high[1]+.35, origin[2]],
                       [low[0]-.35, high[1]+.35, origin[2]]])
    projection = np.vstack([cloud, ground]) @ basis[:2].T
    lower, upper = projection.min(axis=0), projection.max(axis=0)
    margin = (upper-lower)*.04
    lower -= margin
    upper += margin
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.85), facecolor="white")
    fig.subplots_adjust(left=.012, right=.988, top=.99, bottom=.20, wspace=.025)
    try:
        for index, (ax, (triangles, normals, record)) in enumerate(zip(axes, results)):
            ax.add_collection(PolyCollection([ground @ basis[:2].T], facecolors="#5a5046",
                                             edgecolors="none", zorder=0, rasterized=False))
            ax.add_collection(LineCollection(edges @ basis[:2].T, colors="#3977a8",
                linewidths=.65, linestyles=(0, (3, 2)), zorder=1, rasterized=False))
            polygons, colors = vector_mesh_faces(triangles, basis, vertex_normals=normals)
            ax.add_collection(PolyCollection(polygons, facecolors=colors, edgecolors="face",
                linewidths=.12, antialiaseds=False, zorder=2, rasterized=False))
            ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), aspect="equal")
            ax.set_axis_off()
            record["original_triangle_count"] = len(triangles)
            record["visible_triangle_count"] = len(polygons)
            bounds = ax.get_position()
            center = bounds.x0+bounds.width/2
            name = "MCTS" if record["method"] == "mcts" else "Greedy"
            fig.text(center, .14, f"({chr(97+index)}) {name}", ha="center", va="center", fontsize=11)
            fig.text(center, .065,
                f"{record['placed']} stones; {100*record['metrics']['target_coverage']:.1f}% target coverage",
                ha="center", va="center", fontsize=8.5)
        paths = []
        for extension in ("pdf", "svg", "png"):
            path = directory / f"mcts_greedy_seed{results[0][2]['seed']}.{extension}"
            fig.savefig(path, dpi=600, facecolor="white")
            paths.append(path)
    finally:
        plt.close(fig)
    return paths, dict(basis=basis.tolist(), shared_projected_bounds=[lower.tolist(), upper.tolist()],
        ground_corners=ground.tolist(), size_inches=[7.16, 2.85], png_dpi=600,
        stone_color=[.5, .5, .5], ground_color="#5a5046", target_outline="#3977a8",
        mesh_simplification=False, original_poses=True, projection="orthographic, world Z up",
        visibility="back-face culling and global far-to-near triangle sorting",
        shading="vertex-normal lighting averaged per original face; solid vector fills",
        pdf_content="vector mesh faces, ground, target outline, and text; no embedded bitmaps")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for method in ("mcts", "greedy"):
        print(f"Loading full-inventory {method}, seed {args.seed}", flush=True)
        result = load_result(args.benchmark, method, args.seed)
        results.append(result)
        print(f"  {result[2]['placed']} stones; {100*result[2]['metrics']['target_coverage']:.2f}% coverage", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    paths, settings = render_comparison(args.output, results)
    record = dict(benchmark=str(args.benchmark.resolve()), seed=args.seed,
        rendering=settings, results=[result[2] for result in results],
        coverage_definition=COVERAGE_DEFINITION)
    numerical_paths = write_numerical_results(args.output, record["results"])
    record["numerical_results"] = [path.name for path in numerical_paths]
    (args.output / "comparison.json").write_text(json.dumps(record, indent=2)+"\n")
    for path in paths + numerical_paths:
        print(path, flush=True)


if __name__ == "__main__":
    main()
