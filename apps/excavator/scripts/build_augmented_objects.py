"""Build a simulation-only catalog: real stones plus freshly fitted mesh deformations.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python apps/excavator/scripts/build_augmented_objects.py

The default output contains 100 stones (49 originals + 51 synthetic variants).
Finished objects are verified and reused on restart; incompatible settings or
changed sources are rejected. The normal excavator inventory is not modified.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import argparse
import hashlib
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages/stacking-common/src"))

from stacking_common.reconstruction.augmentation import fit_augmented_stone
from stacking_common.profiles.inventory import inventory_choices, load_inventory_objects
from stacking_common.reconstruction.config import (
    ConvexDecompositionProfile, DsfExportProfile, DsfFittingProfile, DsfRansacProfile)


def profiles():
    """Excavator SI fitting values; deformation itself uses normalized coordinates."""
    return (
        ConvexDecompositionProfile(enabled=True, threshold=.02, maximum_components=4,
            preprocess_mode="auto", preprocess_resolution=50, resolution=2000,
            mcts_nodes=20, mcts_iterations=150, mcts_max_depth=3, pca=False,
            merge=True, decimate=False, maximum_hull_vertices=256, seed=0, real_metric=False),
        DsfFittingProfile(enabled=True, fit_source="both", total_vertices=48,
            direction_subdivisions=3, maximum_iterations=300, support_quantile=.999,
            loss="huber", loss_scale=.02, center_interior_upper_bound=.8,
            maximum_sharpness=60, split_inside_tolerance=.005, split_minimum_points=20,
            ransac=DsfRansacProfile(enabled=False, iterations=5, sample_ratio=.5,
                minimum_sample_points=200, maximum_sample_points=2000,
                fit_maximum_iterations=25, inlier_threshold=.02,
                minimum_inlier_ratio=.5, score_direction_subdivisions=2, seed=0)),
        DsfExportProfile(maximum_nodes=128, sharpness=20, friction=.4))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-inventory", type=Path, default=ROOT / "apps/excavator/config/inventory.yml")
    parser.add_argument("--output", type=Path, default=ROOT / "apps/excavator/assets/objects_augmented")
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    _, choices = inventory_choices(args.source_inventory)
    source, _ = load_inventory_objects(args.source_inventory,
        inventory_name="real" if "real" in choices else None)
    ids = source["ids"]
    if args.total <= len(ids) or args.workers < 1 or args.seed < 0:
        parser.error("total must exceed the source count; workers must be positive and seed nonnegative")
    source_directory = ROOT / source["model_directory"]
    args.output = args.output.resolve()
    if args.output == source_directory.resolve():
        parser.error("augmented output must differ from the original model directory")
    args.output.mkdir(parents=True, exist_ok=True)
    convex, fitting, exporter = profiles()
    parents = []
    rng = np.random.default_rng(args.seed)
    while len(parents) < args.total-len(ids):
        parents.extend(int(value) for value in rng.permutation(ids))
    parents = parents[:args.total-len(ids)]
    planned = [dict(object_id=max(ids)+1+index, parent_id=parent,
        seed=int(np.random.SeedSequence([args.seed, max(ids)+1+index, parent]).generate_state(1)[0]))
        for index, parent in enumerate(parents)]
    settings = dict(algorithm="principal-axis-anisotropic-gaussian-v1",
        software_versions={name: version(name) for name in ("numpy", "scipy", "open3d", "coacd")},
        seed=args.seed, total=args.total, source_ids=ids, planned=planned,
        convex_profile=asdict(convex), fitting_profile=asdict(fitting), export_profile=asdict(exporter))
    settings["source_hashes"] = {path.name: digest(path) for oid in ids
        for path in (source_directory / f"model_{oid}.obj", source_directory / f"model_{oid}_mesh.ply")}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["settings"] != settings:
            raise ValueError("existing catalog settings/sources differ; choose another output directory")
    else:
        manifest = dict(status="building", settings=settings, models={})
    manifest["status"] = "building"
    manifest["failures"] = {}
    for oid in ids:
        files = {}
        for suffix in (".obj", "_mesh.ply"):
            name = f"model_{oid}{suffix}"
            destination = args.output / name
            expected = settings["source_hashes"][name]
            if destination.exists() and digest(destination) != expected:
                raise ValueError(f"copied original changed: {destination}")
            if not destination.exists():
                shutil.copy2(source_directory / name, destination)
            files[name] = expected
        manifest["models"][str(oid)] = dict(kind="real", object_id=oid, files=files)
    save(manifest_path, manifest)
    # Bound native/BLAS parallelism inside each independent fitting process.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    futures = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for plan in planned:
            oid, parent = plan["object_id"], plan["parent_id"]
            path = args.output / f"model_{oid}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if record["deformation"]["seed"] != plan["seed"]:
                    raise ValueError(f"saved deformation differs: {path}")
                if not all(digest(args.output / name) == expected for name, expected in record["files"].items()):
                    raise ValueError(f"saved augmented asset changed: {path}")
                record["parent_id"] = parent
                record.setdefault("fitting_retry_count", 0)
                save(path, record)
                manifest["models"][str(oid)] = record
                continue
            future = pool.submit(fit_augmented_stone,
                source_directory / f"model_{parent}_mesh.ply", source_directory / f"model_{parent}.obj",
                args.output, oid, plan["seed"], convex, fitting, exporter)
            futures[future] = plan
        for future in as_completed(futures):
            plan = futures[future]
            try:
                record = future.result()
            except Exception as error:
                manifest["failures"][str(plan["object_id"])] = repr(error)
                save(manifest_path, manifest)
                print(f"Failed synthetic {plan['object_id']}: {error}", flush=True)
                continue
            record["parent_id"] = plan["parent_id"]
            save(args.output / f"model_{plan['object_id']}.json", record)
            manifest["models"][str(plan["object_id"])] = record
            save(manifest_path, manifest)
            print(f"Completed {len(manifest['models'])}/{args.total}: synthetic {plan['object_id']} "
                  f"from {plan['parent_id']}, {record['convex_components']} DSFs", flush=True)
    manifest["status"] = "failed" if manifest["failures"] else "complete"
    manifest["object_ids"] = sorted(int(oid) for oid in manifest["models"])
    manifest["implementation_sha256"] = {str(path.relative_to(ROOT)): digest(path) for path in (
        Path(__file__).resolve(),
        *(ROOT / "packages/stacking-common/src/stacking_common/reconstruction" / name
          for name in ("augmentation.py", "convex_decomposition.py", "dsf_fitting.py", "geometry.py")))}
    save(manifest_path, manifest)
    if manifest["failures"]:
        raise RuntimeError(f"{len(manifest['failures'])} synthetic fits failed; see {manifest_path}")
    print(f"Catalog complete: {len(ids)} real + {len(planned)} synthetic = {len(manifest['models'])}", flush=True)


if __name__ == "__main__":
    main()
