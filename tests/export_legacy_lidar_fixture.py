"""Run the trusted legacy ray caster without its ROS/URDF import dependencies.

This explicit fixture exporter executes only simulate_lidar's AST. The model
adapter supplies already projected meshes and a fixed sensor pose; the original
ray generation, intersections and sensor-frame conversion run unmodified.
Runtime and tests consume the resulting NPZ, never the legacy checkout.
"""

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d

from stacking_excavator.acquisition.emulation import demonstration
from stacking_excavator.acquisition.storage import file_hash


def export(legacy_root, repository, output):
    source = legacy_root / "perception/utils/simulate_lidar.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "simulate_lidar")
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    scope = {"np": np, "o3d": o3d, "update_urdf_mesh": lambda model, meshes, q: meshes}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    values = {"source": str(source.relative_to(legacy_root)), "source_sha256": file_hash(source),
              "open3d_version": o3d.__version__, "fov": [30., 24.], "cases": {}}
    arrays = {}
    for kind in ("field", "inhand", "scene"):
        meshes, sensors, gripper, joints, crop = demonstration(repository, kind)
        values["cases"][kind] = {"sensors": [s.tolist() for s in sensors],
            "world_T_gripper": gripper.tolist(), "joint_positions": joints, "crop": crop.values()}
        for index, sensor in enumerate(sensors):
            link = SimpleNamespace(GetRotation=lambda: sensor[:3, :3], GetPosition=lambda: sensor[:3, 3])
            model = SimpleNamespace(GetLink=lambda name: link)
            cloud, transform = scope["simulate_lidar"](model, {}, dict(enumerate(meshes)),
                values["fov"], "emulated-lidar", np.asarray(joints), False)
            np.testing.assert_array_equal(transform, sensor)
            arrays[f"{kind}_{index}"] = np.asarray(cloud.points)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    values["fixture_sha256"] = file_hash(output)
    values["asset_hashes"] = {str(path.relative_to(repository)): file_hash(path)
        for path in sorted((repository / "apps/excavator/assets").rglob("*"))
        if path.suffix in {".stl", ".urdf"} or path.name == "model_10_mesh.ply"}
    output.with_suffix(".json").write_text(json.dumps(values, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_root", type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/legacy_planner/lidar.npz"))
    args = parser.parse_args()
    export(args.legacy_root, args.repository, args.output)
