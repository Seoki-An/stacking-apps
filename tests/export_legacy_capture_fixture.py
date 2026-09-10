"""Extract compact real recorded frame/transform references, without legacy imports."""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from stacking_excavator.acquisition.storage import file_hash


def export(legacy, output):
    roots = {
        "field": "logs/260707/plan_260707_7/exec_232318_1_nuc/step_44/field_scan",
        "scene": "logs/260706/plan_260706_4/exec_164224_1_nuc/step_39/scene_scan",
        "inhand": "logs/260623/plan_260623_1/exec_121628_1/step_17",
    }
    arrays, metadata = {}, {"sources": {}, "timestamp_source": "recorded_unknown", "cases": {}}
    for kind, relative in roots.items():
        root = legacy / relative
        if kind == "inhand":
            raw = root / "pcd_0.ply"
            points = np.asarray(o3d.io.read_point_cloud(str(raw)).points)
            indices = np.arange(0, len(points), max(1, len(points)//1000))
            arrays["inhand_points"] = points[indices]
            q_path = root / "q_list.npy"
            arrays["inhand_joints"] = np.load(q_path, allow_pickle=False)[0]
            paths = [raw, q_path]
            metadata["cases"][kind] = {"frame": "stone_grab_geometry", "original_sensor_frame": False}
        else:
            raw = root / f"{kind}_raw_iv_points21_swing0.npy"
            tf = root / f"{kind}_lidar_frame_iv_points21_swing0.npy"
            q_path = root / f"{kind}_scan_q_actual_swing0.npy"
            base_path = root / f"{kind}_base_iv_points21_swing0.ply"
            points = np.load(raw, allow_pickle=False)
            base = np.asarray(o3d.io.read_point_cloud(str(base_path)).points)
            if len(points) != len(base):
                raise ValueError("legacy base cloud no longer corresponds row-for-row to raw")
            indices = np.arange(0, len(points), max(1, len(points)//1000))
            arrays[kind + "_points"] = points[indices]
            arrays[kind + "_world_points"] = base[indices]
            arrays[kind + "_world_T_sensor"] = np.load(tf, allow_pickle=False)
            arrays[kind + "_joints"] = np.load(q_path, allow_pickle=False)
            paths = [raw, tf, q_path, base_path]
            metadata["cases"][kind] = {"frame": "iv_points21", "original_sensor_frame": True}
        metadata["sources"].update({str(p.relative_to(legacy)): file_hash(p) for p in paths})
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    metadata["sha256"] = file_hash(output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/legacy_planner/captures.npz"))
    args = parser.parse_args()
    export(args.legacy, args.output)
