"""Explicitly migrate measured catalog XYZ from a trusted legacy checkout."""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

from stacking_excavator.acquisition.storage import file_hash


def export(legacy, repository):
    profile = yaml.safe_load((repository / "apps/excavator/config/inventory.yml").read_text())
    directory = repository / "apps/excavator/assets/perception"
    directory.mkdir(exist_ok=True)
    manifest = {"source_project": "stacking-planner", "encoding": "exact measured XYZ, compressed NPZ", "models": {}}
    for object_id in profile["objects"]["ids"]:
        source = legacy / f"assets/stone/model_{object_id}.pcd"
        path = directory / f"model_{object_id}.npz"
        cloud = o3d.io.read_point_cloud(str(source))
        points = np.asarray(cloud.points)
        if not len(points) or not np.all(np.isfinite(points)):
            raise ValueError(f"invalid reference cloud: {source}")
        np.savez_compressed(path, xyz=points)
        manifest["models"][str(object_id)] = {"source": str(source.relative_to(legacy)),
            "source_sha256": file_hash(source), "sha256": file_hash(path), "points": len(points)}
    (directory / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy", type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args()
    export(args.legacy, args.repository)
