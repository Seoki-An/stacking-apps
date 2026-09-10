"""Explicit sidecar import of recorded sensor clouds; never infer frame or timing."""

import json
from pathlib import Path
import numpy as np

from .capture import CaptureFrame
from .storage import create_capture, file_hash


def read_sensor_file(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False), {}
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as values:
            if "xyz" not in values or any(k != "xyz" and not k.startswith("attr_") for k in values.files):
                raise ValueError("recorded NPZ requires xyz and optional attr_ channels")
            return values["xyz"], {k[5:]: values[k] for k in values.files if k.startswith("attr_")}
    if path.suffix in {".pcd", ".ply"}:
        import open3d as o3d
        cloud = o3d.t.io.read_point_cloud(str(path))
        channels = {name: cloud.point[name].numpy() for name in cloud.point}
        if "positions" not in channels:
            raise ValueError("recorded cloud has no XYZ positions")
        return channels.pop("positions"), channels
    raise ValueError("recorded sensor file must be NPY, NPZ, PCD or PLY")


def import_recording(sidecar, destination):
    """Frame metadata is CaptureFrame.metadata(), plus source and source_sha256.

    Legacy files lacking timestamps must say recorded_unknown with null times.
    They can be replayed, but cannot establish sensor freshness for execution.
    """
    sidecar = Path(sidecar)
    values = json.loads(sidecar.read_text())
    if values["schema_version"] != 1:
        raise ValueError("unsupported recorded sidecar version")
    frames = []
    for entry in values["frames"]:
        entry = dict(entry)
        path = sidecar.parent / entry.pop("source")
        expected = entry.pop("source_sha256")
        if file_hash(path) != expected:
            raise ValueError("recorded source hash mismatch")
        points, attributes = read_sensor_file(path)
        frames.append(CaptureFrame(**entry, points=points, attributes=attributes))
    return create_capture(destination, values["capture_id"], values["kind"],
        values["scene_revision"], frames,
        recording={"sidecar_sha256": file_hash(sidecar),
                   "sources": [{"frame_id": e["frame_id"], "source": e["source"],
                                "sha256": e["source_sha256"]} for e in values["frames"]]})
