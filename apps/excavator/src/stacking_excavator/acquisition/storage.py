"""Immutable capture/crop products and verified local transfer emulation.

The local receiver deliberately exposes interruption by byte count for tests.
It exercises persistence/integrity, not a claim about WiFi or controller timing.
"""

from dataclasses import replace
from pathlib import Path
import hashlib
import json
import os
import tempfile

import numpy as np

from .capture import CaptureFrame, CropRequest, KINDS, digest, identifier, integer


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def finish_manifest(directory, values):
    manifest = {"schema_version": 1, **values}
    manifest["sha256"] = digest(manifest)
    atomic_json(Path(directory) / "manifest.json", manifest)
    return manifest


def load_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = manifest.pop("sha256")
    if manifest.get("schema_version") != 1 or digest(manifest) != expected:
        raise ValueError("invalid manifest version or hash")
    manifest["sha256"] = expected
    identifier(manifest["capture_id"])
    integer(manifest["scene_revision"], "scene revision")
    if manifest["kind"] not in KINDS or not manifest["frames"]:
        raise ValueError("capture needs a purpose and explicit frame set")
    names = []
    for item in manifest["frames"]:
        identifier(item["frame_id"])
        if item["file"] != item["frame_id"] + ".npz":
            raise ValueError("frame filename does not match its identity")
        integer(item["bytes"], "frame byte count", 1)
        integer(item["points"], "frame point count")
        names.append(item["frame_id"])
    if len(names) != len(set(names)):
        raise ValueError("duplicate capture frames")
    if "product_id" in manifest:
        identifier(manifest["product_id"])
        crop = CropRequest(**manifest["crop"])
        if crop.kind != manifest["kind"] or crop.scene_revision != manifest["scene_revision"]:
            raise ValueError("crop context differs from capture")
    return manifest


def save_frame(directory, frame, *, source=None):
    path = Path(directory) / (frame.frame_id + ".npz")
    with path.open("xb") as stream:
        np.savez(stream, xyz=frame.points, **{"attr_" + k: v for k, v in frame.attributes.items()})
        stream.flush()
        os.fsync(stream.fileno())
    return {**frame.metadata(), "file": path.name, "points": len(frame.points),
            "bytes": path.stat().st_size, "sha256": file_hash(path),
            **({"source_sha256": source["sha256"], "source_points": source["points"]} if source else {})}


def read_frame(directory, item):
    path = Path(directory) / item["file"]
    if path.is_symlink() or path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
        raise ValueError("frame size/hash mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        if "xyz" not in arrays.files or any(k != "xyz" and not k.startswith("attr_") for k in arrays.files):
            raise ValueError("unexpected sensor channel encoding")
        metadata = {k: v for k, v in item.items()
                    if k not in {"file", "points", "bytes", "sha256", "source_sha256", "source_points"}}
        frame = CaptureFrame(**metadata, points=arrays["xyz"],
                             attributes={k[5:]: arrays[k] for k in arrays.files if k.startswith("attr_")})
    if len(frame.points) != item["points"]:
        raise ValueError("frame point count mismatch")
    return frame


def create_capture(directory, capture_id, kind, scene_revision, frames, *, recording=None):
    """Write a recorded/emulated capture; incomplete directories have no manifest."""
    identifier(capture_id)
    integer(scene_revision, "scene revision")
    if kind not in KINDS:
        raise ValueError("unsupported scan purpose")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    items = [save_frame(directory, frame) for frame in frames]
    if not items or len({item["frame_id"] for item in items}) != len(items):
        raise ValueError("capture needs a nonempty unique frame set")
    return finish_manifest(directory, {"capture_id": capture_id, "kind": kind,
        "scene_revision": scene_revision, "frames": items,
        **({"recording": recording} if recording is not None else {})})


def prepare_crop(source, destination, product_id, crop, *, maximum_timestamp_skew_ns):
    """Derive a new product without modifying or recapturing the raw source."""
    identifier(product_id)
    manifest = load_manifest(source)
    if "product_id" in manifest:
        raise ValueError("recrop requires retained raw capture, not an earlier crop")
    if crop.kind != manifest["kind"] or crop.scene_revision != manifest["scene_revision"]:
        raise ValueError("crop purpose/revision differs from capture")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    items = []
    for item in manifest["frames"]:
        frame = read_frame(source, item)
        mask = crop.select(frame, maximum_timestamp_skew_ns=maximum_timestamp_skew_ns)
        selected = replace(frame, points=frame.points[mask],
                           attributes={k: values[mask] for k, values in frame.attributes.items()})
        items.append(save_frame(destination, selected, source=item))
    return finish_manifest(destination, {"capture_id": manifest["capture_id"],
        "product_id": product_id, "source_manifest_sha256": manifest["sha256"],
        "kind": manifest["kind"], "scene_revision": manifest["scene_revision"],
        "crop": crop.values(), "frames": items})


def receive_local(source, destination, *, byte_budget=None):
    """Resume a local transfer; publish manifest/receipt only after full verification.

    Return False on an injected interruption. Callers must serialize access to
    one destination, just as the production transfer worker will do.
    """
    if byte_budget is not None:
        integer(byte_budget, "transfer byte budget", 1)
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("source and receiver must be separate stores")
    manifest = load_manifest(source)
    if "product_id" not in manifest:
        raise ValueError("receiver requires an identified crop product")
    destination.mkdir(parents=True, exist_ok=True)
    pending = destination / "transfer.json"
    identity = {"product_id": manifest["product_id"], "sha256": manifest["sha256"]}
    if pending.exists() and json.loads(pending.read_text()) != identity:
        raise ValueError("receiver is bound to another immutable product")
    atomic_json(pending, identity)
    remaining = byte_budget
    for item in manifest["frames"]:
        final = destination / item["file"]
        if final.exists():
            read_frame(destination, item)
            continue
        partial = destination / (item["file"] + ".partial")
        if partial.is_symlink():
            raise ValueError("partial transfer must be a local regular file")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > item["bytes"]:
            raise ValueError("partial transfer exceeds expected size")
        if remaining == 0:
            return False
        with (source / item["file"]).open("rb") as src, partial.open("ab") as dst:
            src.seek(offset)
            while offset < item["bytes"]:
                count = min(65536, item["bytes"] - offset)
                if remaining is not None:
                    count = min(count, remaining)
                if count == 0:
                    break
                data = src.read(count)
                if not data:
                    raise ValueError("source truncated during transfer")
                dst.write(data)
                offset += len(data)
                if remaining is not None:
                    remaining -= len(data)
            dst.flush()
            os.fsync(dst.fileno())
        if offset != item["bytes"]:
            return False
        if file_hash(partial) != item["sha256"]:
            raise ValueError("transferred frame hash mismatch; discard corrupt partial before retry")
        os.replace(partial, final)
        fsync_directory(destination)
        read_frame(destination, item)
    # Repeated calls after a lost ACK return the same receipt; source is retained.
    atomic_json(destination / "manifest.json", manifest)
    atomic_json(destination / "receipt.json", {"capture_id": manifest["capture_id"], **identity})
    return True


def load_product(directory, *, expected_revision=None):
    manifest = load_manifest(directory)
    if "product_id" not in manifest:
        raise ValueError("perception requires a derived transfer product")
    if expected_revision is not None and manifest["scene_revision"] != expected_revision:
        raise ValueError("stale scene revision")
    frames = tuple(read_frame(directory, item) for item in manifest["frames"])
    crop = CropRequest(**manifest["crop"])
    if any(frame.calibration_sha256 != crop.calibration_sha256 for frame in frames):
        raise ValueError("product calibration mismatch")
    return manifest, frames
