"""Resolved planning inputs retained independently of a platform profile."""

from dataclasses import dataclass, field
from collections.abc import Mapping
import hashlib
from pathlib import Path

import numpy as np

from stacking_common.domain._validation import pose_array
from stacking_common.execution.values import (
    content_digest, freeze_json, plain_json, required_text, sha256_text,
)


@dataclass(frozen=True)
class FixedCollisionGeometry:
    """A fixed box (full XYZ size), +Z plane, or mesh asset in the run frame.

    Mesh bytes are identified by content hash; their local origin uses ``pose``.
    Plane pose defines its origin and normal. All lengths are metres.
    """

    name: str
    kind: str
    pose: np.ndarray
    friction: float
    size: tuple[float, ...] = ()
    asset_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", required_text(self.name, "geometry name"))
        if self.kind not in {"box", "plane", "mesh"}:
            raise ValueError("fixed geometry must be box, plane, or mesh")
        pose = pose_array(self.pose, "fixed geometry pose")
        pose.setflags(write=False)
        object.__setattr__(self, "pose", pose)
        size = tuple(float(value) for value in self.size)
        if self.kind == "box":
            if len(size) != 3 or not np.all(np.isfinite(size)) or min(size) <= 0:
                raise ValueError("box size must have three positive finite lengths")
        elif size:
            raise ValueError("only box geometry has a size")
        if self.kind == "mesh":
            sha256_text(self.asset_sha256, "mesh asset hash")
        elif self.asset_sha256 is not None:
            raise ValueError("only mesh geometry has an asset hash")
        if not np.isfinite(self.friction) or self.friction < 0:
            raise ValueError("geometry friction must be finite and non-negative")
        object.__setattr__(self, "size", size)


@dataclass(frozen=True)
class PlanningRunContext:
    """One content-addressed snapshot supplied by the platform composition.

    Profiles contain resolved values, not paths to mutable YAML. Asset hashes
    cover robot/gripper/object meshes and other source data needed for replay.
    Common code cannot infer whether the caller included every native input.
    """

    frame_id: str
    joint_names: tuple[str, ...]
    profiles: Mapping[str, object]
    fixed_geometry: tuple[FixedCollisionGeometry, ...] = ()
    asset_hashes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", required_text(self.frame_id, "context frame"))
        names = tuple(required_text(name, "joint name") for name in self.joint_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("context joint names must be nonempty and unique")
        geometry = tuple(self.fixed_geometry)
        if not all(isinstance(value, FixedCollisionGeometry) for value in geometry):
            raise TypeError("context geometry must be FixedCollisionGeometry values")
        if len({value.name for value in geometry}) != len(geometry):
            raise ValueError("fixed geometry names must be unique")
        for role, digest in self.asset_hashes.items():
            required_text(role, "asset role")
            sha256_text(digest, "asset hash")
        for item in geometry:
            if item.kind == "mesh" and item.asset_sha256 not in self.asset_hashes.values():
                raise ValueError("fixed mesh must be listed in context asset hashes")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "fixed_geometry", geometry)
        object.__setattr__(self, "profiles", freeze_json(self.profiles))
        object.__setattr__(self, "asset_hashes", freeze_json(self.asset_hashes))

    @property
    def sha256(self) -> str:
        return content_digest(encode_run_context(self))


def encode_run_context(value: PlanningRunContext) -> dict[str, object]:
    return plain_json({
        "schema_version": 1,
        "frame_id": value.frame_id,
        "joint_names": value.joint_names,
        "profiles": value.profiles,
        "asset_hashes": value.asset_hashes,
        "fixed_geometry": [
            {"name": item.name, "kind": item.kind, "pose": item.pose,
             "friction": item.friction, "size": item.size,
             "asset_sha256": item.asset_sha256}
            for item in value.fixed_geometry
        ],
    })


def decode_run_context(values: Mapping[str, object]) -> PlanningRunContext:
    if values.get("schema_version") != 1:
        raise ValueError("unsupported planning context schema version")
    return PlanningRunContext(
        values["frame_id"], tuple(values["joint_names"]), values["profiles"],
        tuple(FixedCollisionGeometry(**item) for item in values["fixed_geometry"]),
        values["asset_hashes"],
    )


def verify_context_assets(
    context: PlanningRunContext, assets: Mapping[str, bytes | Path],
) -> None:
    """Verify resolved local files or embedded bytes against all recorded roles."""
    if set(assets) != set(context.asset_hashes):
        raise ValueError("resolved asset roles must match the recorded context")
    for role, source in assets.items():
        if isinstance(source, bytes):
            digest = hashlib.sha256(source).hexdigest()
        elif isinstance(source, Path):
            with source.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        else:
            raise TypeError("resolved asset must be bytes or a Path")
        if digest != context.asset_hashes[role]:
            raise ValueError(f"asset hash mismatch for {role}")
