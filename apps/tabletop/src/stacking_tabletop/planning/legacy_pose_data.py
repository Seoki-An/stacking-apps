"""Trusted migration adapter for legacy tabletop ``pose_data.pkl`` files."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from stacking_common.domain.geometry import transform_points
from stacking_common.planning import PickSceneSnapshot


@dataclass(frozen=True)
class PickSupportEstimate:
    """Legacy one-percent model bottoms and their robust table height."""

    height: float
    bottom_by_object: Mapping[int, float]


def infer_legacy_pick_support(
    scene: PickSceneSnapshot,
    catalog,
    *,
    bottom_quantile: float = 0.01,
) -> PickSupportEstimate:
    """Reproduce legacy pickup-table inference from registered model clouds."""

    quantile = float(bottom_quantile)
    if not np.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError("pick-support bottom quantile must be in [0, 1]")
    bottoms: dict[int, float] = {}
    for object_id in scene.object_ids:
        try:
            model = catalog.load(object_id)
        except (KeyError, FileNotFoundError, ValueError):
            continue
        points = np.asarray(model.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            continue
        world = transform_points(points, scene.pose(object_id))
        if not np.all(np.isfinite(world)):
            continue
        bottoms[object_id] = float(np.quantile(world[:, 2], quantile))
    if not bottoms:
        raise ValueError(
            "cannot infer pickup table height: pose data has no matching "
            "finite object model point clouds"
        )
    return PickSupportEstimate(
        float(np.median(tuple(bottoms.values()))),
        MappingProxyType(bottoms),
    )


def load_legacy_pick_scene(
    path: str | Path,
    *,
    frame_id: str = "world",
    object_ids: Sequence[int] | None = None,
) -> PickSceneSnapshot:
    """Load the legacy ``{id: (position, quaternion)}`` pickle contract.

    Pickle is intentionally retained only at this migration boundary. Callers
    must supply a trusted local artifact produced by legacy stacking-tabletop.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"missing legacy pick pose data: {source}")
    try:
        with source.open("rb") as stream:
            values = pickle.load(stream)
    except (OSError, pickle.PickleError, EOFError) as error:
        raise ValueError(f"cannot read legacy pick pose data: {source}") from error
    if not isinstance(values, Mapping):
        raise TypeError("legacy pick pose data must contain an object mapping")

    poses: dict[int, np.ndarray] = {}
    for raw_id, raw_value in values.items():
        object_id = int(raw_id)
        if object_id in poses:
            raise ValueError(f"duplicate legacy pick-pose object ID: {object_id}")
        if not isinstance(raw_value, (tuple, list)) or len(raw_value) != 2:
            raise TypeError(
                f"legacy pick pose for object {object_id} must be "
                "(position, quaternion)"
            )
        position = np.asarray(raw_value[0], dtype=float)
        quaternion = np.asarray(raw_value[1], dtype=float)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError(
                f"legacy pick pose for object {object_id} must have "
                "position (3,) and quaternion (4,)"
            )
        poses[object_id] = np.concatenate([position, quaternion])

    snapshot = PickSceneSnapshot(
        frame_id=frame_id,
        object_poses=poses,
        source=str(source),
        metadata={
            "format": "legacy_pose_data_pickle",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    )
    return snapshot if object_ids is None else snapshot.select(object_ids)
