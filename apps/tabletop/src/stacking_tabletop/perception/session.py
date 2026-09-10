"""Persistent, GUI-independent reconstruction session state."""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from stacking_common.pointcloud import VoxelMergeStats, validate_rigid_transform


SESSION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ReconstructionViewRecord:
    name: str
    path: str
    accepted: bool = True
    capture_index: int | None = None
    tool_T_camera: np.ndarray | None = None
    transform: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.path:
            raise ValueError("reconstruction view name and path are required")
        if self.capture_index is not None and self.capture_index < 0:
            raise ValueError("reconstruction capture index cannot be negative")
        if self.tool_T_camera is not None:
            viewpoint = validate_rigid_transform(
                self.tool_T_camera,
                "reconstruction tool_T_camera",
            )
            viewpoint.setflags(write=False)
            object.__setattr__(self, "tool_T_camera", viewpoint)
        transform = np.eye(4) if self.transform is None else self.transform
        value = validate_rigid_transform(
            transform,
            "reconstruction view transform",
        )
        value.setflags(write=False)
        object.__setattr__(self, "transform", value)


@dataclass(frozen=True)
class ReconstructionSession:
    object_id: int
    views: tuple[ReconstructionViewRecord, ...]
    loop_edges: tuple[tuple[int, int], ...] = ()
    status: str = "created"
    pose_graph_optimized: bool = False
    pose_graph_reason: str | None = None
    merge_stats: VoxelMergeStats | None = None
    unit_density_mass: float | None = None
    inertia: np.ndarray | None = None
    center_of_mass: np.ndarray | None = None
    outputs: tuple[tuple[str, str], ...] = ()
    schema_version: int = SESSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.object_id <= 0:
            raise ValueError("reconstruction object ID must be positive")
        if self.schema_version != SESSION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported reconstruction schema {self.schema_version}"
            )
        names = [view.name for view in self.views]
        if len(names) != len(set(names)):
            raise ValueError("reconstruction view names must be unique")
        capture_indices = [
            view.capture_index
            for view in self.views
            if view.capture_index is not None
        ]
        if capture_indices and len(capture_indices) != len(self.views):
            raise ValueError("capture indices must be present for every view or none")
        if any(
            second <= first
            for first, second in zip(capture_indices, capture_indices[1:])
        ):
            raise ValueError("reconstruction capture indices must be strictly increasing")
        for source, target in self.loop_edges:
            if (
                source < 0
                or target < 0
                or source >= len(self.views)
                or target >= len(self.views)
                or source == target
            ):
                raise ValueError(
                    f"invalid session loop edge ({source}, {target})"
                )
        inertial_values = (
            self.unit_density_mass,
            self.inertia,
            self.center_of_mass,
        )
        if any(value is not None for value in inertial_values):
            if any(value is None for value in inertial_values):
                raise ValueError("session inertial metadata must be complete")
            from .reconstruction import InertialMetadata

            inertial = InertialMetadata(
                float(self.unit_density_mass),
                self.inertia,
                self.center_of_mass,
            )
            object.__setattr__(self, "inertia", inertial.inertia)
            object.__setattr__(
                self,
                "center_of_mass",
                inertial.center_of_mass,
            )

    def accepted_views(self) -> tuple[ReconstructionViewRecord, ...]:
        return tuple(view for view in self.views if view.accepted)

    def with_refinement(
        self,
        transforms: tuple[np.ndarray, ...],
        *,
        optimized: bool,
        reason: str | None,
    ) -> "ReconstructionSession":
        accepted = self.accepted_views()
        if len(transforms) != len(accepted):
            raise ValueError(
                "refinement transform count does not match accepted views"
            )
        transform_by_name = {
            view.name: transform for view, transform in zip(accepted, transforms)
        }
        views = tuple(
            replace(
                view,
                transform=transform_by_name.get(view.name, view.transform),
            )
            for view in self.views
        )
        return replace(
            self,
            views=views,
            status="refined",
            pose_graph_optimized=optimized,
            pose_graph_reason=reason,
        )

    def with_outputs(
        self,
        stats: VoxelMergeStats,
        outputs: dict[str, str],
        *,
        unit_density_mass: float,
        inertia: np.ndarray,
        center_of_mass: np.ndarray,
    ) -> "ReconstructionSession":
        return replace(
            self,
            status="exported",
            merge_stats=stats,
            unit_density_mass=unit_density_mass,
            inertia=inertia,
            center_of_mass=center_of_mass,
            outputs=tuple(sorted(outputs.items())),
        )


class ReconstructionSessionStore:
    """Read and atomically replace a reconstruction JSON manifest."""

    def save(self, session: ReconstructionSession, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_session_to_dict(session), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    def load(self, path: str | Path) -> ReconstructionSession:
        source = Path(path)
        values = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise TypeError("reconstruction manifest must contain an object")
        return _session_from_dict(values)

    @staticmethod
    def resolve_view_path(
        manifest_path: str | Path,
        view: ReconstructionViewRecord,
    ) -> Path:
        path = Path(view.path)
        return path if path.is_absolute() else Path(manifest_path).parent / path


def _session_to_dict(session: ReconstructionSession) -> dict[str, object]:
    merge_stats = None
    if session.merge_stats is not None:
        merge_stats = dict(session.merge_stats.__dict__)
    return {
        "schema_version": session.schema_version,
        "object_id": session.object_id,
        "status": session.status,
        "views": [
            {
                "name": view.name,
                "path": view.path,
                "accepted": view.accepted,
                "capture_index": view.capture_index,
                "tool_T_camera": (
                    None if view.tool_T_camera is None else view.tool_T_camera.tolist()
                ),
                "transform": view.transform.tolist(),
            }
            for view in session.views
        ],
        "loop_edges": [list(edge) for edge in session.loop_edges],
        "pose_graph": {
            "optimized": session.pose_graph_optimized,
            "reason": session.pose_graph_reason,
        },
        "merge_stats": merge_stats,
        "inertial": (
            None
            if session.unit_density_mass is None
            else {
                "unit_density_mass": session.unit_density_mass,
                "inertia": session.inertia.tolist(),
                "center_of_mass": session.center_of_mass.tolist(),
            }
        ),
        "outputs": dict(session.outputs),
    }


def _session_from_dict(values: dict[str, object]) -> ReconstructionSession:
    view_values = values.get("views")
    if not isinstance(view_values, list):
        raise TypeError("reconstruction manifest views must be a list")
    pose_graph = values.get("pose_graph", {})
    if not isinstance(pose_graph, dict):
        raise TypeError("reconstruction pose_graph must be an object")
    stats_values = values.get("merge_stats")
    stats = None
    if stats_values is not None:
        if not isinstance(stats_values, dict):
            raise TypeError("reconstruction merge_stats must be an object")
        stats = VoxelMergeStats(**stats_values)
    output_values = values.get("outputs", {})
    if not isinstance(output_values, dict):
        raise TypeError("reconstruction outputs must be an object")
    loop_values = values.get("loop_edges", [])
    if not isinstance(loop_values, list):
        raise TypeError("reconstruction loop_edges must be a list")
    inertial_values = values.get("inertial")
    if inertial_values is not None and not isinstance(inertial_values, dict):
        raise TypeError("reconstruction inertial metadata must be an object")
    return ReconstructionSession(
        schema_version=int(values.get("schema_version", -1)),
        object_id=int(values["object_id"]),
        status=str(values.get("status", "created")),
        views=tuple(_view_from_dict(view) for view in view_values),
        loop_edges=tuple((int(edge[0]), int(edge[1])) for edge in loop_values),
        pose_graph_optimized=bool(pose_graph.get("optimized", False)),
        pose_graph_reason=(
            None if pose_graph.get("reason") is None else str(pose_graph["reason"])
        ),
        merge_stats=stats,
        unit_density_mass=(
            None
            if inertial_values is None
            else float(inertial_values["unit_density_mass"])
        ),
        inertia=(
            None
            if inertial_values is None
            else np.asarray(inertial_values["inertia"], dtype=np.float64)
        ),
        center_of_mass=(
            None
            if inertial_values is None
            else np.asarray(
                inertial_values["center_of_mass"], dtype=np.float64
            )
        ),
        outputs=tuple(
            sorted(
                (str(key), str(value))
                for key, value in output_values.items()
            )
        ),
    )


def _view_from_dict(value: object) -> ReconstructionViewRecord:
    if not isinstance(value, dict):
        raise TypeError("each reconstruction view must be an object")
    return ReconstructionViewRecord(
        name=str(value["name"]),
        path=str(value["path"]),
        accepted=bool(value.get("accepted", True)),
        capture_index=(
            None if value.get("capture_index") is None else int(value["capture_index"])
        ),
        tool_T_camera=(
            None
            if value.get("tool_T_camera") is None
            else np.asarray(value["tool_T_camera"], dtype=np.float64)
        ),
        transform=np.asarray(value["transform"], dtype=np.float64),
    )
