"""Platform-neutral array observations for stacking environments."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from stacking_common.domain import ObjectInventory, StackObject, State

from ._geometry import transform_points
from .contracts import HeightMapRenderer


class ObservationTarget(Protocol):
    points: np.ndarray
    faces: np.ndarray
    width: float
    length: float
    height: float
    origin: np.ndarray


class ObservationSource(Protocol):
    objects: Sequence[StackObject]
    points: np.ndarray
    faces: np.ndarray
    sharpness: np.ndarray
    object_surface_points: np.ndarray
    object_surface_normals: np.ndarray
    object_surface_mask: np.ndarray
    object_physical_features: np.ndarray
    target: ObservationTarget
    target_height_map: np.ndarray | None
    xlim: np.ndarray
    ylim: np.ndarray

    def get_height_map(self, state: State) -> np.ndarray | None:
        """Render the current state on the configured height-map grid."""

    def initial_pose(self, object_id: int) -> np.ndarray:
        """Return an object's pose before it enters the placed scene."""


@dataclass(frozen=True)
class ObservationConfig:
    max_poses_per_object: int
    height_map_resolution: tuple[int, int]
    render_height_map: bool
    action_sample_count: int | None = None
    banned_object_ids: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.max_poses_per_object <= 0:
            raise ValueError("max poses per object must be positive")
        if (
            len(self.height_map_resolution) != 2
            or any(value <= 0 for value in self.height_map_resolution)
        ):
            raise ValueError(
                "height-map resolution must contain two positive values"
            )
        if self.action_sample_count is not None and self.action_sample_count <= 0:
            raise ValueError("action sample count must be positive")
        banned = frozenset(int(value) for value in self.banned_object_ids)
        if any(value < 0 for value in banned):
            raise ValueError("banned object IDs must be non-negative")
        object.__setattr__(self, "banned_object_ids", banned)


@dataclass(frozen=True)
class ObservationGeometryBatch:
    """Padded object geometry used by array observations."""

    points: np.ndarray
    faces: np.ndarray
    sharpness: np.ndarray
    surface_points: np.ndarray
    surface_normals: np.ndarray
    surface_mask: np.ndarray
    physical_features: np.ndarray


class InventoryObservationAdapter:
    """Adapt a domain inventory to the arrays required by observations."""

    def __init__(
        self,
        inventory: ObjectInventory,
        surface_sample_count: int,
        height_map_renderer: HeightMapRenderer,
    ) -> None:
        batch = _geometry_batch(inventory.objects, surface_sample_count)
        self.inventory = inventory
        self.objects = inventory.objects
        self.target = inventory.target
        self.points = batch.points
        self.faces = batch.faces
        self.sharpness = batch.sharpness
        self.object_surface_points = batch.surface_points
        self.object_surface_normals = batch.surface_normals
        self.object_surface_mask = batch.surface_mask
        self.object_physical_features = batch.physical_features
        self.xlim = _axis_limits(height_map_renderer.xlim, "x")
        self.ylim = _axis_limits(height_map_renderer.ylim, "y")
        self.target_height_map = _height_map(
            height_map_renderer.render_target(), "target height map"
        )
        self.height_map_renderer = height_map_renderer

    def initial_pose(self, object_id: int) -> np.ndarray:
        return self.inventory.initial_pose(object_id)

    def get_height_map(self, state: State) -> np.ndarray | None:
        return _height_map(
            self.height_map_renderer.render(state), "rendered height map"
        )


@dataclass
class Observation:
    """One observation with immutable static geometry and owned scene arrays."""

    pending_points: np.ndarray
    pending_faces: np.ndarray
    stacked_points: np.ndarray
    stacked_faces: np.ndarray
    sharpness: np.ndarray
    target_points: np.ndarray
    target_faces: np.ndarray
    scene_mask: np.ndarray
    action_mask: np.ndarray
    scene_height_map: np.ndarray | None
    target_height_map: np.ndarray
    target_width: np.ndarray
    target_length: np.ndarray
    target_height: np.ndarray
    target_origin: np.ndarray
    height_map_xlim: np.ndarray
    height_map_ylim: np.ndarray
    object_ids: np.ndarray | None = None
    object_poses: np.ndarray | None = None
    object_surface_points: np.ndarray | None = None
    object_surface_normals: np.ndarray | None = None
    object_surface_mask: np.ndarray | None = None
    object_physical_features: np.ndarray | None = None


class ArrayObservationBuilder:
    """Build an array observation from an explicit inventory view."""

    def __init__(
        self,
        inventory: ObservationSource,
        config: ObservationConfig,
    ) -> None:
        self.inventory = inventory
        self.config = config

        object_count = len(inventory.objects)
        for name in (
            "points",
            "faces",
            "sharpness",
            "object_surface_points",
            "object_surface_normals",
            "object_surface_mask",
            "object_physical_features",
        ):
            if len(getattr(inventory, name)) != object_count:
                raise ValueError(f"observation {name} must have one row per object")
        if (
            inventory.target_height_map is not None
            and inventory.target_height_map.shape != config.height_map_resolution
        ):
            raise ValueError(
                "target height map does not match its configured resolution"
            )

        self._points = _readonly_array(inventory.points, dtype=float)
        self._faces = _readonly_array(inventory.faces, dtype=int)
        self._sharpness = _readonly_array(inventory.sharpness, dtype=float)
        self._surface_points = _readonly_array(
            inventory.object_surface_points
        )
        self._surface_normals = _readonly_array(
            inventory.object_surface_normals
        )
        self._surface_mask = _readonly_array(
            inventory.object_surface_mask, dtype=bool
        )
        self._physical_features = _readonly_array(
            inventory.object_physical_features
        )
        self._target_points = _readonly_array(
            inventory.target.points, dtype=float
        )
        self._target_faces = _readonly_array(
            inventory.target.faces, dtype=int
        )
        self._target_height_map = (
            None
            if inventory.target_height_map is None
            else _readonly_array(inventory.target_height_map, dtype=np.float32)
        )
        self._xlim = _readonly_array(inventory.xlim, dtype=np.float32)
        self._ylim = _readonly_array(inventory.ylim, dtype=np.float32)

    def build(self, state: State) -> Observation:
        inventory = self.inventory
        inventory_ids = [
            int(stack_object.object_id) for stack_object in inventory.objects
        ]
        if state.object_ids != inventory_ids:
            raise ValueError("state inventory does not match observation inventory")
        pending_points = self._points
        pending_faces = self._faces
        object_count = len(state.object_ids)

        stacked_points = np.full_like(pending_points, np.inf, dtype=float)
        stacked_faces = np.full_like(pending_faces, -1, dtype=int)
        for index in state.placed_object_indices:
            stack_object = inventory.objects[index]
            pose = state.object_poses.get(
                stack_object.object_id,
                inventory.initial_pose(stack_object.object_id),
            )
            stacked_points[index] = transform_points(
                pending_points[index], pose
            )
            stacked_faces[index] = pending_faces[index]

        scene_mask = np.zeros(object_count, dtype=bool)
        scene_mask[state.placed_object_indices] = True
        action_mask = self._action_mask(state)

        scene_height_map = (
            inventory.get_height_map(state)
            if self.config.render_height_map
            else None
        )
        scene_height_map = _height_map(scene_height_map, "scene height map")
        if (
            scene_height_map is not None
            and scene_height_map.shape != self.config.height_map_resolution
        ):
            raise ValueError(
                "scene height map does not match its configured resolution"
            )
        if scene_height_map is None:
            scene_height_map = np.zeros(
                self.config.height_map_resolution, dtype=np.float32
            )
        target_height_map = (
            self._target_height_map
            if self._target_height_map is not None
            else np.zeros_like(scene_height_map, dtype=np.float32)
        )

        object_poses = np.zeros((object_count, 7), dtype=np.float32)
        object_poses[:, 6] = 1.0
        for index in state.placed_object_indices:
            stack_object = inventory.objects[index]
            object_poses[index] = state.object_poses.get(
                stack_object.object_id,
                inventory.initial_pose(stack_object.object_id),
            )

        target = inventory.target
        return Observation(
            pending_points=pending_points,
            pending_faces=pending_faces,
            stacked_points=stacked_points,
            stacked_faces=stacked_faces,
            sharpness=self._sharpness,
            target_points=self._target_points,
            target_faces=self._target_faces,
            scene_mask=scene_mask,
            action_mask=action_mask,
            scene_height_map=scene_height_map,
            target_height_map=target_height_map,
            target_width=np.float32(target.width),
            target_length=np.float32(target.length),
            target_height=np.float32(target.height),
            target_origin=np.asarray(target.origin, dtype=np.float32),
            height_map_xlim=self._xlim,
            height_map_ylim=self._ylim,
            object_ids=np.asarray(state.object_ids, dtype=np.int32),
            object_poses=object_poses,
            object_surface_points=self._surface_points,
            object_surface_normals=self._surface_normals,
            object_surface_mask=self._surface_mask,
            object_physical_features=self._physical_features,
        )

    def _action_mask(self, state: State) -> np.ndarray:
        available = sum(
            index not in state.placed_object_indices
            and int(object_id) not in self.config.banned_object_ids
            for index, object_id in enumerate(state.object_ids)
        )
        max_real = available * self.config.max_poses_per_object
        sample_count = self.config.action_sample_count
        if sample_count is None:
            sample_count = (
                len(state.object_ids) * self.config.max_poses_per_object
            )
        real_count = min(sample_count, max_real)
        mask = np.full(sample_count, -np.inf)
        mask[:real_count] = 0.0
        return mask


def _geometry_batch(
    objects: Sequence[StackObject], surface_sample_count: int
) -> ObservationGeometryBatch:
    if surface_sample_count <= 0:
        raise ValueError("surface sample count must be positive")
    if not objects:
        return ObservationGeometryBatch(
            points=np.zeros((0, 0, 3), dtype=float),
            faces=np.zeros((0, 0, 3), dtype=int),
            sharpness=np.zeros(0, dtype=float),
            surface_points=np.zeros(
                (0, surface_sample_count, 3), dtype=np.float32
            ),
            surface_normals=np.zeros(
                (0, surface_sample_count, 3), dtype=np.float32
            ),
            surface_mask=np.zeros((0, surface_sample_count), dtype=bool),
            physical_features=np.zeros((0, 14), dtype=np.float32),
        )

    meshes = [stack_object.mesh_arrays() for stack_object in objects]
    max_points = max(len(points) for points, _ in meshes)
    max_faces = max(len(faces) for _, faces in meshes)
    points = np.full((len(objects), max_points, 3), np.inf, dtype=float)
    faces = np.full((len(objects), max_faces, 3), -1, dtype=int)
    for index, (object_points, object_faces) in enumerate(meshes):
        points[index, : len(object_points)] = object_points
        faces[index, : len(object_faces)] = object_faces

    samples = [
        stack_object.surface_samples(surface_sample_count)
        for stack_object in objects
    ]
    return ObservationGeometryBatch(
        points=points,
        faces=faces,
        sharpness=np.asarray(
            [stack_object.sharpness for stack_object in objects]
        ),
        surface_points=np.stack([sample[0] for sample in samples]),
        surface_normals=np.stack([sample[1] for sample in samples]),
        surface_mask=np.stack([sample[2] for sample in samples]),
        physical_features=np.stack(
            [stack_object.physical_features() for stack_object in objects]
        ),
    )


def _axis_limits(values: np.ndarray, axis: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if (
        result.shape != (2,)
        or not np.all(np.isfinite(result))
        or result[0] >= result[1]
    ):
        raise ValueError(f"{axis}-axis limits must be two increasing finite values")
    return result.copy()


def _height_map(values, name: str) -> np.ndarray | None:
    if values is None:
        return None
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite two-dimensional array")
    return result.copy()


def _readonly_array(values, dtype=None) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result
