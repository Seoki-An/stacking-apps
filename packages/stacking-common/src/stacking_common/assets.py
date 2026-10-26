"""Loading of commissioned platform convex-component OBJ assets."""

import ast
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stacking_common.domain import (
    ConvexGeometry,
    ObjectInventory,
    StackObject,
    TargetRegion,
)


@dataclass(frozen=True)
class _Component:
    geometry: ConvexGeometry
    friction: float


@dataclass(frozen=True)
class _ObjectAsset:
    mass: float
    inertia: np.ndarray
    components: tuple[_Component, ...]


class ObjectAssetLoader:
    """Build domain and stacking-core models from platform OBJ assets."""

    def __init__(
        self,
        model_directory: str | Path,
        density: float,
        target: TargetRegion,
        density_overrides: Mapping[int, float] | None = None,
        core_module=None,
    ) -> None:
        self.model_directory = Path(model_directory)
        if not self.model_directory.is_dir():
            raise FileNotFoundError(
                f"missing object model directory: {self.model_directory}"
            )
        self.density = float(density)
        self.density_overrides = {
            int(object_id): float(value)
            for object_id, value in (density_overrides or {}).items()
        }
        if not np.isfinite(self.density) or self.density <= 0.0:
            raise ValueError("object density must be positive and finite")
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in self.density_overrides.values()
        ):
            raise ValueError("object density overrides must be positive")
        self.target = target
        if core_module is None:
            try:
                import stacking_core as core_module
            except ImportError as error:
                raise RuntimeError(
                    "stacking-core Python bindings are required"
                ) from error
        self.core = core_module

    def load(
        self,
        object_ids: Sequence[int],
        initial_poses: Mapping[int, np.ndarray],
    ) -> ObjectInventory:
        objects = [self.load_object(object_id) for object_id in object_ids]
        return ObjectInventory(objects, initial_poses, self.target)

    def load_object(self, object_id: int) -> StackObject:
        object_id = int(object_id)
        path = self.model_directory / f"model_{object_id}.obj"
        if not path.is_file():
            raise FileNotFoundError(f"missing object model: {path}")
        asset = _parse_object_asset(path)
        density = self.density_overrides.get(object_id, self.density)
        mass = density * asset.mass
        inertia = density * asset.inertia
        geometries = tuple(
            component.geometry for component in asset.components
        )
        try:
            body_model = self.core.BodyModel.dsf_components(
                object_id,
                [geometry.points for geometry in geometries],
                mass=mass,
                inertia=inertia,
                frictions=[
                    component.friction for component in asset.components
                ],
                sharpnesses=[
                    int(round(geometry.sharpness)) for geometry in geometries
                ],
            )
        except AttributeError as error:
            raise RuntimeError(
                "stacking-core bindings must provide BodyModel.dsf_components"
            ) from error
        return StackObject(object_id, body_model, geometries)


def _parse_object_asset(path: Path) -> _ObjectAsset:
    lines = path.read_text().splitlines()
    header_lines = []
    vertices: list[list[float]] = []
    components: list[dict[str, object]] = []
    current: dict[str, object] | None = None

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("o "):
            current = {
                "vertices": [],
                "faces": [],
                "sharpness": None,
                "friction": None,
            }
            components.append(current)
        elif line.startswith("#"):
            comment = line[1:].strip()
            if current is None:
                header_lines.append(comment)
            elif comment.startswith("sharpness:"):
                current["sharpness"] = float(comment.partition(":")[2])
            elif comment.startswith("mu:"):
                current["friction"] = float(comment.partition(":")[2])
        elif line.startswith("v "):
            values = [float(value) for value in line.split()[1:]]
            if len(values) != 3:
                raise ValueError(f"{path}:{line_number}: vertex must be XYZ")
            vertices.append(values)
            if current is not None:
                current["vertices"].append(len(vertices) - 1)
        elif line.startswith("f "):
            if current is None:
                raise ValueError(f"{path}:{line_number}: face has no component")
            indices = [
                _vertex_index(token, len(vertices), path, line_number)
                for token in line.split()[1:]
            ]
            if len(indices) < 3:
                raise ValueError(
                    f"{path}:{line_number}: face needs three vertices"
                )
            faces = current["faces"]
            for index in range(1, len(indices) - 1):
                faces.append([indices[0], indices[index], indices[index + 1]])

    mass, inertia = _header_inertial(path, header_lines)
    point_values = np.asarray(vertices, dtype=float)
    parsed_components = []
    for index, component in enumerate(components):
        faces = np.asarray(component["faces"], dtype=int)
        sharpness = component["sharpness"]
        friction = component["friction"]
        if len(faces) == 0 or sharpness is None or friction is None:
            raise ValueError(f"{path}: component {index} is incomplete")
        declared = np.asarray(component["vertices"], dtype=int)
        used = np.unique(
            np.concatenate((declared, faces.reshape(-1)))
        )
        remap = np.full(len(point_values), -1, dtype=int)
        remap[used] = np.arange(len(used))
        geometry = ConvexGeometry(
            point_values[used],
            remap[faces],
            sharpness=float(sharpness),
        )
        parsed_components.append(_Component(geometry, float(friction)))
    if not parsed_components:
        raise ValueError(f"{path}: object contains no convex components")
    return _ObjectAsset(mass, inertia, tuple(parsed_components))


def _header_inertial(
    path: Path, header_lines: list[str]
) -> tuple[float, np.ndarray]:
    header = " ".join(header_lines)
    mass_match = re.search(r"(?:^|\s)mass:\s*([^\s]+)", header)
    inertia_match = re.search(r"inertia:\s*(\[\[.*?\]\])", header)
    if mass_match is None or inertia_match is None:
        raise ValueError(f"{path}: missing mass or inertia metadata")
    mass = float(mass_match.group(1))
    inertia = np.asarray(ast.literal_eval(inertia_match.group(1)), dtype=float)
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"{path}: mass metadata must be positive")
    if inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)):
        raise ValueError(f"{path}: inertia metadata must have shape (3, 3)")
    if np.min(np.linalg.eigvalsh(inertia)) <= 0.0:
        raise ValueError(f"{path}: inertia metadata must be positive definite")
    return mass, inertia


def _vertex_index(
    token: str, vertex_count: int, path: Path, line_number: int
) -> int:
    raw_index = int(token.partition("/")[0])
    index = raw_index - 1 if raw_index > 0 else vertex_count + raw_index
    if index < 0 or index >= vertex_count:
        raise ValueError(f"{path}:{line_number}: invalid vertex index")
    return index
