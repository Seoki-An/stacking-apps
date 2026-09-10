"""Backend-neutral DSF component parsing and numeric support surfaces."""

from pathlib import Path

import numpy as np


def dsf_support_points(nodes, directions, sharpness):
    """Evaluate the centered positive-power support map used by stacking-core.

    The gradient of h(d) = ||max((v-c) dot d, 0)||_p gives the surface
    point with outward normal d. Keep the native coefficient cutoff and
    normalization so large sharpness values do not overflow.
    """
    nodes = np.asarray(nodes, dtype=float)
    directions = np.asarray(directions, dtype=float)
    sharpness = float(sharpness)
    if (nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) < 4
            or not np.all(np.isfinite(nodes))):
        raise ValueError("DSF nodes must be finite Nx3 points with at least four nodes")
    if (directions.ndim != 2 or directions.shape[1] != 3
            or not np.all(np.isfinite(directions))
            or np.any(np.linalg.norm(directions, axis=1) == 0)):
        raise ValueError("DSF directions must be finite nonzero Nx3 vectors")
    if not np.isfinite(sharpness) or sharpness < 2 or sharpness != round(sharpness):
        raise ValueError("DSF sharpness must be an integer >= 2")
    center = nodes.mean(axis=0)
    local = nodes - center
    projection = np.maximum(directions @ local.T, 0.)
    scale = projection.max(axis=1, keepdims=True)
    normalized = np.divide(projection, scale, out=np.zeros_like(projection), where=scale > 0)
    normalized[normalized <= np.finfo(float).eps ** (1. / sharpness)] = 0.
    powers = normalized ** (sharpness - 1.)
    total = (powers * normalized).sum(axis=1)
    factor = np.zeros_like(total)
    active = total > 0
    factor[active] = total[active] ** (1. / sharpness - 1.)
    return center + (powers @ local) * factor[:, None]


def read_dsf_components(path):
    """Read OBJ node groups and global/per-component sharpness metadata.

    Native URDF assets allow metadata before the first group and faces that
    reference shared or negative OBJ indices. Faces identify component nodes;
    their triangles are not the DSF surface.
    """
    path = Path(path)
    vertices, groups = [], []
    defaults = {"sharpness": 20}
    current = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            key, _, value = line[1:].strip().partition(":")
            if key.strip() == "sharpness":
                (defaults if current is None else current)["sharpness"] = float(value)
            continue
        fields = line.split()
        if fields[0] in {"o", "g"}:
            current = {**defaults, "indices": set()}
            groups.append(current)
        elif fields[0] == "v":
            vertices.append([float(value) for value in fields[1:]])
            if current is not None:
                current["indices"].add(len(vertices) - 1)
        elif fields[0] == "f":
            if current is None:
                current = {**defaults, "indices": set()}
                groups.append(current)
            for value in fields[1:]:
                index = int(value.split("/")[0])
                resolved = index - 1 if index > 0 else len(vertices) + index
                if index == 0 or not 0 <= resolved < len(vertices):
                    raise ValueError(f"Invalid DSF face index in {path}: {value}")
                current["indices"].add(resolved)
    if not groups:
        groups = [{**defaults, "indices": set(range(len(vertices)))}]
    points = np.asarray(vertices, dtype=float)
    components = tuple((points[sorted(group["indices"])], group["sharpness"])
                       for group in groups if group["indices"])
    if not components:
        raise ValueError(f"No DSF components in {path}")
    return components


def object_surface_path(path, mode):
    """Resolve a commissioned OBJ/PLY pair without silently changing modes."""
    path = Path(path)
    if mode == "dsf":
        result = path.with_name(path.stem.removesuffix("_mesh") + ".obj")
    elif mode == "ply":
        result = path if path.suffix.lower() == ".ply" else path.with_name(path.stem + "_mesh.ply")
    else:
        raise ValueError(f"Unknown object geometry mode: {mode}")
    if not result.is_file():
        raise FileNotFoundError(f"Missing {mode.upper()} object surface: {result}")
    return result
