"""Pure array score policy migrated from stacking-planner support_map.py.

Only the dependencies of scored sampling are retained; platform inputs arrive
through the canonical inventory adapter in scored.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Iterable

import numpy as np

from scipy.ndimage import convolve

from scipy.spatial import ConvexHull, QhullError

from scipy.spatial.transform import Rotation

@dataclass(frozen=True)
class StateMapBundle:
    surface_height: np.ndarray
    smoothed_height: np.ndarray
    valid: np.ndarray

@dataclass(frozen=True)
class BottomHeightKernel:
    y_offsets: np.ndarray
    x_offsets: np.ndarray
    bottom_height: np.ndarray

    def __len__(self) -> int:
        return int(len(self.bottom_height))

@dataclass(frozen=True)
class KernelSupportMaps:
    support: np.ndarray
    contact_height: np.ndarray
    placement_height: np.ndarray

@dataclass(frozen=True)
class ConvexVolume:
    """World-space half-spaces ``A @ xyz <= b`` for one stone component."""

    A: np.ndarray
    b: np.ndarray

def transformed_stone_volumes(stone, pose: np.ndarray) -> tuple[ConvexVolume, ...]:
    """Transform the stone's existing convex DSF components into world space."""
    pose = np.asarray(pose, dtype=float)
    rotation = Rotation.from_quat(pose[3:7]).as_matrix()
    translation = pose[:3]
    volumes = []
    for geometry in stone.geometries:
        local_A, local_b = geometry.support_planes
        world_A = np.asarray(local_A, dtype=float) @ rotation.T
        world_b = np.asarray(local_b, dtype=float) + world_A @ translation
        volumes.append(ConvexVolume(A=world_A, b=world_b))
    return tuple(volumes)

def volume_slice_mask(
    volumes: Iterable[ConvexVolume],
    xx: np.ndarray,
    yy: np.ndarray,
    z_lower: float,
    z_upper: float,
) -> np.ndarray:
    """Rasterize where convex volumes intersect a horizontal z slab."""
    xx = np.asarray(xx, dtype=float)
    yy = np.asarray(yy, dtype=float)
    if xx.shape != yy.shape:
        raise ValueError("xx and yy must have the same shape")

    slab_lower = min(float(z_lower), float(z_upper))
    slab_upper = max(float(z_lower), float(z_upper))
    occupied = np.zeros(xx.shape, dtype=bool)
    for volume in volumes:
        occupied |= _convex_volume_slice_mask(
            volume,
            xx,
            yy,
            slab_lower,
            slab_upper,
        )
    return occupied

def _convex_volume_slice_mask(
    volume: ConvexVolume,
    xx: np.ndarray,
    yy: np.ndarray,
    z_lower: float,
    z_upper: float,
) -> np.ndarray:
    A = np.asarray(volume.A, dtype=float)
    b = np.asarray(volume.b, dtype=float).reshape(-1)
    if A.ndim != 2 or A.shape[1] != 3 or A.shape[0] != b.size:
        raise ValueError("convex volume must contain compatible A and b arrays")

    rhs = (
        b[:, None, None]
        - A[:, 0, None, None] * xx[None, ...]
        - A[:, 1, None, None] * yy[None, ...]
    )
    z_coeff = A[:, 2]
    lower_faces = z_coeff < -1e-10
    upper_faces = z_coeff > 1e-10
    vertical_faces = ~(lower_faces | upper_faces)

    column_lower = np.full(xx.shape, -np.inf, dtype=float)
    column_upper = np.full(xx.shape, np.inf, dtype=float)
    if np.any(lower_faces):
        column_lower = np.max(
            rhs[lower_faces] / z_coeff[lower_faces, None, None],
            axis=0,
        )
    if np.any(upper_faces):
        column_upper = np.min(
            rhs[upper_faces] / z_coeff[upper_faces, None, None],
            axis=0,
        )

    inside = column_lower <= column_upper + 1e-8
    if np.any(vertical_faces):
        inside &= np.all(rhs[vertical_faces] >= -1e-8, axis=0)
    return (
        inside
        & (column_upper >= float(z_lower) - 1e-8)
        & (column_lower <= float(z_upper) + 1e-8)
    )

def build_state_map_bundle(
    scene_height_map: np.ndarray,
    shape: tuple[int, int],
    ground_z: float,
    valid: np.ndarray,
    smooth_kernel_size: int = 5,
) -> StateMapBundle:
    surface = resize_height_map_max(scene_height_map, shape)
    surface = np.maximum(surface, float(ground_z))
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != surface.shape:
        raise ValueError("state-map valid mask must match the score-grid shape")
    smoothed = masked_mean_filter(surface, valid, smooth_kernel_size)
    return StateMapBundle(
        surface_height=surface,
        smoothed_height=smoothed,
        valid=valid.copy(),
    )

def resize_height_map_max(
    height_map: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Resize a top-surface map while preserving narrow high supports."""
    source = np.asarray(height_map, dtype=float)
    if source.ndim != 2 or source.size == 0:
        return np.zeros(shape, dtype=float)
    if source.shape == tuple(shape):
        return source.copy()

    dst_h, dst_w = (int(value) for value in shape)
    src_h, src_w = source.shape
    y_edges = np.linspace(0.0, float(src_h), dst_h + 1)
    x_edges = np.linspace(0.0, float(src_w), dst_w + 1)
    # Block maxima over finite values: maxima of row slabs, then column slabs.
    finite = np.where(np.isfinite(source), source, -np.inf)
    rows = np.empty((dst_h, src_w))
    for y_idx in range(dst_h):
        y0 = min(int(np.floor(y_edges[y_idx])), src_h - 1)
        y1 = max(min(int(np.ceil(y_edges[y_idx + 1])), src_h), y0 + 1)
        rows[y_idx] = finite[y0:y1].max(axis=0)
    resized = np.empty((dst_h, dst_w), dtype=float)
    for x_idx in range(dst_w):
        x0 = min(int(np.floor(x_edges[x_idx])), src_w - 1)
        x1 = max(min(int(np.ceil(x_edges[x_idx + 1])), src_w), x0 + 1)
        resized[:, x_idx] = rows[:, x0:x1].max(axis=1)
    resized[resized == -np.inf] = 0.0
    return resized

def masked_mean_filter(
    values: np.ndarray,
    valid: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """Smooth a grid without allowing masked neighbors into the mean."""
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("values and valid must be same-shaped 2D grids")
    kernel_size = max(int(kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    included = valid & np.isfinite(values)
    kernel = np.ones((kernel_size, kernel_size), dtype=float)
    sums = convolve(
        np.where(included, values, 0.0),
        kernel,
        mode="constant",
        cval=0.0,
    )
    counts = convolve(
        included.astype(float),
        kernel,
        mode="constant",
        cval=0.0,
    )
    smoothed = values.copy()
    np.divide(sums, counts, out=smoothed, where=counts > 0.0)
    return smoothed

def bottom_height_kernel(
    vertices: np.ndarray,
    quaternion: np.ndarray,
    dx: float,
    dy: float,
) -> BottomHeightKernel:
    """Rasterize the lower surface of an oriented convex stone hull."""
    vertices = np.asarray(vertices, dtype=float)
    rotation = Rotation.from_quat(np.asarray(quaternion, dtype=float)).as_matrix()
    oriented = vertices @ rotation.T
    try:
        hull = ConvexHull(oriented)
    except QhullError:
        return rectangular_bottom_kernel(
            np.ptp(oriented[:, :2], axis=0),
            dx,
            dy,
            bottom_height=float(np.min(oriented[:, 2])),
        )

    dx = max(abs(float(dx)), 1e-6)
    dy = max(abs(float(dy)), 1e-6)
    x_offsets = np.arange(
        int(np.floor(np.min(oriented[:, 0]) / dx)),
        int(np.ceil(np.max(oriented[:, 0]) / dx)) + 1,
        dtype=int,
    )
    y_offsets = np.arange(
        int(np.floor(np.min(oriented[:, 1]) / dy)),
        int(np.ceil(np.max(oriented[:, 1]) / dy)) + 1,
        dtype=int,
    )
    xx, yy = np.meshgrid(x_offsets * dx, y_offsets * dy, indexing="xy")
    x_flat = xx.reshape(-1)
    y_flat = yy.reshape(-1)

    equations = np.asarray(hull.equations, dtype=float)
    horizontal = (
        equations[:, 0, None] * x_flat[None, :]
        + equations[:, 1, None] * y_flat[None, :]
        + equations[:, 3, None]
    )
    z_coeff = equations[:, 2]
    lower_faces = z_coeff < -1e-9
    upper_faces = z_coeff > 1e-9
    vertical_faces = ~(lower_faces | upper_faces)

    lower = np.full(len(x_flat), -np.inf, dtype=float)
    upper = np.full(len(x_flat), np.inf, dtype=float)
    if np.any(lower_faces):
        lower = np.max(
            -horizontal[lower_faces] / z_coeff[lower_faces, None],
            axis=0,
        )
    if np.any(upper_faces):
        upper = np.min(
            -horizontal[upper_faces] / z_coeff[upper_faces, None],
            axis=0,
        )
    inside = lower <= upper + 1e-8
    if np.any(vertical_faces):
        inside &= np.all(horizontal[vertical_faces] <= 1e-8, axis=0)

    if not np.any(inside):
        nearest = int(np.argmin(x_flat**2 + y_flat**2))
        inside[nearest] = True
        lower[nearest] = float(np.min(oriented[:, 2]))

    grid_y, grid_x = np.meshgrid(y_offsets, x_offsets, indexing="ij")
    return BottomHeightKernel(
        y_offsets=grid_y.reshape(-1)[inside].astype(int),
        x_offsets=grid_x.reshape(-1)[inside].astype(int),
        bottom_height=lower[inside].astype(float),
    )

def rectangular_bottom_kernel(
    xy_extent: np.ndarray,
    dx: float,
    dy: float,
    bottom_height: float = 0.0,
) -> BottomHeightKernel:
    extent = np.maximum(np.asarray(xy_extent, dtype=float), [abs(dx), abs(dy)])
    half_x = max(int(np.floor(0.5 * extent[0] / max(abs(dx), 1e-6))), 0)
    half_y = max(int(np.floor(0.5 * extent[1] / max(abs(dy), 1e-6))), 0)
    x_offsets = np.arange(-half_x, half_x + 1, dtype=int)
    y_offsets = np.arange(-half_y, half_y + 1, dtype=int)
    grid_y, grid_x = np.meshgrid(y_offsets, x_offsets, indexing="ij")
    return BottomHeightKernel(
        y_offsets=grid_y.reshape(-1),
        x_offsets=grid_x.reshape(-1),
        bottom_height=np.full(grid_x.size, float(bottom_height), dtype=float),
    )

def support_likelihood_maps_by_kernel(
    surface_height: np.ndarray,
    kernels: Iterable[BottomHeightKernel],
    contact_tolerance: float,
) -> KernelSupportMaps:
    """Return support maps without averaging away the kernel orientation."""
    surface = np.asarray(surface_height, dtype=float)
    tolerance = max(float(contact_tolerance), 0.0)
    support_maps = []
    contact_height_maps = []
    placement_height_maps = []

    for kernel in kernels:
        if len(kernel) == 0:
            empty = np.full(surface.shape, np.nan, dtype=float)
            support_maps.append(empty.copy())
            contact_height_maps.append(empty.copy())
            placement_height_maps.append(empty)
            continue
        samples = _shifted_samples(surface, kernel.y_offsets, kernel.x_offsets, -np.inf)
        covered = np.all(np.isfinite(samples), axis=0)
        clearance = samples - kernel.bottom_height[:, None, None]
        placement = np.max(clearance, axis=0)
        contacts = covered[None, ...] & (
            clearance >= placement[None, ...] - tolerance
        )
        contact_count = np.sum(contacts, axis=0)
        support = contact_count.astype(float) / float(len(kernel))
        contact_height = np.divide(
            np.sum(np.where(contacts, samples, 0.0), axis=0),
            contact_count,
            out=np.full(surface.shape, np.nan, dtype=float),
            where=contact_count > 0,
        )
        support[~covered] = np.nan
        placement[~covered] = np.nan

        support_maps.append(support)
        contact_height_maps.append(contact_height)
        placement_height_maps.append(placement)

    empty_shape = (0, *surface.shape)
    return KernelSupportMaps(
        support=(
            np.stack(support_maps)
            if support_maps
            else np.empty(empty_shape, dtype=float)
        ),
        contact_height=(
            np.stack(contact_height_maps)
            if contact_height_maps
            else np.empty(empty_shape, dtype=float)
        ),
        placement_height=(
            np.stack(placement_height_maps)
            if placement_height_maps
            else np.empty(empty_shape, dtype=float)
        ),
    )

def _shifted_samples(
    values: np.ndarray,
    y_offsets: np.ndarray,
    x_offsets: np.ndarray,
    fill: float,
) -> np.ndarray:
    """Return output[k, y, x] = values[y + dy_k, x + dx_k], or fill outside the grid."""
    values = np.asarray(values, dtype=float)
    height, width = values.shape
    y_offsets = np.asarray(y_offsets, dtype=int)
    x_offsets = np.asarray(x_offsets, dtype=int)
    pad = int(max(np.max(np.abs(y_offsets)), np.max(np.abs(x_offsets))))
    padded = np.pad(values, pad, mode="constant", constant_values=fill)
    rows = np.arange(height)[None, :, None] + (y_offsets + pad)[:, None, None]
    columns = np.arange(width)[None, None, :] + (x_offsets + pad)[:, None, None]
    return padded[rows, columns]
