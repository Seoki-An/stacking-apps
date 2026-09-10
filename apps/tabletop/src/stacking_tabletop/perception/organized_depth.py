"""Temporal filtering and back-projection of recorded Femto depth bursts."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DepthFilterConfig:
    minimum_temporal_support: int = 5
    temporal_tolerance: float = 0.005
    enable_boundary_filter: bool = False
    discontinuity_threshold: float = 0.015
    discontinuity_radius_pixels: int = 2
    boundary_dilation_pixels: int = 2

    def __post_init__(self) -> None:
        if self.minimum_temporal_support <= 0:
            raise ValueError("minimum temporal support must be positive")
        metric = (self.temporal_tolerance, self.discontinuity_threshold)
        if not np.all(np.isfinite(metric)) or any(value <= 0 for value in metric):
            raise ValueError("depth filter metric values must be positive")
        if self.discontinuity_radius_pixels <= 0:
            raise ValueError("depth discontinuity radius must be positive")
        if self.boundary_dilation_pixels < 0:
            raise ValueError("boundary dilation radius must be non-negative")


@dataclass(frozen=True)
class DepthFilterStats:
    total_pixels: int
    median_valid_pixels: int
    temporal_supported_pixels: int
    temporal_consistent_pixels: int
    boundary_rejected_pixels: int
    kept_pixels: int


def filter_depth_burst(
    depth_frames: tuple[np.ndarray, ...],
    depth_unit: float,
    config: DepthFilterConfig,
) -> tuple[np.ndarray, np.ndarray, DepthFilterStats]:
    """Return median axial depth and the temporally supported pixel mask."""

    if len(depth_frames) < config.minimum_temporal_support:
        raise ValueError(
            f"need at least {config.minimum_temporal_support} depth frames; "
            f"found {len(depth_frames)}"
        )
    if not np.isfinite(depth_unit) or depth_unit <= 0.0:
        raise ValueError("organized depth unit must be positive")
    shapes = {np.asarray(frame).shape for frame in depth_frames}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError("organized depth frames must share one 2-D shape")

    depth = np.stack(depth_frames).astype(np.float32) * np.float32(depth_unit)
    valid = np.isfinite(depth) & (depth > 0.0)
    support = np.count_nonzero(valid, axis=0)
    sorted_depth = depth.copy()
    sorted_depth[~valid] = np.inf
    sorted_depth.sort(axis=0)
    lower = np.maximum(support - 1, 0) // 2
    upper = support // 2
    median = np.float32(0.5) * (
        np.take_along_axis(sorted_depth, lower[None, ...], axis=0)[0]
        + np.take_along_axis(sorted_depth, upper[None, ...], axis=0)[0]
    )
    median[support == 0] = np.nan
    consistent = valid & (
        np.abs(depth - median[np.newaxis, ...]) <= config.temporal_tolerance
    )
    temporal_supported = support >= config.minimum_temporal_support
    temporal_consistent = (
        temporal_supported
        & (np.count_nonzero(consistent, axis=0) >= config.minimum_temporal_support)
        & np.isfinite(median)
    )
    boundary = (
        _dilate(
            _depth_boundary_mask(
                median,
                temporal_consistent,
                config.discontinuity_threshold,
                config.discontinuity_radius_pixels,
            ),
            config.boundary_dilation_pixels,
        )
        if config.enable_boundary_filter
        else np.zeros_like(temporal_consistent)
    )
    keep = temporal_consistent & ~boundary
    return median, keep, DepthFilterStats(
        total_pixels=median.size,
        median_valid_pixels=int(np.count_nonzero(np.isfinite(median))),
        temporal_supported_pixels=int(np.count_nonzero(temporal_supported)),
        temporal_consistent_pixels=int(np.count_nonzero(temporal_consistent)),
        boundary_rejected_pixels=int(np.count_nonzero(temporal_consistent & boundary)),
        kept_pixels=int(np.count_nonzero(keep)),
    )


def camera_rays(camera_info: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return distortion-corrected x/z and y/z for every depth pixel."""

    width = int(camera_info.get("width", 0))
    height = int(camera_info.get("height", 0))
    matrix = np.asarray(camera_info.get("k", []), dtype=np.float64)
    if width <= 0 or height <= 0 or matrix.size != 9:
        raise ValueError("depth CameraInfo has invalid dimensions or K matrix")
    matrix = matrix.reshape(3, 3)
    fx, fy = matrix[0, 0], matrix[1, 1]
    cx, cy = matrix[0, 2], matrix[1, 2]
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("depth CameraInfo focal lengths must be positive")
    model = str(camera_info.get("distortion_model", "")).strip()
    if model not in ("", "plumb_bob", "rational_polynomial"):
        raise ValueError(f"unsupported depth distortion model {model!r}")
    columns, rows = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    distorted_x = (columns - cx) / fx
    distorted_y = (rows - cy) / fy
    ray_x, ray_y = _undistort_normalized(
        distorted_x,
        distorted_y,
        list(camera_info.get("d", [])),
    )
    return ray_x.astype(np.float32), ray_y.astype(np.float32)


def depth_to_points(
    depth: np.ndarray,
    keep: np.ndarray,
    ray_x: np.ndarray,
    ray_y: np.ndarray,
) -> np.ndarray:
    if not (depth.shape == keep.shape == ray_x.shape == ray_y.shape):
        raise ValueError("depth, mask, and camera-ray shapes must match")
    selected = depth[keep]
    return np.column_stack(
        (ray_x[keep] * selected, ray_y[keep] * selected, selected)
    )


class OrganizedDepthDataset:
    """Validated access to organized bursts in a raw capture directory."""

    def __init__(self, capture_directory: str | Path, expected_frame: str, depth_mode: str):
        self.root = Path(capture_directory).expanduser().resolve()
        manifest_path = self.root / "organized_depth_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing organized-depth manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not self.manifest.get("values_are_raw", False):
            raise ValueError("organized depth must contain raw values")
        self.depth_unit = float(self.manifest.get("depth_unit_m", 0.0))
        recorded_mode = str(self.manifest.get("depth_mode", "")).strip()
        if depth_mode and recorded_mode != depth_mode:
            raise ValueError("organized-depth mode differs from camera metadata")
        info_name = str(self.manifest.get("camera_info", "")).strip()
        info_path = self.root / info_name
        if not info_name or not info_path.is_file():
            raise FileNotFoundError(f"missing organized-depth CameraInfo: {info_path}")
        self.camera_info = json.loads(info_path.read_text(encoding="utf-8"))
        if str(self.camera_info.get("frame_id", "")).strip() != expected_frame:
            raise ValueError("organized-depth frame differs from camera metadata")
        self.ray_x, self.ray_y = camera_rays(self.camera_info)
        self.views = {}
        for view in self.manifest.get("views", []):
            filename = Path(str(view.get("representative_pcd", ""))).name
            if not filename or filename in self.views:
                raise ValueError("organized-depth view filenames must be unique")
            self.views[filename] = view

    def points_for_view(
        self,
        pcd_filename: str,
        config: DepthFilterConfig,
    ) -> tuple[np.ndarray, DepthFilterStats]:
        try:
            view = self.views[Path(pcd_filename).name]
        except KeyError as error:
            raise ValueError(f"no organized-depth view matches {pcd_filename!r}") from error
        frames = []
        for record in view.get("depth_frames", []):
            path = self.root / str(record.get("filename", ""))
            if not path.is_file():
                raise FileNotFoundError(f"missing organized depth frame: {path}")
            frame = np.load(path, allow_pickle=False)
            if frame.shape != self.ray_x.shape:
                raise ValueError(f"organized depth frame has invalid shape: {path}")
            frames.append(frame)
        median, keep, stats = filter_depth_burst(
            tuple(frames), self.depth_unit, config
        )
        return depth_to_points(median, keep, self.ray_x, self.ray_y), stats

    def report(self, config: DepthFilterConfig) -> dict:
        return {"manifest": str(self.root / "organized_depth_manifest.json"), **asdict(config)}


def _overlap_slices(height: int, width: int, dy: int, dx: int):
    first = (
        slice(max(0, -dy), min(height, height - dy)),
        slice(max(0, -dx), min(width, width - dx)),
    )
    second = (
        slice(max(0, dy), min(height, height + dy)),
        slice(max(0, dx), min(width, width + dx)),
    )
    return first, second


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    result = mask.copy()
    height, width = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            source, target = _overlap_slices(height, width, dy, dx)
            result[target] |= mask[source]
    return result


def _depth_boundary_mask(
    depth: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    radius: int,
) -> np.ndarray:
    height, width = depth.shape
    boundary = np.zeros_like(valid)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            first, second = _overlap_slices(height, width, dy, dx)
            first_valid, second_valid = valid[first], valid[second]
            rejected = (first_valid ^ second_valid) | (
                first_valid & second_valid & (np.abs(depth[first] - depth[second]) > threshold)
            )
            boundary[first] |= rejected & first_valid
            boundary[second] |= rejected & second_valid
    return boundary


def _undistort_normalized(
    distorted_x: np.ndarray,
    distorted_y: np.ndarray,
    coefficients: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(8, dtype=np.float64)
    supplied = np.asarray(coefficients, dtype=np.float64)
    if supplied.size not in (0, 4, 5, 8):
        raise ValueError("depth distortion must have 0, 4, 5, or 8 coefficients")
    values[: supplied.size] = supplied
    k1, k2, p1, p2, k3, k4, k5, k6 = values
    x, y = distorted_x.copy(), distorted_y.copy()
    for _ in range(20):
        r2 = x * x + y * y
        r4, r6 = r2 * r2, r2 * r2 * r2
        denominator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        if np.any(np.abs(denominator) < 1e-12):
            raise ValueError("camera distortion inversion became singular")
        inverse_radial = (1.0 + k4 * r2 + k5 * r4 + k6 * r6) / denominator
        delta_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        delta_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (distorted_x - delta_x) * inverse_radial
        y = (distorted_y - delta_y) * inverse_radial
    return x, y
