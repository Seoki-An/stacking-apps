"""Geometry-scale-aware coordinate frames for tabletop visualizations."""

import numpy as np


_REFERENCE_FRAME_SIZE = 0.02
_REFERENCE_LABEL_FONT_SIZE = 10
_MINIMUM_LABEL_FONT_SIZE = 6
_MAXIMUM_LABEL_FONT_SIZE = 18


def coordinate_frame_size(
    points: np.ndarray,
    *,
    fraction: float = 0.2,
    minimum: float = 1e-4,
) -> float:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("coordinate frame scaling requires XYZ points")
    if not np.all(np.isfinite(values)):
        raise ValueError("coordinate frame scaling points must be finite")
    if not np.isfinite(fraction) or fraction <= 0.0:
        raise ValueError("coordinate frame fraction must be positive")
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("coordinate frame minimum must be positive")
    diagonal = float(np.linalg.norm(np.ptp(values, axis=0)))
    return max(minimum, fraction * diagonal)


def coordinate_label_font_size(points: np.ndarray) -> int:
    """Scale axis-caption text with the metric coordinate-frame size."""
    scaled = round(
        _REFERENCE_LABEL_FONT_SIZE
        * coordinate_frame_size(points)
        / _REFERENCE_FRAME_SIZE
    )
    return max(
        _MINIMUM_LABEL_FONT_SIZE,
        min(_MAXIMUM_LABEL_FONT_SIZE, scaled),
    )


def add_scaled_pyvista_axes(plotter, points: np.ndarray):
    size = coordinate_frame_size(points)
    actor = plotter.add_axes_at_origin(line_width=2.0)
    actor.SetTotalLength(size, size, size)
    font_size = coordinate_label_font_size(points)
    captions = (
        actor.GetXAxisCaptionActor2D(),
        actor.GetYAxisCaptionActor2D(),
        actor.GetZAxisCaptionActor2D(),
    )
    for caption in captions:
        caption.GetTextActor().SetTextScaleModeToNone()
        caption.GetCaptionTextProperty().SetFontSize(font_size)
    return actor


def open3d_coordinate_frame(o3d, points: np.ndarray):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=coordinate_frame_size(points)
    )
