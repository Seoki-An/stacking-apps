import numpy as np
import pytest

from stacking_tabletop.visualization import (
    add_scaled_pyvista_axes,
    coordinate_frame_size,
    coordinate_label_font_size,
)


def test_coordinate_frame_tracks_point_cloud_diagonal() -> None:
    small = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    large = 5.0 * small

    assert coordinate_frame_size(small) == pytest.approx(0.02)
    assert coordinate_frame_size(large) == pytest.approx(0.10)


def test_coordinate_labels_track_frame_size_with_readable_limits() -> None:
    tiny = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
    tabletop = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    large = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    assert coordinate_label_font_size(tiny) == 6
    assert coordinate_label_font_size(tabletop) == 10
    assert coordinate_label_font_size(large) == 18


def test_pyvista_axes_apply_scaled_label_text() -> None:
    plotter = _Plotter()
    points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

    actor = add_scaled_pyvista_axes(plotter, points)

    assert actor.total_length == pytest.approx((0.02, 0.02, 0.02))
    assert all(caption.text_actor.scale_mode_none for caption in actor.captions)
    assert all(caption.text_property.font_size == 10 for caption in actor.captions)


class _TextActor:
    def __init__(self) -> None:
        self.scale_mode_none = False

    def SetTextScaleModeToNone(self) -> None:
        self.scale_mode_none = True


class _TextProperty:
    def __init__(self) -> None:
        self.font_size = None

    def SetFontSize(self, value: int) -> None:
        self.font_size = value


class _Caption:
    def __init__(self) -> None:
        self.text_actor = _TextActor()
        self.text_property = _TextProperty()

    def GetTextActor(self):
        return self.text_actor

    def GetCaptionTextProperty(self):
        return self.text_property


class _AxesActor:
    def __init__(self) -> None:
        self.total_length = None
        self.captions = (_Caption(), _Caption(), _Caption())

    def SetTotalLength(self, x: float, y: float, z: float) -> None:
        self.total_length = (x, y, z)

    def GetXAxisCaptionActor2D(self):
        return self.captions[0]

    def GetYAxisCaptionActor2D(self):
        return self.captions[1]

    def GetZAxisCaptionActor2D(self):
        return self.captions[2]


class _Plotter:
    def add_axes_at_origin(self, *, line_width: float):
        assert line_width == 2.0
        return _AxesActor()
