from pathlib import Path

import pytest

from stacking_tabletop.perception.prepare_capture_cli import (
    discover_capture_directories,
    plan_capture_preparation,
)


def test_discovers_a_selected_raw_capture(tmp_path: Path) -> None:
    capture = _raw_capture(tmp_path / "capture")

    assert discover_capture_directories(capture) == (capture.resolve(),)


def test_discovers_immediate_raw_capture_children_in_name_order(
    tmp_path: Path,
) -> None:
    second = _raw_capture(tmp_path / "object_2")
    first = _raw_capture(tmp_path / "object_1")
    (tmp_path / "not_a_capture").mkdir()

    assert discover_capture_directories(tmp_path) == (
        first.resolve(),
        second.resolve(),
    )


def test_batch_output_directory_keeps_capture_names(tmp_path: Path) -> None:
    first = _raw_capture(tmp_path / "object_1")
    second = _raw_capture(tmp_path / "object_2")
    output = tmp_path / "prepared"

    assert plan_capture_preparation(tmp_path, output) == (
        (first.resolve(), (output / "object_1").resolve()),
        (second.resolve(), (output / "object_2").resolve()),
    )


def test_rejects_directory_without_raw_captures(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no raw captures found"):
        discover_capture_directories(tmp_path)


def _raw_capture(path: Path) -> Path:
    (path / "raw_pcd").mkdir(parents=True)
    (path / "camera_frame.json").touch()
    (path / "frames.csv").touch()
    (path / "raw_pcd" / "0.pcd").touch()
    return path
