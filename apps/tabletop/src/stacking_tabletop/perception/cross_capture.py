"""Tabletop operator UI and artifact adapters for cross-capture merging."""

from pathlib import Path

import numpy as np

from stacking_common.pointcloud import (
    CrossCaptureAlignment,
    CrossCaptureInitializer,
    CrossCaptureMerger,
    CrossCaptureMergeResult,
    PointCloudData,
)
from stacking_tabletop.config import CrossCaptureMergeProfile
from stacking_tabletop.visualization import open3d_coordinate_frame

from .reconstruction import load_point_cloud


class InteractiveCrossCaptureInitializer:
    """Keyboard-controlled tabletop initialization before ICP refinement."""

    def __init__(self, profile: CrossCaptureMergeProfile) -> None:
        self.profile = profile

    def initialize(
        self,
        source: PointCloudData,
        target: PointCloudData,
        source_index: int,
    ) -> np.ndarray:
        o3d = _open3d()
        fixed = _cloud(o3d, _voxel_downsample(target, self.profile.preview_voxel_size))
        moving = _cloud(o3d, _voxel_downsample(source, self.profile.preview_voxel_size))
        fixed.paint_uniform_color([0.15, 0.45, 1.0])
        moving.paint_uniform_color([1.0, 0.35, 0.05])
        initial = np.eye(4)
        initial[:3, 3] = np.asarray(fixed.get_center()) - np.asarray(moving.get_center())
        moving.transform(initial)
        state = {
            "transform": initial.copy(),
            "translation_step": self.profile.initial_translation_step,
            "rotation_step": np.deg2rad(self.profile.initial_rotation_step_degrees),
        }
        print(
            f"Cross-capture initialization {source_index}:\n"
            "  blue = fixed accumulated target, orange = moving capture\n"
            "  translate: A/D=x, S/W=y, F/R=z\n"
            "  rotate: J/L=yaw, I/K=pitch, U/O=roll\n"
            "  [ and ] change step size; X resets; close when aligned"
        )
        visualizer = o3d.visualization.VisualizerWithKeyCallback()
        if not visualizer.create_window(
            window_name=f"Align capture {source_index + 1}",
            width=1280,
            height=720,
        ):
            raise RuntimeError("failed to create cross-capture alignment window")
        visualizer.add_geometry(fixed)
        visualizer.add_geometry(moving)
        frame_points = np.vstack((np.asarray(fixed.points), np.asarray(moving.points)))
        visualizer.add_geometry(open3d_coordinate_frame(o3d, frame_points))

        def apply(transform: np.ndarray):
            moving.transform(transform)
            state["transform"] = transform @ state["transform"]
            visualizer.update_geometry(moving)
            visualizer.update_renderer()
            return False

        def translate(axis: int, sign: float):
            def callback(_):
                transform = np.eye(4)
                transform[axis, 3] = sign * state["translation_step"]
                return apply(transform)

            return callback

        def rotate(axis: np.ndarray, sign: float):
            def callback(_):
                center = np.asarray(moving.get_center())
                rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(
                    axis * sign * state["rotation_step"]
                )
                transform = np.eye(4)
                transform[:3, :3] = rotation
                transform[:3, 3] = center - rotation @ center
                return apply(transform)

            return callback

        def scale_step(factor: float):
            def callback(_):
                state["translation_step"] *= factor
                state["rotation_step"] *= factor
                print(
                    "alignment step: "
                    f"{state['translation_step'] * 1000.0:.2f} mm, "
                    f"{np.rad2deg(state['rotation_step']):.2f} deg"
                )
                return False

            return callback

        def reset(_):
            moving.transform(np.linalg.inv(state["transform"]))
            moving.transform(initial)
            state["transform"] = initial.copy()
            visualizer.update_geometry(moving)
            visualizer.update_renderer()
            return False

        callbacks = {
            "A": translate(0, -1.0),
            "D": translate(0, 1.0),
            "S": translate(1, -1.0),
            "W": translate(1, 1.0),
            "F": translate(2, -1.0),
            "R": translate(2, 1.0),
            "J": rotate(np.array([0.0, 0.0, 1.0]), 1.0),
            "L": rotate(np.array([0.0, 0.0, 1.0]), -1.0),
            "I": rotate(np.array([0.0, 1.0, 0.0]), 1.0),
            "K": rotate(np.array([0.0, 1.0, 0.0]), -1.0),
            "U": rotate(np.array([1.0, 0.0, 0.0]), 1.0),
            "O": rotate(np.array([1.0, 0.0, 0.0]), -1.0),
            "[": scale_step(0.5),
            "]": scale_step(2.0),
            "X": reset,
        }
        for key, callback in callbacks.items():
            visualizer.register_key_callback(ord(key), callback)
            if key.isalpha():
                visualizer.register_key_callback(ord(key.lower()), callback)
        visualizer.run()
        visualizer.destroy_window()
        return np.asarray(state["transform"], dtype=np.float64)


# Compatibility name for existing tabletop callers. The implementation is common.
TabletopCrossCaptureMerger = CrossCaptureMerger


def resolve_capture_cloud(path: str | Path) -> Path:
    """Resolve a prepared capture directory or an explicit PCD/PLY file."""
    value = Path(path).expanduser().resolve()
    if value.is_file():
        return value
    if not value.is_dir():
        raise FileNotFoundError(f"capture path does not exist: {value}")
    for relative in (
        "object_merged_tool0.pcd",
        "prepared/object_merged_tool0.pcd",
        "stone_manual_tool0.pcd",
        "output/merged.pcd",
    ):
        candidate = value / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "capture has no prepared object_merged_tool0.pcd or compatible "
        f"legacy object cloud: {value}"
    )


def load_capture_cloud(path: str | Path) -> PointCloudData:
    return load_point_cloud(resolve_capture_cloud(path))


def show_cross_capture_result(result: CrossCaptureMergeResult) -> None:
    o3d = _open3d()
    colors = (
        [0.15, 0.45, 1.0],
        [1.0, 0.35, 0.05],
        [0.2, 0.85, 0.3],
        [0.8, 0.25, 1.0],
    )
    geometries = []
    for index, cloud in enumerate(result.aligned_clouds):
        value = _cloud(o3d, cloud)
        value.paint_uniform_color(colors[index % len(colors)])
        geometries.append(value)
    geometries.append(open3d_coordinate_frame(o3d, result.cloud.points))
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Aligned cross-capture object scans",
    )


def _voxel_downsample(cloud: PointCloudData, voxel_size: float) -> PointCloudData:
    o3d = _open3d()
    value = _cloud(o3d, cloud).voxel_down_sample(voxel_size)
    return PointCloudData(np.asarray(value.points))


def _cloud(o3d, cloud: PointCloudData):
    value = o3d.geometry.PointCloud()
    value.points = o3d.utility.Vector3dVector(np.array(cloud.points, copy=True))
    return value


def _open3d():
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D is required for cross-capture merging") from error
    return o3d


__all__ = [
    "CrossCaptureAlignment",
    "CrossCaptureInitializer",
    "CrossCaptureMergeResult",
    "InteractiveCrossCaptureInitializer",
    "TabletopCrossCaptureMerger",
    "load_capture_cloud",
    "resolve_capture_cloud",
    "show_cross_capture_result",
]
