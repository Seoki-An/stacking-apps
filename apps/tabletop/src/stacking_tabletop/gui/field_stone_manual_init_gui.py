"""Manual per-stone ICP initialization for fixed-Femto field PoseID."""

from __future__ import annotations

import copy

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from stacking_tabletop.perception.assignment import initial_table_transform as make_initial_transform


DEFAULT_TRANSLATION_STEP_M = 0.005
DEFAULT_ROTATION_STEP_DEG = 5.0
COORDINATE_FRAME_SIZE_M = 0.05


def selected_manual_assignments(
    assignments: dict[int, int],
    target_stone_ids: set[int] | None,
) -> list[tuple[int, int]]:
    """Return sorted cluster/stone pairs selected for manual initialization."""
    pairs = sorted((int(cluster), int(stone)) for cluster, stone in assignments.items())
    if target_stone_ids is None:
        return pairs

    targets = {int(stone_id) for stone_id in target_stone_ids}
    assigned = {stone_id for _, stone_id in pairs}
    missing = sorted(targets - assigned)
    if missing:
        raise ValueError(
            "Manual-initial-pose stone IDs are not assigned to clusters: "
            + ", ".join(map(str, missing))
        )
    return [pair for pair in pairs if pair[1] in targets]


def _translation_transform(axis: int, distance: float) -> np.ndarray:
    transform = np.eye(4)
    transform[axis, 3] = float(distance)
    return transform


def _rotation_transform(axis, angle_rad: float, center) -> np.ndarray:
    rotation = Rotation.from_rotvec(
        np.asarray(axis, dtype=np.float64) * float(angle_rad)
    ).as_matrix()
    center = np.asarray(center, dtype=np.float64).reshape(3)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = center - rotation @ center
    return transform


def _pose_summary(transform: np.ndarray) -> str:
    xyz = 1000.0 * transform[:3, 3]
    rpy = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=True)
    return (
        f"xyz=[{xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}] mm, "
        f"rpy=[{rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}] deg"
    )


def run_manual_initial_pose_gui(
    assignments: dict[int, int],
    clusters: list[o3d.geometry.PointCloud],
    models: dict[int, o3d.geometry.PointCloud],
    table_plane,
    *,
    target_stone_ids: set[int] | None = None,
    translation_step_m: float = DEFAULT_TRANSLATION_STEP_M,
    rotation_step_deg: float = DEFAULT_ROTATION_STEP_DEG,
    window_name: str = "Field PoseID - manual initial poses",
) -> dict[int, np.ndarray]:
    """Adjust selected assigned models and return transforms into the scan frame."""
    if not np.isfinite(translation_step_m) or translation_step_m <= 0.0:
        raise ValueError("manual translation step must be positive and finite")
    if not np.isfinite(rotation_step_deg) or rotation_step_deg <= 0.0:
        raise ValueError("manual rotation step must be positive and finite")

    editable = selected_manual_assignments(assignments, target_stone_ids)
    if not editable:
        raise ValueError("No assigned stones were selected for manual initialization")

    plane = (
        None
        if table_plane is None
        else np.asarray(table_plane, dtype=np.float64).reshape(4)
    )
    cluster_views = []
    for cluster in clusters:
        view = copy.deepcopy(cluster)
        view.paint_uniform_color([0.45, 0.45, 0.45])
        cluster_views.append(view)

    state = {
        "selected": 0,
        "translation_step_m": float(translation_step_m),
        "rotation_step_rad": float(np.deg2rad(rotation_step_deg)),
        "cancelled": False,
    }
    transforms = {}
    original_transforms = {}
    model_views = {}
    for cluster_index, stone_id in editable:
        if stone_id not in models:
            raise KeyError(f"No loaded model PCD for assigned stone {stone_id}")
        model = models[stone_id]
        transform = make_initial_transform(
            np.asarray(model.points, dtype=np.float64),
            np.asarray(clusters[cluster_index].points, dtype=np.float64),
            np.eye(3),
            plane,
        )
        transforms[stone_id] = transform
        original_transforms[stone_id] = transform.copy()
        view = copy.deepcopy(model)
        view.transform(transform)
        model_views[stone_id] = view

    print("\nManual field-pose initialization:")
    print("  N/P or 1-9 : select next/previous or indexed stone")
    print("  A/D        : move -/+ planning X")
    print("  S/W        : move -/+ planning Y")
    print("  F/R        : move -/+ planning Z")
    print("  J/L        : rotate -/+ yaw about displayed-model center")
    print("  I/K        : rotate +/- pitch about displayed-model center")
    print("  U/O        : rotate -/+ roll about displayed-model center")
    print("  [/]        : halve/double translation and rotation steps")
    print("  X          : reset selected stone")
    print("  C          : confirm all selected manual seeds")
    print("  Q          : cancel manual initialization")

    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    if not visualizer.create_window(
        window_name=window_name,
        width=1280,
        height=720,
    ):
        raise RuntimeError("Could not open the manual initial-pose viewer")
    for cluster in cluster_views:
        visualizer.add_geometry(cluster)
    for model in model_views.values():
        visualizer.add_geometry(model)
    visualizer.add_geometry(
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=COORDINATE_FRAME_SIZE_M
        )
    )
    render_option = visualizer.get_render_option()
    render_option.background_color = np.asarray([0.12, 0.12, 0.12])
    render_option.point_size = 3.0

    def selected_pair():
        return editable[state["selected"]]

    def print_selected(prefix="selected"):
        cluster_index, stone_id = selected_pair()
        print(
            f"  [{prefix}] {state['selected'] + 1}/{len(editable)}: "
            f"cluster {cluster_index} -> stone {stone_id}, "
            f"{_pose_summary(transforms[stone_id])}"
        )

    def repaint():
        selected_cluster, selected_stone = selected_pair()
        for cluster_index, cluster in enumerate(cluster_views):
            color = (
                [0.10, 0.80, 0.90]
                if cluster_index == selected_cluster
                else [0.40, 0.40, 0.40]
            )
            cluster.paint_uniform_color(color)
            visualizer.update_geometry(cluster)
        for stone_id, model in model_views.items():
            model.paint_uniform_color(
                [1.0, 0.40, 0.05]
                if stone_id == selected_stone
                else [0.20, 0.45, 0.95]
            )
            visualizer.update_geometry(model)
        visualizer.update_renderer()

    def apply_increment(increment):
        _, stone_id = selected_pair()
        model_views[stone_id].transform(increment)
        transforms[stone_id] = increment @ transforms[stone_id]
        visualizer.update_geometry(model_views[stone_id])
        visualizer.update_renderer()
        print_selected("adjusted")
        return False

    def move(axis, sign):
        return lambda _vis: apply_increment(
            _translation_transform(axis, sign * state["translation_step_m"])
        )

    def rotate(axis, sign):
        def callback(_vis):
            _, stone_id = selected_pair()
            increment = _rotation_transform(
                axis,
                sign * state["rotation_step_rad"],
                model_views[stone_id].get_center(),
            )
            return apply_increment(increment)

        return callback

    def select(offset):
        def callback(_vis):
            state["selected"] = (state["selected"] + offset) % len(editable)
            repaint()
            print_selected()
            return False

        return callback

    def select_index(index):
        def callback(_vis):
            if index < len(editable):
                state["selected"] = index
                repaint()
                print_selected()
            return False

        return callback

    def scale_step(factor):
        def callback(_vis):
            state["translation_step_m"] *= factor
            state["rotation_step_rad"] *= factor
            print(
                "  step: "
                f"{1000.0 * state['translation_step_m']:.2f} mm, "
                f"{np.rad2deg(state['rotation_step_rad']):.2f} deg"
            )
            return False

        return callback

    def reset(_vis):
        _, stone_id = selected_pair()
        increment = original_transforms[stone_id] @ np.linalg.inv(
            transforms[stone_id]
        )
        model_views[stone_id].transform(increment)
        transforms[stone_id] = original_transforms[stone_id].copy()
        visualizer.update_geometry(model_views[stone_id])
        visualizer.update_renderer()
        print_selected("reset")
        return False

    def finish(cancelled):
        def callback(active_visualizer):
            state["cancelled"] = cancelled
            active_visualizer.close()
            return False

        return callback

    callbacks = {
        "N": select(1),
        "P": select(-1),
        "A": move(0, -1.0),
        "D": move(0, 1.0),
        "S": move(1, -1.0),
        "W": move(1, 1.0),
        "F": move(2, -1.0),
        "R": move(2, 1.0),
        "J": rotate([0.0, 0.0, 1.0], -1.0),
        "L": rotate([0.0, 0.0, 1.0], 1.0),
        "I": rotate([0.0, 1.0, 0.0], 1.0),
        "K": rotate([0.0, 1.0, 0.0], -1.0),
        "U": rotate([1.0, 0.0, 0.0], -1.0),
        "O": rotate([1.0, 0.0, 0.0], 1.0),
        "[": scale_step(0.5),
        "]": scale_step(2.0),
        "X": reset,
        "C": finish(False),
        "Q": finish(True),
    }
    for key, callback in callbacks.items():
        visualizer.register_key_callback(ord(key), callback)
        if key.isalpha():
            visualizer.register_key_callback(ord(key.lower()), callback)
    for index in range(min(9, len(editable))):
        visualizer.register_key_callback(ord(str(index + 1)), select_index(index))

    repaint()
    print_selected()
    visualizer.run()
    visualizer.destroy_window()
    if state["cancelled"]:
        print("  Manual initial-pose edits cancelled; using automatic initialization.")
        return {}

    print(
        "  Confirmed manual initial poses for stones: "
        + ", ".join(str(stone_id) for _, stone_id in editable)
    )
    return {stone_id: transforms[stone_id].copy() for _, stone_id in editable}
