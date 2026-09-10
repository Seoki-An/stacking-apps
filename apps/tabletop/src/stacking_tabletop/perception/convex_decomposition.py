"""Tabletop operator adapters for shared convex decomposition."""

from pathlib import Path

import numpy as np

from stacking_common.pointcloud import PointCloudData
from stacking_common.reconstruction import (
    ConvexDecomposer,
    ConvexPart,
    DsfPartitionOptions,
    build_dsf_point_sets,
    split_points_by_convex_parts,
    split_points_by_seeds,
)
from stacking_tabletop.visualization import add_scaled_pyvista_axes


TabletopConvexDecomposer = ConvexDecomposer


def select_maximum_components(
    cloud: PointCloudData,
    initial: int,
) -> int:
    """Let the operator select the CoACD component cap for this object."""
    if initial < 1:
        raise ValueError("initial convex component count must be positive")
    pv = _pyvista()
    selected = {"value": int(initial)}
    plotter = pv.Plotter(title="Select CoACD maximum components")
    add_scaled_pyvista_axes(plotter, cloud.points)
    _add_cloud(pv, plotter, cloud)

    def text() -> str:
        return (
            f"Maximum CoACD components: {selected['value']}\n\n"
            "Insert / + / Up: increase\n"
            "Delete / - / Down: decrease\n"
            "Enter / Space: continue"
        )

    actor = plotter.add_text(
        text(),
        position="upper_left",
        font_size=10,
        name="component_count",
    )

    def update(delta: int) -> None:
        selected["value"] = max(1, selected["value"] + delta)
        actor.SetInput(text())
        plotter.render()

    def key_press(interactor, _event) -> None:
        key = interactor.GetKeySym()
        if key in ("Insert", "plus", "equal", "KP_Add", "Up", "Right"):
            update(1)
        elif key in ("Delete", "minus", "KP_Subtract", "Down", "Left"):
            update(-1)
        elif key in ("Return", "KP_Enter", "space", "q", "Escape"):
            interactor.TerminateApp()

    plotter.iren.interactor.AddObserver("KeyPressEvent", key_press)
    plotter.show()
    plotter.close()
    return selected["value"]


def load_or_select_split_seeds(
    cloud: PointCloudData,
    *,
    path: str | Path | None = None,
    force_selection: bool = False,
    left_clicking: bool = False,
    marker_radius: float = 0.003,
) -> np.ndarray:
    destination = None if path is None else Path(path).expanduser().resolve()
    if destination is not None and destination.is_file() and not force_selection:
        seeds = np.asarray(np.load(destination), dtype=np.float64)
        if seeds.ndim != 2 or seeds.shape[1] != 3 or not len(seeds):
            raise ValueError(f"invalid manual split seed file: {destination}")
        return seeds
    seeds = _select_split_seeds(
        cloud,
        left_clicking=left_clicking,
        marker_radius=marker_radius,
    )
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, seeds)
    return seeds


def show_convex_decomposition(
    cloud: PointCloudData,
    parts: tuple[ConvexPart, ...],
) -> None:
    pv = _pyvista()
    plotter = pv.Plotter(title="CoACD convex decomposition")
    add_scaled_pyvista_axes(plotter, cloud.points)
    _add_cloud(pv, plotter, cloud)
    colors = ("red", "green", "blue", "yellow", "cyan", "magenta")
    for index, part in enumerate(parts):
        plotter.add_mesh(
            _mesh(pv, part.vertices, part.triangles),
            color=colors[index % len(colors)],
            opacity=0.3,
            show_edges=True,
            name=f"convex_{index}",
        )
    plotter.show()
    plotter.close()


def show_point_partitions(point_sets: tuple[np.ndarray, ...]) -> None:
    pv = _pyvista()
    plotter = pv.Plotter(title="Point-cloud DSF partitions")
    all_points = np.vstack(point_sets)
    add_scaled_pyvista_axes(plotter, all_points)
    colors = ("red", "green", "blue", "yellow", "cyan", "magenta")
    for index, points in enumerate(point_sets):
        plotter.add_points(
            pv.PolyData(points),
            color=colors[index % len(colors)],
            point_size=5,
            name=f"partition_{index}",
        )
    plotter.show()
    plotter.close()


def _select_split_seeds(
    cloud: PointCloudData,
    *,
    left_clicking: bool,
    marker_radius: float,
) -> np.ndarray:
    if not np.isfinite(marker_radius) or marker_radius <= 0.0:
        raise ValueError("manual split marker radius must be positive")
    pv = _pyvista()
    picked = []
    plotter = pv.Plotter(title="Manual DSF partitions")
    add_scaled_pyvista_axes(plotter, cloud.points)
    _add_cloud(pv, plotter, cloud)
    plotter.add_text(
        "Pick one seed near each desired DSF component.\n"
        "Press P to pick and U to undo. Close when complete.",
        position="upper_left",
        font_size=10,
    )

    def pick(point) -> None:
        if point is None:
            return
        value = np.asarray(point, dtype=np.float64)
        picked.append(value)
        plotter.add_mesh(
            pv.Sphere(radius=marker_radius, center=value),
            color="red",
            opacity=0.45,
            name=f"seed_{len(picked)}",
            pickable=False,
        )

    def undo() -> None:
        if picked:
            plotter.remove_actor(f"seed_{len(picked)}")
            picked.pop()

    plotter.enable_point_picking(
        callback=pick,
        left_clicking=left_clicking,
        show_point=True,
        point_size=12,
        color="red",
    )
    plotter.add_key_event("u", undo)
    plotter.add_key_event("U", undo)
    plotter.show()
    plotter.close()
    if not picked:
        raise ValueError("no manual DSF split seeds were selected")
    return np.asarray(picked, dtype=np.float64)


def _add_cloud(pv, plotter, cloud: PointCloudData) -> None:
    points = np.asarray(cloud.points)
    value = pv.PolyData(points)
    value["z"] = points[:, 2]
    plotter.add_points(value, scalars="z", point_size=4, name="object")


def _mesh(pv, vertices: np.ndarray, triangles: np.ndarray):
    faces = np.column_stack(
        (np.full(len(triangles), 3, dtype=np.int64), triangles)
    )
    return pv.PolyData(vertices, faces)


def _pyvista():
    try:
        import pyvista as pv
    except ImportError as error:
        raise RuntimeError(
            "PyVista is required for decomposition visualization and manual splits"
        ) from error
    return pv

