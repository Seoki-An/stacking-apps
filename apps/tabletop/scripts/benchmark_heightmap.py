"""Compare tabletop NumPy and legacy Open3D height-map rendering."""

import argparse
import time
from pathlib import Path

import numpy as np

from stacking_common.domain import State
from stacking_common.domain.geometry import transform_points
from stacking_tabletop import TabletopApplication, TabletopHeightMapRenderer
from stacking_tabletop.environment import TabletopObjectAssetLoader
from stacking_tabletop.environment.heightmap_reference import (
    render_open3d_height_map,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path)
    parser.add_argument("--object-ids", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("benchmark repeat count must be positive")

    application = TabletopApplication.from_repository(args.repository)
    profile = application.inventory_profile
    model_directory = args.model_directory
    if model_directory is None:
        model_directory = args.repository / profile.model_directory
    initial_poses = {
        object_id: np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        )
        for object_id in args.object_ids
    }
    loader = TabletopObjectAssetLoader(
        model_directory,
        density=profile.density,
        density_overrides=dict(profile.density_overrides),
        target=profile.target.geometry(),
    )
    inventory = loader.load(args.object_ids, initial_poses)
    renderer = TabletopHeightMapRenderer(
        inventory,
        resolution=application.environment_profile.observation.height_map_resolution,
        margin=application.environment_profile.height_map_margin,
    )
    state = State(
        object_ids=list(inventory.object_ids),
        placed_object_indices=list(range(len(inventory))),
        object_poses=initial_poses,
        trajectories=[],
        action_history=[],
    )
    meshes = []
    for stack_object in inventory:
        points, faces = stack_object.mesh_arrays()
        meshes.append(
            (
                transform_points(
                    points, initial_poses[stack_object.object_id]
                ),
                faces,
            )
        )

    renderer.render(state)
    render_open3d_height_map(
        meshes, renderer.resolution, renderer.xlim, renderer.ylim
    )
    numpy_seconds = _median_seconds(
        lambda: renderer.render(state), args.repeats
    )
    open3d_seconds = _median_seconds(
        lambda: render_open3d_height_map(
            meshes, renderer.resolution, renderer.xlim, renderer.ylim
        ),
        args.repeats,
    )
    numpy_map = renderer.render(state)
    open3d_map = render_open3d_height_map(
        meshes, renderer.resolution, renderer.xlim, renderer.ylim
    )
    disagreement = np.count_nonzero((numpy_map > 0.0) != (open3d_map > 0.0))
    print(f"NumPy median: {numpy_seconds * 1e3:.3f} ms")
    print(f"Open3D median: {open3d_seconds * 1e3:.3f} ms")
    print(f"speed ratio (NumPy/Open3D): {numpy_seconds / open3d_seconds:.3f}")
    print(f"maximum absolute error: {np.max(np.abs(numpy_map - open3d_map)):.6g} m")
    print(f"occupancy disagreements: {disagreement}/{numpy_map.size}")


def _median_seconds(function, repeats: int) -> float:
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        values.append(time.perf_counter() - start)
    return float(np.median(values))


if __name__ == "__main__":
    main()
