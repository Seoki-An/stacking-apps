"""Headless multi-view tabletop object reconstruction command."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from stacking_common.pointcloud import (
    MultiViewVoxelMerge,
    PointCloudData,
    VoxelMergeConfig,
)
from stacking_tabletop.config import load_perception_profile

from .pose_graph import TabletopPoseGraphRefiner
from .extraction import InteractiveObjectSelector
from .in_hand import TabletopRawCapturePreprocessor
from .reconstruction import (
    TabletopDsfExporter,
    TabletopMeshReconstructor,
    estimate_hull_inertial,
    load_point_cloud,
    write_point_cloud,
    write_triangle_mesh,
)
from .session import (
    ReconstructionSession,
    ReconstructionSessionStore,
    ReconstructionViewRecord,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refine, merge, mesh, and export tabletop object views."
    )
    parser.add_argument("repository", type=Path)
    parser.add_argument(
        "session_directory",
        type=Path,
        nargs="?",
        help=(
            "Session directory. New sessions default below "
            "apps/tabletop/data/reconstruction."
        ),
    )
    parser.add_argument("--object-id", type=int)
    parser.add_argument(
        "--session-id",
        help="Session directory name used with the default data location.",
    )
    view_source = parser.add_mutually_exclusive_group()
    view_source.add_argument(
        "--view",
        type=Path,
        action="append",
        default=[],
        help="Input PCD/PLY view; repeat in capture order.",
    )
    view_source.add_argument(
        "--capture-directory",
        type=Path,
        help=(
            "Capture directory containing prepared manual_stone_views_tool0/*.pcd."
        ),
    )
    view_source.add_argument(
        "--raw-capture-directory",
        type=Path,
        help=(
            "Raw Femto capture containing camera_frame.json, frames.csv, and "
            "organized_depth_manifest.json; opens the object separation GUI."
        ),
    )
    parser.add_argument(
        "--reject-view",
        type=int,
        action="append",
        default=[],
        help="Zero-based input view index to exclude.",
    )
    parser.add_argument(
        "--loop-edge",
        action="append",
        default=[],
        metavar="SOURCE:TARGET",
        help="Explicit zero-based loop closure; repeat as needed.",
    )
    return parser


def discover_capture_views(
    capture_directory: str | Path,
) -> tuple[Path, ...]:
    """Load object-only tool-frame views from one recorded capture."""

    capture = Path(capture_directory)
    if not capture.is_dir():
        raise FileNotFoundError(f"capture directory does not exist: {capture}")
    view_directory = capture / "manual_stone_views_tool0"
    if not view_directory.is_dir():
        raise FileNotFoundError(
            "capture is missing manual_stone_views_tool0: "
            f"{capture}"
        )
    views = tuple(sorted(view_directory.glob("*.pcd"), key=_view_sort_key))
    if not views:
        raise FileNotFoundError(
            f"capture contains no object-only PCD views: {view_directory}"
        )
    return views


def _view_sort_key(path: Path) -> tuple[int, int | str]:
    try:
        return 0, int(path.stem)
    except ValueError:
        return 1, path.name


def default_session_directory(
    repository: str | Path,
    object_id: int,
    session_id: str | None = None,
) -> Path:
    """Resolve a safe app-local directory for a reconstruction session."""

    if object_id <= 0:
        raise ValueError("reconstruction object ID must be positive")
    resolved_session_id = session_id or datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )
    if (
        not resolved_session_id
        or resolved_session_id in {".", ".."}
        or Path(resolved_session_id).name != resolved_session_id
    ):
        raise ValueError("session ID must be one path-segment name")
    return (
        Path(repository).resolve()
        / "apps"
        / "tabletop"
        / "data"
        / "reconstruction"
        / str(object_id)
        / resolved_session_id
    )


def run_reconstruction(
    repository: str | Path,
    session_directory: str | Path,
    *,
    object_id: int | None = None,
    view_paths: tuple[Path, ...] = (),
    rejected_views: tuple[int, ...] = (),
    loop_edges: tuple[tuple[int, int], ...] = (),
    capture_indices: tuple[int, ...] | None = None,
    viewpoints: tuple[np.ndarray, ...] | None = None,
) -> ReconstructionSession:
    root = Path(repository).resolve()
    session_root = Path(session_directory).resolve()
    manifest = session_root / "session.json"
    store = ReconstructionSessionStore()
    if view_paths:
        if object_id is None:
            raise ValueError("--object-id is required when creating a session")
        rejected = set(rejected_views)
        invalid = sorted(
            index
            for index in rejected
            if index < 0 or index >= len(view_paths)
        )
        if invalid:
            raise ValueError(f"rejected view indices are invalid: {invalid}")
        if capture_indices is not None and len(capture_indices) != len(view_paths):
            raise ValueError("capture indices must match reconstruction views")
        if viewpoints is not None and len(viewpoints) != len(view_paths):
            raise ValueError("viewpoints must match reconstruction views")
        session = ReconstructionSession(
            object_id=object_id,
            views=tuple(
                ReconstructionViewRecord(
                    name=f"view_{index:03d}",
                    path=str(Path(path).resolve()),
                    accepted=index not in rejected,
                    capture_index=(
                        None if capture_indices is None else capture_indices[index]
                    ),
                    tool_T_camera=(
                        None if viewpoints is None else viewpoints[index]
                    ),
                )
                for index, path in enumerate(view_paths)
            ),
            loop_edges=loop_edges,
        )
    else:
        if not manifest.is_file():
            raise FileNotFoundError(
                "session does not exist; provide --object-id and --view"
            )
        session = store.load(manifest)
        if object_id is not None and object_id != session.object_id:
            raise ValueError("requested object ID differs from the session")

    profile = load_perception_profile(
        root / "apps" / "tabletop" / "config" / "perception.yml"
    ).reconstruction
    accepted_records = session.accepted_views()
    if len(accepted_records) < profile.minimum_view_support:
        raise ValueError("too few accepted views for configured voxel support")
    clouds = tuple(
        load_point_cloud(store.resolve_view_path(manifest, view))
        for view in accepted_records
    )
    accepted_indices = [
        index for index, view in enumerate(session.views) if view.accepted
    ]
    compact_index = {
        original: compact for compact, original in enumerate(accepted_indices)
    }
    compact_loops = tuple(
        (compact_index[source], compact_index[target])
        for source, target in session.loop_edges
        if source in compact_index and target in compact_index
    )
    refinement = TabletopPoseGraphRefiner(profile.pose_graph).refine(
        clouds,
        compact_loops,
        capture_indices=(
            None
            if any(view.capture_index is None for view in accepted_records)
            else tuple(int(view.capture_index) for view in accepted_records)
        ),
        viewpoints=(
            None
            if any(view.tool_T_camera is None for view in accepted_records)
            else tuple(view.tool_T_camera for view in accepted_records)
        ),
    )
    session = session.with_refinement(
        refinement.transforms,
        optimized=refinement.optimized,
        reason=refinement.reason,
    )
    store.save(session, manifest)

    aligned = tuple(
        cloud.transformed(transform)
        for cloud, transform in zip(clouds, refinement.transforms)
    )
    merge = MultiViewVoxelMerge(
        VoxelMergeConfig(
            profile.merge_voxel_size,
            profile.minimum_view_support,
        )
    ).merge(aligned)
    if not len(merge.cloud):
        raise ValueError("multi-view consensus produced an empty point cloud")
    output = session_root / "output"
    inertial = estimate_hull_inertial(merge.cloud.points)
    centered = PointCloudData(
        merge.cloud.points - inertial.center_of_mass
    )
    merged_path = write_point_cloud(output / "merged.pcd", centered)
    mesh = TabletopMeshReconstructor(profile.mesh).reconstruct(centered)
    mesh_path = write_triangle_mesh(output / "mesh.ply", mesh)
    dsf_exporter = TabletopDsfExporter(profile.dsf)
    dsf_path = dsf_exporter.write(
        output / "dsf.obj",
        (dsf_exporter.component(mesh),),
        inertial,
    )
    outputs = {
        "merged_point_cloud": str(merged_path.relative_to(session_root)),
        "mesh": str(mesh_path.relative_to(session_root)),
        "dsf": str(dsf_path.relative_to(session_root)),
    }
    session = session.with_outputs(
        merge.stats,
        outputs,
        unit_density_mass=inertial.mass,
        inertia=inertial.inertia,
        center_of_mass=inertial.center_of_mass,
    )
    store.save(session, manifest)
    return session


def _parse_loop_edge(value: str) -> tuple[int, int]:
    try:
        source, target = value.split(":", maxsplit=1)
        return int(source), int(target)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            f"invalid loop edge '{value}'; expected SOURCE:TARGET"
        ) from error


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        view_paths = (
            discover_capture_views(args.capture_directory)
            if args.capture_directory is not None
            else tuple(args.view)
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    session_directory = args.session_directory
    if session_directory is None:
        if args.object_id is None:
            parser.error(
                "--object-id is required when session_directory is omitted"
            )
        if (
            not view_paths
            and args.raw_capture_directory is None
            and args.session_id is None
        ):
            parser.error(
                "--session-id is required to resume a default session"
            )
        session_directory = default_session_directory(
            args.repository,
            args.object_id,
            args.session_id,
        )
    capture_indices = None
    viewpoints = None
    if args.raw_capture_directory is not None:
        if args.object_id is None:
            parser.error("--object-id is required for a raw capture")
        try:
            profile = load_perception_profile(
                Path(args.repository).resolve()
                / "apps"
                / "tabletop"
                / "config"
                / "perception.yml"
            )
            selector = InteractiveObjectSelector(
                profile.in_hand_capture.manual_remove_distance,
                profile.in_hand_capture.manual_max_remove_ratio,
            )
            prepared = TabletopRawCapturePreprocessor(profile).prepare(
                args.raw_capture_directory,
                Path(session_directory) / "preprocessing",
                selector,
            )
            view_paths = prepared.paths
            capture_indices = prepared.capture_indices
            viewpoints = prepared.viewpoints
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            parser.error(str(error))
    session = run_reconstruction(
        args.repository,
        session_directory,
        object_id=args.object_id,
        view_paths=view_paths,
        rejected_views=tuple(args.reject_view),
        loop_edges=tuple(_parse_loop_edge(value) for value in args.loop_edge),
        capture_indices=capture_indices,
        viewpoints=viewpoints,
    )
    print(
        f"reconstruction {session.status}: "
        f"{session_directory / 'session.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
