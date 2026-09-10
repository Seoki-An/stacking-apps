"""Interactive alignment and merge of prepared capture artifacts."""

import argparse
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from stacking_tabletop.config import load_perception_profile

from .cross_capture import (
    CrossCaptureInitializer,
    InteractiveCrossCaptureInitializer,
    TabletopCrossCaptureMerger,
    load_capture_cloud,
    resolve_capture_cloud,
    show_cross_capture_result,
)
from .reconstruction import write_point_cloud


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manually initialize, ICP-align, and merge independently "
            "grasped tabletop scans."
        )
    )
    parser.add_argument("repository", type=Path)
    parser.add_argument(
        "capture_directory",
        type=Path,
        nargs="+",
        help=(
            "Prepared capture directories, cloud files, or a parent directory "
            "whose immediate children are prepared captures; the first capture "
            "defines the output frame."
        ),
    )
    parser.add_argument("--object-id", type=int, required=True)
    parser.add_argument(
        "--session-id",
        help="Optional output session directory name.",
    )
    parser.add_argument(
        "--no-show-result",
        action="store_true",
        help="Do not open the final aligned-capture preview.",
    )
    return parser


def discover_merge_captures(
    paths: Iterable[str | Path],
) -> tuple[Path, ...]:
    """Expand each input into prepared captures while preserving input order."""
    captures = []
    seen = set()
    for path in paths:
        value = Path(path).expanduser().resolve()
        try:
            resolve_capture_cloud(value)
            candidates = (value,)
        except FileNotFoundError as direct_error:
            if not value.is_dir():
                raise direct_error
            candidates = tuple(
                child
                for child in sorted(value.iterdir(), key=lambda item: item.name)
                if child.is_dir() and _has_capture_cloud(child)
            )
            if not candidates:
                raise ValueError(
                    f"no prepared captures found in {value}; expected child "
                    "directories containing prepared/object_merged_tool0.pcd"
                ) from direct_error
        for candidate in candidates:
            key = resolve_capture_cloud(candidate)
            if key not in seen:
                captures.append(candidate)
                seen.add(key)
    if len(captures) < 2:
        raise ValueError(
            "at least two prepared capture directories are required after "
            "expanding parent directories"
        )
    return tuple(captures)


def default_merge_directory(
    repository: str | Path,
    object_id: int,
    session_id: str | None = None,
) -> Path:
    if object_id <= 0:
        raise ValueError("object ID must be positive")
    name = session_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("session ID must be one path-segment name")
    return (
        Path(repository).resolve()
        / "apps"
        / "tabletop"
        / "data"
        / "merged"
        / str(object_id)
        / name
    )


def run_cross_capture_merge(
    repository: str | Path,
    session_directory: str | Path,
    *,
    object_id: int,
    capture_directories: tuple[Path, ...],
    initializer: CrossCaptureInitializer | None = None,
    show_result: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
) -> Path:
    if object_id <= 0:
        raise ValueError("object ID must be positive")
    if len(capture_directories) < 2:
        raise ValueError("at least two capture directories are required")
    root = Path(repository).resolve()
    session = Path(session_directory).resolve()
    profile = load_perception_profile(
        root / "apps" / "tabletop" / "config" / "perception.yml"
    )
    capture_paths = tuple(
        resolve_capture_cloud(path) for path in capture_directories
    )
    clouds = tuple(load_capture_cloud(path) for path in capture_paths)
    seed_builder = initializer or InteractiveCrossCaptureInitializer(
        profile.cross_capture_merge
    )
    result = TabletopCrossCaptureMerger(profile.cross_capture_merge).merge(
        clouds,
        initializer=seed_builder,
        progress=progress,
    )
    if show_result:
        show_cross_capture_result(result)

    merged_path = write_point_cloud(
        session / "merged.pcd",
        result.cloud,
    )
    report = {
        "schema_version": 1,
        "status": "merged",
        "object_id": object_id,
        "coordinate_frame": "first_capture_tool0",
        "captures": [str(path) for path in capture_paths],
        "profile": {
            "registration": asdict(profile.cross_capture_merge.registration),
            "quality": asdict(profile.cross_capture_merge.quality),
            "preview_voxel_size": profile.cross_capture_merge.preview_voxel_size,
            "output_voxel_size": profile.cross_capture_merge.output_voxel_size,
        },
        "alignments": [
            {
                "source_index": alignment.source_index,
                "transform": alignment.transform.tolist(),
                "target_coverage": alignment.registration.target_coverage,
                "source_coverage": alignment.registration.source_coverage,
                "rmse": alignment.registration.rmse,
                "correspondences": alignment.registration.correspondences,
                "decision": {
                    "accepted": alignment.decision.accepted,
                    "reasons": list(alignment.decision.reasons),
                    "correction_translation": (
                        alignment.decision.correction_translation
                    ),
                    "correction_rotation_degrees": (
                        alignment.decision.correction_rotation_degrees
                    ),
                },
            }
            for alignment in result.alignments
        ],
        "point_count": len(result.cloud),
        "merged_point_cloud": str(merged_path.relative_to(session)),
    }
    report_path = session / "merge.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        captures = discover_merge_captures(args.capture_directory)
        session = default_merge_directory(
            args.repository,
            args.object_id,
            args.session_id,
        )
        print(f"merging {len(captures)} prepared captures:")
        for capture in captures:
            print(f"  {capture}")
        report = run_cross_capture_merge(
            args.repository,
            session,
            object_id=args.object_id,
            capture_directories=captures,
            show_result=not args.no_show_result,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"cross-capture merge: {report}")
    return 0


def _has_capture_cloud(directory: Path) -> bool:
    try:
        resolve_capture_cloud(directory)
    except FileNotFoundError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
