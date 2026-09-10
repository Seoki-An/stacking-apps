"""Prepare one or more raw captures as object-only point-cloud artifacts."""

import argparse
from collections.abc import Callable
from pathlib import Path

from stacking_tabletop.config import load_perception_profile

from .extraction import InteractiveObjectSelector
from .in_hand import PreparedCapture, TabletopRawCapturePreprocessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter, align, crop, and consensus-merge one raw tabletop capture "
            "or every raw capture immediately below a directory."
        )
    )
    parser.add_argument("repository", type=Path)
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument(
        "--output-directory",
        type=Path,
        help=(
            "Single capture: exact output directory. Batch: output root with one "
            "directory per capture. Defaults to CAPTURE/prepared."
        ),
    )
    return parser


def discover_capture_directories(directory: str | Path) -> tuple[Path, ...]:
    """Resolve a raw capture or its immediate raw-capture children."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"capture directory does not exist: {root}")
    if _is_raw_capture_directory(root):
        return (root,)
    captures = tuple(
        child
        for child in sorted(root.iterdir(), key=lambda path: path.name)
        if child.is_dir() and _is_raw_capture_directory(child)
    )
    if not captures:
        raise ValueError(
            f"no raw captures found in {root}; expected camera_frame.json, "
            "frames.csv, and raw_pcd/*.pcd"
        )
    return captures


def plan_capture_preparation(
    capture_directory: str | Path,
    output_directory: str | Path | None = None,
) -> tuple[tuple[Path, Path], ...]:
    """Map discovered captures to unambiguous prepared artifact directories."""
    captures = discover_capture_directories(capture_directory)
    if output_directory is None:
        return tuple((capture, capture / "prepared") for capture in captures)
    output = Path(output_directory).expanduser().resolve()
    if len(captures) == 1:
        return ((captures[0], output),)
    return tuple((capture, output / capture.name) for capture in captures)


def run_prepare_capture(
    repository: str | Path,
    capture_directory: str | Path,
    output_directory: str | Path | None = None,
    *,
    selector: object | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    icp_inspector: object | None = None,
) -> PreparedCapture:
    root = Path(repository).resolve()
    capture = Path(capture_directory).resolve()
    output = (
        capture / "prepared"
        if output_directory is None
        else Path(output_directory).resolve()
    )
    profile = load_perception_profile(
        root / "apps" / "tabletop" / "config" / "perception.yml"
    )
    if selector is None:
        selector = InteractiveObjectSelector(
            profile.in_hand_capture.manual_remove_distance,
            profile.in_hand_capture.manual_max_remove_ratio,
        )
    return TabletopRawCapturePreprocessor(profile).prepare(
        capture,
        output,
        selector,
        progress=progress,
        icp_inspector=icp_inspector,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        jobs = plan_capture_preparation(
            args.capture_directory,
            args.output_directory,
        )
        prepared_captures = []
        for index, (capture, output) in enumerate(jobs, start=1):
            print(
                f"preparing capture [{index}/{len(jobs)}]: {capture}",
                flush=True,
            )
            prepared_captures.append(
                run_prepare_capture(args.repository, capture, output)
            )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    for prepared in prepared_captures:
        print(f"prepared capture: {prepared.report_path}")
        print(f"prepared object cloud: {prepared.merged_path}")
    print(f"prepared {len(prepared_captures)} capture(s)")
    return 0


def _is_raw_capture_directory(directory: Path) -> bool:
    raw = directory / "raw_pcd"
    return (
        (directory / "camera_frame.json").is_file()
        and (directory / "frames.csv").is_file()
        and raw.is_dir()
        and any(
            path.is_file() and path.suffix.lower() == ".pcd"
            for path in raw.iterdir()
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
