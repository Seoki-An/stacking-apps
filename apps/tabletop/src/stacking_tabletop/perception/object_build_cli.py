"""One resumable command for tabletop prepare, merge, and reconstruct."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

from stacking_common.workflows.object_build import ObjectBuildRunner, ObjectBuildWorkflow
from stacking_tabletop.config import load_inventory_profile

from .workflow import TabletopObjectBuildAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume the tabletop object-build workflow."
    )
    parser.add_argument("repository", type=Path)
    parser.add_argument("--object-id", type=int)
    parser.add_argument("--mass-kg", type=float)
    parser.add_argument("--session-id")
    parser.add_argument("--resume", type=Path, help="Existing workflow directory.")
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "merge", "reconstruct"),
        default="all",
    )
    parser.add_argument("--capture-directory", type=Path)
    return parser


def default_workflow_directory(
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
        / "object-build"
        / str(object_id)
        / name
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        repository = args.repository.resolve()
        if args.resume is not None:
            workflow = ObjectBuildWorkflow.resume(args.resume)
        else:
            if args.object_id is None:
                raise ValueError("--object-id is required for a new workflow")
            density = tabletop_density(repository, args.object_id)
            root = default_workflow_directory(
                repository,
                args.object_id,
                args.session_id,
            )
            workflow = ObjectBuildWorkflow.create(
                root,
                object_id=args.object_id,
                session_id=root.name,
                platform="tabletop",
                measured_mass_kg=args.mass_kg,
                density_kg_m3=density,
            )
        runner = ObjectBuildRunner(
            workflow,
            TabletopObjectBuildAdapter(repository, workflow.root),
        )
        stages = (
            ("prepare", "merge", "reconstruct")
            if args.stage == "all"
            else (args.stage,)
        )
        for stage in stages:
            print(f"running {stage}: {workflow.root}", flush=True)
            if stage == "prepare":
                if args.capture_directory is None:
                    raise ValueError("--capture-directory is required for prepare")
                attempt = runner.run_prepare(args.capture_directory)
            elif stage == "merge":
                attempt = runner.run_merge()
            else:
                attempt = runner.run_reconstruct()
            print(f"accepted {stage} attempt: {attempt.attempt_id}")
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(f"object-build workflow: {workflow.manifest}")
    return 0


def tabletop_density(repository: str | Path, object_id: int) -> float:
    root = Path(repository).resolve()
    profile = load_inventory_profile(
        root / "apps" / "tabletop" / "config" / "inventory.yml"
    )
    return profile.density_for(object_id)


if __name__ == "__main__":
    raise SystemExit(main())
