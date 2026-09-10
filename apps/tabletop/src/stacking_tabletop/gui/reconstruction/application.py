"""Entry point for the optional PySide/PyVista object-build workbench."""

import argparse
from pathlib import Path

from stacking_common.workflows.object_build import ObjectBuildWorkflow
from stacking_tabletop.perception.object_build_cli import (
    default_workflow_directory,
    tabletop_density,
)
from stacking_tabletop.gui.theme import DARK_STYLESHEET


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open the tabletop prepare–merge–reconstruct workbench."
    )
    parser.add_argument("repository", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--object-id", type=int)
    parser.add_argument("--mass-kg", type=float)
    parser.add_argument("--session-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository = args.repository.resolve()
    root = None
    if args.resume is not None:
        root = args.resume.resolve()
        ObjectBuildWorkflow.resume(root)
    elif args.object_id is not None:
        root = default_workflow_directory(
            repository,
            args.object_id,
            args.session_id,
        )
        ObjectBuildWorkflow.create(
            root,
            object_id=args.object_id,
            session_id=root.name,
            platform="tabletop",
            measured_mass_kg=args.mass_kg,
            density_kg_m3=tabletop_density(repository, args.object_id),
        )
    try:
        from PySide6.QtWidgets import QApplication
        from .integrated_window import ObjectBuildWindow
    except ImportError as error:
        raise SystemExit(
            "The workbench requires the tabletop workbench extra: "
            "python -m pip install -e 'apps/tabletop[perception,workbench]'"
        ) from error
    application = QApplication.instance() or QApplication([])
    application.setStyle("Fusion")
    application.setStyleSheet(DARK_STYLESHEET)
    window = ObjectBuildWindow(repository, root)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
