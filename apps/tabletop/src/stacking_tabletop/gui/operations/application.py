"""Entry point for the optional tabletop stacking operations console."""

import argparse
from pathlib import Path

from stacking_tabletop.gui.theme import DARK_STYLESHEET


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Open the tabletop planning, search inspection, and execution console."
        )
    )
    parser.add_argument("repository", type=Path, nargs="?", default=Path("."))
    parser.add_argument(
        "--run",
        "--debug-artifact",
        dest="run",
        type=Path,
        help="Open a native .stack run (JSON is accepted for legacy fixtures).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository = args.repository.resolve()
    try:
        from PySide6.QtWidgets import QApplication

        from .window import StackingOperationsWindow
    except ImportError as error:
        raise SystemExit(
            "The stacking operations console requires: "
            "python -m pip install -e 'apps/tabletop[operations]'"
        ) from error
    application = QApplication.instance() or QApplication([])
    application.setStyle("Fusion")
    application.setStyleSheet(DARK_STYLESHEET)
    window = StackingOperationsWindow(repository, args.run)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
