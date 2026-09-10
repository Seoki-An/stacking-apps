"""Entry point for the optional offline excavator Operations Console."""

import argparse
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--run", type=Path, help="Open a recorded excavator .stack run")
    parser.add_argument("--session", type=Path, help="Open/create a desktop perception and simulation session")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        from PySide6.QtWidgets import QApplication
        from stacking_excavator.gui.theme import DARK_STYLESHEET
        from .window import ExcavatorOperationsWindow
    except ImportError as error:
        raise SystemExit("Install console dependencies: python -m pip install -e 'apps/excavator[operations]'") from error
    application = QApplication.instance() or QApplication([])
    application.setStyle("Fusion")
    application.setStyleSheet(DARK_STYLESHEET)
    window = ExcavatorOperationsWindow(args.repository, args.run, args.session)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
