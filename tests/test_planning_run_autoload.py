"""Planning progress windows auto-open readable runs without minimizing their owner."""

import os
from pathlib import Path
from unittest.mock import Mock

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PySide6')
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget
from stacking_common.artifacts import PlanningRunContext, StackRunStore
from stacking_common.visualization import MCTSDebugArtifact

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('platform', ['tabletop', 'excavator'])
def test_wait_for_readable_run_and_minimize_independently(platform, tmp_path):
    if platform == 'tabletop':
        from stacking_tabletop.gui.operations.planning_run import PlanningRunDialog
    else:
        from stacking_excavator.gui.operations.planning_run import PlanningRunDialog
    app = QApplication.instance() or QApplication([])
    viewer = QWidget()
    dialog = PlanningRunDialog(ROOT, parent=viewer)
    opened = []
    dialog.run_available.connect(opened.append)
    viewer.show()
    dialog.show()
    try:
        assert dialog.windowType() == Qt.WindowType.Window
        assert dialog.windowFlags() & Qt.WindowType.WindowMinimizeButtonHint
        assert not dialog.isModal()
        dialog.destination = tmp_path / 'new.stack'
        dialog._run_timer.start()
        dialog._check_run_available()
        assert not opened
        dialog.destination.write_bytes(b'incomplete database')
        dialog._check_run_available()
        assert not opened and dialog._run_timer.isActive()
        dialog.destination = tmp_path / 'readable.stack'
        with StackRunStore.create_mcts(dialog.destination, MCTSDebugArtifact('world', ())) as writer:
            if platform == 'excavator':
                dialog._check_run_available()
                assert not opened  # Context is committed separately from the header.
                writer.append_run_context(PlanningRunContext('world', ('joint',), {}, ()))
            dialog.showMinimized()
            QTest.qWait(650)
            assert dialog.isMinimized()
            assert viewer.isVisible() and not viewer.isMinimized()
            assert opened == [dialog.destination]
            assert not dialog._run_timer.isActive()
            assert dialog.open_button.isEnabled()
            dialog._check_run_available()
            assert len(opened) == 1
            dialog.showNormal()
            assert not dialog.isMinimized()
    finally:
        dialog.close()
        viewer.close()
        app.processEvents()


@pytest.mark.parametrize('platform', ['tabletop', 'excavator'])
def test_autoload_refreshes_already_open_run(platform, tmp_path):
    pytest.importorskip('pyvistaqt')
    if platform == 'tabletop':
        from stacking_tabletop.gui.operations.window import StackingOperationsWindow as Window
    else:
        from stacking_excavator.gui.operations.window import ExcavatorOperationsWindow as Window
    window = Mock()
    path = tmp_path / 'run.stack'
    window._artifact_path = None
    Window._open_planning_run(window, path)
    window.open_artifact.assert_called_with(path, refresh=False)
    window._artifact_path = path
    Window._open_planning_run(window, path)
    window.open_artifact.assert_called_with(path, refresh=True)
