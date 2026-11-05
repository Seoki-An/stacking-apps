"""Recovery blocks competing operations and preserves failed-step evidence."""

import threading
import time

import pytest
pytest.importorskip('PySide6')
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QCloseEvent

from stacking_common.artifacts import StackRunStore
from stacking_tabletop.execution.sequence import execute_saved_run
from stacking_tabletop.gui.operations.execution_view import ExecutionView
from test_tabletop_saved_execution import ROOT, saved, executor_for


class Recovery(QThread):
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.release = threading.Event()

    def run(self):
        self.release.wait(5.)
        self.completed.emit('Recovered; torque OFF.')


def test_recovery_blocks_execution_and_keeps_reconciliation(saved, monkeypatch):
    app = QApplication.instance() or QApplication([])
    transport, executor = executor_for(saved)
    transport.feedback.receive_fault('overload')
    assert execute_saved_run(saved, executor)[0].status.value == 'failed'
    original = StackRunStore.load_execution_events(saved.path)
    monkeypatch.setattr('stacking_tabletop.gui.operations.execution_view.confirm_gripper_recovery', lambda _: True)
    view = ExecutionView(ROOT, recovery_worker_factory=Recovery)
    view.load(saved.path)
    assert view.recover_gripper_button.isEnabled()
    view.recover_gripper_button.click()
    try:
        assert view.running and view.recovering
        assert not view.recover_gripper_button.isEnabled()
        assert not view.start_button.isEnabled() and not view.reconcile_button.isEnabled()
        assert not view.continue_button.isEnabled() and not view.abort_button.isEnabled()
        view.abort()  # Cannot cancel a reboot that may have already been sent.
        view.approve()
        view.adjust(1)
        view.toggle_hold()
        event = QCloseEvent()
        view.closeEvent(event)
        assert not event.isAccepted()
        with pytest.raises(ValueError):
            view.load(saved.path)
        view.recovery_worker.release.set()
        assert view.recovery_worker.wait(2000)
        app.processEvents()
        assert not view.running
        assert view.reconcile_button.isEnabled() and not view.start_button.isEnabled()
        assert 'physical reconciliation' in view.status.text()
        assert StackRunStore.load_execution_events(saved.path) == original
        assert StackRunStore.load_execution_checkpoint(saved.path).requires_reconciliation
    finally:
        view.recovery_worker.release.set()
        view.recovery_worker.wait(2000)
        view.close()


def test_cancel_confirmation_and_other_workflow_never_create_recovery(saved, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr('stacking_tabletop.gui.operations.execution_view.confirm_gripper_recovery', lambda _: False)
    view = ExecutionView(ROOT, recovery_worker_factory=lambda _: pytest.fail('unexpected recovery'))
    view.load(saved.path)
    view.recover_gripper_button.click()
    assert not view.running
    view.can_start = lambda: False
    view._controls()
    assert not view.recover_gripper_button.isEnabled()
    view.recover_gripper()
    view.close()
