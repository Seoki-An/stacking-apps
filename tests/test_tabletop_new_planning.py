"""New tabletop plans use the existing offline CLI and process supervisor."""

import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip('PySide6')
from PySide6.QtCore import QProcess
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from stacking_common.artifacts import StackRunStore, decode_legacy_mcts_debug
from stacking_tabletop.gui.operations.planning_run import PlanningRunDialog

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def qt():
    return QApplication.instance() or QApplication([])


def wait_for(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(20)
    assert predicate()


def test_new_dialog_is_passive_and_builds_tabletop_cli_arguments(qt, tmp_path):
    dialog = PlanningRunDialog(ROOT)
    try:
        assert not dialog.running
        assert dialog.start_button.isEnabled()
        assert not dialog.open_button.isEnabled()
        with pytest.raises(ValueError, match='output'):
            dialog.arguments()
        dialog.output_path.setText(str(tmp_path / 'new plan'))
        args = dialog.arguments()
        assert args[:2] == ['-u', str(ROOT / 'apps/tabletop/scripts/debug/mcts.py')]
        assert '--plan-motion' in args and '--resume-run' not in args
        assert '--object-ids' not in args and '--pick-pose-data' not in args
        assert dialog.pose_from_scan.currentData() is None
        assert '--pose-from-scan' not in args and '--no-pose-from-scan' not in args
        assert dialog.destination == tmp_path / 'new plan.stack'
        assert not dialog.destination.exists()
        pickup = tmp_path / 'trusted pickup.pkl'
        pickup.touch()
        dialog.pick_path.setText(str(pickup))
        dialog.object_ids.setText('1, 2 3')
        dialog.steps.setValue(3)
        dialog.wall_seconds.setValue(120)
        dialog.search_only.setChecked(True)
        args = dialog.arguments()
        assert args[args.index('--pick-pose-data') + 1] == str(pickup)
        assert args[args.index('--object-ids') + 1:] == ['1', '2', '3']
        assert args[args.index('--steps') + 1] == '3'
        assert args[args.index('--max-run-time') + 1] == '120'
        assert '--no-plan-motion' in args and '--plan-motion' not in args
        for value, flag in ((True, '--pose-from-scan'), (False, '--no-pose-from-scan')):
            dialog.pose_from_scan.setCurrentIndex(dialog.pose_from_scan.findData(value))
            args = dialog.arguments()
            assert flag in args
            assert args[args.index('--pick-pose-data') + 1] == str(pickup)
        assert 'apps/excavator' not in ' '.join(args)
        dialog.destination.write_text('preserve me')
        with pytest.raises(FileExistsError):
            dialog.arguments()
        assert dialog.destination.read_text() == 'preserve me'
    finally:
        dialog.close()


@pytest.mark.parametrize('value', ['1 1', '0', '-2', '1.2', 'one', ', ,'])
def test_object_id_validation(qt, tmp_path, value):
    dialog = PlanningRunDialog(ROOT)
    try:
        dialog.output_path.setText(str(tmp_path / 'run.stack'))
        dialog.object_ids.setText(value)
        with pytest.raises(ValueError, match='Object IDs'):
            dialog.arguments()
    finally:
        dialog.close()


def test_paths_are_relative_to_repository_and_missing_inputs_fail(qt, tmp_path):
    dialog = PlanningRunDialog(ROOT)
    try:
        assert dialog.config_path.text() == str(ROOT / 'apps/tabletop/config/mcts_cpu.yml')
        dialog.output_path.setText(str(tmp_path / 'run.stack'))
        dialog.config_path.setText('apps/tabletop/config/mcts_cpu.yml')
        args = dialog.arguments()
        assert args[args.index('--config') + 1] == str(ROOT / 'apps/tabletop/config/mcts_cpu.yml')
        dialog.pick_path.setText(str(tmp_path / 'missing.pkl'))
        with pytest.raises(ValueError, match='pickup'):
            dialog.arguments()
        dialog.pick_path.clear()
        dialog.config_path.setText(str(tmp_path / 'missing.yml'))
        with pytest.raises(ValueError, match='YAML'):
            dialog.arguments()
    finally:
        dialog.close()


@pytest.mark.parametrize('scan_override', [None, True, False])
def test_cli_search_completes_and_open_result_emits_path(qt, tmp_path, scan_override):
    profile = yaml.safe_load((ROOT / 'apps/tabletop/config/mcts_cpu.yml').read_text())
    profile['debug'].update(object_ids=[1], steps=1, iterations=1, max_search_time=2.,
                            transition_backend='kinematic', pose_solver='passthrough', pick_pose_data=None)
    profile['parallel']['enabled'] = False
    config = tmp_path / 'mcts.yml'
    config.write_text(yaml.safe_dump(profile))
    dialog = PlanningRunDialog(ROOT)
    output = tmp_path / 'computed.stack'
    dialog.config_path.setText(str(config))
    dialog.output_path.setText(str(output))
    dialog.search_only.setChecked(True)
    dialog.pose_from_scan.setCurrentIndex(dialog.pose_from_scan.findData(scan_override))
    dialog.wall_seconds.setValue(15)
    opened = []
    dialog.run_available.connect(opened.append)
    dialog.start_button.click()
    try:
        assert not dialog.inputs.isEnabled()
        assert not dialog.start_button.isEnabled()
        wait_for(lambda: not dialog.running, timeout=25)
        assert dialog.process.exitCode() == 0, dialog.log.toPlainText()
        info = StackRunStore.info(output)
        assert info.status == 'complete'
        assert info.metadata['request']['plan_motion'] is False
        assert info.metadata['request']['pose_from_scan'] is (scan_override is not False)
        assert not StackRunStore.load_execution_commands(output)
        assert dialog.inputs.isEnabled() and dialog.start_button.isEnabled()
        assert dialog.open_button.isEnabled()
        assert opened and all(path == output for path in opened)
        opened.clear()
        dialog.open_button.click()
        assert opened == [output]
    finally:
        if dialog.running:
            dialog.reject()
            dialog.process.waitForFinished(10000)
        dialog.close()


def test_stop_reaps_worker_and_keeps_result(qt, tmp_path, monkeypatch):
    fixture = ROOT / 'tests/fixtures/legacy_visualization/mcts_candidate_viewer.json'
    source = tmp_path / 'source.stack'
    StackRunStore.save_mcts(decode_legacy_mcts_debug(json.loads(fixture.read_text())), source)
    output = tmp_path / 'stopped.stack'
    worker = f'import shutil,time; shutil.copyfile({str(source)!r}, {str(output)!r}); print("ready",flush=True); time.sleep(60)'
    supervisor = ('import sys; from stacking_tabletop.planning.process import supervise_planning; '
                  f'sys.exit(supervise_planning([sys.executable,"-u","-c",{worker!r}],destination={str(output)!r}))')
    dialog = PlanningRunDialog(ROOT)
    dialog.destination = output
    monkeypatch.setattr(dialog, 'arguments', lambda: ['-u', '-c', supervisor])
    dialog.start()
    try:
        wait_for(lambda: 'ready' in dialog.log.toPlainText())
        dialog.reject()
        wait_for(lambda: not dialog.running)
        assert dialog.process.exitCode() == 130
        assert StackRunStore.info(output).status == 'cancelled'
        assert dialog.open_button.isEnabled()
        assert dialog.stop_button.isEnabled()
    finally:
        if dialog.running:
            dialog.reject()
            dialog.process.waitForFinished(10000)
        dialog.close()


def test_failed_process_start_restores_controls(qt, tmp_path, monkeypatch):
    import stacking_tabletop.gui.operations.planning_run as module
    dialog = PlanningRunDialog(ROOT)
    dialog.output_path.setText(str(tmp_path / 'run.stack'))
    monkeypatch.setattr(module.sys, 'executable', str(tmp_path / 'missing-python'))
    dialog.start()
    try:
        wait_for(lambda: not dialog.running and dialog.start_button.isEnabled())
        assert dialog.process.error() == QProcess.ProcessError.FailedToStart
        assert dialog.inputs.isEnabled() and dialog.stop_button.isEnabled()
        assert not dialog.open_button.isEnabled()
        assert dialog.log.toPlainText()
    finally:
        dialog.close()


def test_window_new_action_single_dialog_and_hardware_guard(qt, monkeypatch, tmp_path):
    import stacking_tabletop.gui.operations.window as windows
    warnings = []
    monkeypatch.setattr('stacking_tabletop.gui.operations.planning_run.QMessageBox.warning',
                        lambda *args: warnings.append(args[-1]))
    class View(QWidget):
        def __init__(self, parent, *args):
            super().__init__(parent)
        def show_empty(self):
            pass
    monkeypatch.setattr(windows, 'OperationsSceneView', View)
    window = windows.StackingOperationsWindow(ROOT)
    try:
        action = next(a for a in window.findChildren(windows.QAction) if a.text() == 'New planning run…')
        assert action.shortcut().toString() == 'Ctrl+N'
        action.trigger()
        dialog = window._planning_dialog
        assert dialog is not None and not dialog.running
        # Refocus a running process instead of starting a second planner.
        active = SimpleNamespace(running=True, showNormal=lambda: None, raise_=lambda: None)
        window._planning_dialog = active
        action.trigger()
        assert window._planning_dialog is active
        window._planning_dialog = dialog
        execution_view = window.execution_view
        window.execution_view = SimpleNamespace(running=True)
        action.trigger()
        assert window._planning_dialog is dialog
        dialog.output_path.setText(str(tmp_path / 'blocked.stack'))
        dialog.start()
        assert warnings and not dialog.running
        window.execution_view = execution_view
    finally:
        window._planning_dialog = None
        dialog.close()
        window.close()
