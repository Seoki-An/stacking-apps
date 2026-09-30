"""Both planning GUIs publish replacements only after successful completion."""
import json
from pathlib import Path
import sys
import sqlite3
import time

import pytest

from stacking_common.artifacts import PlanningRunContext, StackRunStore, decode_legacy_mcts_debug
from stacking_common.workflows.planning_output import PlanningRunOutput

ROOT = Path(__file__).resolve().parents[1]


def run_file(path, label):
    fixture = ROOT / 'tests/fixtures/legacy_visualization/mcts_candidate_viewer.json'
    artifact = decode_legacy_mcts_debug(json.loads(fixture.read_text()))
    StackRunStore.save_mcts(artifact, path, metadata={'label': label})
    return path


def test_output_default_rejects_overwrite_and_resume_source(tmp_path):
    destination = run_file(tmp_path / 'run.stack', 'original')
    original = destination.read_bytes()
    with pytest.raises(FileExistsError):
        PlanningRunOutput(destination)
    with pytest.raises(ValueError, match='source'):
        PlanningRunOutput(destination, overwrite=True, source=destination)
    link = tmp_path / 'source-link.stack'
    link.hardlink_to(destination)
    with pytest.raises(ValueError, match='source'):
        PlanningRunOutput(link, overwrite=True, source=destination)
    with pytest.raises(ValueError, match='directory'):
        PlanningRunOutput(tmp_path, overwrite=True)
    assert destination.read_bytes() == original


def test_incomplete_or_unreadable_result_does_not_replace_original(tmp_path):
    target = run_file(tmp_path / 'run.stack', 'original')
    original = target.read_bytes()
    output = PlanningRunOutput(target, overwrite=True)
    assert not output.path.exists()
    run_file(output.path, 'incomplete')
    with StackRunStore.open_append(output.path) as writer:
        writer.finish('failed')
    assert output.finish(True) == output.path != target
    assert target.read_bytes() == original
    output.path.write_bytes(b'not a run')
    with pytest.raises((OSError, ValueError, sqlite3.Error)):
        output.finish(True)
    assert target.read_bytes() == original


@pytest.mark.parametrize('platform', ['tabletop', 'excavator'])
@pytest.mark.parametrize('outcome', ['success', 'failure', 'stop'])
def test_gui_overwrite_process_and_open_result(tmp_path, monkeypatch, platform, outcome):
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest
    if platform == 'tabletop':
        from stacking_tabletop.gui.operations.planning_run import PlanningRunDialog
    else:
        from stacking_excavator.gui.operations.planning_run import PlanningRunDialog
    app = QApplication.instance() or QApplication([])
    target = run_file(tmp_path / 'existing.stack', 'original')
    template = run_file(tmp_path / 'replacement.stack', 'replacement')
    if platform == 'excavator':
        with StackRunStore.open_append(template) as writer:
            writer.append_run_context(PlanningRunContext('world', ('joint',), {}, ()))
    original = target.read_bytes()
    dialog = PlanningRunDialog(ROOT)
    dialog.output_path.setText(str(target))
    dialog.search_only.setChecked(True)
    dialog.pose_from_scan.setCurrentIndex(dialog.pose_from_scan.findData(False))
    with pytest.raises(FileExistsError):
        dialog.arguments()
    dialog.overwrite.setChecked(True)
    original_arguments = dialog.arguments
    staged = []
    def arguments():
        args = original_arguments()
        pending = Path(args[args.index('--save-run') + 1])
        staged.append(pending)
        assert pending != target and not pending.exists()
        assert target.read_bytes() == original
        worker = (f'import shutil,time,sys; shutil.copyfile({str(template)!r},{str(pending)!r}); '
                  f'print("ready",flush=True); time.sleep({60 if outcome == "stop" else .3}); '
                  f'sys.exit({1 if outcome == "failure" else 0})')
        supervisor = ('import sys; from stacking_common.workflows.process import supervise_planning; '
                      f'sys.exit(supervise_planning([sys.executable,"-u","-c",{worker!r}],destination={str(pending)!r}))')
        return ['-u', '-c', supervisor]
    monkeypatch.setattr(dialog, 'arguments', arguments)
    opened = []
    dialog.run_available.connect(opened.append)
    dialog.start()
    def wait_for(predicate):
        deadline = time.monotonic() + 10
        while not predicate() and time.monotonic() < deadline:
            QTest.qWait(20)
        assert predicate(), dialog.log.toPlainText()
    try:
        wait_for(lambda: 'ready' in dialog.log.toPlainText())
        assert target.read_bytes() == original
        assert not dialog.overwrite.isEnabled()
        if outcome == 'stop':
            dialog.reject()
        wait_for(lambda: not dialog.running)
        assert dialog.open_button.isEnabled()
        # Completion auto-opens the published path, including after promotion
        # of the pending replacement. The explicit button remains available.
        assert opened and opened[-1] == dialog.destination
        assert all(path in (staged[0], dialog.destination) for path in opened)
        opened.clear()
        dialog.open_button.click()
        if outcome == 'success':
            assert StackRunStore.info(target).metadata['label'] == 'replacement'
            assert not staged[0].exists()
            assert opened == [target]
        else:
            assert target.read_bytes() == original
            assert staged[0].is_file()
            assert StackRunStore.info(staged[0]).status == ('cancelled' if outcome == 'stop' else 'failed')
            assert opened == [staged[0]]
        assert dialog.destination == opened[0]
    finally:
        if dialog.running:
            dialog.reject()
            dialog.process.waitForFinished(10000)
        dialog.close()
        app.processEvents()


def test_tabletop_resume_existing_output_waits_for_overwrite_selection(tmp_path):
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from stacking_tabletop.gui.operations.planning_run import PlanningRunDialog
    from test_tabletop_planning_restart import _checkpoint_run
    app = QApplication.instance() or QApplication([])
    source = tmp_path / 'source.stack'
    _checkpoint_run(source)
    destination = run_file(tmp_path / 'resumed.stack', 'old branch')
    dialog = PlanningRunDialog(ROOT, source, destination, ROOT / 'apps/tabletop/config/mcts.yml')
    try:
        assert not dialog.running and not dialog.inputs.isHidden()
        assert not dialog.start_button.isHidden()
        with pytest.raises(FileExistsError):
            dialog.arguments()
        dialog.overwrite.setChecked(True)
        args = dialog.arguments()
        assert args[args.index('--resume-run') + 1] == str(source)
        assert Path(args[args.index('--save-run') + 1]) != destination
        dialog.output_path.setText(str(source))
        with pytest.raises(ValueError, match='source'):
            dialog.arguments()
    finally:
        dialog.close()
        app.processEvents()
