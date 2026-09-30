"""Optional GPU runtime selection without camera or model dependencies."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from stacking_tabletop.acquisition import hand_eye_fast_stereo as stereo


def setup(tmp_path):
    vendor = tmp_path/'.data/vendor/Fast-FoundationStereo'
    python = tmp_path/'.data/venvs/fast-stereo/bin/python'
    python.parent.mkdir(parents=True)
    python.touch()
    output = tmp_path/'result'
    output.mkdir()
    (output/'summary.json').write_text(json.dumps({'status': 'complete'}))
    return vendor, python, output


def test_optional_runtime_invocation_preserves_paths(tmp_path, monkeypatch):
    vendor, python, output = setup(tmp_path)
    child = Mock()
    child.wait.return_value = 0
    start = Mock(return_value=child)
    monkeypatch.setattr(stereo.subprocess, 'Popen', start)
    result = stereo.reconstruct_with_runtime(tmp_path/'session', tmp_path/'baseline', tmp_path/'refinement',
        output, repository=vendor, checkpoint=vendor/'weights/model.pth')
    command = start.call_args.args[0]
    assert command[:4] == [str(python), '-u', '-m', 'stacking_tabletop.acquisition.hand_eye_fast_stereo']
    assert command[command.index('--session')+1] == str(tmp_path/'session')
    assert result['status'] == 'complete'
    child.terminate.assert_not_called()


def test_cancel_terminates_inference_child(tmp_path, monkeypatch):
    vendor, python, output = setup(tmp_path)
    child = Mock()
    child.wait.side_effect = [KeyboardInterrupt(), 0]
    child.poll.return_value = None
    monkeypatch.setattr(stereo.subprocess, 'Popen', Mock(return_value=child))
    with pytest.raises(KeyboardInterrupt):
        stereo.reconstruct_with_runtime('session', 'baseline', 'refinement', output,
            repository=vendor, checkpoint=vendor/'model.pth')
    child.terminate.assert_called_once()
    assert child.wait.call_args.kwargs == {'timeout': 5}


def test_missing_runtime_explains_colmap_alternative(tmp_path):
    with pytest.raises(RuntimeError, match='--dense-backend colmap'):
        stereo.reconstruct_with_runtime('session', 'baseline', 'refinement', 'output',
            repository=tmp_path/'.data/vendor/model', checkpoint=tmp_path/'weights')


def test_recorded_pose_mode_omits_refinement_and_checks_execution_cancellation(tmp_path, monkeypatch):
    vendor, python, output = setup(tmp_path)
    child = Mock()
    child.poll.return_value = None
    start = Mock(return_value=child)
    monkeypatch.setattr(stereo.subprocess, 'Popen', start)
    def cancelled():
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        stereo.reconstruct_with_runtime(tmp_path/'session', None, None, output,
            repository=vendor, checkpoint=vendor/'weights/model.pth',
            region_file=tmp_path/'region.json', check=cancelled)
    command = start.call_args.args[0]
    assert '--baseline' not in command and '--refinement' not in command
    assert command[command.index('--region-file')+1] == str(tmp_path/'region.json')
    child.terminate.assert_called_once()
