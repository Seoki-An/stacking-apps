"""Operator folder choices resolve to the intended saved field capture."""

import json

import pytest

from stacking_tabletop.perception.field_source import resolve_field_capture_directory


@pytest.fixture
def capture(tmp_path):
    merged = tmp_path / 'merged'
    merged.mkdir()
    (merged / 'field_scan.json').write_text('{}')
    results = merged / 'pose_identification'
    results.mkdir()
    (results / 'stone_poses.json').write_text(json.dumps({'source_directory': str(merged)}))
    (tmp_path / 'field_workflow.json').write_text(json.dumps({
        'cameras': ['primary', 'secondary'], 'pose_input_dir': str(merged),
    }))
    return merged


def test_capture_workflow_and_results_resolve_to_same_input(capture):
    for selection in (capture, capture.parent, capture / 'pose_identification'):
        assert resolve_field_capture_directory(selection) == capture


def test_relocated_workflow_prefers_its_own_capture(capture):
    (capture.parent / 'field_workflow.json').write_text(json.dumps({
        'pose_input_dir': '/old/workspace/run/merged',
    }))
    assert resolve_field_capture_directory(capture.parent) == capture


def test_separate_identification_results_resolve_recorded_source(capture):
    retry = capture.parent / 'retry' / 'pose_identification'
    retry.mkdir(parents=True)
    (retry / 'stone_poses.json').write_text(json.dumps({'source_directory': str(capture)}))
    assert resolve_field_capture_directory(retry) == capture
    assert resolve_field_capture_directory(retry.parent) == capture


def test_ambiguous_or_unfinished_dual_capture_is_not_silently_downgraded(tmp_path):
    for name in ('primary', 'secondary'):
        camera = tmp_path / name
        camera.mkdir()
        (camera / 'field_scan.json').write_text('{}')
    with pytest.raises(ValueError, match='multiple camera captures'):
        resolve_field_capture_directory(tmp_path)
    (tmp_path / 'field_workflow.json').write_text(json.dumps({'cameras': ['primary', 'secondary']}))
    with pytest.raises(ValueError, match='no merged capture'):
        resolve_field_capture_directory(tmp_path)
    assert resolve_field_capture_directory(tmp_path / 'primary') == tmp_path / 'primary'


def test_invalid_and_cyclic_capture_references_are_actionable(tmp_path):
    with pytest.raises(ValueError, match='select a capture containing field_scan.json'):
        resolve_field_capture_directory(tmp_path)
    report = tmp_path / 'stone_poses.json'
    report.write_text(json.dumps({'source_directory': str(tmp_path)}))
    with pytest.raises(ValueError, match='cyclic'):
        resolve_field_capture_directory(tmp_path)
    report.write_text('broken JSON')
    with pytest.raises(ValueError, match='cannot read field capture reference'):
        resolve_field_capture_directory(tmp_path)
