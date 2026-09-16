"""Synthetic end-to-end replay, portability, provenance and rejection tests."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from stacking_tabletop.acquisition.hand_eye_replay import (
    assemble_session, inspect_session, replay_session, read_json, native_depth_rays)
from stacking_tabletop.acquisition.hand_eye_synthetic import generate_demo, SCENE
from stacking_tabletop.acquisition.hand_eye_capture import digest
from stacking_common.pointcloud.synthetic_depth import raycast_ellipsoids


@pytest.fixture(scope='module')
def demo(tmp_path_factory):
    root = tmp_path_factory.mktemp('replay') / 'demo'
    generate_demo(root, visualize=False)
    return root


def test_synthetic_end_to_end_preserves_frame_pixel_provenance_and_table_height(demo):
    report = read_json(demo / 'replay/replay.json')
    assert report['status'] == 'complete' and report['data_origin'] == 'synthetic'
    assert len(report['frames']) == 12 and len(report['references']) == 2
    assert not report['accuracy_1mm_verified']
    with np.load(demo / 'replay/aggregate.npz') as data:
        points, frames, pixels = data['points'], data['frame_index'], data['pixel_index']
    assert len(points) == report['point_count'] > 100000
    errors = []
    for index, frame in enumerate(report['frames']):
        mask = frames == index
        path = demo / 'session' / frame['source']
        root = path.parent.parent
        geometry = read_json(root / 'device.json')['streams']['depth']
        record = read_json(str(path) + '.json')
        truth, labels = raycast_ellipsoids(native_depth_rays(geometry), record['base_link_T_depth'], **SCENE)
        with np.load(str(path) + '.npz') as raw:
            np.testing.assert_array_equal(pixels[mask], np.flatnonzero(raw['depth']))
            camera_z = ((points[mask]-np.asarray(record['base_link_T_depth'])[:3, 3])
                        @ np.asarray(record['base_link_T_depth'])[:3, :3])[:, 2]
            np.testing.assert_allclose(camera_z, raw['depth'].ravel()[pixels[mask]]*.00025, atol=1e-12)
        table = labels.ravel()[pixels[mask]] == 1
        errors.extend(points[mask][table, 2] - SCENE['table_z'])
    assert abs(np.mean(errors)) < .00002
    assert .0001 < np.std(errors) < .0004


def test_session_can_move_and_replay_without_original_inputs_or_hardware_imports(demo, tmp_path):
    relocated = tmp_path / 'moved'
    shutil.copytree(demo / 'session', relocated)
    # Block hardware imports even if ROS happens to be installed in the environment.
    script = '''
import sys
class BlockHardware:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'rclpy', 'pyrealsense2', 'tf2_ros'}:
            raise RuntimeError('unexpected hardware dependency: '+fullname)
sys.meta_path.insert(0, BlockHardware())
from stacking_tabletop.acquisition.hand_eye_replay import replay_session
replay_session(sys.argv[1], sys.argv[2], visualize=False)
'''
    subprocess.run([sys.executable, '-c', script, str(relocated), str(tmp_path / 'again')], check=True)
    with np.load(demo / 'replay/aggregate.npz') as original, np.load(tmp_path / 'again/aggregate.npz') as repeated:
        for name in original.files:
            np.testing.assert_array_equal(original[name], repeated[name])


def test_assembly_rejects_duplicate_reverse_incompatible_and_incomplete_views(demo, tmp_path):
    views = sorted((demo / 'inputs').glob('view_*'))
    with pytest.raises(ValueError, match='distinct'):
        assemble_session(tmp_path / 'duplicate', [views[0], views[0]])
    with pytest.raises(ValueError, match='source time'):
        assemble_session(tmp_path / 'reverse', views[::-1])
    with pytest.raises(ValueError, match='both FEMTO'):
        assemble_session(tmp_path / 'missing_after', views, femto_before=demo / 'inputs/femto_before')
    with pytest.raises(ValueError, match='bracket'):
        assemble_session(tmp_path / 'wrong_order', views, femto_before=demo / 'inputs/femto_after',
                         femto_after=demo / 'inputs/femto_before')
    changed = tmp_path / 'changed'
    shutil.copytree(views[1], changed)
    manifest = read_json(changed / 'manifest.json')
    manifest['status'] = 'incomplete'
    (changed / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='incomplete'):
        assemble_session(tmp_path / 'bad', [views[0], changed])


@pytest.mark.parametrize('relative', ['views/000/frames/0000.npz', 'views/001/device.json',
                                     'femto_after/field_fused_base_link.pcd', 'reference.json'])
def test_session_rejects_tampering(demo, tmp_path, relative):
    shutil.copytree(demo / 'session', tmp_path / 'session')
    with (tmp_path / 'session' / relative).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        inspect_session(tmp_path / 'session')


def test_no_overwrite_or_replay_inside_inputs(demo, tmp_path):
    with pytest.raises(ValueError, match='separate'):
        replay_session(demo / 'session', demo / 'session/results', visualize=False)
    with pytest.raises(FileExistsError):
        replay_session(demo / 'session', demo / 'replay', visualize=False)


def test_unknown_nonzero_native_distortion_is_rejected():
    with pytest.raises(ValueError, match='unsupported'):
        native_depth_rays(dict(width=2, height=2, K=np.eye(3),
                               distortion_model='distortion.inverse_brown_conrady', distortion=[.1, 0, 0, 0, 0]))


@pytest.mark.parametrize('change', ['geometry', 'origin'])
def test_valid_but_incompatible_view_is_rejected(demo, tmp_path, change):
    first = demo / 'inputs/view_000'
    changed = tmp_path / 'changed'
    shutil.copytree(demo / 'inputs/view_001', changed)
    device, manifest = read_json(changed / 'device.json'), read_json(changed / 'manifest.json')
    if change == 'geometry':
        device['streams']['depth']['K'][0][0] += 1
    else:
        device['data_origin'] = manifest['data_origin'] = 'hardware'
    (changed / 'device.json').write_text(json.dumps(device))
    manifest['files']['device.json'] = digest(changed / 'device.json')
    (changed / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='incompatible'):
        assemble_session(tmp_path / 'bad', [first, changed])


def test_render_failure_leaves_incomplete_output_and_unchanged_inputs(demo, tmp_path, monkeypatch):
    import stacking_common.rendering.pointcloud as rendering
    def fail(*args, **kwargs):
        raise RuntimeError('render failed')
    monkeypatch.setattr(rendering, 'render_clouds', fail)
    before = inspect_session(demo / 'session')
    with pytest.raises(RuntimeError, match='render failed'):
        replay_session(demo / 'session', tmp_path / 'failed')
    assert read_json(tmp_path / 'failed/replay.json')['status'] == 'incomplete'
    assert inspect_session(demo / 'session') == before
