"""Calibration consistency checks independent of devices and local PoseID files."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from stacking_tabletop.acquisition.hand_eye import HandEyeCalibration
from stacking_tabletop.acquisition.hand_eye_cli import main
from stacking_tabletop.config.hand_eye import load_hand_eye_profile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def profile_path(tmp_path):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('xyz', [4, -8, 12], degrees=True).as_matrix()
    transform[:3, 3] = [-.03, -.075, .083]
    matrix = np.array([[903., 0., 648.], [0., 904., 371.], [0., 0., 1.]])
    dist = np.array([.14, -.46, -.001, .0002, .39, 0., 0., 0.])
    np.savez(tmp_path/'hand_eye.npz', T_gripper_to_camera=transform,
             T_EE_to_cam1=transform, T_gripper_to_camera_tf=transform,
             cameraMatrix=matrix, distCoeffs=dist, method='Tsai',
             robot_pose_source='tf', robot_pose_parent_frame='base', robot_pose_child_frame='tool0')
    np.savez(tmp_path/'intrinsics.npz', K=matrix, dist=dist,
             K_factory=matrix+np.diag([3., 2., 0.]), dist_factory=np.zeros(8),
             camera='D435_COLOR', method='d435_handeye_log_charuco_intrinsic',
             image_size=[1280, 720], distortion_model='k3')
    (tmp_path/'installation.yml').write_text(yaml.safe_dump({'handeye_calib': {'ros__parameters': {
        'rs_serial': 'camera-A', 'rs_width': 1280, 'rs_height': 720, 'tf_base_frame': 'base',
        'tf_tool_frame': 'tool0', 'robot_pose_source': 'tf'}}}))
    values = dict(schema_version=1, serial='camera-A', color_frame='d435_color_optical_frame',
                  base_frame='base', tool_frame='tool0', robot_pose_source='tf')
    for key, name in [('hand_eye','hand_eye.npz'), ('intrinsics','intrinsics.npz'),
                      ('installation','installation.yml')]:
        values[key] = {'path': name, 'sha256': digest(tmp_path/name)}
    path = tmp_path/'profile.yml'
    path.write_text(yaml.safe_dump(values))
    return path


def change_npz(profile_path, name, **changes):
    config = yaml.safe_load(profile_path.read_text())
    path = profile_path.parent/config[name]['path']
    with np.load(path, allow_pickle=False) as z:
        values = dict(z)
    values.update(changes)
    np.savez(path, **values)
    config[name]['sha256'] = digest(path)
    profile_path.write_text(yaml.safe_dump(config))


def test_load_and_compose_physical_tool_in_correct_base(profile_path):
    calibration = HandEyeCalibration.load(profile_path)
    robot = np.eye(4)
    robot[:3, :3] = Rotation.from_euler('xyz', [10, 20, 35], degrees=True).as_matrix()
    robot[:3, 3] = [.3, -.4, .2]
    expected = np.diag([-1., -1., 1., 1.]) @ robot @ calibration.tool_T_color
    np.testing.assert_allclose(calibration.base_link_T_color(robot), expected)
    point = [.1, -.02, .4, 1.]
    np.testing.assert_allclose(calibration.base_link_T_color(robot) @ point, expected @ point)
    assert calibration.report()['status'] == 'offline_consistent'
    assert not calibration.report()['accuracy_1mm_verified']
    assert 'not a live-device' in calibration.report()['identity_provenance']
    with pytest.raises(ValueError):
        calibration.tool_T_color[0, 3] = 0.
    with pytest.raises(TypeError):
        calibration.artifacts['hand_eye'] = b'changed'


def test_base_link_contract_is_not_rotated_twice(profile_path):
    change_npz(profile_path, 'hand_eye', robot_pose_parent_frame='base_link')
    config = yaml.safe_load(profile_path.read_text())
    binding_path = profile_path.parent/'installation.yml'
    binding = yaml.safe_load(binding_path.read_text())
    binding['handeye_calib']['ros__parameters']['tf_base_frame'] = 'base_link'
    binding_path.write_text(yaml.safe_dump(binding))
    config['installation']['sha256'] = digest(binding_path)
    config['base_frame'] = 'base_link'
    profile_path.write_text(yaml.safe_dump(config))
    calibration = HandEyeCalibration.load(profile_path)
    np.testing.assert_allclose(calibration.base_link_T_color(np.eye(4)), calibration.tool_T_color)


@pytest.mark.parametrize('changes,reason', [
    ({'cameraMatrix': np.diag([800., 800., 1.])}, 'intrinsics mismatch'),
    ({'distCoeffs': np.zeros(8)}, 'distortion mismatch'),
    ({'image_size': [640,480]}, 'image size mismatch'),
    ({'distortion_model': 'inverse_brown_conrady'}, 'distortion model differs'),
    ({'robot_pose_parent_frame': 'base_link'}, 'parent_frame'),
    ({'robot_pose_child_frame': 'grasp_frame'}, 'child_frame'),
    ({'robot_pose_source': 'custom'}, 'pose_source'),
    ({'camera_serial': 'camera-B'}, 'serial'),
    ({'observation_frame': 'depth_optical_frame'}, 'optical frame'),
    ({'T_EE_to_cam1': np.eye(4)}, 'transform aliases'),
    ({'T_gripper_to_camera_tf': np.eye(4)}, 'robot-source transform'),
    ({'T_gripper_to_camera': np.diag([2., 1., 1., 1.])}, 'orthonormal'),
    ({'method': 'joint_bundle_adjustment_v1', 'residuals_translation_mm': np.zeros(10)}, 'solver-provenance'),
    ({'refined_from': '/some/old/result.npz'}, 'solver-provenance'),
])
def test_rejects_incompatible_hand_eye_even_with_updated_digest(profile_path, changes, reason):
    change_npz(profile_path, 'hand_eye', **changes)
    with pytest.raises(ValueError, match=reason):
        HandEyeCalibration.load(profile_path)


@pytest.mark.parametrize('changes,reason', [
    ({'image_size': [1280.,720.]}, 'integer width'),
    ({'image_size': [640,480]}, 'rs_width'),
    ({'K': np.diag([np.nan, 900., 1.])}, 'intrinsic matrix'),
    ({'dist': np.full(8,np.inf)}, 'finite OpenCV'),
    ({'distortion_model': 'inverse_brown_conrady'}, 'distortion model'),
    ({'distortion_model': 'k2'}, 'coefficients disagree'),
    ({'camera': 'D435_DEPTH'}, 'color intrinsic'),
    ({'camera_serial': 'camera-B'}, 'serial'),
])
def test_rejects_bad_intrinsics(profile_path, changes, reason):
    change_npz(profile_path, 'intrinsics', **changes)
    with pytest.raises(ValueError, match=reason):
        HandEyeCalibration.load(profile_path)


def test_source_changes_fail_but_existing_snapshot_survives(profile_path, tmp_path):
    calibration = HandEyeCalibration.load(profile_path)
    frozen = calibration.snapshot(tmp_path/'snapshot')
    original_bytes = calibration.artifacts['hand_eye']
    (profile_path.parent/'hand_eye.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        HandEyeCalibration.load(profile_path)
    relocated = tmp_path/'relocated'
    shutil.move(frozen.parent, relocated)
    loaded = HandEyeCalibration.load(relocated/'profile.yml')
    assert loaded.artifacts['hand_eye'] == original_bytes
    np.testing.assert_array_equal(loaded.tool_T_color, calibration.tool_T_color)
    assert not (relocated/'profile.yml.tmp').exists()
    (relocated/'color_intrinsics.npz').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        HandEyeCalibration.load(relocated/'profile.yml')


def test_profile_rejects_unsupported_schema_and_unpinned_inputs(profile_path):
    values = yaml.safe_load(profile_path.read_text())
    values['schema_version'] = True
    profile_path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match='schema'):
        load_hand_eye_profile(profile_path)
    values['schema_version'] = 1
    values['intrinsics']['sha256'] = 'not-a-digest'
    profile_path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match='SHA-256'):
        load_hand_eye_profile(profile_path)


def test_installation_binding_cannot_override_recorded_serial(profile_path):
    profile = load_hand_eye_profile(profile_path)
    with pytest.raises(ValueError, match='rs_serial'):
        HandEyeCalibration(replace(profile, serial='camera-B'))


def test_snapshot_failure_does_not_publish_loadable_profile(profile_path, tmp_path, monkeypatch):
    calibration = HandEyeCalibration.load(profile_path)
    original_write = Path.write_bytes

    def failed_write(path, data):
        if path.name == 'color_intrinsics.npz':
            raise OSError('storage failed')
        return original_write(path, data)

    monkeypatch.setattr(Path, 'write_bytes', failed_write)
    output = tmp_path/'incomplete'
    with pytest.raises(OSError, match='storage failed'):
        calibration.snapshot(output)
    assert (output/'hand_eye.npz').exists()
    assert not (output/'profile.yml').exists()


def test_malformed_yaml_returns_cli_error(profile_path, capsys):
    profile_path.write_text('schema_version: [\n')
    assert main(['--profile',str(profile_path)]) == 1
    assert 'profile YAML' in capsys.readouterr().err


def test_live_color_checks_use_factory_intrinsics(profile_path):
    calibration = HandEyeCalibration.load(profile_path)
    values = dict(serial='camera-A', width=1280, height=720,
                  factory_matrix=calibration.factory_matrix, factory_distortion=np.zeros(5))
    calibration.validate_color_stream(**values)
    for change in ({'serial':'camera-B'}, {'width':640},
                   {'factory_matrix':calibration.color_matrix},
                   {'factory_distortion':calibration.color_distortion}):
        with pytest.raises(ValueError):
            calibration.validate_color_stream(**(values|change))


def test_compare_does_not_select_or_trust_alternative(profile_path, tmp_path):
    calibration = HandEyeCalibration.load(profile_path)
    with np.load(profile_path.parent/'hand_eye.npz') as z:
        candidate = dict(z)
    candidate['cameraMatrix'] = candidate['cameraMatrix']+np.diag([4.,4.,0.])
    candidate['T_gripper_to_camera'][:3,3] += [.001,0,0]
    alternative = tmp_path/'refined.npz'
    np.savez(alternative, **candidate)
    result = calibration.compare(alternative)
    assert not result['compatible']
    assert result['difference']['translation_m'] == pytest.approx(.001)
    assert 'intrinsics mismatch' in result['reason']
    assert calibration.method == 'Tsai'


def test_cli_audits_without_writes_and_snapshot_refuses_overwrite(profile_path, tmp_path, capsys):
    before = set(tmp_path.iterdir())
    assert main(['--profile',str(profile_path)]) == 0
    assert set(tmp_path.iterdir()) == before
    assert json.loads(capsys.readouterr().out)['status'] == 'offline_consistent'
    output = tmp_path/'bundle'
    assert main(['--profile',str(profile_path),'--output',str(output)]) == 0
    assert HandEyeCalibration.load(output/'profile.yml').width == 1280
    original = (output/'hand_eye.npz').read_bytes()
    assert main(['--profile',str(profile_path),'--output',str(output)]) == 1
    assert (output/'hand_eye.npz').read_bytes() == original
    (profile_path.parent/'hand_eye.npz').write_bytes(b'changed')
    failed = tmp_path/'failed'
    assert main(['--profile',str(profile_path),'--output',str(failed)]) == 1
    assert not failed.exists()
