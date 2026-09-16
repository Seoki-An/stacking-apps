"""Validate and snapshot a coherent staged PoseID hand-eye/color calibration.

This module reads files only. Installation binding is explicitly distinguished
from a live camera check and from independently measured physical accuracy.
"""

from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import yaml

from stacking_common.pointcloud import validate_rigid_transform
from stacking_tabletop.config.hand_eye import load_hand_eye_profile
from .calibration import BASE_LINK_T_UR_BASE, camera_cloud_transform_difference


ARTIFACT_NAMES = {'hand_eye': 'hand_eye.npz', 'intrinsics': 'color_intrinsics.npz',
                  'installation': 'installation.yml'}
STAGED_METHODS = {'Tsai', 'Park', 'Horaud', 'Andreff', 'Daniilidis'}


def _npz(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _text(data, key):
    value = np.asarray(data[key])
    if value.shape != () or value.dtype.kind not in 'US' or not str(value.item()).strip():
        raise ValueError(f'{key} must be a nonempty scalar string')
    return str(value.item())


def _matrix(value, name):
    value = np.asarray(value, dtype=float)
    if (value.shape != (3, 3) or not np.isfinite(value).all()
            or value[0, 0] <= 0 or value[1, 1] <= 0
            or not np.allclose(value[2], [0, 0, 1], rtol=0, atol=1e-8)
            or not np.allclose([value[0, 1], value[1, 0]], 0, rtol=0, atol=1e-8)):
        raise ValueError(f'{name} must be a finite pinhole intrinsic matrix')
    return value


def _distortion(value):
    value = np.asarray(value, dtype=float)
    if value.shape not in ((4,), (5,), (8,)) or not np.isfinite(value).all():
        raise ValueError('distortion requires 4, 5 or 8 finite OpenCV coefficients')
    # OpenCV k2/k3 exports can carry zero-padded rational coefficients.
    return np.pad(value, (0, 8 - value.size))


def _same(left, right, name):
    if np.shape(left) != np.shape(right) or not np.allclose(left, right, rtol=0, atol=1e-6):
        raise ValueError(f'{name} mismatch; select artifacts from the same calibration solve')


def _readonly(value):
    value = np.array(value, copy=True)
    value.setflags(write=False)
    return value


class HandEyeCalibration:
    def __init__(self, profile):
        self.profile = profile
        raw = {}
        for name in ARTIFACT_NAMES:
            artifact = getattr(profile, name)
            raw[name] = artifact.path.read_bytes()
            if hashlib.sha256(raw[name]).hexdigest() != artifact.sha256:
                raise ValueError(f'{name} SHA-256 mismatch; source changed since explicit selection')
        self.artifacts = MappingProxyType(raw)
        try:
            intrinsic = _npz(raw['intrinsics'])
            hand_eye = _npz(raw['hand_eye'])
            self._load_intrinsics(intrinsic)
            transform = self._validate_hand_eye(hand_eye)
            self._validate_installation(yaml.safe_load(raw['installation']))
            self._validate_identity(intrinsic)
        except KeyError as error:
            raise ValueError(f'missing calibration metadata: {error.args[0]}') from error
        except yaml.YAMLError as error:
            raise ValueError(f'invalid installation YAML: {error}') from error
        self.tool_T_color = _readonly(transform)
        self.method = _text(hand_eye, 'method')

    @classmethod
    def load(cls, path):
        return cls(load_hand_eye_profile(path))

    def _load_intrinsics(self, data):
        if _text(data, 'camera') != 'D435_COLOR':
            raise ValueError('expected a D435 color intrinsic artifact')
        if _text(data, 'method') != 'd435_handeye_log_charuco_intrinsic':
            raise ValueError('unsupported color intrinsic producer')
        self.distortion_model = _text(data, 'distortion_model')
        if self.distortion_model not in ('k2', 'k3', 'rational'):
            raise ValueError('unsupported fitted color distortion model')
        size = np.asarray(data['image_size'])
        if size.shape != (2,) or size.dtype.kind not in 'iu' or np.any(size <= 0):
            raise ValueError('image_size must contain positive integer width and height')
        self.width, self.height = map(int, size)
        self.color_matrix = _readonly(_matrix(data['K'], 'color K'))
        self.color_distortion = _readonly(_distortion(data['dist']))
        if (self.distortion_model in ('k2', 'k3') and np.any(self.color_distortion[5:] != 0)
                or self.distortion_model == 'k2' and self.color_distortion[4] != 0):
            raise ValueError('coefficients disagree with the declared distortion model')
        self.factory_matrix = _readonly(_matrix(data['K_factory'], 'factory K'))
        self.factory_distortion = _readonly(_distortion(data['dist_factory']))

    def _validate_identity(self, data):
        # Older PoseID exports omit these fields. The report explicitly retains
        # that limitation; present fields are never overridden by configuration.
        for key in ('camera_serial', 'rs_serial'):
            if key in data and _text(data, key) != self.profile.serial:
                raise ValueError('artifact camera serial differs from installation binding')
        for key in ('observation_frame', 'color_frame'):
            if key in data and _text(data, key) != self.profile.color_frame:
                raise ValueError('artifact color optical frame differs from profile')

    def _validate_hand_eye(self, data):
        _same(_matrix(data['cameraMatrix'], 'hand-eye K'), self.color_matrix, 'color intrinsics')
        _same(_distortion(data['distCoeffs']), self.color_distortion, 'color distortion')
        if 'image_size' in data:
            _same(data['image_size'], [self.width, self.height], 'hand-eye image size')
        if 'distortion_model' in data and _text(data, 'distortion_model') != self.distortion_model:
            raise ValueError('hand-eye distortion model differs from the intrinsic artifact')
        for key, expected in (
                ('robot_pose_parent_frame', self.profile.base_frame),
                ('robot_pose_child_frame', self.profile.tool_frame),
                ('robot_pose_source', self.profile.robot_pose_source)):
            if _text(data, key) != expected:
                raise ValueError(f'{key} differs from the configured robot pose contract')
        transform = validate_rigid_transform(data['T_gripper_to_camera'])
        _same(validate_rigid_transform(data['T_EE_to_cam1']), transform, 'hand-eye transform aliases')
        self._validate_identity(data)
        if _text(data, 'method') not in STAGED_METHODS or 'refined_from' in data:
            raise ValueError('joint-refined/unknown export has no supported solver-provenance contract; '
                             'copied staged residuals cannot validate it')
        selected = 'T_gripper_to_camera_' + self.profile.robot_pose_source
        if selected in data:
            _same(validate_rigid_transform(data[selected]), transform, 'selected robot-source transform')
        return transform

    def _validate_installation(self, data):
        if not isinstance(data, dict):
            raise ValueError('installation binding must contain the PoseID YAML profile')
        node = data.get('handeye_calib')
        if not isinstance(node, dict) or not isinstance(node.get('ros__parameters'), dict):
            raise ValueError('installation binding requires handeye_calib.ros__parameters')
        params = node['ros__parameters']
        for key, expected in (('rs_serial', self.profile.serial),
                              ('rs_width', self.width), ('rs_height', self.height),
                              ('tf_base_frame', self.profile.base_frame),
                              ('tf_tool_frame', self.profile.tool_frame),
                              ('robot_pose_source', self.profile.robot_pose_source)):
            if params[key] != expected:
                raise ValueError(f'installation binding {key} differs from selected calibration')

    def validate_color_stream(self, *, serial, width, height, factory_matrix, factory_distortion):
        """Check live identity/geometry without substituting fitted K for factory K.

        The caller must read these from the device's native color stream. This
        does not validate depth geometry, timing, robot FK or physical accuracy.
        """
        if (serial, width, height) != (self.profile.serial, self.width, self.height):
            raise ValueError('live color serial or native image size differs from calibration')
        _same(_matrix(factory_matrix, 'live factory K'), self.factory_matrix, 'live factory intrinsics')
        _same(_distortion(factory_distortion), self.factory_distortion, 'live factory distortion')

    def base_link_T_color(self, base_T_tool):
        """Compose a measured transform in the declared robot base, in metres."""
        base_T_tool = validate_rigid_transform(base_T_tool)
        base_link_T_base = BASE_LINK_T_UR_BASE if self.profile.base_frame == 'base' else np.eye(4)
        return base_link_T_base @ base_T_tool @ self.tool_T_color

    def report(self):
        return {
            'schema_version': 1, 'status': 'offline_consistent',
            'camera_model': 'D435', 'configured_serial': self.profile.serial,
            'identity_provenance': 'configured_poseid_installation; not a live-device verification',
            'color_frame': self.profile.color_frame,
            'color_frame_provenance': 'application optical-frame label for PoseID color coordinates',
            'robot_base_frame': self.profile.base_frame, 'tool_frame': self.profile.tool_frame,
            'robot_pose_source': self.profile.robot_pose_source,
            'transform_convention': 'A_T_B maps B coordinates into A; translations in metres',
            'tool_T_color': self.tool_T_color.tolist(), 'method': self.method,
            'color': {'width': self.width, 'height': self.height,
                      'K': self.color_matrix.tolist(), 'dist': self.color_distortion.tolist(),
                      'distortion_convention': 'opencv_brown_conrady',
                      'fit_model': self.distortion_model},
            'artifacts': {name: {'sha256': getattr(self.profile, name).sha256}
                          for name in ARTIFACT_NAMES},
            'remaining_checks': ['live_device_identity_and_native_stream',
                                 'native_depth_geometry_units_and_color_extrinsics',
                                 'exposure_clock_mapping_and_robot_model',
                                 'independent_physical_accuracy'],
            'accuracy_1mm_verified': False,
        }

    def compare(self, path):
        """Diagnose an alternative; never select it or consume its residuals."""
        raw = Path(path).expanduser().read_bytes()
        result = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
        try:
            data = _npz(raw)
            alternative = validate_rigid_transform(data['T_gripper_to_camera'])
            difference = camera_cloud_transform_difference(self.tool_T_color, np.eye(4), alternative)
            result['difference'] = asdict(difference)
            self._validate_hand_eye(data)
            result['compatible'] = True
        except (KeyError, ValueError) as error:
            result.update(compatible=False, reason=str(error))
        return result

    def snapshot(self, directory, *, comparisons=()):
        """Write exact validated bytes; profile.yml is published last."""
        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=False)
        values = asdict(self.profile)
        values['schema_version'] = 1
        for name, filename in ARTIFACT_NAMES.items():
            (directory / filename).write_bytes(self.artifacts[name])
            values[name]['path'] = filename
        report = self.report()
        report['comparisons'] = list(comparisons)
        report['source_paths'] = {name: str(getattr(self.profile, name).path) for name in ARTIFACT_NAMES}
        (directory / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary = directory / 'profile.yml.tmp'
        temporary.write_text(yaml.safe_dump(values, sort_keys=False))
        temporary.replace(directory / 'profile.yml')
        return directory / 'profile.yml'
