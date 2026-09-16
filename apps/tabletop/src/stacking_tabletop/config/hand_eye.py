"""Explicit, content-pinned PoseID calibration inputs for the wrist D435."""

from dataclasses import dataclass
from pathlib import Path
import re

import yaml


@dataclass(frozen=True)
class CalibrationArtifact:
    path: Path
    sha256: str

    def __post_init__(self):
        object.__setattr__(self, 'path', Path(self.path).expanduser().resolve())
        if not isinstance(self.sha256, str) or not re.fullmatch('[0-9a-f]{64}', self.sha256):
            raise ValueError('calibration artifacts require a lowercase SHA-256 digest')


@dataclass(frozen=True)
class HandEyeProfile:
    serial: str
    color_frame: str
    base_frame: str
    tool_frame: str
    robot_pose_source: str
    hand_eye: CalibrationArtifact
    intrinsics: CalibrationArtifact
    installation: CalibrationArtifact

    def __post_init__(self):
        for name in ('serial', 'color_frame', 'base_frame', 'tool_frame'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f'{name} must be a nonempty frame/identity string')
        if self.base_frame not in ('base', 'base_link') or self.tool_frame != 'tool0':
            raise ValueError('tabletop hand-eye requires UR base/base_link and physical tool0')
        if self.color_frame in (self.base_frame, self.tool_frame):
            raise ValueError('color optical frame must differ from robot frames')
        if self.robot_pose_source not in ('tf', 'custom'):
            raise ValueError('robot pose source must be tf or custom')


def load_hand_eye_profile(path):
    path = Path(path).expanduser().resolve()
    try:
        values = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ValueError(f'invalid hand-eye profile YAML: {error}') from error
    if not isinstance(values, dict) or type(values.get('schema_version')) is not int or values['schema_version'] != 1:
        raise ValueError('unsupported hand-eye profile schema')
    values = dict(values)
    values.pop('schema_version')
    for name in ('hand_eye', 'intrinsics', 'installation'):
        record = values.get(name)
        if not isinstance(record, dict) or set(record) != {'path', 'sha256'}:
            raise ValueError(f'{name} requires path and sha256')
        if not isinstance(record['path'], str) or not record['path'].strip():
            raise ValueError(f'{name} path must be a nonempty string')
        asset = Path(record['path']).expanduser()
        values[name] = CalibrationArtifact(
            asset if asset.is_absolute() else path.parent / asset, record['sha256'])
    try:
        return HandEyeProfile(**values)
    except TypeError as error:
        raise ValueError(f'invalid hand-eye profile: {error}') from error
