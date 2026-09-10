"""Predefined FEMTO identities and finite field capture bounds."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class FemtoProfile:
    serial: str
    pose_calibration: Path
    depth_calibration: Path
    namespace: str
    base_frame: str
    crop_min: tuple[float, float, float]
    crop_max: tuple[float, float, float]
    voxel_size: float

    def __post_init__(self):
        if not self.serial.strip() or not self.base_frame.strip():
            raise ValueError('camera serial and base frame are required')
        if not self.namespace.startswith('/') or self.namespace == '/':
            raise ValueError('camera namespace must be an absolute ROS namespace')
        lower, upper = np.asarray(self.crop_min), np.asarray(self.crop_max)
        if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all() or np.any(lower >= upper):
            raise ValueError('field crop must have finite ordered XYZ bounds')
        if not np.isfinite(self.voxel_size) or self.voxel_size <= 0:
            raise ValueError('field voxel size must be positive')
        object.__setattr__(self, 'namespace', self.namespace.rstrip('/'))
        for name in ('pose_calibration', 'depth_calibration'):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())


def load_femto_profile(path, camera='primary'):
    path = Path(path).expanduser().resolve()
    values = yaml.safe_load(path.read_text())
    identity = dict(values['cameras'][camera])
    for name in ('pose_calibration', 'depth_calibration'):
        asset = Path(identity[name]).expanduser()
        identity[name] = asset if asset.is_absolute() else path.parent / asset
    return FemtoProfile(**identity, namespace=values['namespace'], base_frame=values['base_frame'],
        crop_min=tuple(values['field']['crop_min']), crop_max=tuple(values['field']['crop_max']),
        voxel_size=float(values['field']['voxel_size']))
