"""Native capture artifacts consumed by tabletop field and in-hand processing."""

import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from stacking_common.pointcloud import PointCloudData
from stacking_common.pointcloud.processing import fixed_grid_centroids
from stacking_tabletop.execution.motion import UR_JOINT_NAMES
from stacking_tabletop.perception.depth import correct_points_along_rays
from stacking_tabletop.perception.organized_depth import DepthFilterConfig, filter_depth_burst, camera_rays, depth_to_points


def write_xyz_pcd(path, points):
    points = np.asarray(points, dtype='<f4')
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('PCD requires finite XYZ points')
    header = ('# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n'
              f'WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA binary\n')
    with Path(path).open('xb') as stream:
        stream.write(header.encode('ascii'))
        stream.write(points.tobytes())


def _json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class CaptureSession:
    """A new directory only; incomplete captures never acquire complete status."""
    def __init__(self, directory, calibration, *, in_hand=False):
        self.directory = Path(directory).expanduser().resolve()
        self.calibration, self.in_hand = calibration, in_hand
        self.views, self.rows = [], []
        self._finished = False
        self.directory.mkdir(parents=True, exist_ok=False)
        for name in ('raw_pcd', 'organized_depth', 'calibration_artifacts'):
            (self.directory / name).mkdir()
        for name, data in calibration.artifacts.items():
            (self.directory / 'calibration_artifacts' / name).write_bytes(data)
        self.metadata = {
            'version': 1, 'status': 'running', 'camera_serial': calibration.profile.serial,
            'depth_mode': calibration.depth.depth_mode, 'cloud_frame': calibration.depth.depth_frame,
            'base_frame': calibration.profile.base_frame,
            'depth_scale': calibration.depth.scale, 'depth_offset_m': calibration.depth.offset,
            'camera_calibration_file': 'calibration_artifacts/camera_pose.npz',
            'depth_calibration_file': 'calibration_artifacts/camera_depth.npz',
            'camera_calibration_sha256': calibration.hashes['camera_pose.npz'],
            'depth_calibration_sha256': calibration.hashes['camera_depth.npz'],
            'camera_transform_source': 'pose_calibration_and_verified_factory_optical_tf',
            'camera_calibration_correction': calibration.pose.correction,
            'raw_shots_depth_corrected': False,
        }
        _json(self.directory / 'capture_status.json', {'status': 'running'})
        _json(self.directory / 'camera_frame.json', self.metadata)

    def add(self, burst):
        if self._finished:
            raise ValueError('capture session already finished')
        if len(burst) != 7:
            raise ValueError('a FEMTO view requires seven synchronized frames')
        first = burst[0]
        index = len(self.views)
        if not self.in_hand and index:
            raise ValueError('a fixed field capture contains one view')
        if any((item.robot is not None) != self.in_hand for item in burst):
            raise ValueError('capture robot metadata does not match session type')
        frame = first.frame
        for item in burst:
            if (item.frame.frame_id != self.calibration.depth.depth_frame
                    or item.frame.camera_info != frame.camera_info
                    or item.frame.encoding != frame.encoding
                    or not np.allclose(item.base_T_cloud, self.calibration.base_T_depth, atol=1e-8, rtol=0)):
                raise ValueError('burst stream or transform differs from session calibration')
        if any(b.frame.depth_stamp_ns <= a.frame.depth_stamp_ns or b.frame.cloud_stamp_ns <= a.frame.cloud_stamp_ns
               for a, b in zip(burst, burst[1:])):
            raise ValueError('burst exposure timestamps must strictly increase')
        view_name = f'view_{index:03d}.pcd'
        records, shots = [], []
        for shot_index, item in enumerate(burst):
            stem = f'view_{index:03d}_shot_{shot_index:02d}_{item.frame.depth_stamp_ns}'
            np.save(self.directory / 'organized_depth' / (stem + '.npy'), item.frame.depth, allow_pickle=False)
            write_xyz_pcd(self.directory / 'raw_pcd' / (stem + '.pcd'), item.frame.points)
            record = {'filename': f'organized_depth/{stem}.npy', 'stamp_ns': item.frame.depth_stamp_ns,
                      'cloud_stamp_ns': item.frame.cloud_stamp_ns}
            records.append(record)
            shots.append({'filename': stem + '.pcd', 'stamp_ns': item.frame.cloud_stamp_ns,
                'cloud_frame': frame.frame_id, 'transform_status': 'exact',
                'base_T_cloud': item.base_T_cloud.tolist(), 'organized_depth_filename': record['filename']})
        _json(self.directory / 'depth_camera_info.json', frame.camera_info)
        median, keep, stats = filter_depth_burst(tuple(item.frame.depth for item in burst), frame.depth_unit, DepthFilterConfig())
        rays = camera_rays(frame.camera_info)
        points = depth_to_points(median, keep, *rays)
        if not len(points):
            raise ValueError('temporal depth filtering removed every point')
        if self.in_hand:
            # Raw optical representative. Correction happens exactly once in
            # TabletopRawCapture, after reprojecting the original depth burst.
            write_xyz_pcd(self.directory / 'raw_pcd' / view_name, points)
            robot = first.robot
            row = {'filename': view_name, 'view_index': index, 'cloud_frame': frame.frame_id,
                   'transform_status': 'exact', 'stamp_ns': frame.cloud_stamp_ns,
                   'gripper_position': robot['gripper_position'],
                   'joint_stamp_ns': robot['joint_stamp_ns'], 'gripper_stamp_ns': robot['gripper_stamp_ns'],
                   'joint_sample_mode': robot['joint_sample_mode'],
                   **dict(zip(UR_JOINT_NAMES, map(float, robot['joints'])))}
            for prefix, matrix in (('base_T_cloud', first.base_T_cloud), ('base_T_tool', robot['base_T_tool'])):
                row.update({f'{prefix}_{i}{j}': float(matrix[i, j]) for i in range(4) for j in range(4)})
            self.rows.append(row)
            with (self.directory / 'frames.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(self.rows)
            representative = view_name
        else:
            corrected = correct_points_along_rays(points, self.calibration.depth.scale, self.calibration.depth.offset)
            transformed = corrected @ first.base_T_cloud[:3, :3].T + first.base_T_cloud[:3, 3]
            profile = self.calibration.profile
            inside = np.all((transformed >= profile.crop_min) & (transformed <= profile.crop_max), axis=1)
            if not np.any(inside):
                raise ValueError('field crop removed every point')
            fused = fixed_grid_centroids(PointCloudData(transformed[inside]), profile.voxel_size).points
            representative = f'field_fused_{profile.base_frame}.pcd'
            write_xyz_pcd(self.directory / representative, fused)
            self.field_metadata = {**self.metadata, 'version': 5, 'status': 'complete',
                'output_frame': profile.base_frame, 'capture_count': 7, 'warmup_frames': 10,
                'capture_period': .2, 'merge_voxel_size': profile.voxel_size,
                'base_frame_crop': {'min': list(profile.crop_min), 'max': list(profile.crop_max)},
                'fused_depth_corrected': True, 'fused_source': 'organized_depth_burst',
                'fused_file': representative, 'fused_point_count': len(fused), 'shots': shots,
                'organized_depth_manifest': 'organized_depth_manifest.json',
                'organized_depth_filter': asdict(stats)}
        self.views.append({'view_index': index, 'representative_pcd': representative, 'depth_frames': records})
        _json(self.directory / 'organized_depth_manifest.json', {
            'depth_mode': self.calibration.depth.depth_mode, 'encoding': frame.encoding,
            'depth_unit_m': frame.depth_unit, 'values_are_raw': True,
            'depth_calibration_file': self.metadata['depth_calibration_file'],
            'camera_info': 'depth_camera_info.json', 'views': self.views})

    def finish(self):
        if self._finished:
            return self.directory
        if not self.views:
            raise ValueError('cannot complete an empty capture')
        if not self.in_hand:
            _json(self.directory / 'field_scan.json', self.field_metadata)
        _json(self.directory / 'camera_frame.json', {**self.metadata, 'status': 'complete'})
        _json(self.directory / 'capture_status.json', {'status': 'complete', 'view_count': len(self.views)})
        self._finished = True
        return self.directory

    def __enter__(self):
        return self

    def __exit__(self, kind, error, _traceback):
        if kind is None:
            try:
                self.finish()
            except Exception as failure:
                _json(self.directory / 'capture_status.json', {'status': 'failed', 'reason': str(failure)})
                raise
        else:
            _json(self.directory / 'capture_status.json', {'status': 'failed', 'reason': str(error)})
