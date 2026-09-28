"""Serialize rendered scene frames through the normal offline capture contracts."""

from pathlib import Path

import numpy as np

from .calibration import BASE_LINK_T_UR_BASE
from .hand_eye_capture import atomic_json, capture_view
from .hand_eye_replay import assemble_session, inspect_session
from .hand_eye_synthetic import (
    SyntheticCamera, SyntheticClock, calibration_fixture, geometry, synthetic_femto,
)


class RenderedCamera(SyntheticCamera):
    def __init__(self, calibration, world_T_color, joints, rgb, depth, clock):
        self.clock, self.number = clock, 0
        self.rgb = rgb
        self.raw = np.clip(np.rint(depth/.00025), 0, 65535).astype(np.uint16)
        self.joints = np.asarray(joints)
        self.base_T_tool = world_T_color @ np.linalg.inv(calibration.tool_T_color)
        info = geometry(calibration.width, calibration.height)
        info['K'] = calibration.color_matrix.tolist()
        color = dict(info, format='format.rgb8')
        self.report = dict(schema_version=1, data_origin='synthetic',
            device=dict(serial_number='SYNTHETIC-D435', name='Operations triangle scene renderer'),
            streams=dict(color=color, depth=info), depth_metres_per_unit=.00025,
            color_T_depth=np.eye(4).tolist(), accuracy_1mm_verified=False,
            timestamp_mapping='fabricated deterministic sequence; no physical clocks',
            image_model='ideal pinhole RGB and colocated ideal depth; no distortion or sensor noise')

    def read(self):
        self.clock.now += 67_000_000
        self.number += 1
        pair = dict(receipt_ns=self.clock.now+10_000_000,
                    receipt_monotonic_ns=int(self.clock.monotonic()*1e9)+10_000_000)
        for name, image in [('color', self.rgb), ('depth', self.raw)]:
            stamp = self.clock.now
            pair[name] = dict(image=image, frame_number=self.number, timestamp_ns=stamp,
                timestamp_ms=stamp/1e6, timestamp_domain='timestamp_domain.global_time',
                metadata=dict(synthetic=True))
        return pair

    def measure(self, start, end, stamps):
        packet = super().measure(start, end, stamps)
        packet['joint_positions'] = np.tile(self.joints, (len(packet['joint_stamps_ns']), 1)).tolist()
        packet['pose_source'] = 'synthetic exact saved-plan FK; no physical measurement'
        return packet


class SimulatedCapture:
    """A GUI-readable wrist+FEMTO capture, always labelled synthetic."""
    def __init__(self, directory, calibration, plan, femto_depth, femto_pose, femto_info, crop):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.clock = SyntheticClock()
        self.calibration = calibration_fixture(self.directory/'fixture_calibration',
            color_size=(calibration.width, calibration.height), color_matrix=calibration.color_matrix,
            tool_T_color=calibration.tool_T_color)
        self.views = []
        self.femto = dict(depth=femto_depth, pose=femto_pose, info=femto_info,
                          crop_min=crop[0], crop_max=crop[1], voxel_size=.001)
        self.status = dict(schema_version=1, kind='automated_scene_capture', status='running',
            data_origin='synthetic', output_frame='base_link', completed_views=[],
            motion_sent=False, accuracy_1mm_verified=False,
            scene_stability='synthetic_static', camera_model='ideal_raycaster')
        atomic_json(self.directory/'capture.json', self.status)
        atomic_json(self.directory/'executed_plan.json', dict(plan, data_origin='synthetic', motion_sent=False))
        synthetic_femto(self.directory/'femto_before', self.clock, **self.femto)

    def add(self, view, rgb, depth):
        index = len(self.views)
        path = self.directory/'wrist'/f'view_{index:03d}_{view["label"]}'
        world_T_color = BASE_LINK_T_UR_BASE @ np.asarray(view['base_link_T_color'])
        camera = RenderedCamera(self.calibration, world_T_color, view['path'][-1], rgb, depth, self.clock)
        capture_view(path, self.calibration, camera, camera, count=3, clock=self.clock)
        self.views.append(path)
        self.status['completed_views'].append(str(path.relative_to(self.directory)))
        atomic_json(self.directory/'capture.json', self.status)

    def finish(self):
        synthetic_femto(self.directory/'femto_after', self.clock, **self.femto)
        assemble_session(self.directory/'session', self.views,
                         femto_before=self.directory/'femto_before', femto_after=self.directory/'femto_after')
        inspect_session(self.directory/'session')
        self.status.update(status='complete', stage='complete')
        atomic_json(self.directory/'capture.json', self.status)
        return self.directory
