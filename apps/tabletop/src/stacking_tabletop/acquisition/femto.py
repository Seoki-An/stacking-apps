"""Bounded, cancellable FEMTO depth bursts independent of ROS message classes."""

from collections import deque
import copy
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import tempfile
import threading
import time

import numpy as np

from stacking_common.pointcloud import validate_rigid_transform
from stacking_tabletop.perception.depth import load_femto_depth_calibration
from stacking_tabletop.perception.organized_depth import camera_rays
from .calibration import load_camera_calibration, BASE_LINK_T_UR_BASE, camera_cloud_transform_difference

DEPTH_MODE = 'WFOV_UNBINNED_1024x1024_15'
DEPTH_SHAPE = (1024, 1024)
SYNC_NS = 30_000_000


class CaptureError(RuntimeError):
    pass


class CaptureCancelled(CaptureError):
    pass


class FemtoCalibration:
    """Parse immutable snapshots, so later file edits cannot alter this capture."""
    def __init__(self, profile):
        self.profile = profile
        self.artifacts = {
            'camera_pose.npz': profile.pose_calibration.read_bytes(),
            'camera_depth.npz': profile.depth_calibration.read_bytes(),
        }
        self.hashes = {name: hashlib.sha256(data).hexdigest() for name, data in self.artifacts.items()}
        with tempfile.TemporaryDirectory(prefix='tabletop-femto-') as directory:
            paths = {}
            for name, data in self.artifacts.items():
                paths[name] = Path(directory) / name
                paths[name].write_bytes(data)
            self.depth = load_femto_depth_calibration(paths['camera_depth.npz'],
                expected_camera_serial=profile.serial, expected_depth_mode=DEPTH_MODE)
            self.pose = load_camera_calibration(paths['camera_pose.npz'], paths['camera_depth.npz'], profile.serial)
        if self.pose.parent_frame != profile.base_frame:
            raise ValueError('capture base frame must match the calibrated parent frame')
        with np.load(io.BytesIO(self.artifacts['camera_pose.npz']), allow_pickle=False) as data:
            base_T_color = validate_rigid_transform(data['T_base_to_observation'])
        with np.load(io.BytesIO(self.artifacts['camera_depth.npz']), allow_pickle=False) as data:
            depth_T_color = data['capture_depth_T_color'][0]
        if 'legacy_ur_base_to_base_link' in self.pose.correction:
            base_T_color = BASE_LINK_T_UR_BASE @ base_T_color
        self.base_T_depth = validate_rigid_transform(base_T_color @ np.linalg.inv(depth_T_color))

    def transform(self, camera_T_depth):
        # Check the live driver's factory optical frame against the calibration
        # recorded with depth enabled; never use a color-only camera_link origin.
        error = camera_cloud_transform_difference(self.pose.transform, camera_T_depth, self.base_T_depth)
        if error.translation_m > .0001 or error.rotation_deg > .01:
            raise CaptureError('live camera optical transform differs from calibration')
        return self.base_T_depth.copy()


@dataclass(frozen=True)
class FemtoFrame:
    sequence: int
    depth_stamp_ns: int
    cloud_stamp_ns: int
    received_at: float
    frame_id: str
    encoding: str
    depth: np.ndarray
    points: np.ndarray
    camera_info: dict

    @property
    def depth_unit(self):
        return 1.0 if self.encoding.lower() == '32fc1' else .001


class FemtoBuffer:
    """Bounded immutable snapshots; freshness uses local receipt time."""
    def __init__(self, calibration, *, clock=time.monotonic):
        self.calibration, self.clock = calibration, clock
        self._lock = threading.Lock()
        self._frames = deque(maxlen=16)
        self._sequence = 0
        self._last_depth = self._last_cloud = 0
        self._encoding = self._info = self._fault = None

    def fail(self, error):
        with self._lock:
            if self._fault is None:
                self._fault = str(error)

    def push(self, *, depth_stamp_ns, cloud_stamp_ns, frame_id, encoding, depth, points, camera_info):
        if min(depth_stamp_ns, cloud_stamp_ns) <= 0:
            raise CaptureError('capture timestamps must be positive')
        if abs(depth_stamp_ns - cloud_stamp_ns) > SYNC_NS:
            return False
        if frame_id != self.calibration.depth.depth_frame or camera_info['frame_id'] != frame_id:
            raise CaptureError('depth, cloud, intrinsics and calibration frames must agree')
        if encoding.lower() not in ('16uc1', 'mono16', '32fc1'):
            raise CaptureError('unsupported native depth encoding')
        depth = np.asarray(depth)
        if depth.shape != DEPTH_SHAPE or (camera_info['height'], camera_info['width']) != DEPTH_SHAPE:
            raise CaptureError('FEMTO requires 1024x1024 WFOV unbinned depth')
        expected = 'f' if encoding.lower() == '32fc1' else 'u'
        size = 4 if expected == 'f' else 2
        if depth.dtype.kind != expected or depth.dtype.itemsize != size:
            raise CaptureError('depth dtype does not match its encoding')
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise CaptureError('cloud must contain XYZ points')
        points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
        if not len(points):
            return False
        with self._lock:
            if depth_stamp_ns <= self._last_depth or cloud_stamp_ns <= self._last_cloud:
                return False
            if self._encoding is not None and self._encoding != encoding.lower():
                raise CaptureError('depth encoding changed during capture')
            if self._info is not None and self._info != camera_info:
                raise CaptureError('depth intrinsics changed during capture')
            if self._info is None:
                if not np.isfinite(camera_info['k']).all() or not np.isfinite(camera_info['d']).all():
                    raise CaptureError('depth intrinsics must be finite')
                if camera_info.get('binning_x', 0) not in (0, 1) or camera_info.get('binning_y', 0) not in (0, 1):
                    raise CaptureError('FEMTO depth must be unbinned')
                roi = camera_info.get('roi', {})
                if roi.get('x_offset', 0) or roi.get('y_offset', 0) or roi.get('width', 0) not in (0, 1024) or roi.get('height', 0) not in (0, 1024):
                    raise CaptureError('cropped depth CameraInfo is unsupported')
                rays = camera_rays(camera_info)
                if not all(np.isfinite(ray).all() for ray in rays):
                    raise CaptureError('invalid calibrated camera rays')
            self._info = copy.deepcopy(camera_info)
            self._encoding = encoding.lower()
            self._last_depth, self._last_cloud = depth_stamp_ns, cloud_stamp_ns
            self._sequence += 1
            depth, points = depth.copy(), points.copy()
            depth.setflags(write=False)
            points.setflags(write=False)
            self._frames.append(FemtoFrame(self._sequence, depth_stamp_ns, cloud_stamp_ns,
                self.clock(), frame_id, encoding, depth, points, self._info))
        return True

    def cursor(self):
        with self._lock:
            return self._sequence

    def check(self):
        with self._lock:
            if self._fault is not None:
                raise CaptureError(self._fault)

    def after(self, sequence):
        with self._lock:
            if self._fault is not None:
                raise CaptureError(self._fault)
            return tuple(frame for frame in self._frames if frame.sequence > sequence)


@dataclass(frozen=True)
class CapturedFrame:
    frame: FemtoFrame
    base_T_cloud: np.ndarray
    robot: dict | None = None


class FemtoCamera:
    """Acquire after a new capture request, including warmup and exact-time TF."""
    def __init__(self, transport, calibration, *, cancel_event=None, clock=time.monotonic, sleep=time.sleep):
        self.transport, self.calibration = transport, calibration
        self.cancel = cancel_event if cancel_event is not None else threading.Event()
        self.clock, self.sleep = clock, sleep
        self._capturing = threading.Lock()

    def capture(self, *, in_hand=False, check=None, timeout=60., progress=None):
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('capture timeout must be finite and positive')
        if not self._capturing.acquire(blocking=False):
            raise CaptureError('a camera burst is already active')
        try:
            deadline = self.clock() + timeout
            cursor = self.transport.buffer.cursor()
            def health():
                self._check()
                if check is not None:
                    check()
            # Verify actual device identity, not just the selected YAML/NPZ.
            self.transport.verify_serial(self.calibration.profile.serial, deadline, health)
            cursor = self.transport.buffer.cursor()  # Ignore pre-identity frames.
            seen = 0
            last_stamp = 0
            captured = []
            waiting_for = 'fresh synchronized depth/cloud frames'
            stale = 0
            while len(captured) < 7:
                health()
                if self.clock() >= deadline:
                    diagnostics = getattr(self.transport, 'diagnostics', lambda: {})()
                    details = ', '.join(f'{key}={value}' for key, value in diagnostics.items())
                    raise CaptureError(
                        f'FEMTO capture timed out; last wait: {waiting_for}; '
                        f'fresh_pairs={seen}, stale_pairs={stale}, captured={len(captured)}/7'
                        + (f'; transport since connection: {details}' if details else ''))
                for frame in self.transport.buffer.after(cursor):
                    cursor = frame.sequence
                    if self.clock() - frame.received_at > .5:
                        stale += 1
                        waiting_for = 'fresh frames (received pairs are stale)'
                        continue
                    seen += 1
                    if seen <= 10:
                        waiting_for = f'warmup frames ({seen}/10)'
                        continue
                    if frame.cloud_stamp_ns - last_stamp < 200_000_000:
                        waiting_for = 'next exposure at least 200 ms after the last capture'
                        continue
                    factory = self.transport.lookup(self.calibration.pose.child_frame, frame.frame_id, frame.cloud_stamp_ns)
                    if factory is None:
                        waiting_for = (f'exact transform {self.calibration.pose.child_frame}'
                                       f' <- {frame.frame_id} at the exposure timestamp')
                        continue
                    base = self.calibration.transform(factory)
                    robot = self.transport.robot_at(frame.cloud_stamp_ns, self.calibration.profile.base_frame) if in_hand else None
                    if in_hand and robot is None:
                        waiting_for = 'timestamp-matched robot/gripper state and tool transform'
                        continue
                    if captured and robot is not None:
                        initial = captured[0].robot
                        if np.max(np.abs(np.asarray(robot['joints']) - initial['joints'])) > np.deg2rad(.25):
                            raise CaptureError('robot moved during settled-view depth burst')
                        if abs(robot['gripper_position'] - initial['gripper_position']) > 5:
                            raise CaptureError('gripper moved during settled-view depth burst')
                        diff = camera_cloud_transform_difference(initial['base_T_tool'], np.eye(4), robot['base_T_tool'])
                        if diff.translation_m > .0005 or diff.rotation_deg > .05:
                            raise CaptureError('tool transform moved during settled-view depth burst')
                    captured.append(CapturedFrame(frame, base, robot))
                    last_stamp = frame.cloud_stamp_ns
                    waiting_for = 'next fresh synchronized depth/cloud frame'
                    if progress is not None:
                        progress(len(captured), 7)
                    if len(captured) == 7:
                        break
                self.sleep(.01)
            health()
            return tuple(captured)
        finally:
            self._capturing.release()

    def _check(self):
        if self.cancel.is_set():
            raise CaptureCancelled('FEMTO capture cancelled')
        if not self.transport.ok():
            raise CaptureError('FEMTO ROS transport stopped')
        self.transport.buffer.check()
