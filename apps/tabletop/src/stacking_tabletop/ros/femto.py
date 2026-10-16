"""FEMTO ROS acquisition only: no robot commands, serial bus or driver launch."""

from collections import deque
import threading
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from stacking_tabletop.acquisition.femto import FemtoBuffer, CaptureError
from stacking_tabletop.execution.motion import canonical_joint_positions


# Native depth/cloud delivery can lag exposure by about 0.9 s. Keep the
# exposure-time robot evidence longer than that delay; matching stays at 50 ms.
ROBOT_HISTORY_SECONDS = 2.


def stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def decode_depth(message):
    """Decode native Image bytes, including padded rows and byte order."""
    encoding = message.encoding.lower()
    if encoding not in ('16uc1', 'mono16', '32fc1'):
        raise CaptureError('unsupported depth Image encoding')
    dtype = np.dtype(('>' if message.is_bigendian else '<') + ('f4' if encoding == '32fc1' else 'u2'))
    if message.height <= 0 or message.width <= 0 or message.step < message.width * dtype.itemsize:
        raise CaptureError('invalid depth Image dimensions/step')
    if len(message.data) < message.step * message.height:
        raise CaptureError('truncated depth Image data')
    return np.ndarray((message.height, message.width), dtype=dtype, buffer=message.data,
                      strides=(message.step, dtype.itemsize)).copy()


def camera_info(message):
    return {'frame_id': message.header.frame_id, 'width': message.width, 'height': message.height,
            'distortion_model': message.distortion_model, 'k': list(message.k), 'd': list(message.d),
            'r': list(message.r), 'p': list(message.p),
            'binning_x': message.binning_x, 'binning_y': message.binning_y,
            'roi': {'x_offset': message.roi.x_offset, 'y_offset': message.roi.y_offset,
                    'height': message.roi.height, 'width': message.roi.width,
                    'do_rectify': message.roi.do_rectify}}


def transform_matrix(message):
    result = np.eye(4)
    value = message.transform
    result[:3, :3] = Rotation.from_quat([value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w]).as_matrix()
    result[:3, 3] = [value.translation.x, value.translation.y, value.translation.z]
    return result


class FemtoRosTransport:
    """Own a private ROS context or borrow an already-spun node."""
    def __init__(self, calibration, *, node=None, bindings=None, clock=time.monotonic):
        api = _bindings() if bindings is None else bindings
        self.api, self.clock = api, clock
        self.buffer = FemtoBuffer(calibration, clock=clock)
        self._owns_node = node is None
        self._context = self.executor = self.thread = self.listener = None
        self._closed = False
        self._subscriptions, self._clients = [], []
        self._lock = threading.Lock()
        self._depths, self._clouds = deque(maxlen=16), deque(maxlen=16)
        self._joints, self._gripper = deque(maxlen=2000), deque(maxlen=2000)
        self._robot_match = {}
        self._info = None
        self._received = {'depth_messages': 0, 'cloud_messages': 0, 'matched_pairs': 0}
        self._error = None
        if node is None:
            self._context = api.Context()
            api.rclpy.init(context=self._context)
            try:
                node = api.Node('tabletop_femto_capture', context=self._context)
            except Exception:
                self._context.try_shutdown()
                raise
        self.node = node
        try:
            self.tf = api.Buffer(node=node)
            self.listener = api.TransformListener(self.tf, node, spin_thread=False)
            namespace = calibration.profile.namespace
            for kind, topic, callback in (
                (api.Image, namespace + '/depth/image_raw', self._depth),
                (api.PointCloud2, namespace + '/depth/points', self._cloud),
                (api.CameraInfo, namespace + '/depth/camera_info', self._camera_info),
                (api.JointState, '/joint_states', self._joint),
                (api.JointState, '/gripper/state', self._hand),
            ):
                self._subscriptions.append(node.create_subscription(kind, topic, callback, api.sensor_qos))
            self.device_info = node.create_client(api.GetDeviceInfo, namespace + '/get_device_info')
            self._clients.append(self.device_info)
            if self._owns_node:
                self.executor = api.SingleThreadedExecutor(context=self._context)
                self.executor.add_node(node)
                self.thread = threading.Thread(target=self._spin, daemon=True, name='tabletop_femto')
                self.thread.start()
        except Exception:
            self.close()
            raise

    def _spin(self):
        try:
            self.executor.spin()
        except Exception as error:
            self._error = error
            self.buffer.fail(error)

    def ok(self):
        if any(subscription.get_publisher_count() > 1 for subscription in self._subscriptions[:2]):
            raise CaptureError('multiple FEMTO publishers share the selected namespace; run one camera at a time')
        return not self._closed and self._error is None and self.node.context.ok()

    def verify_serial(self, expected, deadline, check):
        future = None
        try:
            while True:
                check()
                if self.clock() >= deadline:
                    raise CaptureError('FEMTO device identity service timed out')
                if future is None and self.device_info.service_is_ready():
                    future = self.device_info.call_async(self.api.GetDeviceInfo.Request())
                if future is not None and future.done():
                    response = future.result()
                    if response is None or not response.success or response.info.serial_number != expected:
                        raise CaptureError('connected FEMTO serial differs from selected calibration')
                    return
                time.sleep(.01)
        finally:
            if future is not None and not future.done():
                future.cancel()

    def _camera_info(self, message):
        try:
            value = camera_info(message)
            with self._lock:
                if self._info is not None and self._info != value:
                    raise CaptureError('CameraInfo changed while connected')
                self._info = value
        except Exception as error:
            self.buffer.fail(error)

    def _depth(self, message):
        self._receive(message, self._depths, self._clouds, True)

    def diagnostics(self):
        """Receipt counts distinguish absent streams from rejected frame pairs."""
        with self._lock:
            return {**self._received, 'camera_info_received': self._info is not None,
                    **self._robot_match}

    def _cloud(self, message):
        self._receive(message, self._clouds, self._depths, False)

    def _receive(self, message, own, other, is_depth):
        try:
            stamp = stamp_ns(message.header.stamp)
            if stamp <= 0:
                raise CaptureError('FEMTO stream has a zero source timestamp')
            now = self.clock()
            with self._lock:
                self._received['depth_messages' if is_depth else 'cloud_messages'] += 1
                # Buffer the delayed stream without reusing an exposure twice.
                while other and now - other[0][2] > .5:
                    other.popleft()
                match = min(other, key=lambda item: abs(item[0] - stamp), default=None)
                if match is None or abs(match[0] - stamp) > 30_000_000:
                    own.append((stamp, message, now))
                    return
                other.remove(match)
                self._received['matched_pairs'] += 1
                info = self._info
            if info is None:
                return
            depth, cloud = (message, match[1]) if is_depth else (match[1], message)
            if depth.header.frame_id != cloud.header.frame_id:
                raise CaptureError('paired depth/cloud frame mismatch')
            points = self.api.read_points_numpy(cloud, field_names=('x', 'y', 'z'), skip_nans=True)
            self.buffer.push(depth_stamp_ns=stamp_ns(depth.header.stamp), cloud_stamp_ns=stamp_ns(cloud.header.stamp),
                frame_id=cloud.header.frame_id, encoding=depth.encoding, depth=decode_depth(depth),
                points=np.asarray(points).reshape(-1, 3), camera_info=info)
        except Exception as error:
            self.buffer.fail(error)

    def _joint(self, message):
        value = canonical_joint_positions(message.name, message.position)
        stamp = stamp_ns(message.header.stamp)
        if value is not None and stamp > 0:
            with self._lock:
                self._joints.append((stamp, value.copy(), self.clock()))

    def _hand(self, message):
        stamp = stamp_ns(message.header.stamp)
        if stamp > 0 and len(message.position) == 1 and np.isfinite(message.position[0]):
            with self._lock:
                self._gripper.append((stamp, float(message.position[0]), self.clock()))

    def lookup(self, parent, child, timestamp_ns):
        try:
            return transform_matrix(self.tf.lookup_transform(parent, child,
                self.api.Time(nanoseconds=timestamp_ns), timeout=self.api.Duration(seconds=0.)))
        except self.api.TransformException:
            return None  # No latest-TF fallback: captures retain exposure-time TF.

    def robot_at(self, timestamp_ns, base_frame):
        with self._lock:
            now = self.clock()
            joints = sorted(((stamp, value) for stamp, value, received in self._joints
                             if now - received <= ROBOT_HISTORY_SECONDS), key=lambda item: item[0])
            gripper = [(stamp, value) for stamp, value, received in self._gripper
                       if now - received <= ROBOT_HISTORY_SECONDS]
        if not joints or not gripper:
            self._record_robot_match('missing joint history' if not joints else 'missing gripper history')
            return None
        nearest = min(joints, key=lambda pair: abs(pair[0] - timestamp_ns))
        hand = min(gripper, key=lambda pair: abs(pair[0] - timestamp_ns))
        if max(abs(nearest[0] - timestamp_ns), abs(hand[0] - timestamp_ns)) > 50_000_000:
            self._record_robot_match('state timestamp mismatch', nearest[0] - timestamp_ns,
                                     hand[0] - timestamp_ns)
            return None
        before = next((pair for pair in reversed(joints) if pair[0] <= timestamp_ns), None)
        after = next((pair for pair in joints if pair[0] >= timestamp_ns), None)
        q, mode = nearest[1], 'nearest'
        if before is not None and after is not None and before[0] != after[0] and max(timestamp_ns - before[0], after[0] - timestamp_ns) <= 50_000_000:
            fraction = (timestamp_ns - before[0]) / (after[0] - before[0])
            q = before[1] + fraction * (after[1] - before[1])
            mode = 'interpolated'
        tool = self.lookup(base_frame, 'tool0', timestamp_ns)
        if tool is None:
            self._record_robot_match('missing exposure-time tool TF', nearest[0] - timestamp_ns,
                                     hand[0] - timestamp_ns)
            return None
        self._record_robot_match('matched', nearest[0] - timestamp_ns, hand[0] - timestamp_ns)
        return {'joints': q.copy(), 'gripper_position': hand[1], 'base_T_tool': tool,
                'joint_stamp_ns': nearest[0], 'gripper_stamp_ns': hand[0], 'joint_sample_mode': mode}

    def _record_robot_match(self, reason, joint_delta_ns=None, gripper_delta_ns=None):
        with self._lock:
            self._robot_match = {'robot_state_match': reason}
            if joint_delta_ns is not None:
                self._robot_match.update(joint_offset_ms=joint_delta_ns / 1e6,
                                         gripper_offset_ms=gripper_delta_ns / 1e6)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.executor is not None:
            self.executor.shutdown(timeout_sec=2.)
        if self.thread is not None:
            self.thread.join(timeout=2.)
        if self.listener is not None:
            self.listener.unregister()
        buffer = getattr(self, 'tf', None)
        if buffer is not None:
            if hasattr(buffer, 'srv'):
                self.node.destroy_service(buffer.srv)
            if hasattr(buffer, 'jump_handle'):
                buffer.jump_handle.unregister()
        for subscription in self._subscriptions:
            self.node.destroy_subscription(subscription)
        for client in self._clients:
            self.node.destroy_client(client)
        if self._owns_node:
            self.node.destroy_node()
            self._context.try_shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _bindings():
    try:
        import rclpy
        from rclpy.context import Context
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.time import Time
        from rclpy.duration import Duration
        from sensor_msgs.msg import Image, PointCloud2, CameraInfo, JointState
        from sensor_msgs_py.point_cloud2 import read_points_numpy
        from orbbec_camera_msgs.srv import GetDeviceInfo
        from tf2_ros import Buffer, TransformListener, TransformException
    except ImportError as error:
        raise RuntimeError('Source ROS Jazzy and ~/ros2_ws/install/setup.bash, including orbbec_camera_msgs') from error
    return SimpleNamespace(**locals(), sensor_qos=qos_profile_sensor_data)
