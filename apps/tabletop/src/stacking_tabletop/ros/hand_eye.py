"""Read-only joint/TF evidence for wrist captures; never publishes commands."""

from collections import deque
import threading
import time
from types import SimpleNamespace

import numpy as np

from stacking_tabletop.acquisition.hand_eye_capture import LIMITS
from stacking_tabletop.execution.motion import canonical_joint_positions
from .femto import stamp_ns, transform_matrix


def bracketed_joints(samples, start_ns, end_ns):
    """Require samples on both sides, retaining all intervening feedback."""
    samples = sorted(samples, key=lambda item: item[0])
    before = [i for i, (stamp, _) in enumerate(samples) if stamp <= start_ns]
    after = [i for i, (stamp, _) in enumerate(samples) if stamp >= end_ns]
    if not before or not after:
        return None
    selected = samples[before[-1]:after[0] + 1]
    stamps = np.array([stamp for stamp, _ in selected], dtype=np.int64)
    if (start_ns - stamps[0] > LIMITS['max_joint_gap_ns']
            or stamps[-1] - end_ns > LIMITS['max_joint_gap_ns']
            or np.any(np.diff(stamps) <= 0) or np.any(np.diff(stamps) > LIMITS['max_joint_gap_ns'])):
        return None
    return dict(joint_stamps_ns=stamps.tolist(), joint_positions=[q.tolist() for _, q in selected])


class HandEyeRobot:
    def __init__(self, calibration, *, bindings=None):
        if calibration.profile.robot_pose_source != 'tf':
            raise ValueError('live capture currently supports the pinned TF pose source only')
        self.api = api = _bindings() if bindings is None else bindings
        self.calibration = calibration
        self.context = api.Context()
        self.node = self.executor = self.thread = self.listener = self.subscription = None
        self._error = None
        self._lock = threading.Lock()
        self._joints = deque(maxlen=10000)
        api.rclpy.init(context=self.context)
        try:
            self.node = api.Node('tabletop_hand_eye_capture', context=self.context)
            if self.node.get_parameter('use_sim_time').value:
                raise ValueError('live camera capture requires ROS system time, not simulated time')
            self.tf = api.Buffer(node=self.node)
            self.listener = api.TransformListener(self.tf, self.node, spin_thread=False)
            self.subscription = self.node.create_subscription(api.JointState, '/joint_states',
                                                              self._joint, api.sensor_qos)
            self.executor = api.SingleThreadedExecutor(context=self.context)
            self.executor.add_node(self.node)
            self.thread = threading.Thread(target=self._spin, daemon=True, name='wrist_capture_state')
            self.thread.start()
        except BaseException:
            self.close()
            raise

    def _spin(self):
        try:
            self.executor.spin()
        except Exception as error:
            self._error = error

    def _joint(self, message):
        value = canonical_joint_positions(message.name, message.position)
        stamp = stamp_ns(message.header.stamp)
        if value is None or stamp <= 0:
            self._error = ValueError('invalid joint feedback timestamp or positions')
            return
        with self._lock:
            if self._joints and stamp <= self._joints[-1][0]:
                self._error = ValueError('joint feedback clock reset/duplicate/out-of-order sample')
            self._joints.append((stamp, value, time.monotonic_ns()))

    def measure(self, start_ns, end_ns, stamps):
        deadline = time.monotonic() + .5
        while time.monotonic() < deadline:
            if self._error is not None:
                raise RuntimeError(f'robot feedback failed: {self._error}')
            if not self.context.ok():
                raise RuntimeError('robot feedback ROS context stopped')
            if self.subscription.get_publisher_count() != 1:
                raise ValueError('expected one authoritative joint-state publisher')
            ros_now = self.node.get_clock().now().nanoseconds
            ros_offset = ros_now - time.time_ns()
            if abs(ros_offset) > LIMITS['max_clock_step_ns']:
                raise ValueError('ROS clock and camera host system clock disagree')
            with self._lock:
                samples = [(stamp, q.copy()) for stamp, q, received in self._joints
                           if time.monotonic_ns() - received < 2_000_000_000]
            packet = bracketed_joints(samples, start_ns, end_ns)
            if packet is not None:
                try:
                    for name, stamp in stamps.items():
                        transform = self.tf.lookup_transform(self.calibration.profile.base_frame,
                            self.calibration.profile.tool_frame, self.api.Time(nanoseconds=stamp),
                            timeout=self.api.Duration(seconds=0.))
                        packet[name + '_base_T_tool'] = transform_matrix(transform).tolist()
                    packet.update(interval_start_ns=start_ns, interval_end_ns=end_ns,
                                  pose_stamps_ns=stamps, pose_source='tf_at_sdk_global_frame_time',
                                  ros_system_offset_ns=ros_offset)
                    return packet
                except self.api.TransformException:
                    pass  # Wait for exact-time TF; never substitute latest TF.
            time.sleep(.005)
        raise ValueError('missing timestamp-bracketed joints or exact-time TF for capture')

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(timeout_sec=2.)
        if self.thread is not None:
            self.thread.join(timeout=2.)
        if self.listener is not None:
            self.listener.unregister()
        if self.node is not None:
            if self.subscription is not None:
                self.node.destroy_subscription(self.subscription)
            buffer = getattr(self, 'tf', None)
            if buffer is not None and hasattr(buffer, 'srv'):
                self.node.destroy_service(buffer.srv)
            if buffer is not None and hasattr(buffer, 'jump_handle'):
                buffer.jump_handle.unregister()
            self.node.destroy_node()
            self.node = None
        self.context.try_shutdown()

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
        from sensor_msgs.msg import JointState
        from tf2_ros import Buffer, TransformListener, TransformException
    except ImportError as error:
        raise RuntimeError('Source ROS Jazzy and the UR robot state/TF environment') from error
    return SimpleNamespace(**locals(), sensor_qos=qos_profile_sensor_data)
