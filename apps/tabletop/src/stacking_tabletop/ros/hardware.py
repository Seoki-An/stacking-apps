"""ROS 2 Jazzy transport for the existing UR5e/Dynamixel controller graph.

No driver launch, serial access, or commands occur on construction. ROS imports
are lazy. An injected node remains owned/spun by its caller; otherwise this
transport owns a private ROS context and executor, never the global context.
"""

import threading
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from ..execution.feedback import HardwareFeedback
from ..execution.motion import validate_pose


class TabletopRosTransport:
    def __init__(self, *, node=None, feedback=None, node_name="tabletop_hardware", bindings=None):
        api = _bindings() if bindings is None else bindings
        self.api = api
        self.feedback = feedback if feedback is not None else HardwareFeedback()
        self._closed = False
        self._spin_error = None
        self._owns_node = node is None
        self._context = None
        self.executor = None
        self.thread = None
        self._publishers = []
        self._subscriptions = []
        self._clients = []
        if node is None:
            self._context = api.Context()
            api.rclpy.init(context=self._context)
            try:
                node = api.Node(node_name, context=self._context)
            except Exception:
                self._context.try_shutdown()
                raise
        self.node = node
        try:
            latched = api.QoSProfile(depth=1, durability=api.DurabilityPolicy.TRANSIENT_LOCAL,
                                    reliability=api.ReliabilityPolicy.RELIABLE)
            self._subscribe(api.Bool, "/admittance_ready", lambda msg: self.feedback.receive_ready(msg.data), latched)
            self._subscribe(api.JointState, "/joint_states", self._receive_joints, api.sensor_qos)
            self._subscribe(api.PoseStamped, "/ee_pose_ur5e", self._receive_pose, 10)
            self.gripper_subscription = self._subscribe(api.JointState, "/gripper/state", self._receive_gripper, api.sensor_qos)
            self._subscribe(api.String, "/gripper/fault", lambda msg: self.feedback.receive_fault(msg.data), latched)
            self._subscribe(api.Int32, "/gripper/contact_hold_goal", lambda msg: self.feedback.receive_hold(msg.data), latched)
            self.pose_publisher = self._publisher(api.PoseStamped, "/dish_planning/pos_planner")
            self.gain_publisher = self._publisher(api.Float64MultiArray, "/dish_planning/admittance_gain")
            self.wrench_publisher = self._publisher(api.Bool, "/dish_planning/use_wrench")
            self.flag_publisher = self._publisher(api.Int32, "/grip_flag")
            self.position_publisher = self._publisher(api.Int32, "/grip_position_goal")
            self.hold_client = node.create_client(api.Trigger, "/gripper/hold_current_position")
            self._clients.append(self.hold_client)
            if self._owns_node:
                self.executor = api.SingleThreadedExecutor(context=self._context)
                self.executor.add_node(node)
                self.thread = threading.Thread(target=self._spin, name=node_name, daemon=True)
                self.thread.start()
        except Exception:
            self.close()
            raise

    def _subscribe(self, kind, topic, callback, qos):
        subscription = self.node.create_subscription(kind, topic, callback, qos)
        self._subscriptions.append(subscription)
        return subscription

    def _publisher(self, kind, topic):
        publisher = self.node.create_publisher(kind, topic, 10)
        self._publishers.append(publisher)
        return publisher

    def _spin(self):
        try:
            self.executor.spin()
        except self.api.ExternalShutdownException:
            pass
        except Exception as error:
            self._spin_error = error

    def ok(self):
        return not self._closed and self._spin_error is None and self.node.context.ok()

    def arm_connected(self):
        return self.ok() and all(pub.get_subscription_count() > 0 for pub in
                                 (self.pose_publisher, self.gain_publisher, self.wrench_publisher))

    def gripper_connected(self):
        return (self.ok() and self.gripper_subscription.get_publisher_count() > 0
                and self.flag_publisher.get_subscription_count() > 0
                and self.position_publisher.get_subscription_count() > 0
                and self.hold_client.service_is_ready())

    def _receive_joints(self, message):
        self.feedback.receive_joints(message.name, message.position)

    def _receive_pose(self, message):
        if message.header.frame_id != "base":
            return
        p, q = message.pose.position, message.pose.orientation
        values = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        if not np.all(np.isfinite(values)) or np.linalg.norm(values[3:]) <= 1e-12:
            return
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(values[3:]).as_matrix()
        pose[:3, 3] = values[:3]
        self.feedback.receive_pose(pose)

    def _receive_gripper(self, message):
        if len(message.position):
            self.feedback.receive_gripper(message.position[0])

    def publish_pose(self, pose):
        pose = validate_pose(pose)
        message = self.api.PoseStamped()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.header.frame_id = "base"
        message.pose.position.x, message.pose.position.y, message.pose.position.z = pose[:3, 3].tolist()
        q = Rotation.from_matrix(pose[:3, :3]).as_quat().tolist()
        (message.pose.orientation.x, message.pose.orientation.y,
         message.pose.orientation.z, message.pose.orientation.w) = q
        self.pose_publisher.publish(message)

    def publish_gains(self, gains):
        values = np.asarray(gains, dtype=float)
        if values.shape != (18,) or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("admittance gains require 18 finite non-negative values")
        self.gain_publisher.publish(self.api.Float64MultiArray(data=values.tolist()))

    def publish_wrench(self, enabled):
        self.wrench_publisher.publish(self.api.Bool(data=bool(enabled)))

    def publish_gripper_flag(self, command):
        if command not in (0, 1, 2):
            raise ValueError("gripper flag must be CLOSE=0, MIDDLE=1, or OPEN=2")
        self.flag_publisher.publish(self.api.Int32(data=int(command)))

    def publish_gripper_position(self, position):
        if (isinstance(position, bool) or not isinstance(position, (int, np.integer))
                or not 91 <= position <= 1500):
            raise ValueError("raw gripper position must be an integer in 91..1500")
        self.position_publisher.publish(self.api.Int32(data=int(position)))

    def request_gripper_hold(self):
        return self.hold_client.call_async(self.api.Trigger.Request())

    def close(self):
        """Release only owned ROS resources; never send arm/gripper commands."""
        if self._closed:
            return
        self._closed = True
        if self.executor is not None:
            self.executor.shutdown(timeout_sec=2.0)
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self._owns_node:
            self.node.destroy_node()
            self._context.try_shutdown()
        else:
            for entity in self._subscriptions:
                self.node.destroy_subscription(entity)
            for entity in self._publishers:
                self.node.destroy_publisher(entity)
            for entity in self._clients:
                self.node.destroy_client(entity)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _bindings():
    try:
        import rclpy
        from rclpy.context import Context
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
        from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool, Float64MultiArray, Int32, String
        from std_srvs.srv import Trigger
    except ImportError as error:
        raise RuntimeError("source ROS 2 Jazzy and the tabletop workspace before opening hardware transport") from error
    return SimpleNamespace(rclpy=rclpy, Context=Context, Node=Node,
                           SingleThreadedExecutor=SingleThreadedExecutor,
                           ExternalShutdownException=ExternalShutdownException,
                           QoSProfile=QoSProfile, DurabilityPolicy=DurabilityPolicy,
                           ReliabilityPolicy=ReliabilityPolicy, sensor_qos=qos_profile_sensor_data,
                           PoseStamped=PoseStamped, JointState=JointState,
                           Bool=Bool, Float64MultiArray=Float64MultiArray,
                           Int32=Int32, String=String, Trigger=Trigger)
