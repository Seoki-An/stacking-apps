"""ROS graph/ownership contract exercised with a fake node and message types."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from stacking_tabletop.execution import HardwareFeedback, HardwareError, UR_JOINT_NAMES
from stacking_tabletop.ros.hardware import TabletopRosTransport


class Entity:
    def __init__(self, topic, callback=None, qos=None):
        self.topic, self.callback, self.qos = topic, callback, qos
        self.messages = []

    def publish(self, message):
        self.messages.append(message)

    def get_subscription_count(self):
        return 1

    def get_publisher_count(self):
        return 1

    def service_is_ready(self):
        return True

    def call_async(self, request):
        self.messages.append(request)
        return NS(done=lambda: True, result=lambda: NS(success=True))


class Context:
    def __init__(self):
        self.live = True

    def ok(self):
        return self.live

    def try_shutdown(self):
        self.live = False


class Node:
    def __init__(self, name="test", context=None):
        self.context = context if context is not None else Context()
        self.publishers, self.subscriptions, self.clients = {}, {}, {}
        self.destroyed_entities = []
        self.destroyed = False

    def create_publisher(self, kind, topic, qos):
        self.publishers[topic] = Entity(topic, qos=qos)
        return self.publishers[topic]

    def create_subscription(self, kind, topic, callback, qos):
        self.subscriptions[topic] = Entity(topic, callback, qos)
        return self.subscriptions[topic]

    def create_client(self, kind, topic):
        self.clients[topic] = Entity(topic)
        return self.clients[topic]

    def get_clock(self):
        return NS(now=lambda: NS(to_msg=lambda: "ros-stamp"))

    def destroy_node(self):
        self.destroyed = True

    def destroy_subscription(self, entity):
        self.destroyed_entities.append(entity)

    destroy_publisher = destroy_subscription
    destroy_client = destroy_subscription


class Executor:
    def __init__(self, context):
        import threading
        self.stopped = threading.Event()

    def add_node(self, node):
        self.node = node

    def spin(self):
        self.stopped.wait()

    def shutdown(self, timeout_sec):
        self.stopped.set()


def pose_message():
    return NS(header=NS(frame_id="base", stamp=None),
              pose=NS(position=NS(x=0., y=0., z=0.), orientation=NS(x=0., y=0., z=0., w=1.)))


def bindings():
    return NS(rclpy=NS(init=lambda **kw: None), Context=Context, Node=Node,
              SingleThreadedExecutor=Executor, ExternalShutdownException=type("Shutdown", (Exception,), {}),
              QoSProfile=lambda **kw: NS(**kw),
              DurabilityPolicy=NS(TRANSIENT_LOCAL="latched"), ReliabilityPolicy=NS(RELIABLE="reliable"),
              sensor_qos="sensor_data", PoseStamped=pose_message, JointState=NS,
              Bool=NS, Float64MultiArray=NS, Int32=NS, String=NS, Trigger=NS(Request=lambda: NS()))


@pytest.mark.parametrize('publishers', [1, 2])
def test_robot_description_is_latched_read_only_and_requires_one_publisher(publishers):
    class DescriptionNode(Node):
        def create_subscription(self, kind, topic, callback, qos):
            entity = super().create_subscription(kind, topic, callback, qos)
            if topic == '/robot_description':
                entity.get_publisher_count = lambda: publishers
                callback(NS(data='<robot name="installed"/>'))
            return entity
    node = DescriptionNode()
    with TabletopRosTransport(node=node, bindings=bindings()) as transport:
        if publishers == 1:
            assert transport.read_robot_description() == '<robot name="installed"/>'
        else:
            with pytest.raises(RuntimeError, match='one authoritative'):
                transport.read_robot_description()
        entity = node.subscriptions['/robot_description']
        assert entity.qos.durability == 'latched'
        assert entity in node.destroyed_entities
        assert all(not publisher.messages for publisher in node.publishers.values())


def test_missing_robot_description_timeout_cleans_subscription():
    node = Node()
    with TabletopRosTransport(node=node, bindings=bindings()) as transport:
        with pytest.raises(RuntimeError, match='one authoritative'):
            transport.read_robot_description(timeout=.001)
        assert node.subscriptions['/robot_description'] in node.destroyed_entities


def test_graph_topics_qos_and_no_startup_or_cleanup_commands():
    node = Node()
    transport = TabletopRosTransport(node=node, bindings=bindings())
    assert set(node.publishers) == {"/dish_planning/pos_planner", "/dish_planning/admittance_gain",
                                    "/dish_planning/use_wrench", "/grip_flag", "/grip_position_goal"}
    assert set(node.clients) == {"/gripper/hold_current_position", "/gripper/recover"}
    for topic in ("/admittance_ready", "/gripper/fault", "/gripper/contact_hold_goal"):
        assert node.subscriptions[topic].qos.durability == "latched"
        assert node.subscriptions[topic].qos.reliability == "reliable"
    for topic in ("/joint_states", "/gripper/state"):
        assert node.subscriptions[topic].qos == "sensor_data"
    assert transport.arm_connected() and transport.gripper_connected()
    transport.close()
    transport.close()
    assert not transport.ok()
    assert len(node.destroyed_entities) == 13  # Six subscriptions, five publishers, two clients.
    assert not node.destroyed and node.context.ok()
    assert all(not entity.messages for entity in node.publishers.values())


def test_private_context_shutdown_does_not_touch_external_context():
    external = Context()
    transport = TabletopRosTransport(bindings=bindings())
    assert transport.thread.is_alive()
    transport.close()
    assert transport.node.destroyed and not transport.node.context.ok()
    assert not transport.thread.is_alive()
    assert external.ok()


def test_pose_transport_frame_and_invalid_input_rejection():
    node = Node()
    with TabletopRosTransport(node=node, bindings=bindings()) as transport:
        receive_pose = node.subscriptions["/ee_pose_ur5e"].callback
        node.subscriptions["/joint_states"].callback(NS(name=UR_JOINT_NAMES, position=[0.] * 6))
        message = pose_message()
        message.header.frame_id = "base_link"
        receive_pose(message)
        with pytest.raises(HardwareError):
            transport.feedback.arm_state()
        message.header.frame_id = "base"
        message.pose.orientation.w = float("nan")
        receive_pose(message)
        with pytest.raises(HardwareError):
            transport.feedback.arm_state()
        message.pose.orientation.w = 1.
        receive_pose(message)
        np.testing.assert_allclose(transport.feedback.arm_state()[1], np.eye(4))
        transport.publish_pose(np.eye(4))
        sent = node.publishers["/dish_planning/pos_planner"].messages[-1]
        assert sent.header.frame_id == "base" and sent.header.stamp == "ros-stamp"
        with pytest.raises(ValueError):
            transport.publish_pose(np.full((4, 4), np.nan))


def test_gripper_ros_values_stay_raw_and_hold_ack_is_separate():
    node = Node()
    with TabletopRosTransport(node=node, bindings=bindings()) as transport:
        node.subscriptions["/gripper/state"].callback(NS(position=[796.]))
        node.subscriptions["/gripper/contact_hold_goal"].callback(NS(data=796))
        state = transport.feedback.gripper_state()
        assert state.position == 796 and state.hold_goal == 796 and not state.holding
        transport.publish_gripper_position(169)
        transport.publish_gripper_flag(2)
        assert node.publishers["/grip_position_goal"].messages[-1].data == 169
        assert node.publishers["/grip_flag"].messages[-1].data == 2
        assert transport.request_gripper_hold().result().success
        for value in (50, 1700, 91.1, True):
            with pytest.raises(ValueError):
                transport.publish_gripper_position(value)


def test_partial_setup_failure_releases_only_created_entities():
    class BrokenNode(Node):
        def create_client(self, *_):
            raise RuntimeError("creation failed")

    node = BrokenNode()
    with pytest.raises(RuntimeError, match="creation failed"):
        TabletopRosTransport(node=node, bindings=bindings())
    assert len(node.destroyed_entities) == 11
    assert node.context.ok() and not node.destroyed


def test_robot_description_waits_for_discovery_after_latched_message():
    counts = iter([0, 0, 1])
    class DescriptionNode(Node):
        def create_subscription(self, kind, topic, callback, qos):
            entity = super().create_subscription(kind, topic, callback, qos)
            if topic == '/robot_description':
                entity.get_publisher_count = lambda: next(counts)
                callback(NS(data='<robot/>'))
            return entity
    with TabletopRosTransport(node=DescriptionNode(), bindings=bindings()) as transport:
        assert transport.read_robot_description(timeout=1.) == '<robot/>'
