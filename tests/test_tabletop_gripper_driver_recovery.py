"""Exercise the installed external driver with fake ROS and serial bindings."""

import importlib.util
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace as NS

import pytest

from test_tabletop_gripper_recovery import Clock, Packet, driver


def test_startup_fault_recovers_and_only_explicit_next_command_enables_torque(monkeypatch):
    source = Path.home() / 'ros2_ws/src/ur5e_admittance/scripts/gripper_node.py'
    if not source.exists():
        pytest.skip('external ur5e_admittance driver is not installed')
    class Node:
        def __init__(self, *args):
            self.services, self.published = {}, {}

        def declare_parameter(self, name, value):
            return NS(value=value)

        def get_logger(self):
            return NS(info=lambda *a: None, warning=lambda *a: None, error=lambda *a: None)

        def create_publisher(self, kind, topic, qos):
            self.published[topic] = []
            return NS(publish=self.published[topic].append)

        def create_subscription(self, *args):
            return NS()

        def destroy_subscription(self, *args):
            pass

        def create_service(self, kind, name, callback):
            self.services[name] = callback
            return NS()

        def create_timer(self, *args):
            return NS()

    def module(name, **attrs):
        result = ModuleType(name)
        result.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, result)
    module('rclpy', ok=lambda: True)
    module('rclpy.node', Node=Node)
    module('rclpy.qos', DurabilityPolicy=NS(TRANSIENT_LOCAL=1), QoSProfile=lambda **kw: NS(**kw),
           ReliabilityPolicy=NS(RELIABLE=1))
    module('std_msgs'); module('std_msgs.msg', Int32=NS, String=NS)
    module('std_srvs'); module('std_srvs.srv', Trigger=NS)
    module('sensor_msgs'); module('sensor_msgs.msg', JointState=NS)
    packet = Packet()
    module('dynamixel_sdk', COMM_SUCCESS=0, PacketHandler=lambda _: packet,
           PortHandler=lambda _: NS(openPort=lambda: True, setBaudRate=lambda _: True))
    packet.getRxPacketError = lambda _: 'Hardware alert'
    monkeypatch.setitem(sys.modules, 'gripper_recovery', driver)
    spec = importlib.util.spec_from_file_location('test_external_gripper_node', source)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    node = loaded.GripperNode()
    node._command_worker_thread.join(1.)
    assert not node._command_worker_thread.is_alive()
    assert '/gripper/recover' in node.services
    assert '0x21' in node._fault_message
    assert node._pending_command is None
    assert node.published['/gripper/fault'][-1].data
    assert all(c[0] != 'write' or c[1:] == (64, 0) for c in packet.calls)
    assert not any(c[0] == 'reboot' for c in packet.calls)

    clock = Clock()
    reboot = driver.reboot_and_check
    monkeypatch.setattr(driver, 'reboot_and_check',
                        lambda *args: reboot(*args, clock=clock, sleep=clock.sleep))
    result = node.services['/gripper/recover'](None, NS())
    assert result.success and node._fault_message is None
    assert node._command_worker_thread.is_alive()
    assert node._pending_command is None and node._confirmed_hold_goal is None
    assert node.published['/gripper/fault'][-1].data == ''
    assert packet.registers[64] == 0
    assert not any(c[0] == 'write' and c[1:] != (64, 0) for c in packet.calls)

    def write_goal(port, device, address, value):
        packet.calls.append(('write', address, value))
        packet.registers[address] = value
        return 0, 0
    packet.write4ByteTxRx = write_goal
    packet.calls.clear()
    try:
        node._start_move(1200, source='raw position topic')
        deadline = time.monotonic() + 2.
        while ('write', 116, 1200) not in packet.calls and time.monotonic() < deadline:
            time.sleep(.01)
        writes = [c for c in packet.calls if c[0] == 'write']
        assert writes == [('write', 116, 1003), ('write', 64, 0),
                          ('write', 64, 1), ('write', 116, 1200)]
        assert not node._torque_needs_enable
    finally:
        with node._state_condition:
            node._shutdown_requested = True
            node._state_condition.notify_all()
        node._command_worker_thread.join(2.)
        assert not node._command_worker_thread.is_alive()
