"""Exact ROS contracts with injected bindings; no ROS nodes or devices needed."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from stacking_tabletop.ros.femto import FemtoRosTransport, decode_depth
from stacking_tabletop.acquisition.femto import CaptureError
from stacking_tabletop.execution.motion import UR_JOINT_NAMES
from test_tabletop_ros_hardware import Node, Context, Executor
from test_tabletop_femto_acquisition import calibration, frame_values, info


class Buffer:
    def __init__(self, node):
        self.calls = []
        self.fail = False

    def lookup_transform(self, parent, child, timestamp, timeout):
        self.calls.append((parent, child, timestamp.nanoseconds))
        if self.fail:
            raise LookupError('no exact transform')
        return NS(transform=NS(translation=NS(x=0., y=0., z=0.), rotation=NS(x=0., y=0., z=0., w=1.)))


class Listener:
    def __init__(self, *args, **kwargs):
        self.closed = False

    def unregister(self):
        self.closed = True


def bindings():
    return NS(rclpy=NS(init=lambda **kw: None), Context=Context, Node=Node,
              SingleThreadedExecutor=Executor, Buffer=Buffer, TransformListener=Listener,
              TransformException=LookupError, Time=lambda **kw: NS(**kw), Duration=lambda **kw: NS(**kw),
              Image=NS, PointCloud2=NS, CameraInfo=NS, JointState=NS, sensor_qos='sensor',
              GetDeviceInfo=NS(Request=lambda: NS()), read_points_numpy=lambda *a, **kw: np.array([[0., 0., .5]], np.float32))


def header(stamp=1_000_000_000):
    return NS(frame_id='camera_depth_optical_frame', stamp=NS(sec=stamp // 1_000_000_000, nanosec=stamp % 1_000_000_000))


def depth_message(stamp=1_000_000_000):
    return NS(header=header(stamp), width=1024, height=1024, encoding='16UC1',
              is_bigendian=False, step=2048, data=frame_values()['depth'].tobytes())


def info_message():
    value = info()
    return NS(header=header(), **{key: item for key, item in value.items() if key not in ('frame_id', 'roi')}, roi=NS(**value['roi']))


def test_borrowed_node_contract_and_no_publishers(calibration):
    node = Node()
    transport = FemtoRosTransport(calibration, node=node, bindings=bindings())
    assert set(node.subscriptions) == {'/camera/depth/image_raw', '/camera/depth/points',
        '/camera/depth/camera_info', '/joint_states', '/gripper/state'}
    assert all(value.qos == 'sensor' for value in node.subscriptions.values())
    assert set(node.clients) == {'/camera/get_device_info'}
    assert node.publishers == {}
    transport.close()
    transport.close()
    assert len(node.destroyed_entities) == 6
    assert transport.listener.closed and not node.destroyed and node.context.ok()


def test_owned_resources_are_closed(calibration):
    transport = FemtoRosTransport(calibration, bindings=bindings())
    transport.close()
    assert transport.node.destroyed
    assert not transport.node.context.ok()
    assert not transport.thread.is_alive()


def test_partial_constructor_failure_cleans_borrowed_resources(calibration):
    class FailingNode(Node):
        def create_client(self, *args):
            raise RuntimeError('client creation failed')
    node = FailingNode()
    with pytest.raises(RuntimeError, match='creation failed'):
        FemtoRosTransport(calibration, node=node, bindings=bindings())
    assert len(node.destroyed_entities) == 5
    assert not node.destroyed and node.context.ok()


@pytest.mark.parametrize('encoding,dtype', [('16UC1', '<u2'), ('mono16', '>u2'), ('32FC1', '>f4')])
def test_native_image_decoding_keeps_units_endianness_and_row_padding(encoding, dtype):
    array = np.array([[1, 2, 99], [3, 4, 99]], dtype=dtype)
    message = NS(encoding=encoding, height=2, width=2, step=3 * array.dtype.itemsize,
                 is_bigendian=dtype.startswith('>'), data=array.tobytes())
    np.testing.assert_array_equal(decode_depth(message), [[1, 2], [3, 4]])
    message.data = message.data[:-1]
    with pytest.raises(CaptureError, match='truncated'):
        decode_depth(message)


def test_pairing_uses_each_exposure_once_and_rejects_unsynchronized_or_old_data(calibration):
    now = [10.]
    node = Node()
    with FemtoRosTransport(calibration, node=node, bindings=bindings(), clock=lambda: now[0]) as transport:
        transport._camera_info(info_message())
        transport._depth(depth_message())
        transport._cloud(NS(header=header(1_001_000_000)))
        assert len(transport.buffer.after(0)) == 1
        transport._cloud(NS(header=header(1_001_000_000)))
        transport._depth(depth_message())
        assert len(transport.buffer.after(0)) == 1
        transport._depth(depth_message(2_000_000_000))
        transport._cloud(NS(header=header(2_050_000_000)))
        assert len(transport.buffer.after(0)) == 1
        now[0] += 1.
        transport._cloud(NS(header=header(2_001_000_000)))
        assert len(transport.buffer.after(0)) == 1
        transport._cloud(NS(header=header(3_000_000_000)))
        transport._depth(depth_message(3_001_000_000))
        assert len(transport.buffer.after(0)) == 2


def test_callback_frame_error_is_latched_for_command_worker(calibration):
    with FemtoRosTransport(calibration, node=Node(), bindings=bindings()) as transport:
        transport._camera_info(info_message())
        transport._depth(depth_message())
        value = header()
        value.frame_id = 'color_frame'
        transport._cloud(NS(header=value))
        with pytest.raises(CaptureError, match='frame mismatch'):
            transport.buffer.after(0)


def test_exact_transform_and_timestamp_matched_joint_gripper_state(calibration):
    now = [10.]
    with FemtoRosTransport(calibration, node=Node(), bindings=bindings(), clock=lambda: now[0]) as transport:
        for stamp, q in ((980_000_000, 0.), (1_020_000_000, .2)):
            message = NS(header=header(stamp), name=UR_JOINT_NAMES, position=np.full(6, q))
            transport._joint(message)
            transport._joint(message)  # Duplicate source stamps cannot break sorting.
        transport._hand(NS(header=header(1_005_000_000), position=[700.]))
        sample = transport.robot_at(1_000_000_000, 'base_link')
        np.testing.assert_allclose(sample['joints'], np.full(6, .1))
        assert sample['joint_sample_mode'] == 'interpolated'
        assert sample['gripper_position'] == 700.
        assert transport.tf.calls[-1] == ('base_link', 'tool0', 1_000_000_000)
        transport.tf.fail = True
        assert transport.robot_at(1_000_000_000, 'base_link') is None
        assert all(call[2] != 0 for call in transport.tf.calls)
        now[0] += 1.
        assert transport.robot_at(1_000_000_000, 'base_link') is None


def test_serial_service_checks_actual_device_and_cancels_pending_future(calibration):
    with FemtoRosTransport(calibration, node=Node(), bindings=bindings()) as transport:
        response = NS(success=True, info=NS(serial_number='wrong-camera'))
        transport.device_info.call_async = lambda request: NS(done=lambda: True, result=lambda: response)
        with pytest.raises(CaptureError, match='serial differs'):
            transport.verify_serial('test-camera', transport.clock() + 1, lambda: None)
        response.info.serial_number = 'test-camera'
        transport.verify_serial('test-camera', transport.clock() + 1, lambda: None)
        cancelled = []
        transport.device_info.call_async = lambda request: NS(done=lambda: False, cancel=lambda: cancelled.append(True))
        calls = []
        def check():
            calls.append(True)
            if len(calls) == 2:
                raise CaptureError('cancelled')
        with pytest.raises(CaptureError, match='cancelled'):
            transport.verify_serial('test-camera', transport.clock() + 1, check)
        assert cancelled == [True]


def test_duplicate_camera_publishers_fail_before_capture(calibration):
    with FemtoRosTransport(calibration, node=Node(), bindings=bindings()) as transport:
        transport._subscriptions[0].get_publisher_count = lambda: 2
        with pytest.raises(CaptureError, match='multiple FEMTO publishers'):
            transport.ok()


def test_borrowed_tf_buffer_service_and_clock_callback_are_released(calibration):
    node = Node()
    released = []
    node.destroy_service = lambda service: released.append(service)
    with FemtoRosTransport(calibration, node=node, bindings=bindings()) as transport:
        transport.tf.srv = 'tf2_frames'
        transport.tf.jump_handle = NS(unregister=lambda: released.append('clock_jump'))
    assert released == ['tf2_frames', 'clock_jump']
    assert not node.destroyed
