"""Thin conversion from ROS PointCloud2 messages to common capture values."""

import numpy as np

from stacking_common.pointcloud import CapturedPointCloud, PointCloudData
from stacking_tabletop.config import CameraProfile


class RosPointCloudAdapter:
    """Convert one subscribed ROS message without owning a ROS node."""

    def __init__(self, camera: CameraProfile) -> None:
        self.camera = camera

    def convert(self, message, *, reader=None) -> CapturedPointCloud:
        if reader is None:
            try:
                from sensor_msgs_py.point_cloud2 import read_points_numpy
            except ImportError as error:
                raise RuntimeError(
                    "ROS sensor_msgs_py is required for PointCloud2 conversion"
                ) from error
            reader = read_points_numpy
        values = reader(
            message,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )
        points = _xyz_array(values)
        stamp = message.header.stamp
        timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        return CapturedPointCloud(
            PointCloudData(points),
            str(message.header.frame_id),
            timestamp_ns,
            self.camera.serial,
        )


class RosPointCloudSubscription:
    """Attach the converter to an existing ROS node and configured topic."""

    def __init__(
        self,
        node,
        camera: CameraProfile,
        callback,
        *,
        qos=10,
        message_type=None,
        reader=None,
    ) -> None:
        if message_type is None:
            try:
                from sensor_msgs.msg import PointCloud2 as message_type
            except ImportError as error:
                raise RuntimeError(
                    "ROS sensor_msgs is required for a point-cloud subscription"
                ) from error
        self.adapter = RosPointCloudAdapter(camera)
        self.callback = callback
        self.reader = reader
        self.subscription = node.create_subscription(
            message_type,
            camera.pointcloud_topic,
            self._receive,
            qos,
        )

    def _receive(self, message) -> None:
        self.callback(self.adapter.convert(message, reader=self.reader))


def _xyz_array(values) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.names:
        required = ("x", "y", "z")
        if not all(name in array.dtype.names for name in required):
            raise ValueError("PointCloud2 data is missing XYZ fields")
        return np.column_stack([array[name] for name in required])
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("PointCloud2 reader must return XYZ columns")
    return array[:, :3]
