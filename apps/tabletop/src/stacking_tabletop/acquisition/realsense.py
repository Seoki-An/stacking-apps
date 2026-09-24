"""Native wrist RGB-D transport. SDK import and device ownership are explicit."""

import importlib.metadata
import time

import numpy as np

from stacking_common.pointcloud import validate_rigid_transform


def stream_geometry(profile):
    profile = profile.as_video_stream_profile()
    intr = profile.get_intrinsics()
    if (intr.width <= 0 or intr.height <= 0 or intr.fx <= 0 or intr.fy <= 0
            or not np.isfinite([intr.fx, intr.fy, intr.ppx, intr.ppy, *intr.coeffs]).all()):
        raise ValueError('invalid native stream intrinsics')
    return dict(width=intr.width, height=intr.height, fps=profile.fps(),
                format=str(profile.format()), distortion_model=str(intr.model),
                K=[[intr.fx, 0., intr.ppx], [0., intr.fy, intr.ppy], [0., 0., 1.]],
                distortion=list(intr.coeffs))


def extrinsic_matrix(extrinsic):
    transform = np.eye(4)
    # librealsense rs2_extrinsics stores a column-major rotation, metres.
    transform[:3, :3] = np.asarray(extrinsic.rotation).reshape(3, 3, order='F')
    transform[:3, 3] = extrinsic.translation
    return validate_rigid_transform(transform)


def bindings():
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise RuntimeError('Install pyrealsense2 in this Python environment for live capture') from error
    return rs


def list_profiles(calibration, *, rs=None):
    """Enumerate the pinned device without starting streams or setting options."""
    rs = bindings() if rs is None else rs
    context = rs.context()
    for device in context.query_devices():
        if device.get_info(rs.camera_info.serial_number) == calibration.profile.serial:
            return dict(serial=calibration.profile.serial,
                        profiles=sorted({str(p) for s in device.query_sensors()
                                         for p in s.get_stream_profiles()}),
                        simultaneous_stream_combination_verified=False)
    raise ValueError('pinned D435 serial is not connected')


class RealSenseCamera:
    """One exact native RGB8/Z16 profile; no alignment, filtering or fallback.

    Only global_time_enabled is changed, and restored on close. Existing camera
    drivers must release the device before use. No driver killing/device reset.
    """

    def __init__(self, calibration, *, depth_size=(848, 480), fps=15, rs=None):
        self.rs = rs = bindings() if rs is None else rs
        self.pipeline = None
        self._restore = []
        self.calibration = calibration
        self.context = rs.context()
        devices = [d for d in self.context.query_devices()
                   if d.get_info(rs.camera_info.serial_number) == calibration.profile.serial]
        if len(devices) != 1:
            raise ValueError('expected exactly one connected device matching the pinned D435 serial')
        self.device = devices[0]
        self.sensors = self.device.query_sensors()
        self.available_profiles = sorted({str(p) for s in self.sensors for p in s.get_stream_profiles()})
        try:
            config = rs.config()
            config.enable_device(calibration.profile.serial)
            config.enable_stream(rs.stream.color, calibration.width, calibration.height, rs.format.rgb8, fps)
            config.enable_stream(rs.stream.depth, *depth_size, rs.format.z16, fps)
            pipeline = rs.pipeline(self.context)
            # Start first, so opening a busy camera fails before changing options.
            try:
                active = pipeline.start(config)
            except RuntimeError as error:
                raise RuntimeError(
                    f'pinned D435 {calibration.profile.serial} could not start color '
                    f'{calibration.width}x{calibration.height} RGB8 and depth '
                    f'{depth_size[0]}x{depth_size[1]} Z16 at {fps} fps: {error}. '
                    'Check camera power, USB connection and external camera drivers; '
                    'then run hand_eye_capture_cli profiles.'
                ) from error
            self.pipeline = pipeline
            self.device = active.get_device()
            self.sensors = self.device.query_sensors()
            for sensor in (self.device.first_color_sensor(), self.device.first_depth_sensor()):
                option = rs.option.global_time_enabled
                if not sensor.supports(option):
                    raise ValueError('both sensors must support SDK global timestamps')
                self._restore.append((sensor, option, sensor.get_option(option)))
                sensor.set_option(option, 1.)
                if sensor.get_option(option) != 1.:
                    raise ValueError('cannot enable global timestamps')
            self.profiles = {name: active.get_stream(getattr(rs.stream, name))
                             for name in ('color', 'depth')}
            geometry = {name: stream_geometry(profile) for name, profile in self.profiles.items()}
            color, depth = geometry['color'], geometry['depth']
            serial = self.device.get_info(rs.camera_info.serial_number)
            calibration.validate_color_stream(serial=serial, width=color['width'], height=color['height'],
                factory_matrix=color['K'], factory_distortion=color['distortion'])
            if (color['fps'] != fps or depth['fps'] != fps
                    or (depth['width'], depth['height']) != tuple(depth_size)
                    or color['format'] != str(rs.format.rgb8) or depth['format'] != str(rs.format.z16)):
                raise ValueError('active stream differs from the explicitly requested native profile')
            scale = float(self.device.first_depth_sensor().get_depth_scale())
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('invalid native depth scale')
            transform = extrinsic_matrix(self.profiles['depth'].get_extrinsics_to(self.profiles['color']))
            info = {name: self.device.get_info(getattr(rs.camera_info, name))
                    if self.device.supports(getattr(rs.camera_info, name)) else None
                    for name in ('name', 'serial_number', 'firmware_version', 'usb_type_descriptor')}
            try:
                version = importlib.metadata.version('pyrealsense2')
            except importlib.metadata.PackageNotFoundError:
                version = getattr(rs, '__version__', 'unknown')
            self.report = dict(schema_version=1, device=info, sdk_version=version,
                streams=geometry, depth_metres_per_unit=scale, color_T_depth=transform.tolist(),
                available_profiles=self.available_profiles, sensor_options=self.options(),
                timestamp_mapping='SDK global_time to host system clock; uncertainty unmeasured',
                pose_time_semantics='SDK frame timestamp; not asserted to be exposure midpoint',
                accuracy_1mm_verified=False)
        except BaseException:
            self.close()
            raise

    def options(self):
        result = []
        for sensor in self.sensors:
            result.append({str(option): float(sensor.get_option(option))
                           for option in sensor.get_supported_options()})
        return result

    def read(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        receipt_ns, receipt_monotonic_ns = time.time_ns(), time.monotonic_ns()
        result = dict(receipt_ns=receipt_ns, receipt_monotonic_ns=receipt_monotonic_ns)
        for name in ('color', 'depth'):
            frame = getattr(frames, 'get_' + name + '_frame')()
            if not frame:
                raise ValueError('missing native ' + name + ' frame')
            if stream_geometry(frame.profile) != self.report['streams'][name]:
                raise ValueError('native stream geometry changed during capture')
            metadata = {}
            for key in ('frame_timestamp', 'sensor_timestamp', 'actual_exposure', 'gain_level',
                        'auto_exposure', 'frame_counter', 'backend_timestamp', 'actual_fps'):
                field = getattr(self.rs.frame_metadata_value, key, None)
                metadata[key] = (int(frame.get_frame_metadata(field))
                                 if field is not None and frame.supports_frame_metadata(field) else None)
            timestamp_ms = float(frame.get_timestamp())
            if not np.isfinite(timestamp_ms):
                raise ValueError('invalid SDK frame timestamp')
            result[name] = dict(image=np.asanyarray(frame.get_data()).copy(),
                frame_number=int(frame.get_frame_number()),
                timestamp_ms=timestamp_ms, timestamp_ns=int(round(timestamp_ms * 1_000_000)),
                timestamp_domain=str(frame.get_frame_timestamp_domain()), metadata=metadata)
        if self.device.first_depth_sensor().get_depth_scale() != self.report['depth_metres_per_unit']:
            raise ValueError('native depth units changed during capture')
        return result

    def close(self):
        errors = []
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception as error:
                errors.append(error)
            self.pipeline = None
        for sensor, option, original in reversed(self._restore):
            try:
                sensor.set_option(option, original)
            except Exception as error:
                errors.append(error)
        self._restore.clear()
        if errors:
            raise RuntimeError(f'camera cleanup/option restoration failed: {errors}')

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
