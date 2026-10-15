"""SDK contract and resource cleanup with native-profile/frame doubles."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from stacking_tabletop.acquisition.hand_eye import HandEyeCalibration
from stacking_tabletop.acquisition.realsense import RealSenseCamera, list_profiles
from test_tabletop_hand_eye_calibration import profile_path


class Profile:
    def __init__(self, width, height, matrix, format):
        self.intr = NS(width=width, height=height, fx=matrix[0, 0], fy=matrix[1, 1],
                       ppx=matrix[0, 2], ppy=matrix[1, 2], model='brown_conrady', coeffs=[0.] * 5)
        self.encoding = format
    def as_video_stream_profile(self):
        return self
    def get_intrinsics(self):
        return self.intr
    def fps(self):
        return 15
    def format(self):
        return self.encoding
    def get_extrinsics_to(self, profile):
        assert self.encoding == 'z16' and profile.encoding == 'rgb8'
        return NS(rotation=np.eye(3).flatten(), translation=[.015, 0., 0.])


class Sensor:
    def __init__(self, profile):
        self.profile = profile
        self.global_time = 0.
        self.scale = .00025
    def supports(self, option):
        return True
    def get_option(self, option):
        return self.global_time
    def set_option(self, option, value):
        self.global_time = value
    def get_supported_options(self):
        return ['global_time_enabled']
    def get_stream_profiles(self):
        return [self.profile]
    def get_depth_scale(self):
        return self.scale


def sdk(calibration):
    profiles = dict(color=Profile(1280, 720, calibration.factory_matrix, 'rgb8'),
                    depth=Profile(848, 480, np.diag([420., 420., 1.]), 'z16'))
    sensors = {name: Sensor(profile) for name, profile in profiles.items()}
    info = dict(serial_number=calibration.profile.serial, name='D435', firmware_version='test-fw', usb_type_descriptor='3.2')
    device = NS(get_info=lambda key: info[key], supports=lambda key: True,
                query_sensors=lambda: list(sensors.values()),
                first_color_sensor=lambda: sensors['color'], first_depth_sensor=lambda: sensors['depth'])
    active = NS(get_device=lambda: device, get_stream=lambda name: profiles[name])
    state = NS(starts=0, stops=0, requested=[], info=info, profiles=profiles, sensors=sensors, frame_number=10)
    class Frame:
        def __init__(self, name):
            self.name = name
            self.profile = profiles[name]
        def get_data(self):
            shape = (720, 1280, 3) if self.name == 'color' else (480, 848)
            return np.ones(shape, dtype=np.uint8 if self.name == 'color' else np.uint16)
        def get_frame_number(self):
            return state.frame_number
        def get_timestamp(self):
            return 1_800_000_000_001.25
        def get_frame_timestamp_domain(self):
            return 'timestamp_domain.global_time'
        def supports_frame_metadata(self, key):
            return key != 'sensor_timestamp'
        def get_frame_metadata(self, key):
            return 1234
    class Pipeline:
        def __init__(self, context):
            pass
        def start(self, config):
            state.starts += 1
            return active
        def stop(self):
            state.stops += 1
        def wait_for_frames(self, timeout_ms):
            state.frame_number += 1
            return NS(get_color_frame=lambda: Frame('color'), get_depth_frame=lambda: Frame('depth'))
    metadata = {key: key for key in ('frame_timestamp', 'sensor_timestamp', 'actual_exposure', 'gain_level',
                                   'auto_exposure', 'frame_counter', 'backend_timestamp', 'actual_fps')}
    api = NS(context=lambda: NS(query_devices=lambda: [device]), pipeline=Pipeline,
        config=lambda: NS(enable_device=lambda serial: state.requested.append(serial),
                          enable_stream=lambda *args: state.requested.append(args)),
        camera_info=NS(**{key: key for key in info}), option=NS(global_time_enabled='global_time_enabled'),
        stream=NS(color='color', depth='depth'), format=NS(rgb8='rgb8', z16='z16'),
        frame_metadata_value=NS(**metadata))
    return api, state


def test_native_profile_depth_units_metadata_and_option_restoration(profile_path):
    calibration = HandEyeCalibration.load(profile_path)
    api, state = sdk(calibration)
    assert list_profiles(calibration, rs=api)['serial'] == 'camera-A'
    assert state.starts == 0
    with RealSenseCamera(calibration, rs=api) as camera:
        assert state.requested == ['camera-A', ('color', 1280, 720, 'rgb8', 15), ('depth', 848, 480, 'z16', 15)]
        assert camera.report['depth_metres_per_unit'] == .00025
        assert camera.report['color_T_depth'][0][3] == .015
        frame = camera.read()
        assert frame['depth']['image'].dtype == np.uint16
        assert frame['color']['image'].shape == (720, 1280, 3)
        assert frame['color']['timestamp_ms'] == 1_800_000_000_001.25
        assert abs(frame['color']['timestamp_ns'] - 1_800_000_000_001_250_000) <= 256
        assert frame['depth']['metadata']['sensor_timestamp'] is None
        assert frame['depth']['metadata']['actual_exposure'] == 1234
        assert all(sensor.global_time == 1. for sensor in state.sensors.values())
    assert state.stops == 1
    assert all(sensor.global_time == 0. for sensor in state.sensors.values())
    camera.close()
    assert state.stops == 1


@pytest.mark.parametrize('mutation,reason', [
    (lambda s: s.info.update(serial_number='different-camera'), 'exactly one'),
    (lambda s: setattr(s.profiles['color'].intr, 'fx', 700.), 'factory intrinsics'),
    (lambda s: setattr(s.profiles['depth'].intr, 'width', 640), 'requested native profile'),
    (lambda s: setattr(s.sensors['depth'], 'scale', 0.), 'depth scale'),
    (lambda s: setattr(s.sensors['depth'], 'supports', lambda option: False), 'both sensors'),
])
def test_bad_live_binding_cleans_started_pipeline(profile_path, mutation, reason):
    calibration = HandEyeCalibration.load(profile_path)
    api, state = sdk(calibration)
    mutation(state)
    with pytest.raises(ValueError, match=reason):
        RealSenseCamera(calibration, rs=api)
    assert state.stops == state.starts
    assert all(sensor.global_time == 0. for sensor in state.sensors.values())


def test_mid_capture_geometry_or_depth_scale_change_is_rejected(profile_path):
    api, state = sdk(HandEyeCalibration.load(profile_path))
    with RealSenseCamera(HandEyeCalibration.load(profile_path), rs=api) as camera:
        state.profiles['depth'].intr.fx = 500.
        with pytest.raises(ValueError, match='geometry changed'):
            camera.read()
        state.profiles['depth'].intr.fx = 420.
        state.sensors['depth'].scale *= 2.
        with pytest.raises(ValueError, match='units changed'):
            camera.read()


def test_unresolved_stream_request_reports_pinned_device_and_modes(profile_path):
    calibration = HandEyeCalibration.load(profile_path)
    api, _state = sdk(calibration)
    class RejectedPipeline:
        def __init__(self, _context): pass
        def start(self, _config): raise RuntimeError("Couldn't resolve requests")
    api.pipeline = RejectedPipeline
    with pytest.raises(RuntimeError, match=r'pinned D435 camera-A.*1280x720.*848x480.*15 fps'):
        RealSenseCamera(calibration, rs=api)


@pytest.mark.parametrize('detected, detail', [
    ([], 'none (SDK enumerated zero devices)'),
    (['camera-B'], 'camera-B'),
    (['camera-A', 'camera-A'], 'matches=2'),
])
@pytest.mark.parametrize('operation', [list_profiles, RealSenseCamera])
def test_enumeration_failure_reports_identity_without_starting_streams(profile_path, detected, detail, operation):
    calibration = HandEyeCalibration.load(profile_path)
    api, state = sdk(calibration)
    devices = [NS(get_info=lambda key, serial=serial: serial) for serial in detected]
    api.context = lambda: NS(query_devices=lambda: devices)
    with pytest.raises(ValueError) as error:
        operation(calibration, rs=api)
    assert 'pinned D435 camera-A' in str(error.value)
    assert detail in str(error.value)
    assert state.starts == 0
