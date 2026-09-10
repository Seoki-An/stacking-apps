"""Device behavior against deterministic telemetry, without ROS or hardware."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stacking_common.execution import ExecutionFailure
from stacking_tabletop.execution import (
    AdapterRuntime, CartesianStreamLimits, DynamixelAdapter, HardwareError,
    HardwareFeedback, UR5eAdapter, UR5eKinematics, UR_JOINT_NAMES,
)
from stacking_tabletop.execution.motion import sample_joint_path, pose_error
from stacking_tabletop.execution.ur5e import CONTACT_ADMITTANCE_GAINS, RIGID_ADMITTANCE_GAINS


class Kinematics:
    def pose(self, q):
        result = np.eye(4)
        result[0, 3] = q[0]
        result[:3, :3] = Rotation.from_rotvec([0, 0, q[1]]).as_matrix()
        return result


class Future:
    def __init__(self, *, success=True, done=True):
        self.success = success
        self.complete = done
        self.cancelled = False

    def done(self):
        return self.complete

    def result(self):
        return SimpleNamespace(success=self.success, message="driver response")

    def cancel(self):
        self.cancelled = True


class FakeTransport:
    def __init__(self):
        self.now = 10.0
        self.live = True
        self.arm_online = self.gripper_online = True
        self.arm_telemetry = self.gripper_telemetry = True
        self.follow = True
        self.clear_hold = self.ack_hold = True
        self.joints = np.zeros(6)
        self.pose = np.eye(4)
        self.position = 1500
        self.contact_position = None
        self.calls = []
        self.on_sleep = None
        self.feedback = HardwareFeedback(clock=lambda: self.now)
        self.feedback.receive_ready(True)
        self.future = Future()
        self.runtime = AdapterRuntime(clock=lambda: self.now, sleep=self.sleep)
        self.refresh()

    def refresh(self):
        if self.arm_telemetry:
            self.feedback.receive_joints(UR_JOINT_NAMES, self.joints)
            self.feedback.receive_pose(self.pose)
        if self.gripper_telemetry:
            self.feedback.receive_gripper(self.position)

    def sleep(self, duration):
        self.now += duration
        if self.on_sleep is not None:
            self.on_sleep()
        self.refresh()

    def ok(self):
        return self.live

    def arm_connected(self):
        return self.arm_online

    def gripper_connected(self):
        return self.gripper_online

    def publish_pose(self, pose):
        self.calls.append(("pose", self.now, pose.copy()))
        if self.follow:
            self.pose = pose.copy()
            self.joints[0] = pose[0, 3]
            self.joints[1] = Rotation.from_matrix(pose[:3, :3]).as_rotvec()[2]
        self.refresh()

    def publish_gains(self, gains):
        self.calls.append(("gains", self.now, tuple(gains)))

    def publish_wrench(self, enabled):
        self.calls.append(("wrench", self.now, enabled))

    def publish_gripper_flag(self, flag):
        self.calls.append(("flag", self.now, flag))
        self.position = 1500 if flag == 2 else 91
        self.feedback.receive_hold(0)

    def publish_gripper_position(self, position):
        self.calls.append(("position", self.now, position))
        self.position = max(position, self.contact_position or 0)
        if self.clear_hold:
            self.feedback.receive_hold(0)

    def request_gripper_hold(self):
        self.calls.append(("hold", self.now, None))
        if self.ack_hold:
            self.feedback.receive_hold(int(self.position))
        return self.future


def devices():
    transport = FakeTransport()
    gripper = DynamixelAdapter(transport, runtime=transport.runtime)
    arm = UR5eAdapter(transport, Kinematics(), gripper=gripper, runtime=transport.runtime)
    return transport, gripper, arm


def path():
    end = np.zeros(6)
    end[:2] = [0.02, 0.03]
    return [np.zeros(6), end]


def test_construct_and_ready_never_command_devices():
    transport, gripper, arm = devices()
    arm.wait_until_ready()
    gripper.wait_until_ready()
    assert transport.calls == []


def test_arm_inherits_gripper_cancellation_and_rejects_unrelated_transport():
    transport, gripper, _ = devices()
    arm = UR5eAdapter(transport, Kinematics(), gripper=gripper)
    assert arm.runtime is gripper.runtime
    with pytest.raises(ValueError, match="share"):
        UR5eAdapter(FakeTransport(), Kinematics(), gripper=gripper)


def test_hold_service_exception_is_a_gripper_failure():
    transport, gripper, _ = devices()
    def fail():
        raise OSError("service connection lost")
    transport.future.result = fail
    with pytest.raises(HardwareError) as error:
        gripper.close(0.45)
    assert error.value.failure is ExecutionFailure.GRIPPER_FAULT
    assert not transport.feedback.gripper_state().holding


def test_arm_deadline_covers_stream_and_does_not_catch_up_after_slow_callback():
    transport, _, arm = devices()
    def slow(*_):
        transport.sleep(0.05)
    with pytest.raises(HardwareError) as error:
        arm.execute_path(path(), timeout=.2, progress=slow)
    assert error.value.failure is ExecutionFailure.TIMEOUT
    moving = [c for c in transport.calls if c[0] == "pose"][:-3]  # Final three request hold.
    assert len(moving) >= 2
    assert all(b[1] - a[1] >= .05 - 1e-9 for a, b in zip(moving, moving[1:]))


def test_close_requires_fresh_hold_then_attached_path_and_explicit_release():
    transport, gripper, arm = devices()
    gripper.close(0.45)
    assert transport.feedback.gripper_state().holding
    assert abs(next(c[2] for c in reversed(transport.calls) if c[0] == "position") - 169) <= 20
    gripper.require_hold()
    arm.execute_path(path(), attached=True)
    assert not any(c[0] == "flag" for c in transport.calls)
    gripper.open()
    assert not transport.feedback.gripper_state().holding
    assert transport.calls[-1][0] == "flag"
    assert transport.calls[-1][2] == 2


def test_stable_contact_stops_closing_above_goal_and_installs_hold():
    transport, gripper, _ = devices()
    transport.contact_position = 600
    gripper.close(0.45)
    state = transport.feedback.gripper_state()
    assert state.holding and state.position == 600 and state.hold_goal == 600


@pytest.mark.parametrize("clear,ack", [(False, True), (True, False)])
def test_cached_hold_or_service_success_without_ack_cannot_authorize_lift(clear, ack):
    transport, gripper, arm = devices()
    transport.feedback.receive_hold(500)
    transport.clear_hold, transport.ack_hold = clear, ack
    with pytest.raises(HardwareError, match="freshly"):
        gripper.close(0.45)
    assert not transport.feedback.gripper_state().holding
    with pytest.raises(HardwareError, match="acknowledged"):
        arm.execute_path(path(), attached=True)
    assert all(np.allclose(c[2], np.eye(4)) for c in transport.calls if c[0] == "pose")


@pytest.mark.parametrize("success,done", [(False, True), (True, False)])
def test_hold_service_failure_or_timeout_never_arms_guard(success, done):
    transport, gripper, _ = devices()
    transport.future = Future(success=success, done=done)
    with pytest.raises(HardwareError):
        gripper.close(0.45)
    assert not transport.feedback.gripper_state().holding
    assert transport.future.cancelled == (not done)


def test_close_timeout_includes_ramp_and_cancel_does_not_open():
    transport, gripper, _ = devices()
    with pytest.raises(HardwareError) as error:
        gripper.close(0.45, timeout=0.15)
    assert error.value.failure is ExecutionFailure.TIMEOUT
    assert transport.now <= 10.15
    transport.on_sleep = lambda: transport.runtime.cancel_event.set()
    with pytest.raises(HardwareError) as error:
        gripper.close(0.45)
    assert error.value.failure is ExecutionFailure.CANCELLED
    assert not any(c[0] in ("hold", "flag") for c in transport.calls)


def test_invalid_gripper_samples_do_not_refresh_telemetry_or_confirm_open():
    transport, gripper, _ = devices()
    transport.gripper_telemetry = False
    transport.on_sleep = lambda: transport.feedback.receive_gripper(float("nan"))
    with pytest.raises(HardwareError, match="stale"):
        gripper.open()
    assert transport.feedback.gripper_state().sequence == 1


def test_duplicate_polled_samples_do_not_establish_contact():
    transport, gripper, _ = devices()
    transport.gripper_telemetry = False
    with pytest.raises(HardwareError, match="stale"):
        gripper.close(0.45)
    assert not any(c[0] == "hold" for c in transport.calls)


@pytest.mark.parametrize("fault", ["driver", "lost_hold", "changed_hold", "opening", "stale"])
def test_hold_faults_latch_and_block_attached_motion(fault):
    transport, gripper, arm = devices()
    gripper.close(0.45)
    if fault == "driver":
        transport.feedback.receive_fault("hardware overload")
        transport.feedback.receive_fault("")
    elif fault == "lost_hold":
        transport.feedback.receive_hold(0)
    elif fault == "changed_hold":
        transport.feedback.receive_hold(500)
    elif fault == "opening":
        for _ in range(5):
            transport.feedback.receive_gripper(300)
    else:
        transport.gripper_telemetry = False
        transport.sleep(3.1)
    with pytest.raises(HardwareError) as error:
        arm.execute_path(path(), attached=True)
    assert error.value.failure is ExecutionFailure.GRIPPER_FAULT
    with pytest.raises(HardwareError):
        gripper.require_hold()
    assert not any(c[0] == "flag" for c in transport.calls)


def test_arm_stream_is_bounded_and_feedback_is_not_commanded_state():
    transport, _, arm = devices()
    samples = []
    arm.execute_path(path(), progress=lambda i, n, q: samples.append((i, n, q)))
    assert samples[0][0] == 0 and samples[-1][0] == samples[-1][1] - 1
    commands = [c for c in transport.calls if c[0] == "pose"]
    for a, b in zip(commands, commands[1:]):
        dt = b[1] - a[1]
        position, angle = pose_error(a[2], b[2])
        assert dt >= 0.01 - 1e-9
        assert position <= 0.08 * dt + 1e-9
        assert angle <= 0.70 * dt + 1e-9
    np.testing.assert_allclose(transport.feedback.arm_state()[0], path()[-1])


@pytest.mark.parametrize("kind", ["cancel", "stale", "ready", "disconnect", "gripper_fault"])
def test_fault_during_motion_requests_rigid_actual_hold_without_release(kind):
    transport, _, arm = devices()

    def interrupt():
        if transport.now < 10.08:
            return
        if kind == "cancel":
            transport.runtime.cancel_event.set()
        elif kind == "stale":
            transport.arm_telemetry = False
        elif kind == "ready":
            transport.feedback.receive_ready(False)
        elif kind == "disconnect":
            transport.arm_online = False
        else:
            transport.feedback.receive_fault("gripper link lost")

    transport.on_sleep = interrupt
    with pytest.raises(HardwareError):
        arm.execute_path(path(), timeout=4.0)
    if kind != "stale":
        assert ("wrench", transport.calls[-2][1], False) == transport.calls[-2]
        assert transport.calls[-1][0] == "pose"
    else:
        assert arm.hold_actual_pose() is False
    assert not any(c[0] == "flag" for c in transport.calls)


def test_model_fk_mismatch_and_start_mismatch_rejected():
    transport, _, arm = devices()
    transport.pose[0, 3] = 0.02
    transport.refresh()
    with pytest.raises(HardwareError, match="FK mismatch"):
        arm.validate_start(np.zeros(6))
    transport.pose = np.eye(4)
    transport.joints[5] = 2 * np.pi
    transport.refresh()
    with pytest.raises(HardwareError, match="2 degrees"):
        arm.validate_start(np.zeros(6))


@pytest.mark.parametrize("value", [float("nan"), 5.5])
def test_bad_path_rejected_without_commands(value):
    transport, _, arm = devices()
    bad = np.zeros((2, 6))
    bad[1, 5] = value
    with pytest.raises(ValueError):
        arm.execute_path(bad)
    assert transport.calls == []


def test_raw_wrist_branch_jump_rejected():
    transport, _, arm = devices()
    bad = np.zeros((2, 6))
    bad[:, 5] = [-2, 2]
    with pytest.raises(ValueError, match="branch"):
        arm.execute_path(bad)
    assert transport.calls == []


def test_endpoint_timeout_and_profile_gains():
    transport, _, arm = devices()
    arm.set_contact_compliance()
    assert transport.now == pytest.approx(11.75)
    assert all(c[2] == CONTACT_ADMITTANCE_GAINS for c in transport.calls if c[0] == "gains")
    arm.set_rigid_tracking()
    assert transport.calls[-2][2] == RIGID_ADMITTANCE_GAINS
    transport.follow = False
    with pytest.raises(HardwareError, match="settle"):
        arm.execute_path(path(), settle_timeout=0.4)


def test_feedback_rejects_ambiguous_joints_and_invalid_rigid_pose():
    feedback = HardwareFeedback()
    assert not feedback.receive_joints([*UR_JOINT_NAMES, "arm/elbow_joint"], [0] * 7)
    invalid = np.eye(4)
    invalid[0, 0] = 2
    assert not feedback.receive_pose(invalid)


def test_nonlinear_fk_sampling_obeys_interval_limits():
    class Curved(Kinematics):
        def pose(self, q):
            result = super().pose(q)
            result[0, 3] = np.sin(q[0])
            return result

    limits = CartesianStreamLimits(rate_hz=20, max_joint_speed=1., max_tcp_speed=.1, max_angular_speed=.5)
    endpoint = np.array([2., .5, 0, 0, 0, 0])
    samples = sample_joint_path([np.zeros(6), endpoint], Curved(), limits)
    for (qa, a), (qb, b) in zip(samples, samples[1:]):
        distance, angle = pose_error(a, b)
        assert np.max(np.abs(qb - qa)) <= limits.max_joint_speed / limits.rate_hz + 1e-9
        assert distance <= limits.max_tcp_speed / limits.rate_hz + 1e-9
        assert angle <= limits.max_angular_speed / limits.rate_hz + 1e-9


def test_native_fk_uses_tool0_relative_to_base_not_virtual_grasp_frame():
    core = pytest.importorskip("stacking_core")
    urdf = Path(__file__).resolve().parents[1] / "apps/tabletop/assets/ur5e/ur5e_ik.urdf"
    kin = UR5eKinematics(urdf, core_module=core)
    q = np.deg2rad([90, -105, 85, -70, -90, 0])
    actual = kin.pose(q)
    state = core.KinematicState(1, kin.model.kinematics)
    state.positions = q
    tool_pose = np.asarray(state.link_pose(kin.model.kinematics.find_link("tool0")))
    grasp_pose = np.asarray(state.link_pose(kin.model.kinematics.find_link("grasp_frame")))
    np.testing.assert_allclose(actual[:3, 3], tool_pose[:3], atol=1e-10)
    assert not np.allclose(actual[:3, 3], grasp_pose[:3])


def test_native_controller_fk_is_independent_of_urdf_world_mount(tmp_path):
    import xml.etree.ElementTree as ET
    core = pytest.importorskip("stacking_core")
    urdf = Path(__file__).resolve().parents[1] / "apps/tabletop/assets/ur5e/ur5e_ik.urdf"
    tree = ET.parse(urdf)
    origin = tree.getroot().find("joint[@name='world_joint']/origin")
    origin.set("xyz", "1 2 3")
    origin.set("rpy", "0 0 0")
    repository = Path(__file__).resolve().parents[1]
    for mesh in tree.getroot().iter():
        if mesh.tag in {"mesh", "dsf_vert"} and "filename" in mesh.attrib:
            filename = mesh.attrib["filename"]
            asset = repository / filename
            if not asset.is_file():
                asset = urdf.parent / filename
            mesh.set("filename", str(asset.resolve()))
    mounted = tmp_path / "mounted.urdf"
    tree.write(mounted)
    q = np.deg2rad([90, -105, 85, -70, -90, 0])
    reference = UR5eKinematics(urdf, core_module=core).pose(q)
    actual = UR5eKinematics(mounted, core_module=core).pose(q)
    np.testing.assert_allclose(actual, reference, atol=1e-10)
