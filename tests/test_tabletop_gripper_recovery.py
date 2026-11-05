"""Recovery cannot authorize a grasp, replay goals, or clear an old session."""

import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace as NS

import pytest

from stacking_tabletop.execution.gripper_recovery import recover_gripper
from test_tabletop_hardware_adapters import FakeTransport, Future

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'driver_gripper_recovery', ROOT / 'apps/tabletop/driver/gripper_recovery.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class Packet:
    def __init__(self, *, persists=False, lost_ack=False):
        self.registers = {34: 95, 32: 160, 144: 118, 31: 80, 146: 37,
                          64: 0, 70: 0x21, 132: 1003}
        self.calls = []
        self.persists, self.lost_ack = persists, lost_ack

    def read1ByteTxRx(self, port, device, address):
        self.calls.append(('read', address))
        return self.registers[address], 0, 128 if self.registers[70] else 0

    read2ByteTxRx = read4ByteTxRx = read1ByteTxRx

    def write1ByteTxRx(self, port, device, address, value):
        self.calls.append(('write', address, value))
        self.registers[address] = value
        return 0, 128 if self.registers[70] else 0

    def reboot(self, port, device):
        self.calls.append(('reboot',))
        if not self.persists:
            self.registers[70] = 0
        return (-3002, 0) if self.lost_ack else (0, 128)


def reset(packet):
    clock = Clock()
    return driver.reboot_and_check(packet, object(), 1, clock=clock, sleep=clock.sleep)


def test_reboot_verifies_health_without_enabling_torque_or_position_write():
    packet = Packet()
    assert reset(packet) == dict(position=1003, voltage=11.8, temperature=37)
    assert [c for c in packet.calls if c[0] != 'read'] == [('write', 64, 0), ('reboot',)]
    assert sum(c == ('read', 132) for c in packet.calls) >= 2


@pytest.mark.parametrize('register,value,reason', [(144, 91, 'voltage'), (146, 80, 'temperature')])
def test_bad_environment_refuses_reboot(register, value, reason):
    packet = Packet()
    packet.registers[register] = value
    with pytest.raises(RuntimeError, match=reason):
        reset(packet)
    assert not any(c[0] in ('write', 'reboot') for c in packet.calls)


@pytest.mark.parametrize('packet,reason', [(Packet(persists=True), '0x21'),
                                          (Packet(lost_ack=True), 'not acknowledged')])
def test_reboot_failure_never_enables_torque(packet, reason):
    with pytest.raises(RuntimeError, match=reason):
        reset(packet)
    assert all(c != ('write', 64, 1) for c in packet.calls)
    assert packet.calls.count(('reboot',)) == 1


def test_post_reboot_position_must_be_commissioned():
    packet = Packet()
    packet.registers[132] = 4095
    with pytest.raises(RuntimeError, match='outside commissioned'):
        reset(packet)


class Host(driver.GripperRecoveryMixin):
    def __init__(self):
        self._state_condition = threading.Condition()
        self._bus_lock = threading.Lock()
        self._recovering = self._shutdown_requested = False
        self._fault_message = 'old fault'
        self._pending_command = (1, 91, 'old close')
        self._command_generation = 1
        self._command_results = {1: (False, 'old')}
        self.packetHandler, self.portHandler, self.DXL_ID = Packet(), object(), 1
        self._confirmed_hold_goal = 800
        self.reports, self.subscriptions = [], []
        self.worker_runs = 0

    def _replace_command_subscriptions(self, *, enabled):
        self.subscriptions.append(enabled)

    def _publish_fault_text(self, reason):
        self.reports.append(reason)

    def _set_confirmed_hold_goal(self, goal, **kw):
        self._confirmed_hold_goal = goal

    def _command_worker(self):
        self.worker_runs += 1


def test_driver_invalidates_old_commands_and_hold_and_restarts_worker(monkeypatch):
    host = Host()
    monkeypatch.setattr(driver, 'reboot_and_check', lambda *args: dict(position=1003, voltage=11.8, temperature=37))
    result = host._recover_callback(None, NS())
    host._command_worker_thread.join(1.)
    assert result.success and host._fault_message is None
    assert host._confirmed_hold_goal is None and host._pending_command is None
    assert host._command_results == {} and host._command_generation == 2
    assert host._last_state is None and host._torque_needs_enable
    assert host.subscriptions == [False, True] and host.reports == ['']
    assert host.worker_runs == 1 and not host._recovering


def test_driver_failure_retains_fault_and_does_not_recreate_command_readers(monkeypatch):
    host = Host()
    def fail(*args):
        raise RuntimeError('persistent overload')
    monkeypatch.setattr(driver, 'reboot_and_check', fail)
    result = host._recover_callback(None, NS())
    assert not result.success and 'persistent overload' in host._fault_message
    assert host.reports == [host._fault_message] and host.subscriptions == [False]
    assert host.worker_runs == 0 and not host._recovering
    assert host._confirmed_hold_goal is None


def test_healthy_driver_rejects_unnecessary_reboot():
    host = Host()
    host._fault_message = None
    assert not host._recover_callback(None, NS()).success
    assert host.subscriptions == [] and host.packetHandler.calls == []


def test_stuck_command_worker_blocks_reboot(monkeypatch):
    host = Host()
    host._command_worker_thread = NS(join=lambda **kw: None, is_alive=lambda: True)
    monkeypatch.setattr(driver, 'reboot_and_check', lambda *args: pytest.fail('worker still running'))
    result = host._recover_callback(None, NS())
    assert not result.success and 'did not stop' in result.message


def test_first_explicit_command_seeds_position_before_torque():
    host = Host()
    host._torque_needs_enable = True
    calls = []
    host._read_hw_error = lambda: 0
    host._read4_allow_alert = lambda *args: 1003
    host._write_goal_position = lambda goal: calls.append(('goal', goal))
    host._enable_torque_safely = lambda: calls.append(('enable',))
    host._read1_allow_alert = lambda *args: 1
    host._prepare_recovered_command()
    assert calls == [('goal', 1003), ('enable',)] and not host._torque_needs_enable
    host._prepare_recovered_command()
    assert len(calls) == 2


def transports(*, success=True, clear=True, refault=False, pending=False):
    old, new = FakeTransport(), FakeTransport()
    old.feedback.receive_fault('latched fault')
    old.recovery_client = NS(service_is_ready=lambda: True)
    future = Future(success=success, done=not pending)
    old.request_gripper_recovery = lambda: future
    old.closed = new.closed = False
    old.close = lambda: setattr(old, 'closed', True)
    new.close = lambda: setattr(new, 'closed', True)
    # Sharing the injected clock makes freshness deterministic across connections.
    new.feedback.clock = lambda: old.now
    old.on_sleep = new.refresh
    new.feedback.receive_hold(0)
    if clear:
        new.feedback.receive_fault('')
    if refault:
        new.feedback.receive_fault('new overload')
    return old, new, future


def test_client_uses_fresh_session_and_preserves_old_fault():
    old, new, _ = transports()
    sessions = iter([old, new])
    assert recover_gripper(lambda: next(sessions), runtime=old.runtime) == 'driver response'
    with pytest.raises(RuntimeError, match='latched fault'):
        old.feedback.check_gripper()
    assert old.closed and new.closed and old.calls == new.calls == []


@pytest.mark.parametrize('options,reason', [({'success': False}, 'driver response'),
    ({'clear': False}, 'fresh fault-clear'), ({'refault': True}, 'new overload'),
    ({'pending': True}, 'timed out')])
def test_client_rejects_unsuccessful_or_unverified_reset(options, reason):
    old, new, future = transports(**options)
    sessions = iter([old, new])
    with pytest.raises(RuntimeError, match=reason):
        recover_gripper(lambda: next(sessions), runtime=old.runtime)
    assert old.closed and old.calls == new.calls == []
    if options.get('pending'):
        assert future.cancelled


def test_empty_fault_message_cannot_clear_active_session_latch():
    old, _, _ = transports()
    old.feedback.receive_fault('')
    with pytest.raises(RuntimeError, match='latched fault'):
        old.feedback.check_gripper()


def test_existing_hold_cannot_authorize_recovery_client():
    old, new, _ = transports()
    new.feedback.receive_hold(800)
    sessions = iter([old, new])
    with pytest.raises(RuntimeError, match='hold-clear'):
        recover_gripper(lambda: next(sessions), runtime=old.runtime)
    assert old.calls == new.calls == []


def test_missing_service_does_not_send_reboot():
    old, _, _ = transports()
    old.recovery_client.service_is_ready = lambda: False
    old.request_gripper_recovery = lambda: pytest.fail('missing service')
    with pytest.raises(RuntimeError, match='service unavailable'):
        recover_gripper(lambda: old, runtime=old.runtime)
    assert old.closed
