"""Software recovery for the single-owner ur5e_admittance gripper driver.

Installed beside gripper_node.py; deliberately independent of ROS imports so
register ordering and fault handling can be tested without a connected motor.
Recovery never enables torque or writes a position goal.
"""

import threading
import time


def reboot_and_check(packet, port, device_id, *, clock=time.monotonic, sleep=time.sleep):
    """Reboot once, then require healthy registers with torque off for 0.25 s."""
    def read(address, size, *, allow_alert=False):
        value, comm, error = getattr(packet, f'read{size}ByteTxRx')(port, device_id, address)
        if comm != 0 or error not in ((0, 128) if allow_alert else (0,)):
            raise RuntimeError(f'read register {address}: communication={comm}, error={error}')
        return value

    def environment(*, allow_alert=False):
        low = read(34, 2, allow_alert=allow_alert)
        high = read(32, 2, allow_alert=allow_alert)
        voltage = read(144, 2, allow_alert=allow_alert)
        limit = read(31, 1, allow_alert=allow_alert)
        temperature = read(146, 1, allow_alert=allow_alert)
        if not 0 < low < high or not low <= voltage <= high:
            raise RuntimeError(f'voltage {voltage / 10:g} V outside {low / 10:g}..{high / 10:g} V')
        if not 0 < limit or temperature >= limit:
            raise RuntimeError(f'temperature {temperature} C reaches limit {limit} C')
        return voltage, temperature

    environment(allow_alert=True)
    comm, error = packet.write1ByteTxRx(port, device_id, 64, 0)
    if comm != 0 or error not in (0, 128) or read(64, 1, allow_alert=True) != 0:
        raise RuntimeError('cannot confirm torque off before reboot')
    comm, error = packet.reboot(port, device_id)
    if comm != 0 or error not in (0, 128):
        raise RuntimeError(f'reboot not acknowledged: communication={comm}, error={error}')

    deadline, healthy_since, last_error = clock() + 8., None, 'waiting for actuator'
    sleep(.25)
    while clock() < deadline:
        try:
            hardware = read(70, 1, allow_alert=True)
            if hardware:
                raise RuntimeError(f'Hardware Error Status remains 0x{hardware:02X}')
            if read(64, 1) != 0:
                raise RuntimeError('torque is unexpectedly enabled after reboot')
            voltage, temperature = environment()
            position = read(132, 4)
            if not 91 <= position <= 1500:
                raise RuntimeError(f'position {position} outside commissioned range 91..1500')
            if healthy_since is None:
                healthy_since = clock()
            if clock() - healthy_since >= .25:
                return dict(position=position, voltage=voltage / 10., temperature=temperature)
        except RuntimeError as error:
            last_error, healthy_since = str(error), None
        sleep(.05)
    raise RuntimeError(f'reboot verification timed out: {last_error}')


class GripperRecoveryMixin:
    """A fault remains latched until the actuator and command worker recover."""

    def _recover_callback(self, _request, response):
        response.success = False
        with self._state_condition:
            if self._recovering or self._shutdown_requested:
                response.message = 'Recovery unavailable during recovery/shutdown'
                return response
            if self._fault_message is None:
                response.message = 'Gripper is not faulted; reboot was not requested'
                return response
            self._recovering = True
            self._pending_command = None
            self._state_condition.notify_all()
        try:
            self._set_confirmed_hold_goal(None, force_publish=True)
            # Destroy command readers before clearing the fault: buffered goals
            # from the failed execution must not replay after reboot.
            self._replace_command_subscriptions(enabled=False)
            worker = getattr(self, '_command_worker_thread', None)
            if worker is not None:
                worker.join(timeout=3.)
                if worker.is_alive():
                    raise RuntimeError('previous command worker did not stop')
            with self._bus_lock:
                state = reboot_and_check(self.packetHandler, self.portHandler, self.DXL_ID)
            with self._state_condition:
                self._active_generation = self._active_goal = None
                self._command_generation += 1
                self._command_results.clear()
                self._last_state = self._last_state_time = None
                self._last_velocity = self._last_current = float('nan')
                self._input_voltage_alert_since = None
                self._input_voltage_relief_installed = False
                self._input_voltage_recovery_reported = False
                self._known_retained_voltage_alert = False
                self._consecutive_position_read_errors = 0
                self._set_confirmed_hold_goal(None, force_publish=True)
                self._hold_recovery_attempts = self._hold_watchdog_error_count = 0
                self._hold_recovery_reference_position = None
                self._torque_needs_enable = True
                self._replace_command_subscriptions(enabled=True)
                self._fault_message = None
                self._command_worker_thread = threading.Thread(
                    target=self._command_worker, name='dynamixel-command-worker', daemon=True)
                self._command_worker_thread.start()
                self._publish_fault_text('')
            response.success = True
            response.message = (
                f"Recovered: {state['voltage']:g} V, {state['temperature']} C, "
                f"position {state['position']}; torque OFF. Inspect the scene before retry.")
        except Exception as error:
            with self._state_condition:
                self._fault_message = f'Gripper recovery failed: {error}'
                self._state_condition.notify_all()
            self._publish_fault_text(self._fault_message)
            response.message = self._fault_message
        finally:
            with self._state_condition:
                self._recovering = False
        return response

    def _prepare_recovered_command(self):
        """Only an explicit later command may re-enable torque after recovery."""
        if not self._torque_needs_enable:
            return
        hardware = self._read_hw_error()
        if hardware:
            raise RuntimeError(f'recovered actuator faulted again: 0x{hardware:02X}')
        position = self._read4_allow_alert(132, 'read recovered position')
        if not 91 <= position <= 1500:
            raise RuntimeError(f'recovered position {position} is outside range')
        # A reboot can reset the goal register. Seed the measured position
        # before enabling torque so an old/default goal cannot command a jump.
        self._write_goal_position(position)
        self._enable_torque_safely()
        torque = self._read1_allow_alert(64, 'verify recovered torque')
        if torque != 1:
            raise RuntimeError('torque did not enable for the requested command')
        self._torque_needs_enable = False
