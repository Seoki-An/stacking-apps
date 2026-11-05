"""Explicit driver reboot followed by verification through a fresh ROS session."""

from .dynamixel import DynamixelAdapter
from .runtime import AdapterRuntime


def recover_gripper(transport_factory, *, runtime=None):
    """Caller confirms stopped motion and supported load before invoking this.

    No execution journal is changed. A faulted session remains faulted; only a
    new transport can verify the successful recovery. No position/torque command
    is published by this client, including on timeout or failed verification.
    """
    runtime = runtime or AdapterRuntime()
    transport = transport_factory()
    try:
        check = lambda: runtime.check(transport)
        runtime.until(transport.recovery_client.service_is_ready, runtime.deadline(5.), check,
                      'Gripper recovery service unavailable; restart the updated gripper driver')
        future = transport.request_gripper_recovery()
        try:
            runtime.until(future.done, runtime.deadline(15.), check,
                          'Gripper recovery timed out; inspect driver status before retrying')
            result = future.result()
            if result is None or not result.success:
                raise RuntimeError('No recovery response' if result is None else result.message)
            message = result.message
        finally:
            if not future.done():
                # Cancels client waiting, not a reboot already executing in the driver.
                future.cancel()
    finally:
        transport.close()

    transport = transport_factory()
    try:
        hand = DynamixelAdapter(transport, runtime=runtime)

        def ready():
            state = transport.feedback.gripper_state()
            return (transport.feedback.gripper_fault_report() == ''
                    and transport.gripper_connected() and state.position is not None
                    and runtime.clock() - state.stamp <= 1.
                    and state.hold_sequence > 0 and state.hold_goal is None)

        runtime.until(ready, runtime.deadline(5.), hand.check,
                      'Recovery reported success but fresh fault-clear/hold-clear/telemetry is missing')
        return message
    finally:
        transport.close()
