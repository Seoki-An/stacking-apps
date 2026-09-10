"""Cancellation, deadlines, and failures shared by the two device adapters."""

import threading
import time

import numpy as np

from stacking_common.execution import ExecutionFailure


class HardwareError(RuntimeError):
    """A device failure classified for the common execution journal."""

    def __init__(self, failure: ExecutionFailure, reason: str):
        super().__init__(reason)
        self.failure = ExecutionFailure(failure)


class AdapterRuntime:
    """One command worker; cancellation may be requested from another thread.

    Clock and sleep injection lets device tests run against simulated time.
    Device commands must be serialized by the owning tabletop coordinator.
    """

    def __init__(self, *, cancel_event=None, clock=time.monotonic, sleep=time.sleep):
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.clock = clock
        self.sleep = sleep
        self.poll = None
        self.paused_seconds = 0.0

    def active_time(self):
        """Motion time excludes supervised pauses, but feedback uses wall time."""
        return self.clock() - self.paused_seconds

    def check(self, transport):
        if self.cancel_event.is_set():
            raise HardwareError(ExecutionFailure.CANCELLED, "execution cancelled by operator")
        if not transport.ok():
            raise HardwareError(ExecutionFailure.CONTROLLER_FAULT, "ROS transport stopped")
        if self.poll is not None:
            self.poll()

    def deadline(self, timeout):
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        return self.clock() + timeout

    def wait(self, duration, check):
        end = self.clock() + max(0.0, duration)
        while True:
            check()
            remaining = end - self.clock()
            if remaining <= 0:
                return
            self.sleep(min(0.02, remaining))

    def until(self, predicate, deadline, check, reason):
        while True:
            check()
            if self.clock() >= deadline:
                raise HardwareError(ExecutionFailure.TIMEOUT, reason)
            if predicate():
                return
            remaining = deadline - self.clock()
            if remaining > 0:
                self.sleep(min(0.02, remaining))
