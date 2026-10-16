"""Cancellable JSON requests to the local, hardware-launch-owned perception process."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
from uuid import uuid4


def service_directory(repository):
    key = hashlib.sha256(str(Path(repository).resolve()).encode()).hexdigest()[:12]
    domain = int(os.environ.get('ROS_DOMAIN_ID', '0'))
    return Path(tempfile.gettempdir()) / f'stacking-perception-{os.getuid()}-{domain}-{key}'


def request(repository, operation, *, check=lambda: None, timeout=120., **arguments):
    message = dict(id=uuid4().hex, operation=operation, arguments=arguments)
    deadline = time.monotonic() + timeout
    check()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(.2)
        try:
            connection.connect(str(service_directory(repository) / 'service.sock'))
        except OSError as error:
            raise RuntimeError('Perception service unavailable; start hardware.launch.py '
                               'with start_perception:=true') from error
        connection.sendall(json.dumps(message, allow_nan=False).encode() + b'\n')
        received = bytearray()
        while b'\n' not in received:
            check()
            if time.monotonic() >= deadline:
                raise TimeoutError(f'perception request {operation} timed out')
            try:
                chunk = connection.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                raise RuntimeError('perception service disconnected')
            received.extend(chunk)
            if len(received) > 4 * 1024 * 1024:
                raise ValueError('perception response exceeds size limit')
        check()
        reply = json.loads(received)
        if reply.get('id') != message['id']:
            raise ValueError('perception response does not match request')
        if not reply.get('ok'):
            raise RuntimeError(reply.get('error', 'perception request failed'))
        return reply['result']


@contextmanager
def server_wrist(executor, calibration, repository):
    from stacking_tabletop.acquisition.hand_eye_capture import digest
    # Check the persistent camera and the exact calibration before scan motion.
    # Hash comparison prevents a stale service from using a different calibration.
    profile = Path(repository) / 'apps/tabletop/config/hand_eye.yml'
    request(repository, 'wrist_ready', check=executor._health,
            profile_sha256=digest(profile), serial=calibration.profile.serial)
    def capture(directory):
        before = executor.runtime.paused_seconds
        request(repository, 'wrist_capture', check=executor._health,
                directory=str(Path(directory).resolve()), profile_sha256=digest(profile),
                serial=calibration.profile.serial)
        if executor.runtime.paused_seconds != before:
            raise RuntimeError('hold interrupted wrist capture; partial scene retained for review')
    yield capture


class RemoteStackProcessing:
    """Submit perception while the executor independently returns the arm home."""

    def __init__(self, saved, repository, output, step):
        self.output = Path(output)
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self.error = None
        def check():
            if self.cancelled.is_set():
                raise InterruptedError('stack perception cancelled')
        def run():
            try:
                request(repository, 'stack_scene', check=check, timeout=1800.,
                        run=str(saved.path), capture=str(self.output), step=int(step))
            except Exception as error:
                self.error = error
            finally:
                self.done.set()
        self.thread = threading.Thread(target=run, daemon=True, name='stack-perception-client')
        self.thread.start()

    def abort(self):
        self.cancelled.set()
        self.thread.join(timeout=1.)

    def finish(self, executor):
        from stacking_common.execution import ExecutionFailure
        from stacking_tabletop.execution.runtime import HardwareError
        from stacking_tabletop.acquisition.stack_scene_capture import _finish_stack_scene
        while not self.done.is_set():
            executor.runtime.wait(.1, executor._health)
        if self.error is not None:
            raise HardwareError(ExecutionFailure.PERCEPTION_FAILED, str(self.error))
        return _finish_stack_scene(executor, self.output)
