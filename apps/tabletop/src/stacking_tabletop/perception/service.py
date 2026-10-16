"""Persistent local perception: no motion, gripper, or controller commands."""

import argparse
from contextlib import ExitStack
import fcntl
from functools import lru_cache
import json
import os
from pathlib import Path
import select
import signal
import socket
import socketserver
import threading
from types import SimpleNamespace

from .service_client import service_directory


class PerceptionResources:
    def __init__(self, repository, *, wrist_camera=True):
        self.repository = Path(repository).resolve()
        self.wrist_enabled = wrist_camera
        self.application = None
        self.camera = None
        self.wrist = ExitStack()
        self.calibration_hash = None

    def inhand_application(self):
        if self.application is None:
            from stacking_tabletop.composition import TabletopApplication
            app = TabletopApplication.from_repository(self.repository)
            registrar = app.build_in_hand_registrar()
            catalog = registrar.catalog
            catalog.load = lru_cache(maxsize=len(catalog.object_ids))(catalog.load)
            self.application = SimpleNamespace(
                perception_profile=app.perception_profile,
                build_pcd_catalog=lambda: catalog,
                build_in_hand_registrar=lambda: registrar,
            )
        return self.application

    def wrist_ready(self, check):
        if not self.wrist_enabled:
            raise RuntimeError('wrist camera disabled in perception service')
        from stacking_tabletop.acquisition.hand_eye import HandEyeCalibration
        from stacking_tabletop.acquisition.hand_eye_capture import digest, warmup
        from stacking_tabletop.acquisition.realsense import RealSenseCamera
        from stacking_tabletop.ros.hand_eye import HandEyeRobot
        if self.camera is None:
            try:
                path = self.repository / 'apps/tabletop/config/hand_eye.yml'
                self.calibration = HandEyeCalibration.load(path)
                self.calibration_hash = digest(path)
                self.robot = self.wrist.enter_context(HandEyeRobot(self.calibration))
                self.camera = RealSenseCamera(self.calibration, depth_size=(848, 480), fps=15)
                self.wrist.callback(self.camera.close)
            except Exception:
                self.close_wrist()
                raise
        camera = self.camera
        resources = self
        class CheckedCamera:
            report = camera.report
            def read(self):
                check()
                try:
                    result = camera.read()
                except Exception:
                    resources.close_wrist()
                    raise
                check()
                return result
        try:
            warmup(CheckedCamera(), count=3)
        except Exception:
            self.close_wrist()
            raise
        return CheckedCamera()

    def close_wrist(self):
        self.wrist.close()
        self.wrist = ExitStack()
        self.camera = None

    def close(self):
        self.close_wrist()

    def dispatch(self, operation, arguments, check):
        check()
        if operation == 'inhand':
            import numpy as np
            from .inhand_execution import identify_inhand_capture
            result = identify_inhand_capture(
                self.inhand_application(), arguments['capture'], arguments['object_id'],
                np.asarray(arguments['initial_tool_T_object'], dtype=float), check=check,
            )
            return result.report()
        if operation in ('wrist_ready', 'wrist_capture'):
            from stacking_tabletop.acquisition.hand_eye_capture import capture_view
            camera = self.wrist_ready(check)
            if (arguments['profile_sha256'] != self.calibration_hash
                    or arguments['serial'] != self.calibration.profile.serial):
                raise ValueError('wrist calibration changed; restart the perception service')
            if operation == 'wrist_capture':
                capture_view(arguments['directory'], self.calibration, camera, self.robot, count=3)
            return {'ready': True}
        if operation == 'stack_scene':
            from stacking_tabletop.acquisition.stack_scene_processing import process_stack_scene
            from stacking_tabletop.acquisition.hand_eye_fast_stereo import reconstruct_with_runtime
            def reconstruct(*args, **kwargs):
                return reconstruct_with_runtime(*args, **kwargs, check=check)
            return process_stack_scene(repository=self.repository, reconstruct=reconstruct, **arguments)
        if operation == 'refine_remaining':
            from stacking_tabletop.execution.scene_refinement import refine_execution_run
            return refine_execution_run(repository=self.repository, check=check, **arguments)
        raise ValueError(f'unsupported perception operation: {operation}')


class PerceptionServer(socketserver.ThreadingUnixStreamServer):
    # Close waits for handlers; cooperative checks interrupt camera/processing waits.
    daemon_threads = False

    def __init__(self, address, resources):
        self.resources = resources
        self.busy = threading.Lock()
        self.stopping = threading.Event()
        super().__init__(address, RequestHandler)


class RequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        message = {}
        self.request.settimeout(5.)
        def check():
            if self.server.stopping.is_set():
                raise InterruptedError('perception service stopping')
            readable, _, _ = select.select([self.request], [], [], 0)
            if readable and not self.request.recv(1, socket.MSG_PEEK):
                raise InterruptedError('perception client disconnected')
        try:
            line = self.rfile.readline(65537)
            if len(line) > 65536 or not line.endswith(b'\n'):
                raise ValueError('invalid perception request size')
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError('perception request must be an object')
            message = parsed
            if message['operation'] == 'ping':
                result = {'ready': True, 'busy': self.server.busy.locked()}
            else:
                if not self.server.busy.acquire(blocking=False):
                    raise RuntimeError('perception service busy; retry after the current job')
                try:
                    result = self.server.resources.dispatch(message['operation'], message['arguments'], check)
                    check()
                finally:
                    self.server.busy.release()
            reply = dict(id=message.get('id'), ok=True, result=result)
        except Exception as error:
            reply = dict(id=message.get('id'), ok=False, error=f'{type(error).__name__}: {error}')
        try:
            self.wfile.write(json.dumps(reply, allow_nan=False).encode() + b'\n')
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--wrist-camera', choices=['true', 'false'], default='true')
    args = parser.parse_args()
    directory = service_directory(args.repository)
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.is_symlink() or directory.stat().st_uid != os.getuid():
        raise RuntimeError('perception service directory is not owned by this user')
    directory.chmod(0o700)
    try:
        os.nice(10)
    except OSError:
        pass
    with (directory / 'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        address = directory / 'service.sock'
        address.unlink(missing_ok=True)  # exclusive lock proves no service owns it
        resources = PerceptionResources(args.repository, wrist_camera=args.wrist_camera == 'true')
        try:
            with PerceptionServer(str(address), resources) as server:
                def stop(*_):
                    server.stopping.set()
                    threading.Thread(target=server.shutdown, daemon=True).start()
                signal.signal(signal.SIGTERM, stop)
                signal.signal(signal.SIGINT, stop)
                def warm():
                    with server.busy:
                        for name, initialize in [
                            ('in-hand registration', resources.inhand_application),
                            *([('wrist camera', lambda: resources.wrist_ready(lambda: None))]
                              if resources.wrist_enabled else []),
                        ]:
                            try:
                                initialize()
                                print(f'[perception] {name} ready', flush=True)
                            except Exception as error:
                                print(f'[perception] {name} unavailable: {error}; requests may retry', flush=True)
                warmer = threading.Thread(target=warm, name='perception-warmup')
                warmer.start()
                print(f'[perception] listening on {address}', flush=True)
                server.serve_forever(poll_interval=.1)
                warmer.join()
        finally:
            resources.close()
            address.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
