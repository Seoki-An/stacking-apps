"""One Qt worker owns the hardware session and its execution journal writer."""

import threading

from PySide6.QtCore import QThread, Signal

from stacking_tabletop.execution import AdapterRuntime, DynamixelAdapter, UR5eAdapter, UR5eKinematics
from stacking_tabletop.execution.sequence import TabletopMotionExecutor, execute_saved_run


def connect_devices(saved, runtime):
    from stacking_tabletop.ros.hardware import TabletopRosTransport
    kinematics = UR5eKinematics(saved.assets["robot/urdf"])
    transport = TabletopRosTransport()
    gripper = DynamixelAdapter(transport, runtime=runtime)
    arm = UR5eAdapter(transport, kinematics, gripper=gripper, runtime=runtime)
    return transport, arm, gripper


class ExecutionWorker(QThread):
    scan_prepared = Signal(object)
    gate_changed = Signal(object)
    telemetry = Signal(object)
    event_received = Signal(object)
    plan_selected = Signal(object)
    failed = Signal(str)
    completed = Signal(object)

    def __init__(self, saved, parent=None, *, device_factory=connect_devices, camera_factory=None, repository=None):
        super().__init__(parent)
        self.saved = saved
        self.device_factory = device_factory
        self.camera_factory = camera_factory
        self.repository = repository
        self.runtime = AdapterRuntime(cancel_event=threading.Event())
        self.executor = None

    def abort(self):
        self.runtime.cancel_event.set()

    def _scan(self, executor, request):
        from contextlib import ExitStack
        from stacking_tabletop.acquisition.scanning import scan_held_object
        with ExitStack() as resources:
            if self.camera_factory is not None:
                camera = resources.enter_context(self.camera_factory(request[1], self.runtime))
            else:
                from pathlib import Path
                from stacking_tabletop.config.femto import load_femto_profile
                from stacking_tabletop.acquisition.femto import FemtoCalibration, FemtoCamera
                from stacking_tabletop.ros.femto import FemtoRosTransport
                if self.repository is None:
                    raise ValueError('scan worker needs the tabletop repository path')
                profile = load_femto_profile(Path(self.repository) / 'apps/tabletop/config/femto.yml', request[1])
                calibration = FemtoCalibration(profile)
                transport = resources.enter_context(FemtoRosTransport(calibration))
                camera = FemtoCamera(transport, calibration, cancel_event=self.runtime.cancel_event)
            scan_held_object(executor, request, camera, prepared=self.scan_prepared.emit)

    def run(self):
        transport = None
        try:
            self.saved.verify_assets()
            if self.runtime.cancel_event.is_set():
                return
            transport, arm, gripper = self.device_factory(self.saved, self.runtime)
            self.executor = TabletopMotionExecutor(self.saved, arm, gripper,
                gate_changed=self.gate_changed.emit, telemetry=self.telemetry.emit, scan_handler=self._scan)
            results = execute_saved_run(self.saved, self.executor,
                events=self.event_received.emit, next_step=self.plan_selected.emit)
            self.completed.emit(results)
        except Exception as error:
            self.failed.emit(str(error))
        finally:
            self.executor = None
            if transport is not None:
                try:
                    transport.close()
                except Exception as error:
                    self.failed.emit(f"Hardware session cleanup failed: {error}")
