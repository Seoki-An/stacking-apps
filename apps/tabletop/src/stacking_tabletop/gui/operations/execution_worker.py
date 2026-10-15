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
        self.capture_stack_scene = False
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

    def _stack_scene(self, executor):
        from stacking_tabletop.acquisition.stack_scene_capture import collect_stack_scene
        from stacking_tabletop.acquisition.scene_capture_cli import ensure_ownership
        ensure_ownership(executor.arm.transport)
        return collect_stack_scene(executor, self.repository, prepared=self.scan_prepared.emit)

    def run(self):
        transport = None
        try:
            self.saved.verify_assets()
            if self.runtime.cancel_event.is_set():
                return
            from stacking_tabletop.execution.recovery import pending_released_stack_capture
            recovering_capture = any(
                pending_released_stack_capture(self.saved.path, plan.step) is not None
                for plan in self.saved.plans
            )
            capture_stack_scene = self.capture_stack_scene or recovering_capture
            if capture_stack_scene:
                from stacking_tabletop.acquisition.stack_scene_capture import check_stack_capture_runtime
                check_stack_capture_runtime(self.repository)
            transport, arm, gripper = self.device_factory(self.saved, self.runtime)
            stack_scene_handler = self._stack_scene if capture_stack_scene else None
            if recovering_capture and not self.capture_stack_scene:
                def stack_scene_handler(executor):
                    try:
                        return self._stack_scene(executor)
                    finally:
                        # The unchecked option suppresses capture on later full steps.
                        executor.stack_scene_handler = None
            self.executor = TabletopMotionExecutor(self.saved, arm, gripper,
                gate_changed=self.gate_changed.emit, telemetry=self.telemetry.emit, scan_handler=self._scan,
                stack_scene_handler=stack_scene_handler)
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
