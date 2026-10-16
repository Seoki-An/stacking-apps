"""One Qt worker owns the hardware session and its execution journal writer."""

import threading

from PySide6.QtCore import QThread, Signal

from stacking_tabletop.execution import AdapterRuntime, DynamixelAdapter, UR5eAdapter, UR5eKinematics
from stacking_tabletop.execution.runtime import HardwareError
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
    refined_run_ready = Signal(str)
    refinement_status = Signal(str)
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
        from stacking_tabletop.execution.sequence import InHandPreparationError
        from stacking_tabletop.perception.service_client import request as perception_request
        from stacking_tabletop.perception.inhand_execution import (
            failed_inhand_identification,
            identification_from_report,
        )
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
                try:
                    profile = load_femto_profile(Path(self.repository) / 'apps/tabletop/config/femto.yml', request[1])
                    calibration = FemtoCalibration(profile)
                    transport = resources.enter_context(FemtoRosTransport(calibration))
                    camera = FemtoCamera(transport, calibration, cancel_event=self.runtime.cancel_event)
                except Exception as error:
                    raise InHandPreparationError(str(error)) from error
            directory = scan_held_object(
                executor,
                request,
                camera,
                prepared=self.scan_prepared.emit,
            )
            object_id = executor.held_object_id
            executor._phase = "inhand_identification"
            executor._emit(
                "running",
                inhand_identification_started=str(directory),
                inhand_object_id=object_id,
            )
            try:
                report = perception_request(
                    self.repository, 'inhand', check=executor._health,
                    capture=str(directory), object_id=object_id,
                    initial_tool_T_object=executor.initial_tool_T_object().tolist(),
                )
                result = identification_from_report(report)
                if result.object_id != object_id or result.capture_directory != directory:
                    raise ValueError('identification belongs to another held object/capture')
                return result
            except HardwareError:
                raise
            except Exception as error:
                return failed_inhand_identification(directory, object_id, error)

    def _stack_scene(self, executor):
        from stacking_tabletop.acquisition.stack_scene_capture import collect_stack_scene
        from stacking_tabletop.acquisition.scene_capture_cli import ensure_ownership
        from stacking_tabletop.perception.service_client import server_wrist, RemoteStackProcessing
        ensure_ownership(executor.arm.transport)
        return collect_stack_scene(
            executor, self.repository, prepared=self.scan_prepared.emit,
            wrist_factory=lambda ex, calibration: server_wrist(ex, calibration, self.repository),
            processing_factory=RemoteStackProcessing,
        )

    def _refine_remaining(self, pending, executor=None):
        from stacking_common.execution import ExecutionFailure
        from stacking_tabletop.perception.service_client import request

        def check():
            if self.runtime.cancel_event.is_set():
                raise HardwareError(ExecutionFailure.CANCELLED,
                                    'scene refinement cancelled by operator')
            if executor is not None:
                # The command has already ended. Do not publish more events for
                # it or install its polling callback while perception runs.
                executor.arm._check_motion(False)
                hand = executor.gripper.feedback.gripper_state()
                if hand.holding or hand.hold_goal is not None:
                    raise HardwareError(ExecutionFailure.GRIPPER_FAULT,
                                        'scene refinement requires an empty gripper')

        self.refinement_status.emit(
            'Refining the remaining planned poses and motions; no new pickup '
            'will start. Large scene changes require separate replanning.'
        )
        reply = request(
            self.repository, 'refine_remaining', run=str(self.saved.path),
            capture=str(pending['capture_directory']), step=int(pending['step']),
            check=check, timeout=1800.,
        )
        from pathlib import Path
        from stacking_common.artifacts import StackRunStore
        from stacking_tabletop.execution.saved_plan import load_execution_run
        from stacking_tabletop.execution.scene_refinement import pending_scene_refinement
        branch = load_execution_run(reply['run'], self.repository)
        if branch.path == Path(self.saved.path).resolve():
            raise ValueError('refinement must produce a separate plan branch')
        original = StackRunStore.load_execution_checkpoint(self.saved.path)
        refined = StackRunStore.load_execution_checkpoint(branch.path)
        if (refined.requires_reconciliation
                or refined.completed_steps != original.completed_steps
                or refined.pending_steps != original.pending_steps
                or branch.context.sha256 != self.saved.context.sha256
                or pending_scene_refinement(branch.path) is not None):
            raise ValueError('refined branch does not preserve completed physical steps')
        check()
        self.refined_run_ready.emit(str(branch.path))

    def run(self):
        transport = None
        try:
            from stacking_common.artifacts import StackRunStore
            from stacking_tabletop.execution.scene_refinement import pending_scene_refinement
            self.saved.verify_assets()
            if self.runtime.cancel_event.is_set():
                return
            from stacking_tabletop.execution.recovery import pending_released_stack_capture
            pending = pending_scene_refinement(self.saved.path)
            if pending is not None:
                checkpoint = StackRunStore.load_execution_checkpoint(self.saved.path)
                if checkpoint.requires_reconciliation:
                    raise ValueError('reconcile the interrupted physical step before retrying scene refinement')
                recovery = pending_released_stack_capture(self.saved.path, pending['step'])
                if recovery is None or not checkpoint.pending_steps or checkpoint.pending_steps[0] != pending['step']:
                    # Retrying a failed perception job must not reconnect or move
                    # hardware. Explicitly reconciled scan-only recovery is the
                    # sole exception and still cannot replay pickup/placement.
                    self._refine_remaining(pending)
                    self.completed.emit(())
                    return
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
            pending = self.executor.pending_scene_refinement
            if pending is not None and results and results[-1].status.value == 'succeeded':
                self._refine_remaining(pending, self.executor)
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
