"""Automatic held-object scans serialized inside the execution command worker."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from stacking_common.execution import ExecutionFailure
from stacking_tabletop.execution.runtime import HardwareError
from stacking_tabletop.execution.motion import CartesianStreamLimits
from .femto import CaptureError, CaptureCancelled
from .scan_paths import build_scan, joint_leg, Q_SCAN, PIVOT_POINT_TOOL_M
from .storage import CaptureSession


class CheckedKinematics:
    def __init__(self, model, check):
        self.model, self.check = model, check
        self.lower, self.upper = model.lower, model.upper

    def pose(self, q):
        self.check()
        return self.model.pose(q)

    def solve(self, target, seed):
        return self.model.solve(target, seed, check=self.check)


def scan_held_object(executor, request, camera, *, prepared=lambda value: None,
                     planner=build_scan, session_factory=CaptureSession):
    """Return to the exact gate branch on success; abort never initiates return."""
    kind, camera_name, directory = request
    arm = executor.arm
    anchor = arm.feedback.arm_state()[0].copy()
    executor._health()
    executor.gripper.require_hold()
    executor.saved.verify_assets()
    model = CheckedKinematics(arm.kinematics, executor._health)
    motions = planner(kind, model, anchor, check=executor._health)
    capture_count = sum(m.capture for m in motions)
    previous = anchor
    preview_legs = [anchor[None, :]]
    for motion in motions:
        preview_legs.append(joint_leg(previous, motion.target)[1:])
        previous = motion.target
    preview_path = np.vstack(preview_legs)
    output = Path(directory).expanduser().resolve()
    model_hash = hashlib.sha256(arm.kinematics.urdf_path.read_bytes()).hexdigest()
    details = {'kind': kind, 'camera': camera_name, 'output': str(output),
        'anchor': anchor.tolist(), 'q_scan': Q_SCAN.tolist(),
        'pivot_point_tool_m': PIVOT_POINT_TOOL_M.tolist(),
        'robot_urdf_sha256': model_hash, 'context_sha256': executor.saved.context.sha256,
        'command_id': executor.active_command_id, 'frame_id': 'base',
        'motions': [{**asdict(motion), 'target': motion.target.tolist()} for motion in motions]}
    try:
        # Check identity, optical TF and synchronized held state before entry.
        # The preflight burst is discarded; every view gets a fresh burst.
        _capture(executor, camera, anchor)
        with session_factory(output, camera.calibration, in_hand=True) as session:
            (output / 'scan_plan.json').write_text(json.dumps(details, indent=2, allow_nan=False) + '\n')
            prepared({'positions': preview_path.tolist(), 'capture_count': capture_count,
                      'kind': kind, 'output': str(output)})
            executor._gate_wait('scan_review',
                f'Review the {kind} scan, its entry/return motion and clearance in the viewport. '
                f'{capture_count} views will be saved to {output}. '
                'Check clearance around the robot and held object; this scan is not collision checked. '
                'Continue starts motion.')
            executor._profile(False)
            executor._emit('running', scan_started=str(output), scan_kind=kind)
            previous = anchor
            view = 0
            for index, motion in enumerate(motions):
                executor._phase = f'scan/{motion.label}'
                executor._health()
                # Wrist-only motions preserve the slower legacy 0.25 rad/s cap.
                wrist_only = np.allclose(previous[:5], motion.target[:5], atol=1e-8, rtol=0)
                limits = CartesianStreamLimits(100., .25 if wrist_only else .40, .08, .70)
                arm.execute_path(joint_leg(previous, motion.target), attached=True, limits=limits,
                    progress=lambda _i, _n, q: setattr(executor, '_reference', q))
                executor._emit('running', scan_motion_index=index, scan_motion_count=len(motions))
                if motion.capture:
                    _settle(executor, motion.target)
                    burst = _capture(executor, camera, motion.target)
                    session.add(burst)
                    view += 1
                    executor._emit('running', scan_view=view, scan_view_count=capture_count,
                                   capture_directory=str(output))
                previous = motion.target
            _settle(executor, anchor)
            executor._reference = anchor
            executor._health()
        executor._emit('held', scan_completed=str(output), scan_view_count=view)
        return output
    except CaptureCancelled as error:
        raise HardwareError(ExecutionFailure.CANCELLED, str(error)) from error
    except CaptureError as error:
        raise HardwareError(ExecutionFailure.PERCEPTION_FAILED, str(error)) from error


def _settle(executor, target):
    runtime = executor.runtime
    deadline = runtime.active_time() + 15.
    since = None
    while True:
        executor._health()
        now = runtime.active_time()
        if now >= deadline:
            raise HardwareError(ExecutionFailure.TRACKING_ERROR, 'scan joints did not settle within 0.25 degrees')
        q = executor.arm.feedback.arm_state()[0]
        if np.max(np.abs(q - target)) <= np.deg2rad(.25):
            if since is None:
                since = now
            elif now - since >= .25:
                return
        else:
            since = None
        runtime.sleep(.02)


def _capture(executor, camera, target):
    while True:
        paused = executor.runtime.paused_seconds
        try:
            burst = camera.capture(in_hand=True, check=executor._health)
        except CaptureError:
            executor._health()
            if executor.runtime.paused_seconds != paused:
                continue
            raise
        executor._health()
        if executor.runtime.paused_seconds != paused:
            # A resumed capture starts a whole new burst after warmup. Never
            # combine exposures from either side of an operator hold.
            continue
        if any(item.robot is None or np.max(np.abs(item.robot['joints'] - target)) > np.deg2rad(.25) for item in burst):
            raise CaptureError('capture joints disagree with the settled scan target')
        return burst
