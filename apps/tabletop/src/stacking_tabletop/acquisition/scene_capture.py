"""Serialize FEMTO brackets and an already reviewed automatic wrist scan."""

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile

import numpy as np

from stacking_tabletop.execution.motion import CartesianStreamLimits
from .hand_eye_capture import atomic_json

SCAN_LIMITS = CartesianStreamLimits(100., .15, .03, .30)


@contextmanager
def capture_lock():
    """Exclude other automatic scan processes on this host/ROS domain."""
    import fcntl
    domain = int(os.environ.get('ROS_DOMAIN_ID', '0'))
    path = Path(tempfile.gettempdir()) / f'stacking-scene-capture-{os.getuid()}-{domain}.lock'
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another automated scene capture owns this ROS domain') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def settle_arm(arm, target, *, timeout=15.):
    """Check actual joints, then let the recorder enforce tighter stationarity."""
    runtime = arm.runtime
    deadline = runtime.deadline(timeout)
    since = None
    while runtime.clock() < deadline:
        arm._check_motion()
        q = arm.feedback.arm_state()[0]
        now = runtime.clock()
        if np.max(np.abs(q - target)) <= np.deg2rad(.25):
            since = now if since is None else since
            if now - since >= .75:
                return
        else:
            since = None
        runtime.wait(.02, arm._check_motion)
    raise RuntimeError('scan joints did not settle at the requested viewpoint')


def execute_capture(plan, output, arm, femto_capture, wrist_context, assemble, *,
                    settle=settle_arm, progress=print, snapshot=lambda directory: None):
    """Devices are injected; abort holds in place and never initiates a return.

    The caller validates/reviews the full plan and owns controller exclusivity.
    FEMTO brackets run outside wrist_context to avoid simultaneous active depth.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    status = dict(schema_version=1, kind='automated_scene_capture', status='running',
                  plan_sha256=plan['approved_sha256'], output_frame='base_link', completed_views=[],
                  accuracy_1mm_verified=False, scene_stability='not_evaluated',
                  sensor_mode='wrist-only' if femto_capture is None else 'wrist-femto')
    def stage(name):
        status['stage'] = name
        atomic_json(output / 'capture.json', status)
        progress(name)
    seed = np.asarray(plan['seed_joints'])
    fixed_center = plan.get('schema_version') == 5
    start = np.asarray(plan['start_joints']) if fixed_center else seed
    views = []
    try:
        stage('preflight')
        atomic_json(output / 'executed_plan.json', plan)
        snapshot(output)
        arm.validate_start(start)
        # Hold the measured pose while stationary acquisition starts.
        arm.set_rigid_tracking()
        if not arm.hold_actual_pose():
            raise RuntimeError('could not request initial actual-pose hold')
        settle(arm, start)
        if femto_capture is not None:
            stage('femto_before')
            femto_capture(output / 'femto_before')
        stage('wrist_start')
        with wrist_context() as wrist_capture:
            if fixed_center:
                stage('move_capture_init')
                arm.execute_path(np.asarray(plan['entry_path']), limits=SCAN_LIMITS, attached=False)
                settle(arm, seed)
            for index, view in enumerate(plan['views']):
                path = np.asarray(view['path'])
                stage('move_' + view['label'])
                arm.execute_path(path, limits=SCAN_LIMITS, attached=False)
                settle(arm, path[-1])
                stage('capture_' + view['label'])
                directory = output / 'wrist' / f'view_{index:03d}_{view["label"]}'
                wrist_capture(directory)
                views.append(directory)
                status['completed_views'].append(str(directory.relative_to(output)))
                stage('return_' + view['label'])
                arm.execute_path(path[::-1], limits=SCAN_LIMITS, attached=False)
                settle(arm, seed)
        if fixed_center:
            stage('return_to_start')
            arm.execute_path(np.asarray(plan['return_path']), limits=SCAN_LIMITS, attached=False)
            settle(arm, start)
        if femto_capture is not None:
            stage('femto_after')
            femto_capture(output / 'femto_after')
        stage('assembly')
        references = (dict(femto_before=output / 'femto_before', femto_after=output / 'femto_after')
                      if femto_capture is not None else {})
        assemble(output / 'session', views, **references)
        status['status'] = 'complete'
        stage('complete')
        return status
    except BaseException as error:
        status.update(status='cancelled' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error=f'{type(error).__name__}: {error}',
                      hold_requested=bool(arm.hold_actual_pose()))
        atomic_json(output / 'capture.json', status)
        raise
