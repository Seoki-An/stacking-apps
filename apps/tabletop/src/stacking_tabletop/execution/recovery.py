"""Record explicit operator reconciliation; never move or reopen a device."""

import time

from stacking_common.artifacts import ExecutionReconciliation, StackRunStore
from stacking_common.execution import ExecutionEvent, ExecutionFailure


def reconcile_saved_run(path, reason, *, robot_stopped, scene_restored, gripper_checked):
    """The caller must end the live worker before collecting this evidence."""
    if not str(reason).strip() or not all(value is True for value in
            (robot_stopped, scene_restored, gripper_checked)):
        raise ValueError("reconciliation requires notes and all three physical checks")
    checkpoint = StackRunStore.load_execution_checkpoint(path)
    events = {event.command_id: event for event in StackRunStore.load_execution_events(path)}
    with StackRunStore.open_append(path) as writer:
        for command_id in checkpoint.unconfirmed_command_ids:
            latest = events.get(command_id)
            if latest is None or not latest.status.terminal:
                writer.append_execution_event(ExecutionEvent(
                    command_id, 1 if latest is None else latest.sequence + 1,
                    time.time_ns(), "failed", failure=ExecutionFailure.INTERRUPTED,
                    reason="operator confirmed interrupted controller session ended",
                ))
            writer.append_execution_reconciliation(ExecutionReconciliation(
                command_id, time.time_ns(), str(reason).strip(),
                {"robot_stopped": True, "scene_restored": True, "gripper_checked": True,
                 "source": "operator", "retry": "whole_step_from_home"},
            ))
