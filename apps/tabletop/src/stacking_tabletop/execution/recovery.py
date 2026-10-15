"""Record explicit operator reconciliation; never move or reopen a device."""

from dataclasses import dataclass
import time

from stacking_common.artifacts import ExecutionReconciliation, StackRunStore
from stacking_common.execution import ExecutionEvent, ExecutionFailure


@dataclass(frozen=True)
class ReleasedPlacementRecovery:
    step: int
    release_command_id: str
    release_pose: tuple[float, ...]


def released_placement_recovery(path):
    """Return a post-release recovery candidate backed by durable evidence."""
    checkpoint = StackRunStore.load_execution_checkpoint(path)
    if not checkpoint.requires_reconciliation or not checkpoint.pending_steps:
        return None
    commands = StackRunStore.load_execution_commands(path)
    events = StackRunStore.load_execution_events(path)
    by_command = {command.command_id: [] for command in commands}
    for event in events:
        by_command[event.command_id].append(event)
    unconfirmed = set(checkpoint.unconfirmed_command_ids)
    recovery_commands = [command for command in commands if command.command_id in unconfirmed]
    if not recovery_commands:
        return None
    step = checkpoint.pending_steps[0]
    if any(command.step != step for command in recovery_commands):
        return None
    candidates = []
    for index, command in enumerate(commands):
        if command.step != step:
            continue
        release = next((event for event in reversed(by_command[command.command_id])
                        if event.diagnostics.get("release_completed") is True
                        and event.diagnostics.get("release_pose") is not None), None)
        if release is not None:
            candidates.append((index, command, release))
    if not candidates:
        return None
    source_index, source, release = candidates[-1]
    recovery_index = max(index for index, command in enumerate(commands)
                         if command.command_id in unconfirmed)
    if source_index > recovery_index:
        return None
    if any(event.segment_index == 0
           for command in commands[source_index + 1:recovery_index + 1]
           for event in by_command[command.command_id]):
        return None
    return ReleasedPlacementRecovery(
        step, source.command_id, tuple(float(value)
                                       for value in release.diagnostics["release_pose"]),
    )


def accept_released_placement(path, reason, *, robot_stopped,
                              placed_scene_confirmed, gripper_empty):
    """Accept a durably recorded release and advance without moving hardware."""
    if not str(reason).strip() or not all(value is True for value in (
            robot_stopped, placed_scene_confirmed, gripper_empty)):
        raise ValueError("released-placement recovery requires notes and all three physical checks")
    recovery = released_placement_recovery(path)
    if recovery is None:
        raise ValueError("no safely recoverable released placement is recorded")
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
                command_id, time.time_ns(), str(reason).strip(), {
                    "robot_stopped": True,
                    "placed_scene_confirmed": True,
                    "gripper_empty": True,
                    "source": "operator",
                    "resolution": "accept_released_placement",
                    "release_command_id": recovery.release_command_id,
                    "release_pose": recovery.release_pose,
                },
            ))
    return recovery


def retry_released_stack_capture(path, reason, *, robot_stopped,
                                 placed_scene_confirmed, gripper_empty):
    """Reconcile the failed attempt for a scan-only execution retry."""
    if not str(reason).strip() or not all(value is True for value in (
            robot_stopped, placed_scene_confirmed, gripper_empty)):
        raise ValueError("stack-capture retry requires notes and all three physical checks")
    recovery = released_placement_recovery(path)
    if recovery is None:
        raise ValueError("no safely recoverable released placement is recorded")
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
                command_id, time.time_ns(), str(reason).strip(), {
                    "robot_stopped": True,
                    "placed_scene_confirmed": True,
                    "gripper_empty": True,
                    "source": "operator",
                    "retry": "stack_scene_only",
                    "release_command_id": recovery.release_command_id,
                    "release_pose": recovery.release_pose,
                },
            ))
    return recovery


def pending_released_stack_capture(path, step):
    """Return the latest scan-only retry request for a still-pending step."""
    checkpoint = StackRunStore.load_execution_checkpoint(path)
    if step not in checkpoint.pending_steps or checkpoint.requires_reconciliation:
        return None
    commands = {command.command_id: command
                for command in StackRunStore.load_execution_commands(path)}
    for reconciliation in reversed(StackRunStore.load_execution_reconciliations(path)):
        command = commands[reconciliation.command_id]
        if command.step != step:
            continue
        if reconciliation.diagnostics.get("retry") != "stack_scene_only":
            return None
        return ReleasedPlacementRecovery(
            step,
            str(reconciliation.diagnostics["release_command_id"]),
            tuple(float(value) for value in reconciliation.diagnostics["release_pose"]),
        )
    return None


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
