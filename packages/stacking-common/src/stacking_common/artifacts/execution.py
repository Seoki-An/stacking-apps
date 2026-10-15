"""Execution journal codecs and conservative recovery assessment."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import uuid4

from stacking_common.execution import ExecutionCommand, ExecutionEvent, ExecutionStatus
from stacking_common.execution.values import (
    content_digest, freeze_json, plain_json, positive_int, required_text,
)
from stacking_common.planning import MotionPlanSegment

from .planning import PlacementPlanRecord, encode_placement_plan, encode_grasp_event, decode_grasp_event
from .run_context import PlanningRunContext


def build_execution_command(
    plan: PlacementPlanRecord,
    context: PlanningRunContext,
    *,
    execution_attempt: int = 1,
    command_id: str | None = None,
) -> ExecutionCommand:
    if not plan.result.succeeded:
        raise ValueError("only successful planning attempts can be executed")
    if plan.problem.frame_id != context.frame_id:
        raise ValueError("plan and context frames differ")
    if plan.context_sha256 != context.sha256:
        raise ValueError("plan is not bound to this recorded context")
    return ExecutionCommand(
        command_id if command_id is not None else uuid4().hex,
        plan.step, plan.attempt, execution_attempt,
        content_digest(encode_placement_plan(plan)), context.sha256,
        context.frame_id, context.joint_names, plan.result.segments,
        plan.result.grasp_events,
    )


def encode_execution_command(value: ExecutionCommand) -> dict[str, object]:
    return plain_json({
        "schema_version": 1,
        "command_id": value.command_id, "step": value.step,
        "plan_attempt": value.plan_attempt, "execution_attempt": value.execution_attempt,
        "plan_sha256": value.plan_sha256, "context_sha256": value.context_sha256,
        "frame_id": value.frame_id, "joint_names": value.joint_names,
        "segments": [
            {"name": item.name, "positions": item.positions,
             **({"mode": item.mode} if item.mode is not None else {}),
             "times": item.times, "openings": item.openings}
            for item in value.segments
        ],
        **({"grasp_events": [encode_grasp_event(event) for event in value.grasp_events]}
           if value.grasp_events else {}),
    })


def decode_execution_command(values: Mapping[str, object]) -> ExecutionCommand:
    fields = _fields(values)
    fields["segments"] = tuple(MotionPlanSegment(**item) for item in fields["segments"])
    fields["grasp_events"] = tuple(decode_grasp_event(item) for item in fields.get("grasp_events", ()))
    return ExecutionCommand(**fields)


def encode_execution_event(value: ExecutionEvent) -> dict[str, object]:
    return plain_json({
        "schema_version": 1, "command_id": value.command_id,
        "sequence": value.sequence, "timestamp_ns": value.timestamp_ns,
        "status": value.status.value, "segment_index": value.segment_index,
        "sample_index": value.sample_index,
        "failure": None if value.failure is None else value.failure.value,
        "reason": value.reason, "diagnostics": value.diagnostics,
    })


def decode_execution_event(values: Mapping[str, object]) -> ExecutionEvent:
    return ExecutionEvent(**_fields(values))


def validate_execution_event(
    command: ExecutionCommand,
    previous: ExecutionEvent | None,
    event: ExecutionEvent,
) -> None:
    if event.command_id != command.command_id:
        raise ValueError("event command ID does not match")
    expected = 1 if previous is None else previous.sequence + 1
    if event.sequence != expected:
        raise ValueError(f"execution event sequence must be {expected}")
    if previous is not None and previous.status.terminal:
        raise ValueError("cannot append events after a terminal outcome")
    if previous is None and event.status not in {
        ExecutionStatus.RUNNING, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
    }:
        raise ValueError("first execution event must start or reject the command")
    if event.segment_index is not None:
        if event.segment_index >= len(command.segments):
            raise ValueError("event segment index is outside the command")
        if event.sample_index is not None and event.sample_index >= len(
            command.segments[event.segment_index].positions
        ):
            raise ValueError("event sample index is outside the segment")


@dataclass(frozen=True)
class ExecutionCheckpoint:
    """Journal assessment, not permission to move a robot.

    Unconfirmed attempts include queued, interrupted, failed, and cancelled
    commands. A platform must reconcile the physical scene before a retry.
    No mid-trajectory restart is inferred from the last sample index.
    """

    completed_steps: tuple[int, ...]
    pending_steps: tuple[int, ...]
    unconfirmed_command_ids: tuple[str, ...]

    @property
    def requires_reconciliation(self) -> bool:
        return bool(self.unconfirmed_command_ids)


@dataclass(frozen=True)
class ExecutionReconciliation:
    """Platform confirmation that a failed/cancelled attempt can be retried.

    The caller first confirms the robot is stopped and the scene is restored
    or replanned. This records its evidence; it does not perform that check.
    """

    command_id: str
    timestamp_ns: int
    reason: str
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_id", required_text(self.command_id, "command ID"))
        positive_int(self.timestamp_ns, "reconciliation timestamp")
        object.__setattr__(self, "reason", required_text(self.reason, "reconciliation reason"))
        object.__setattr__(self, "diagnostics", freeze_json(self.diagnostics))


def encode_execution_reconciliation(value):
    return plain_json({
        "schema_version": 1, "command_id": value.command_id,
        "timestamp_ns": value.timestamp_ns, "reason": value.reason,
        "diagnostics": value.diagnostics,
    })


def decode_execution_reconciliation(values):
    return ExecutionReconciliation(**_fields(values))


def execution_checkpoint(
    planned_steps: Sequence[int],
    commands: Sequence[ExecutionCommand],
    events: Sequence[ExecutionEvent],
    reconciliations: Sequence[ExecutionReconciliation] = (),
) -> ExecutionCheckpoint:
    by_id = {command.command_id: command for command in commands}
    if len(by_id) != len(commands):
        raise ValueError("duplicate execution command IDs")
    latest = {}
    for event in events:
        if event.command_id not in by_id:
            raise ValueError("event references an unknown execution command")
        validate_execution_event(by_id[event.command_id], latest.get(event.command_id), event)
        latest[event.command_id] = event
    completed = {
        command.step for command in commands
        if command.command_id in latest
        and latest[command.command_id].status is ExecutionStatus.SUCCEEDED
    }
    command_index = {command.command_id: index for index, command in enumerate(commands)}
    events_by_command = {
        command.command_id: tuple(event for event in events
                                  if event.command_id == command.command_id)
        for command in commands
    }
    reconciled = {item.command_id for item in reconciliations}
    for reconciliation in reconciliations:
        command_id = reconciliation.command_id
        if command_id not in latest or latest[command_id].status not in {
            ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
        }:
            raise ValueError("reconciliation requires a failed/cancelled command")
        accept_release = (
            reconciliation.diagnostics.get("resolution") == "accept_released_placement"
        )
        retry_capture = reconciliation.diagnostics.get("retry") == "stack_scene_only"
        if not accept_release and not retry_capture:
            continue
        source_id = reconciliation.diagnostics.get("release_command_id")
        if source_id not in by_id or by_id[source_id].step != by_id[command_id].step:
            raise ValueError("released-placement reconciliation has an invalid source command")
        source_index = command_index[source_id]
        recovery_index = command_index[command_id]
        release = next((event for event in reversed(events_by_command[source_id])
                        if event.diagnostics.get("release_completed") is True
                        and event.diagnostics.get("release_pose") is not None), None)
        if source_index > recovery_index or release is None:
            raise ValueError("released-placement reconciliation lacks durable release evidence")
        if reconciliation.diagnostics.get("release_pose") != release.diagnostics["release_pose"]:
            raise ValueError("released-placement reconciliation release pose differs from evidence")
        later_commands = commands[source_index + 1:recovery_index + 1]
        if any(event.segment_index == 0 for command in later_commands
               for event in events_by_command[command.command_id]):
            raise ValueError("a later execution attempt entered pickup motion")
        if accept_release:
            completed.add(by_id[command_id].step)
    unconfirmed = tuple(
        command.command_id for command in commands
        if command.command_id not in reconciled and (
            command.command_id not in latest
            or latest[command.command_id].status is not ExecutionStatus.SUCCEEDED
        )
    )
    return ExecutionCheckpoint(
        tuple(sorted(completed)), tuple(step for step in sorted(set(planned_steps)) if step not in completed),
        unconfirmed,
    )


def _fields(values):
    fields = dict(values)
    if fields.pop("schema_version", None) != 1:
        raise ValueError("unsupported execution record schema version")
    return fields
