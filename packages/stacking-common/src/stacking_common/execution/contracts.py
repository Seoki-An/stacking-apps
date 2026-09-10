"""Commands and observations at the platform executor boundary (SI units)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np

from stacking_common.planning import GraspEvent, MotionPlanSegment
from stacking_common.planning.contracts import validate_grasp_events

from .values import freeze_json, positive_int, required_text, sha256_text


class ExecutionStatus(str, Enum):
    RUNNING = "running"
    HELD = "held"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {self.SUCCEEDED, self.FAILED, self.CANCELLED}


class ExecutionFailure(str, Enum):
    INVALID_PLAN = "invalid_plan"
    TIMEOUT = "timeout"
    CONTROLLER_FAULT = "controller_fault"
    TRACKING_ERROR = "tracking_error"
    GRIPPER_FAULT = "gripper_fault"
    PERCEPTION_FAILED = "perception_failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ExecutionCommand:
    """One placement attempt bound to a stored plan and resolved run context.

    Segment samples remain a planning path when times are absent. The platform
    executor owns time parameterization, controller limits, and segment gates.
    A command ID identifies one attempt and must never be reused for a retry.
    """

    command_id: str
    step: int
    plan_attempt: int
    execution_attempt: int
    plan_sha256: str
    context_sha256: str
    frame_id: str
    joint_names: tuple[str, ...]
    segments: tuple[MotionPlanSegment, ...]
    grasp_events: tuple[GraspEvent, ...] = ()

    def __post_init__(self) -> None:
        for name in ("command_id", "frame_id"):
            object.__setattr__(self, name, required_text(getattr(self, name), name))
        for name in ("step", "plan_attempt", "execution_attempt"):
            positive_int(getattr(self, name), name)
        for name in ("plan_sha256", "context_sha256"):
            sha256_text(getattr(self, name), name)
        names = tuple(required_text(name, "joint name") for name in self.joint_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("joint names must be nonempty and unique")
        segments = []
        for value in self.segments:
            if not isinstance(value, MotionPlanSegment):
                raise TypeError("execution segments must be MotionPlanSegment values")
            segment = MotionPlanSegment(value.name, value.positions, value.times, value.openings, value.mode)
            if segment.positions.shape[1] != len(names):
                raise ValueError("execution joint names must match segment columns")
            if segment.times is not None and np.any(np.diff(segment.times) <= 0):
                raise ValueError("execution segment times must strictly increase")
            for array in (segment.positions, segment.times, segment.openings):
                if array is not None:
                    array.setflags(write=False)
            segments.append(segment)
        if not segments or len({value.name for value in segments}) != len(segments):
            raise ValueError("execution requires segments with unique names")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "segments", tuple(segments))
        object.__setattr__(self, "grasp_events", validate_grasp_events(self.grasp_events, segments))


@dataclass(frozen=True)
class ExecutionEvent:
    """One observed status; indices are zero-based within a commanded path.

    Event sequence numbers start at one per command. Timestamps are Unix
    nanoseconds; sequence numbers determine journal ordering.
    """

    command_id: str
    sequence: int
    timestamp_ns: int
    status: ExecutionStatus
    segment_index: int | None = None
    sample_index: int | None = None
    failure: ExecutionFailure | None = None
    reason: str | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_id", required_text(self.command_id, "command ID"))
        positive_int(self.sequence, "event sequence")
        positive_int(self.timestamp_ns, "event timestamp")
        for name in ("segment_index", "sample_index"):
            value = getattr(self, name)
            if value is not None:
                positive_int(value, name, minimum=0)
        if self.sample_index is not None and self.segment_index is None:
            raise ValueError("sample index requires a segment index")
        status, failure, reason = _outcome(self.status, self.failure, self.reason)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "failure", failure)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "diagnostics", freeze_json(self.diagnostics))


@dataclass(frozen=True)
class ExecutionResult:
    command_id: str
    status: ExecutionStatus
    failure: ExecutionFailure | None = None
    reason: str | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_id", required_text(self.command_id, "command ID"))
        status, failure, reason = _outcome(self.status, self.failure, self.reason)
        if not status.terminal:
            raise ValueError("execution result requires a terminal status")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "failure", failure)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "diagnostics", freeze_json(self.diagnostics))


class MotionExecutor(Protocol):
    """Platform driver contract; common code never sends hardware commands.

    Execute emits ordered events including a terminal event before returning.
    Hold/cancel request a controller action; completion is reported by events,
    not implied by the request returning. Implementations define thread access.
    """

    def execute(
        self,
        command: ExecutionCommand,
        *,
        progress: Callable[[ExecutionEvent], None],
    ) -> ExecutionResult: ...

    def hold(self, command_id: str) -> None: ...

    def resume(self, command_id: str) -> None:
        """Continue a held command in the same live controller session."""
        ...

    def cancel(self, command_id: str) -> None: ...


def _outcome(status, failure, reason):
    status = ExecutionStatus(status)
    failure = None if failure is None else ExecutionFailure(failure)
    reason = None if reason is None else required_text(reason, "reason")
    if status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
        if failure is None or reason is None:
            raise ValueError("failed/cancelled execution requires failure and reason")
    elif failure is not None:
        raise ValueError("failure category requires failed/cancelled status")
    if status is ExecutionStatus.SUCCEEDED and reason is not None:
        raise ValueError("successful execution must not have a failure reason")
    return status, failure, reason
