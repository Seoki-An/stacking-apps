"""Single-file stacking run documents.

The public format is a versioned application document. SQLite provides the
transaction and indexing layer; JSON records and compressed typed arrays are
implementation details hidden behind this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Iterator, Mapping
from uuid import uuid4
import zlib

import numpy as np

from stacking_common.execution import ExecutionCommand, ExecutionEvent, ExecutionResult
from stacking_common.execution.values import canonical_json

from stacking_common.visualization import MCTSDebugArtifact, MCTSDebugStep

from .mcts_debug import (
    decode_mcts_debug,
    decode_mcts_step,
    encode_mcts_debug,
    encode_mcts_step,
)
from .planning import (
    PlacementPlanRecord,
    decode_placement_plan,
    encode_placement_plan,
)
from .run_context import PlanningRunContext, decode_run_context, encode_run_context
from .execution import (
    ExecutionCheckpoint, ExecutionReconciliation, build_execution_command,
    decode_execution_command, decode_execution_event, decode_execution_reconciliation,
    encode_execution_command, encode_execution_event, encode_execution_reconciliation,
    execution_checkpoint, validate_execution_event,
)
from .planning_checkpoint import PlanningCheckpoint, decode_planning_checkpoint, encode_planning_checkpoint


STACK_SUFFIX = ".stack"
STACK_FORMAT_VERSION = 1
_APPLICATION_ID = 0x53544143  # ASCII "STAC"
_ARRAY_MARKER = "$stack_array"
_ARRAY_CODEC = "zlib"
_MIN_ARRAY_VALUES = 16


@dataclass(frozen=True)
class StackRunInfo:
    """Small summary that can be read without decoding result arrays."""

    path: Path
    status: str
    created_utc: str
    completed_utc: str | None
    content_types: tuple[str, ...]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class StackAttachmentInfo:
    """Metadata for one embedded asset or derived export."""

    name: str
    role: str
    media_type: str
    sha256: str
    size: int
    metadata: Mapping[str, object]


class StackRunWriter:
    """Incrementally append committed records to a new ``.stack`` file."""

    def __init__(
        self, path: Path, connection: sqlite3.Connection, *, finalize_on_exit: bool = True,
    ) -> None:
        self.path = path
        self._connection = connection
        self._closed = False
        self._finalize_on_exit = finalize_on_exit

    def append_mcts_step(
        self,
        step: MCTSDebugStep,
        *,
        statistics: Mapping[str, object] | None = None,
        checkpoint: PlanningCheckpoint | None = None,
        context_sha256: str | None = None,
    ) -> None:
        """Commit one complete search step without rewriting earlier steps."""

        self._require_open()
        if checkpoint is not None and checkpoint.step != step.step:
            raise ValueError("checkpoint and committed step numbers differ")
        payload, arrays = _externalize_arrays(encode_mcts_step(step))
        with self._connection:
            # Viewing a search step must not require a restart checkpoint.
            # Copied steps may belong to an older context than the branch's
            # restart state; callers can preserve that association explicitly.
            if context_sha256 is None and checkpoint is not None:
                context_sha256 = checkpoint.context_sha256
            if context_sha256 is None:
                row = self._connection.execute(
                    "SELECT record_id FROM records WHERE kind='planning_context' "
                    "ORDER BY ordinal DESC LIMIT 1"
                ).fetchone()
                context_sha256 = None if row is None else row[0]
            if context_sha256 is not None:
                _load_context(self._connection, context_sha256)
                payload["context_sha256"] = context_sha256
            _write_record(
                self._connection,
                kind="mcts_step",
                record_id=str(step.step),
                ordinal=step.step,
                schema_version=1,
                payload=payload,
                arrays=arrays,
            )
            if statistics is not None:
                statistics_payload, statistics_arrays = _externalize_arrays(
                    dict(statistics)
                )
                _write_record(
                    self._connection,
                    kind="mcts_search",
                    record_id=str(step.step),
                    ordinal=step.step,
                    schema_version=1,
                    payload=statistics_payload,
                    arrays=statistics_arrays,
                )
            if checkpoint is not None:
                self._append_planning_checkpoint(checkpoint)
            _set_metadata(self._connection, "updated_utc", _utc_now())

    def _append_planning_checkpoint(self, checkpoint: PlanningCheckpoint) -> None:
        context = _load_context(self._connection, checkpoint.context_sha256)
        if context.frame_id != checkpoint.world_T_stack.parent_frame:
            raise ValueError("checkpoint and context frames differ")
        step = _read_record(self._connection, "mcts_step", str(checkpoint.step))
        if step.get("selected_candidate_id") is None:
            raise ValueError("planning checkpoint requires a selected step")
        selected = next(
            item for item in step["candidates"]
            if item["candidate_id"] == step["selected_candidate_id"]
        )
        state = checkpoint.state
        index = int(selected["object_index"])
        if index not in state.placed_object_indices or state.object_ids[index] != selected["object_id"]:
            raise ValueError("checkpoint state does not contain the selected placement")
        self._append_record(
            "planning_checkpoint", str(checkpoint.step), encode_planning_checkpoint(checkpoint),
        )

    def finish(self, status: str = "complete") -> None:
        """Record the terminal run status and close the document."""

        self._require_open()
        status = str(status).strip()
        if status not in {"complete", "failed", "cancelled"}:
            raise ValueError("stack run status must be complete, failed, or cancelled")
        with self._connection:
            _set_metadata(self._connection, "status", status)
            _set_metadata(self._connection, "updated_utc", _utc_now())
            _set_metadata(self._connection, "completed_utc", _utc_now())
        self.close()

    def append_placement_plan(self, record: PlacementPlanRecord) -> None:
        """Commit one motion-planning attempt alongside its search step."""

        self._require_open()
        if not isinstance(record, PlacementPlanRecord):
            raise TypeError("stack run planning record has the wrong type")
        if record.context_sha256 is not None:
            context = _load_context(self._connection, record.context_sha256)
            if context.frame_id != record.problem.frame_id:
                raise ValueError("plan and recorded context frames differ")
        payload, arrays = _externalize_arrays(encode_placement_plan(record))
        record_id = f"{record.step}:{record.attempt}"
        with self._connection:
            _write_record(
                self._connection,
                kind="placement_plan",
                record_id=record_id,
                ordinal=_next_ordinal(self._connection, "placement_plan"),
                schema_version=1,
                payload=payload,
                arrays=arrays,
            )
            _add_content_type(self._connection, "placement_plan")
            _set_metadata(self._connection, "updated_utc", _utc_now())

    def append_run_context(self, context: PlanningRunContext) -> str:
        """Record resolved planning inputs before writing plans bound to them."""
        self._require_open()
        if not isinstance(context, PlanningRunContext):
            raise TypeError("run context must be a PlanningRunContext")
        digest = context.sha256
        with self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            existing = self._connection.execute(
                "SELECT 1 FROM records WHERE kind = 'planning_context' AND record_id = ?",
                (digest,),
            ).fetchone()
            if existing is None:
                self._append_record("planning_context", digest, encode_run_context(context))
            else:
                _load_context(self._connection, digest)
        return digest

    def append_execution_command(self, command: ExecutionCommand) -> None:
        """Durably queue one attempt before the platform sends it to a robot.

        Earlier planned steps must have succeeded. Unconfirmed attempts must be
        reconciled before retry; queuing never implies physical completion.
        """
        self._require_open()
        if not isinstance(command, ExecutionCommand):
            raise TypeError("expected an ExecutionCommand")
        with self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            plan = decode_placement_plan(_read_record(
                self._connection, "placement_plan", f"{command.step}:{command.plan_attempt}",
            ))
            context = _load_context(self._connection, command.context_sha256)
            expected = build_execution_command(
                plan, context, execution_attempt=command.execution_attempt,
                command_id=command.command_id,
            )
            if canonical_json(encode_execution_command(command)) != canonical_json(
                encode_execution_command(expected)
            ):
                raise ValueError("execution command differs from its stored plan/context")
            step = _read_record(self._connection, "mcts_step", str(command.step))
            if plan.candidate_id is None or step["selected_candidate_id"] != plan.candidate_id:
                raise ValueError("execution requires the committed selected candidate")
            commands, events, reconciliations = _execution_records(self._connection)
            planned = _planned_steps(self._connection)
            checkpoint = execution_checkpoint(planned, commands, events, reconciliations)
            if checkpoint.requires_reconciliation:
                raise ValueError("previous execution requires reconciliation")
            if command.step in checkpoint.completed_steps:
                raise ValueError("step has already been executed successfully")
            if any(step < command.step for step in checkpoint.pending_steps):
                raise ValueError("earlier planned steps must execute first")
            attempts = [item.execution_attempt for item in commands if item.step == command.step]
            if command.execution_attempt != max(attempts, default=0) + 1:
                raise ValueError("execution attempt must be the next attempt for this step")
            self._append_record("execution_command", command.command_id, encode_execution_command(command))

    def append_execution_event(self, event: ExecutionEvent) -> None:
        """Commit ordered observations, including the final execution result."""
        self._require_open()
        if not isinstance(event, ExecutionEvent):
            raise TypeError("expected an ExecutionEvent")
        with self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            command = decode_execution_command(_read_record(
                self._connection, "execution_command", event.command_id,
            ))
            events = [
                decode_execution_event(payload)
                for payload in _read_records(self._connection, "execution_event")
                if payload["command_id"] == event.command_id
            ]
            validate_execution_event(command, events[-1] if events else None, event)
            self._append_record(
                "execution_event", f"{event.command_id}:{event.sequence}", encode_execution_event(event),
            )

    def append_execution_reconciliation(self, record: ExecutionReconciliation) -> None:
        """Record platform evidence permitting a new attempt after a failure."""
        self._require_open()
        if not isinstance(record, ExecutionReconciliation):
            raise TypeError("expected an ExecutionReconciliation")
        with self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            commands, events, reconciliations = _execution_records(self._connection)
            execution_checkpoint(
                _planned_steps(self._connection), commands, events, (*reconciliations, record),
            )
            self._append_record(
                "execution_reconciliation", record.command_id,
                encode_execution_reconciliation(record),
            )

    def _append_record(self, kind: str, record_id: str, values) -> None:
        payload, arrays = _externalize_arrays(values)
        _write_record(
            self._connection, kind=kind, record_id=record_id,
            ordinal=_next_ordinal(self._connection, kind), schema_version=1,
            payload=payload, arrays=arrays,
        )
        _add_content_type(self._connection, kind)
        _set_metadata(self._connection, "updated_utc", _utc_now())

    def add_attachment(
        self,
        name: str,
        data: bytes,
        *,
        role: str,
        media_type: str,
        metadata: Mapping[str, object] | None = None,
    ) -> StackAttachmentInfo:
        """Embed a replay asset or derived export in the run document."""

        self._require_open()
        name = str(name).strip()
        role = str(role).strip()
        media_type = str(media_type).strip()
        if not name or not role or not media_type:
            raise ValueError("stack attachment name, role, and media type are required")
        if not isinstance(data, bytes):
            raise TypeError("stack attachment data must be bytes")
        details = dict(metadata or {})
        sha256 = hashlib.sha256(data).hexdigest()
        with self._connection:
            _write_blob(self._connection, sha256, data, prefer_compression=False)
            self._connection.execute(
                """
                INSERT INTO attachments(
                    name, role, media_type, sha256, size, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (name, role, media_type, sha256, len(data), _json(details)),
            )
            _set_metadata(self._connection, "updated_utc", _utc_now())
        return StackAttachmentInfo(
            name, role, media_type, sha256, len(data), details
        )

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("stack run writer is closed")

    def __enter__(self) -> StackRunWriter:
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._closed:
            return
        if not self._finalize_on_exit:
            self.close()
            return
        if exc_type is None:
            self.finish("complete")
        else:
            self.finish("failed")


class StackRunStore:
    """Create and read the native single-file stacking run format."""

    @staticmethod
    def open_append(path: str | Path) -> StackRunWriter:
        """Reopen an existing run without replacing data or changing its status.

        Context exit closes the connection; it does not mark planning complete.
        Each journal append is its own transaction. Open this writer in the
        thread that will consume execution events, or route events to it.
        """
        source = _stack_path(path)
        connection = sqlite3.connect(source.as_uri() + "?mode=rw", uri=True)
        try:
            _validate_document(connection)
            _configure_writer(connection)
        except BaseException:
            connection.close()
            raise
        return StackRunWriter(source, connection, finalize_on_exit=False)

    @staticmethod
    def load_run_context(path: str | Path, sha256: str) -> PlanningRunContext:
        with _open_reader(Path(path)) as connection:
            return _load_context(connection, sha256)

    @staticmethod
    def load_run_contexts(path: str | Path) -> tuple[PlanningRunContext, ...]:
        """Read verified contexts, including runs without completed attempts."""
        with _open_reader(Path(path)) as connection:
            rows = connection.execute(
                "SELECT record_id FROM records WHERE kind = 'planning_context' ORDER BY ordinal"
            ).fetchall()
            return tuple(_load_context(connection, row[0]) for row in rows)

    @staticmethod
    def load_mcts_step_contexts(path: str | Path) -> dict[int, str]:
        """Read search/view context bindings without decoding scene arrays.

        Older single-context runs and exact checkpoints remain supported.
        Ambiguous historical steps are left unresolved rather than assigned
        another experiment's settings.
        """
        with _open_reader(Path(path)) as connection:
            connection.execute("BEGIN")
            contexts = [row[0] for row in connection.execute(
                "SELECT record_id FROM records WHERE kind='planning_context' ORDER BY ordinal")]
            checkpoints = dict(connection.execute(
                "SELECT CAST(record_id AS INTEGER), json_extract(payload_json, '$.context_sha256') "
                "FROM records WHERE kind='planning_checkpoint'"))
            result = {}
            for step, digest, succeeded in connection.execute(
                "SELECT CAST(record_id AS INTEGER), json_extract(payload_json, '$.context_sha256'), "
                "json_extract(payload_json, '$.succeeded') FROM records WHERE kind='mcts_step'"
            ):
                digest = digest or checkpoints.get(step)
                if digest is None and contexts and (len(contexts) == 1 or not succeeded):
                    digest = contexts[-1]
                if digest is not None:
                    if digest not in contexts:
                        raise ValueError(f"search step {step} references an unknown planning context")
                    result[step] = digest
            return result

    @staticmethod
    def load_execution_commands(path: str | Path) -> tuple[ExecutionCommand, ...]:
        with _open_reader(Path(path)) as connection:
            return tuple(decode_execution_command(item) for item in _read_records(connection, "execution_command"))

    @staticmethod
    def load_execution_events(path: str | Path) -> tuple[ExecutionEvent, ...]:
        with _open_reader(Path(path)) as connection:
            return tuple(decode_execution_event(item) for item in _read_records(connection, "execution_event"))

    @staticmethod
    def load_execution_reconciliations(path: str | Path) -> tuple[ExecutionReconciliation, ...]:
        with _open_reader(Path(path)) as connection:
            return tuple(decode_execution_reconciliation(item)
                         for item in _read_records(connection, "execution_reconciliation"))

    @staticmethod
    def load_execution_results(path: str | Path) -> tuple[ExecutionResult, ...]:
        return tuple(
            ExecutionResult(event.command_id, event.status, event.failure, event.reason, event.diagnostics)
            for event in StackRunStore.load_execution_events(path) if event.status.terminal
        )

    @staticmethod
    def load_execution_checkpoint(path: str | Path) -> ExecutionCheckpoint:
        with _open_reader(Path(path)) as connection:
            connection.execute("BEGIN")
            commands, events, reconciliations = _execution_records(connection)
            return execution_checkpoint(_planned_steps(connection), commands, events, reconciliations)

    @staticmethod
    def load_committed_placement_plans(path: str | Path) -> tuple[PlacementPlanRecord, ...]:
        """Resolve selected steps to exactly one successful recorded plan each.

        A search-only run is readable by the viewer but cannot supply a motion
        sequence. Old plans lacking context remain readable; command creation
        requires a plan produced with a recorded context.
        """
        with _open_reader(Path(path)) as connection:
            connection.execute("BEGIN")
            plans = tuple(decode_placement_plan(item) for item in _read_records(connection, "placement_plan"))
            selected = []
            for step in _read_records(connection, "mcts_step"):
                candidate_id = step.get("selected_candidate_id")
                if candidate_id is None:
                    continue
                matches = [
                    plan for plan in plans
                    if plan.step == step["step"] and plan.candidate_id == candidate_id
                    and plan.result.succeeded
                ]
                if len(matches) != 1:
                    raise ValueError(f"step {step['step']} requires exactly one selected successful motion plan")
                selected.append(matches[0])
            return tuple(selected)

    @staticmethod
    def load_planning_checkpoint(path: str | Path, step: int | None = None) -> PlanningCheckpoint:
        """Load a committed restart state, the latest by default; never infer it from a viewer."""
        with _open_reader(Path(path)) as connection:
            connection.execute("BEGIN")
            steps = _planned_steps(connection)
            if not steps:
                raise ValueError("run has no committed planning step")
            if step is not None and step not in steps:
                raise ValueError(f"run has no committed planning step {step}")
            checkpoint = decode_planning_checkpoint(_read_record(
                connection, "planning_checkpoint", str(max(steps) if step is None else step),
            ))
            context = _load_context(connection, checkpoint.context_sha256)
            if context.frame_id != checkpoint.world_T_stack.parent_frame:
                raise ValueError("checkpoint and context frames differ")
            return checkpoint

    @staticmethod
    def create_mcts(
        path: str | Path,
        artifact: MCTSDebugArtifact,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> StackRunWriter:
        """Create a run containing an MCTS header and no search steps yet."""

        if artifact.steps:
            raise ValueError("new stack run MCTS header must not contain steps")
        destination = _stack_path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{uuid4().hex}.tmp"
        )
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(temporary)
            _configure_writer(connection)
            _create_schema(connection)
            header, arrays = _externalize_arrays(encode_mcts_debug(artifact))
            now = _utc_now()
            with connection:
                for key, value in {
                    "format": "stack",
                    "format_version": STACK_FORMAT_VERSION,
                    "status": "running",
                    "created_utc": now,
                    "updated_utc": now,
                    "completed_utc": None,
                    "content_types": ["mcts_debug"],
                    "metadata": dict(metadata or {}),
                }.items():
                    _set_metadata(connection, key, value)
                _write_record(
                    connection,
                    kind="mcts_header",
                    record_id="main",
                    ordinal=0,
                    schema_version=artifact.schema_version,
                    payload=header,
                    arrays=arrays,
                )
            connection.close()
            connection = None
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        except BaseException:
            if connection is not None:
                connection.close()
            temporary.unlink(missing_ok=True)
            raise
        opened = sqlite3.connect(destination)
        _configure_writer(opened)
        _validate_document(opened)
        return StackRunWriter(destination, opened)

    @classmethod
    def save_mcts(
        cls,
        artifact: MCTSDebugArtifact,
        path: str | Path,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        """Persist an already complete artifact as one ``.stack`` document."""

        header = MCTSDebugArtifact(
            frame_id=artifact.frame_id,
            target=artifact.target,
            camera=artifact.camera,
            steps=(),
            schema_version=artifact.schema_version,
        )
        writer = cls.create_mcts(path, header, metadata=metadata)
        try:
            for step in artifact.steps:
                writer.append_mcts_step(step)
            writer.finish("complete")
        except BaseException:
            if not writer._closed:
                writer.finish("failed")
            raise
        return writer.path

    @staticmethod
    def load_mcts(path: str | Path) -> MCTSDebugArtifact:
        """Load the MCTS content, verifying every compressed array."""

        source = Path(path)
        with _open_reader(source) as connection:
            header = _read_record(connection, "mcts_header", "main")
            header_artifact = decode_mcts_debug(header)
            steps = tuple(
                decode_mcts_step(payload, frame_id=header_artifact.frame_id)
                for payload in _read_records(connection, "mcts_step")
            )
        return MCTSDebugArtifact(
            frame_id=header_artifact.frame_id,
            target=header_artifact.target,
            camera=header_artifact.camera,
            steps=steps,
            schema_version=header_artifact.schema_version,
        )

    @staticmethod
    def info(path: str | Path) -> StackRunInfo:
        """Read lightweight run metadata without loading trajectories."""

        source = Path(path)
        with _open_reader(source) as connection:
            values = {
                row[0]: json.loads(row[1])
                for row in connection.execute(
                    "SELECT key, value_json FROM metadata"
                )
            }
        return StackRunInfo(
            path=source.resolve(),
            status=str(values["status"]),
            created_utc=str(values["created_utc"]),
            completed_utc=values.get("completed_utc"),
            content_types=tuple(values.get("content_types", [])),
            metadata=dict(values.get("metadata", {})),
        )

    @staticmethod
    def load_mcts_statistics(
        path: str | Path,
    ) -> tuple[Mapping[str, object], ...]:
        """Load compact per-step search accounting in committed order."""

        with _open_reader(Path(path)) as connection:
            return tuple(_read_records(connection, "mcts_search"))

    @staticmethod
    def load_placement_plans(
        path: str | Path,
    ) -> tuple[PlacementPlanRecord, ...]:
        """Load motion-planning attempts in committed order."""

        with _open_reader(Path(path)) as connection:
            return tuple(
                decode_placement_plan(payload)
                for payload in _read_records(connection, "placement_plan")
            )

    @staticmethod
    def attachments(path: str | Path) -> tuple[StackAttachmentInfo, ...]:
        """List embedded assets and exports without loading their bytes."""

        with _open_reader(Path(path)) as connection:
            rows = connection.execute(
                """
                SELECT name, role, media_type, sha256, size, metadata_json
                FROM attachments ORDER BY name
                """
            ).fetchall()
        return tuple(
            StackAttachmentInfo(
                name,
                role,
                media_type,
                sha256,
                size,
                json.loads(metadata_json),
            )
            for name, role, media_type, sha256, size, metadata_json in rows
        )

    @staticmethod
    def read_attachment(path: str | Path, name: str) -> bytes:
        """Read and verify one embedded attachment."""

        with _open_reader(Path(path)) as connection:
            row = connection.execute(
                """
                SELECT blobs.codec, blobs.sha256, blobs.raw_size, blobs.data
                FROM attachments JOIN blobs USING (sha256)
                WHERE attachments.name = ?
                """,
                (str(name),),
            ).fetchone()
        if row is None:
            raise KeyError(name)
        codec, sha256, raw_size, data = row
        return _decode_blob(codec, sha256, raw_size, data, str(name))


class StackRunViewReader:
    """Reuse decoded viewer records across refreshes of one run.

    A refresh reads one committed SQLite snapshot. Only new or changed records
    decompress arrays and rebuild contracts; file replacement resets the cache.
    Connections are closed after each read so an idle viewer holds no locks.
    """

    def __init__(self):
        self._identity = None
        self._records = {}

    def load(self, path: str | Path):
        path = Path(path).resolve()
        stat = path.stat()
        identity = (path, stat.st_dev, stat.st_ino)
        previous = self._records if identity == self._identity else {}
        records = {}
        with closing(_open_reader(path)) as connection, connection:
            connection.execute("BEGIN")
            rows = connection.execute(
                "SELECT kind, record_id, payload_json FROM records "
                "WHERE kind IN ('mcts_header', 'mcts_step', 'placement_plan', 'mcts_search') "
                "ORDER BY ordinal"
            ).fetchall()
            # Compare array descriptors too: JSON alone only contains array names.
            descriptors = {}
            for kind, record_id, name, dtype, shape, digest in connection.execute(
                "SELECT record_kind, record_id, name, dtype, shape_json, sha256 FROM arrays "
                "WHERE record_kind IN ('mcts_header', 'mcts_step', 'placement_plan', 'mcts_search') "
                "ORDER BY name"
            ):
                descriptors.setdefault((kind, record_id), []).append((name, dtype, shape, digest))

            def read(kind, record_id, payload):
                key = (kind, record_id)
                signature = (hashlib.sha256(payload.encode()).digest(),
                             tuple(descriptors.get(key, ())))
                cached = previous.get(key)
                if cached is not None and cached[0] == signature:
                    value = cached[1]
                else:
                    values = _materialize_arrays(json.loads(payload), _arrays(connection, *key))
                    if kind == "mcts_header":
                        value = decode_mcts_debug(values)
                    elif kind == "mcts_step":
                        value = decode_mcts_step(values, frame_id=header.frame_id)
                    elif kind == "placement_plan":
                        value = decode_placement_plan(values)
                    else:
                        value = values
                records[key] = (signature, value)
                return value

            header_row = next((row for row in rows if row[:2] == ("mcts_header", "main")), None)
            if header_row is None:
                raise ValueError("stack run is missing mcts_header/main")
            header = read(*header_row)
            if previous.get(("mcts_header", "main"), (None,))[0] != records[("mcts_header", "main")][0]:
                # Steps are decoded in the header's coordinate frame.
                previous = {}
            groups = {"mcts_step": [], "placement_plan": [], "mcts_search": []}
            for kind, record_id, payload in rows:
                if kind in groups:
                    groups[kind].append(read(kind, record_id, payload))
            artifact = MCTSDebugArtifact(
                frame_id=header.frame_id, target=header.target, camera=header.camera,
                steps=tuple(groups["mcts_step"]), schema_version=header.schema_version,
            )
        # Failed refreshes leave the previous complete cache usable.
        self._identity, self._records = identity, records
        return artifact, tuple(groups["placement_plan"]), tuple(groups["mcts_search"])


def _load_context(connection, digest):
    context = decode_run_context(_read_record(connection, "planning_context", digest))
    if context.sha256 != digest:
        raise ValueError("planning context content hash mismatch")
    return context


def _planned_steps(connection):
    return tuple(
        int(item["step"]) for (payload,) in connection.execute(
            "SELECT payload_json FROM records WHERE kind = 'mcts_step' ORDER BY ordinal"
        )
        for item in (json.loads(payload),)
        if item.get("selected_candidate_id") is not None
    )


def _execution_records(connection):
    return (
        tuple(decode_execution_command(item) for item in _read_records(connection, "execution_command")),
        tuple(decode_execution_event(item) for item in _read_records(connection, "execution_event")),
        tuple(decode_execution_reconciliation(item) for item in _read_records(connection, "execution_reconciliation")),
    )


def _stack_path(path: str | Path) -> Path:
    result = Path(path)
    if result.suffix.lower() != STACK_SUFFIX:
        raise ValueError(f"stack run path must end with {STACK_SUFFIX}")
    return result.resolve()


def _configure_writer(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = DELETE")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA busy_timeout = 5000")


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
    connection.execute(f"PRAGMA user_version = {STACK_FORMAT_VERSION}")
    connection.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        ) STRICT;

        CREATE TABLE records (
            kind TEXT NOT NULL,
            record_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            schema_version INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (kind, record_id),
            UNIQUE (kind, ordinal)
        ) STRICT;

        CREATE TABLE arrays (
            record_kind TEXT NOT NULL,
            record_id TEXT NOT NULL,
            name TEXT NOT NULL,
            dtype TEXT NOT NULL,
            shape_json TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            PRIMARY KEY (record_kind, record_id, name),
            FOREIGN KEY (record_kind, record_id)
                REFERENCES records(kind, record_id) ON DELETE CASCADE,
            FOREIGN KEY (sha256) REFERENCES blobs(sha256)
        ) STRICT;

        CREATE TABLE blobs (
            sha256 TEXT PRIMARY KEY,
            codec TEXT NOT NULL,
            raw_size INTEGER NOT NULL,
            data BLOB NOT NULL
        ) STRICT;

        CREATE TABLE attachments (
            name TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            media_type TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            size INTEGER NOT NULL,
            metadata_json TEXT NOT NULL,
            FOREIGN KEY (sha256) REFERENCES blobs(sha256)
        ) STRICT;

        CREATE INDEX records_kind_ordinal
            ON records(kind, ordinal);
        """
    )


def _validate_document(connection: sqlite3.Connection) -> None:
    application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if application_id != _APPLICATION_ID:
        raise ValueError("file is not a stacking .stack document")
    if version != STACK_FORMAT_VERSION:
        raise ValueError(f"unsupported .stack format version: {version}")


def _open_reader(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"missing stack run: {path}")
    uri = path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA foreign_keys = ON")
        _validate_document(connection)
    except BaseException:
        connection.close()
        raise
    return connection


def _set_metadata(
    connection: sqlite3.Connection, key: str, value: object
) -> None:
    connection.execute(
        """
        INSERT INTO metadata(key, value_json) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
        """,
        (str(key), _json(value)),
    )


def _add_content_type(connection: sqlite3.Connection, content_type: str) -> None:
    row = connection.execute(
        "SELECT value_json FROM metadata WHERE key = 'content_types'"
    ).fetchone()
    values = [] if row is None else list(json.loads(row[0]))
    if content_type not in values:
        values.append(content_type)
        _set_metadata(connection, "content_types", values)


def _next_ordinal(connection: sqlite3.Connection, kind: str) -> int:
    row = connection.execute(
        "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM records WHERE kind = ?",
        (kind,),
    ).fetchone()
    return int(row[0])


def _write_record(
    connection: sqlite3.Connection,
    *,
    kind: str,
    record_id: str,
    ordinal: int,
    schema_version: int,
    payload: object,
    arrays: Mapping[str, np.ndarray],
) -> None:
    connection.execute(
        """
        INSERT INTO records(kind, record_id, ordinal, schema_version, payload_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (kind, record_id, int(ordinal), int(schema_version), _json(payload)),
    )
    for name, value in arrays.items():
        array = np.ascontiguousarray(value)
        raw = array.tobytes(order="C")
        sha256 = hashlib.sha256(raw).hexdigest()
        _write_blob(connection, sha256, raw, prefer_compression=True)
        connection.execute(
            """
            INSERT INTO arrays(
                record_kind, record_id, name, dtype, shape_json,
                sha256
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                kind,
                record_id,
                name,
                array.dtype.str,
                _json(list(array.shape)),
                sha256,
            ),
        )


def _read_record(
    connection: sqlite3.Connection, kind: str, record_id: str
) -> dict[str, object]:
    row = connection.execute(
        "SELECT payload_json FROM records WHERE kind = ? AND record_id = ?",
        (kind, record_id),
    ).fetchone()
    if row is None:
        raise ValueError(f"stack run is missing {kind}/{record_id}")
    return _materialize_arrays(
        json.loads(row[0]), _arrays(connection, kind, record_id)
    )


def _read_records(
    connection: sqlite3.Connection, kind: str
) -> Iterator[dict[str, object]]:
    rows = connection.execute(
        "SELECT record_id, payload_json FROM records "
        "WHERE kind = ? ORDER BY ordinal",
        (kind,),
    ).fetchall()
    for record_id, payload_json in rows:
        yield _materialize_arrays(
            json.loads(payload_json), _arrays(connection, kind, record_id)
        )


def _arrays(
    connection: sqlite3.Connection, kind: str, record_id: str
) -> dict[str, np.ndarray]:
    result = {}
    rows = connection.execute(
        """
        SELECT arrays.name, arrays.dtype, arrays.shape_json,
               blobs.codec, blobs.sha256, blobs.raw_size, blobs.data
        FROM arrays JOIN blobs USING (sha256)
        WHERE arrays.record_kind = ? AND arrays.record_id = ?
        """,
        (kind, record_id),
    ).fetchall()
    for name, dtype, shape_json, codec, expected_hash, raw_size, data in rows:
        raw = _decode_blob(codec, expected_hash, raw_size, data, name)
        shape = tuple(int(value) for value in json.loads(shape_json))
        array = np.frombuffer(raw, dtype=np.dtype(dtype))
        expected_size = int(np.prod(shape, dtype=np.int64))
        if array.size != expected_size:
            raise ValueError(f"stack array shape does not match data: {name}")
        result[name] = array.reshape(shape).copy()
    return result


def _write_blob(
    connection: sqlite3.Connection,
    sha256: str,
    raw: bytes,
    *,
    prefer_compression: bool,
) -> None:
    compressed = zlib.compress(raw)
    if prefer_compression or len(compressed) < len(raw):
        codec, data = _ARRAY_CODEC, compressed
    else:
        codec, data = "none", raw
    connection.execute(
        """
        INSERT OR IGNORE INTO blobs(sha256, codec, raw_size, data)
        VALUES (?, ?, ?, ?)
        """,
        (sha256, codec, len(raw), data),
    )


def _decode_blob(
    codec: str,
    expected_hash: str,
    raw_size: int,
    data: bytes,
    name: str,
) -> bytes:
    if codec == _ARRAY_CODEC:
        try:
            raw = zlib.decompress(data)
        except zlib.error as error:
            raise ValueError(f"corrupt stack blob: {name}") from error
    elif codec == "none":
        raw = data
    else:
        raise ValueError(f"unsupported stack blob codec: {codec}")
    if len(raw) != raw_size:
        raise ValueError(f"stack blob byte size mismatch: {name}")
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError(f"stack blob checksum mismatch: {name}")
    return raw


def _externalize_arrays(
    value: object,
) -> tuple[object, dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}

    def visit(item: object, path: tuple[str, ...]) -> object:
        array = _numeric_array(item)
        if array is not None:
            name = "/".join(path) or "value"
            arrays[name] = array
            return {_ARRAY_MARKER: name}
        if isinstance(item, dict):
            return {
                str(key): visit(child, path + (str(key),))
                for key, child in item.items()
            }
        if isinstance(item, (list, tuple)):
            return [
                visit(child, path + (str(index),))
                for index, child in enumerate(item)
            ]
        return item

    return visit(value, ()), arrays


def _numeric_array(value: object) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.dtype.kind not in "biuf" or array.dtype.hasobject:
        return None
    if array.size < _MIN_ARRAY_VALUES:
        return None
    return np.ascontiguousarray(array)


def _materialize_arrays(
    value: object, arrays: Mapping[str, np.ndarray]
) -> object:
    if isinstance(value, dict):
        if set(value) == {_ARRAY_MARKER}:
            name = str(value[_ARRAY_MARKER])
            try:
                return arrays[name]
            except KeyError as error:
                raise ValueError(
                    f"stack record references missing array: {name}"
                ) from error
        return {key: _materialize_arrays(child, arrays) for key, child in value.items()}
    if isinstance(value, list):
        return [_materialize_arrays(child, arrays) for child in value]
    return value


def _json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
