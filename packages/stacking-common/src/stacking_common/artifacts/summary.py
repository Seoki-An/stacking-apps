"""Derived summaries for reports; the .stack document remains authoritative."""

import csv
import json
from dataclasses import asdict
from pathlib import Path

from stacking_common.execution.values import plain_json

from .execution import execution_checkpoint
from .run_document import _execution_records, _open_reader, _planned_steps, _read_record


def summarize_run(path: str | Path) -> dict[str, object]:
    """Read one consistent snapshot without exporting trajectory arrays."""
    with _open_reader(Path(path)) as connection:
        connection.execute("BEGIN")
        metadata = {
            key: json.loads(value)
            for key, value in connection.execute("SELECT key, value_json FROM metadata")
        }
        # Summaries never need materialized mesh, pose-track, or path arrays.
        steps = _payloads(connection, "mcts_step")
        statistics = {
            str(record_id): _read_record(connection, "mcts_search", record_id)
            for (record_id,) in connection.execute(
                "SELECT record_id FROM records WHERE kind = 'mcts_search' ORDER BY ordinal"
            )
        }
        plans = [
            {"step": item["step"], "attempt": item["attempt"],
             "candidate_id": item["candidate_id"], "context_sha256": item.get("context_sha256"),
             "status": item["result"]["status"], "reason": item["result"]["reason"],
             "segment_count": len(item["result"]["segments"])}
            for item in _payloads(connection, "placement_plan")
        ]
        commands, events, reconciliations = _execution_records(connection)
        checkpoint = execution_checkpoint(_planned_steps(connection), commands, events, reconciliations)
        latest = {event.command_id: event for event in events}
        executions = []
        for command in commands:
            event = latest.get(command.command_id)
            executions.append({
                "command_id": command.command_id, "step": command.step,
                "plan_attempt": command.plan_attempt, "execution_attempt": command.execution_attempt,
                "status": "queued" if event is None else event.status.value,
                "failure": None if event is None or event.failure is None else event.failure.value,
                "reason": None if event is None else event.reason,
                "diagnostics": {} if event is None else plain_json(event.diagnostics),
            })
        return {
            "schema_version": 1,
            "run": metadata,
            "steps": [
                {"step": item["step"], "selected_candidate_id": item["selected_candidate_id"],
                 "succeeded": item["succeeded"], "candidate_count": len(item["candidates"]),
                 "failure_reason": item.get("failure_reason"),
                 "statistics": plain_json(statistics.get(str(item["step"]), {}))}
                for item in steps
            ],
            "plans": plans,
            "executions": executions,
            "checkpoint": asdict(checkpoint),
        }


def export_run_summary(path: str | Path, destination: str | Path) -> Path:
    """Create JSON or one-row-per-step CSV. Existing outputs are preserved."""
    destination = Path(destination)
    if destination.suffix.lower() not in {".json", ".csv"}:
        raise ValueError("summary destination must end with .json or .csv")
    summary = summarize_run(path)
    if destination.suffix.lower() == ".json":
        text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        fields = (
            "step", "selected_candidate_id", "succeeded", "candidate_count", "failure_reason",
            "statistics_json", "plans_json", "executions_json",
        )
        rows = []
        for step in summary["steps"]:
            row = {key: value for key, value in step.items() if key != "statistics"}
            row["statistics_json"] = json.dumps(step["statistics"], allow_nan=False)
            for name in ("plans", "executions"):
                row[f"{name}_json"] = json.dumps(
                    [item for item in summary[name] if item["step"] == step["step"]], allow_nan=False,
                )
            rows.append(row)
        with destination.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return destination


def _payloads(connection, kind):
    return [json.loads(payload) for (payload,) in connection.execute(
        "SELECT payload_json FROM records WHERE kind = ? ORDER BY ordinal", (kind,),
    )]
