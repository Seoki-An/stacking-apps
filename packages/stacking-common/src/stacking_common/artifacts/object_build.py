"""Versioned lineage for prepare, merge, and reconstruct attempts."""

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Mapping


STAGES = ("prepare", "merge", "reconstruct")
STATUSES = (
    "pending",
    "running",
    "waiting_for_operator",
    "accepted",
    "failed",
    "cancelled",
    "stale",
)


@dataclass(frozen=True)
class ArtifactReference:
    artifact_id: str
    path: str
    frame: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.path or not self.frame:
            raise ValueError("artifact ID, relative path, and frame are required")
        path = Path(self.path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("artifact paths must be safe session-relative paths")


@dataclass(frozen=True)
class StageAttempt:
    stage: str
    attempt_id: str
    status: str
    parents: tuple[str, ...] = ()
    artifacts: tuple[ArtifactReference, ...] = ()
    profile: Mapping[str, object] = field(default_factory=dict)
    metrics: Mapping[str, object] = field(default_factory=dict)
    decisions: Mapping[str, object] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"unsupported object-build stage: {self.stage}")
        if not self.attempt_id or Path(self.attempt_id).name != self.attempt_id:
            raise ValueError("attempt ID must be one path-segment name")
        if self.status not in STATUSES:
            raise ValueError(f"unsupported stage status: {self.status}")
        if self.status == "failed" and not self.error:
            raise ValueError("failed stage attempt requires an error")


@dataclass(frozen=True)
class ObjectBuildSession:
    object_id: int
    session_id: str
    platform: str
    attempts: tuple[StageAttempt, ...] = ()
    measured_mass_kg: float | None = None
    density_kg_m3: float | None = None
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.object_id <= 0:
            raise ValueError("object ID must be positive")
        if not self.session_id or Path(self.session_id).name != self.session_id:
            raise ValueError("session ID must be one path-segment name")
        if not self.platform:
            raise ValueError("session platform is required")
        if self.measured_mass_kg is not None and self.measured_mass_kg <= 0.0:
            raise ValueError("measured mass must be positive")
        if self.density_kg_m3 is not None and self.density_kg_m3 <= 0.0:
            raise ValueError("configured density must be positive")
        identities = {(value.stage, value.attempt_id) for value in self.attempts}
        if len(identities) != len(self.attempts):
            raise ValueError("stage attempt IDs must be unique within each stage")
        artifact_ids = [
            artifact.artifact_id
            for attempt in self.attempts
            for artifact in attempt.artifacts
        ]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("artifact IDs must be unique within a workflow")

    def latest(self, stage: str) -> StageAttempt | None:
        values = [value for value in self.attempts if value.stage == stage]
        return values[-1] if values else None

    def with_attempt(self, attempt: StageAttempt) -> "ObjectBuildSession":
        attempts = list(self.attempts)
        if attempt.stage in STAGES:
            stage_index = STAGES.index(attempt.stage)
            attempts = [
                replace(value, status="stale")
                if (
                    STAGES.index(value.stage) > stage_index
                    and value.status == "accepted"
                )
                else value
                for value in attempts
            ]
        attempts.append(attempt)
        return replace(self, attempts=tuple(attempts))


class ObjectBuildSessionStore:
    """Atomically persist the single source of workflow truth."""

    def save(self, session: ObjectBuildSession, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_session_to_dict(session), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    def load(self, path: str | Path) -> ObjectBuildSession:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise TypeError("object-build manifest must be a JSON object")
        return _session_from_dict(values)


def _session_to_dict(session: ObjectBuildSession) -> dict[str, object]:
    return {
        "schema_version": session.schema_version,
        "object_id": session.object_id,
        "session_id": session.session_id,
        "platform": session.platform,
        "measured_mass_kg": session.measured_mass_kg,
        "density_kg_m3": session.density_kg_m3,
        "attempts": [
            {
                "stage": attempt.stage,
                "attempt_id": attempt.attempt_id,
                "status": attempt.status,
                "parents": list(attempt.parents),
                "artifacts": [
                    {
                        "artifact_id": artifact.artifact_id,
                        "path": artifact.path,
                        "frame": artifact.frame,
                        "sha256": artifact.sha256,
                    }
                    for artifact in attempt.artifacts
                ],
                "profile": dict(attempt.profile),
                "metrics": dict(attempt.metrics),
                "decisions": dict(attempt.decisions),
                "error": attempt.error,
            }
            for attempt in session.attempts
        ],
    }


def _session_from_dict(values: Mapping[str, object]) -> ObjectBuildSession:
    attempts = []
    for item in values.get("attempts", []):
        if not isinstance(item, Mapping):
            raise TypeError("stage attempt must be a mapping")
        artifacts = tuple(
            ArtifactReference(
                artifact_id=str(artifact["artifact_id"]),
                path=str(artifact["path"]),
                frame=str(artifact["frame"]),
                sha256=(
                    None if artifact.get("sha256") is None else str(artifact["sha256"])
                ),
            )
            for artifact in item.get("artifacts", [])
        )
        attempts.append(
            StageAttempt(
                stage=str(item["stage"]),
                attempt_id=str(item["attempt_id"]),
                status=str(item["status"]),
                parents=tuple(str(value) for value in item.get("parents", [])),
                artifacts=artifacts,
                profile=dict(item.get("profile", {})),
                metrics=dict(item.get("metrics", {})),
                decisions=dict(item.get("decisions", {})),
                error=None if item.get("error") is None else str(item["error"]),
            )
        )
    mass = values.get("measured_mass_kg")
    density = values.get("density_kg_m3")
    return ObjectBuildSession(
        schema_version=int(values.get("schema_version", 1)),
        object_id=int(values["object_id"]),
        session_id=str(values["session_id"]),
        platform=str(values["platform"]),
        measured_mass_kg=None if mass is None else float(mass),
        density_kg_m3=None if density is None else float(density),
        attempts=tuple(attempts),
    )
