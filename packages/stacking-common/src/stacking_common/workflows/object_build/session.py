"""Durable state transitions for a prepare/merge/reconstruct session."""

from pathlib import Path
from typing import Mapping

from stacking_common.artifacts import (
    ArtifactReference,
    ObjectBuildSession,
    ObjectBuildSessionStore,
    StageAttempt,
)


class ObjectBuildWorkflow:
    def __init__(self, root: str | Path, session: ObjectBuildSession) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "workflow.json"
        self.store = ObjectBuildSessionStore()
        self.session = session
        self.store.save(session, self.manifest)

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        object_id: int,
        session_id: str,
        platform: str,
        measured_mass_kg: float | None = None,
        density_kg_m3: float | None = None,
    ) -> "ObjectBuildWorkflow":
        return cls(
            root,
            ObjectBuildSession(
                object_id=object_id,
                session_id=session_id,
                platform=platform,
                measured_mass_kg=measured_mass_kg,
                density_kg_m3=density_kg_m3,
            ),
        )

    @classmethod
    def resume(cls, root: str | Path) -> "ObjectBuildWorkflow":
        path = Path(root).resolve()
        store = ObjectBuildSessionStore()
        session = store.load(path / "workflow.json")
        workflow = cls.__new__(cls)
        workflow.root = path
        workflow.manifest = path / "workflow.json"
        workflow.store = store
        workflow.session = session
        return workflow

    def begin(
        self,
        stage: str,
        attempt_id: str,
        *,
        parents: tuple[str, ...] = (),
        profile: Mapping[str, object] | None = None,
    ) -> StageAttempt:
        return self._record(
            StageAttempt(
                stage=stage,
                attempt_id=attempt_id,
                status="running",
                parents=parents,
                profile={} if profile is None else profile,
            )
        )

    def waiting_for_operator(
        self,
        attempt: StageAttempt,
        *,
        decisions: Mapping[str, object] | None = None,
    ) -> StageAttempt:
        return self._replace(
            attempt,
            status="waiting_for_operator",
            decisions={} if decisions is None else decisions,
        )

    def accept(
        self,
        attempt: StageAttempt,
        *,
        artifacts: tuple[ArtifactReference, ...],
        metrics: Mapping[str, object] | None = None,
        decisions: Mapping[str, object] | None = None,
    ) -> StageAttempt:
        return self._replace(
            attempt,
            status="accepted",
            artifacts=artifacts,
            metrics={} if metrics is None else metrics,
            decisions={} if decisions is None else decisions,
        )

    def fail(self, attempt: StageAttempt, error: Exception | str) -> StageAttempt:
        return self._replace(attempt, status="failed", error=str(error))

    def cancel(self, attempt: StageAttempt) -> StageAttempt:
        return self._replace(attempt, status="cancelled")

    def _replace(self, previous: StageAttempt, **changes) -> StageAttempt:
        attempts = tuple(
            value
            for value in self.session.attempts
            if not (
                value.stage == previous.stage
                and value.attempt_id == previous.attempt_id
            )
        )
        current = StageAttempt(
            stage=previous.stage,
            attempt_id=previous.attempt_id,
            status=changes.get("status", previous.status),
            parents=previous.parents,
            artifacts=changes.get("artifacts", previous.artifacts),
            profile=previous.profile,
            metrics=changes.get("metrics", previous.metrics),
            decisions=changes.get("decisions", previous.decisions),
            error=changes.get("error", previous.error),
        )
        self.session = ObjectBuildSession(
            object_id=self.session.object_id,
            session_id=self.session.session_id,
            platform=self.session.platform,
            measured_mass_kg=self.session.measured_mass_kg,
            density_kg_m3=self.session.density_kg_m3,
            schema_version=self.session.schema_version,
            attempts=attempts + (current,),
        )
        self.store.save(self.session, self.manifest)
        return current

    def _record(self, attempt: StageAttempt) -> StageAttempt:
        self.session = self.session.with_attempt(attempt)
        self.store.save(self.session, self.manifest)
        return attempt
