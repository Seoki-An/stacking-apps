"""Headless orchestration for the prepare, merge, reconstruct sequence."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Protocol

from stacking_common.artifacts import ArtifactReference, StageAttempt

from .session import ObjectBuildWorkflow


@dataclass(frozen=True)
class StageExecutionResult:
    artifacts: tuple[ArtifactReference, ...]
    metrics: Mapping[str, object] = field(default_factory=dict)
    decisions: Mapping[str, object] = field(default_factory=dict)


class ObjectBuildPlatformAdapter(Protocol):
    """The platform-specific edge of a common object-build workflow."""

    platform: str

    def prepare(
        self,
        raw_capture_root: Path,
        output_directory: Path,
    ) -> StageExecutionResult: ...

    def merge(
        self,
        prepared_artifacts: tuple[ArtifactReference, ...],
        output_directory: Path,
    ) -> StageExecutionResult: ...

    def reconstruct(
        self,
        merged_artifact: ArtifactReference,
        output_directory: Path,
    ) -> StageExecutionResult: ...


class ObjectBuildRunner:
    """Run stages while keeping one durable lineage manifest."""

    def __init__(
        self,
        workflow: ObjectBuildWorkflow,
        adapter: ObjectBuildPlatformAdapter,
    ) -> None:
        if workflow.session.platform != adapter.platform:
            raise ValueError("workflow and adapter platforms do not match")
        self.workflow = workflow
        self.adapter = adapter

    def run_prepare(self, raw_capture_root: str | Path) -> StageAttempt:
        return self._run(
            "prepare",
            (),
            lambda output: self.adapter.prepare(Path(raw_capture_root), output),
        )

    def run_merge(self) -> StageAttempt:
        prepared = self._accepted_artifacts("prepare")
        return self._run(
            "merge",
            tuple(value.artifact_id for value in prepared),
            lambda output: self.adapter.merge(prepared, output),
        )

    def run_reconstruct(self) -> StageAttempt:
        merged = self._accepted_artifacts("merge")
        if len(merged) != 1:
            raise ValueError("merge stage must expose exactly one merged cloud")
        return self._run(
            "reconstruct",
            (merged[0].artifact_id,),
            lambda output: self.adapter.reconstruct(merged[0], output),
        )

    def _run(self, stage: str, parents: tuple[str, ...], operation) -> StageAttempt:
        attempt_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        attempt = self.workflow.begin(stage, attempt_id, parents=parents)
        output = self.workflow.root / stage / attempt_id
        try:
            result = operation(output)
        except Exception as error:
            self.workflow.fail(attempt, error)
            raise
        return self.workflow.accept(
            attempt,
            artifacts=result.artifacts,
            metrics=result.metrics,
            decisions=result.decisions,
        )

    def _accepted_artifacts(self, stage: str) -> tuple[ArtifactReference, ...]:
        attempt = self.workflow.session.latest(stage)
        if attempt is None or attempt.status != "accepted":
            raise ValueError(f"{stage} must be accepted before the next stage")
        return attempt.artifacts
