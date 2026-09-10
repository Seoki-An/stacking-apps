"""Qt-independent state projection for the reconstruction workbench."""

from dataclasses import dataclass
from pathlib import Path

from stacking_common.artifacts import ArtifactReference
from stacking_common.workflows.object_build import ObjectBuildWorkflow


def artifact_render_format(path: Path) -> str | None:
    """Classify one workflow artifact for layered status rendering."""

    suffix = path.suffix.lower()
    if suffix == ".pcd":
        return "pcd"
    if suffix == ".obj" and _is_dsf_artifact(path):
        return "dsf"
    if suffix in {".obj", ".ply"}:
        return "mesh"
    return None


@dataclass(frozen=True)
class WorkbenchStage:
    name: str
    status: str
    attempt_id: str | None
    artifacts: tuple[ArtifactReference, ...]


class WorkbenchController:
    """Read workflow truth and expose allowed stage transitions to any GUI."""

    stages = ("prepare", "merge", "reconstruct")

    def __init__(self, workflow_root: str | Path) -> None:
        self.root = Path(workflow_root).resolve()
        self.workflow = ObjectBuildWorkflow.resume(self.root)

    def refresh(self) -> tuple[WorkbenchStage, ...]:
        self.workflow = ObjectBuildWorkflow.resume(self.root)
        return tuple(self.stage(name) for name in self.stages)

    def stage(self, name: str) -> WorkbenchStage:
        attempt = self.workflow.session.latest(name)
        return WorkbenchStage(
            name=name,
            status="pending" if attempt is None else attempt.status,
            attempt_id=None if attempt is None else attempt.attempt_id,
            artifacts=() if attempt is None else attempt.artifacts,
        )

    def can_run(self, name: str) -> bool:
        if name == "prepare":
            return True
        upstream = "prepare" if name == "merge" else "merge"
        return self.stage(upstream).status == "accepted"

    def artifact_path(self, artifact: ArtifactReference) -> Path:
        return self.root / artifact.path

    def preview_artifact(self, stage: str) -> Path | None:
        """Return the first renderable artifact from an accepted stage."""
        for artifact in self.stage(stage).artifacts:
            path = self.artifact_path(artifact)
            if path.suffix.lower() in {".pcd", ".ply", ".obj"} and path.is_file():
                return path
        return None

    def finish_interrupted_process(
        self,
        stage: str,
        *,
        cancelled: bool,
        exit_code: int,
    ) -> None:
        self.workflow = ObjectBuildWorkflow.resume(self.root)
        attempt = self.workflow.session.latest(stage)
        if attempt is None or attempt.status != "running":
            return
        if cancelled:
            self.workflow.cancel(attempt)
        else:
            self.workflow.fail(
                attempt,
                f"stage process exited unexpectedly with code {exit_code}",
            )


def _is_dsf_artifact(path: Path) -> bool:
    with path.open(encoding="utf-8") as stream:
        return any("# dsf_count:" in line for _, line in zip(range(8), stream))
