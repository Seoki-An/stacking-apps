from pathlib import Path

from stacking_common.artifacts import ArtifactReference
from stacking_common.workflows.object_build import (
    ObjectBuildRunner,
    ObjectBuildWorkflow,
    StageExecutionResult,
)
from stacking_tabletop.gui.reconstruction import WorkbenchController
from stacking_tabletop.gui.reconstruction.controller import artifact_render_format
from stacking_tabletop.gui.reconstruction.application import (
    build_parser as build_workbench_parser,
)
from stacking_tabletop.perception.object_build_cli import (
    build_parser as build_object_build_parser,
    tabletop_density,
)


ROOT = Path(__file__).resolve().parents[1]


class FakePlatformAdapter:
    platform = "tabletop"

    def __init__(self, root: Path) -> None:
        self.root = root

    def prepare(self, raw_capture_root: Path, output_directory: Path):
        return StageExecutionResult(
            (
                ArtifactReference(f"prepared-{output_directory.name}-1", "prepared/one.pcd", "tool0"),
                ArtifactReference(f"prepared-{output_directory.name}-2", "prepared/two.pcd", "tool0"),
            ),
            metrics={"capture_count": 2},
        )

    def merge(self, prepared_artifacts, output_directory: Path):
        assert len(prepared_artifacts) == 2
        return StageExecutionResult(
            (ArtifactReference(f"merged-{output_directory.name}", "merge/merged.pcd", "capture0"),)
        )

    def reconstruct(self, merged_artifact, output_directory: Path):
        assert merged_artifact.artifact_id.startswith("merged-")
        return StageExecutionResult(
            (ArtifactReference(f"object-dsf-{output_directory.name}", "output/object.obj", "object"),)
        )


def test_tabletop_object_build_uses_configured_density_without_mass() -> None:
    args = build_object_build_parser().parse_args(
        [".", "--object-id", "1", "--capture-directory", "captures"]
    )

    assert args.mass_kg is None
    assert tabletop_density(ROOT, args.object_id) == 1000.0


def test_workbench_can_start_without_preselecting_an_object() -> None:
    args = build_workbench_parser().parse_args(["."])

    assert args.object_id is None
    assert args.resume is None


def test_workbench_classifies_all_status_rendering_formats(
    tmp_path: Path,
) -> None:
    pcd = tmp_path / "object.pcd"
    mesh = tmp_path / "mesh.ply"
    dsf = tmp_path / "dsf.obj"
    metadata = tmp_path / "object.json"
    pcd.touch()
    mesh.touch()
    dsf.write_text("# dsf_count: 1\n")
    metadata.touch()

    assert artifact_render_format(pcd) == "pcd"
    assert artifact_render_format(mesh) == "mesh"
    assert artifact_render_format(dsf) == "dsf"
    assert artifact_render_format(metadata) is None


def test_object_build_runner_persists_stage_lineage(tmp_path: Path) -> None:
    workflow = ObjectBuildWorkflow.create(
        tmp_path,
        object_id=3,
        session_id="build-a",
        platform="tabletop",
        measured_mass_kg=0.42,
        density_kg_m3=1000.0,
    )
    runner = ObjectBuildRunner(workflow, FakePlatformAdapter(tmp_path))

    prepared = runner.run_prepare(tmp_path / "raw")
    merged = runner.run_merge()
    reconstructed = runner.run_reconstruct()

    loaded = ObjectBuildWorkflow.resume(tmp_path).session
    assert prepared.status == "accepted"
    assert all(value.startswith("prepared-") for value in merged.parents)
    assert reconstructed.parents[0].startswith("merged-")
    assert loaded.measured_mass_kg == 0.42
    assert loaded.density_kg_m3 == 1000.0
    assert loaded.latest("reconstruct").artifacts[0].artifact_id.startswith(
        "object-dsf-"
    )


def test_rerunning_upstream_marks_downstream_attempts_stale(tmp_path: Path) -> None:
    workflow = ObjectBuildWorkflow.create(
        tmp_path,
        object_id=3,
        session_id="build-a",
        platform="tabletop",
        measured_mass_kg=0.42,
    )
    runner = ObjectBuildRunner(workflow, FakePlatformAdapter(tmp_path))
    runner.run_prepare(tmp_path / "raw")
    runner.run_merge()
    runner.run_reconstruct()

    runner.run_prepare(tmp_path / "new-raw")

    assert runner.workflow.session.latest("prepare").status == "accepted"
    assert runner.workflow.session.latest("merge").status == "stale"
    assert runner.workflow.session.latest("reconstruct").status == "stale"


def test_workbench_controller_uses_manifest_as_state(tmp_path: Path) -> None:
    ObjectBuildWorkflow.create(
        tmp_path,
        object_id=1,
        session_id="build-a",
        platform="tabletop",
        measured_mass_kg=0.1,
    )
    controller = WorkbenchController(tmp_path)

    assert [stage.status for stage in controller.refresh()] == [
        "pending",
        "pending",
        "pending",
    ]
    assert controller.can_run("prepare")
    assert not controller.can_run("merge")


def test_workbench_controller_routes_accepted_merge_to_preview(tmp_path: Path) -> None:
    workflow = ObjectBuildWorkflow.create(
        tmp_path,
        object_id=1,
        session_id="preview-a",
        platform="tabletop",
    )
    merged = tmp_path / "merge/attempt-a/merged.pcd"
    merged.parent.mkdir(parents=True)
    merged.touch()
    attempt = workflow.begin("merge", "attempt-a")
    workflow.accept(
        attempt,
        artifacts=(
            ArtifactReference(
                "merged-attempt-a",
                "merge/attempt-a/merged.pcd",
                "first_capture_tool0",
            ),
        ),
    )

    controller = WorkbenchController(tmp_path)

    assert controller.preview_artifact("merge") == merged
