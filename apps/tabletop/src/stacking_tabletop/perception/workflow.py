"""Tabletop adapter for the common object-build workflow."""

import json
from collections.abc import Callable
from pathlib import Path

from stacking_common.artifacts import ArtifactReference
from stacking_common.workflows.object_build import StageExecutionResult
from stacking_common.reconstruction import (
    ConvexDecompositionProfile,
    DsfPartitionOptions,
    DsfFittingProfile,
    ReconstructionOperatorOptions,
)
from stacking_tabletop.config import load_inventory_profile

from .merge_captures_cli import run_cross_capture_merge
from .object_reconstruct_cli import run_object_reconstruction
from .prepare_capture_cli import plan_capture_preparation, run_prepare_capture


class TabletopObjectBuildAdapter:
    """Keep tabletop paths and operator UIs outside the common workflow."""

    platform = "tabletop"

    def __init__(
        self,
        repository: str | Path,
        workflow_root: str | Path,
        *,
        selector=None,
        icp_inspector=None,
        initializer=None,
        operator_options: ReconstructionOperatorOptions | None = None,
        partition_options: DsfPartitionOptions | None = None,
        convex_decomposition_profile: ConvexDecompositionProfile | None = None,
        dsf_fitting_profile: DsfFittingProfile | None = None,
        manual_remover=None,
        manual_seed_provider=None,
        result_reviewer=None,
        operation_progress: Callable[[int, int, str], None] | None = None,
        prepare_continuation: Callable[[int, int, Path], bool] | None = None,
    ) -> None:
        self.repository = Path(repository).resolve()
        self.workflow_root = Path(workflow_root).resolve()
        self.selector = selector
        self.icp_inspector = icp_inspector
        self.initializer = initializer
        self.operator_options = operator_options
        self.partition_options = partition_options
        self.convex_decomposition_profile = convex_decomposition_profile
        self.dsf_fitting_profile = dsf_fitting_profile
        self.manual_remover = manual_remover
        self.manual_seed_provider = manual_seed_provider
        self.result_reviewer = result_reviewer
        self.operation_progress = operation_progress
        self.prepare_continuation = prepare_continuation

    def prepare(
        self,
        raw_capture_root: Path,
        output_directory: Path,
    ) -> StageExecutionResult:
        jobs = plan_capture_preparation(raw_capture_root, output_directory)
        prepared = []
        total = len(jobs)
        for index, (capture, output) in enumerate(jobs, start=1):
            prepared.append(
                run_prepare_capture(
                    self.repository,
                    capture,
                    output,
                    selector=self.selector,
                    icp_inspector=self.icp_inspector,
                    progress=(
                        None
                        if self.operation_progress is None
                        else lambda completed, count, message, name=capture.name: (
                            self.operation_progress(
                                completed,
                                count,
                                f"{name}: {message}",
                            )
                        )
                    ),
                )
            )
            if index < total and self.prepare_continuation is not None:
                next_capture = jobs[index][0]
                if not self.prepare_continuation(index, total, next_capture):
                    raise RuntimeError(
                        "operator stopped batch preparation after "
                        f"{index} of {total} captures"
                    )
        artifacts = tuple(
            self._artifact(
                f"prepared-{output_directory.name}-{index}",
                result.merged_path,
                "tool0",
            )
            for index, result in enumerate(prepared, start=1)
        )
        return StageExecutionResult(
            artifacts,
            metrics={
                "capture_count": len(prepared),
                "point_clouds": [
                    _report_point_count(result.report_path) for result in prepared
                ],
            },
        )

    def merge(
        self,
        prepared_artifacts: tuple[ArtifactReference, ...],
        output_directory: Path,
    ) -> StageExecutionResult:
        report_path = run_cross_capture_merge(
            self.repository,
            output_directory,
            object_id=_workflow_object_id(self.workflow_root),
            capture_directories=tuple(
                self.workflow_root / artifact.path
                for artifact in prepared_artifacts
            ),
            initializer=self.initializer,
            show_result=self.initializer is None,
            progress=self.operation_progress,
        )
        report = _read_report(report_path)
        merged = output_directory / str(report["merged_point_cloud"])
        return StageExecutionResult(
            (
                self._artifact(
                    f"merged-{output_directory.name}",
                    merged,
                    "first_capture_tool0",
                ),
            ),
            metrics={
                "point_count": int(report["point_count"]),
                "capture_count": len(prepared_artifacts),
            },
        )

    def reconstruct(
        self,
        merged_artifact: ArtifactReference,
        output_directory: Path,
    ) -> StageExecutionResult:
        workflow = _read_report(self.workflow_root / "workflow.json")
        measured_value = workflow.get("measured_mass_kg")
        measured_mass = None if measured_value is None else float(measured_value)
        density_value = workflow.get("density_kg_m3")
        if density_value is None:
            inventory = load_inventory_profile(
                self.repository
                / "apps"
                / "tabletop"
                / "config"
                / "inventory.yml"
            )
            density = inventory.density_for(_workflow_object_id(self.workflow_root))
        else:
            density = float(density_value)
        report_path = run_object_reconstruction(
            self.repository,
            self.workflow_root / merged_artifact.path,
            output_directory,
            object_id=_workflow_object_id(self.workflow_root),
            measured_mass_kg=measured_mass,
            density_kg_m3=density,
            convex_decomposition_profile=self.convex_decomposition_profile,
            dsf_fitting_profile=self.dsf_fitting_profile,
            operator_options=self.operator_options,
            partition_options=self.partition_options,
            manual_remover=self.manual_remover,
            manual_seed_provider=self.manual_seed_provider,
            result_reviewer=self.result_reviewer,
            progress=self.operation_progress,
        )
        report = _read_report(report_path)
        output = report["outputs"]
        unit_density_mass = float(report["inertial"]["unit_density_mass"])
        metadata_path = output_directory / str(output["metadata"])
        artifacts = (
            self._artifact(
                f"object-cloud-{output_directory.name}",
                output_directory / str(output["merged_point_cloud"]),
                "object_center_of_mass",
            ),
            self._artifact(
                f"object-mesh-{output_directory.name}",
                output_directory / str(output["mesh"]),
                "object_center_of_mass",
            ),
            self._artifact(
                f"object-dsf-{output_directory.name}",
                output_directory / str(output["dsf"]),
                "object_center_of_mass",
            ),
            self._artifact(
                f"object-metadata-{output_directory.name}",
                metadata_path,
                "object_center_of_mass",
            ),
        )
        return StageExecutionResult(
            artifacts,
            metrics={
                "point_count": int(report["cleaned_point_count"]),
                "component_count": int(
                    report["convex_decomposition"]["component_count"]
                ),
                "unit_density_mass": unit_density_mass,
                "mass_kg": float(report["inertial"]["mass_kg"]),
                "mass_source": str(report["inertial"]["mass_source"]),
                "effective_density_kg_m3": float(
                    report["inertial"]["effective_density_kg_m3"]
                ),
            },
        )

    def _artifact(self, artifact_id: str, path: Path, frame: str) -> ArtifactReference:
        return ArtifactReference(
            artifact_id=artifact_id,
            path=str(path.resolve().relative_to(self.workflow_root)),
            frame=frame,
        )


def _workflow_object_id(root: Path) -> int:
    return int(_read_report(root / "workflow.json")["object_id"])


def _read_report(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"report must contain a JSON object: {path}")
    return value


def _report_point_count(report_path: Path) -> int:
    report = _read_report(report_path)
    merge = report.get("object_merge", {})
    if isinstance(merge, dict):
        return int(merge.get("output_points", 0))
    return 0
