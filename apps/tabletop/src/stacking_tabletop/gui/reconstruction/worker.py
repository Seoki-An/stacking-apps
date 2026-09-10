"""Background execution of one object-build stage."""

from pathlib import Path
from time import monotonic

from PySide6.QtCore import QThread, Signal

from stacking_common.reconstruction import (
    ConvexDecompositionProfile,
    DsfFittingProfile,
    DsfPartitionOptions,
    ReconstructionOperatorOptions,
)
from stacking_common.workflows.object_build import (
    ObjectBuildRunner,
    ObjectBuildWorkflow,
)
from stacking_tabletop.perception.workflow import TabletopObjectBuildAdapter

from .operators import WorkbenchOperators


class StageWorker(QThread):
    succeeded = Signal(str, str)
    failed = Signal(str, str)
    progress = Signal(int, int, str)

    def __init__(
        self,
        repository: Path,
        workflow_root: Path,
        stage: str,
        capture_directory: Path | None,
        operators: WorkbenchOperators,
        *,
        manual_cleanup: bool,
        manual_partition: bool,
        review_result: bool,
        inspect_icp: bool,
        convex_decomposition_profile: ConvexDecompositionProfile,
        dsf_fitting_profile: DsfFittingProfile,
    ) -> None:
        super().__init__()
        self.repository = repository
        self.workflow_root = workflow_root
        self.stage = stage
        self.capture_directory = capture_directory
        self.operators = operators
        self.manual_cleanup = manual_cleanup
        self.manual_partition = manual_partition
        self.review_result = review_result
        self.inspect_icp = inspect_icp
        self.convex_decomposition_profile = convex_decomposition_profile
        self.dsf_fitting_profile = dsf_fitting_profile
        self.completed_attempt_id: str | None = None
        self._last_progress_time = 0.0
        self._last_progress_message: str | None = None

    def run(self) -> None:
        try:
            workflow = ObjectBuildWorkflow.resume(self.workflow_root)
            adapter = TabletopObjectBuildAdapter(
                self.repository,
                workflow.root,
                selector=self.operators,
                icp_inspector=(
                    self.operators if self.inspect_icp else None
                ),
                initializer=self.operators,
                operator_options=ReconstructionOperatorOptions(
                    manual_remove=self.manual_cleanup,
                ),
                partition_options=DsfPartitionOptions(
                    manual_split=self.manual_partition,
                    review_result=self.review_result,
                ),
                convex_decomposition_profile=self.convex_decomposition_profile,
                dsf_fitting_profile=self.dsf_fitting_profile,
                manual_remover=self.operators.remove,
                manual_seed_provider=self.operators.split_seeds,
                result_reviewer=(
                    self.operators.review if self.review_result else None
                ),
                operation_progress=self._report_progress,
                prepare_continuation=self.operators.continue_prepare,
            )
            runner = ObjectBuildRunner(workflow, adapter)
            if self.stage == "prepare":
                if self.capture_directory is None:
                    raise ValueError("capture directory is required for prepare")
                attempt = runner.run_prepare(self.capture_directory)
            elif self.stage == "merge":
                attempt = runner.run_merge()
            elif self.stage == "reconstruct":
                attempt = runner.run_reconstruct()
            else:
                raise ValueError(f"unsupported stage: {self.stage}")
        except Exception as error:
            self.failed.emit(self.stage, str(error))
            return
        self.completed_attempt_id = attempt.attempt_id
        self.succeeded.emit(self.stage, attempt.attempt_id)

    def _report_progress(self, completed: int, total: int, message: str) -> None:
        """Limit repaint traffic while preserving phase and completion events."""

        now = monotonic()
        phase_changed = message != self._last_progress_message
        completed_operation = total > 0 and completed >= total
        if (
            phase_changed
            or completed_operation
            or now - self._last_progress_time >= 0.05
        ):
            self.progress.emit(completed, total, message)
            self._last_progress_time = now
            self._last_progress_message = message
