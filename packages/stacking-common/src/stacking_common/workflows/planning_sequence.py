"""Reusable MCTS-to-motion-planning sequence orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from collections.abc import Callable, Mapping
from typing import Generic, Protocol, TypeVar

import numpy as np

from stacking_common.artifacts import PlacementPlanRecord, PlanningCheckpoint, StackRunWriter
from stacking_common.actions.cem import CEMStatistics
from stacking_common.domain import Action, State
from stacking_common.planning import PlanningStatus
from stacking_common.search import MCTSSearchRequest, MCTSSession
from stacking_common.visualization import MCTSDebugStep, project_mcts_step
from stacking_common.workflows.mcts_reporting import tree_statistics


ObservationT = TypeVar("ObservationT")


class PlacementAttemptRunner(Protocol):
    """Build and solve one selected search action."""

    def attempt(
        self,
        state: State,
        action: Action,
        *,
        step: int,
        attempt: int,
        evaluated_state: State | None = None,
        candidate_id: str | None = None,
        writer=None,
    ) -> PlacementPlanRecord: ...


class PlanningSequenceStatus(str, Enum):
    COMPLETE = "complete"
    SEARCH_EXHAUSTED = "search_exhausted"
    SEARCH_BACKEND_FAILED = "search_backend_failed"
    PLANNING_EXHAUSTED = "planning_exhausted"
    PLANNING_FAILED = "planning_failed"


@dataclass(frozen=True)
class SequenceSearchAttempt:
    """Compact search accounting retained across planning retries."""

    attempt: int
    elapsed_seconds: float
    completed_iterations: int
    generated_candidates: int
    evaluated_candidates: int
    failed_evaluations: int
    selected_candidate_id: str | None
    cem_statistics: tuple[CEMStatistics, ...] = ()
    effective_max_depth: int | None = None
    timed_out: bool = False
    execution_rejected_candidates: int = 0
    tree_nodes: int = 0
    tree_leaves: int = 0
    backend_failure: dict[str, object] | None = None


@dataclass(frozen=True)
class PlannedSequenceStep:
    """One committed search step and all of its planning attempts."""

    step: int
    search_attempts: tuple[SequenceSearchAttempt, ...]
    planning_attempts: tuple[PlacementPlanRecord, ...]
    debug_step: MCTSDebugStep
    selected_action: Action | None
    elapsed_seconds: float

    @property
    def succeeded(self) -> bool:
        return self.selected_action is not None


@dataclass(frozen=True)
class PlanningSequenceResult:
    """Terminal result of bounded search-and-plan orchestration."""

    status: PlanningSequenceStatus
    steps: tuple[PlannedSequenceStep, ...]
    final_state: State
    reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is PlanningSequenceStatus.COMPLETE


class MCTSPlanningSequenceCoordinator(Generic[ObservationT]):
    """Search, motion-plan, reject infeasible actions, and advance the root."""

    def __init__(
        self,
        session: MCTSSession[ObservationT],
        request: MCTSSearchRequest,
        planner: PlacementAttemptRunner,
        *,
        frame_id: str,
        writer: StackRunWriter | None = None,
        progress: Callable[[str, Mapping[str, object]], None] | None = None,
        checkpoint_builder: Callable[[int, State], PlanningCheckpoint] | None = None,
        committed_state_evaluator: Callable[
            [State], tuple[ObservationT, float, Mapping[str, object]]
        ] | None = None,
    ) -> None:
        frame_id = str(frame_id).strip()
        if not frame_id:
            raise ValueError("planning sequence frame ID must not be empty")
        if not hasattr(planner, "attempt") or not callable(planner.attempt):
            raise TypeError("planning sequence requires an attempt runner")
        self.session = session
        self.request = request
        self.planner = planner
        self.frame_id = frame_id
        self.writer = writer
        self.progress = progress
        self.checkpoint_builder = checkpoint_builder
        self.committed_state_evaluator = committed_state_evaluator

    def run(
        self,
        max_steps: int,
        *,
        max_planning_attempts_per_step: int | None = None,
        first_step: int = 1,
    ) -> PlanningSequenceResult:
        max_steps = int(max_steps)
        if isinstance(first_step, bool) or not isinstance(first_step, int) or first_step <= 0:
            raise ValueError("first planning step must be a positive integer")
        max_attempts = (
            None
            if max_planning_attempts_per_step is None
            else int(max_planning_attempts_per_step)
        )
        if max_steps <= 0:
            raise ValueError("planning sequence step limit must be positive")
        if max_attempts is not None and max_attempts <= 0:
            raise ValueError("planning attempt limit must be positive")

        steps: list[PlannedSequenceStep] = []
        status = PlanningSequenceStatus.COMPLETE
        reason = None
        for step_number in range(first_step, first_step + max_steps):
            if self.session.root.terminal:
                break
            self._emit("step_started", step=step_number)
            step, status, reason = self._run_step(step_number, max_attempts)
            steps.append(step)
            self._emit(
                "step_completed",
                step=step_number,
                status=status.value,
                elapsed_seconds=step.elapsed_seconds,
            )
            if not step.succeeded:
                break
        return PlanningSequenceResult(
            status,
            tuple(steps),
            self.session.root.state.copy(),
            reason,
        )

    def _run_step(
        self,
        step: int,
        max_attempts: int | None,
    ) -> tuple[PlannedSequenceStep, PlanningSequenceStatus, str | None]:
        started = time.perf_counter()
        searches: list[SequenceSearchAttempt] = []
        plans: list[PlacementPlanRecord] = []
        selected = None
        status = PlanningSequenceStatus.PLANNING_EXHAUSTED
        reason: str | None = None
        planning_attempt = 0
        search_attempt = 0
        failure_counts: dict[int, int] = {}
        skipped_candidates: list[dict[str, object]] = []

        while max_attempts is None or planning_attempt < max_attempts:
            search_attempt += 1
            self._emit(
                "search_started",
                step=step,
                search_attempt=search_attempt,
            )
            search_started = time.perf_counter()
            result = self.session.search(self.request)
            search_elapsed = time.perf_counter() - search_started
            tree = tree_statistics(result.root)
            generator = getattr(self.session.search_engine, "root_action_generator", None)
            cem_statistics = tuple(getattr(generator, "statistics_history", ()))
            self._assign_candidate_ids(result.root, step)
            self._emit(
                "search_completed",
                step=step,
                search_attempt=search_attempt,
                elapsed_seconds=search_elapsed,
                generated_candidates=result.generated_candidates,
                evaluated_candidates=result.evaluated_candidates,
                ranked_candidates=len(result.ranked_candidates),
                cem_statistics=tuple(value.to_dict() for value in cem_statistics),
                execution_rejected_candidates=getattr(result, "execution_rejected_candidates", 0),
                tree_nodes=tree["nodes"],
                tree_leaves=tree["leaves"],
            )
            candidate_id = self._candidate_id(result.selected)
            searches.append(
                SequenceSearchAttempt(
                    search_attempt,
                    search_elapsed,
                    result.completed_iterations,
                    result.generated_candidates,
                    result.evaluated_candidates,
                    result.failed_evaluations,
                    candidate_id,
                    cem_statistics,
                    getattr(result, "effective_max_depth", None),
                    bool(getattr(result, "timed_out", False)),
                    int(getattr(result, "execution_rejected_candidates", 0)),
                    tree_nodes=tree["nodes"],
                    tree_leaves=tree["leaves"],
                    backend_failure=getattr(result, "backend_failure", None),
                )
            )
            if result.selected is None:
                failure = getattr(result, "backend_failure", None)
                status = (PlanningSequenceStatus.SEARCH_BACKEND_FAILED if failure
                          else PlanningSequenceStatus.SEARCH_EXHAUSTED)
                reason = failure["message"] if failure else "MCTS produced no selectable root candidate"
                break

            ranked = tuple(getattr(result, "ranked_candidates", ()))
            if not ranked:
                ranked = (result.selected,)
            pending_rejections = []
            for rank, node in enumerate(ranked, start=1):
                if not node.selectable:
                    details = dict(
                        step=step, search_attempt=search_attempt,
                        candidate_rank=rank, candidate_count=len(ranked),
                        candidate_id=self._candidate_id(node),
                        reason=node.rejection_reason or "not_selectable",
                    )
                    skipped_candidates.append(details)
                    self._emit("candidate_skipped", **details)
                    continue
                if max_attempts is not None and planning_attempt >= max_attempts:
                    reason = "planning attempt limit reached"
                    break
                planning_attempt += 1
                candidate_id = self._candidate_id(node)
                self._emit(
                    "candidate_started",
                    step=step,
                    planning_attempt=planning_attempt,
                    candidate_rank=rank,
                    candidate_count=len(ranked),
                    candidate_id=candidate_id,
                    object_id=node.action.object_id,
                )
                record = self.planner.attempt(
                    result.root.state,
                    node.action,
                    step=step,
                    attempt=planning_attempt,
                    evaluated_state=node.state,
                    candidate_id=candidate_id,
                    writer=self.writer,
                )
                plans.append(record)
                self._emit(
                    "candidate_completed",
                    step=step,
                    planning_attempt=planning_attempt,
                    candidate_id=candidate_id,
                    object_id=node.action.object_id,
                    status=record.result.status.value,
                    reason=record.result.reason,
                )
                if record.result.succeeded:
                    self._restore_execution_candidate(node)
                    self._commit_planned_pose(result.root.state, node, record)
                    selected = node
                    status = PlanningSequenceStatus.COMPLETE
                    reason = None
                    break

                object_id = int(node.action.object_id)
                motion_attempted = bool(record.result.diagnostics.get(
                    "motion_planning_attempted",
                    not (record.result.reason or "").startswith(("active_floor_", "fixed_scene_resettle_")),
                ))
                if motion_attempted and record.result.status is not PlanningStatus.CANCELLED:
                    failure_counts[object_id] = failure_counts.get(object_id, 0) + 1
                # Legacy tries the entire returned batch before feeding back
                # neighborhood/object exclusions to the next MCTS search.
                pending_rejections.append(record.problem.action.copy())
                node.reject("execution_rejected_action", info={
                    **node.info, "planning_failure_reason": record.result.reason,
                })
                if record.result.status is PlanningStatus.CANCELLED:
                    status = PlanningSequenceStatus.PLANNING_FAILED
                    reason = record.result.reason or "motion planning cancelled"
                    break
            if selected is None:
                for action in pending_rejections:
                    self.session.reject_execution(
                        action,
                        reject_object=failure_counts.get(action.object_id, 0) >= 2,
                    )
            if selected is not None or status is PlanningSequenceStatus.PLANNING_FAILED:
                break

        if selected is None and reason is None:
            reason = "planning attempt limit reached"

        debug_step = project_mcts_step(
            self.session.root,
            selected,
            step=step,
            frame_id=self.frame_id,
            failure_reason=(None if selected is not None else status.value),
        )
        elapsed = time.perf_counter() - started
        if self.writer is not None:
            checkpoint_args = {}
            if selected is not None and self.checkpoint_builder is not None:
                checkpoint_args["checkpoint"] = self.checkpoint_builder(step, selected.state.copy())
            self.writer.append_mcts_step(
                debug_step,
                statistics=self._statistics(
                    step,
                    searches,
                    plans,
                    status,
                    elapsed,
                    skipped_candidates,
                ),
                **checkpoint_args,
            )
        selected_action = None if selected is None else selected.action.copy()
        sequence_step = PlannedSequenceStep(
            step,
            tuple(searches),
            tuple(plans),
            debug_step,
            selected_action,
            elapsed,
        )
        if selected is not None:
            self.session.promote(selected, preserve_subtree=True)
            self.session.clear_execution_rejections()
        return sequence_step, status, reason

    def _commit_planned_pose(self, parent_state, node, record) -> None:
        """Commit the fixed-scene pose that motion planning actually used."""

        if node.state is None or node.action is None:
            raise ValueError("a successful planning candidate must be evaluated")
        action = record.problem.action.copy()
        committed = node.state.copy()
        for object_id, pose in parent_state.object_poses.items():
            committed.object_poses[object_id] = pose.copy()
        committed.object_poses[action.object_id] = action.pose.copy()
        poses_changed = any(
            object_id not in node.state.object_poses
            or not np.array_equal(pose, node.state.object_poses[object_id])
            for object_id, pose in committed.object_poses.items()
        )
        if poses_changed and self.committed_state_evaluator is None:
            raise ValueError(
                "committing changed placement poses requires a committed-state evaluator"
            )
        if committed.action_history:
            committed.action_history[-1].pose = action.pose.copy()

        info = dict(node.info)
        if poses_changed:
            node.clear_children()
            node.reset_statistics(recursive=True)
            # Contacts, forces, and the latest trajectory describe the old
            # simulated scene. They cannot establish contacts at the committed
            # pose without another simulation, which would move that pose again.
            committed.contact_points = []
            if committed.trajectories:
                committed.trajectories[-1] = {}
            for key in (
                "posegen_contact_points", "posegen_contact_normals", "posegen_contact_forces"
            ):
                info.pop(key, None)
            info.pop("authoritative_fidelity", None)
            info["committed_pose_changed"] = True
            info["committed_contacts_available"] = False

        observation, reward = node.observation, node.reward
        if self.committed_state_evaluator is not None:
            observation, reward, diagnostics = self.committed_state_evaluator(
                committed.copy()
            )
            info.update(diagnostics)
        if poses_changed:
            # These are unknown after invalidating the simulation contacts,
            # rather than evidence of an unsupported placement.
            info.pop("support_count", None)
            info.pop("support_has_ground", None)
        node.set_result(committed, observation, reward, info=info)
        node.action.pose = action.pose.copy()

    @staticmethod
    def _restore_execution_candidate(node) -> None:
        if node.rejection_reason not in {
            "execution_rejected_action",
            "execution_rejected_object",
        }:
            return
        node.rejection_reason = None
        node.failed = bool(node.state is not None and node.state.failed)

    def _emit(self, event: str, **details: object) -> None:
        if self.progress is not None:
            self.progress(event, details)

    @staticmethod
    def _candidate_id(node) -> str | None:
        if node is None:
            return None
        return str(node.info["candidate_id"])

    @staticmethod
    def _assign_candidate_ids(root, step: int) -> None:
        used = {
            str(child.info["candidate_id"])
            for child in root.children
            if "candidate_id" in child.info
        }
        next_index = 1
        for child in root.children:
            if "candidate_id" in child.info:
                continue
            while f"step-{step}-candidate-{next_index}" in used:
                next_index += 1
            candidate_id = f"step-{step}-candidate-{next_index}"
            child.info["candidate_id"] = candidate_id
            used.add(candidate_id)
            next_index += 1

    @staticmethod
    def _statistics(step, searches, plans, status, elapsed, skipped_candidates=()):
        return {
            "step": int(step),
            "elapsed_seconds": float(elapsed),
            "status": status.value,
            "skipped_candidates": list(skipped_candidates),
            "search_attempts": [
                {
                    "attempt": value.attempt,
                    "elapsed_seconds": value.elapsed_seconds,
                    "completed_iterations": value.completed_iterations,
                    "generated_candidates": value.generated_candidates,
                    "evaluated_candidates": value.evaluated_candidates,
                    "failed_evaluations": value.failed_evaluations,
                    "selected_candidate_id": value.selected_candidate_id,
                    "cem_statistics": [entry.to_dict() for entry in value.cem_statistics],
                    "effective_max_depth": value.effective_max_depth,
                    "timed_out": value.timed_out,
                    "execution_rejected_candidates": value.execution_rejected_candidates,
                    "tree_nodes": value.tree_nodes,
                    "tree_leaves": value.tree_leaves,
                    "backend_failure": value.backend_failure,
                }
                for value in searches
            ],
            "planning_attempts": [
                {
                    "attempt": value.attempt,
                    "candidate_id": value.candidate_id,
                    "status": value.result.status.value,
                    "reason": value.result.reason,
                    "elapsed_seconds": value.result.diagnostics.get(
                        "elapsed_seconds"
                    ),
                }
                for value in plans
            ],
        }
