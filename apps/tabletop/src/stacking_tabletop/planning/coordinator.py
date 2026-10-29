"""One-attempt orchestration from an MCTS action to a stored motion plan."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np

from stacking_common.artifacts import PlacementPlanRecord
from stacking_common.domain import Action, ObjectInventory, State
from stacking_common.planning import (
    FrameTransform,
    PlacementPlanner,
    PlacementPlanningProblem,
    PlacementPlanningResult,
    PlanningStatus,
)

from .safety import TabletopPlacementSafety


class TabletopPlanningCoordinator:
    """Build, solve, and optionally append one placement-planning attempt."""

    def __init__(
        self,
        inventory: ObjectInventory,
        problem_builder: Callable[..., PlacementPlanningProblem],
        planner: PlacementPlanner,
        *,
        placement_backend=None,
        world_T_stack: FrameTransform | None = None,
        progress: Callable[[str, Mapping[str, object]], None] | None = None,
        context_sha256: str | None = None,
    ) -> None:
        if not callable(problem_builder):
            raise TypeError("planning problem builder must be callable")
        if not hasattr(planner, "plan") or not callable(planner.plan):
            raise TypeError("planning coordinator requires a placement planner")
        self.inventory = inventory
        self.problem_builder = problem_builder
        self.planner = planner
        self.placement_backend = placement_backend
        self.world_T_stack = world_T_stack
        self.progress = progress
        self.context_sha256 = context_sha256

    def attempt(
        self,
        state: State,
        action: Action,
        *,
        step: int,
        attempt: int,
        evaluated_state: State | None = None,
        pick_pose: np.ndarray | None = None,
        candidate_id: str | None = None,
        writer=None,
    ) -> PlacementPlanRecord:
        if tuple(state.object_ids) != self.inventory.object_ids:
            raise ValueError("planning state does not match the inventory")
        if action.object_index >= len(self.inventory):
            raise ValueError("planning action is outside the inventory")
        object_id = self.inventory.object_ids[action.object_index]
        if action.object_id != object_id:
            raise ValueError("planning action ID does not match the inventory")
        evaluated = state if evaluated_state is None else evaluated_state
        if tuple(evaluated.object_ids) != self.inventory.object_ids:
            raise ValueError("evaluated planning state does not match the inventory")
        try:
            evaluated_pose = evaluated.object_poses[object_id]
        except KeyError as error:
            if evaluated_state is not None:
                raise ValueError(
                    "evaluated candidate has no target placement pose"
                ) from error
            evaluated_pose = action.pose
        planning_action = action.copy()
        planning_action.pose = np.asarray(evaluated_pose, dtype=float).copy()
        resolved_pick = (
            self.inventory.initial_pose(object_id)
            if pick_pose is None
            else np.asarray(pick_pose, dtype=float)
        )
        self._emit("fixed_scene_started", step=step, attempt=attempt)
        preparation = self._prepare_fixed_scene(state, planning_action)
        planning_action = preparation[0]
        preparation_failure = preparation[1]
        diagnostics = preparation[2]
        self._emit(
            "fixed_scene_completed",
            step=step,
            attempt=attempt,
            accepted=preparation_failure is None,
        )
        problem = self.problem_builder(
            state,
            planning_action,
            pick_pose=resolved_pick,
            metadata={
                "planning_attempt": int(attempt),
                **diagnostics,
            },
        )
        if preparation_failure is not None:
            result = preparation_failure
        else:
            self._emit("motion_started", step=step, attempt=attempt)
            result = self.planner.plan(problem)
            self._emit(
                "motion_completed",
                step=step,
                attempt=attempt,
                status=result.status.value,
            )
        if result.succeeded:
            self._emit("support_validation_started", step=step, attempt=attempt)
            result = self._validate_released_pose(state, problem, result)
            self._emit(
                "support_validation_completed",
                step=step,
                attempt=attempt,
                status=result.status.value,
            )
        result = replace(result, diagnostics={
            **result.diagnostics,
            "motion_planning_attempted": preparation_failure is None,
        })
        record = PlacementPlanRecord(
            step=step,
            attempt=attempt,
            problem=problem,
            result=result,
            candidate_id=candidate_id,
            context_sha256=self.context_sha256,
        )
        if writer is not None:
            append = getattr(writer, "append_placement_plan", None)
            if not callable(append):
                raise TypeError("planning writer cannot append placement plans")
            append(record)
        return record

    def _prepare_fixed_scene(
        self,
        state: State,
        action: Action,
    ) -> tuple[Action, PlacementPlanningResult | None, dict[str, object]]:
        diagnostics: dict[str, object] = {}
        prepared = action.copy()
        if self.placement_backend is not None:
            fixed_state = state.copy()
            fixed_state.pose_identified_object_ids.update(
                fixed_state.object_ids[index]
                for index in fixed_state.placed_object_indices
            )
            settle = getattr(self.placement_backend, "settle_fixed", None)
            placement = (
                settle(fixed_state, prepared)
                if callable(settle)
                else self.placement_backend.place(fixed_state, prepared)
            )
            diagnostics.update(
                {
                    "fixed_scene_resettle_enabled": True,
                    "fixed_scene_resettle_valid": not placement.failed,
                    **dict(placement.diagnostics),
                }
            )
            if placement.failed:
                reason = str(
                    placement.diagnostics.get(
                        "rejection_reason", "invalid"
                    )
                )
                diagnostics["fixed_scene_resettle_reject_reason"] = reason
                return prepared, self._infeasible(
                    f"fixed_scene_resettle_{reason}", diagnostics
                ), diagnostics
            previous_pose = prepared.pose.copy()
            settled_pose = np.asarray(
                placement.object_poses[prepared.object_id], dtype=float
            )
            diagnostics["fixed_scene_resettle_position_delta"] = float(
                np.linalg.norm(settled_pose[:3] - previous_pose[:3])
            )
            diagnostics["fixed_scene_resettle_rotation_delta"] = (
                self._quaternion_distance(
                    previous_pose[3:], settled_pose[3:]
                )
            )
            prepared.pose = settled_pose.copy()

        return prepared, None, diagnostics

    @staticmethod
    def _quaternion_distance(left: np.ndarray, right: np.ndarray) -> float:
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        cosine = abs(float(np.dot(left, right))) / denominator
        return float(2.0 * np.arccos(np.clip(cosine, -1.0, 1.0)))

    def _validate_released_pose(
        self,
        state: State,
        problem: PlacementPlanningProblem,
        result: PlacementPlanningResult,
    ) -> PlacementPlanningResult:
        if self.placement_backend is None or self.world_T_stack is None:
            return result
        achieved = result.diagnostics.get("achieved_place_pose")
        if achieved is None:
            return result
        achieved_action = problem.action.copy()
        achieved_action.pose = self.world_T_stack.parent_pose_to_child(
            np.asarray(achieved, dtype=float)
        )
        fixed_state = state.copy()
        fixed_state.pose_identified_object_ids.update(
            fixed_state.object_ids[index]
            for index in fixed_state.placed_object_indices
        )
        placement = self.placement_backend.place(fixed_state, achieved_action)
        settled = placement.object_poses.get(achieved_action.object_id)
        gap = (
            float("inf")
            if settled is None
            else float(np.linalg.norm(settled[:3] - achieved_action.pose[:3]))
        )
        diagnostics = dict(result.diagnostics)
        diagnostics["place_support_gap"] = gap
        tolerance = TabletopPlacementSafety.support_tolerance
        diagnostics["place_support_tolerance"] = tolerance
        if placement.failed or gap > tolerance:
            return PlacementPlanningResult(
                PlanningStatus.INFEASIBLE,
                segments=result.segments,
                grasp_events=result.grasp_events,
                reason="place_target_unsupported",
                score=result.score,
                diagnostics={**diagnostics, "retryable": True},
            )
        return PlacementPlanningResult(
            result.status,
            segments=result.segments,
            grasp_events=result.grasp_events,
            score=result.score,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _infeasible(reason: str, diagnostics) -> PlacementPlanningResult:
        return PlacementPlanningResult(
            PlanningStatus.INFEASIBLE,
            reason=reason,
            diagnostics={**diagnostics, "retryable": True},
        )

    def _emit(self, event: str, **details: object) -> None:
        if self.progress is not None:
            self.progress(event, details)
