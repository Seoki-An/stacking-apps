"""Keep committed stack poses separate from lifted excavator motion targets."""

from dataclasses import replace

import numpy as np

from stacking_common.artifacts import PlacementPlanRecord
from stacking_common.planning import PlacementPlanningResult, PlanningStatus
from stacking_common.search import EvaluationFidelity


class ExcavatorPlanningCoordinator:
    def __init__(self, application, inventory, planner, *, placement_backend=None,
                 progress=None, context_sha256=None):
        self.application = application
        self.inventory = inventory
        self.planner = planner
        self.backend = placement_backend or application.build_core_backend(inventory, EvaluationFidelity.SHORT)
        self.backend.reset(inventory.object_ids)
        self.progress = progress
        self.context_sha256 = context_sha256

    def close(self):
        self.backend.close()

    def attempt(self, state, action, *, step, attempt, evaluated_state=None,
                pick_pose=None, candidate_id=None, writer=None):
        if tuple(state.object_ids) != self.inventory.object_ids:
            raise ValueError("planning state and inventory differ")
        if action.object_index < 0 or action.object_index >= len(self.inventory):
            raise ValueError("planning target is outside the inventory")
        if self.inventory[action.object_index].object_id != action.object_id:
            raise ValueError("planning target ID does not match its index")
        prepared = action.copy()
        if evaluated_state is not None:
            if tuple(evaluated_state.object_ids) != self.inventory.object_ids:
                raise ValueError("evaluated state and inventory differ")
            prepared.pose = np.asarray(evaluated_state.object_poses[action.object_id], dtype=float).copy()
        fixed = state.copy()
        fixed.pose_identified_object_ids.update(fixed.object_ids[i] for i in fixed.placed_object_indices)
        if self.progress:
            self.progress("fixed_scene_started", {"step": step, "attempt": attempt})
        placement = self.backend.settle_fixed(fixed, prepared)
        diagnostics = {"fixed_scene_resettle_enabled": True, **dict(placement.diagnostics)}
        rejection = None
        if placement.failed:
            rejection = "fixed_scene_resettle_" + str(placement.diagnostics.get("rejection_reason", "invalid"))
        else:
            prepared.pose = np.asarray(placement.object_poses[action.object_id], dtype=float).copy()
            if self.application.action_generation_profile.strict_floor_first:
                reason, floor = self.application.build_floor_policy(self.inventory).active_floor_rejection(state, prepared)
                diagnostics.update(floor)
                relaxed = bool(action.diagnostics.get("floor_fill_fallback", False))
                if relaxed:
                    diagnostics.update(floor_fill_fallback=True, floor_fill_bypassed_reason=reason)
                if reason and not relaxed:
                    rejection = "active_floor_" + reason
        pick = self.inventory.initial_pose(action.object_id) if pick_pose is None else pick_pose
        remaining = {object_id: self.inventory.initial_pose(object_id)
                     for i, object_id in enumerate(state.object_ids) if i not in state.placed_object_indices}
        problem = self.application.build_placement_planning_problem(
            state, prepared, pick_pose=pick, pick_scene_poses=remaining,
            metadata={"planning_attempt": attempt, **diagnostics})
        if rejection:
            result = PlacementPlanningResult(PlanningStatus.INFEASIBLE, reason=rejection)
        else:
            if self.progress:
                self.progress("motion_started", {"step": step, "attempt": attempt})
            result = self.planner.plan(problem)
        result = replace(result, diagnostics={**result.diagnostics,
                                              "motion_planning_attempted": rejection is None})
        record = PlacementPlanRecord(step, attempt, problem, result,
                                     candidate_id, self.context_sha256)
        if writer is not None:
            writer.append_placement_plan(record)
        return record
