"""Small composable filters for solved placement candidates."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from stacking_common.domain import Action, ObjectInventory, State

from .types import (
    InitializedPlacement,
    PlacementFilterDecision,
    PlacementSolveResult,
)


class InitialXYDisplacementFilter:
    """Keep optimized and simulated endpoints near this proposal's initial XY.

    The length scale is the candidate stone's cube-root volume. This is a
    proposal-locality restriction, not a structural stability test.
    """

    def __init__(self, inventory, max_ratio):
        if isinstance(max_ratio, bool) or not np.isfinite(max_ratio) or max_ratio <= 0:
            raise ValueError("max initial XY displacement ratio must be finite and positive")
        volumes = np.asarray([obj.volume for obj in inventory], dtype=float)
        if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
            raise ValueError("initial XY displacement filtering requires positive finite stone volumes")
        self.length_scales = np.cbrt(volumes)
        self.max_ratio = float(max_ratio)

    def check(self, action, final_pose=None):
        length = float(self.length_scales[action.object_index])
        solved_shift = float(np.linalg.norm(action.solved_pose[:2] - action.init_pose[:2]))
        final_shift = (None if final_pose is None else
                       float(np.linalg.norm(final_pose[:2] - action.init_pose[:2])))
        shift = solved_shift if final_shift is None else max(solved_shift, final_shift)
        valid = np.isfinite(solved_shift) and (final_shift is None or np.isfinite(final_shift))
        diagnostics = {
            "posegen_initial_xy_displacement": solved_shift,
            "final_initial_xy_displacement": final_shift,
            "initial_xy_displacement_ratio": shift / length,
            "initial_xy_displacement_ratio_limit": self.max_ratio,
            "initial_xy_displacement_limit": self.max_ratio * length,
            "initial_xy_displacement_stage": "posegen" if final_pose is None else "simulation",
        }
        if not valid or shift > self.max_ratio * length:
            return PlacementFilterDecision.reject("initial_xy_displacement", **diagnostics)
        return PlacementFilterDecision.accept(**diagnostics)

    def evaluate(self, inventory, state, object_index, initial, solved):
        return self.check(Action(object_index, solved.pose, initial.pose))


class ActiveLayerAdmissionFilter:
    """Restrict solved proposals to the parent's lowest incomplete layer.

    This is a construction-order restriction, not a physical stability check.
    Connectivity, empty-area overlap and repair are deliberately independent.
    """

    def __init__(self, floor_policy, *, relaxed=False) -> None:
        self.floor_policy = floor_policy
        self.relaxed = relaxed

    def prepare_scene(self, inventory, state):
        return self.floor_policy.context_details(state)

    def evaluate(self, inventory, state, object_index, initial, solved, *, scene_context=None):
        action = Action(object_index, solved.pose, initial.pose,
                        object_id=inventory[object_index].object_id)
        diagnostics = self.floor_policy.pose_diagnostics(state, action, context=scene_context)
        diagnostics["active_layer_admission_checked"] = True
        if self.relaxed:
            return PlacementFilterDecision.accept(
                **diagnostics, floor_fill_fallback=True, active_layer_admission_relaxed=True,
                floor_fill_bypassed_reason=(
                    "above_active_layer" if diagnostics["fixed_scene_active_layer_above"] else None),
            )
        if diagnostics["fixed_scene_active_layer_above"]:
            return PlacementFilterDecision.reject("above_active_layer", **diagnostics)
        return PlacementFilterDecision.accept(**diagnostics)


class ScanOrientationFilter:
    """Preserve legacy scan-relative tilt checks and the lower-fill exception."""

    def __init__(self, floor_policy) -> None:
        self.floor_policy = floor_policy

    def evaluate(self, inventory, state, object_index, initial, solved):
        object_id = inventory[object_index].object_id
        scan = Rotation.from_quat(inventory.initial_pose(object_id)[3:])
        difference = Rotation.from_quat(solved.pose[3:]) * scan.inv()
        xy_norm = float(np.linalg.norm(difference.as_rotvec()[:2]))
        limit = float(np.pi / 4.0)
        matrix_zz = float(difference.as_matrix()[2, 2])
        too_tilted = xy_norm > limit
        flipped = matrix_zz < 0.0
        orientation_ok = not (too_tilted or flipped)

        action = Action(object_index, solved.pose, initial.pose, object_id=object_id)
        # Lower-fill overlap is an exception to the optional scan tilt check.
        floor = self.floor_policy.pose_diagnostics(state, action)
        lower_fill = (
            floor["fixed_scene_active_layer_fill_score"] > 0.0
            and not floor["fixed_scene_active_layer_above"]
        )
        diagnostics = {
            "scan_orientation_rotvec_xy_norm": xy_norm,
            "scan_orientation_rotvec_xy_limit": limit,
            "scan_orientation_matrix_zz": matrix_zz,
            "scan_orientation_too_tilted": bool(too_tilted),
            "scan_orientation_flipped": bool(flipped),
            "scan_orientation_ok": bool(orientation_ok),
            "lower_fill_candidate": bool(lower_fill),
            "scan_orientation_soft_lower_fill": bool(not orientation_ok and lower_fill),
        }
        if not orientation_ok and not lower_fill:
            return PlacementFilterDecision.reject("scan_orientation", **diagnostics)
        return PlacementFilterDecision.accept(**diagnostics)


class PosegenFeasibilityFilter:
    """Reject failed solves and excessive pose-generator gap residuals."""

    def __init__(
        self,
        gap_threshold: float | None,
        equilibrium_threshold: float | None = None,
        *, require_solver_convergence: bool = True,
        reject_equilibrium_at_threshold: bool = False,
    ) -> None:
        self.require_solver_convergence = require_solver_convergence
        self.reject_equilibrium_at_threshold = reject_equilibrium_at_threshold
        if gap_threshold is not None and (
            not np.isfinite(gap_threshold) or gap_threshold < 0.0
        ):
            raise ValueError("pose-generator gap threshold must be non-negative")
        self.gap_threshold = (
            None if gap_threshold is None else float(gap_threshold)
        )
        if equilibrium_threshold is not None and (
            not np.isfinite(equilibrium_threshold)
            or equilibrium_threshold < 0.0
        ):
            raise ValueError(
                "pose-generator equilibrium threshold must be non-negative"
            )
        self.equilibrium_threshold = (
            None
            if equilibrium_threshold is None
            else float(equilibrium_threshold)
        )

    def evaluate(
        self,
        inventory: ObjectInventory,
        state: State,
        object_index: int,
        initial: InitializedPlacement,
        solved: PlacementSolveResult,
    ) -> PlacementFilterDecision:
        del inventory, state, object_index, initial
        diagnostics = {
            "posegen_c_feq": solved.c_feq,
            "posegen_equilibrium_residual_normalized": solved.c_feq,
            "posegen_equilibrium_threshold": self.equilibrium_threshold,
            "posegen_c_gap": solved.c_gap,
            "posegen_gap_threshold": self.gap_threshold,
        }
        if not solved.valid:
            return PlacementFilterDecision.reject(solved.rejection_reason, **diagnostics)
        if self.require_solver_convergence and not solved.converged:
            return PlacementFilterDecision.reject(
                solved.rejection_reason or "pose_solver",
                **diagnostics,
            )
        equilibrium_ok = (
            self.equilibrium_threshold is None
            or (solved.c_feq < self.equilibrium_threshold if self.reject_equilibrium_at_threshold
                else solved.c_feq <= self.equilibrium_threshold)
        )
        diagnostics["posegen_equilibrium_ok"] = bool(equilibrium_ok)
        if not equilibrium_ok:
            return PlacementFilterDecision.reject(
                "posegen_c_feq", **diagnostics
            )
        gap_ok = self.gap_threshold is None or solved.c_gap <= self.gap_threshold
        diagnostics["posegen_gap_ok"] = bool(gap_ok)
        if not gap_ok:
            return PlacementFilterDecision.reject("posegen_gap", **diagnostics)
        return PlacementFilterDecision.accept(**diagnostics)


class TargetContainmentFilter:
    """Reject solved poses that already fall outside the target threshold."""

    def __init__(self, overlap_evaluator, threshold: float) -> None:
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("target containment threshold must be between zero and one")
        self.overlap_evaluator = overlap_evaluator
        self.threshold = float(threshold)

    def evaluate(
        self,
        inventory: ObjectInventory,
        state: State,
        object_index: int,
        initial: InitializedPlacement,
        solved: PlacementSolveResult,
    ) -> PlacementFilterDecision:
        del state, initial
        containment = float(
            self.overlap_evaluator.object_containment(
                inventory[object_index], solved.pose
            )
        )
        diagnostics = {
            "presimulation_target_containment_fraction": containment,
            "presimulation_target_containment_threshold": self.threshold,
        }
        if not np.isfinite(containment):
            return PlacementFilterDecision.reject(
                "nonfinite_target_containment", **diagnostics
            )
        if containment < self.threshold:
            return PlacementFilterDecision.reject(
                "stone_outside_target", **diagnostics
            )
        return PlacementFilterDecision.accept(**diagnostics)
