"""Normalize stacking-core-style plan results for common orchestration."""

from __future__ import annotations

from collections.abc import Callable
import time

import numpy as np

from stacking_common.planning import (
    GraspEvent,
    MotionPlanSegment,
    PlacementPlanningProblem,
    PlacementPlanningResult,
    PlanningStatus,
)


class NativeMotionPlannerAdapter:
    """Adapt an injected native platform solver to the common result shape.

    The injected callable owns robot, gripper, phase-scene, and native problem
    construction. This boundary can call ``stacking_core.solve_pick_place`` as
    soon as the planner types are present in its nanobind module.
    """

    def __init__(self, solve: Callable[[PlacementPlanningProblem], object]) -> None:
        if not callable(solve):
            raise TypeError("platform motion planner solve callback must be callable")
        self.solve_native = solve

    @classmethod
    def from_stacking_core(
        cls,
        problem_builder: Callable[[PlacementPlanningProblem, object], object],
        *,
        config: object | None = None,
        core_module=None,
    ) -> "NativeMotionPlannerAdapter":
        """Build the adapter around the nanobind pick-place entry point."""

        if not callable(problem_builder):
            raise TypeError("native planning problem builder must be callable")

        def solve(problem: PlacementPlanningProblem):
            core = core_module
            if core is None:
                try:
                    import stacking_core as core
                except ImportError as error:
                    raise RuntimeError(
                        "stacking-core Python bindings are required for planning"
                    ) from error
            native_solve = getattr(core, "solve_pick_place", None)
            if not callable(native_solve):
                raise RuntimeError(
                    "the stacking-core nanobind module does not expose "
                    "solve_pick_place yet"
                )
            native_problem = problem_builder(problem, core)
            if config is None:
                return native_solve(native_problem)
            return native_solve(native_problem, config)

        return cls(solve)

    def plan(self, problem: PlacementPlanningProblem) -> PlacementPlanningResult:
        if not isinstance(problem, PlacementPlanningProblem):
            raise TypeError("platform motion planner requires a planning problem")
        started = time.perf_counter()
        try:
            native = self.solve_native(problem)
        except (RuntimeError, ValueError) as error:
            return PlacementPlanningResult(
                PlanningStatus.FAILED,
                reason=f"native_planner_error: {error}",
                diagnostics={
                    "exception_type": type(error).__name__,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )

        elapsed = time.perf_counter() - started
        status_name = _enum_name(getattr(native, "status", "invalid_problem"))
        status = {
            "success": PlanningStatus.SUCCESS,
            "infeasible": PlanningStatus.INFEASIBLE,
            "cancelled": PlanningStatus.CANCELLED,
            "max_iters": PlanningStatus.FAILED,
            "invalid_problem": PlanningStatus.FAILED,
        }.get(status_name, PlanningStatus.FAILED)
        failure = getattr(native, "failure", None)
        diagnostics = {
            "native_status": status_name,
            "candidate_count": len(tuple(getattr(native, "candidates", ()))),
            "selected_index": _optional_int(
                getattr(native, "selected_index", None)
            ),
            "elapsed_seconds": elapsed,
        }
        timings = getattr(native, "timings", None)
        if timings is not None:
            diagnostics["native_timings"] = {
                "grasp_generation_seconds": float(
                    getattr(timings, "grasp_generation_seconds", 0.0)
                ),
                "simulation_refinement_seconds": float(
                    getattr(timings, "simulation_refinement_seconds", 0.0)
                ),
                "trajectory_optimization_seconds": float(
                    getattr(timings, "trajectory_optimization_seconds", 0.0)
                ),
                "total_seconds": float(
                    getattr(timings, "total_seconds", 0.0)
                ),
                "grasp_candidates": int(
                    getattr(timings, "grasp_candidates", 0)
                ),
                "refined_candidates": int(
                    getattr(timings, "refined_candidates", 0)
                ),
                "motion_candidates": int(
                    getattr(timings, "motion_candidates", 0)
                ),
            }
        if failure is not None:
            diagnostics.update(
                {
                    "failure_code": str(getattr(failure, "code", "")),
                    "failure_message": str(getattr(failure, "message", "")),
                    "retryable": bool(getattr(failure, "retryable", False)),
                }
            )
        if status is not PlanningStatus.SUCCESS:
            return PlacementPlanningResult(
                status,
                reason=_failure_reason(failure, status_name),
                diagnostics=diagnostics,
            )

        candidate = _selected_candidate(native)
        if candidate is None:
            return PlacementPlanningResult(
                PlanningStatus.FAILED,
                reason="native_success_without_selected_candidate",
                diagnostics=diagnostics,
            )
        try:
            segments = tuple(
                _motion_segment(segment) for segment in candidate.segments
            )
            events = tuple(GraspEvent(
                _enum_name(event.event), int(event.segment_index), int(event.sample_index),
                np.asarray(event.grasp.frame_from_grasp, dtype=float), float(event.grasp.opening),
            ) for event in getattr(candidate, "grasp_events", ()))
            from stacking_common.planning.contracts import validate_grasp_events
            validate_grasp_events(events, segments)
        except (AttributeError, TypeError, ValueError) as error:
            return PlacementPlanningResult(
                PlanningStatus.FAILED,
                reason=f"invalid_native_plan: {error}",
                diagnostics=diagnostics,
            )
        if not segments:
            return PlacementPlanningResult(
                PlanningStatus.FAILED,
                reason="native_success_without_motion_segments",
                diagnostics=diagnostics,
            )
        diagnostics.update(
            {
                "grasp_event_count": len(
                    tuple(getattr(candidate, "grasp_events", ()))
                ),
                "candidate_diagnostics": _candidate_diagnostics(candidate),
            }
        )
        achieved_place_pose = _achieved_place_pose(candidate)
        if achieved_place_pose is not None:
            diagnostics["achieved_place_pose"] = achieved_place_pose
        return PlacementPlanningResult(
            PlanningStatus.SUCCESS,
            segments=segments,
            grasp_events=events,
            score=float(getattr(candidate, "score", 0.0)),
            diagnostics=diagnostics,
        )


def _selected_candidate(native):
    selected = getattr(native, "selected_candidate", None)
    if callable(selected):
        return selected()
    candidates = tuple(getattr(native, "candidates", ()))
    index = getattr(native, "selected_index", None)
    if index is None:
        return None
    index = int(index)
    return candidates[index] if 0 <= index < len(candidates) else None


def _achieved_place_pose(candidate) -> np.ndarray | None:
    for segment in reversed(tuple(getattr(candidate, "segments", ()))):
        samples = tuple(getattr(getattr(segment, "trajectory", None), "samples", ()))
        for sample in reversed(samples):
            pose = getattr(sample, "frame_from_target", None)
            if pose is None:
                continue
            value = np.asarray(pose, dtype=float)
            if value.shape == (7,) and np.all(np.isfinite(value)):
                return value.copy()
    return None


def _motion_segment(native) -> MotionPlanSegment:
    trajectory = getattr(native, "trajectory")
    samples = tuple(getattr(trajectory, "samples"))
    if not samples:
        raise ValueError("native motion segment has no trajectory samples")
    positions = np.vstack(
        [np.asarray(sample.robot.positions, dtype=float) for sample in samples]
    )
    openings = np.asarray(
        [float(sample.robot.gripper_opening) for sample in samples],
        dtype=float,
    )
    name = _enum_name(getattr(native, "stage", "motion"))
    phases = {"pick_approach": "pick", "pick_retreat": "pick",
              "transfer": "transfer", "handoff_place": "handoff",
              "handoff_retreat": "handoff", "handoff_approach": "handoff",
              "handoff_pick": "handoff", "place_approach": "place",
              "place_retreat": "place"}
    poses = tuple(getattr(sample, "frame_from_target", None) for sample in samples)
    return MotionPlanSegment(
        name=name,
        positions=positions,
        openings=openings,
        mode=None if getattr(native, "mode", None) is None else _enum_name(native.mode),
        target_poses=poses if any(pose is not None for pose in poses) else None,
        scene_phase=phases.get(name),
        times=getattr(trajectory, "times", None),
    )


def _failure_reason(failure, fallback: str) -> str:
    if failure is None:
        return fallback
    code = str(getattr(failure, "code", "")).strip()
    message = str(getattr(failure, "message", "")).strip()
    if code and message:
        return f"{code}: {message}"
    return code or message or fallback


def _candidate_diagnostics(candidate) -> list[dict[str, object]]:
    values = []
    for item in tuple(getattr(candidate, "diagnostics", ())):
        failure = getattr(item, "failure", None)
        values.append(
            {
                "stage": (
                    None
                    if getattr(item, "stage", None) is None
                    else _enum_name(item.stage)
                ),
                "code": str(getattr(failure, "code", "")),
                "message": str(getattr(failure, "message", "")),
                "retryable": bool(getattr(failure, "retryable", False)),
            }
        )
    return values


def _optional_int(value) -> int | None:
    return None if value is None else int(value)


def _enum_name(value: object) -> str:
    name = getattr(value, "name", value)
    return str(name).rsplit(".", 1)[-1].strip().lower()
