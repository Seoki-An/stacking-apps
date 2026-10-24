"""Offline snapshots of actual accepted native pose-optimization updates."""

from dataclasses import dataclass

import numpy as np

from stacking_common.actions.core_solver import CorePlacementPoseSolver
from stacking_common.actions.types import PlacementSolveResult


@dataclass(frozen=True)
class PoseOptimizationSample:
    stage: int
    iteration: int
    pose: np.ndarray


@dataclass(frozen=True)
class PoseOptimizationTrace:
    samples: tuple[PoseOptimizationSample, ...]
    result: PlacementSolveResult
    stage_iterations: tuple[int, ...]
    stage_accepted_iterations: tuple[int, ...]


def trace_pose_optimization(inventory, profile, state, object_index, initial, *, max_samples=24):
    """Replay the CPU schedule and sample accepted trust-region iterates.

    The existing native profiler identifies accepted trials but does not expose
    their poses. Recover those poses by replaying deterministic solve prefixes,
    including every preceding schedule stage to preserve native warm starts.
    This is intentionally expensive offline work. Normal search is unaffected.
    No poses are interpolated, and iteration limits are never used as restarts.
    The returned forces belong to the full reference solve's final pose.
    """
    if max_samples < 2:
        raise ValueError("pose trace needs at least two samples")
    solver = CorePlacementPoseSolver(inventory, profile)
    solver._validate_request(inventory, state, object_index)
    pose, gap = initial.pose.copy(), 0.0
    times = profile.schedule["times"]
    results, profiles = [], []
    for stage, time in enumerate(times):
        solver.generator.config = solver._scheduled_config(time, gap, final=stage == len(times) - 1)
        result, measurements = solver.generator.solve_profiled(solver._problem(state, object_index, pose))
        results.append(result)
        profiles.append(measurements)
        pose, gap = result.optimal_pose, result.c_gap

    accepted = [(stage, iteration) for stage, measurements in enumerate(profiles)
                for iteration, evaluation in enumerate(measurements.evaluations)
                if iteration > 0 and evaluation.accepted]
    # Preserve endpoints of every stage as well as evenly sampled accepted updates.
    selected = set(accepted[i] for i in np.linspace(
        0, len(accepted) - 1, min(max_samples - 2, len(accepted)), dtype=int)) if accepted else set()
    selected.update((stage, result.solver.iters) for stage, result in enumerate(results))
    samples = [PoseOptimizationSample(0, 0, initial.pose.copy())]
    for stage, iteration in sorted(selected):
        if iteration == results[stage].solver.iters:
            sampled_pose = results[stage].optimal_pose
        else:
            # Rebuild the same solver history; a fresh stage alone can have a
            # different force-graph warm start and therefore a different path.
            replay = CorePlacementPoseSolver(inventory, profile)
            replay_pose, replay_gap = initial.pose.copy(), 0.0
            for prior_stage in range(stage + 1):
                config = replay._scheduled_config(times[prior_stage], replay_gap,
                                                   final=prior_stage == len(times) - 1)
                if prior_stage == stage:
                    trust = config.trust_region
                    trust.max_iters = iteration
                    config.trust_region = trust
                replay.generator.config = config
                prefix, measurements = replay.generator.solve_profiled(
                    replay._problem(state, object_index, replay_pose))
                replay_pose, replay_gap = prefix.optimal_pose, prefix.c_gap
                expected = profiles[prior_stage].evaluations[:len(measurements.evaluations)]
                if ([e.accepted for e in measurements.evaluations] != [e.accepted for e in expected]
                        or prefix.solver.iters != (iteration if prior_stage == stage
                                                   else results[prior_stage].solver.iters)):
                    raise RuntimeError("native pose prefix did not reproduce the reference solve")
                if prior_stage < stage and not np.allclose(
                        replay_pose, results[prior_stage].optimal_pose, rtol=1e-10, atol=1e-10):
                    raise RuntimeError("native pose stage did not reproduce the reference pose")
            sampled_pose = replay_pose
        samples.append(PoseOptimizationSample(stage + 1, iteration, np.array(sampled_pose, copy=True)))
    return PoseOptimizationTrace(
        tuple(samples), solver._placement_result(results[-1], solver._native_mass_scale(state, object_index)),
        tuple(r.solver.iters for r in results), tuple(r.solver.accepted_iters for r in results),
    )
