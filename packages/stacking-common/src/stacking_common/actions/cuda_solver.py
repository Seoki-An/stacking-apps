"""Batched CUDA continuation; failed candidates are rejected without CPU retries."""

import numpy as np

from .core_solver import CorePlacementPoseSolver
from .types import PlacementSolveResult


class CudaPlacementPoseSolver(CorePlacementPoseSolver):
    """Own one reusable GPU workspace; calls are serialized by action generation."""

    # Fit a full 256-node excavator generation batch in one GPU chunk: each
    # call costs nearly its trust-region iteration count regardless of size, so
    # splitting a batch repeats that cost. The Small workspace reserves about
    # 727 MiB per 128 jobs. Fixed-proposal tests preserve results across
    # 32/64/128/256-job capacities.
    batch_size = 256

    def __init__(self, inventory, profile, *, core_module=None):
        if core_module is None:
            import stacking_core as core_module
        capabilities = getattr(core_module, "cuda_capabilities", lambda: {})()
        if not capabilities.get("posegen_solve") or not capabilities.get("devices"):
            raise RuntimeError(
                "posegen backend 'cuda' requires a CUDA-enabled stacking-core build "
                "and a visible CUDA device; install CUDA bindings or set parallel.posegen_backend: cpu in the MCTS profile"
            )
        super().__init__(inventory, profile, core_module=core_module)
        self.cuda_generator = core_module.CudaPoseGenerator(
            core_module.CudaCapacity.small(self.batch_size)
        )

    def solve(self, inventory, state, object_index, initial):
        return self.solve_batch(inventory, ((state, object_index, initial),))[0]

    def solve_batch(self, inventory, requests):
        """Return ordered results for (state, object index, initial placement) jobs."""
        requests = tuple(requests)
        for state, object_index, _ in requests:
            self._validate_request(inventory, state, object_index)
        results = []
        for start in range(0, len(requests), self.batch_size):
            results.extend(self._solve_chunk(inventory, requests[start:start + self.batch_size]))
        return tuple(results)

    def _solve_chunk(self, inventory, requests):
        # Requests retain the states, so identity keys cannot be recycled.
        # Native body inputs are immutable here and live only within this chunk.
        scenes = {}
        for state, _, _ in requests:
            if id(state) not in scenes:
                scenes[id(state)] = self._prepare_problem_scene(state)
        poses = [initial.pose for _, _, initial in requests]
        gaps = [0.0 for _ in requests]
        results = [None for _ in requests]
        stages = [[] for _ in requests]
        active = list(range(len(requests)))
        times = self.profile.schedule["times"]
        for step, time in enumerate(times):
            if not active:
                break
            items = [self.core.CudaPosegenBatchItem(
                index,
                self._problem(requests[index][0], requests[index][1], poses[index],
                              scene_context=scenes[id(requests[index][0])]),
                self._scheduled_config(float(time), gaps[index], final=step == len(times) - 1),
            ) for index in active]
            outputs = self.cuda_generator.solve(items)
            by_id = {out.request_id: out for out in outputs}
            if len(outputs) != len(active) or set(by_id) != set(active):
                raise RuntimeError("CUDA posegen returned mismatched request IDs")
            next_active = []
            for index in active:
                out = by_id[index]
                native = out.result
                status = out.status.name
                stages[index].append(status)
                limited = out.status in (
                    self.core.CudaPosegenStatus.precision_limit,
                    self.core.CudaPosegenStatus.iteration_limit,
                )
                reason = None
                if not limited and out.status != self.core.CudaPosegenStatus.converged:
                    reason = status
                elif native is None:
                    reason = "missing_result"
                elif not native.solver.force_solver.converged:
                    reason = "force_nonconvergence"
                elif not limited and not native.solver.converged:
                    reason = "solver_nonconvergence"
                elif not (np.isfinite(native.optimal_pose).all()
                          and np.isfinite(native.c_gap) and np.isfinite(native.c_feq)):
                    reason = "nonfinite_result"
                if reason is not None:
                    # These are explicitly invalid placeholders, not a usable
                    # original/partial pose. Action generation rejects them
                    # independently of optional convergence/feasibility filters.
                    result = PlacementSolveResult(
                        requests[index][2].pose, 0., 0., converged=False,
                        rejection_reason=f"posegen_{reason}", valid=False,
                        diagnostics=self._diagnostics(stages[index], reason, False),
                    )
                    results[index] = result
                    continue
                poses[index] = native.optimal_pose
                gaps[index] = float(native.c_gap)
                if step == len(times) - 1:
                    results[index] = self._placement_result(
                        native, stages[index], status if limited else None)
                else:
                    next_active.append(index)
            active = next_active
        return results

    def _placement_result(self, result, stages, limited=None):
        values = self._placement_values(result)
        if limited is not None:
            values.update(converged=False, rejection_reason=f"posegen_{limited}")
        values["diagnostics"].update(self._diagnostics(stages, None, True))
        # Take ownership once, after all CUDA metadata and status are final.
        return PlacementSolveResult(**values)

    @staticmethod
    def _diagnostics(stages, failure_reason, valid):
        return {
            "posegen_backend": "cuda",
            "posegen_result_backend": "cuda",
            "posegen_cuda_stage_statuses": tuple(stages),
            "posegen_cpu_retry": False,
            "posegen_cpu_retry_reason": None,
            "posegen_cuda_failure_reason": failure_reason,
            "posegen_result_valid": valid,
        }
