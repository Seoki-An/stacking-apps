"""Batched resident GPU placement simulation using the platform physics profiles."""

from dataclasses import replace
import math
import time

import numpy as np

from stacking_common.domain import StackObjectTrajectory
from stacking_common.domain.geometry import quaternion_matrix

from .core_backend import CorePlacementBackend, _StructuralSimulation
from .robustness import pose_displacement
from .transition import PlacementResult


def _steps_after(seconds, dt):
    """First step strictly beyond a platform time bound."""
    return math.floor(seconds / dt + 1e-9) + 1


class CudaPlacementBackend(CorePlacementBackend):
    """One GPU workspace per fidelity; each call starts independent branches."""

    # Each native call costs about its step limit regardless of job count, so
    # fewer, larger chunks win once CPU result work is cheap; 256-placement
    # chunks lose the CPU/GPU overlap. 512 jobs fit four robustness trials per
    # placement in about 2-3 GiB of GPU memory.
    batch_size = 512  # Native capacity: four robustness trials per placement chunk.
    placement_batch_size = 128  # Branches simulated per GPU chunk before CPU results.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        info = getattr(self.core, 'cuda_capabilities', lambda: {})()
        if not info.get('simulation_solve') or not info.get('devices'):
            raise RuntimeError('CUDA mode requires stacking-core GPU simulation and a visible GPU; rebuild stacking-core')
        if not hasattr(self.core.CudaSimulationPhase(), 'max_total_steps'):
            raise RuntimeError('rebuild stacking-core for CUDA continued-phase time limits')
        cap = self.core.CudaCapacity.small(self.batch_size)
        cap.bodies = max(cap.bodies, len(self.models) + 1)
        geometries = 1 + sum(model.geometry_count for model in self.models.values())
        cap.geometries = max(cap.geometries, geometries)
        cap.pairs = max(cap.pairs, geometries * (geometries - 1) // 2)
        self.cuda_simulator = self.core.CudaSimulator(cap)

    def _new_simulator(self):
        return None

    def _new_settling_simulator(self):
        return None

    def _release_clearance_checker(self):
        if self._release_clearance is None:
            from .release_clearance import ReleaseClearance
            from ._cuda_release_clearance import CudaReleaseQueries
            self._release_clearance = ReleaseClearance(self, CudaReleaseQueries(self))
        return self._release_clearance

    def restore(self, state):
        if list(state.object_ids) != self.object_ids:
            raise ValueError('state inventory does not match the backend')

    def close(self):
        self.cuda_simulator = None
        super().close()

    def _validate_job(self, state, action):
        if not self.object_ids:
            raise RuntimeError('placement backend has not been reset')
        self.restore(state)
        if not 0 <= action.object_index < len(self.object_ids):
            raise ValueError('action object index is outside the inventory')
        if action.object_id != self.object_ids[action.object_index]:
            raise ValueError('action object ID does not match the inventory')
        if action.object_index in state.placed_object_indices:
            raise ValueError('action object has already been placed')

    def _problem(self, request_id, scene, config, phases):
        problem = self.core.CudaSimulationProblem()
        problem.request_id, problem.scene = request_id, scene
        problem.config, problem.phases = config, phases
        return problem

    def _run(self, problems):
        outputs = []
        for start in range(0, len(problems), self.batch_size):
            chunk = problems[start:start + self.batch_size]
            results = self.cuda_simulator.run(chunk)
            by_id = {out.job.request_id: out for out in results}
            if len(results) != len(chunk) or set(by_id) != {p.request_id for p in chunk}:
                raise RuntimeError('CUDA simulation returned mismatched request IDs')
            outputs.extend(by_id[p.request_id] for p in chunk)
        return outputs

    def _failure(self, state, action, out, stage):
        # No failed/partial native snapshot is used as a valid placement.
        return PlacementResult(
            object_poses={**state.object_poses, action.object_id: action.pose.copy()},
            failed=True, settled=False,
            diagnostics={'simulation_backend': 'cuda', 'simulation_cuda_stage': stage,
                         'simulation_cuda_status': out.status.name,
                         'rejection_reason': 'simulation_cuda_' + out.status.name},
        )

    def _settling(self, states, actions):
        problems = []
        for i, (state, action) in enumerate(zip(states, actions)):
            indices = [*state.placed_object_indices, action.object_index]
            scene = self._scene(state, action, indices, dynamic_indices={action.object_index})
            phase = self.core.CudaSimulationPhase()
            phase.dt, phase.steps = self.config.settle_dt, self.config.settle_steps
            phase.sample_stride = max(1, phase.steps // self.config.settle_sample_count)
            problems.append(self._problem(i, scene, self.settling_simulation_config or self.simulation_config, [phase]))
        return self._run(problems)

    def _settle_trajectory(self, action, out):
        entity = self.scene_builder.entity_id(action.object_index)
        poses = (np.vstack((action.pose, out.sample_poses(entity)))
                 if hasattr(out, 'sample_poses') else
                 [action.pose.copy(), *(s.snapshot.body_pose(entity) for s in out.samples)])
        trajectory = StackObjectTrajectory(action.object_id)
        trajectory.settle_pose_count = len(poses)
        trajectory.settle_position_delta = float(np.linalg.norm(poses[-1][:3] - poses[0][:3]))
        trajectory.settle_path_length = float(np.linalg.norm(np.diff(np.asarray(poses)[:, :3], axis=0), axis=1).sum())
        trajectory.add_pose(out.snapshot.body_pose(entity), time=0.)
        return trajectory

    def _phase(self, config, indices, *, final_object, continuation=False):
        phase = self.core.CudaSimulationPhase()
        phase.dt = config.dt
        phase.steps = _steps_after(config.max_time, config.dt)
        phase.max_total_steps = phase.steps
        phase.min_steps = _steps_after(config.min_time, config.dt) - 1
        phase.quiet_steps = (config.final_extra_steps if final_object and config.final_extra_steps is not None
                             else config.extra_steps)
        phase.monitored = [self.scene_builder.entity_id(i) for i in indices]
        phase.energy_threshold = config.energy_threshold
        phase.motion_limit = config.velocity_integral_threshold
        phase.continue_previous = continuation
        phase.sample_stride = 1
        return phase

    def place(self, state, action):
        return self.place_batch((state,), (action,))[0]

    def place_batch(self, states, actions):
        if len(states) != len(actions):
            raise ValueError('placement batch states and actions must align')
        for state, action in zip(states, actions):
            self._validate_job(state, action)
        results = []
        for start in range(0, len(states), self.batch_size):
            results.extend(self._place_chunk(states[start:start + self.batch_size], actions[start:start + self.batch_size]))
        return tuple(results)

    def _place_chunk(self, states, actions):
        started = time.perf_counter()
        settled = self._settling(states, actions)
        settling_wall = time.perf_counter() - started
        results, problems, contexts = [None] * len(states), [], {}
        for i, (state, action, out) in enumerate(zip(states, actions, settled)):
            if out.snapshot is None:
                results[i] = self._failure(state, action, out, 'settling')
                continue
            indices = [*state.placed_object_indices, action.object_index]
            frozen = self._short_frozen_indices(out.snapshot, indices, action.object_index, out.contacts)
            scene = self._scene_from_snapshot(out.snapshot, indices, state, action, frozen_indices=frozen)
            phases = [self._phase(config, indices, final_object=len(indices) == len(self.object_ids),
                                  continuation=j > 0) for j, config in enumerate(
                ([self.config.short_gate] if self.config.short_gate else []) + [self.config.structural])]
            problems.append(self._problem(i, scene, self.simulation_config, phases))
            contexts[i] = (indices, frozen, scene)
        structural_started = time.perf_counter()
        outputs = self._run(problems)
        structural_wall = time.perf_counter() - structural_started
        valid = [(p.request_id, out) for p, out in zip(problems, outputs) if out.snapshot is not None]
        robustness_started = time.perf_counter()
        robustness = self._robustness([(states[i], actions[i], settled[i].snapshot) for i, _ in valid])
        robustness_wall = time.perf_counter() - robustness_started
        robust_by_id = {i: value for (i, _), value in zip(valid, robustness)}
        for problem, out in zip(problems, outputs):
            i = problem.request_id
            state, action = states[i], actions[i]
            if out.snapshot is None:
                results[i] = self._failure(state, action, out, 'structural')
                continue
            indices, frozen, scene = contexts[i]
            simulation = self._structural_result(scene, indices, action, settled[i], out)
            result = self._finish_placement(
                state, action, settled[i].snapshot, simulation, frozen,
                settling_wall / len(states), structural_wall / len(problems), started,
                robustness=robust_by_id[i],
            )
            results[i] = replace(result, diagnostics={
                **result.diagnostics,
                'simulation_backend': 'cuda', 'simulation_cuda_status': out.status.name,
                'simulation_cuda_phase_statuses': tuple(p.status.name for p in out.phases),
                'simulation_cuda_unconverged_steps': sum(p.unconverged_steps for p in (*settled[i].phases, *out.phases)),
                'simulation_robustness_wall_seconds': robustness_wall / max(len(valid), 1),
                'simulation_total_wall_seconds': (time.perf_counter() - started) / len(states),
            })
        return results

    def _structural_result(self, scene, indices, action, settling, out):
        trajectories = {self.object_ids[i]: StackObjectTrajectory(self.object_ids[i]) for i in indices}
        trajectories[action.object_id] = self._settle_trajectory(action, settling)
        for i in indices:
            if i != action.object_index:
                trajectories[self.object_ids[i]].add_pose(scene.body_pose(self.scene_builder.entity_id(i)), time=0.)
        elapsed = 0.
        times = {}
        names = ('short', 'long_extension') if self.config.short_gate else ('structural',)
        for j, phase in enumerate(out.phases):
            config = self.config.short_gate if j == 0 and self.config.short_gate else self.config.structural
            times[names[j]] = phase.steps * config.dt
            elapsed += times[names[j]]
        for name in names:
            times.setdefault(name, 0.)
        times['total'] = elapsed
        if hasattr(out, 'sample_poses'):
            sample_times, selected = [0.], []
            for index, sample_time in enumerate(out.sample_times):
                if sample_time > sample_times[-1]:
                    selected.append(index)
                    sample_times.append(float(sample_time))
            for i in indices:
                object_id = self.object_ids[i]
                initial = trajectories[object_id]
                values = out.sample_poses(self.scene_builder.entity_id(i))
                packed = StackObjectTrajectory.from_samples(
                    object_id, np.vstack((initial.poses[0], values[selected])), sample_times)
                packed.settle_pose_count = initial.settle_pose_count
                packed.settle_position_delta = initial.settle_position_delta
                packed.settle_path_length = initial.settle_path_length
                trajectories[object_id] = packed
        else:
            for sample in out.samples:
                for i in indices:
                    trajectory = trajectories[self.object_ids[i]]
                    if sample.time > trajectory.times[-1]:
                        trajectory.add_pose(sample.snapshot.body_pose(self.scene_builder.entity_id(i)), time=sample.time)
        last = out.phases[-1]
        integrals = {}
        for i in indices:
            entity, object_id = self.scene_builder.entity_id(i), self.object_ids[i]
            integral = last.motion_integrals.get(entity, 0.)
            integrals[object_id] = integral
            trajectories[object_id].vel_integral = integral
            linear = out.snapshot.body_linear_velocity(entity)
            angular = out.snapshot.body_angular_velocity(entity)
            trajectories[object_id].energy = self._normalized_energy(object_id, linear, angular)
        # If the short phase failed, the long phase was never entered.
        config = self.config.short_gate if self.config.short_gate and len(out.phases) == 1 else self.config.structural
        required_quiet = (config.final_extra_steps if len(indices) == len(self.object_ids)
                          and config.final_extra_steps is not None else config.extra_steps)
        is_settled = last.quiet_steps >= required_quiet and elapsed > config.min_time
        return _StructuralSimulation(
            out.snapshot, trajectories, is_settled,
            out.contacts, out.status == self.core.CudaSimulationStatus.motion_limit,
            integrals, elapsed, last.quiet_steps, times, config.velocity_integral_threshold,
        )

    def settle_fixed(self, state, action):
        self._validate_job(state, action)
        started = time.perf_counter()
        out, = self._settling((state,), (action,))
        if out.snapshot is None:
            return self._failure(state, action, out, 'settling')
        trajectory = self._settle_trajectory(action, out)
        indices = [*state.placed_object_indices, action.object_index]
        poses = {self.object_ids[i]: out.snapshot.body_pose(self.scene_builder.entity_id(i)) for i in indices}
        return PlacementResult(object_poses=poses, trajectories={action.object_id: trajectory}, diagnostics={
            'simulation_backend': 'cuda', 'simulation_cuda_status': out.status.name,
            'fixed_scene_resettle_steps': self.config.settle_steps,
            'fixed_scene_resettle_bodies': len(state.placed_object_indices),
            'fixed_scene_resettle_finite': True,
            'fixed_scene_resettle_position_delta': trajectory.settle_position_delta,
            'fixed_scene_resettle_path_length': trajectory.settle_path_length,
            'simulation_settling_wall_seconds': time.perf_counter() - started,
            'simulation_structural_wall_seconds': 0.,
            'simulation_total_wall_seconds': time.perf_counter() - started,
        })

    def _robustness(self, jobs):
        config = self.place_stability
        if not config.active:
            return [(0., [], []) for _ in jobs]
        from .release_clearance import prepare_release_trials
        prepared = prepare_release_trials(self, [
            (state, action, snapshot.body_pose(self.scene_builder.entity_id(action.object_index)))
            for state, action, snapshot in jobs])
        problems, trials = [], []
        displacements = [[float('inf')] * len(group) for group in prepared]
        reasons = [[trial.failure_reason for trial in group] for group in prepared]
        for i, ((state, action, _), group) in enumerate(zip(jobs, prepared)):
            indices = [*state.placed_object_indices, action.object_index]
            for number, release in enumerate(group):
                if release.failure_reason is not None:
                    continue
                noisy = release.pose
                trial = action.copy()
                trial.pose = noisy
                scene = self._scene(state, trial, indices, dynamic_indices={action.object_index})
                structural = self.config.structural
                extra = structural.extra_steps
                if len(indices) == len(self.object_ids):
                    extra = structural.final_extra_steps or 2 * extra
                phase_config = replace(structural, dt=2 * structural.dt, extra_steps=extra)
                phase = self._robustness_phase(phase_config, action.object_index)
                native_stop = getattr(phase, 'robustness', None) is not None
                problems.append(self._problem(len(problems), scene, self.simulation_config, [phase]))
                trials.append((i, number, action, noisy, phase_config, native_stop))
        outputs = self._run(problems) if problems else []
        for (i, number, action, noisy, config_phase, native_stop), out in zip(trials, outputs, strict=True):
            stop = self._native_robustness_stop if native_stop else self._robustness_stop
            final, reason, finished = stop(action, noisy, config_phase, out)
            if not finished:
                displacement, reason = float('inf'), 'simulation_cuda_' + out.status.name
            else:
                displacement = pose_displacement(noisy, final, config.rotation_weight)
            displacements[i][number] = displacement
            reasons[i][number] = reason
        return [(max(d, default=0.), d, r) for d, r in zip(displacements, reasons)]

    def _robustness_phase(self, config, object_index):
        phase = self._phase(config, [object_index], final_object=False)
        stopping_type = getattr(self.core, 'CudaRobustnessStopping', None)
        if stopping_type is None:
            # Compatibility with older core installations: retain exact replay.
            phase.quiet_steps = 0
            phase.motion_limit = 0.
        else:
            stopping = stopping_type()
            stopping.min_time, stopping.max_time = config.min_time, config.max_time
            phase.robustness = stopping
            phase.sample_stride = 0
        return phase

    def _native_robustness_stop(self, action, noisy, config, out):
        if out.snapshot is None or not out.phases:
            return noisy, None, False
        phase = out.phases[-1]
        elapsed = phase.steps * config.dt
        integral = phase.max_motion_integral
        finished = (phase.quiet_steps >= config.extra_steps or elapsed > config.max_time
                    or integral > config.velocity_integral_threshold)
        reason = ('target_motion_limit' if integral > config.velocity_integral_threshold else
                  'not_settled' if phase.settled_steps < config.extra_steps or elapsed <= config.min_time else None)
        return out.snapshot.body_pose(self.scene_builder.entity_id(action.object_index)), reason, finished

    def _robustness_stop(self, action, noisy, phase, out):
        entity = self.scene_builder.entity_id(action.object_index)
        quiet = settled_steps = 0
        integral = 0.
        pose = noisy
        elapsed = 0.
        finished = False
        for sample in out.samples:
            pose = sample.snapshot.body_pose(entity)
            linear = sample.snapshot.body_linear_velocity(entity)
            angular = quaternion_matrix(pose[3:]).T @ sample.snapshot.body_angular_velocity(entity)
            energy = self._normalized_energy(action.object_id, linear, angular)
            integral += np.sqrt(max(energy, 0.)) * phase.dt
            elapsed = sample.step * phase.dt
            settled_steps = settled_steps + 1 if energy < phase.energy_threshold and elapsed > phase.min_time else 0
            quiet = quiet + 1 if (energy < phase.energy_threshold or integral > .9 * phase.velocity_integral_threshold) and elapsed > phase.min_time else 0
            if quiet >= phase.extra_steps or elapsed > phase.max_time or integral > phase.velocity_integral_threshold:
                finished = True
                break
        reason = ('target_motion_limit' if integral > phase.velocity_integral_threshold else
                  'not_settled' if settled_steps < phase.extra_steps or elapsed <= phase.min_time else None)
        return pose, reason, finished
