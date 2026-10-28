"""Batched resident GPU placement simulation using the platform physics profiles."""

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import replace
import math
import time

import numpy as np

from stacking_common.domain import StackObjectTrajectory

from .core_backend import CorePlacementBackend, _StructuralSimulation
from .robustness import disturbance_scene, disturbance_phase, scene_trial_outcome, robustness_dynamic_entities
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
        self._sample_body_capacity = cap.bodies
        self._compact_robustness_history = hasattr(self.core.CudaSimulationPhase, 'sampled_entities')
        self.cuda_simulator = self.core.CudaSimulator(cap)
        self._robustness_executor = None

    def _new_simulator(self):
        return None

    def _new_settling_simulator(self):
        return None

    def restore(self, state):
        if list(state.object_ids) != self.object_ids:
            raise ValueError('state inventory does not match the backend')

    def close(self):
        if self._robustness_executor is not None:
            self._robustness_executor.shutdown(wait=True)
            self._robustness_executor = None
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
        return self._place_batch(states, actions)

    def place_batch_with_preparation(self, states, actions, prepare_nominal):
        """Notify the caller of nominal poses while robustness remains private."""
        return self._place_batch(states, actions, prepare_nominal)

    def _place_batch(self, states, actions, prepare_nominal=None):
        if len(states) != len(actions):
            raise ValueError('placement batch states and actions must align')
        for state, action in zip(states, actions):
            self._validate_job(state, action)
        results = []
        for start in range(0, len(states), self.batch_size):
            chunk_states = states[start:start + self.batch_size]
            chunk_actions = actions[start:start + self.batch_size]
            if prepare_nominal is None:
                results.extend(self._place_chunk(chunk_states, chunk_actions))
            else:
                results.extend(self._place_chunk(chunk_states, chunk_actions,
                    prepare_nominal=lambda poses, start=start: prepare_nominal(start, poses)))
        return tuple(results)

    def _place_chunk(self, states, actions, *, prepare_nominal=None):
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
        summarize = self._structural_summary if prepare_nominal is not None else self._structural_result
        simulations = {p.request_id: summarize(
            contexts[p.request_id][2], contexts[p.request_id][0], actions[p.request_id],
            settled[p.request_id], out)
            for p, out in zip(problems, outputs) if out.snapshot is not None}
        valid = [(i, simulation) for i, simulation in simulations.items()
                 if self._robustness_eligible(states[i], actions[i], simulation)]
        robustness_started = time.perf_counter()
        jobs = [(states[i], actions[i], simulation.snapshot, simulation.contacts)
                for i, simulation in valid]
        if prepare_nominal is None:
            robustness = self._robustness(jobs)
        else:
            if self._robustness_executor is None:
                self._robustness_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="stacking-robustness")
            future = self._robustness_executor.submit(self._robustness, jobs)
            try:
                poses = [None] * len(states)
                for i, simulation in simulations.items():
                    poses[i] = {self.object_ids[index]: simulation.snapshot.body_pose(
                        self.scene_builder.entity_id(index)) for index in contexts[i][0]}
                prepare_nominal(poses)
                for problem, out in zip(problems, outputs):
                    i = problem.request_id
                    if i in simulations:
                        self._populate_trajectories(contexts[i][2], contexts[i][0], actions[i],
                                                    settled[i], out, simulations[i])
                robustness = future.result()
            finally:
                # No native work may outlive its inputs, including on CPU errors.
                wait((future,))
        robustness_wall = time.perf_counter() - robustness_started
        robust_by_id = {i: value for (i, _), value in zip(valid, robustness)}
        for problem, out in zip(problems, outputs):
            i = problem.request_id
            state, action = states[i], actions[i]
            if out.snapshot is None:
                results[i] = self._failure(state, action, out, 'structural')
                continue
            indices, frozen, scene = contexts[i]
            simulation = simulations[i]
            result = self._finish_placement(
                state, action, simulation, frozen,
                settling_wall / len(states), structural_wall / len(problems), started,
                robustness=robust_by_id.get(i, (0., [], [])),
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
        simulation = self._structural_summary(scene, indices, action, settling, out)
        self._populate_trajectories(scene, indices, action, settling, out, simulation)
        return simulation

    def _structural_summary(self, scene, indices, action, settling, out):
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
        last = out.phases[-1]
        integrals = {}
        for i in indices:
            entity, object_id = self.scene_builder.entity_id(i), self.object_ids[i]
            integrals[object_id] = last.motion_integrals.get(entity, 0.)
        config = self.config.short_gate if self.config.short_gate and len(out.phases) == 1 else self.config.structural
        required_quiet = (config.final_extra_steps if len(indices) == len(self.object_ids)
                          and config.final_extra_steps is not None else config.extra_steps)
        is_settled = last.quiet_steps >= required_quiet and elapsed > config.min_time
        simulation = _StructuralSimulation(
            out.snapshot, {}, is_settled,
            out.contacts, out.status == self.core.CudaSimulationStatus.motion_limit,
            integrals, elapsed, last.quiet_steps, times, config.velocity_integral_threshold,
        )
        return simulation

    def _populate_trajectories(self, scene, indices, action, settling, out, simulation):
        trajectories = {self.object_ids[i]: StackObjectTrajectory(self.object_ids[i]) for i in indices}
        trajectories[action.object_id] = self._settle_trajectory(action, settling)
        for i in indices:
            if i != action.object_index:
                trajectories[self.object_ids[i]].add_pose(scene.body_pose(self.scene_builder.entity_id(i)), time=0.)
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
        for i in indices:
            entity, object_id = self.scene_builder.entity_id(i), self.object_ids[i]
            trajectories[object_id].vel_integral = simulation.velocity_integrals[object_id]
            trajectories[object_id].energy = self._normalized_energy(
                object_id, out.snapshot.body_linear_velocity(entity),
                out.snapshot.body_angular_velocity(entity))
        simulation.trajectories.update(trajectories)

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
        problems, trials = [], []
        displacements = [[float('inf')] * config.n_noise for _ in jobs]
        reasons = [[None] * config.n_noise for _ in jobs]
        for i, (state, action, snapshot, contacts) in enumerate(jobs):
            phase_config = disturbance_phase(self, state)
            dynamic = robustness_dynamic_entities(self, snapshot, action, contacts)
            for number, (linear, angular) in enumerate(config.velocity_disturbances(
                    step=len(state.placed_object_indices) + 1,
                    object_index=action.object_index, object_id=action.object_id)):
                scene = disturbance_scene(self, snapshot, action, linear, angular, dynamic)
                initial = {body.id: np.asarray(body.pose) for body in scene.bodies
                           if body.mobility == self.core.Mobility.DYNAMIC}
                phase = self._robustness_phase(phase_config, initial)
                problems.append(self._problem(len(problems), scene, self.simulation_config, [phase]))
                trials.append((i, number, initial, phase_config))
        # Keep every timestep, but compact-capable native builds store only
        # monitored bodies. Older builds still allocate full-capacity histories.
        # The allowance covers device histories, downloads and host conversion.
        steps = max((p.phases[0].steps + 1 for p in problems), default=1)
        sample_bodies = max((len(p.phases[0].sampled_entities)
                            if self._compact_robustness_history and p.phases[0].sampled_entities
                            else self._sample_body_capacity for p in problems), default=1)
        chunk_size = max(1, min(self.batch_size,
            (1536 * 1024**2) // (steps * sample_bodies * 320)))
        for start in range(0, len(problems), chunk_size):
            outputs = self._run(problems[start:start + chunk_size])
            for context, out in zip(trials[start:start + chunk_size], outputs, strict=True):
                i, number, initial, config_phase = context
                value, reason = self._robustness_outcome(initial, config_phase, out)
                displacements[i][number], reasons[i][number] = value, reason
            del outputs
        return [(max(d, default=0.), d, r) for d, r in zip(displacements, reasons)]

    def _robustness_phase(self, config, entities):
        # The ordinary phase supports all dynamic bodies. The older native
        # CudaRobustnessStopping API only monitors one body, missing support motion.
        phase = self._phase(config, [], final_object=False)
        phase.monitored = list(entities)
        if self._compact_robustness_history and entities:
            phase.sampled_entities = list(entities)
        phase.motion_limit = 0.  # Intentional disturbances use peak excursion instead.
        return phase

    def _robustness_outcome(self, initial, config, out):
        if out.snapshot is None or out.status.name not in ('quiet', 'complete'):
            return float('inf'), 'simulation_cuda_' + out.status.name
        histories = {entity: (out.sample_poses(entity) if hasattr(out, 'sample_poses') else
                     np.array([s.snapshot.body_pose(entity) for s in out.samples]))
                     for entity in initial}
        if not out.phases or any(len(poses) != out.phases[-1].steps for poses in histories.values()):
            return float('inf'), 'incomplete_robustness_history'
        settled = out.phases[-1].quiet_steps >= config.extra_steps
        return scene_trial_outcome(initial, histories, self.place_stability, config,
                                   energy_settled=settled)
