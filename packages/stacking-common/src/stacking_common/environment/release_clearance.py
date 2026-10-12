"""Repair small noisy-release penetrations without changing XY or orientation."""

from dataclasses import dataclass
import time

import numpy as np


# Numerical clearance for the FP32 collision queries, in the repository's SI units.
GAP_TOLERANCE = 1e-5
LIFT_CLEARANCE = 1e-4
MAX_ROUNDS = 12


@dataclass(frozen=True)
class ReleaseTrial:
    pose: np.ndarray
    failure_reason: str | None = None
    lift: float = 0.


class ReleaseClearance:
    """Shared bounded vertical correction, with CPU or CUDA contact queries.

    The lift budget is the sampled perturbation's maximum surface displacement
    plus the existing contact tolerance. This allows numerical penetration at
    the settled pose without moving a noisy trial arbitrarily above its scene.
    A blocked local correction does not prove that all robot paths are blocked.
    """

    def __init__(self, backend, queries):
        self.backend = backend
        self.queries = queries
        self._radii = {}

    def _radius(self, object_id):
        if object_id not in self._radii:
            radius = 0.
            for geometry in self.backend.models[object_id].geometries:
                extent = np.maximum(np.abs(geometry['bounds_min']),
                                    np.abs(geometry['bounds_max']))
                # Rotation preserves the norm of this conservative local bound.
                radius = max(radius, np.linalg.norm(geometry['body_from_geometry'][:3])
                             + np.linalg.norm(extent))
            self._radii[object_id] = radius
        return self._radii[object_id]

    def correct(self, rows, poses, bases):
        started = time.perf_counter()
        corrected = [np.array(p, dtype=float, copy=True) for p in poses]
        reasons = [None] * len(rows)
        budgets = np.zeros(len(rows))
        for i, ((_, action), pose, base) in enumerate(zip(rows, corrected, bases)):
            if not (np.all(np.isfinite(pose)) and np.all(np.isfinite(base))
                    and np.linalg.norm(pose[3:]) > 0 and np.linalg.norm(base[3:]) > 0):
                reasons[i] = 'initial_clearance_nonfinite'
                continue
            a, b = pose[3:] / np.linalg.norm(pose[3:]), base[3:] / np.linalg.norm(base[3:])
            rotation = 2 * self._radius(action.object_id) * np.sqrt(max(0., 1 - float(a @ b)**2))
            budgets[i] = (np.linalg.norm(pose[:3] - base[:3]) + rotation
                          + self.backend.config.contact_gap_tolerance)
        original_z = np.array([p[2] for p in corrected])
        initial_gaps = np.full(len(rows), np.inf)
        final_gaps = np.full(len(rows), np.inf)
        queries = rounds = 0
        for iteration in range(MAX_ROUNDS):
            active = [i for i, reason in enumerate(reasons) if reason is None]
            if not active:
                break
            rounds += 1
            contacts, errors = self.queries.query(
                [rows[i] for i in active], [corrected[i] for i in active])
            queries += len(contacts)
            increments = np.zeros(len(rows))
            for local, reason in errors.items():
                reasons[active[local]] = reason
            for local, gap, dz in contacts:
                i = active[local]
                if not np.isfinite(gap) or not np.isfinite(dz):
                    reasons[i] = 'initial_clearance_query_failure'
                    continue
                if iteration == 0:
                    initial_gaps[i] = min(initial_gaps[i], gap)
                # Reset below before processing each query batch, including ground.
                final_gaps[i] = min(final_gaps[i], gap)
                if gap >= -GAP_TOLERANCE or reasons[i] is not None:
                    continue
                if dz <= 1e-6:
                    reasons[i] = 'initial_clearance_lateral_contact'
                    continue
                lift = (LIFT_CLEARANCE - gap) / dz
                if corrected[i][2] + lift - original_z[i] > budgets[i]:
                    reasons[i] = 'initial_clearance_lift_limit'
                else:
                    increments[i] = max(increments[i], lift)
            for i in active:
                if reasons[i] is not None:
                    increments[i] = 0.
                elif iteration == MAX_ROUNDS - 1 and increments[i] > 0:
                    reasons[i] = 'initial_clearance_iteration_limit'
                    increments[i] = 0.
            if not np.any(increments):
                break
            for i, increment in enumerate(increments):
                corrected[i][2] += increment
            final_gaps[:] = np.inf
        lifts = np.zeros(len(corrected))
        finite_z = np.isfinite(original_z)
        lifts[finite_z] = np.array([p[2] for p in corrected])[finite_z] - original_z[finite_z]
        valid_gaps = [gap for gap, reason in zip(final_gaps, reasons)
                      if reason is None and np.isfinite(gap)]
        self.last = {
            'seconds': time.perf_counter() - started, 'trials': len(rows),
            'rounds': rounds, 'queries': queries,
            'initially_penetrating': int(np.sum(initial_gaps < -GAP_TOLERANCE)),
            'invalid_trials': sum(reason is not None for reason in reasons),
            'max_lift_m': float(np.max(lifts, initial=0.)),
            'final_valid_min_gap_m': min(valid_gaps, default=None),
        }
        return [ReleaseTrial(p, reason, float(lift)) for p, reason, lift in zip(corrected, reasons, lifts)]


def prepare_release_trials(backend, jobs):
    """Return ordered trial groups from (state, action, settled pose) jobs."""
    if not jobs:
        return []
    config = backend.place_stability
    if not config.active:
        return [[] for _ in jobs]
    rows, poses, bases = [], [], []
    for state, action, base in jobs:
        noisy = (list(config.noisy_poses(base, step=len(state.placed_object_indices) + 1,
                                         object_index=action.object_index, object_id=action.object_id))
                 if np.all(np.isfinite(base)) else [base] * config.n_noise)
        rows.extend((state, action) for _ in noisy)
        poses.extend(noisy)
        bases.extend(base for _ in noisy)
    if not rows:
        return [[] for _ in jobs]
    trials = backend._release_clearance_checker().correct(rows, poses, bases)
    return [trials[i:i + config.n_noise] for i in range(0, len(trials), config.n_noise)]
