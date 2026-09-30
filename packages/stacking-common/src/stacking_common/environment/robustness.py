"""Legacy noisy-placement robustness, independent of the physics backend."""

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
from stacking_common.domain.geometry import quaternion_matrix


@dataclass(frozen=True)
class PlaceStabilityConfig:
    enabled: bool = False
    n_noise: int = 0
    position_std: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_std_deg: float = 0.0
    rotation_weight: float = 0.0
    fail_displacement: float | None = None
    seed: int = 0
    reject_failed_trials: bool = False

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('place_stability.enabled must be boolean')
        if not isinstance(self.reject_failed_trials, bool):
            raise ValueError('place_stability.reject_failed_trials must be boolean')
        if type(self.n_noise) is not int or self.n_noise < 0:
            raise ValueError('place_stability.n_noise must be a non-negative integer')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('place_stability.seed must be a non-negative integer')
        std = np.asarray(self.position_std, dtype=float)
        if std.ndim == 0:
            std = np.repeat(std, 3)
        if std.shape != (3,) or not np.all(np.isfinite(std)) or np.any(std < 0):
            raise ValueError('place_stability.position_std must be non-negative XYZ deviations')
        object.__setattr__(self, 'position_std', tuple(float(v) for v in std))
        limits = (self.rotation_std_deg, self.rotation_weight)
        if self.fail_displacement is not None:
            limits += (self.fail_displacement,)
        for value in limits:
            if value is None or not np.isfinite(value) or value < 0:
                raise ValueError('place_stability limits and deviations must be finite and non-negative')

    @classmethod
    def from_mapping(cls, values):
        if not isinstance(values, Mapping):
            raise TypeError('reward.place_stability must be a mapping')
        return cls(**values)

    @property
    def active(self):
        return self.enabled and self.n_noise > 0

    def failed(self, displacement, trial_failure_reasons=()):
        if not self.active:
            return False
        return bool(
            (self.reject_failed_trials and (
                any(reason is not None for reason in trial_failure_reasons)
                or not np.isfinite(displacement)
            ))
            or (self.fail_displacement is not None and (
                not np.isfinite(displacement) or displacement > self.fail_displacement
            ))
        )

    def noisy_poses(self, pose, *, step, object_index, object_id):
        seed = self.seed + 1009 * step + 6361 * object_index + 9176 * object_id
        rng = np.random.default_rng(seed)
        for _ in range(self.n_noise if self.active else 0):
            noisy = np.array(pose, dtype=float, copy=True)
            noisy[:3] += rng.normal(0.0, self.position_std, size=3)
            if self.rotation_std_deg > 0:
                noise = rng.normal(0.0, np.deg2rad(self.rotation_std_deg), size=3)
                angle = np.linalg.norm(noise)
                delta = np.r_[noise * (0.5 * np.sinc(angle / (2 * np.pi))), np.cos(angle / 2)]
                base = noisy[3:] / np.linalg.norm(noisy[3:])
                noisy[3:] = np.r_[delta[3] * base[:3] + base[3] * delta[:3]
                                  + np.cross(delta[:3], base[:3]),
                                  delta[3] * base[3] - delta[:3] @ base[:3]]
                noisy[3:] /= np.linalg.norm(noisy[3:])
            yield noisy


def pose_displacement(initial, final, rotation_weight):
    """Metres of translation plus angular distance weighted in metres/radian."""
    if not np.all(np.isfinite(final)):
        return float('inf')
    distance = float(np.linalg.norm(final[:3] - initial[:3]))
    if rotation_weight > 0:
        a = initial[3:] / np.linalg.norm(initial[3:])
        b = final[3:] / np.linalg.norm(final[3:])
        distance += rotation_weight * 2 * np.arccos(np.clip(abs(a @ b), 0.0, 1.0))
    return float(distance)


def evaluate_place_stability(backend, state, action, settled_pose):
    config = backend.place_stability
    if not config.active:
        return 0.0, [], []
    if not np.all(np.isfinite(settled_pose)):
        return float('inf'), [float('inf')], ['nonfinite_simulation']
    displacements = []
    failure_reasons = []
    for noisy in config.noisy_poses(
        settled_pose, step=len(state.placed_object_indices) + 1,
        object_index=action.object_index, object_id=action.object_id,
    ):
        final, reason = _simulate_noisy_place(backend, state, action, noisy)
        displacements.append(pose_displacement(noisy, final, config.rotation_weight))
        failure_reasons.append(reason)
    # Preserve raw failures; reward clipping must not hide failed trials.
    return max(displacements, default=0.0), displacements, failure_reasons


def _simulate_noisy_place(backend, state, action, noisy_pose):
    trial = action.copy()
    trial.pose = noisy_pose.copy()
    indices = [*state.placed_object_indices, action.object_index]
    scene = backend._scene(state, trial, indices, dynamic_indices={action.object_index})
    simulator = backend._new_simulator()
    phase = backend.config.structural
    extra_steps = phase.extra_steps
    if len(indices) == len(backend.object_ids):
        extra_steps = phase.final_extra_steps or 2 * extra_steps
    dt = 2 * phase.dt
    elapsed, quiet, integral = 0.0, 0, 0.0
    settled_steps = 0
    pose = noisy_pose.copy()
    entity = backend.scene_builder.entity_id(action.object_index)
    model = backend.models[action.object_id]
    while quiet < extra_steps:
        scene = simulator.step(scene, dt).snapshot
        elapsed += dt
        pose = np.asarray(scene.body_pose(entity), dtype=float)
        if not np.all(np.isfinite(pose)):
            return np.full(7, np.inf), 'nonfinite_simulation'
        linear = np.asarray(scene.body_linear_velocity(entity), dtype=float)
        angular = quaternion_matrix(pose[3:]).T @ np.asarray(
            scene.body_angular_velocity(entity), dtype=float,
        )
        energy = float(linear @ linear + angular @ np.asarray(model.inertia) @ angular / model.mass)
        if not np.isfinite(energy):
            return np.full(7, np.inf), 'nonfinite_simulation'
        integral += np.sqrt(max(energy, 0.0)) * dt
        # Keep legacy stopping behavior for the score, but count genuine rest
        # separately: its near-motion-limit shortcut does not prove settling.
        if energy < phase.energy_threshold and elapsed > phase.min_time:
            settled_steps += 1
        else:
            settled_steps = 0
        if (energy < phase.energy_threshold or integral > .9 * phase.velocity_integral_threshold) and elapsed > phase.min_time:
            quiet += 1
        else:
            quiet = 0
        if elapsed > phase.max_time or integral > phase.velocity_integral_threshold:
            break
    if not np.isfinite(integral):
        reason = 'nonfinite_simulation'
    elif integral > phase.velocity_integral_threshold:
        reason = 'target_motion_limit'
    elif settled_steps < extra_steps or elapsed <= phase.min_time:
        reason = 'not_settled'
    else:
        reason = None
    return pose.copy(), reason
