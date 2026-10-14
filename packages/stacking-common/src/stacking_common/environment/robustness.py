"""Placement robustness under velocity disturbances of the settled scene."""

from dataclasses import dataclass, replace
from collections.abc import Mapping

import numpy as np


@dataclass(frozen=True)
class PlaceStabilityConfig:
    enabled: bool = False
    n_noise: int = 0
    linear_velocity_std: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity_std_deg: float = 0.0
    rotation_weight: float = 0.0
    fail_displacement: float | None = None
    seed: int = 0
    reject_failed_trials: bool = False
    settling_position_tolerance: float = 0.0
    settling_rotation_tolerance_deg: float = 0.0

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('place_stability.enabled must be boolean')
        if not isinstance(self.reject_failed_trials, bool):
            raise ValueError('place_stability.reject_failed_trials must be boolean')
        if type(self.n_noise) is not int or self.n_noise < 0:
            raise ValueError('place_stability.n_noise must be a non-negative integer')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('place_stability.seed must be a non-negative integer')
        std = np.asarray(self.linear_velocity_std, dtype=float)
        if std.ndim == 0:
            std = np.repeat(std, 3)
        if std.shape != (3,) or not np.all(np.isfinite(std)) or np.any(std < 0):
            raise ValueError('place_stability.linear_velocity_std must be non-negative XYZ deviations')
        object.__setattr__(self, 'linear_velocity_std', tuple(float(v) for v in std))
        limits = (self.angular_velocity_std_deg, self.rotation_weight,
                  self.settling_position_tolerance, self.settling_rotation_tolerance_deg)
        if self.fail_displacement is not None:
            limits += (self.fail_displacement,)
        for value in limits:
            if value is None or not np.isfinite(value) or value < 0:
                raise ValueError('place_stability limits and deviations must be finite and non-negative')

    @classmethod
    def from_mapping(cls, values):
        if not isinstance(values, Mapping):
            raise TypeError('reward.place_stability must be a mapping')
        if 'position_std' in values or 'rotation_std_deg' in values:
            raise ValueError('pose-noise place_stability settings are obsolete; configure '
                             'linear_velocity_std (m/s) and angular_velocity_std_deg (deg/s)')
        return cls(**values)

    @property
    def active(self):
        return self.enabled and self.n_noise > 0

    def failed(self, displacement, trial_failure_reasons=()):
        if not self.active:
            return False
        # Invalid simulations and excessive excursions always reject.
        if not np.isfinite(displacement) or any(
                (reason or '').startswith('simulation_cuda_')
                or reason in ('excursion_limit', 'nonfinite_simulation', 'incomplete_robustness_history')
                for reason in trial_failure_reasons):
            return True
        return bool(
            (self.reject_failed_trials and any(
                reason not in (None, 'bounded_rocking') for reason in trial_failure_reasons))
            or (self.fail_displacement is not None and displacement > self.fail_displacement)
        )

    def velocity_disturbances(self, *, step, object_index, object_id):
        """Seeded world linear (m/s) and body angular (rad/s) velocity samples."""
        seed = self.seed + 1009 * step + 6361 * object_index + 9176 * object_id
        rng = np.random.default_rng(seed)
        for _ in range(self.n_noise if self.active else 0):
            yield (rng.normal(0., self.linear_velocity_std, size=3),
                   rng.normal(0., np.deg2rad(self.angular_velocity_std_deg), size=3))


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


def pose_window_extent(poses):
    """Conservative translation/rotation diameters; invariant to quaternion sign.

    Bounding-box diagonals bound every pairwise distance without an O(N^2)
    comparison. The quaternion chord bound is converted to an angle in radians.
    """
    poses = np.asarray(poses, dtype=float)
    position = float(np.linalg.norm(np.ptp(poses[:, :3], axis=0)))
    q = poses[:, 3:] / np.linalg.norm(poses[:, 3:], axis=1)[:, None]
    q = np.where((q @ q[0])[:, None] < 0, -q, q)
    angle = min(np.pi, 4 * np.arcsin(min(1., np.linalg.norm(np.ptp(q, axis=0)) / 2)))
    return position, float(angle)


def trial_outcome(initial, poses, config, phase, *, energy_settled):
    """Peak excursion penalty and terminal-window classification of a trial.

    `poses` contains every completed integration step, excluding the initial
    pose. Neither a large excursion followed by a return nor missing history
    can pass as a small final displacement.
    """
    poses = np.asarray(poses, dtype=float)
    if (poses.ndim != 2 or poses.shape[1:] != (7,) or not len(poses)
            or not np.all(np.isfinite(poses))
            or np.any(np.linalg.norm(poses[:, 3:], axis=1) == 0)):
        return float('inf'), 'nonfinite_simulation'
    translation = np.linalg.norm(poses[:, :3] - initial[:3], axis=1)
    q = poses[:, 3:] / np.linalg.norm(poses[:, 3:], axis=1)[:, None]
    reference = initial[3:] / np.linalg.norm(initial[3:])
    angle = 2 * np.arccos(np.clip(np.abs(q @ reference), 0., 1.))
    peak = float(np.max(translation + config.rotation_weight * angle))
    if config.fail_displacement is not None and peak > config.fail_displacement:
        return peak, 'excursion_limit'
    if energy_settled:
        return peak, None
    window = phase.extra_steps + 1
    if (config.settling_position_tolerance > 0 and config.settling_rotation_tolerance_deg > 0
            and len(poses) >= window
            and (len(poses) - phase.extra_steps) * phase.dt > phase.min_time):
        position, angle = pose_window_extent(poses[-window:])
        if (position <= config.settling_position_tolerance
                and angle <= np.deg2rad(config.settling_rotation_tolerance_deg)):
            return peak, 'bounded_rocking'
    return peak, 'not_settled'


def disturbance_scene(backend, snapshot, action, linear, angular):
    """Keep nominal poses/mobility, reset residual motion, kick only the new stone.

    Native linear velocity is in world axes; angular velocity is in body axes.
    No geometric clearance correction or displacement is applied.
    """
    entity = backend.scene_builder.entity_id(action.object_index)
    zero = np.zeros(3)
    bodies = [backend.core.BodyInstance(
        body.id, body.model, body.pose,
        linear=linear if body.id == entity else zero,
        angular=angular if body.id == entity else zero, mobility=body.mobility,
    ) for body in snapshot.bodies]
    return backend.core.SceneSnapshot(snapshot.frame_id, bodies)


def disturbance_phase(backend, state):
    phase = backend.config.structural
    extra = phase.extra_steps
    if len(state.placed_object_indices) + 1 == len(backend.object_ids):
        extra = phase.final_extra_steps or 2 * extra
    return replace(phase, dt=2 * phase.dt, extra_steps=extra)


def scene_trial_outcome(initial, histories, config, phase, *, energy_settled):
    """Worst excursion across dynamic bodies; every body must recover."""
    outcomes = [trial_outcome(pose, histories[entity], config, phase,
                             energy_settled=energy_settled)
                for entity, pose in initial.items()]
    peak = max((value for value, _ in outcomes), default=float('inf'))
    reasons = {reason for _, reason in outcomes}
    for reason in ('nonfinite_simulation', 'excursion_limit', 'not_settled', 'bounded_rocking'):
        if reason in reasons:
            return peak, reason
    return peak, None


def evaluate_place_stability(backend, state, action, snapshot):
    config = backend.place_stability
    if not config.active:
        return 0., [], []
    outcomes = [_simulate_velocity_disturbance(backend, state, action, snapshot, linear, angular)
                for linear, angular in config.velocity_disturbances(
                    step=len(state.placed_object_indices) + 1,
                    object_index=action.object_index, object_id=action.object_id)]
    values, reasons = map(list, zip(*outcomes))
    return max(values), values, reasons


def _simulate_velocity_disturbance(backend, state, action, snapshot, linear, angular):
    scene = disturbance_scene(backend, snapshot, action, linear, angular)
    simulator = backend._new_simulator()
    phase = disturbance_phase(backend, state)
    bodies = [body for body in scene.bodies if body.mobility == backend.core.Mobility.DYNAMIC]
    initial = {body.id: np.asarray(body.pose) for body in bodies}
    histories = {body.id: [] for body in bodies}
    settled_steps = 0
    for step in range(int(np.floor(phase.max_time / phase.dt + 1e-9)) + 1):
        scene = simulator.step(scene, phase.dt).snapshot
        energies = []
        for body in bodies:
            pose = np.asarray(scene.body_pose(body.id), dtype=float)
            v = np.asarray(scene.body_linear_velocity(body.id))
            w = np.asarray(scene.body_angular_velocity(body.id))
            energy = float(v @ v + w @ np.asarray(body.model.inertia) @ w / body.model.mass)
            if not np.all(np.isfinite(pose)) or not np.isfinite(energy):
                return float('inf'), 'nonfinite_simulation'
            histories[body.id].append(pose.copy())
            energies.append(energy)
        quiet = max(energies, default=float('inf')) < phase.energy_threshold
        settled_steps = settled_steps + 1 if quiet and (step + 1) * phase.dt > phase.min_time else 0
        if settled_steps >= phase.extra_steps:
            break
        limit = backend.place_stability.fail_displacement
        if limit is not None and any(pose_displacement(initial[e], h[-1],
                backend.place_stability.rotation_weight) > limit for e, h in histories.items()):
            break
    return scene_trial_outcome(initial, histories, backend.place_stability, phase,
                               energy_settled=settled_steps >= phase.extra_steps)
