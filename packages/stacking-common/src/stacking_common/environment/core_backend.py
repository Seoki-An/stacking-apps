"""Placement simulation backed by the stacking-core Python extension."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import time
from typing import Sequence

import numpy as np

from stacking_common.domain import (
    Action,
    ContactPoint,
    StackObjectTrajectory,
    State,
)
from stacking_common.planning import (
    CoreSceneBuilder,
    SceneMobility,
    SceneObjectInstance,
)

from .transition import PlacementResult
from .robustness import PlaceStabilityConfig, evaluate_place_stability


@dataclass
class _StructuralSimulation:
    snapshot: object
    trajectories: dict[int, StackObjectTrajectory]
    settled: bool
    contacts: Sequence[dict]
    unstable: bool
    velocity_integrals: dict[int, float]
    elapsed: float
    quiet_steps: int
    stage_times: dict[str, float]
    velocity_threshold: float


@dataclass(frozen=True)
class SimulationPhaseConfig:
    dt: float
    energy_threshold: float
    velocity_integral_threshold: float
    min_time: float
    max_time: float
    extra_steps: int
    freeze_radius: float | None = None
    final_extra_steps: int | None = None

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.dt,
                self.energy_threshold,
                self.velocity_integral_threshold,
                self.min_time,
                self.max_time,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("simulation phase values must be finite")
        if self.dt <= 0.0:
            raise ValueError("simulation dt must be positive")
        if self.energy_threshold < 0.0:
            raise ValueError("energy threshold must be non-negative")
        if self.velocity_integral_threshold <= 0.0:
            raise ValueError("velocity integral threshold must be positive")
        if self.min_time < 0.0 or self.max_time <= self.min_time:
            raise ValueError("simulation time limits are invalid")
        if not isinstance(self.extra_steps, int) or self.extra_steps <= 0:
            raise ValueError("extra simulation steps must be positive")
        if self.final_extra_steps is not None and (
            not isinstance(self.final_extra_steps, int) or self.final_extra_steps <= 0
        ):
            raise ValueError("final extra simulation steps must be positive")
        if self.freeze_radius is not None and (
            not np.isfinite(self.freeze_radius) or self.freeze_radius <= 0.0
        ):
            raise ValueError("simulation freeze radius must be positive and finite")


@dataclass(frozen=True)
class CorePlacementConfig:
    structural: SimulationPhaseConfig
    settle_dt: float
    settle_steps: int
    contact_gap_tolerance: float
    minimum_support_count: int
    short_gate: SimulationPhaseConfig | None = None
    ground_z: float = 0.0
    ground_friction: float = 1.0
    settle_sample_count: int = 64
    frame_id: int = 1
    support_chain_z_tolerance: float = 0.005

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.settle_dt,
                self.contact_gap_tolerance,
                self.ground_z,
                self.ground_friction,
                self.support_chain_z_tolerance,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("placement configuration values must be finite")
        if self.settle_dt <= 0.0:
            raise ValueError("settling dt must be positive")
        if self.contact_gap_tolerance < 0.0:
            raise ValueError("contact gap tolerance must be non-negative")
        if (
            not isinstance(self.minimum_support_count, int)
            or self.minimum_support_count <= 0
        ):
            raise ValueError("minimum support count must be positive")
        if not isinstance(self.settle_steps, int) or self.settle_steps < 0:
            raise ValueError("settling step count must be non-negative")
        if (
            not isinstance(self.settle_sample_count, int)
            or self.settle_sample_count <= 0
        ):
            raise ValueError("settling sample count must be positive")
        if self.ground_friction < 0.0:
            raise ValueError("ground friction must be non-negative")
        if self.support_chain_z_tolerance < 0.0:
            raise ValueError("support-chain Z tolerance must be non-negative")
        if not isinstance(self.frame_id, int) or self.frame_id < 0:
            raise ValueError("frame ID must be a non-negative integer")


@dataclass(frozen=True)
class CorePhysicsProfile:
    """Native physics settings, independently selectable for each phase."""

    gravity: np.ndarray
    contact_model: str
    detection_margin: float
    solver_tol_abs: float
    solver_tol_rel: float
    damping: float = field(default=0.0, kw_only=True)
    linear_damping: float | None = field(default=None, kw_only=True,
        metadata={"omit_default_from_context": True})
    angular_damping: float | None = field(default=None, kw_only=True,
        metadata={"omit_default_from_context": True})
    error_reduction_ratio: float | None = field(default=None, kw_only=True)
    patch_eps: float | None = field(default=None, kw_only=True)
    solver_constraint_scale_reference: float | None = field(default=None, kw_only=True)
    solver_beta_init: float | None = field(default=None, kw_only=True)
    solver_beta_update_interval: int | None = field(default=None, kw_only=True)
    solver_max_iters: int | None = field(default=None, kw_only=True)
    solver_stagnation_window: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        gravity = np.asarray(self.gravity, dtype=float)
        if not np.isfinite(self.damping) or self.damping < 0.0:
            raise ValueError("damping must be finite and non-negative")
        for name in ("linear_damping", "angular_damping"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError("simulation gravity must be a finite XYZ vector")
        if self.contact_model not in {"single_point", "limit_surface_4d", "limit_surface_6d"}:
            raise ValueError("unsupported simulation contact model")
        if (
            not np.isfinite(self.detection_margin)
            or self.detection_margin < 0.0
        ):
            raise ValueError("contact detection margin must be non-negative")
        for tol in (self.solver_tol_abs, self.solver_tol_rel):
            if not np.isfinite(tol) or tol < 0.0:
                raise ValueError(
                    "simulation solver tolerances must be finite and non-negative"
                )
        for name in ("error_reduction_ratio", "patch_eps", "solver_constraint_scale_reference", "solver_beta_init"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.solver_beta_init == 0.0:
            raise ValueError("solver_beta_init must be positive")
        if self.solver_constraint_scale_reference == 0.0:
            raise ValueError("solver_constraint_scale_reference must be positive")
        if self.error_reduction_ratio is not None and self.error_reduction_ratio > 1.0:
            raise ValueError("error_reduction_ratio must be at most one")
        for name in ("solver_beta_update_interval", "solver_max_iters"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        window = self.solver_stagnation_window
        if window is not None and (
            isinstance(window, bool) or not isinstance(window, int) or window < 0
        ):
            raise ValueError("solver_stagnation_window must be a non-negative integer")
        gravity = gravity.copy()
        gravity.setflags(write=False)
        object.__setattr__(self, "gravity", gravity)

    def native_config(self, core_module=None):
        """Build a fresh native configuration without sharing mutable settings."""
        core = core_module
        if core is None:
            import stacking_core as core
        config = core.SimulationConfig()
        config.gravity = self.gravity.copy()
        if hasattr(config, "damping"):
            config.damping = self.damping
        elif self.damping != 0.0:
            raise RuntimeError("rebuild stacking-core to enable physical dynamics damping")
        for name in ("linear_damping", "angular_damping"):
            value = getattr(self, name)
            if hasattr(config, name):
                setattr(config, name, value)
            elif value is not None:
                raise RuntimeError("rebuild stacking-core to enable separate linear/angular damping")
        contact = core.ContactConfig()
        try:
            contact.model = getattr(core.ContactModel, self.contact_model.upper())
        except AttributeError as exc:
            raise RuntimeError(
                f"rebuild stacking-core to enable {self.contact_model} contacts"
            ) from exc
        contact.detection_margin = self.detection_margin
        if self.error_reduction_ratio is not None:
            contact.error_reduction_ratio = self.error_reduction_ratio
        if self.patch_eps is not None:
            contact.patch_eps = self.patch_eps
        config.contact = contact
        solver = core.SolverConfig()
        solver.tol_abs = self.solver_tol_abs
        solver.tol_rel = self.solver_tol_rel
        for name in ("constraint_scale_reference", "beta_init", "beta_update_interval", "max_iters", "stagnation_window"):
            value = getattr(self, f"solver_{name}")
            if value is not None:
                setattr(solver, name, value)
        config.solver = solver
        return config


@dataclass(frozen=True)
class CoreSimulationProfile(CorePhysicsProfile):
    """Structural physics, placement policy, and optional settling physics."""

    placement: CorePlacementConfig
    settling_physics: CorePhysicsProfile | None = field(default=None, kw_only=True)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "CoreSimulationProfile":
        """Parse a fully resolved SI-unit simulation profile."""

        if values.get("units") != "si":
            raise ValueError("simulation profile units must be 'si'")
        if "backend" in values:
            raise ValueError("move simulation backend to parallel.simulation_backend in the MCTS profile")
        integration = _profile_mapping(values, "integration")
        if "physics" in values or "phases" in values:
            # Do not silently ignore settings left behind during migration.
            legacy_keys = {
                "solver", "contact", "settling", "structural", "short_structural"
            }
            if legacy_keys.intersection(values):
                raise ValueError("simulation profile cannot mix grouped and flat settings")
            physics = _profile_mapping(values, "physics")
            structural_physics = _profile_mapping(physics, "structural")
            settling_settings = _profile_mapping(physics, "settling")
            phases = _profile_mapping(values, "phases")
        else:
            # Keep existing platform profiles readable without changing physics.
            structural_physics = values
            settling_settings = _profile_mapping(values, "settling")
            phases = values
        solver = _profile_mapping(structural_physics, "solver")
        contact = _profile_mapping(structural_physics, "contact")
        settling = _profile_mapping(phases, "settling")
        structural = _profile_mapping(phases, "structural")
        short_structural = phases.get("short_structural")
        if short_structural is not None and not isinstance(
            short_structural, Mapping
        ):
            raise TypeError("simulation profile 'short_structural' must be a mapping")
        validation = _profile_mapping(values, "validation")
        ground = _profile_mapping(values, "ground")
        scene = _profile_mapping(values, "scene")

        placement = CorePlacementConfig(
            structural=_simulation_phase(integration, structural),
            settle_dt=float(_profile_value(settling, "dt")),
            settle_steps=int(_profile_value(settling, "steps")),
            short_gate=(
                None
                if short_structural is None
                else _simulation_phase(integration, short_structural)
            ),
            contact_gap_tolerance=float(
                _profile_value(validation, "contact_gap_tolerance")
            ),
            minimum_support_count=int(
                _profile_value(validation, "minimum_support_count")
            ),
            support_chain_z_tolerance=float(
                validation.get("support_chain_z_tolerance", 0.005)
            ),
            settle_sample_count=int(
                _profile_value(settling, "sample_count")
            ),
            ground_z=float(_profile_value(ground, "z")),
            ground_friction=float(_profile_value(ground, "friction")),
            frame_id=int(_profile_value(scene, "frame_id")),
        )
        settling_physics = None
        if (
            "physics" in values
            or "contact" in settling_settings
            or "solver" in settling_settings
        ):
            settling_physics = CorePhysicsProfile(**_physics_values(
                integration, _profile_mapping(settling_settings, "contact"),
                _profile_mapping(settling_settings, "solver"),
                damping=settling_settings.get("damping"),
            ))
        return cls(
            **_physics_values(integration, contact, solver, damping=structural_physics.get("damping")),
            placement=placement, settling_physics=settling_physics,
        )

    def settling_native_config(self, core_module=None):
        return (self.settling_physics or self).native_config(core_module)


def _physics_values(integration, contact, solver, *, damping=None):
    if "damping" in solver:
        raise ValueError(
            "solver.damping has been renamed to solver.constraint_scale_reference; "
            "integration.damping is unchanged"
        )
    values = dict(
        gravity=np.asarray(_profile_value(integration, "gravity"), dtype=float),
        damping=float(integration.get("damping", 0.0)),
        contact_model=str(_profile_value(contact, "model")),
        detection_margin=float(_profile_value(contact, "detection_margin")),
        solver_tol_abs=float(_profile_value(solver, "tol_abs")),
        solver_tol_rel=float(_profile_value(solver, "tol_rel")),
    )
    if damping is not None:
        if not isinstance(damping, Mapping) or set(damping) != {"linear", "angular"}:
            raise ValueError("physics damping requires both linear and angular rates in s^-1")
        values.update(linear_damping=float(damping["linear"]),
                      angular_damping=float(damping["angular"]))
    for name in ("error_reduction_ratio", "patch_eps"):
        if name in contact:
            values[name] = float(contact[name])
    for name in ("constraint_scale_reference", "beta_init", "beta_update_interval", "max_iters", "stagnation_window"):
        if name in solver:
            # Preserve integer type validation instead of truncating fractions.
            values[f"solver_{name}"] = solver[name]
    return values


class CorePlacementBackend:
    """Settle and structurally evaluate placements with stacking-core."""

    _ground_entity_id = 0
    early_quiet_stop = False

    @classmethod
    def from_profile(
        cls,
        models: Mapping[int, object],
        profile: CoreSimulationProfile,
        *,
        place_stability: PlaceStabilityConfig | None = None,
        settling_pose_filter=None,
        core_module=None,
    ) -> "CorePlacementBackend":
        core = core_module
        if core is None:
            try:
                import stacking_core as core
            except ImportError as error:
                raise RuntimeError(
                    "stacking-core Python bindings are required by "
                    "CorePlacementBackend"
                ) from error
        return cls(
            models,
            profile.placement,
            simulation_config=profile.native_config(core),
            settling_simulation_config=profile.settling_native_config(core),
            place_stability=place_stability,
            settling_pose_filter=settling_pose_filter,
            core_module=core,
        )

    def __init__(
        self,
        models: Mapping[int, object],
        config: CorePlacementConfig,
        simulation_config=None,
        *,
        settling_simulation_config=None,
        place_stability: PlaceStabilityConfig | None = None,
        settling_pose_filter=None,
        core_module=None,
    ) -> None:
        if core_module is None:
            try:
                import stacking_core as core_module
            except ImportError as error:
                raise RuntimeError(
                    "stacking-core Python bindings are required by "
                    "CorePlacementBackend"
                ) from error

        self.core = core_module
        self.models = {int(key): value for key, value in models.items()}
        self.config = config
        self.place_stability = place_stability or PlaceStabilityConfig()
        self.settling_pose_filter = settling_pose_filter
        self.simulation_config = simulation_config
        self.settling_simulation_config = settling_simulation_config
        self.object_ids: list[int] = []
        self.scene_builder = CoreSceneBuilder(
            self.models,
            frame_id=config.frame_id,
            core_module=self.core,
        )
        self.simulator = self._new_simulator()
        self.settling_simulator = self._new_settling_simulator()
        self.ground_model = self.core.BodyModel.plane(
            0, friction=config.ground_friction, geometry_id=0
        )

    def reset(self, object_ids: Sequence[int]) -> None:
        resolved = [int(object_id) for object_id in object_ids]
        missing = [
            object_id for object_id in resolved if object_id not in self.models
        ]
        if missing:
            raise ValueError(f"missing stacking-core body models: {missing}")
        self.object_ids = resolved
        self.simulator = self._new_simulator()
        self.settling_simulator = self._new_settling_simulator()

    def place(self, state: State, action: Action) -> PlacementResult:
        if not self.object_ids:
            raise RuntimeError("placement backend has not been reset")
        if action.object_id != self.object_ids[action.object_index]:
            raise ValueError("action object ID does not match the inventory")

        placement_started = time.perf_counter()
        indices = [*state.placed_object_indices, action.object_index]
        self.settling_simulator.clear_warm_start()
        settling_scene = self._scene(
            state,
            action,
            indices,
            dynamic_indices={action.object_index},
        )
        settling_started = time.perf_counter()
        settling_scene, settle_trajectory, settle_contacts = self._settle(
            settling_scene, action
        )
        settling_wall_seconds = time.perf_counter() - settling_started

        rejected = self._settling_rejection(
            state, action, settling_scene, settle_trajectory,
            settling_wall_seconds, placement_started,
        )
        if rejected is not None:
            return rejected

        self.simulator.clear_warm_start()
        frozen_indices = self._structural_frozen_indices(
            settling_scene,
            indices,
            action.object_index,
            settle_contacts,
        )
        structural_scene = self._scene_from_snapshot(
            settling_scene,
            indices,
            state,
            action,
            frozen_indices=frozen_indices,
        )
        structural_started = time.perf_counter()
        structural_result = self._simulate_structural(
            structural_scene,
            indices,
            action,
            settle_trajectory,
        )
        structural_wall_seconds = time.perf_counter() - structural_started

        return self._finish_placement(
            state, action, structural_result, frozen_indices,
            settling_wall_seconds, structural_wall_seconds, placement_started,
        )

    def _settling_rejection(
        self, state, action, snapshot, trajectory, settling_seconds, started,
    ):
        """Discard refined poses outside the target before structural simulation."""
        if self.settling_pose_filter is None:
            return None
        pose = np.asarray(
            snapshot.body_pose(self.scene_builder.entity_id(action.object_index)),
            dtype=float,
        )
        decision = self.settling_pose_filter(action, pose)
        if decision.accepted:
            return None
        return PlacementResult(
            object_poses={**state.object_poses, action.object_id: pose},
            trajectories={action.object_id: trajectory},
            failed=True, settled=False,
            diagnostics={
                **decision.diagnostics,
                "rejection_reason": decision.reason,
                "simulation_rejection_stage": "settling",
                "simulation_settling_wall_seconds": settling_seconds,
                "simulation_structural_wall_seconds": 0.,
                "simulation_robustness_wall_seconds": 0.,
                "simulation_total_wall_seconds": time.perf_counter() - started,
            },
        )

    def _finish_placement(
        self, state, action, structural_result, frozen_indices,
        settling_wall_seconds, structural_wall_seconds, placement_started,
        robustness=None,
    ):
        indices = [*state.placed_object_indices, action.object_index]
        snapshot = structural_result.snapshot
        trajectories = structural_result.trajectories
        poses = {
            self.object_ids[index]: np.asarray(
                snapshot.body_pose(self.scene_builder.entity_id(index)), dtype=float
            )
            for index in indices
        }
        finite = all(np.all(np.isfinite(pose)) for pose in poses.values())
        contacts = self._contact_points(structural_result.contacts)
        diagnostics = self._simulation_diagnostics(
            state,
            action,
            structural_result,
            contacts,
            poses,
        )
        diagnostics.update(
            {
                "simulation_settling_wall_seconds": settling_wall_seconds,
                "simulation_structural_wall_seconds": structural_wall_seconds,
                "simulation_total_wall_seconds": (
                    time.perf_counter() - placement_started
                ),
                "simulation_frozen_object_indices": tuple(
                    sorted(frozen_indices)
                ),
                "simulation_frozen_object_count": len(frozen_indices),
            }
        )
        rejection_reason = self._physical_rejection_reason(
            state,
            action,
            structural_result,
            contacts,
            finite,
            diagnostics,
        )
        robustness_started = time.perf_counter()
        if robustness is None:
            robustness = (evaluate_place_stability(self, state, action, snapshot, structural_result.contacts)
                          if rejection_reason is None and structural_result.settled
                          else (0., [], []))
        displacement, trials, trial_failure_reasons = robustness
        robustness_failed = self.place_stability.failed(displacement, trial_failure_reasons)
        if robustness_failed and rejection_reason is None:
            rejection_reason = 'place_robustness'
        diagnostics.update({
            'place_robustness_enabled': self.place_stability.active,
            'place_robustness_model': ('force_disturbance' if self.place_stability.force_ratio is not None
                                       else 'velocity_disturbance'),
            'place_robustness_contact_hops': self.place_stability.contact_hops,
            'place_robustness_evaluated': bool(trials),
            'place_robustness_trial_displacements': trials,
            'place_robustness_trial_failure_reasons': [
                None if reason == 'bounded_rocking' else reason for reason in trial_failure_reasons],
            'place_robustness_trial_outcomes': [reason or 'settled' for reason in trial_failure_reasons],
            'place_robustness_raw_displacement': displacement,
            'place_robustness_failed': robustness_failed,
            'simulation_robustness_wall_seconds': time.perf_counter() - robustness_started,
            'simulation_total_wall_seconds': time.perf_counter() - placement_started,
        })
        if self.place_stability.force_ratio is not None:
            diagnostics['place_robustness_force_ratio'] = self.place_stability.force_ratio
            diagnostics['place_robustness_force_duration'] = self.place_stability.force_duration
        if rejection_reason is not None:
            diagnostics["rejection_reason"] = rejection_reason
        return PlacementResult(
            object_poses=poses,
            trajectories=trajectories,
            contact_points=contacts,
            failed=bool(rejection_reason is not None),
            settled=bool(finite and structural_result.settled),
            place_robustness_displacement=displacement,
            diagnostics=diagnostics,
        )

    def settle_fixed(self, state: State, action: Action) -> PlacementResult:
        """Settle one target while preserving every existing placed pose."""

        if not self.object_ids:
            raise RuntimeError("placement backend has not been reset")
        if action.object_id != self.object_ids[action.object_index]:
            raise ValueError("action object ID does not match the inventory")
        indices = [*state.placed_object_indices, action.object_index]
        self.settling_simulator.clear_warm_start()
        scene = self._scene(
            state,
            action,
            indices,
            dynamic_indices={action.object_index},
        )
        settling_started = time.perf_counter()
        scene, trajectory, _ = self._settle(scene, action)
        settling_wall_seconds = time.perf_counter() - settling_started
        poses = {
            self.object_ids[index]: np.asarray(
                scene.body_pose(self.scene_builder.entity_id(index)), dtype=float
            )
            for index in indices
        }
        finite = all(np.all(np.isfinite(pose)) for pose in poses.values())
        diagnostics = {
            "fixed_scene_resettle_steps": self.config.settle_steps,
            "fixed_scene_resettle_bodies": len(state.placed_object_indices),
            "fixed_scene_resettle_finite": finite,
            "fixed_scene_resettle_position_delta": (
                trajectory.settle_position_delta
            ),
            "fixed_scene_resettle_path_length": trajectory.settle_path_length,
            "simulation_settling_wall_seconds": settling_wall_seconds,
            "simulation_structural_wall_seconds": 0.0,
            "simulation_total_wall_seconds": settling_wall_seconds,
        }
        if not finite:
            diagnostics["rejection_reason"] = "nonfinite"
        return PlacementResult(
            object_poses=poses,
            trajectories={action.object_id: trajectory},
            failed=not finite,
            settled=finite,
            diagnostics=diagnostics,
        )

    def restore(self, state: State) -> None:
        if [int(value) for value in state.object_ids] != self.object_ids:
            raise ValueError("state inventory does not match the backend")
        self.simulator.clear_warm_start()
        self.settling_simulator.clear_warm_start()

    def close(self) -> None:
        self.simulator = None
        self.settling_simulator = None

    def _new_settling_simulator(self):
        if self.settling_simulation_config is None:
            return self.simulator
        return self.core.Simulator(self.settling_simulation_config)

    def _new_simulator(self):
        if self.simulation_config is None:
            return self.core.Simulator()
        return self.core.Simulator(self.simulation_config)

    def _scene(
        self,
        state: State,
        action: Action,
        indices: Sequence[int],
        dynamic_indices: set[int],
    ):
        objects = []
        for index in indices:
            object_id = self.object_ids[index]
            object_pose = (
                action.pose
                if index == action.object_index
                else state.object_poses[object_id]
            )
            mobility = (
                SceneMobility.DYNAMIC
                if index in dynamic_indices
                else SceneMobility.STATIC
            )
            objects.append(
                SceneObjectInstance(
                    object_index=index,
                    object_id=object_id,
                    pose=object_pose,
                    mobility=mobility,
                )
            )
        return self.scene_builder.build(objects, [self._ground_body()])

    def _scene_from_snapshot(
        self,
        snapshot,
        indices: Sequence[int],
        state: State,
        action: Action,
        *,
        frozen_indices: set[int] | None = None,
    ):
        frozen = set() if frozen_indices is None else set(frozen_indices)
        objects = []
        for index in indices:
            object_id = self.object_ids[index]
            mobility = (
                SceneMobility.STATIC
                if (
                    index in frozen
                    or (
                        object_id in state.pose_identified_object_ids
                        and index != action.object_index
                    )
                )
                else SceneMobility.DYNAMIC
            )
            objects.append(
                SceneObjectInstance(
                    object_index=index,
                    object_id=object_id,
                    pose=snapshot.body_pose(
                        self.scene_builder.entity_id(index)
                    ),
                    mobility=mobility,
                )
            )
        return self.scene_builder.build(objects, [self._ground_body()])

    def _settle(self, scene, action: Action):
        trajectory = StackObjectTrajectory(action.object_id)
        initial_pose = np.asarray(
            scene.body_pose(
                self.scene_builder.entity_id(action.object_index)
            ),
            dtype=float,
        )
        trajectory.add_pose(initial_pose.copy())
        steps = self.config.settle_steps
        stride = max(1, steps // self.config.settle_sample_count)
        contacts = ()
        for start in range(0, steps, stride):
            count = min(stride, steps - start)
            result = self.settling_simulator.step_n(
                scene, self.config.settle_dt, count
            )
            scene = result.snapshot
            contacts = tuple(result.contacts)
            current_pose = np.asarray(
                scene.body_pose(
                    self.scene_builder.entity_id(action.object_index)
                ),
                dtype=float,
            )
            if np.all(np.isfinite(current_pose)):
                trajectory.add_pose(current_pose.copy())

        poses = np.asarray(trajectory.poses, dtype=float)
        trajectory.settle_pose_count = len(poses)
        if len(poses) >= 2:
            trajectory.settle_position_delta = float(
                np.linalg.norm(poses[-1, :3] - poses[0, :3])
            )
            trajectory.settle_path_length = float(
                np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1).sum()
            )
        if len(poses):
            trajectory.poses = [poses[-1].copy()]
        return scene, trajectory, contacts

    def _structural_frozen_indices(
        self,
        snapshot,
        indices: Sequence[int],
        target_index: int,
        contacts: Sequence[dict],
    ) -> set[int]:
        """Freeze distant stones while retaining the descending support chain."""
        radius = self.config.structural.freeze_radius
        if radius is None:
            return set()

        entity_to_index = {
            self.scene_builder.entity_id(index): index for index in indices
        }
        adjacency: dict[int, set[int]] = {}
        for contact in contacts:
            first = entity_to_index.get(int(contact["entity_first"]))
            second = entity_to_index.get(int(contact["entity_second"]))
            if first is None or second is None:
                continue
            adjacency.setdefault(first, set()).add(second)
            adjacency.setdefault(second, set()).add(first)

        def position(index: int) -> np.ndarray:
            return np.asarray(
                snapshot.body_pose(self.scene_builder.entity_id(index)),
                dtype=float,
            )[:3]

        support_chain = {target_index}
        stack = [target_index]
        while stack:
            current = stack.pop()
            current_z = float(position(current)[2])
            for neighbor in adjacency.get(current, ()):
                if neighbor in support_chain:
                    continue
                if (
                    float(position(neighbor)[2])
                    < current_z + self.config.support_chain_z_tolerance
                ):
                    support_chain.add(neighbor)
                    stack.append(neighbor)

        target_xy = position(target_index)[:2]
        return {
            index
            for index in indices
            if index != target_index
            and index not in support_chain
            and np.linalg.norm(position(index)[:2] - target_xy) > radius
        }

    def _simulate_structural(
        self,
        scene,
        indices: Sequence[int],
        action: Action,
        settle_trajectory: StackObjectTrajectory,
    ) -> _StructuralSimulation:
        trajectories = {
            self.object_ids[index]: StackObjectTrajectory(self.object_ids[index])
            for index in indices
        }
        trajectories[action.object_id] = settle_trajectory
        # Structural replay begins after fixed-scene settling (whose poses
        # are reduced to the final pose); all bodies share the same clock.
        settle_trajectory.times = [0.0] if settle_trajectory.poses else []
        for index in indices:
            object_id = self.object_ids[index]
            if index != action.object_index:
                trajectories[object_id].add_pose(
                    np.asarray(
                        scene.body_pose(
                            self.scene_builder.entity_id(index)
                        ),
                        dtype=float,
                    ),
                    time=0.0,
                )

        velocity_integrals = {self.object_ids[index]: 0.0 for index in indices}
        elapsed = 0.0
        quiet_steps = 0
        contacts = []
        settled = False
        unstable = False
        stage_times: dict[str, float] = {}
        phases = (
            (
                ("short", self.config.short_gate),
                ("long_extension", self.config.structural),
            )
            if self.config.short_gate is not None
            else (("structural", self.config.structural),)
        )
        velocity_threshold = self.config.structural.velocity_integral_threshold
        for stage_index, (stage_name, phase) in enumerate(phases):
            if len(indices) == len(self.object_ids) and phase.final_extra_steps is not None:
                phase = replace(phase, extra_steps=phase.final_extra_steps)
            if stage_index > 0 and _phase_cannot_continue(
                velocity_integrals, elapsed, phase
            ):
                settled = (
                    quiet_steps >= phase.extra_steps
                    and elapsed > phase.min_time
                )
                unstable = True
                stage_times[stage_name] = 0.0
                velocity_threshold = phase.velocity_integral_threshold
                break
            stage_start = elapsed
            (
                scene,
                elapsed,
                quiet_steps,
                contacts,
                settled,
                unstable,
            ) = self._simulate_phase(
                scene,
                indices,
                phase,
                trajectories,
                velocity_integrals,
                elapsed,
                quiet_steps,
                **({'stop_if_quiet_impossible': True}
                   if self.early_quiet_stop and stage_index == len(phases) - 1 else {}),
            )
            stage_times[stage_name] = max(elapsed - stage_start, 0.0)
            velocity_threshold = phase.velocity_integral_threshold
        stage_times["total"] = elapsed
        return _StructuralSimulation(
            snapshot=scene,
            trajectories=trajectories,
            settled=settled,
            contacts=contacts,
            unstable=unstable,
            velocity_integrals=velocity_integrals,
            elapsed=elapsed,
            quiet_steps=quiet_steps,
            stage_times=stage_times,
            velocity_threshold=velocity_threshold,
        )

    def _simulate_phase(
        self,
        scene,
        indices: Sequence[int],
        phase: SimulationPhaseConfig,
        trajectories: dict[int, StackObjectTrajectory],
        velocity_integrals: dict[int, float],
        elapsed: float,
        quiet_steps: int,
        *,
        stop_if_quiet_impossible: bool = False,
    ):
        contacts = []
        # Reproduce the existing floating-point clock, including its last step
        # strictly beyond max_time. Count the most optimistic quiet remainder.
        future_quiet = []
        if stop_if_quiet_impossible:
            future_elapsed = elapsed
            while True:
                future_elapsed += phase.dt
                future_quiet.append(future_elapsed > phase.min_time)
                if future_elapsed > phase.max_time:
                    break
        remaining_quiet = sum(future_quiet)
        completed_steps = 0
        while quiet_steps < phase.extra_steps or elapsed <= phase.min_time:
            result = self.simulator.step(scene, phase.dt)
            scene = result.snapshot
            contacts = result.contacts
            elapsed += phase.dt
            energies = []
            for index in indices:
                object_id = self.object_ids[index]
                entity_id = self.scene_builder.entity_id(index)
                object_pose = np.asarray(scene.body_pose(entity_id), dtype=float)
                linear = np.asarray(
                    scene.body_linear_velocity(entity_id), dtype=float
                )
                angular = np.asarray(
                    scene.body_angular_velocity(entity_id), dtype=float
                )
                energy = self._normalized_energy(object_id, linear, angular)
                energies.append(energy)
                velocity_integrals[object_id] += np.sqrt(energy) * phase.dt
                if np.all(np.isfinite(object_pose)):
                    trajectories[object_id].add_pose(object_pose.copy(), time=elapsed)
                trajectories[object_id].energy = energy
                trajectories[object_id].vel_integral = velocity_integrals[
                    object_id
                ]

            if max(energies, default=0.0) < phase.energy_threshold:
                if elapsed > phase.min_time:
                    quiet_steps += 1
            else:
                quiet_steps = 0
            if stop_if_quiet_impossible:
                remaining_quiet -= future_quiet[completed_steps]
                completed_steps += 1
            if (
                elapsed > phase.max_time
                or max(velocity_integrals.values(), default=0.0)
                > phase.velocity_integral_threshold
            ):
                break
            if stop_if_quiet_impossible and quiet_steps + remaining_quiet < phase.extra_steps:
                break

        settled = quiet_steps >= phase.extra_steps and elapsed > phase.min_time
        unstable = bool(
            not all(np.isfinite(value) for value in velocity_integrals.values())
            or max(velocity_integrals.values(), default=0.0)
            > phase.velocity_integral_threshold
        )
        return scene, elapsed, quiet_steps, contacts, settled, unstable

    def _simulation_diagnostics(
        self,
        state: State,
        action: Action,
        simulation: _StructuralSimulation,
        contacts: Sequence[ContactPoint],
        poses: Mapping[int, np.ndarray],
    ) -> dict[str, object]:
        target_value = simulation.velocity_integrals.get(
            action.object_id, float("nan")
        )
        previous_values = {
            self.object_ids[index]: simulation.velocity_integrals[
                self.object_ids[index]
            ]
            for index in state.placed_object_indices
        }
        previous_displacements = {
            object_id: float(
                np.linalg.norm(poses[object_id][:3] - state.object_poses[object_id][:3])
            )
            for object_id in previous_values
        }
        support = _contact_validation(
            action.object_index,
            contacts,
            placed_count=len(state.placed_object_indices),
            gap_tolerance=self.config.contact_gap_tolerance,
            minimum_support_count=self.config.minimum_support_count,
        )[1]
        return {
            "simulation_stage_times": dict(simulation.stage_times),
            "simulation_elapsed": simulation.elapsed,
            "simulation_settled": simulation.settled,
            "simulation_velocity_integral_threshold": (
                simulation.velocity_threshold
            ),
            "simulation_velocity_integrals": dict(
                simulation.velocity_integrals
            ),
            "simulation_target_velocity_integral": target_value,
            "simulation_previous_velocity_integrals": previous_values,
            "simulation_previous_displacements": previous_displacements,
            "simulation_max_previous_displacement": max(
                previous_displacements.values(), default=0.0
            ),
            "short_gate_used": self.config.short_gate is not None,
            "short_gate_passed": bool(
                self.config.short_gate is None
                or simulation.stage_times.get("long_extension", 0.0) > 0.0
            ),
            **support,
        }

    def _robustness_eligible(self, state, action, simulation):
        if not self.place_stability.active or not simulation.settled:
            return False
        finite = all(np.all(np.isfinite(simulation.snapshot.body_pose(
            self.scene_builder.entity_id(i))))
            for i in [*state.placed_object_indices, action.object_index])
        return self._physical_rejection_reason(
            state, action, simulation, self._contact_points(simulation.contacts), finite, {}) is None

    def _physical_rejection_reason(
        self,
        state: State,
        action: Action,
        simulation: _StructuralSimulation,
        contacts: Sequence[ContactPoint],
        finite: bool,
        diagnostics: dict[str, object],
    ) -> str | None:
        if not finite or any(
            not np.isfinite(value)
            for value in simulation.velocity_integrals.values()
        ):
            return "nonfinite_simulation"
        threshold = simulation.velocity_threshold
        target_value = simulation.velocity_integrals.get(action.object_id, 0.0)
        if target_value > threshold:
            return "target_motion_limit"
        if any(
            simulation.velocity_integrals[self.object_ids[index]] > threshold
            for index in state.placed_object_indices
        ):
            return "scene_motion_limit"
        if simulation.unstable:
            return "simulation_failure"
        contact_reason, contact_diagnostics = _contact_validation(
            action.object_index,
            contacts,
            placed_count=len(state.placed_object_indices),
            gap_tolerance=self.config.contact_gap_tolerance,
            minimum_support_count=self.config.minimum_support_count,
        )
        diagnostics.update(contact_diagnostics)
        return contact_reason

    def _normalized_energy(
        self,
        object_id: int,
        linear: np.ndarray,
        angular: np.ndarray,
    ) -> float:
        model = self.models[object_id]
        mass = float(model.mass)
        inertia = np.asarray(model.inertia, dtype=float)
        value = (
            mass * float(linear @ linear)
            + float(angular @ inertia @ angular)
        ) / mass
        return value if np.isfinite(value) else float("inf")

    def _contact_points(self, contacts: Sequence[dict]) -> list[ContactPoint]:
        entity_to_index = {
            self.scene_builder.entity_id(index): index
            for index in range(len(self.object_ids))
        }
        values = []
        for contact in contacts:
            first = int(contact["entity_first"])
            second = int(contact["entity_second"])
            values.append(
                ContactPoint(
                    entity_id_first=first,
                    entity_id_second=second,
                    object_index_first=entity_to_index.get(first),
                    object_index_second=entity_to_index.get(second),
                    ground_first=first == self._ground_entity_id,
                    ground_second=second == self._ground_entity_id,
                    point_first=contact["point_first"],
                    point_second=contact["point_second"],
                    normal=contact["normal"],
                    gap=contact["gap"],
                )
            )
        return values

    def _ground_pose(self) -> np.ndarray:
        return np.array(
            [0.0, 0.0, self.config.ground_z, 0.0, 0.0, 0.0, 1.0]
        )

    def _ground_body(self):
        return self.core.BodyInstance(
            self._ground_entity_id,
            self.ground_model,
            self._ground_pose(),
            mobility=self.core.Mobility.STATIC,
        )


def _phase_cannot_continue(
    velocity_integrals: Mapping[int, float],
    elapsed: float,
    phase: SimulationPhaseConfig,
) -> bool:
    maximum = max(velocity_integrals.values(), default=0.0)
    return bool(
        not np.isfinite(maximum)
        or maximum > phase.velocity_integral_threshold
        or elapsed >= phase.max_time
    )


def _contact_validation(
    object_index: int,
    contacts: Sequence[ContactPoint],
    *,
    placed_count: int,
    gap_tolerance: float,
    minimum_support_count: int,
) -> tuple[str | None, dict[str, object]]:
    sources: set[int | str] = set()
    ground_gaps: list[float] = []
    scene_gaps: list[float] = []
    for contact in contacts:
        counterpart: int | str | None = None
        if contact.object_index_first == object_index:
            if contact.ground_second:
                counterpart = "ground"
                ground_gaps.append(contact.gap)
            elif contact.object_index_second is not None:
                counterpart = contact.object_index_second
                scene_gaps.append(contact.gap)
        elif contact.object_index_second == object_index:
            if contact.ground_first:
                counterpart = "ground"
                ground_gaps.append(contact.gap)
            elif contact.object_index_first is not None:
                counterpart = contact.object_index_first
                scene_gaps.append(contact.gap)
        if counterpart is not None and contact.gap <= gap_tolerance:
            sources.add(counterpart)

    diagnostics: dict[str, object] = {
        "contact_gap_tolerance": gap_tolerance,
        "support_count": len(sources),
        "support_has_ground": "ground" in sources,
        "support_sources": tuple(
            sorted(sources, key=lambda value: -1 if value == "ground" else int(value))
        ),
    }
    if ground_gaps:
        diagnostics["place_plane_min_gap"] = min(ground_gaps)
    if scene_gaps:
        diagnostics["place_scene_min_gap"] = min(scene_gaps)
    if ground_gaps and min(ground_gaps) < -gap_tolerance:
        return "place_plane_gap", diagnostics
    if scene_gaps and min(scene_gaps) < -gap_tolerance:
        return "place_scene_gap", diagnostics
    if placed_count == 0:
        return None, diagnostics
    required = 1 if "ground" in sources else minimum_support_count
    diagnostics["minimum_support_count"] = required
    if len(sources) < required:
        return "final_support", diagnostics
    return None, diagnostics


def _simulation_phase(
    integration: Mapping[str, object],
    values: Mapping[str, object],
) -> SimulationPhaseConfig:
    return SimulationPhaseConfig(
        dt=float(_profile_value(integration, "dt")),
        energy_threshold=float(_profile_value(values, "energy_threshold")),
        velocity_integral_threshold=float(
            _profile_value(values, "velocity_integral_threshold")
        ),
        min_time=float(_profile_value(values, "min_time")),
        max_time=float(_profile_value(values, "max_time")),
        extra_steps=int(_profile_value(values, "extra_steps")),
        final_extra_steps=values.get("final_extra_steps"),
        freeze_radius=(
            None
            if values.get("freeze_radius") is None
            else float(values["freeze_radius"])
        ),
    )


def _profile_mapping(
    values: Mapping[str, object], key: str
) -> Mapping[str, object]:
    result = _profile_value(values, key)
    if not isinstance(result, Mapping):
        raise TypeError(f"simulation profile '{key}' must be a mapping")
    return result


def _profile_value(values: Mapping[str, object], key: str):
    if key not in values:
        raise ValueError(f"simulation profile is missing '{key}'")
    return values[key]
