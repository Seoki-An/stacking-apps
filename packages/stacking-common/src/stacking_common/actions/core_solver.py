"""Placement-pose solving through the canonical stacking-core scene API."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from stacking_common.domain import ObjectInventory, State
from stacking_common.planning import CoreSceneBuilder, SceneMobility, SceneObjectInstance

from .types import InitializedPlacement, PlacementSolveResult


_CONFIG_FIELDS = {
    "trust_region": frozenset(
        {
            "max_iters", "eps", "delta_init", "delta_max",
            "delta_reduction_rate", "delta_expansion_rate",
            "delta_lower_thresh", "delta_upper_thresh", "improvement_thresh",
            "tol",
        }
    ),
    "hausdorff": frozenset(
        {
            "eps", "max_iters", "error_tol", "radius_init",
            "radius_reduction_rate", "radius_expansion_rate",
            "gain_ratio_lower_thresh", "gain_ratio_upper_thresh",
            "gain_ratio_max",
        }
    ),
    "objective": frozenset(
        {
            "rho", "narrow_phase_scene_tol", "narrow_phase_candidate_tol",
            "eps_gap", "eps_comp", "eps_cone", "eps_target", "k_comp",
            "k_wrench", "k_gap", "k_gap_c", "k_target", "k_potential",
            "k_xy", "k_box", "k_reg", "k_lower", "w_box", "gravity",
            "ground_height",
        }
    ),
    "force_solver": frozenset(
        {
            "max_iters", "beta_consensus", "beta_contact",
            "beta_update_interval", "tol_abs", "tol_rel",
        }
    ),
}

_SCHEDULE_FIELDS = frozenset({
    "times", "max_iters_warmup", "max_iters_final", "objective_weights",
})

# Compatibility for saved profiles from before schedules were platform-owned.
# Current platform YAML files explicitly supply these coefficients.
_LEGACY_SCHEDULE_WEIGHTS = {
    "k_wrench": {"ramp": 5.0},
    "k_comp": {"ramp": 100.0},
    "k_gap_c": {"constant": 40.0, "ramp": 200.0, "gap_feedback": 1e5},
    "k_gap": {"sine": 20.0},
    "k_target": {"constant": 30.0, "sine": 300.0},
    "k_xy": {"anti_ramp": 0.0},
    "k_potential": {"constant": 0.10, "anti_ramp": 0.02},
    "k_reg": {"anti_ramp": 0.0},
    "k_lower": {"anti_ramp": 0.0},
}
_SCHEDULE_BASES = frozenset({"constant", "ramp", "anti_ramp", "sine", "gap_feedback"})


@dataclass(frozen=True)
class CorePosegenProfile:
    """Resolved native pose-generation and per-scene settings."""

    frame_id: int
    target_model_id: int
    target_entity_id: int
    target_friction: float
    target_sharpness: int
    target_epsilon: float
    placed_as_boundaries: bool = False
    boundary_radius: float | None = field(default=None, metadata={"omit_default_from_context": True})
    trust_region: Mapping[str, object] = field(default_factory=dict)
    hausdorff: Mapping[str, object] = field(default_factory=dict)
    objective: Mapping[str, object] = field(default_factory=dict)
    force_solver: Mapping[str, object] = field(default_factory=dict)
    schedule: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.boundary_radius is not None and (not math.isfinite(self.boundary_radius) or self.boundary_radius < 0):
            raise ValueError("posegen boundary radius must be finite and nonnegative")
        for name in ("frame_id", "target_model_id", "target_entity_id"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"posegen {name} must be non-negative")
            object.__setattr__(self, name, int(getattr(self, name)))
        if self.target_entity_id == 0:
            raise ValueError("posegen target entity ID must not be zero")
        if not math.isfinite(self.target_friction) or self.target_friction < 0.0:
            raise ValueError("posegen target friction must be non-negative")
        if self.target_sharpness <= 0:
            raise ValueError("posegen target sharpness must be positive")
        if not math.isfinite(self.target_epsilon) or self.target_epsilon <= 0.0:
            raise ValueError("posegen target epsilon must be positive")
        for section, allowed in _CONFIG_FIELDS.items():
            values = dict(getattr(self, section))
            unknown = sorted(set(values) - allowed)
            if unknown:
                raise ValueError(
                    f"unknown posegen {section} settings: {unknown}"
                )
            object.__setattr__(self, section, MappingProxyType(values))
        schedule = dict(self.schedule)
        unknown = sorted(set(schedule) - _SCHEDULE_FIELDS)
        if unknown:
            raise ValueError(f"unknown posegen schedule settings: {unknown}")
        times = tuple(float(value) for value in schedule.get("times", (1.0,)))
        if not times or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in times
        ):
            raise ValueError("posegen schedule times must lie between zero and one")
        if tuple(sorted(times)) != times:
            raise ValueError("posegen schedule times must be ordered")
        schedule["times"] = times
        for key in ("max_iters_warmup", "max_iters_final"):
            value = int(schedule.get(key, self.trust_region.get("max_iters", 100)))
            if value <= 0:
                raise ValueError(f"posegen schedule {key} must be positive")
            schedule[key] = value
        weights = schedule.get("objective_weights", _LEGACY_SCHEDULE_WEIGHTS)
        if not isinstance(weights, Mapping) or set(weights) != set(_LEGACY_SCHEDULE_WEIGHTS):
            raise ValueError("posegen schedule objective_weights must specify all scheduled weights")
        resolved_weights = {}
        for name, terms in weights.items():
            if not isinstance(terms, Mapping) or not terms or set(terms) - _SCHEDULE_BASES:
                raise ValueError(f"invalid posegen schedule coefficients for {name}")
            coefficients = {basis: float(value) for basis, value in terms.items()}
            if any(not math.isfinite(value) or value < 0.0 for value in coefficients.values()):
                raise ValueError(f"posegen schedule {name} coefficients must be finite and non-negative")
            resolved_weights[name] = MappingProxyType(coefficients)
        schedule["objective_weights"] = MappingProxyType(resolved_weights)
        object.__setattr__(self, "schedule", MappingProxyType(schedule))
        object.__setattr__(self, "target_friction", float(self.target_friction))
        object.__setattr__(self, "target_sharpness", int(self.target_sharpness))
        object.__setattr__(self, "target_epsilon", float(self.target_epsilon))
        object.__setattr__(self, "placed_as_boundaries", bool(self.placed_as_boundaries))

    def native_config(self, core_module):
        """Build the mutable pybind configuration only at the native boundary."""

        config = core_module.PosegenConfig()
        for section in _CONFIG_FIELDS:
            native_section = getattr(config, section)
            for key, value in getattr(self, section).items():
                setattr(
                    native_section,
                    key,
                    np.asarray(value, dtype=float) if key == "k_wrench" else value,
                )
            setattr(config, section, native_section)
        return config


class CorePlacementPoseSolver:
    """Adapt immutable stacking-core pose-generation results to common actions."""

    def __init__(
        self,
        inventory: ObjectInventory,
        profile: CorePosegenProfile,
        *,
        core_module=None,
    ) -> None:
        if core_module is None:
            try:
                import stacking_core as core_module
            except ImportError as error:
                raise RuntimeError(
                    "stacking-core pose-generation bindings are required"
                ) from error
        self.core = core_module
        self.inventory = inventory
        self.profile = profile
        self.scene_builder = CoreSceneBuilder(
            inventory.models,
            frame_id=profile.frame_id,
            core_module=core_module,
        )
        if profile.target_entity_id <= len(inventory):
            raise ValueError("posegen target entity ID collides with an object entity")
        self.target_model = core_module.BodyModel.dsf(
            profile.target_model_id,
            inventory.target.points,
            mass=1.0,
            inertia=np.eye(3),
            friction=profile.target_friction,
            sharpness=profile.target_sharpness,
            epsilon=profile.target_epsilon,
        )
        self.target_body = core_module.BodyInstance(
            profile.target_entity_id,
            self.target_model,
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            mobility=core_module.Mobility.STATIC,
        )
        config = profile.native_config(core_module)
        self.gravity = float(config.objective.gravity)
        if not math.isfinite(self.gravity) or self.gravity <= 0.0:
            raise ValueError("equilibrium normalization requires positive finite gravity")
        self.generator = core_module.PoseGenerator(config)

    def solve(
        self,
        inventory: ObjectInventory,
        state: State,
        object_index: int,
        initial: InitializedPlacement,
    ) -> PlacementSolveResult:
        self._validate_request(inventory, state, object_index)
        pose = initial.pose
        c_gap = 0.0
        result = None
        schedule_times = self.profile.schedule["times"]
        for step, schedule_time in enumerate(schedule_times):
            self.generator.config = self._scheduled_config(
                float(schedule_time), c_gap, final=step == len(schedule_times) - 1,
            )
            result = self.generator.solve(self._problem(state, object_index, pose))
            pose = np.asarray(result.optimal_pose, dtype=float)
            c_gap = float(result.c_gap)
        if result is None:
            raise RuntimeError("posegen schedule did not execute")
        return self._placement_result(result, self._native_mass_scale(state, object_index))

    def _validate_request(self, inventory, state, object_index):
        if inventory is not self.inventory:
            raise ValueError("pose solver inventory does not match its composition")
        if tuple(state.object_ids) != inventory.object_ids:
            raise ValueError("state inventory does not match pose solver inventory")
        if object_index in state.placed_object_indices:
            raise ValueError("posegen candidate is already placed")

    def _scene_objects(self, state):
        objects = [
            SceneObjectInstance(
                object_index=placed_index,
                object_id=state.object_ids[placed_index],
                pose=state.object_poses[state.object_ids[placed_index]],
                mobility=SceneMobility.DYNAMIC,
            )
            for placed_index in state.placed_object_indices
        ]
        boundaries = (
            [self.scene_builder.entity_id(index) for index in state.placed_object_indices]
            if self.profile.placed_as_boundaries
            else []
        )
        if not self.profile.placed_as_boundaries and self.profile.boundary_radius is not None:
            target = self.inventory.target
            half = np.array([target.width, target.length]) / 2.0
            boundaries = [self.scene_builder.entity_id(index)
                for index in state.placed_object_indices
                if np.linalg.norm(np.maximum(np.abs(
                    state.object_poses[state.object_ids[index]][:2] - target.origin[:2]
                ) - half, 0.0)) > self.profile.boundary_radius]
        return objects, boundaries

    def _prepare_problem_scene(self, state):
        objects, boundaries = self._scene_objects(state)
        scene = self.scene_builder.build(objects, fixed_bodies=(self.target_body,))
        return scene.bodies, boundaries

    def _problem(self, state, object_index, pose, *, scene_context=None):
        if scene_context is None:
            objects, boundaries = self._scene_objects(state)
            fixed_bodies = (self.target_body,)
        else:
            fixed_bodies, boundaries = scene_context
            objects = []
        candidate = self.scene_builder.entity_id(object_index)
        candidate_object = SceneObjectInstance(
            object_index=object_index,
            object_id=state.object_ids[object_index],
            pose=pose,
            mobility=SceneMobility.DYNAMIC,
        )
        scene = self.scene_builder.build(
            [*objects, candidate_object], fixed_bodies=fixed_bodies,
        )
        return self.core.PosegenProblem(
            scene, candidate, targets=[self.profile.target_entity_id], boundaries=boundaries,
        )

    def _native_mass_scale(self, state, object_index):
        """Return the mean mass native posegen divides every body mass by.

        The native force solve runs on masses divided by the mean mass of the
        candidate and all placed bodies (fixed boundaries included, targets
        excluded), so its wrenches and contact forces are newtons per this many
        kilograms.
        """
        masses = [self.inventory[index].mass
                  for index in (*state.placed_object_indices, object_index)]
        masses = [mass for mass in masses if math.isfinite(mass) and mass > 0.0]
        return sum(masses) / len(masses) if masses else 1.0

    def _placement_result(self, result, native_mass_scale):
        return PlacementSolveResult(**self._placement_values(result, native_mass_scale))

    def _placement_values(self, result, native_mass_scale):
        stats = result.solver
        diagnostics = {
            "posegen_iterations": int(stats.iters),
            "posegen_objective_evaluations": int(stats.objective_evals),
            "posegen_gradient_norm": float(stats.grad_norm),
            "posegen_pose_converged": bool(stats.converged),
            "posegen_force_converged": bool(stats.force_solver.converged),
            "posegen_force_iterations": getattr(stats.force_solver, "iters", None),
            "posegen_force_primal_residual": getattr(stats.force_solver, "primal_residual", None),
            "posegen_force_dual_residual": getattr(stats.force_solver, "dual_residual", None),
            "posegen_trust_region_radius": getattr(stats, "trust_region_radius", None),
            "posegen_scene_graph_rebuilds": int(stats.scene_graph_rebuilds),
            "posegen_scene_graph_reuses": int(stats.scene_graph_reuses),
            "posegen_c_comp": float(result.c_comp),
            "posegen_contact_points": dict(result.contact_point),
            "posegen_contact_normals": dict(result.contact_normal),
            "posegen_contact_forces": {
                entity_id: [np.asarray(force, dtype=float) * native_mass_scale for force in forces]
                for entity_id, forces in result.contact_force.items()
            },
        }
        converged = bool(stats.converged and stats.force_solver.converged)
        # Restore newtons; the native c_feq fallback is quadratic in the wrench.
        wrench_norm = max(
            (
                float(np.linalg.norm(np.asarray(wrench, dtype=float), ord=1)) * native_mass_scale
                for wrench in result.net_wrench.values()
            ),
            default=float(result.c_feq) * native_mass_scale**2,
        )
        max_mass = max(stack_object.mass for stack_object in self.inventory)
        if result.net_wrench:
            # Native entity IDs are inventory indices + 1, not object IDs.
            # Use each participating body's own weight and volume length scale.
            residuals = []
            for entity_id, wrench in result.net_wrench.items():
                index = int(entity_id) - 1
                if not 0 <= index < len(self.inventory):
                    raise ValueError("equilibrium wrench has an unknown body entity ID")
                body = self.inventory[index]
                if not (math.isfinite(body.mass) and body.mass > 0.0
                        and math.isfinite(body.volume) and body.volume > 0.0):
                    raise ValueError("equilibrium normalization requires positive finite mass and volume")
                wrench = np.asarray(wrench, dtype=float)
                if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
                    raise ValueError("equilibrium wrench must be a finite six-vector")
                wrench = wrench * native_mass_scale
                weight = body.mass * self.gravity
                length = float(np.cbrt(body.volume))
                residuals.append(float(np.linalg.norm(wrench[:3]) / weight
                                       + np.linalg.norm(wrench[3:]) / (weight * length)))
            diagnostics["posegen_equilibrium_residual_normalized"] = max(residuals)
        # Feasibility filters use this legacy-form metric (N per kg of the
        # heaviest object); only reward/prior scoring uses the residual above.
        return dict(
            pose=np.asarray(result.optimal_pose, dtype=float),
            c_feq=wrench_norm / max_mass,
            c_gap=float(result.c_gap),
            converged=converged,
            rejection_reason=None if converged else "posegen_nonconvergence",
            diagnostics=diagnostics,
        )

    def _scheduled_config(self, time_value: float, c_gap: float, *, final: bool):
        config = self.profile.native_config(self.core)
        ramp = 0.5 * (1.0 - math.cos(math.pi * time_value))
        anti_ramp = 0.5 * (1.0 + math.cos(math.pi * time_value))
        sine = math.sin(math.pi * time_value)
        objective = config.objective
        bases = {"constant": 1.0, "ramp": ramp, "anti_ramp": anti_ramp,
                 "sine": sine, "gap_feedback": time_value * c_gap}
        for name, coefficients in self.profile.schedule["objective_weights"].items():
            value = sum(coefficient * bases[basis] for basis, coefficient in coefficients.items())
            if name == "k_wrench":
                value = value * np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
            setattr(objective, name, value)
        config.objective = objective
        trust = config.trust_region
        trust.tol = 10.0 ** (-4.0 * (time_value + 1.0))
        trust.max_iters = int(
            self.profile.schedule[
                "max_iters_final" if final else "max_iters_warmup"
            ]
        )
        config.trust_region = trust
        return config
