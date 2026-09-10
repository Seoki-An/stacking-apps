"""Strict translation of platform solver YAML into native planner config."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import yaml

from .angles import resolve_degree_fields


_GRASP_FIELDS = {
    "sampling": frozenset(
        {
            "strategy",
            "max_seeds",
            "dir_samples",
            "spin_samples",
            "spin_step",
            "include_flipped",
            "retreat_distance",
            "fallback_retreat_distances",
            "fallback_min_candidates",
            "max_target_width",
            "aperture_margin",
            "scene_clearance_margin",
            "preferred_parallel_axis",
            "max_candidates",
        }
    ),
    "generation": frozenset(
        {"separate_margin", "plane_separate_margin", "contact_margin"}
    ),
    "cost": frozenset(
        {
            "antipodal_normal",
            "antipodal_position",
            "align",
            "enclosure",
            "radial_distance",
            "center_distance",
            "contact",
            "teeth_fit",
            "teeth_align",
            "flatness",
        }
    ),
    "score": frozenset(
        {
            "center_distance",
            "align",
            "enclosure",
            "radial_distance",
            "teeth_gap",
            "teeth_fit",
            "teeth_align",
        }
    ),
    "trust_region": frozenset(
        {
            "max_iters",
            "subproblem_tol",
            "radius_max",
            "radius_init",
            "radius_reduction",
            "radius_expansion",
            "gain_ratio_lower",
            "gain_ratio_upper",
            "improvement_tol",
            "step_tol",
        }
    ),
    "alm": frozenset(
        {
            "max_iters",
            "inequality_tol",
            "equality_tol",
            "inequality_penalty_init",
            "equality_penalty_init",
            "inequality_penalty_increase",
            "equality_penalty_increase",
        }
    ),
    "force": frozenset({"enabled", "friction", "wrench_scale", "damping"}),
    "force_weight": frozenset(
        {"wrench", "complementarity", "cone", "moment"}
    ),
    "simulation": frozenset(
        {
            "steps",
            "dt",
            "virtual_closing_effort",
            "virtual_approach_force",
            "damping",
            "friction",
            "pgs_iters",
            "error_reduction_ratio",
            "target_contact_margin",
            "obstacle_margin",
            "plane_obstacle_margin",
            "active_impulse_tol",
            "settled_velocity_tol",
            "relaxed_velocity_tol",
            "accept_relaxed",
            "grasp_offset_tol",
        }
    ),
}

_MOTION_FIELDS = {
    "motion": frozenset(
        {
            "smooth_weight",
            "boundary_weight",
            "collision_weight",
            "joint_limit_weight",
            "smooth_boundary_alpha",
            "swing_smoothness_scale",
            "collision_margin",
            "plane_feasibility_margin",
            "joint_limit_margin",
            "grasped_boundary_pos_scale",
            "grasped_boundary_rot_scale",
            "target_collision_tol",
            "collision_penetration_clamp",
            "collision_alm_enabled",
            "collision_alm_max_iters",
            "collision_alm_beta_init",
            "collision_alm_beta_increase",
            "collision_alm_tol",
            "step_size",
            "max_iters",
            "tol",
            "ik_max_iters",
            "ik_tol",
        }
    ),
    "inverse_kinematics": frozenset({"max_iters", "tol", "initialization"}),
    "direct": frozenset(
        {
            "joint_grasp_recovery",
            "scene_approach",
            "minimum_approach_up_component",
            "simulation_refinement",
            "approach_dir_tool",
            "approach_distance",
            "target_pos_tol",
            "target_rot_tol",
            "move_steps",
            "grasp_steps",
            "max_candidates",
            "worker_count",
        }
    ),
    "regrasp": frozenset(
        {
            "joint_grasp_recovery",
            "scene_approach",
            "minimum_approach_up_component",
            "approach_dir_tool",
            "approach_distance",
            "target_pos_tol",
            "target_rot_tol",
            "max_handoff_orientation_distance",
            "move_steps",
            "grasp_steps",
            "yaw_samples",
            "max_candidates",
            "worker_count",
        }
    ),
    "stable_pose": frozenset({"sampling_level", "stable_eigenvalue_min"}),
}


@dataclass(frozen=True)
class NativeSolverProfile:
    """Resolved native pick-place settings from the two platform profiles."""

    grasp: Mapping[str, Mapping[str, object]]
    motion: Mapping[str, Mapping[str, object]]
    allow_regrasp: bool

    def __post_init__(self):
        trust_region = self.grasp.get("trust_region", {})
        initial = trust_region.get("radius_init")
        maximum = trust_region.get("radius_max")
        if initial is not None and maximum is not None and initial > maximum:
            raise ValueError(
                "grasp trust_region.radius_init must be <= radius_max "
                f"(radius_init={initial}, radius_max={maximum})"
            )

    def native_config(self, core_module):
        config = core_module.PickPlaceConfig()
        grasp = self.grasp
        motion = self.motion

        _apply(config.direct.grasp_sampling, grasp["sampling"], core_module)
        generation = config.direct.grasp_generation
        _apply(generation, grasp["generation"], core_module)
        _apply(generation.cost, grasp["cost"], core_module)
        _apply(generation.score, grasp["score"], core_module)
        _apply(generation.trust_region, grasp["trust_region"], core_module)
        _apply(generation.alm, grasp["alm"], core_module)
        _apply(generation.force, grasp["force"], core_module)
        _apply(generation.force.weight, grasp["force_weight"], core_module)
        _apply(config.direct.grasp_simulation, grasp["simulation"], core_module)

        _apply(config.direct.motion, motion["motion"], core_module)
        _apply(config.regrasp.motion, motion["motion"], core_module)
        _apply(
            config.direct.inverse_kinematics,
            motion["inverse_kinematics"],
            core_module,
        )
        _apply(config.direct, motion["direct"], core_module)
        _apply(config.regrasp, motion["regrasp"], core_module)
        _apply(config.regrasp.stable_pose, motion["stable_pose"], core_module)
        config.allow_regrasp = self.allow_regrasp
        return config


def load_native_solver_profile(
    grasp_path: str | Path,
    motion_path: str | Path,
) -> NativeSolverProfile:
    """Load native grasp and motion profiles with unknown-key rejection."""

    grasp = _load_sections(grasp_path, _GRASP_FIELDS)
    motion_values = _yaml_mapping(motion_path)
    if motion_values.get("units") != "si":
        raise ValueError("motion profile units must be 'si'")
    motion = _sections(motion_values, _MOTION_FIELDS, motion_path)
    allow_regrasp = motion_values.get("allow_regrasp", True)
    if not isinstance(allow_regrasp, bool):
        raise TypeError("motion profile 'allow_regrasp' must be boolean")
    unknown = sorted(
        set(motion_values) - {"units", "allow_regrasp", *_MOTION_FIELDS}
    )
    if unknown:
        raise ValueError(f"unknown motion profile settings: {unknown}")
    return NativeSolverProfile(grasp, motion, allow_regrasp)


def _load_sections(
    path: str | Path,
    fields: Mapping[str, frozenset[str]],
) -> Mapping[str, Mapping[str, object]]:
    values = _yaml_mapping(path)
    if values.get("units") != "si":
        raise ValueError(f"{Path(path).stem} profile units must be 'si'")
    unknown = sorted(set(values) - {"units", *fields})
    if unknown:
        raise ValueError(f"unknown {Path(path).stem} profile settings: {unknown}")
    return _sections(values, fields, path)


def _sections(values, fields, path):
    resolved = {}
    for section, allowed in fields.items():
        raw = values.get(section, {})
        if not isinstance(raw, Mapping):
            raise TypeError(f"{Path(path).stem} profile '{section}' must be a mapping")
        angular_fields = {
            "sampling": ("spin_step",),
            "motion": ("joint_limit_margin",),
            "direct": ("target_rot_tol",),
            "regrasp": ("target_rot_tol", "max_handoff_orientation_distance"),
        }
        raw = resolve_degree_fields(raw, angular_fields.get(section, ()))
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(
                f"unknown {Path(path).stem} {section} settings: {unknown}"
            )
        resolved[section] = MappingProxyType(dict(raw))
    return MappingProxyType(resolved)


def _yaml_mapping(path: str | Path) -> Mapping[str, object]:
    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(f"missing solver profile: {profile_path}")
    values = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise TypeError(f"{profile_path.stem} profile root must be a mapping")
    return values


def _apply(native, values: Mapping[str, object], core_module) -> None:
    for key, raw in values.items():
        value = _native_value(key, raw, core_module)
        setattr(native, key, value)


def _native_value(key: str, value: object, core_module):
    if key == "strategy":
        choices = {
            "antipodal_surface": core_module.GraspSamplingStrategy.ANTIPODAL_SURFACE,
            "parallel_jaw": core_module.GraspSamplingStrategy.PARALLEL_JAW,
        }
        try:
            return choices[str(value)]
        except KeyError as error:
            raise ValueError(f"unsupported grasp sampling strategy: {value}") from error
    if key == "initialization":
        choices = {
            "swing": core_module.InverseKinematicsInitialization.SWING,
            "provided": core_module.InverseKinematicsInitialization.PROVIDED,
        }
        try:
            return choices[str(value)]
        except KeyError as error:
            raise ValueError(f"unsupported IK initialization: {value}") from error
    if key in {"approach_dir_tool", "preferred_parallel_axis", "grasp_offset_tol"}:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"native planner '{key}' must be a finite XYZ vector")
        return vector
    return value
