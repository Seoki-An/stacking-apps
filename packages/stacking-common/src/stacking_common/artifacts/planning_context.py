"""Platform planning provenance and non-destructive checkpoint branching."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from stacking_common.artifacts import (
    FixedCollisionGeometry, PlanningRunContext, StackRunStore,
)


def _values(value):
    if is_dataclass(value):
        return {field.name: _values(getattr(value, field.name)) for field in fields(value)
                if not field.metadata.get("omit_default_from_context", False)
                or getattr(value, field.name) != field.default}
    if isinstance(value, Mapping):
        return {str(key): _values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_values(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_values(item) for item in sorted(value)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_planning_context(application, inventory, pick_scene, solver, runtime):
    """Capture resolved profiles, native joint order, and every model dependency."""
    import stacking_core as core
    import stacking_core._native as native

    root = application.repository_root
    planning = application.planning_profile
    assets = {"stacking_core_native": Path(native.__file__)}
    robot_path = root / planning.robot.urdf
    robot_xml = ET.parse(robot_path)
    model = core.load_urdf_model(str(robot_path)).kinematics
    joints = {}
    for joint in robot_xml.findall("joint"):
        joint_id = model.find_joint(joint.attrib["name"])
        index = model.dof_index(joint_id)
        if index is not None:
            joints[index] = joint.attrib["name"]
    if set(joints) != set(range(model.dof_count)):
        raise ValueError("cannot resolve complete native robot joint order")

    for role, relative in (("robot", planning.robot.urdf), ("gripper", planning.gripper.urdf)):
        path = root / relative
        assets[f"{role}/urdf"] = path
        for element in ET.parse(path).iter():
            filename = element.get("filename")
            if filename is None:
                continue
            # Commissioned DSF references are repository-relative; visual
            # meshes are URDF-relative. Include both collision and visual assets.
            asset = root / filename
            if not asset.is_file():
                asset = path.parent / filename
            if not asset.is_file():
                raise FileNotFoundError(f"unresolved {role} asset: {filename}")
            assets[f"{role}/{filename}"] = asset
    for object_id in inventory.object_ids:
        assets[f"object/{object_id}"] = (
            root / application.inventory_profile.model_directory / f"model_{object_id}.obj"
        )
        surface = root / application.inventory_profile.model_directory / f"model_{object_id}_mesh.ply"
        if surface.is_file():
            assets[f"object_surface/{object_id}"] = surface
    extra_assets = getattr(application, "additional_context_assets", None)
    if extra_assets is not None:
        assets.update(extra_assets(inventory))
    hashes = {}
    for role, path in assets.items():
        with path.open("rb") as stream:
            hashes[role] = hashlib.file_digest(stream, "sha256").hexdigest()
    profiles = {
        name: _values(getattr(application, name))
        for name in (
            "environment_profile", "simulation_profile",
            "posegen_profile", "mcts_profile", "inventory_profile",
            "action_generation_profile", "planning_profile",
        )
    }
    profiles.update(native_solver=_values(solver), runtime=_values(runtime),
                    pick_scene=pick_scene.artifact_metadata())
    from stacking_common.domain import CompositeTargetGeometry
    if isinstance(inventory.target, CompositeTargetGeometry):
        from .target import encode_target
        from stacking_common.visualization import target_display
        profiles["target_geometry"] = encode_target(target_display(inventory.target))
    geometry = tuple(FixedCollisionGeometry(
        item.obstacle_id, "box", np.r_[item.center, 0., 0., 0., 1.],
        planning.obstacle_friction, tuple(item.size),
    ) for item in planning.obstacles)
    return PlanningRunContext(planning.world_T_stack.parent_frame,
                              tuple(joints[i] for i in range(model.dof_count)),
                              profiles, geometry, hashes)


def validate_planning_resume(source, context):
    if StackRunStore.info(source).status == "running":
        raise ValueError("source run is still marked running; stop/finalize it before resuming")
    checkpoint = StackRunStore.load_planning_checkpoint(source)
    recorded = StackRunStore.load_run_context(source, checkpoint.context_sha256)
    if recorded.sha256 != context.sha256:
        profiles = sorted(key for key in recorded.profiles.keys() | context.profiles.keys()
                          if recorded.profiles.get(key) != context.profiles.get(key))
        assets = sorted(key for key in recorded.asset_hashes.keys() | context.asset_hashes.keys()
                        if recorded.asset_hashes.get(key) != context.asset_hashes.get(key))
        raise ValueError(f"resume context mismatch: profiles={profiles}, assets={assets}; "
                         "restore recorded inputs before resuming")
    if StackRunStore.load_execution_commands(source):
        raise ValueError("cannot branch an executed/queued run; physical reconciliation is a separate workflow")
    return checkpoint


def branch_planning_run(source, destination, context):
    """Copy only the committed prefix, leaving failed/partial attempts at source."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or source == destination:
        raise FileExistsError("resume requires a new output .stack path")
    checkpoint = validate_planning_resume(source, context)
    artifact = StackRunStore.load_mcts(source)
    statistics = {int(item["step"]): item for item in StackRunStore.load_mcts_statistics(source)}
    plans = StackRunStore.load_placement_plans(source)
    # Ensure the retained steps have unambiguous successful plans.
    StackRunStore.load_committed_placement_plans(source)
    writer = StackRunStore.create_mcts(destination, replace(artifact, steps=()), metadata={
        **dict(StackRunStore.info(source).metadata),
        "planning_parent": {"path": str(source), "checkpoint_step": checkpoint.step},
    })
    try:
        from .object_assets import copy_object_assets
        copy_object_assets(source, writer)
        writer.append_run_context(context)
        for plan in plans:
            if plan.step <= checkpoint.step:
                if plan.context_sha256 != context.sha256:
                    raise ValueError("retained plan is not bound to the checkpoint context")
                writer.append_placement_plan(plan)
        for step in artifact.steps:
            if step.step <= checkpoint.step:
                writer.append_mcts_step(step, statistics=statistics.get(step.step),
                    checkpoint=checkpoint if step.step == checkpoint.step else None)
    except BaseException:
        writer.finish("failed")
        raise
    return writer, checkpoint
