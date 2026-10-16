"""Resolve and validate an executable saved tabletop sequence before ROS starts."""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from stacking_common.artifacts import StackRunStore, build_execution_command, verify_context_assets
from stacking_common.planning import FrameTransform
from stacking_tabletop.config.planning import (
    TabletopPlanningProfile, TabletopRobotPlanningProfile,
    TabletopGripperPlanningProfile, TabletopPlanningObstacle,
)

from .motion import UR_JOINT_NAMES, validate_joint_path


STAGES = ("pick_approach", "pick_retreat", "transfer", "place_approach", "place_retreat")
MODES = ("free", "attached", "attached", "attached", "free")


@dataclass(frozen=True)
class SavedExecutionRun:
    path: Path
    context: object
    planning: TabletopPlanningProfile
    plans: tuple
    assets: dict
    grasp_steps: int

    def verify_assets(self):
        verify_context_assets(self.context, self.assets)
        for kind in ("robot", "gripper"):
            for element in ET.parse(self.assets[f"{kind}/urdf"]).iter():
                filename = element.get("filename")
                if filename is not None and f"{kind}/{filename}" not in self.context.asset_hashes:
                    raise ValueError(f"recorded context lacks {kind} dependency hash: {filename}")


def recorded_planning_profile(context):
    """Reconstruct resolved values; current YAML does not redefine a saved plan."""
    values = context.profiles["planning_profile"]
    return TabletopPlanningProfile(
        world_T_stack=FrameTransform(**dict(values["world_T_stack"])),
        frame_id=values["frame_id"],
        robot=TabletopRobotPlanningProfile(**dict(values["robot"])),
        gripper=TabletopGripperPlanningProfile(**dict(values["gripper"])),
        obstacle_friction=values["obstacle_friction"],
        handoff_position=values["handoff_position"],
        obstacles=tuple(TabletopPlanningObstacle(**dict(item)) for item in values["obstacles"]),
    )


def resolve_execution_assets(context, repository, planning):
    import stacking_core._native as native

    root = Path(repository).resolve()
    result = {}
    for role in context.asset_hashes:
        if role == "stacking_core_native":
            result[role] = Path(native.__file__)
        elif role in {"robot/urdf", "gripper/urdf"}:
            result[role] = root / getattr(planning, role.split("/")[0]).urdf
        elif role.startswith(("robot/", "gripper/")):
            kind, filename = role.split("/", 1)
            asset = root / filename
            if not asset.is_file():
                asset = (root / getattr(planning, kind).urdf).parent / filename
            result[role] = asset
        elif role.startswith("object/"):
            object_id = int(role.split("/", 1)[1])
            directory = context.profiles["inventory_profile"]["model_directory"]
            result[role] = root / directory / f"model_{object_id}.obj"
        else:
            raise ValueError(f"cannot resolve recorded execution asset role: {role}")
    return result


def load_execution_run(path, repository):
    path = Path(path).resolve()
    info = StackRunStore.info(path)
    if info.status == "running":
        raise ValueError("stop/finalize offline planning before executing its run")
    if 'scene_refinement' in info.metadata and info.status != 'complete':
        raise ValueError('incomplete scene refinement cannot supply executable motions')
    plans = StackRunStore.load_committed_placement_plans(path)
    if not plans or tuple(plan.step for plan in plans) != tuple(range(1, len(plans) + 1)):
        raise ValueError("execution needs contiguous committed motion plans starting at step 1")
    digest = plans[0].context_sha256
    if digest is None or any(plan.context_sha256 != digest for plan in plans):
        raise ValueError("execution requires one recorded context for the saved sequence")
    context = StackRunStore.load_run_context(path, digest)
    try:
        planning = recorded_planning_profile(context)
        solver = context.profiles["native_solver"]
        steps = solver["motion"]["direct"]["grasp_steps"]
        if solver["allow_regrasp"] or isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("execution requires direct planning with positive grasp_steps")
        validate_context(context, planning)
        limit = info.metadata.get('scene_refinement', {}).get('validated_through_step')
        for plan in plans:
            if f"object/{plan.problem.action.object_id}" not in context.asset_hashes:
                raise ValueError("execution object has no recorded asset hash")
            if limit is not None and plan.step > limit:
                if (not plan.result.diagnostics.get('motion_requires_replan')
                        or not plan.result.segments
                        or any(not segment.name.startswith('unvalidated/') for segment in plan.result.segments)):
                    raise ValueError('unvalidated scene continuation must contain blocked motion seeds')
                continue
            command = build_execution_command(plan, context)
            validate_command(command, planning, steps)
        assets = resolve_execution_assets(context, repository, planning)
    except (KeyError, TypeError) as error:
        raise ValueError(f"saved execution context is incomplete: {error}") from error
    result = SavedExecutionRun(path, context, planning, plans, assets, steps)
    result.verify_assets()
    return result


def validate_context(context, planning):
    if context.joint_names != UR_JOINT_NAMES or context.frame_id != "world":
        raise ValueError("execution requires canonical UR5e joints in the tabletop world frame")
    if planning.world_T_stack.parent_frame != context.frame_id:
        raise ValueError("recorded planning frame differs from context")
    robot, hand = planning.robot, planning.gripper
    identity = [0., 0., 0., 0., 0., 0., 1.]
    if (robot.tool_link != "grasp_frame" or hand.root_link != "grasp_frame"
            or not np.allclose(robot.link_from_tool, identity)
            or not np.allclose(hand.grasp_from_root, identity)
            or hand.opening_offset.shape != (1,) or hand.opening_direction.shape != (1,)
            or not np.allclose(hand.opening_offset, [0.])
            or not np.allclose(hand.opening_direction, [1.])):
        raise ValueError("recorded robot/gripper model does not match the tabletop adapter")
    if not -0.1 <= hand.opening_lower <= robot.gripper_opening <= hand.opening_upper <= 0.9:
        raise ValueError("recorded gripper limits are outside the supported angle mapping")
    validate_joint_path([robot.home_positions])
    required = {"robot/urdf", "gripper/urdf", "stacking_core_native"}
    if not required.issubset(context.asset_hashes):
        raise ValueError("recorded execution context lacks robot/gripper/native asset hashes")
    geometry = {item.name: item for item in context.fixed_geometry}
    if set(geometry) != {item.obstacle_id for item in planning.obstacles} or not {"pick_table", "place_table"}.issubset(geometry):
        raise ValueError("recorded finite table geometry is missing or inconsistent")
    for obstacle in planning.obstacles:
        box = geometry[obstacle.obstacle_id]
        if (box.kind != "box" or not np.allclose(box.pose, np.r_[obstacle.center, 0., 0., 0., 1.])
                or not np.allclose(box.size, obstacle.size) or box.friction != planning.obstacle_friction):
            raise ValueError("recorded collision geometry differs from planning inputs")


def validate_command(command, planning, grasp_steps):
    segments = command.segments
    if tuple(item.name for item in segments) != STAGES or tuple(item.mode for item in segments) != MODES:
        raise ValueError("execution needs five direct-plan segments with recorded attachment modes; regenerate old plans")
    if command.joint_names != UR_JOINT_NAMES or command.frame_id != "world":
        raise ValueError("unsupported execution joint ordering or frame")
    for segment in segments:
        validate_joint_path(segment.positions)
        if len(segment.positions) < 2:
            raise ValueError("execution segments require at least two samples")
        if segment.openings is None or np.any(segment.openings < planning.gripper.opening_lower - 1e-9) or np.any(segment.openings > planning.gripper.opening_upper + 1e-9):
            raise ValueError("execution requires sample-aligned openings inside recorded model limits")
    for before, after in zip(segments, segments[1:]):
        if not np.allclose(before.positions[-1], after.positions[0], atol=1e-6, rtol=0):
            raise ValueError("saved motion has a discontinuous raw joint boundary")
    home = planning.robot.home_positions
    if not np.allclose(segments[0].positions[0], home, atol=1e-6, rtol=0) or not np.allclose(segments[-1].positions[-1], home, atol=1e-6, rtol=0):
        raise ValueError("saved sequence must start and return to the recorded home branch")
    if np.max(np.abs(np.diff(segments[1].positions[:2], axis=0))) > .25 + 1e-9:
        raise ValueError("pick retreat starts with an unsafe IK branch jump")
    events = command.grasp_events
    expected = (("acquire", 0), ("release", 3))
    if len(events) != 2:
        raise ValueError("saved plan lacks explicit acquire/release events; regenerate it")
    for event, (kind, index) in zip(events, expected):
        if event.event != kind or event.segment_index != index or event.sample_index != len(segments[index].positions) - 1:
            raise ValueError("acquire/release events must match direct-plan endpoints")
        if not planning.gripper.opening_lower <= event.opening <= planning.gripper.opening_upper:
            raise ValueError("event opening is outside recorded gripper limits")
    if events[0].opening != events[1].opening:
        raise ValueError("direct execution requires one unchanged grasp")
    if len(segments[0].positions) <= grasp_steps:
        raise ValueError("recorded pickup approach needs travel before its grasp_steps")
    # Transfer already reaches pre-place. The placement segment can consist
    # entirely of the final grasp_steps samples, without another travel prefix.
    if len(segments[3].positions) < grasp_steps:
        raise ValueError("recorded placement approach is too short for its grasp_steps")
