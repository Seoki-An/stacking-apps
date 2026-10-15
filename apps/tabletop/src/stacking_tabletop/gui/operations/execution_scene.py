"""Recorded scene plus measured robot and estimated carried/released object."""

import numpy as np
from dataclasses import dataclass, replace

from stacking_common.artifacts import StackRunStore
from stacking_tabletop.planning import TabletopRobotTrajectoryProjector

from .controller import OperationsFrame, RenderObject
from .environment import planning_scene_boxes


@dataclass(frozen=True)
class ExecutionFrame(OperationsFrame):
    observed_scene_points: object = None
    motion_path_points: object = None


class ExecutionScene:
    def __init__(self, saved, repository):
        self.saved = saved
        self.projector = TabletopRobotTrajectoryProjector(saved.planning, repository)
        self.telemetry = None
        self.scan_preview = None
        self.motion_preview = None
        self.show_motion_review = False
        self.released = {}
        self.observed_scene_points = None
        checkpoint = StackRunStore.load_execution_checkpoint(saved.path)
        step = next(iter(checkpoint.pending_steps), saved.plans[-1].step)
        self.plan = next(plan for plan in saved.plans if plan.step == step)
        self.motion_path_points = self._motion_path(self.plan)
        self.boxes = planning_scene_boxes(saved.planning)
        self.mesh_paths = {int(role.split("/")[1]): path
                           for role, path in saved.assets.items() if role.startswith("object/")}

    def select_plan(self, plan):
        self.plan = plan
        self.telemetry = None
        self.motion_preview = None
        self.show_motion_review = False
        self.motion_path_points = self._motion_path(plan)

    def _motion_path(self, plan):
        points = [self.projector.tool_pose(q)[:3]
                  for segment in plan.result.segments for q in segment.positions]
        return None if len(points) < 2 else np.vstack(points)

    def event(self, event):
        pose = event.diagnostics.get("release_pose")
        if (event.status.value == "succeeded" or event.diagnostics.get("release_completed")) and pose is not None:
            plan = next(plan for plan in self.saved.plans if plan.step == event.diagnostics["step"])
            self.released[plan.problem.action.object_id] = np.asarray(pose)

        if 'stack_scene_pcd' in event.diagnostics:
            from pathlib import Path
            from stacking_tabletop.acquisition.hand_eye_capture import digest
            from stacking_tabletop.acquisition.calibration import BASE_LINK_T_UR_BASE
            import open3d as o3d
            path = Path(event.diagnostics['stack_scene_pcd'])
            if digest(path) != event.diagnostics['stack_scene_pcd_sha256']:
                raise ValueError('recorded stacked-scene PCD changed')
            points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
            if not len(points) or not np.isfinite(points).all():
                raise ValueError('stacked-scene PCD is empty or invalid')
            # Captures are in ROS base_link; the workbench uses the UR-base world.
            self.observed_scene_points = points @ BASE_LINK_T_UR_BASE[:3, :3].T
            self.observed_scene_points.setflags(write=False)

    def frame(self, *, geometry_mode="visual"):
        problem = self.plan.problem
        target = problem.action.object_id
        scene = {item.object_id: item.pose for item in (*problem.pick_scene_objects, *problem.place_scene_objects)}
        scene.update(self.released)
        scene.pop(target, None)
        placed = tuple(RenderObject(f"execution/scene/{key}", key, pose, "placed")
                       for key, pose in scene.items())
        state = self.telemetry
        actual = self.released.get(target, problem.pick_pose)
        if state is not None and state["object_pose"] is not None:
            actual = state["object_pose"]
        overlays = (
            RenderObject(f"execution/expected/{target}", target, problem.place_pose,
                         "candidate_solved", opacity=.25, color=(.2, .65, 1.)),
            RenderObject(f"execution/estimated/{target}", target, actual, "placed"),
        )
        parts = []
        opening = self.saved.planning.robot.gripper_opening
        if state is not None:
            # Legacy raw encoder mapping; clamp only the display to URDF limits.
            raw = state["gripper_raw"]
            opening = (self.saved.planning.robot.gripper_opening if raw is None else
                       np.clip((raw - 91.) / (1500. - 91.) * .9,
                               self.saved.planning.gripper.opening_lower,
                               self.saved.planning.gripper.opening_upper))
            parts.extend(self.projector.project(
                state["joints"], float(opening), geometry_mode=geometry_mode
            ))
        if self.scan_preview is not None:
            parts.extend(replace(part, part_id=f'scan_preview/{part.part_id}', color='#1ad9ff')
                for part in self.projector.project(self.scan_preview, float(opening), geometry_mode=geometry_mode))
        motion_preview = getattr(self, "motion_preview", None)
        if motion_preview is not None:
            joints, preview_opening = motion_preview
            parts.extend(replace(part, part_id=f'motion_preview/{part.part_id}', color='#57e389')
                for part in self.projector.project(joints, preview_opening, geometry_mode=geometry_mode))
        return ExecutionFrame(self.plan.step, placed, overlays, (), (), None, None,
            self.plan.candidate_id, 0, 0, tuple(parts), self.boxes, self.mesh_paths,
            observed_scene_points=getattr(self, "observed_scene_points", None),
            motion_path_points=(getattr(self, "motion_path_points", None)
                                if getattr(self, "show_motion_review", False) else None))
