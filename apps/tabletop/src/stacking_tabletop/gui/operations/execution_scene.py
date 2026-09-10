"""Recorded scene plus measured robot and estimated carried/released object."""

import numpy as np
from dataclasses import replace

from stacking_common.artifacts import StackRunStore
from stacking_tabletop.config.visualization import TabletopSceneBox
from stacking_tabletop.planning import TabletopRobotTrajectoryProjector

from .controller import OperationsFrame, RenderObject


class ExecutionScene:
    def __init__(self, saved, repository):
        self.saved = saved
        self.projector = TabletopRobotTrajectoryProjector(saved.planning, repository)
        self.telemetry = None
        self.scan_preview = None
        self.released = {}
        checkpoint = StackRunStore.load_execution_checkpoint(saved.path)
        step = next(iter(checkpoint.pending_steps), saved.plans[-1].step)
        self.plan = next(plan for plan in saved.plans if plan.step == step)
        self.boxes = tuple(TabletopSceneBox(
            box.obstacle_id, "support", box.minimum, box.maximum, (.4, .45, .5), .4,
        ) for box in saved.planning.obstacles)
        self.mesh_paths = {int(role.split("/")[1]): path
                           for role, path in saved.assets.items() if role.startswith("object/")}

    def select_plan(self, plan):
        self.plan = plan
        self.telemetry = None

    def event(self, event):
        pose = event.diagnostics.get("release_pose")
        if event.status.value == "succeeded" and pose is not None:
            plan = next(plan for plan in self.saved.plans if plan.step == event.diagnostics["step"])
            self.released[plan.problem.action.object_id] = np.asarray(pose)

    def frame(self):
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
        parts = ()
        if state is not None:
            # Legacy raw encoder mapping; clamp only the display to URDF limits.
            opening = np.clip((state["gripper_raw"] - 91.) / (1500. - 91.) * .9,
                              self.saved.planning.gripper.opening_lower,
                              self.saved.planning.gripper.opening_upper)
            parts = tuple(self.projector.project(state["joints"], float(opening), geometry_mode="visual"))
            if self.scan_preview is not None:
                parts += tuple(replace(part, part_id=f'scan_preview/{part.part_id}', color='#1ad9ff')
                    for part in self.projector.project(self.scan_preview, float(opening), geometry_mode='visual'))
        return OperationsFrame(self.plan.step, placed, overlays, (), (), None, None,
            self.plan.candidate_id, 0, 0, parts, self.boxes, self.mesh_paths)
