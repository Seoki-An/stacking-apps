"""Construction of native UR5e/SR-gripper pick-place problems."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from stacking_common.domain import ObjectInventory
from stacking_common.planning import (
    CoreSceneBuilder,
    PlacementPlanningProblem,
    SceneMobility,
    SceneObjectInstance,
)

from ..config import TabletopPlanningProfile
from .profile import TabletopNativeSolverProfile


_IDENTITY_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
_ROBOT_COLLISION_LINKS = (
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
)
_GRIPPER_COLLISION_LINKS = (
    "base_link",
    "left_finger_link_1",
    "left_finger_tip",
    "right_finger_link_1",
    "right_finger_tip",
)


class TabletopNativePlanningBackend:
    """Own native models and solve common placement-planning problems."""

    _obstacle_entity_start = 100
    _robot_entity_start = 1001
    _gripper_entity_start = 2001
    _fixed_model_start = 10001

    def __init__(
        self,
        inventory: ObjectInventory,
        planning: TabletopPlanningProfile,
        solver: TabletopNativeSolverProfile,
        repository_root: str | Path,
        *,
        core_module=None,
    ) -> None:
        if core_module is None:
            try:
                import stacking_core as core_module
            except ImportError as error:
                raise RuntimeError(
                    "stacking-core planning bindings are required"
                ) from error
        self.core = core_module
        self.inventory = inventory
        self.planning = planning
        self.repository_root = Path(repository_root).resolve()
        self.scene_builder = CoreSceneBuilder(
            inventory.models,
            frame_id=planning.frame_id,
            core_module=core_module,
        )
        if len(inventory) >= self._obstacle_entity_start:
            raise ValueError("planning fixed entity IDs overlap the inventory")
        self.robot_urdf = self._load_urdf(planning.robot.urdf)
        self.gripper_urdf = self._load_urdf(planning.gripper.urdf)
        self.robot = self._build_robot()
        self.gripper = self._build_gripper()
        self.config = solver.native_config(core_module)
        self._obstacles = self._build_obstacles()

    def solve(self, problem: PlacementPlanningProblem):
        """Build a fresh immutable scene set and invoke native planning."""

        return self.core.solve_pick_place(
            self.build_problem(problem),
            self.config,
        )

    def build_problem(self, problem: PlacementPlanningProblem):
        if not isinstance(problem, PlacementPlanningProblem):
            raise TypeError("native planning requires a placement problem")
        if problem.frame_id != self.planning.world_T_stack.parent_frame:
            raise ValueError(
                "planning problem frame does not match the tabletop world frame"
            )
        target_index = problem.action.object_index
        if target_index >= len(self.inventory):
            raise ValueError("planning target is outside the inventory")
        expected_id = self.inventory.object_ids[target_index]
        if problem.action.object_id != expected_id:
            raise ValueError("planning target ID does not match the inventory")

        pick = self._phase(
            problem,
            problem.pick_pose,
            problem.pick_scene_objects,
        )
        place = self._phase(
            problem,
            problem.place_pose,
            problem.place_scene_objects,
        )
        handoff_pose = problem.pick_pose.copy()
        handoff_pose[:3] = self.planning.handoff_position
        handoff = self._phase(
            problem,
            handoff_pose,
            problem.place_scene_objects,
        )
        direct = self.core.DirectPlanProblem(
            pick,
            place,
            self.robot,
            self.gripper,
        )
        return self.core.PickPlaceProblem(
            direct,
            handoff,
            self.planning.handoff_position.copy(),
        )

    def _phase(self, problem, target_pose, scene_objects):
        target_index = problem.action.object_index
        objects = tuple(
            replace(
                value,
                pose=(
                    np.asarray(target_pose, dtype=float)
                    if value.object_index == target_index
                    else value.pose
                ),
                mobility=(
                    SceneMobility.KINEMATIC
                    if value.object_index == target_index
                    else SceneMobility.STATIC
                ),
            )
            for value in scene_objects
        )
        target_count = sum(value.object_index == target_index for value in objects)
        if target_count != 1:
            raise ValueError("planning scene must contain its target exactly once")
        fixed = list(self._obstacles)
        snapshot = self.scene_builder.build(objects, fixed_bodies=fixed)
        entity_ids = [int(value.id) for value in fixed]
        entity_ids.extend(
            self.scene_builder.entity_id(value.object_index) for value in objects
        )
        return self.core.PhaseScene(
            snapshot,
            entity_ids,
            self.scene_builder.entity_id(target_index),
        )

    def _build_obstacles(self):
        bodies = []
        cube = np.asarray(
            [
                [x, y, z]
                for x in (-0.5, 0.5)
                for y in (-0.5, 0.5)
                for z in (-0.5, 0.5)
            ],
            dtype=float,
        )
        for offset, obstacle in enumerate(self.planning.obstacles):
            model = self.core.BodyModel.dsf(
                self._fixed_model_start + offset,
                cube * obstacle.size,
                mass=1.0,
                inertia=np.eye(3),
                friction=self.planning.obstacle_friction,
            )
            pose = _IDENTITY_POSE.copy()
            pose[:3] = obstacle.center
            bodies.append(
                self.core.BodyInstance(
                    self._obstacle_entity_start + offset,
                    model,
                    pose,
                    mobility=self.core.Mobility.STATIC,
                )
            )
        return tuple(bodies)

    def _build_robot(self):
        profile = self.planning.robot
        model = self.robot_urdf.kinematics
        if model.dof_count != profile.home_positions.size:
            raise ValueError("robot home positions do not match the URDF DoF count")
        tool_link = self._link(model, profile.tool_link, "robot tool")
        state = self.core.KinematicState(self.planning.frame_id, model)
        state.positions = profile.home_positions.copy()
        links = [
            self._link(model, name, "robot collision")
            for name in _ROBOT_COLLISION_LINKS
        ]
        bodies = [
            self.core.MotionLinkBody(
                link,
                self._robot_entity_start + index,
                self.robot_urdf.body_model(link),
            )
            for index, link in enumerate(links)
        ]
        pairs = [
            self.core.CollisionBodyPair(
                self._robot_entity_start + first,
                self._robot_entity_start + second,
            )
            for first in range(len(links))
            for second in range(first + 2, len(links))
        ]
        return self.core.MotionRobot(
            state,
            tool_link,
            profile.link_from_tool.copy(),
            gripper_opening=profile.gripper_opening,
            collision_bodies=bodies,
            self_collision_pairs=pairs,
        )

    def _build_gripper(self):
        profile = self.planning.gripper
        model = self.gripper_urdf.kinematics
        if model.dof_count != profile.opening_offset.size:
            raise ValueError("gripper opening map does not match its URDF DoF count")
        root = self._link(model, profile.root_link, "gripper root")
        if root not in model.roots:
            raise ValueError("configured gripper root link is not a URDF root")
        state = self.core.KinematicState(self.planning.frame_id, model)
        state.positions = (
            profile.opening_offset
            + self.planning.robot.gripper_opening * profile.opening_direction
        )
        links = {
            name: self._link(model, name, "gripper collision")
            for name in _GRIPPER_COLLISION_LINKS
        }
        entity_by_name = {
            name: self._gripper_entity_start + index
            for index, name in enumerate(_GRIPPER_COLLISION_LINKS)
        }
        bodies = [
            self.core.GraspLinkBody(
                links[name],
                entity_by_name[name],
                self.gripper_urdf.body_model(links[name]),
            )
            for name in _GRIPPER_COLLISION_LINKS
        ]
        contacts = [
            self.core.GraspContactGeometry(
                entity_by_name["left_finger_tip"],
                3,
                self.core.GraspContactSide.LEFT,
                self.core.GraspContactType.PAD,
                order=0,
            ),
            self.core.GraspContactGeometry(
                entity_by_name["right_finger_tip"],
                6,
                self.core.GraspContactSide.RIGHT,
                self.core.GraspContactType.PAD,
                order=0,
            ),
            self.core.GraspContactGeometry(
                entity_by_name["right_finger_tip"],
                7,
                self.core.GraspContactSide.RIGHT,
                self.core.GraspContactType.PAD,
                order=1,
            ),
        ]
        return self.core.GripperModel(
            state,
            root,
            profile.grasp_from_root.copy(),
            profile.opening_offset.copy(),
            profile.opening_direction.copy(),
            opening_lower=profile.opening_lower,
            opening_upper=profile.opening_upper,
            collision_bodies=bodies,
            contact_geometries=contacts,
        )

    def _load_urdf(self, relative_path: Path):
        path = relative_path
        if not path.is_absolute():
            path = self.repository_root / path
        if not path.is_file():
            raise FileNotFoundError(f"missing planning URDF: {path}")
        return self.core.load_urdf_model(str(path))

    @staticmethod
    def _link(model, name: str, role: str) -> int:
        link = model.find_link(name)
        if link is None:
            raise ValueError(f"{role} link is missing from its URDF: {name}")
        return int(link)
