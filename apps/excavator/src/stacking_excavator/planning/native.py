"""Commissioned VDK23/stone-grab models and phase-specific native scenes."""

from dataclasses import replace
import xml.etree.ElementTree as ET

import numpy as np

from stacking_common.planning import CoreSceneBuilder, SceneMobility
from stacking_common.planning.frames import compose_poses, inverse_pose
from stacking_common.planning.native import NativeMotionPlannerAdapter

JOINT_NAMES = ("upper_body_joint", "boom_joint", "arm_joint", "bucket_joint", "tilt_joint", "rotate_joint")
IDENTITY = np.array([0., 0., 0., 0., 0., 0., 1.])


def retry_handoff(result):
    """A fixed pick/place-grasp failure cannot be repaired by moving handoff XY."""
    code = str(result.diagnostics.get("failure_code", ""))
    return (result.diagnostics.get("retryable", False) and "regrasp" in code
            and not any(part in code for part in ("place", "pick", "regrasp_b_grasp_generation")))


class ExcavatorNativePlanningBackend:
    def __init__(self, application, inventory, solver, *, core_module=None):
        if core_module is None:
            import stacking_core as core_module
        self.core = core_module
        self.application = application
        self.inventory = inventory
        self.profile = application.planning_profile
        self.settings = application.excavator_profile
        if len(inventory) >= 1000:
            raise ValueError("inventory entity IDs overlap robot/gripper IDs")
        self.builder = CoreSceneBuilder(inventory.models, frame_id=self.profile.frame_id, core_module=self.core)
        root = application.repository_root
        self.robot_urdf = self.core.load_urdf_model(str(root / self.profile.robot.urdf))
        self.gripper_urdf = self.core.load_urdf_model(str(root / self.profile.gripper.urdf))
        self.robot = self._robot()
        self.gripper = self._gripper()
        self.config = solver.native_config(self.core)
        self.obstacles = self._build_obstacles()

    def _build_obstacles(self):
        cube = np.array([[x, y, z] for x in (-.5, .5)
                         for y in (-.5, .5) for z in (-.5, .5)])
        bodies = []
        for offset, obstacle in enumerate(self.profile.obstacles):
            model = self.core.BodyModel.dsf(110000 + offset, cube * obstacle.size,
                mass=1., inertia=np.eye(3), friction=self.profile.obstacle_friction)
            pose = IDENTITY.copy()
            pose[:3] = obstacle.center
            bodies.append(self.core.BodyInstance(10000 + offset, model, pose,
                                                 mobility=self.core.Mobility.STATIC))
        return tuple(bodies)

    def _robot(self):
        profile = self.profile.robot
        model = self.robot_urdf.kinematics
        if model.dof_count != len(profile.home_positions):
            raise ValueError("robot home coordinates do not match the commissioned model")
        for index, name in enumerate(JOINT_NAMES):
            joint = model.find_joint(name)
            if joint is None or model.dof_index(joint) != index:
                raise ValueError("unexpected excavator joint ordering")
        state = self.core.KinematicState(self.profile.frame_id, model)
        state.positions = profile.home_positions
        tool = model.find_link(profile.tool_link)
        end = model.find_link("cs_rotate")
        if tool is None or end is None:
            raise ValueError("commissioned excavator tool links are missing")
        robot = self.core.MotionRobot(state, tool, profile.link_from_tool,
                                       gripper_opening=profile.gripper_opening)
        # The source IK URDF has visual robot meshes and no robot DSF collisions.
        # Preserve that model boundary; gripper and carried-object collisions
        # are explicitly represented below.
        end_from_tool = compose_poses(inverse_pose(state.link_pose(end)), state.link_pose(tool))
        self.core.configure_excavator_ik(robot, list(JOINT_NAMES), "cs_rotate", end_from_tool)
        return robot

    def _gripper(self):
        profile = self.profile.gripper
        model = self.gripper_urdf.kinematics
        root = model.find_link(profile.root_link)
        if root not in model.roots or model.dof_count != len(profile.opening_offset):
            raise ValueError("gripper root or independent opening coordinates differ from the profile")
        state = self.core.KinematicState(self.profile.frame_id, model)
        state.positions = profile.opening_offset + self.profile.robot.gripper_opening * profile.opening_direction
        path = self.application.repository_root / profile.urdf
        bodies, contacts = [], []
        geometry_id = 0
        for element in ET.parse(path).findall("link"):
            name = element.attrib["name"]
            collisions = element.findall("collision")
            if not collisions:
                continue
            link = model.find_link(name)
            body = self.gripper_urdf.body_model(link)
            entity = 2001 + len(bodies)
            bodies.append(self.core.GraspLinkBody(link, entity, body))
            first = geometry_id
            tooth_order = 0
            for collision in collisions:
                dsf = collision.find("geometry/dsf_vert")
                if dsf is None:
                    raise ValueError("commissioned gripper collision must be DSF geometry")
                asset = path.parent / dsf.attrib["filename"]
                count = sum(line.startswith("o ") for line in asset.read_text().splitlines())
                if not count:
                    raise ValueError("gripper DSF has no component declarations")
                if name in ("grip_left", "grip_right") and dsf.get("contact", "false") == "true":
                    side = self.core.GraspContactSide.LEFT if name == "grip_left" else self.core.GraspContactSide.RIGHT
                    kind = self.core.GraspContactType.TOOTH if collision.get("name") == "teeth" else self.core.GraspContactType.PAD
                    for offset in range(count):
                        contacts.append(self.core.GraspContactGeometry(entity, geometry_id + offset,
                                                                      side, kind, order=tooth_order))
                        if kind == self.core.GraspContactType.TOOTH:
                            tooth_order += 1
                geometry_id += count
            if geometry_id - first != body.geometry_count:
                raise ValueError("gripper contact IDs disagree with native geometry loading")
        if not contacts:
            raise ValueError("gripper has no declared grasp contacts")
        return self.core.GripperModel(state, root, profile.grasp_from_root,
            profile.opening_offset, profile.opening_direction,
            opening_lower=profile.opening_lower, opening_upper=profile.opening_upper,
            collision_bodies=bodies, contact_geometries=contacts)

    def _phase(self, problem, target_pose, objects, ground_height):
        target = problem.action.object_index
        values = tuple(replace(obj, pose=target_pose if obj.object_index == target else obj.pose,
                               mobility=SceneMobility.KINEMATIC if obj.object_index == target else SceneMobility.STATIC)
                       for obj in objects)
        if sum(obj.object_index == target for obj in values) != 1:
            raise ValueError("phase must contain the target exactly once")
        ground = self.core.BodyModel.plane(100000, friction=self.profile.obstacle_friction)
        ground_pose = IDENTITY.copy()
        ground_pose[2] = ground_height
        fixed = self.core.BodyInstance(0, ground, ground_pose, mobility=self.core.Mobility.STATIC)
        scene = self.builder.build(values, fixed_bodies=(fixed, *self.obstacles))
        return self.core.PhaseScene(scene, [0, *(10000 + i for i in range(len(self.obstacles))),
                                           *(self.builder.entity_id(obj.object_index) for obj in values)],
                                    self.builder.entity_id(target))

    def build_problem(self, problem, handoff_position=None):
        if problem.frame_id != self.profile.world_T_stack.parent_frame:
            raise ValueError("planning problem must use the excavator world frame")
        index = problem.action.object_index
        if index >= len(self.inventory) or self.inventory[index].object_id != problem.action.object_id:
            raise ValueError("planning target does not match the inventory")
        handoff = np.asarray(self.settings.handoff_positions[0] if handoff_position is None else handoff_position, dtype=float)
        handoff_pose = problem.pick_pose.copy()
        handoff_pose[:3] = handoff
        pick = self._phase(problem, problem.pick_pose, problem.pick_scene_objects, self.settings.pick_ground_height)
        place = self._phase(problem, problem.place_pose, problem.place_scene_objects, self.settings.place_ground_height)
        transfer = self._phase(problem, handoff_pose, problem.place_scene_objects, self.settings.handoff_ground_height)
        direct = self.core.DirectPlanProblem(pick, place, self.robot, self.gripper)
        return self.core.PickPlaceProblem(direct, transfer, handoff)

    def plan(self, problem):
        attempts = []
        for handoff in self.settings.handoff_positions:
            adapter = NativeMotionPlannerAdapter(lambda request: self.core.solve_pick_place(
                self.build_problem(request, handoff), self.config))
            result = adapter.plan(problem)
            attempts.append({"position": list(handoff), "status": result.status.value,
                             "reason": result.reason, "diagnostics": dict(result.diagnostics)})
            if result.succeeded or not self.config.allow_regrasp or not retry_handoff(result):
                break
        total_elapsed = sum(float(item["diagnostics"].get("elapsed_seconds", 0.)) for item in attempts)
        timings = {}
        for item in attempts:
            for name, value in item["diagnostics"].get("native_timings", {}).items():
                timings[name] = timings.get(name, 0) + value
        return replace(result, diagnostics={**result.diagnostics, "handoff_attempts": attempts,
                       "elapsed_seconds": total_elapsed, "native_timings": timings,
                       "selected_handoff_position": list(handoff) if result.succeeded else None})
