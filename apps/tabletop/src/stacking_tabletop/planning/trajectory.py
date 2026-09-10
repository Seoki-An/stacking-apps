"""Tabletop robot appearance policy for common URDF replay."""

from stacking_common.visualization.robot import RobotGeometryPose, RobotTrajectoryProjector

_ROBOT_LINKS = (
    "base_link_inertia",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
)
_GRIPPER_LINKS = (
    "base_link",
    "left_finger_link_1",
    "left_finger_tip",
    "right_finger_link_1",
    "right_finger_tip",
)
_ROBOT_COLORS = {
    "base_link_inertia": "#b8bdc4",
    "shoulder_link": "#167aa3",
    "upper_arm_link": "#b8bdc4",
    "forearm_link": "#b8bdc4",
    "wrist_1_link": "#167aa3",
    "wrist_2_link": "#167aa3",
    "wrist_3_link": "#b8bdc4",
}


class TabletopRobotTrajectoryProjector(RobotTrajectoryProjector):
    def __init__(self, planning, repository_root, *, core_module=None):
        super().__init__(planning, repository_root, robot_links=_ROBOT_LINKS,
                         gripper_links=_GRIPPER_LINKS, robot_colors=_ROBOT_COLORS,
                         core_module=core_module)
