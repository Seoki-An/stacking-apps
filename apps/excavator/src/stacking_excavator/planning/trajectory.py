"""Commissioned excavator appearance for common URDF trajectory projection."""

from stacking_common.visualization.robot import RobotTrajectoryProjector
from stacking_common.domain.geometry import quaternion_matrix
import numpy as np


def pose_matrix(pose):
    transform = np.eye(4)
    transform[:3, :3] = quaternion_matrix(np.asarray(pose)[3:])
    transform[:3, 3] = pose[:3]
    return transform


class ExcavatorRobotTrajectoryProjector(RobotTrajectoryProjector):
    def __init__(self, planning, repository_root, *, core_module=None):
        # The separate gripper owns grip_body; avoid drawing its robot-URDF copy.
        links = ("base_link", "cs_cor", "cover_a", "cover_b", "cs_boom",
                 "cs_arm", "cs_bucket", "cs_tilt", "cs_rotate")
        super().__init__(planning, repository_root, robot_links=links,
            gripper_links=("grip_body", "grip_left", "grip_right"),
            robot_colors={name: "#dca927" for name in links}, gripper_color="#454b52",
            core_module=core_module)
