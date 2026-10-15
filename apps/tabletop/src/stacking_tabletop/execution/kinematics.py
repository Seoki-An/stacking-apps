"""Physical tool0 forward kinematics in the UR controller's base frame."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from stacking_common.domain.geometry import quaternion_matrix

from .motion import UR_JOINT_NAMES, validate_pose


class UR5eKinematics:
    """Load the selected URDF; calibration selection belongs to the caller.

    Unlike replay's virtual grasp_frame, controller commands always use tool0.
    Computing base^-1 * tool0 also handles URDFs rooted in ROS base_link.
    One instance belongs to one command worker (native state is mutable).
    """

    def __init__(self, urdf: str | Path, *, core_module=None):
        path = Path(urdf).expanduser().resolve()
        names = tuple(joint.attrib["name"] for joint in ET.parse(path).getroot().findall("joint")
                      if joint.attrib["type"] != "fixed")
        if names != UR_JOINT_NAMES:
            raise ValueError("UR5e URDF must use the canonical six-joint ordering")
        if core_module is None:
            import stacking_core as core_module
        model = core_module.load_urdf_model(str(path))
        self.model = model
        self.state = core_module.KinematicState(1, model.kinematics)
        self.base = model.kinematics.find_link("base")
        self.tool = model.kinematics.find_link("tool0")
        if self.base is None or self.tool is None:
            raise ValueError("UR5e URDF must contain base and physical tool0 links")
        self.urdf_path = path
        joints = [joint for joint in ET.parse(path).getroot().findall('joint') if joint.attrib['type'] != 'fixed']
        self.lower = np.array([float(joint.find('limit').attrib['lower']) for joint in joints])
        self.upper = np.array([float(joint.find('limit').attrib['upper']) for joint in joints])

    def solve(self, target, seed, *, check=lambda: None):
        """Nearby physical-tool IK against the same URDF used for execution."""
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        target = validate_pose(target)
        seed = np.asarray(seed, dtype=float)
        if seed.shape != (6,) or not np.isfinite(seed).all():
            raise ValueError('IK needs six finite seed joints')
        def residual(q):
            check()
            pose = self.pose(q)
            return np.r_[10 * (pose[:3, 3] - target[:3, 3]),
                Rotation.from_matrix(target[:3, :3].T @ pose[:3, :3]).as_rotvec()]
        result = least_squares(residual, seed, bounds=(self.lower, self.upper),
            max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        error = residual(result.x)
        if not result.success or np.linalg.norm(error[:3]) > 1e-4 or np.linalg.norm(error[3:]) > 1e-4:
            raise ValueError('tool inverse kinematics did not converge on the seeded branch '
                f'(position error {np.linalg.norm(error[:3]) * 100:.3f} mm, '
                f'rotation error {np.rad2deg(np.linalg.norm(error[3:])):.4f} deg, '
                f'evaluations {result.nfev})')
        return result.x.copy()

    def pose(self, positions):
        return self._pose(positions, self.tool)

    def grasp_pose(self, positions):
        link = self.model.kinematics.find_link("grasp_frame")
        if link is None:
            raise ValueError("UR5e model has no grasp_frame")
        return self._pose(positions, link)

    def _pose(self, positions, link):
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (6,) or not np.all(np.isfinite(positions)):
            raise ValueError("UR5e FK requires six finite joint positions")
        self.state.positions = positions
        root_T_base = _matrix(self.state.link_pose(self.base))
        root_T_tool = _matrix(self.state.link_pose(link))
        return validate_pose(np.linalg.solve(root_T_base, root_T_tool))


def _matrix(pose):
    result = np.eye(4)
    result[:3, :3] = quaternion_matrix(np.asarray(pose)[3:])
    result[:3, 3] = np.asarray(pose)[:3]
    return result
