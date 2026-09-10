"""URDF forward-kinematic mesh projection for recorded motion replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from stacking_common.planning import compose_poses

from stacking_common.profiles.planning import PlanningProfile


@dataclass(frozen=True)
class RobotGeometryPose:
    """One URDF visual or collision mesh in the planning world frame."""

    part_id: str
    mesh_path: Path
    pose: np.ndarray
    color: str
    geometry_kind: str = "visual"
    mesh_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        part_id = str(self.part_id).strip()
        path = Path(self.mesh_path)
        pose = np.asarray(self.pose, dtype=float)
        color = str(self.color).strip()
        if not part_id or not color:
            raise ValueError("robot geometry ID and color are required")
        if not path.is_file():
            raise FileNotFoundError(f"missing robot geometry: {path}")
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError("robot geometry pose must be a finite pose")
        if np.linalg.norm(pose[3:]) <= 1e-12:
            raise ValueError("robot geometry quaternion must be nonzero")
        if self.geometry_kind not in {"visual", "collision"}:
            raise ValueError("robot geometry must be visual or collision")
        scale = tuple(float(value) for value in self.mesh_scale)
        if len(scale) != 3 or not np.all(np.isfinite(scale)) or min(scale) <= 0:
            raise ValueError("robot mesh scale must contain three positive values")
        object.__setattr__(self, "mesh_scale", scale)
        owned_pose = pose.copy()
        owned_pose.setflags(write=False)
        object.__setattr__(self, "part_id", part_id)
        object.__setattr__(self, "mesh_path", path.resolve())
        object.__setattr__(self, "pose", owned_pose)
        object.__setattr__(self, "color", color)


@dataclass(frozen=True)
class _GeometrySpec:
    part_id: str
    link: int
    mesh_path: Path
    link_T_geometry: np.ndarray
    color: str
    geometry_kind: str
    mesh_scale: tuple[float, float, float]


class RobotTrajectoryProjector:
    """Project one normalized motion sample into renderable robot parts."""

    def __init__(
        self,
        planning: PlanningProfile,
        repository_root: str | Path,
        *,
        robot_links,
        gripper_links,
        robot_colors,
        gripper_color="#161616",
        core_module=None,
    ) -> None:
        if core_module is None:
            try:
                import stacking_core as core_module
            except ImportError as error:
                raise RuntimeError(
                    "stacking-core kinematics bindings are required for "
                    "robot trajectory replay"
                ) from error
        self.core = core_module
        self.planning = planning
        self.repository_root = Path(repository_root).resolve()
        robot_path = self._resolve(planning.robot.urdf)
        gripper_path = self._resolve(planning.gripper.urdf)
        self.robot_urdf = core_module.load_urdf_model(str(robot_path))
        self.gripper_urdf = core_module.load_urdf_model(str(gripper_path))
        self.robot_state = core_module.KinematicState(
            planning.frame_id,
            self.robot_urdf.kinematics,
        )
        self.gripper_state = core_module.KinematicState(
            planning.frame_id,
            self.gripper_urdf.kinematics,
        )
        self.tool_link = self._link(
            self.robot_urdf.kinematics,
            planning.robot.tool_link,
        )
        self.gripper_root = self._link(
            self.gripper_urdf.kinematics,
            planning.gripper.root_link,
        )
        self.robot_geometry = self._geometry_specs(
            robot_path,
            self.robot_urdf.kinematics,
            robot_links,
            lambda name: robot_colors.get(name, "#b8bdc4"),
            "robot",
        )
        self.gripper_geometry = self._geometry_specs(
            gripper_path,
            self.gripper_urdf.kinematics,
            gripper_links,
            lambda _name: gripper_color,
            "gripper",
        )

    def project(
        self,
        positions: np.ndarray,
        opening: float | None,
        *,
        geometry_mode: str = "visual",
    ) -> tuple[RobotGeometryPose, ...]:
        if geometry_mode not in {"visual", "collision", "visual_collision"}:
            raise ValueError(f"unknown robot geometry mode: {geometry_mode}")
        resolved_opening = (
            self.planning.robot.gripper_opening
            if opening is None
            else float(opening)
        )
        if not np.isfinite(resolved_opening):
            raise ValueError("robot trajectory gripper opening must be finite")
        tool_pose = self.tool_pose(positions)
        root_pose = compose_poses(
            tool_pose,
            self.planning.gripper.grasp_from_root,
        )
        self.gripper_state.set_frame_from_root(
            self.gripper_root,
            root_pose,
        )
        self.gripper_state.positions = (
            self.planning.gripper.opening_offset
            + resolved_opening * self.planning.gripper.opening_direction
        )
        return tuple(
            self._project_spec(self.robot_state, spec)
            for spec in self.robot_geometry
            if geometry_mode == "visual_collision" or spec.geometry_kind == geometry_mode
        ) + tuple(
            self._project_spec(self.gripper_state, spec)
            for spec in self.gripper_geometry
            if geometry_mode == "visual_collision" or spec.geometry_kind == geometry_mode
        )

    def tool_pose(self, positions: np.ndarray) -> np.ndarray:
        """Return the planning grasp frame using the same FK as the meshes."""
        positions = np.asarray(positions, dtype=float)
        expected = self.robot_urdf.kinematics.dof_count
        if positions.shape != (expected,) or not np.all(np.isfinite(positions)):
            raise ValueError(
                f"robot trajectory sample must contain {expected} finite joints"
            )
        self.robot_state.positions = positions
        return compose_poses(
            self.robot_state.link_pose(self.tool_link), self.planning.robot.link_from_tool)

    def _project_spec(self, state, spec: _GeometrySpec) -> RobotGeometryPose:
        return RobotGeometryPose(
            spec.part_id,
            spec.mesh_path,
            compose_poses(state.link_pose(spec.link), spec.link_T_geometry),
            spec.color,
            spec.geometry_kind,
            spec.mesh_scale,
        )

    def _geometry_specs(
        self,
        urdf_path: Path,
        model,
        link_names,
        color,
        group: str,
    ) -> tuple[_GeometrySpec, ...]:
        root = ET.parse(urdf_path).getroot()
        links = {value.attrib.get("name"): value for value in root.findall("link")}
        specs = []
        for link_name in link_names:
            link_element = links.get(link_name)
            if link_element is None:
                raise ValueError(f"trajectory URDF is missing link: {link_name}")
            link_id = self._link(model, link_name)
            for kind in ("visual", "collision"):
                for index, element in enumerate(link_element.findall(kind)):
                    geometry = element.find("geometry")
                    if geometry is None:
                        continue
                    mesh = geometry.find("mesh")
                    if mesh is None and kind == "collision":
                        mesh = geometry.find("dsf_vert")
                    if mesh is None or not mesh.attrib.get("filename"):
                        raise ValueError(f"unsupported {kind} geometry for {link_name}")
                    scale = tuple(float(value) for value in mesh.get("scale", "1 1 1").split())
                    specs.append(
                        _GeometrySpec(
                            f"{group}/{link_name}/{kind}/{index}",
                            link_id,
                            self._mesh_path(urdf_path, mesh.attrib["filename"]),
                            _origin_pose(element.find("origin")),
                            color(link_name),
                            kind,
                            scale,
                        )
                    )
        if not specs:
            raise ValueError(f"trajectory URDF has no {group} geometry")
        return tuple(specs)

    def _mesh_path(self, urdf_path: Path, filename: str) -> Path:
        repository_relative = self.repository_root / filename
        path = (
            repository_relative
            if repository_relative.is_file()
            else urdf_path.parent / filename
        )
        if not path.is_file():
            raise FileNotFoundError(f"missing trajectory mesh: {path}")
        return path.resolve()

    def _resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self.repository_root / path

    @staticmethod
    def _link(model, name: str) -> int:
        link = model.find_link(name)
        if link is None:
            raise ValueError(f"trajectory URDF is missing link: {name}")
        return int(link)


def _origin_pose(element) -> np.ndarray:
    xyz = _vector_attribute(element, "xyz")
    rpy = _vector_attribute(element, "rpy")
    roll, pitch, yaw = 0.5 * rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    quaternion = np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )
    return np.concatenate((xyz, quaternion))


def _vector_attribute(element, name: str) -> np.ndarray:
    if element is None or name not in element.attrib:
        return np.zeros(3)
    values = np.fromstring(element.attrib[name], sep=" ", dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"URDF origin {name} must contain three finite values")
    return values
