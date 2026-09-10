"""Deterministic commissioned catalog and stone_grab surface geometry."""

from pathlib import Path
import json
import numpy as np

from stacking_common.pointcloud.types import PointCloudData
from stacking_common.pointcloud.processing import sample_mesh_surface
from stacking_common.profiles.planning import load_planning_profile
from stacking_excavator.planning.trajectory import ExcavatorRobotTrajectoryProjector, pose_matrix
from stacking_excavator.acquisition.storage import file_hash


def sample_surface(mesh, count=12000):
    return sample_mesh_surface(np.asarray(mesh.vertices), np.asarray(mesh.triangles), count)


class ExcavatorModels:
    def __init__(self, repository):
        self.root = Path(repository)
        self._stones = {}
        self._gripper = {}
        self.catalog = json.loads((self.root / "apps/excavator/assets/perception/provenance.json").read_text())["models"]
        self.profile = load_planning_profile(self.root / "apps/excavator/config/planning.yml")
        self.projector = ExcavatorRobotTrajectoryProjector(self.profile, self.root)

    def stone(self, object_id):
        if object_id not in self._stones:
            path = self.root / f"apps/excavator/assets/perception/model_{object_id}.npz"
            sha = file_hash(path)
            if sha != self.catalog[str(object_id)]["sha256"]:
                raise ValueError("perception catalog differs from its recorded provenance")
            with np.load(path, allow_pickle=False) as values:
                self._stones[object_id] = (PointCloudData(values["xyz"]), sha)
        return self._stones[object_id]

    def gripper(self, opening):
        import open3d as o3d

        opening = float(opening)
        if opening not in self._gripper:
            parts = [part for part in self.projector.project(np.array(self.profile.robot.home_positions), opening)
                     if part.part_id.startswith("gripper/")]
            body = next(part for part in parts if part.part_id.startswith("gripper/grip_body/"))
            geometry_T_world = np.linalg.inv(pose_matrix(body.pose))
            clouds = []
            for part in parts:
                mesh = o3d.io.read_triangle_mesh(str(part.mesh_path))
                mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * part.mesh_scale)
                mesh.transform(geometry_T_world @ pose_matrix(part.pose))
                clouds.append(sample_surface(mesh, 10000).points)
            self._gripper.clear()
            self._gripper[opening] = PointCloudData(np.concatenate(clouds))
        return self._gripper[opening]
