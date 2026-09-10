"""Local virtual NUC: measured joint device and an independent kinematic world.

World poses drive sensor rendering, never perception answers. This emulator
models rigid grasp attachment and release, not contact dynamics or hardware.
"""
import json
from pathlib import Path
import time
import numpy as np

from stacking_excavator.acquisition.storage import atomic_json
from stacking_excavator.planning.trajectory import ExcavatorRobotTrajectoryProjector, pose_matrix
from .simulation import SimulatedDevice


class VirtualNuc(SimulatedDevice):
    def __init__(self, repository, directory, *, clock=time.monotonic):
        super().__init__(repository, clock=clock)
        self.repository = Path(repository).resolve()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.world_path = self.directory / 'world.json'
        self.world = None
        self.projector = None
        self.missing_stone = False
        self.slip = np.zeros(3)
        if self.world_path.exists():
            self.world = json.loads(self.world_path.read_text())
            self.positions = np.array(self.world['joints'])
            self.opening = self.world['opening']
            self.stop()

    def configure(self, record, profile):
        if not self.stopped:
            raise ValueError('stop virtual NUC before loading a world')
        self.projector = ExcavatorRobotTrajectoryProjector(profile, self.repository)
        if self.world is None:
            poses = {str(o.object_id): pose_matrix(o.pose).tolist()
                     for o in (*record.problem.pick_scene_objects, *record.problem.place_scene_objects)}
            poses[str(record.problem.action.object_id)] = pose_matrix(record.problem.pick_pose).tolist()
            self.positions = record.result.segments[0].positions[0].copy()
            openings = record.result.segments[0].openings
            self.opening = 0. if openings is None else float(openings[0])
            self.stop()
            self.world = {'mode': 'kinematic_emulation', 'poses': poses,
                          'attached_id': None, 'gripper_T_object': None}
            self.save_world()

    def gripper_pose(self, positions=None, opening=None):
        parts = self.projector.project(self.positions if positions is None else positions,
                                       self.opening if opening is None else opening)
        body = next(p for p in parts if p.part_id.startswith('gripper/grip_body/'))
        return pose_matrix(body.pose)

    def advance(self):
        super().advance()
        if self.world is not None and self.projector is not None and self.world['attached_id'] is not None:
            pose = self.gripper_pose() @ np.array(self.world['gripper_T_object'])
            self.world['poses'][str(self.world['attached_id'])] = pose.tolist()

    def grasp(self, event, object_id):
        if self.world is None:
            raise ValueError('virtual world is not initialized')
        if event == 'acquire':
            if self.world['attached_id'] is not None:
                raise ValueError('virtual gripper already holds a stone')
            if not self.missing_stone:
                relative = np.linalg.inv(self.gripper_pose()) @ np.array(self.world['poses'][str(object_id)])
                relative[:3, 3] += self.slip
                self.world.update(attached_id=object_id, gripper_T_object=relative.tolist())
                self.advance()
        else:
            self.advance()
            self.world.update(attached_id=None, gripper_T_object=None)
        self.save_world()

    def save_world(self):
        if self.world is not None:
            self.world.update(joints=self.positions.tolist(), opening=float(self.opening))
            atomic_json(self.world_path, self.world)

    def capture_snapshot(self):
        if not self.stopped or self.projector is None:
            raise ValueError('capture requires a stopped, initialized virtual NUC')
        self.advance()
        self.save_world()
        parts = self.projector.project(self.positions, self.opening)
        return {'world': json.loads(json.dumps(self.world)),
                'world_T_gripper': self.gripper_pose().tolist(),
                'timestamp_ns': time.time_ns(),
                'robot_parts': [{'path': str(p.mesh_path), 'scale': list(p.mesh_scale),
                                 'pose': pose_matrix(p.pose).tolist()} for p in parts]}
