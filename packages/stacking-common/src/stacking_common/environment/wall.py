"""Optional wall reward layered onto the existing placement reward."""

from .reward import StackingReward
from ._geometry import exposed_surface_normal, quaternion_matrix
from stacking_common.wall import WallSurfaces
import numpy as np


class WallReward(StackingReward):
    def __init__(self, *args, construction, grid_shape, **kwargs):
        super().__init__(*args, **kwargs)
        self.wall = WallSurfaces(self.inventory, construction, grid_shape)

    def compute(self, state):
        # State-only inspection reports absolute progress, never invents a parent.
        reward, info = super().compute(state)
        info.update(self.wall.diagnostics(state))
        index = state.placed_object_indices[-1]
        pose = state.object_poses[state.object_ids[index]]
        quality = self.wall.quality(self.wall.object_envelope(index, pose))
        sides = np.any(quality > 0, axis=(1, 2))
        info["wall_settled_sides"] = [name for name, present in
            zip(("negative_x", "positive_x"), sides) if present]
        if not state.failed and not sides.any():
            info["rejection_reason"] = "wall_settled_surface_miss"
        return reward, info

    def compute_transition(self, parent, state):
        reward, info = self.compute(state)
        before = self.wall.progress(self.wall.envelope(parent))
        gain = info["wall_surface_progress"] - before
        contribution = self.wall.config.reward_weight * gain
        info.update(wall_previous_surface_progress=before,
                    wall_surface_increment=gain, wall_surface_reward=contribution)
        return reward + contribution, info

    def _inward_orientation_score(self, object_index, pose):
        normals, areas = self.inventory.objects[object_index].representative_face_normals()
        normal = exposed_surface_normal(np.asarray(normals) @ quaternion_matrix(pose[3:]).T, areas)
        if normal is None:
            return 0.
        upward = float(np.clip((normal[2] - .7) / .3, 0., 1.))
        heading = normal[:2]
        if np.linalg.norm(heading) < .05:
            return upward * .5
        area = np.sum(self.wall.quality(self.wall.object_envelope(object_index, pose)), axis=(1, 2))
        if area.sum() == 0:
            return 0.
        minimum = np.cos(np.deg2rad(40.))
        # The horizontal inward normals are +X and -X, independent of Y.
        alignment = np.array([heading[0], -heading[0]]) / np.linalg.norm(heading)
        scores = np.clip((alignment - minimum) / (1 - minimum), 0, 1)
        return upward * float(np.dot(area, scores) / area.sum())
