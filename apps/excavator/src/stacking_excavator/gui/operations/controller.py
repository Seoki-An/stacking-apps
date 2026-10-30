"""Recorded-context excavator replay without GUI or hardware dependencies."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from stacking_common.artifacts import StackRunStore
from stacking_common.artifacts.target import target_from_context
from stacking_common.profiles.visualization import SceneBox
from stacking_common.visualization import MCTSDebugStep, SceneSnapshot
from stacking_common.visualization.operations import (
    OperationsController as CommonOperationsController, OperationsFrame, RenderObject,
)
from stacking_excavator.planning.trajectory import ExcavatorRobotTrajectoryProjector
from stacking_excavator.planning.replay import recorded_planning_profile, verify_replay_assets



def recorded_target(context, artifact_target):
    """Display the physical target recorded by runs predating artifact taper."""
    return target_from_context(context, artifact_target)




class OperationsController(CommonOperationsController):
    def __init__(self, repository):
        super().__init__()
        self.repository = Path(repository)
        self.contexts = {}
        self.projectors = {}
        self.mesh_paths = {}
        self._run_path = None
        self._score_inventory = None
        self._score_view = None

    def load(self, path):
        path = Path(path)
        artifact, records, statistics = self._run_reader.load(path)
        metadata = StackRunStore.info(path).metadata
        if metadata.get("platform") != "excavator":
            raise ValueError("Open an excavator .stack run in this console")
        recorded = {context.sha256: context for context in StackRunStore.load_run_contexts(path)}
        if not recorded:
            raise ValueError("Excavator runs require recorded planning and asset context")
        hashes = set(recorded) | {record.context_sha256 for record in records}
        contexts, projectors, paths = {}, {}, {}
        for digest in hashes:
            if digest is None:
                raise ValueError("Excavator replay requires a recorded planning context")
            context = recorded[digest]
            if context.profiles.get("platform") != "excavator":
                raise ValueError("Planning context belongs to another platform")
            planning = recorded_planning_profile(context)
            paths[digest] = verify_replay_assets(self.repository, context, planning)
            contexts[digest] = context
            projectors[digest] = ExcavatorRobotTrajectoryProjector(planning, self.repository)
        # A cancelled run can have a durable attempt before its search step.
        steps = list(artifact.steps)
        known = {step.step for step in steps}
        for record in records:
            if record.step not in known:
                steps.append(MCTSDebugStep(record.step, SceneSnapshot(artifact.frame_id, ()),
                    (), None, False, state_only=True, failure_reason="Incomplete search step"))
                known.add(record.step)
        artifact = replace(artifact, steps=tuple(sorted(steps, key=lambda step: step.step)))
        self.contexts, self.projectors, self.mesh_paths = contexts, projectors, paths
        self.set_artifact(artifact, planning_records=records, search_statistics=statistics)
        self._run_path = path
        self._score_inventory = None
        self._score_view = None
        return artifact

    def initial_pose_score_view(self):
        """Inspect recorded settings and the pre-search scene, independent of YAML."""
        from stacking_common.assets import ObjectAssetLoader
        from stacking_common.domain import State
        from stacking_common.visualization.sampling import initial_pose_score_view

        step, candidate = self.current_step, self.active_candidate
        if step is None or candidate is None or candidate.init_pose is None:
            raise ValueError("Select a candidate with a recorded initial pose")
        if self.current_plan is not None:
            digest = self.current_plan.context_sha256
        elif len(self.contexts) == 1:
            digest = next(iter(self.contexts))
        else:
            digest = StackRunStore.load_planning_checkpoint(
                self._run_path, step=step.step).context_sha256
        context = self.contexts[digest]
        construction = context.profiles['action_generation_profile'].get('construction', {})
        if construction.get('mode', 'volume_fill') != 'volume_fill':
            raise ValueError("This run used a removed construction mode; its score maps are unavailable")
        settings = context.profiles.get('sampling_settings', {})
        if settings.get('strategy') or not settings.get('score') or not (settings.get('floor') or settings.get('floor_fill')):
            raise ValueError("This run does not record the pre-region sampler settings; its score maps are unavailable")
        key = (digest, step.step, candidate.candidate_id)
        if self._score_view is not None and self._score_view[0] == key:
            return self._score_view[1]
        if self._score_inventory is None or self._score_inventory[0] != digest:
            profile = context.profiles['inventory_profile']
            from stacking_common.profiles.inventory import target_profile_from_record
            target = target_profile_from_record(profile['target']).geometry()
            ids = tuple(map(int, profile['object_ids']))
            inventory = ObjectAssetLoader(self.repository / profile['model_directory'],
                profile['density'], target,
                dict(profile['density_overrides'])).load(ids,
                    {oid: np.array([0., 0., 0., 0., 0., 0., 1.]) for oid in ids})
            self._score_inventory = (digest, inventory)
        inventory = self._score_inventory[1]
        poses = {obj.object_id: obj.pose for obj in step.scene.objects}
        state = State(list(inventory.object_ids),
            [inventory.object_ids.index(oid) for oid in poses], poses, [], [])
        environment = context.profiles['environment_profile']
        result = initial_pose_score_view(inventory, state, candidate.object_id, candidate.init_pose,
            planar_config=context.profiles['action_generation_profile']['planar'],
            floor_settings=context.profiles.get('floor_settings'),
            settings=dict(settings),
            height_map_resolution=tuple(environment['observation']['height_map_resolution']),
            height_map_margin=environment['height_map_margin'])
        self._score_view = (key, result)
        return result

    def frame(self, *, trajectory_time=None):
        frame = super().frame(trajectory_time=trajectory_time)
        if self.mesh_paths:
            digest = next(iter(self.mesh_paths))
            frame = replace(frame, object_mesh_paths=self.mesh_paths[digest],
                            target=recorded_target(self.contexts[digest], frame.target))
        return frame

    def planning_frame(self):
        record = self.current_plan
        if record is None:
            return self.frame()
        self.trajectory_projector = self.projectors[record.context_sha256]
        frame = super().planning_frame()
        context = self.contexts[record.context_sha256]
        segment = self.current_plan_segment
        phase = "pick" if segment is None else (segment.scene_phase or "transfer")
        objects = (record.problem.pick_scene_objects if phase == "pick"
                   else record.problem.place_scene_objects)
        placed = tuple(RenderObject(f"plan/scene/{obj.object_id}", obj.object_id, obj.pose, "placed")
                       for obj in objects if obj.object_index != record.problem.action.object_index)
        overlays = tuple(replace(obj, opacity=.18, selected=False) for obj in frame.overlays)
        if segment is not None and segment.target_poses is not None:
            sample = self.plan_sample_index
            current = segment.target_poses[sample]
            following = segment.target_poses[min(sample + 1, len(segment.target_poses) - 1)]
            if current is not None:
                target = current if following is None else interpolate_pose(current, following, self.plan_sample_fraction)
                overlays += (RenderObject("plan/target", record.problem.action.object_id,
                    target, "carried_object", selected=True, color=(1., .72, .12)),)
        if segment is None:
            frame = replace(frame, robot_parts=self.trajectory_projector.project(
                self.trajectory_projector.planning.robot.home_positions, None,
                geometry_mode=self.robot_geometry_mode))
        return replace(frame, placed_objects=placed, overlays=overlays,
            environment_boxes=self._environment(context, phase),
            object_mesh_paths=self.mesh_paths[record.context_sha256],
            target=recorded_target(context, self.artifact.target),
            target_transform=self.trajectory_projector.planning.world_T_stack.parent_T_child)

    @staticmethod
    def _environment(context, phase):
        boxes = []
        ground_phase = "place" if phase == "transfer" else phase
        for item in context.fixed_geometry:
            if item.kind == "plane" and item.name == f"{ground_phase}_ground":
                height = item.pose[2]
                boxes.append(SceneBox(item.name, "ground", [-10., -9., height-.03],
                                      [6., 8., height], (.353, .314, .275), 1.))
            elif item.kind == "box":
                half = np.asarray(item.size) / 2
                boxes.append(SceneBox(item.name, "obstacle", item.pose[:3]-half,
                                      item.pose[:3]+half, (.65, .4, .3), .4))
        return tuple(boxes)


def interpolate_pose(first, second, fraction):
    fraction = float(np.clip(fraction, 0., 1.))
    quaternion = Slerp([0., 1.], Rotation.from_quat([first[3:], second[3:]]))([fraction]).as_quat()[0]
    return np.r_[(1.-fraction)*first[:3] + fraction*second[:3], quaternion]
