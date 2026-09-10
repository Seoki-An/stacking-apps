"""Portable restart state at a committed planning-step boundary."""

from dataclasses import asdict, dataclass
from collections.abc import Mapping

from stacking_common.domain import ContactPoint, StackObjectTrajectory, State
from stacking_common.execution.values import plain_json, positive_int, sha256_text
from stacking_common.planning import FrameTransform, PickSceneSnapshot

from .action import decode_action, encode_action


@dataclass(frozen=True)
class PlanningCheckpoint:
    """Restart planning with a fresh tree and a rebuilt observation.

    This stores domain state, pickup provenance, frame conversion, and the
    context hash. Native workers, search trees, and RNG internals are not
    serialized; restarting is not promised to reproduce the same search path.
    This planned state is not evidence of physical robot execution.
    """

    step: int
    context_sha256: str
    state: State
    pick_scene: PickSceneSnapshot
    world_T_stack: FrameTransform

    def __post_init__(self) -> None:
        positive_int(self.step, "checkpoint step")
        sha256_text(self.context_sha256, "checkpoint context hash")
        if not isinstance(self.state, State):
            raise TypeError("checkpoint requires a State")
        if not isinstance(self.pick_scene, PickSceneSnapshot):
            raise TypeError("checkpoint requires a PickSceneSnapshot")
        if not isinstance(self.world_T_stack, FrameTransform):
            raise TypeError("checkpoint requires a FrameTransform")
        if self.pick_scene.frame_id != self.world_T_stack.parent_frame:
            raise ValueError("checkpoint pickup and world frames differ")
        if set(self.pick_scene.object_ids) != set(self.state.object_ids):
            raise ValueError("checkpoint pickup scene must cover the full inventory")
        missing = {
            self.state.object_ids[index] for index in self.state.placed_object_indices
        } - set(self.state.object_poses)
        if missing:
            raise ValueError("checkpoint is missing placed-object poses")
        object.__setattr__(self, "state", self.state.copy())
        object.__setattr__(self, "pick_scene", PickSceneSnapshot(
            self.pick_scene.frame_id, self.pick_scene.object_poses,
            self.pick_scene.source, dict(self.pick_scene.metadata),
        ))


def encode_planning_checkpoint(value: PlanningCheckpoint) -> dict[str, object]:
    state = value.state
    return plain_json({
        "schema_version": 1, "step": value.step, "context_sha256": value.context_sha256,
        "world_T_stack": {
            "parent_frame": value.world_T_stack.parent_frame,
            "child_frame": value.world_T_stack.child_frame,
            "parent_T_child": value.world_T_stack.parent_T_child,
        },
        "pick_scene": {
            "frame_id": value.pick_scene.frame_id, "source": value.pick_scene.source,
            "metadata": dict(value.pick_scene.metadata),
            "object_poses": {str(key): pose for key, pose in value.pick_scene.object_poses.items()},
        },
        "state": {
            "object_ids": state.object_ids, "placed_object_indices": state.placed_object_indices,
            "object_poses": {str(key): pose for key, pose in state.object_poses.items()},
            "trajectories": [
                {str(key): asdict(trajectory) for key, trajectory in stage.items()}
                for stage in state.trajectories
            ],
            "action_history": [encode_action(action) for action in state.action_history],
            "terminated": state.terminated, "failed": state.failed,
            "simulation_settled": state.simulation_settled,
            "contact_points": [asdict(contact) for contact in state.contact_points],
            "pose_identified_object_ids": sorted(state.pose_identified_object_ids),
        },
    })


def decode_planning_checkpoint(values: Mapping[str, object]) -> PlanningCheckpoint:
    if values.get("schema_version") != 1:
        raise ValueError("unsupported planning checkpoint schema version")
    state = dict(values["state"])
    state["action_history"] = [decode_action(action) for action in state["action_history"]]
    state["contact_points"] = [ContactPoint(**item) for item in state["contact_points"]]
    state["trajectories"] = [
        {int(key): StackObjectTrajectory(**item) for key, item in stage.items()}
        for stage in state["trajectories"]
    ]
    return PlanningCheckpoint(
        values["step"], values["context_sha256"], State(**state),
        PickSceneSnapshot(**values["pick_scene"]), FrameTransform(**values["world_T_stack"]),
    )
