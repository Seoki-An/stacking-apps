"""Projection of MCTS state into backend-neutral debug values."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from stacking_common.domain import ContactPoint, State
from stacking_common.search.mcts import MCTSNode
from .tree import MCTSTreeNode, project_tree, validate_tree

from .contracts import (
    CameraView,
    ContactMarker,
    ForceMarker,
    PoseTrack,
    SceneObjectPose,
    SceneSnapshot,
    SceneTimeline,
    ScoreField,
    CompositeTargetVolume,
    TargetDisplay,
)


class CandidateOutcome(str, Enum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class CandidateRenderMode(str, Enum):
    ALL = "all"
    ALL_OPAQUE = "all_opaque"
    SELECTED = "selected"
    STATE_ONLY = "no_select"


class CandidatePoseMode(str, Enum):
    INITIAL = "initial"
    SOLVED = "solved"
    ACTION = "action"
    INITIAL_AND_SOLVED = "initial_and_solved"
    SIMULATION_START = "simulation_start"
    FINAL = "final"
    SIMULATION_START_AND_FINAL = "simulation_start_and_final"
    TRAJECTORY = "trajectory"
    BEST_SEQUENCE = "best_sequence"


@dataclass(frozen=True)
class SceneMotionDiagnostic:
    """Motion of an already placed object that invalidated a candidate."""

    object_id: int
    parent_pose: np.ndarray
    failed_final_pose: np.ndarray
    trajectory: PoseTrack

    def __post_init__(self) -> None:
        object_id = int(self.object_id)
        if object_id < 0 or self.trajectory.object_id != object_id:
            raise ValueError("scene-motion object and trajectory must match")
        parent_pose = _pose(self.parent_pose, "scene-motion parent pose")
        final_pose = _pose(
            self.failed_final_pose, "scene-motion failed final pose"
        )
        if not np.allclose(parent_pose, self.trajectory.poses[0]):
            raise ValueError("scene-motion trajectory must begin at the parent pose")
        if not np.allclose(final_pose, self.trajectory.poses[-1]):
            raise ValueError("scene-motion trajectory must end at the failed pose")
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "parent_pose", parent_pose)
        object.__setattr__(self, "failed_final_pose", final_pose)


@dataclass(frozen=True)
class MCTSCandidateDiagnostic:
    """One admitted MCTS candidate and all currently retained debug data."""

    candidate_id: str
    rank: int
    object_index: int
    object_id: int
    outcome: CandidateOutcome
    selected: bool
    init_pose: np.ndarray
    solved_pose: np.ndarray
    action_pose: np.ndarray
    final_pose: np.ndarray | None = None
    rejection_reason: str | None = None
    score: float | None = None
    prior: float = 0.0
    visits: int = 0
    mean_value: float = 0.0
    reward: float | None = None
    score_field_id: str | None = None
    timeline: SceneTimeline = field(default_factory=lambda: SceneTimeline(()))
    simulation_contacts: tuple[ContactMarker, ...] = ()
    pose_solve_forces: tuple[ForceMarker, ...] = ()
    best_sequence: tuple[SceneObjectPose, ...] = ()
    scene_motion: SceneMotionDiagnostic | None = None

    def __post_init__(self) -> None:
        candidate_id = str(self.candidate_id).strip()
        rank = int(self.rank)
        object_index = int(self.object_index)
        object_id = int(self.object_id)
        outcome = CandidateOutcome(self.outcome)
        rejection_reason = self.rejection_reason
        if not candidate_id:
            raise ValueError("candidate ID is required")
        if rank <= 0:
            raise ValueError("candidate rank must be positive")
        if object_index < 0 or object_id < 0:
            raise ValueError("candidate object identity must be non-negative")
        if outcome is CandidateOutcome.FAILED and not rejection_reason:
            raise ValueError("failed candidate requires a rejection reason")
        if rejection_reason is not None:
            rejection_reason = str(rejection_reason).strip()
            if not rejection_reason:
                raise ValueError("candidate rejection reason must not be empty")
            if outcome is not CandidateOutcome.FAILED:
                raise ValueError("only failed candidates may have a rejection reason")
        for name in ("score", "reward"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"candidate {name} must be finite")
            if value is not None:
                object.__setattr__(self, name, float(value))
        for name in ("prior", "mean_value"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"candidate {name} must be finite")
            object.__setattr__(self, name, value)
        visits = int(self.visits)
        if visits < 0:
            raise ValueError("candidate visits must be non-negative")
        score_field_id = (
            None
            if self.score_field_id is None
            else str(self.score_field_id).strip()
        )
        if score_field_id == "":
            raise ValueError("candidate score-field ID must not be empty")
        if not isinstance(self.timeline, SceneTimeline):
            raise TypeError("candidate timeline must be a SceneTimeline")
        simulation_contacts = tuple(self.simulation_contacts)
        pose_solve_forces = tuple(self.pose_solve_forces)
        best_sequence = tuple(self.best_sequence)
        if not all(isinstance(value, ContactMarker) for value in simulation_contacts):
            raise TypeError("simulation contacts must be ContactMarker values")
        if not all(isinstance(value, ForceMarker) for value in pose_solve_forces):
            raise TypeError("pose-solve forces must be ForceMarker values")
        if not all(isinstance(value, SceneObjectPose) for value in best_sequence):
            raise TypeError("best sequence must contain SceneObjectPose values")
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "object_index", object_index)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "selected", bool(self.selected))
        object.__setattr__(self, "init_pose", _pose(self.init_pose, "initial pose"))
        object.__setattr__(self, "solved_pose", _pose(self.solved_pose, "solved pose"))
        object.__setattr__(self, "action_pose", _pose(self.action_pose, "action pose"))
        object.__setattr__(
            self,
            "final_pose",
            None if self.final_pose is None else _pose(self.final_pose, "final pose"),
        )
        object.__setattr__(self, "rejection_reason", rejection_reason)
        object.__setattr__(self, "visits", visits)
        object.__setattr__(self, "score_field_id", score_field_id)
        object.__setattr__(self, "simulation_contacts", simulation_contacts)
        object.__setattr__(self, "pose_solve_forces", pose_solve_forces)
        object.__setattr__(self, "best_sequence", best_sequence)

    @property
    def validated(self) -> bool:
        return self.outcome is not CandidateOutcome.PENDING

    @property
    def failed(self) -> bool:
        return self.outcome is CandidateOutcome.FAILED

    def pose(self, mode: CandidatePoseMode) -> np.ndarray | None:
        """Resolve an explicit pose mode without the legacy settled ambiguity."""

        mode = CandidatePoseMode(mode)
        if mode is CandidatePoseMode.INITIAL:
            return self.init_pose.copy()
        if mode is CandidatePoseMode.SOLVED:
            return self.solved_pose.copy()
        if mode is CandidatePoseMode.ACTION:
            return self.action_pose.copy()
        if mode is CandidatePoseMode.FINAL:
            return None if self.final_pose is None else self.final_pose.copy()
        if mode in {
            CandidatePoseMode.INITIAL_AND_SOLVED,
            CandidatePoseMode.SIMULATION_START_AND_FINAL,
            CandidatePoseMode.TRAJECTORY,
            CandidatePoseMode.BEST_SEQUENCE,
        }:
            raise ValueError(f"{mode.value} is a multi-object display mode")
        try:
            return self.timeline.track(self.object_id).poses[0].copy()
        except KeyError:
            return None


@dataclass(frozen=True)
class MCTSDebugStep:
    """A complete viewer state for one planning step."""

    step: int
    scene: SceneSnapshot
    candidates: tuple[MCTSCandidateDiagnostic, ...]
    selected_candidate_id: str | None
    succeeded: bool
    score_field: ScoreField | None = None
    state_only: bool = False
    failure_reason: str | None = None
    rejected_expansions: Mapping[str, int] = field(default_factory=dict)
    tree_nodes: tuple[MCTSTreeNode, ...] = ()

    def __post_init__(self) -> None:
        step = int(self.step)
        candidates = tuple(self.candidates)
        selected_id = self.selected_candidate_id
        if step <= 0:
            raise ValueError("MCTS debug step must be positive")
        if not isinstance(self.scene, SceneSnapshot):
            raise TypeError("MCTS debug scene must be a SceneSnapshot")
        if not all(isinstance(value, MCTSCandidateDiagnostic) for value in candidates):
            raise TypeError("MCTS debug candidates must be diagnostic values")
        candidate_ids = [value.candidate_id for value in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique within a step")
        selected = [value for value in candidates if value.selected]
        if selected_id is not None:
            selected_id = str(selected_id).strip()
            if selected_id not in candidate_ids:
                raise ValueError("selected candidate ID is not present in the step")
        if len(selected) > 1 or (
            selected and selected[0].candidate_id != selected_id
        ):
            raise ValueError("selected candidate identity is inconsistent")
        if selected_id is not None and not selected:
            raise ValueError("selected candidate must be marked explicitly")
        succeeded = bool(self.succeeded)
        if succeeded and selected_id is None:
            raise ValueError("a successful step requires a selected candidate")
        if succeeded and selected[0].outcome is not CandidateOutcome.SUCCEEDED:
            raise ValueError("a successful step requires a successful candidate")
        if self.score_field is not None:
            if not isinstance(self.score_field, ScoreField):
                raise TypeError("MCTS score field must be a ScoreField")
            if any(
                value.score_field_id not in (None, self.score_field.field_id)
                for value in candidates
            ):
                raise ValueError("candidate references a different score field")
        rejected = {}
        for reason, count in self.rejected_expansions.items():
            reason = str(reason).strip()
            count = int(count)
            if not reason or count <= 0:
                raise ValueError("rejected-expansion counts must be named and positive")
            rejected[reason] = count
        failure_reason = self.failure_reason
        if failure_reason is not None:
            failure_reason = str(failure_reason).strip()
            if not failure_reason or succeeded:
                raise ValueError(
                    "failure reason is valid only for an unsuccessful step"
                )
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "selected_candidate_id", selected_id)
        object.__setattr__(self, "succeeded", succeeded)
        object.__setattr__(self, "state_only", bool(self.state_only))
        object.__setattr__(self, "failure_reason", failure_reason)
        object.__setattr__(self, "rejected_expansions", MappingProxyType(rejected))
        object.__setattr__(self, "tree_nodes", validate_tree(self.tree_nodes))

    def render_candidates(
        self, mode: CandidateRenderMode
    ) -> tuple[MCTSCandidateDiagnostic, ...]:
        """Return candidates in the legacy-compatible drawing order."""

        mode = CandidateRenderMode(mode)
        ranked = tuple(sorted(self.candidates, key=lambda value: value.rank))
        if mode is CandidateRenderMode.STATE_ONLY:
            return ()
        if mode is CandidateRenderMode.SELECTED:
            return tuple(value for value in ranked if value.selected)
        if mode is CandidateRenderMode.ALL_OPAQUE:
            return ranked
        return tuple(value for value in ranked if not value.selected) + tuple(
            value for value in ranked if value.selected
        )


@dataclass(frozen=True)
class MCTSDebugArtifact:
    """Versioned MCTS viewer input independent of a GUI toolkit."""

    frame_id: str
    steps: tuple[MCTSDebugStep, ...]
    target: TargetDisplay | None = None
    camera: CameraView | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        frame_id = str(self.frame_id).strip()
        steps = tuple(self.steps)
        if not frame_id:
            raise ValueError("MCTS debug artifact frame ID is required")
        if int(self.schema_version) not in (1, 2):
            raise ValueError("unsupported MCTS debug artifact schema version")
        if not all(isinstance(value, MCTSDebugStep) for value in steps):
            raise TypeError("MCTS debug artifact steps must be MCTSDebugStep values")
        if any(value.scene.frame_id != frame_id for value in steps):
            raise ValueError("MCTS debug scenes must use the artifact frame")
        step_ids = [value.step for value in steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("MCTS debug step numbers must be unique")
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "schema_version", 2 if isinstance(self.target, CompositeTargetVolume)
                           else int(self.schema_version))


def project_mcts_step(
    root: MCTSNode,
    selected: MCTSNode | None,
    *,
    step: int,
    frame_id: str = "world",
    score_field: ScoreField | None = None,
    state_only: bool = False,
    failure_reason: str | None = None,
) -> MCTSDebugStep:
    """Project one common MCTS root without mutating or extending the tree."""

    if root.state is None:
        raise ValueError("MCTS debug projection requires an evaluated root")
    if selected is not None and selected not in root.children:
        raise ValueError("selected MCTS candidate must be a root child")
    candidates = tuple(
        _project_candidate(
            node,
            root.state,
            step=int(step),
            rank=rank,
            selected=node is selected,
            score_field_id=(None if score_field is None else score_field.field_id),
        )
        for rank, node in enumerate(root.children, start=1)
    )
    selected_id = next(
        (value.candidate_id for value in candidates if value.selected), None
    )
    if failure_reason is None and selected is None and not candidates:
        failure_reason = "no_candidates"
    return MCTSDebugStep(
        step=step,
        scene=_scene_from_state(root.state, frame_id),
        candidates=candidates,
        selected_candidate_id=selected_id,
        succeeded=selected is not None,
        score_field=score_field,
        state_only=state_only,
        failure_reason=failure_reason,
        rejected_expansions=root.rejected_expansions,
        tree_nodes=project_tree(root, selected, candidates),
    )


def _project_candidate(
    node: MCTSNode,
    parent_state: State,
    *,
    step: int,
    rank: int,
    selected: bool,
    score_field_id: str | None,
) -> MCTSCandidateDiagnostic:
    action = node.action
    if action is None:
        raise ValueError("MCTS root child is missing an action")
    candidate_id = str(
        node.info.get("candidate_id", f"step-{step}-candidate-{rank}")
    )
    if node.failed:
        outcome = CandidateOutcome.FAILED
        rejection_reason = node.rejection_reason or str(
            node.info.get("rejection_reason", "evaluation_failed")
        )
    elif node.evaluated:
        outcome = CandidateOutcome.SUCCEEDED
        rejection_reason = None
    else:
        outcome = CandidateOutcome.PENDING
        rejection_reason = None
    final_pose = None
    if node.state is not None:
        final_pose = node.state.object_poses.get(action.object_id)
    timeline = _timeline(node.state)
    return MCTSCandidateDiagnostic(
        candidate_id=candidate_id,
        rank=rank,
        object_index=action.object_index,
        object_id=action.object_id,
        outcome=outcome,
        selected=selected,
        init_pose=action.init_pose,
        solved_pose=action.solved_pose,
        action_pose=action.pose,
        final_pose=final_pose,
        rejection_reason=rejection_reason,
        score=_diagnostic_score(node.info),
        prior=node.prior,
        visits=node.visits,
        mean_value=node.mean_value,
        reward=None if not node.evaluated else node.reward,
        score_field_id=score_field_id,
        timeline=timeline,
        simulation_contacts=_contacts(node.state),
        pose_solve_forces=_forces(action.object_id, node.info),
        best_sequence=_best_sequence(node),
        scene_motion=_scene_motion(parent_state, timeline, node.failed),
    )


def _scene_from_state(state: State, frame_id: str) -> SceneSnapshot:
    objects = []
    for object_index in state.placed_object_indices:
        object_id = state.object_ids[object_index]
        pose = state.object_poses.get(object_id)
        if pose is not None:
            objects.append(SceneObjectPose(object_id, pose, object_index))
    return SceneSnapshot(frame_id, tuple(objects))


def _timeline(state: State | None) -> SceneTimeline:
    if state is None or not state.trajectories:
        return SceneTimeline(())
    return SceneTimeline(
        tuple(
            PoseTrack(object_id, np.asarray(trajectory.poses, dtype=float), trajectory.times)
            for object_id, trajectory in state.trajectories[-1].items()
            if trajectory.poses
        )
    )


def _contacts(state: State | None) -> tuple[ContactMarker, ...]:
    if state is None:
        return ()
    return tuple(_contact_marker(value, state) for value in state.contact_points)


def _contact_marker(contact: ContactPoint, state: State) -> ContactMarker:
    return ContactMarker(
        _contact_object_id(contact.object_index_first, state),
        _contact_object_id(contact.object_index_second, state),
        0.5 * (contact.point_first + contact.point_second),
        contact.normal,
        contact.gap,
    )


def _contact_object_id(index: int | None, state: State) -> int | None:
    return None if index is None else state.object_ids[index]


def _forces(
    object_id: int, diagnostics: Mapping[str, object]
) -> tuple[ForceMarker, ...]:
    points = diagnostics.get("posegen_contact_points")
    normals = diagnostics.get("posegen_contact_normals")
    forces = diagnostics.get("posegen_contact_forces")
    if not all(isinstance(value, Mapping) for value in (points, normals, forces)):
        return ()
    keys = set(points) & set(normals) & set(forces)
    markers = []
    for key in sorted(keys, key=str):
        point_rows = _vector_rows(points[key])
        normal_rows = _vector_rows(normals[key])
        force_rows = _vector_rows(forces[key])
        for point, normal, force in zip(point_rows, normal_rows, force_rows):
            if not all(np.all(np.isfinite(value)) for value in (point, normal, force)):
                continue
            if np.linalg.norm(normal) <= 1e-12:
                continue
            markers.append(
                ForceMarker(
                    object_id=object_id,
                    role="target",
                    point=point,
                    normal=normal,
                    force=force,
                )
            )
    return tuple(markers)


def _vector_rows(values: object) -> tuple[np.ndarray, ...]:
    """Return scalar or batched native Vector3 values as aligned rows."""

    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return ()
    if array.ndim == 1:
        return () if len(array) < 3 else (array[:3].copy(),)
    if array.ndim != 2:
        return ()
    if array.shape[1] >= 3:
        return tuple(row[:3].copy() for row in array)
    if array.shape[0] >= 3:
        return tuple(row[:3].copy() for row in array.T)
    return ()


def _best_sequence(node: MCTSNode) -> tuple[SceneObjectPose, ...]:
    result = []
    current: MCTSNode | None = node
    while current is not None and current.action is not None:
        if current.state is not None:
            pose = current.state.object_poses.get(current.action.object_id)
            if pose is not None:
                result.append(
                    SceneObjectPose(
                        current.action.object_id,
                        pose,
                        current.action.object_index,
                    )
                )
        # Rejected roots can retain ranking diagnostics from an earlier search.
        # Their old continuation is not a feasible path in the current tree.
        if not current.selectable:
            break
        if "deepest_path_child_index" in current.info:
            index = current.info["deepest_path_child_index"]
            if index is None or not 0 <= index < len(current.children):
                break
            child = current.children[index]
            current = child if child.selectable and child.evaluated else None
        else:
            # Trees projected without final ranking retain the historical view.
            children = current.evaluated_children()
            current = max(children, key=lambda value: value.mean_value, default=None)
    return tuple(result)


def _scene_motion(
    parent: State,
    timeline: SceneTimeline,
    failed: bool,
) -> SceneMotionDiagnostic | None:
    if not failed:
        return None
    placed_ids = {parent.object_ids[index] for index in parent.placed_object_indices}
    moved = []
    for track in timeline.tracks:
        parent_pose = parent.object_poses.get(track.object_id)
        if track.object_id not in placed_ids or parent_pose is None:
            continue
        displacement = float(np.linalg.norm(track.poses[-1, :3] - parent_pose[:3]))
        if displacement > 1e-12:
            moved.append((displacement, track, parent_pose))
    if not moved:
        return None
    _, track, parent_pose = max(moved, key=lambda value: value[0])
    poses = track.poses
    if not np.allclose(poses[0], parent_pose):
        poses = np.vstack((parent_pose, poses))
        track = PoseTrack(track.object_id, poses)
    return SceneMotionDiagnostic(
        track.object_id,
        parent_pose,
        track.poses[-1],
        track,
    )


def _diagnostic_score(values: Mapping[str, object]) -> float | None:
    for name in ("cem_score", "score", "planar_score"):
        value = values.get(name)
        if value is not None and math.isfinite(float(value)):
            return float(value)
    return None


def _pose(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite pose")
    if np.linalg.norm(result[3:]) <= 1e-12:
        raise ValueError(f"{name} quaternion must have non-zero norm")
    result = result.copy()
    result.setflags(write=False)
    return result
