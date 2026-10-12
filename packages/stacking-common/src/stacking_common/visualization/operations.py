"""Qt-independent state controller for the stacking operations console."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path

import numpy as np

from stacking_common.planning import compose_poses, inverse_pose

from stacking_common.artifacts import (
    PlacementPlanRecord,
    StackRunStore,
    decode_legacy_mcts_debug,
    decode_mcts_debug,
)
from stacking_common.visualization import (
    CameraView,
    CandidatePoseMode,
    CandidateRenderMode,
    ContactMarker,
    ForceMarker,
    MCTSCandidateDiagnostic,
    MCTSDebugArtifact,
    MCTSDebugStep,
    SceneObjectPose,
    TargetVolume,
)


_CANDIDATE_SELECTED_COLOR = (1.00, 0.80, 0.05, 1.00)
_CANDIDATE_INITIAL_COLOR = (0.92, 0.92, 0.92, 1.00)
_CANDIDATE_SOLVED_COLOR = (0.10, 0.85, 0.25, 1.00)
_CANDIDATE_FAILED_COLOR = (0.25, 0.25, 0.25, 0.35)
_CANDIDATE_SCORE_ANCHORS = np.asarray(
    (
        (0.20, 0.32, 0.85, 0.42),
        (0.00, 0.78, 0.82, 0.48),
        (0.98, 0.86, 0.20, 0.55),
        (0.90, 0.18, 0.16, 0.65),
    ),
    dtype=float,
)


@dataclass(frozen=True)
class RenderObject:
    """One object instance prepared for the platform renderer."""

    instance_id: str
    object_id: int
    pose: np.ndarray
    layer: str
    selected: bool = False
    opacity: float = 1.0
    color: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        instance_id = str(self.instance_id).strip()
        layer = str(self.layer).strip()
        object_id = int(self.object_id)
        pose = np.asarray(self.pose, dtype=float)
        opacity = float(self.opacity)
        color = self.color
        if not instance_id or not layer:
            raise ValueError("render object instance ID and layer are required")
        if object_id < 0:
            raise ValueError("render object ID must be non-negative")
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError("render object pose must be finite with shape (7,)")
        if np.linalg.norm(pose[3:]) <= 1e-12:
            raise ValueError("render object quaternion must have non-zero norm")
        if not np.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
            raise ValueError("render object opacity must be between zero and one")
        if color is not None:
            color_array = np.asarray(color, dtype=float)
            if (
                color_array.shape != (3,)
                or not np.all(np.isfinite(color_array))
                or np.any(color_array < 0.0)
                or np.any(color_array > 1.0)
            ):
                raise ValueError("render object color must be a finite unit RGB")
            color = tuple(float(value) for value in color_array)
        pose = pose.copy()
        pose.setflags(write=False)
        object.__setattr__(self, "instance_id", instance_id)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "pose", pose)
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "selected", bool(self.selected))
        object.__setattr__(self, "opacity", opacity)
        object.__setattr__(self, "color", color)


@dataclass(frozen=True)
class OperationsFrame:
    """Complete render input for one operations-console refresh."""

    step: int
    placed_objects: tuple[RenderObject, ...]
    overlays: tuple[RenderObject, ...]
    contacts: tuple[ContactMarker, ...]
    forces: tuple[ForceMarker, ...]
    target: TargetVolume | None
    camera: CameraView | None
    active_candidate_id: str | None
    timeline_frame: int
    timeline_frame_count: int
    robot_parts: tuple[object, ...] = ()
    environment_boxes: tuple[object, ...] | None = None
    object_mesh_paths: Mapping[int, Path] | None = None
    target_transform: np.ndarray | None = None

    @property
    def objects(self) -> tuple[RenderObject, ...]:
        return self.placed_objects + self.overlays


class OperationsController:
    """Own console selection state while preserving renderer camera policy."""

    def __init__(
        self,
        artifact: MCTSDebugArtifact | None = None,
        *,
        trajectory_projector=None,
    ) -> None:
        self.artifact: MCTSDebugArtifact | None = None
        self.planning_records: tuple[PlacementPlanRecord, ...] = ()
        self.search_statistics: dict[int, Mapping[str, object]] = {}
        self.step_index = 0
        self.plan_attempt: int | None = None
        self.plan_segment_index = 0
        self.plan_sample_index = 0
        self.plan_sample_fraction = 0.0
        self.trajectory_projector = trajectory_projector
        self.robot_geometry_mode = "visual"
        self.active_candidate_id: str | None = None
        self.render_mode = CandidateRenderMode.ALL
        self.pose_mode = CandidatePoseMode.ACTION
        self.timeline_frame = 0
        self.show_contacts = False
        self.show_forces = False
        self._camera_reset_requested = False
        if artifact is not None:
            self.set_artifact(artifact)

    def set_robot_geometry_mode(self, mode: str) -> None:
        if mode not in {"visual", "collision", "visual_collision"}:
            raise ValueError(f"unknown robot geometry mode: {mode}")
        self.robot_geometry_mode = mode

    def load(self, path: str | Path) -> MCTSDebugArtifact:
        """Load a native run or a read-only JSON characterization fixture."""

        path = Path(path)
        if path.suffix.lower() == ".stack":
            artifact = StackRunStore.load_mcts(path)
            planning_records = StackRunStore.load_placement_plans(path)
            search_statistics = StackRunStore.load_mcts_statistics(path)
        else:
            values = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(values, dict):
                raise TypeError("operations-console artifact must be a JSON object")
            artifact = (
                decode_legacy_mcts_debug(values)
                if "fixture_version" in values
                else decode_mcts_debug(values)
            )
            planning_records = ()
            search_statistics = ()
        self.set_artifact(artifact, planning_records=planning_records,
                          search_statistics=search_statistics)
        return artifact

    def reload(self, path: str | Path) -> MCTSDebugArtifact:
        """Read newly committed run data while retaining valid viewer selections."""
        step = self.current_step
        candidate_id = self.active_candidate_id
        replay_frame = self.timeline_frame
        attempt = self.plan_attempt
        segment = self.plan_segment_index
        sample = self.plan_sample_index
        fraction = self.plan_sample_fraction
        camera_reset = self._camera_reset_requested
        artifact = self.load(path)
        if step is not None:
            index = next((index for index, value in enumerate(artifact.steps)
                          if value.step == step.step), None)
            if index is not None:
                self.select_step(index)
                if any(value.candidate_id == candidate_id for value in self.current_step.candidates):
                    self.select_candidate(candidate_id)
                    if self.timeline_frame_count:
                        self.set_timeline_frame(min(replay_frame, self.timeline_frame_count - 1))
                if any(value.attempt == attempt for value in self.current_plans):
                    self.select_plan_attempt(attempt)
                    if self.current_plan.result.segments:
                        self.select_plan_segment(min(segment, len(self.current_plan.result.segments) - 1))
                        if self.plan_sample_count:
                            self.set_plan_sample(min(sample, self.plan_sample_count - 1))
                            self.set_plan_interpolation(fraction)
            self._camera_reset_requested = camera_reset
        return artifact

    def set_artifact(
        self,
        artifact: MCTSDebugArtifact,
        *,
        planning_records: tuple[PlacementPlanRecord, ...] = (),
        search_statistics: tuple[Mapping[str, object], ...] = (),
    ) -> None:
        if not isinstance(artifact, MCTSDebugArtifact):
            raise TypeError("operations console requires an MCTSDebugArtifact")
        records = tuple(planning_records)
        if not all(isinstance(value, PlacementPlanRecord) for value in records):
            raise TypeError("operations console planning records have the wrong type")
        self.artifact = artifact
        self.planning_records = records
        self.search_statistics = {
            int(value["step"]): dict(value) for value in search_statistics if "step" in value
        }
        self.step_index = 0
        self.timeline_frame = 0
        self._select_default_candidate()
        self._select_default_plan()
        self._camera_reset_requested = True

    @property
    def current_step(self) -> MCTSDebugStep | None:
        if self.artifact is None or not self.artifact.steps:
            return None
        return self.artifact.steps[self.step_index]

    @property
    def active_candidate(self) -> MCTSCandidateDiagnostic | None:
        step = self.current_step
        if step is None or self.active_candidate_id is None:
            return None
        return next(
            (
                value
                for value in step.candidates
                if value.candidate_id == self.active_candidate_id
            ),
            None,
        )

    @property
    def search_diagnostics(self) -> str:
        """Describe recorded outcomes without inferring an unrecorded cause."""
        step = self.current_step
        if step is None:
            return "Load a stacking run to inspect search outcomes."
        outcome = "succeeded" if step.succeeded else (step.failure_reason or "unsuccessful")
        lines = [f"Step {step.step}: {outcome}"]

        def reasons(label, counts):
            if counts:
                lines.append(label + ":")
                lines.extend(f"  {reason}: {count}" for reason, count in
                             sorted(counts.items(), key=lambda item: (-item[1], item[0])))

        statistics = self.search_statistics.get(step.step, {})
        attempts = statistics.get("search_attempts")
        if attempts is None:
            attempts = (statistics,) if statistics else ()
        for index, attempt in enumerate(attempts, 1):
            lines.append(f"Search attempt {attempt.get('attempt', index)}:")
            if "timed_out" in attempt:
                lines.append("  Search time limit reached: " + ("yes" if attempt["timed_out"] else "no"))
            for key, label in (("generated_candidates", "Generated candidates"),
                               ("failed_evaluations", "Failed evaluations")):
                if key in attempt:
                    lines.append(f"  {label}: {attempt[key]}")
            failure = attempt.get("backend_failure")
            if failure:
                lines.append(f"  Backend failure: {failure['message']}")
                lines.append(f"  Resource: {failure['resource']}; required: {failure['required']}; "
                             f"capacity: {failure['capacity']}")
            tree = attempt.get("tree", {})
            reasons("Retained tree rejection counts (all depths)", tree.get("rejection_reasons", {}))
            cem = attempt.get("cem_statistics", ())
            if not cem and attempt.get("cem"):
                cem = (attempt["cem"],)
            for pool_index, pool in enumerate(cem, 1):
                reasons(f"CEM pool {pool_index} rejection counts", pool.get("rejection_counts", {}))

        reasons("Displayed root candidate failures", Counter(
            candidate.rejection_reason for candidate in step.candidates if candidate.rejection_reason
        ))
        reasons("Root generation rejections", step.rejected_expansions)
        active = self.active_candidate
        if active is not None:
            lines.append(f"Active candidate {active.candidate_id} (object {active.object_id}): "
                         f"{active.rejection_reason or active.outcome.value}")
        if not step.succeeded and not statistics and not step.rejected_expansions and not any(
            candidate.rejection_reason for candidate in step.candidates
        ):
            lines.append("No detailed rejection reasons were recorded for this step.")
        return "\n".join(lines)

    @property
    def current_plans(self) -> tuple[PlacementPlanRecord, ...]:
        step = self.current_step
        if step is None:
            return ()
        return tuple(
            value for value in self.planning_records if value.step == step.step
        )

    @property
    def current_plan(self) -> PlacementPlanRecord | None:
        plans = self.current_plans
        if not plans:
            return None
        return next(
            (
                value
                for value in plans
                if value.attempt == self.plan_attempt
            ),
            plans[-1],
        )

    @property
    def current_plan_segment(self):
        record = self.current_plan
        if record is None or not record.result.segments:
            return None
        index = min(self.plan_segment_index, len(record.result.segments) - 1)
        return record.result.segments[index]

    @property
    def plan_sample_count(self) -> int:
        segment = self.current_plan_segment
        return 0 if segment is None else int(segment.positions.shape[0])

    def select_step(self, index: int) -> None:
        if self.artifact is None:
            raise RuntimeError("no operations-console artifact is loaded")
        index = int(index)
        if index < 0 or index >= len(self.artifact.steps):
            raise IndexError("operations-console step is outside the artifact")
        if index == self.step_index:
            return
        self.step_index = index
        self.timeline_frame = 0
        self._select_default_candidate()
        self._select_default_plan()

    def select_plan_attempt(self, attempt: int) -> None:
        attempt = int(attempt)
        if attempt not in {value.attempt for value in self.current_plans}:
            raise ValueError("planning attempt is not present in the active step")
        self.plan_attempt = attempt
        self._reset_plan_timeline()

    def select_plan_segment(self, index: int) -> None:
        record = self.current_plan
        index = int(index)
        if record is None or index < 0 or index >= len(record.result.segments):
            raise IndexError("planning segment is outside the active plan")
        self.plan_segment_index = index
        self.plan_sample_index = 0
        self.plan_sample_fraction = 0.0

    def set_plan_sample(self, sample: int) -> None:
        sample = int(sample)
        if sample < 0 or sample >= self.plan_sample_count:
            raise IndexError("planning sample is outside the active segment")
        self.plan_sample_index = sample
        self.plan_sample_fraction = 0.0

    def set_plan_interpolation(self, fraction: float) -> None:
        """Interpolate only for display; never modify the recorded trajectory."""
        self.plan_sample_fraction = float(np.clip(fraction, 0.0, 1.0))

    def select_candidate(self, candidate_id: str | None) -> None:
        step = self.current_step
        if step is None:
            if candidate_id is not None:
                raise RuntimeError("no operations-console step is active")
            return
        if candidate_id is None:
            if step.candidates:
                raise ValueError("an existing candidate must be selected")
            self.active_candidate_id = None
            return
        candidate_id = str(candidate_id)
        if candidate_id not in {value.candidate_id for value in step.candidates}:
            raise ValueError("candidate is not present in the active search step")
        self.active_candidate_id = candidate_id
        self.timeline_frame = 0

    def set_render_mode(self, mode: CandidateRenderMode) -> None:
        self.render_mode = CandidateRenderMode(mode)

    def set_pose_mode(self, mode: CandidatePoseMode) -> None:
        self.pose_mode = CandidatePoseMode(mode)
        self.timeline_frame = min(
            self.timeline_frame,
            max(self.timeline_frame_count - 1, 0),
        )

    def set_timeline_frame(self, frame: int) -> None:
        frame = int(frame)
        if frame < 0 or frame >= self.timeline_frame_count:
            raise IndexError("timeline frame is outside the active candidate")
        self.timeline_frame = frame

    @property
    def timeline_frame_count(self) -> int:
        candidate = self.active_candidate
        return 0 if candidate is None else candidate.timeline.frame_count

    def set_diagnostic_layers(self, *, contacts: bool, forces: bool) -> None:
        self.show_contacts = bool(contacts)
        self.show_forces = bool(forces)

    def request_camera_reset(self) -> None:
        self._camera_reset_requested = True

    def consume_camera_reset_request(self) -> bool:
        result = self._camera_reset_requested
        self._camera_reset_requested = False
        return result

    def frame(self, *, trajectory_time: float | None = None) -> OperationsFrame:
        step = self.current_step
        if step is None:
            raise RuntimeError("no operations-console step is active")
        active = self.active_candidate
        overlays = (
            self._trajectory_overlays(active, time=trajectory_time)
            if self.pose_mode is CandidatePoseMode.TRAJECTORY
            else self._candidate_overlays(step, active)
        )
        replay_object_ids = (
            {value.object_id for value in overlays}
            if self.pose_mode is CandidatePoseMode.TRAJECTORY
            else set()
        )
        placed = tuple(
            _render_object(value, f"placed/{value.object_id}", "placed")
            for value in step.scene.objects
            if value.object_id not in replay_object_ids
        )
        return OperationsFrame(
            step=step.step,
            placed_objects=placed,
            overlays=overlays,
            contacts=(
                ()
                if not self.show_contacts or active is None
                else active.simulation_contacts
            ),
            forces=(
                ()
                if not self.show_forces or active is None
                else active.pose_solve_forces
            ),
            target=None if self.artifact is None else self.artifact.target,
            camera=None if self.artifact is None else self.artifact.camera,
            active_candidate_id=self.active_candidate_id,
            timeline_frame=self.timeline_frame,
            timeline_frame_count=self.timeline_frame_count,
        )

    def planning_frame(self) -> OperationsFrame:
        """Project the active world-frame planning attempt for the viewport."""

        record = self.current_plan
        step = self.current_step
        if record is None or step is None:
            return self.frame()
        problem = record.problem
        target_index = problem.action.object_index
        segment = self.current_plan_segment
        scene_objects = (
            problem.pick_scene_objects
            if segment is None or segment.scene_phase == "pick"
            or (segment.scene_phase is None and segment.name.startswith("pick_"))
            else problem.place_scene_objects
        )
        placed = tuple(
            RenderObject(
                instance_id=f"plan/scene/{value.object_id}",
                object_id=value.object_id,
                pose=value.pose,
                layer="placed",
            )
            for value in scene_objects
            if value.object_index != target_index
        )
        overlays = (
            RenderObject(
                instance_id=f"plan/pick/{problem.action.object_id}",
                object_id=problem.action.object_id,
                pose=problem.pick_pose,
                layer="candidate_initial",
                opacity=0.35,
            ),
            RenderObject(
                instance_id=f"plan/place/{problem.action.object_id}",
                object_id=problem.action.object_id,
                pose=problem.place_pose,
                layer="candidate_solved",
                selected=True,
            ),
        )
        robot_parts = ()
        if self.trajectory_projector is not None and segment is not None:
            sample = min(self.plan_sample_index, segment.positions.shape[0] - 1)
            following = min(sample + 1, segment.positions.shape[0] - 1)
            fraction = getattr(self, "plan_sample_fraction", 0.0)
            positions = (1.0 - fraction) * segment.positions[sample] + fraction * segment.positions[following]
            opening = (
                None
                if segment.openings is None
                else float((1.0 - fraction) * segment.openings[sample] + fraction * segment.openings[following])
            )
            if (sample == len(segment.positions) - 1 and opening is not None
                    and self.plan_segment_index + 1 < len(record.result.segments)):
                next_segment = record.result.segments[self.plan_segment_index + 1]
                if (next_segment.openings is not None
                        and np.allclose(positions, next_segment.positions[0], atol=1e-6, rtol=0)):
                    # Animate an opening change at a stationary segment seam,
                    # but never synthesize a joint path across a discontinuity.
                    opening = float((1.0 - fraction) * segment.openings[sample]
                                    + fraction * next_segment.openings[0])
            target_pose = None if segment.target_poses is None else segment.target_poses[sample]
            if target_pose is not None:
                if segment.mode == "attached":
                    # Keep the saved object-to-tool transform rigid even between
                    # samples: interpolating world poses would cut across the
                    # arc followed by a rotating wrist.
                    tool_pose = self.trajectory_projector.tool_pose
                    tool_from_object = compose_poses(
                        inverse_pose(tool_pose(segment.positions[sample])), target_pose)
                    target_pose = compose_poses(tool_pose(positions), tool_from_object)
                overlays = tuple(replace(value, opacity=0.2) for value in overlays) + (
                    RenderObject(
                        instance_id=f"plan/object/{problem.action.object_id}",
                        object_id=problem.action.object_id,
                        pose=target_pose,
                        layer="trajectory",
                    ),
                )
            robot_parts = tuple(
                self.trajectory_projector.project(
                    positions,
                    opening,
                    geometry_mode=self.robot_geometry_mode,
                )
            )
        return OperationsFrame(
            step=step.step,
            placed_objects=placed,
            overlays=overlays,
            contacts=(),
            forces=(),
            target=self._planning_target(record),
            camera=None,
            active_candidate_id=record.candidate_id,
            timeline_frame=0,
            timeline_frame_count=0,
            robot_parts=robot_parts,
        )

    def _planning_target(self, record) -> TargetVolume | None:
        if self.artifact is None:
            return None
        target = self.artifact.target
        if target is None:
            return None
        transform = record.problem.metadata.get("planning_T_search")
        if transform is None:
            return target
        transform = np.asarray(transform, dtype=float)
        if transform.shape != (7,) or not np.all(np.isfinite(transform)):
            return target
        return TargetVolume(
            origin=target.origin + transform[:3],
            width=target.width,
            length=target.length,
            height=target.height,
            structure_offset=target.structure_offset,
            taper=target.taper,
        )

    def _candidate_overlays(
        self,
        step: MCTSDebugStep,
        active: MCTSCandidateDiagnostic | None,
    ) -> tuple[RenderObject, ...]:
        if self.pose_mode is CandidatePoseMode.TRAJECTORY:
            return self._trajectory_overlays(active)
        if self.pose_mode is CandidatePoseMode.BEST_SEQUENCE:
            return self._sequence_overlays(active)
        candidates = self._visible_candidates(step, active)
        candidate_colors = _candidate_score_colors(
            step.candidates,
            None if active is None else active.candidate_id,
        )
        if self.pose_mode is CandidatePoseMode.INITIAL_AND_SOLVED:
            return tuple(
                overlay
                for candidate in candidates
                for overlay in (
                    _candidate_render_object(
                        candidate,
                        candidate.init_pose,
                        "candidate_initial",
                        self.render_mode,
                        suffix="initial",
                        opacity_scale=0.35,
                        color=_CANDIDATE_INITIAL_COLOR,
                        highlighted=candidate is active,
                    ),
                    _candidate_render_object(
                        candidate,
                        candidate.solved_pose,
                        "candidate_solved",
                        self.render_mode,
                        suffix="solved",
                        color=_CANDIDATE_SOLVED_COLOR,
                        highlighted=candidate is active,
                    ),
                )
            )
        overlays = []
        for candidate in candidates:
            pose = candidate.pose(self.pose_mode)
            if pose is not None:
                overlays.append(
                    _candidate_render_object(
                        candidate,
                        pose,
                        "candidate",
                        self.render_mode,
                        color=candidate_colors[candidate.candidate_id],
                        highlighted=candidate is active,
                    )
                )
        return tuple(overlays)

    def _visible_candidates(
        self,
        step: MCTSDebugStep,
        active: MCTSCandidateDiagnostic | None,
    ) -> tuple[MCTSCandidateDiagnostic, ...]:
        if self.render_mode is CandidateRenderMode.SELECTED:
            return () if active is None else (active,)
        return step.render_candidates(self.render_mode)

    def _trajectory_overlays(
        self, candidate: MCTSCandidateDiagnostic | None, *, time: float | None = None
    ) -> tuple[RenderObject, ...]:
        if candidate is None or candidate.timeline.frame_count == 0:
            return ()
        frame = min(self.timeline_frame, candidate.timeline.frame_count - 1)
        return tuple(
            _render_object(
                value,
                f"trajectory/{candidate.candidate_id}/{value.object_id}",
                "trajectory",
                selected=value.object_id == candidate.object_id,
            )
            for value in (
                candidate.timeline.poses_at(frame) if time is None
                else candidate.timeline.poses_at_time(time)
            )
        )

    @staticmethod
    def _sequence_overlays(
        candidate: MCTSCandidateDiagnostic | None,
    ) -> tuple[RenderObject, ...]:
        if candidate is None:
            return ()
        return tuple(
            _render_object(
                value,
                f"sequence/{index}/{value.object_id}",
                "best_sequence",
                selected=index == 0,
            )
            for index, value in enumerate(candidate.best_sequence)
        )

    def _select_default_candidate(self) -> None:
        step = self.current_step
        if step is None or not step.candidates:
            self.active_candidate_id = None
            return
        self.active_candidate_id = (
            step.selected_candidate_id or step.candidates[0].candidate_id
        )

    def _select_default_plan(self) -> None:
        plans = self.current_plans
        self.plan_attempt = None if not plans else plans[-1].attempt
        self._reset_plan_timeline()

    def _reset_plan_timeline(self) -> None:
        self.plan_segment_index = 0
        self.plan_sample_index = 0
        self.plan_sample_fraction = 0.0


def _candidate_render_object(
    candidate: MCTSCandidateDiagnostic,
    pose: np.ndarray,
    layer: str,
    render_mode: CandidateRenderMode,
    *,
    suffix: str = "pose",
    opacity_scale: float = 1.0,
    color: tuple[float, float, float, float],
    highlighted: bool,
) -> RenderObject:
    opacity = 1.0
    if render_mode is CandidateRenderMode.ALL:
        opacity = 1.0 if highlighted else color[3]
    return RenderObject(
        instance_id=f"candidate/{candidate.candidate_id}/{suffix}",
        object_id=candidate.object_id,
        pose=pose,
        layer=layer,
        selected=candidate.selected,
        opacity=opacity * opacity_scale,
        color=color[:3],
    )


def _candidate_score_colors(
    candidates: tuple[MCTSCandidateDiagnostic, ...],
    highlighted_id: str | None,
) -> dict[str, tuple[float, float, float, float]]:
    """Map scores through the legacy blue-to-red candidate palette."""

    finite_scores = [
        float(value.score) for value in candidates if value.score is not None
    ]
    low = min(finite_scores, default=0.0)
    high = max(finite_scores, default=1.0)
    return {
        candidate.candidate_id: (
            _CANDIDATE_SELECTED_COLOR
            if candidate.candidate_id == highlighted_id
            else _candidate_score_color(candidate.score, low, high)
        )
        for candidate in candidates
    }


def _candidate_score_color(
    score: float | None,
    low: float,
    high: float,
) -> tuple[float, float, float, float]:
    if score is None:
        return _CANDIDATE_FAILED_COLOR
    weight = 0.5 if high <= low else (float(score) - low) / (high - low)
    position = float(np.clip(weight, 0.0, 1.0)) * (
        len(_CANDIDATE_SCORE_ANCHORS) - 1
    )
    index = min(int(np.floor(position)), len(_CANDIDATE_SCORE_ANCHORS) - 2)
    color = (
        _CANDIDATE_SCORE_ANCHORS[index] * (index + 1 - position)
        + _CANDIDATE_SCORE_ANCHORS[index + 1] * (position - index)
    )
    return tuple(float(value) for value in color)


def _render_object(
    value: SceneObjectPose,
    instance_id: str,
    layer: str,
    *,
    selected: bool = False,
) -> RenderObject:
    return RenderObject(
        instance_id=instance_id,
        object_id=value.object_id,
        pose=value.pose,
        layer=layer,
        selected=selected,
    )
