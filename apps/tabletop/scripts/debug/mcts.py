"""Run the migrated tabletop MCTS from one YAML configuration file."""

import argparse
import hashlib
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from stacking_common.actions import InitializedPlacement, PlacementSolveResult
from stacking_common.artifacts import PlanningCheckpoint, StackRunStore, export_run_summary
from stacking_common.planning import PickSceneSnapshot
from stacking_common.domain import Action, State
from stacking_common.environment import PlacementResult
from stacking_common.search import (
    EvaluationFidelity,
    MCTSNode,
    MCTSSession,
    ParallelTransitionEvaluator,
    StackingEnvironmentEvaluator,
)
from stacking_common.visualization import (
    MCTSDebugArtifact,
    project_mcts_step,
)
from stacking_common.workflows.mcts_reporting import (
    ProgressReporter as _ProgressReporter,
    progress_event as _progress_event,
    search_statistics as _search_statistics,
    print_sequence_step as _print_sequence_step,
    print_cem_statistics as _print_cem_statistics,
    print_search_step as _print_step,
    tree_statistics as _tree_statistics,
)
from stacking_common.workflows.search import require_search_target
from stacking_tabletop import TabletopApplication
from stacking_tabletop.config.action import resolve_pose_from_scan
from stacking_tabletop.config import (
    load_mcts_profile,
    planning_profile_with_pick_height,
)
from stacking_tabletop.planning import (
    infer_legacy_pick_support,
    load_legacy_pick_scene,
    load_native_solver_profile,
)
from stacking_tabletop.planning.persistence import (
    branch_planning_run, build_planning_context, validate_planning_resume,
)


class _PassThroughPoseSolver:
    """Debug-only solver that accepts the geometry-initialized pose."""

    def solve(
        self,
        inventory,
        state: State,
        object_index: int,
        initial: InitializedPlacement,
    ) -> PlacementSolveResult:
        del inventory, state, object_index
        return PlacementSolveResult(
            pose=initial.pose,
            c_feq=0.0,
            c_gap=0.0,
            diagnostics={"pose_solver": "debug_passthrough"},
        )


class _KinematicPlacementBackend:
    """Debug-only transition that places an object without simulation."""

    def __init__(self) -> None:
        self.object_ids: tuple[int, ...] = ()

    def reset(self, object_ids) -> None:
        self.object_ids = tuple(int(value) for value in object_ids)

    def place(self, state: State, action: Action) -> PlacementResult:
        del state
        if not self.object_ids:
            raise RuntimeError("kinematic backend has not been reset")
        return PlacementResult(
            object_poses={action.object_id: action.pose},
            c_feq=action.c_feq,
            c_gap=action.c_gap,
            diagnostics={"transition_backend": "debug_kinematic"},
        )

    def restore(self, state: State) -> None:
        if tuple(state.object_ids) != self.object_ids:
            raise ValueError("state inventory does not match kinematic backend")

    def close(self) -> None:
        pass


def main() -> int | None:
    args = _parser().parse_args()
    if args.save_run is not None and args.save_run.exists():
        raise FileExistsError("choose a new --save-run path; existing runs are preserved")
    if args.resume_run is not None and args.save_run is None:
        raise ValueError("--resume-run requires a new --save-run destination")
    if args.export_summary is not None and args.save_run is None:
        raise ValueError("--export-summary requires --save-run")
    config_path = args.config.resolve()
    debug = _load_debug_values(config_path)
    checkpoint = None
    if args.resume_run is not None:
        if args.pick_pose_data is not None or args.planning_pick_position is not None:
            raise ValueError("resume uses checkpoint pickup poses, not pickup overrides")
        checkpoint = StackRunStore.load_planning_checkpoint(args.resume_run)
        recorded_request = StackRunStore.info(args.resume_run).metadata["request"]
        if not recorded_request.get("plan_motion"):
            raise ValueError("planning resume requires a motion-planning run")
        debug.update(recorded_request)
        debug["pick_pose_data"] = None
        debug["planning_pick_position"] = [0.0, -0.7, 0.2]
        debug["steps"] = max(1, len(checkpoint.state.object_ids) - checkpoint.step)
    repository = (
        args.repository.resolve()
        if args.repository is not None
        else _repository_from_config(config_path)
    )
    object_ids = tuple(
        int(value)
        for value in _override(args.object_ids, debug, "object_ids")
    )
    steps = int(_override(args.steps, debug, "steps"))
    iterations = int(_override(args.iterations, debug, "iterations"))
    max_search_time = float(
        _override(args.max_search_time, debug, "max_search_time")
    )
    transition_backend = str(
        _override(args.transition_backend, debug, "transition_backend")
    )
    pose_solver_name = str(_override(args.pose_solver, debug, "pose_solver"))
    initial_yaw_degrees = float(
        _override(args.initial_yaw_degrees, debug, "initial_yaw_degrees")
    )
    seed = int(_override(args.seed, debug, "seed"))
    plan_motion_value = (
        debug.get("plan_motion", False)
        if args.plan_motion is None
        else args.plan_motion
    )
    if not isinstance(plan_motion_value, bool):
        raise TypeError("debug plan_motion must be boolean")
    plan_motion = plan_motion_value
    if checkpoint is not None and not plan_motion:
        raise ValueError("planning resume requires --plan-motion")
    max_planning_attempts_value = _override(
        args.max_planning_attempts_per_step,
        debug,
        "max_planning_attempts_per_step",
        default=None,
    )
    max_planning_attempts = (
        None
        if max_planning_attempts_value is None
        else int(max_planning_attempts_value)
    )
    planning_pick_position = np.asarray(
        _override(
            args.planning_pick_position,
            debug,
            "planning_pick_position",
            default=[0.0, -0.7, 0.2],
        ),
        dtype=float,
    )
    pick_pose_data_value = _override(
        args.pick_pose_data,
        debug,
        "pick_pose_data",
        default=None,
    )
    if pick_pose_data_value is not None and args.planning_pick_position is not None:
        raise ValueError(
            "--pick-pose-data and --planning-pick-position are mutually exclusive"
        )
    _validate_debug_values(
        object_ids,
        steps,
        iterations,
        max_search_time,
        transition_backend,
        pose_solver_name,
        max_planning_attempts,
        planning_pick_position,
    )

    mcts_profile = load_mcts_profile(config_path)
    target_options = {}
    if args.target_shape is not None or checkpoint is not None:
        recorded = (None if checkpoint is None else
            StackRunStore.load_run_context(args.resume_run, checkpoint.context_sha256).profiles["inventory_profile"])
        target_options = dict(target_shape=args.target_shape, recorded_target=recorded)
    application = replace(
        TabletopApplication.from_repository(repository, **target_options),
        mcts_profile=mcts_profile,
    )
    recorded_scan = None
    if checkpoint is not None:
        context = StackRunStore.load_run_context(args.resume_run, checkpoint.context_sha256)
        recorded_scan = context.profiles["action_generation_profile"].get("pose_from_scan", True)
    scan_enabled = resolve_pose_from_scan(application.action_generation_profile.pose_from_scan,
                                         args.pose_from_scan, recorded_scan)
    application = replace(application, action_generation_profile=replace(
        application.action_generation_profile, pose_from_scan=scan_enabled))
    print(f"pose_from_scan: {str(scan_enabled).lower()}", flush=True)
    print(f"target shape: {application.inventory_profile.target_name or 'inline/recorded'}", flush=True)
    pick_scene = None
    if checkpoint is not None:
        if object_ids != tuple(checkpoint.state.object_ids):
            raise ValueError("resume cannot change the checkpoint inventory")
        pick_scene = checkpoint.pick_scene
        if "pick_table_height" in pick_scene.metadata:
            application = replace(application, planning_profile=planning_profile_with_pick_height(
                application.planning_profile, float(pick_scene.metadata["pick_table_height"])))
        initial_poses = {object_id: pick_scene.pose(object_id) for object_id in object_ids}
        pick_scene_description = f"checkpoint from {args.resume_run}; committed step={checkpoint.step}"
    elif pick_pose_data_value is not None:
        pick_pose_path = _repository_path(repository, pick_pose_data_value)
        full_pick_scene = load_legacy_pick_scene(
            pick_pose_path,
            frame_id=application.planning_profile.world_T_stack.parent_frame,
        )
        missing_ids = [object_id for object_id in object_ids if object_id not in full_pick_scene.object_poses]
        if missing_ids:
            print(
                f"[WARN] Skipping requested object IDs absent from pick-pose-data: {missing_ids}",
                flush=True,
            )
            object_ids = tuple(object_id for object_id in object_ids if object_id in full_pick_scene.object_poses)
            if not object_ids:
                print("[WARN] No requested objects have recorded pick poses; nothing to plan.", flush=True)
                return
            steps = min(steps, len(object_ids))
        support = infer_legacy_pick_support(
            full_pick_scene,
            application.build_pcd_catalog(),
        )
        pick_scene = full_pick_scene.select(object_ids)
        pick_scene = replace(
            pick_scene,
            metadata={
                **dict(pick_scene.metadata),
                "bottom_quantile": 0.01,
                "pick_table_height": support.height,
                "pick_bottom_z_by_object": {
                    str(object_id): height
                    for object_id, height in support.bottom_by_object.items()
                },
            },
        )
        application = replace(
            application,
            planning_profile=planning_profile_with_pick_height(
                application.planning_profile,
                support.height,
            ),
        )
        initial_poses = {
            object_id: pick_scene.pose(object_id) for object_id in object_ids
        }
        pick_scene_description = (
            f"legacy pose_data={pick_scene.source}; "
            f"inferred table_z={support.height:.4f}m"
        )
    else:
        yaw = np.deg2rad(initial_yaw_degrees) / 2.0
        initial_position = (
            planning_pick_position
            if plan_motion
            else np.array([0.0, 0.0, 0.20])
        )
        initial_poses = {
            object_id: np.concatenate(
                [
                    initial_position,
                    np.array([0.0, 0.0, np.sin(yaw), np.cos(yaw)]),
                ]
            )
            for object_id in object_ids
        }
        pick_scene_description = "synthetic uniform poses"
    inventory = application.load_inventory(object_ids, initial_poses)
    require_search_target(inventory.target,
        ground_z=float(application.posegen_profile.objective.get("ground_height", 0.)))
    planning_context = None
    native_solver = None
    if plan_motion:
        if pick_scene is None:
            pick_scene = PickSceneSnapshot(application.planning_profile.world_T_stack.parent_frame,
                                           initial_poses, "synthetic_debug_pose")
        native_solver = load_native_solver_profile(application.solver_profiles.grasp,
                                                   application.solver_profiles.motion)
        if args.save_run is not None:
            planning_context = build_planning_context(application, inventory, pick_scene, native_solver, {
                "iterations": iterations, "max_search_time": max_search_time,
                "seed": seed, "transition_backend": transition_backend,
                "pose_solver": pose_solver_name,
                "max_planning_attempts_per_step": max_planning_attempts,
            })
            if checkpoint is not None:
                validate_planning_resume(args.resume_run, planning_context)
    root_environment = _build_root_environment(
        application,
        inventory,
        transition_backend,
    )
    pose_solver = (
        _PassThroughPoseSolver()
        if pose_solver_name == "passthrough"
        else None
    )
    transition_evaluator = None
    worker_count = application.mcts_profile.parallel.effective_workers
    if transition_backend == "kinematic":
        evaluators = [
            StackingEnvironmentEvaluator(
                _build_environments(application, inventory, transition_backend),
                settled_floor_policy=(
                    application.build_floor_policy(inventory)
                    if application.action_generation_profile.strict_floor_first
                    else None
                ),
            )
            for _ in range(worker_count)
        ]
        transition_evaluator = (
            evaluators[0]
            if worker_count == 1
            else ParallelTransitionEvaluator(evaluators, max_workers=worker_count)
        )
    search = application.build_mcts(
        inventory,
        transition_evaluator=transition_evaluator,
        pose_solver=pose_solver,
        rng=random.Random(seed),
    )
    request = replace(
        application.default_mcts_request(),
        iterations=iterations,
        max_time_seconds=max_search_time,
    )
    reset = root_environment.reset()
    root = MCTSNode(state=reset.state, observation=reset.observation)
    if checkpoint is not None:
        root_environment.restore(checkpoint.state)
        root = MCTSNode(state=root_environment.snapshot(), observation=root_environment.observation())
    session = MCTSSession(search, root)
    print(f"config: {config_path}", flush=True)
    print(
        "debug mode: real tabletop assets/actions/rewards; "
        f"{pose_solver_name} pose solve; {transition_backend} transition; "
        f"search_workers={worker_count}; motion_planning={plan_motion}",
        flush=True,
    )
    print(f"pick scene: {pick_scene_description}", flush=True)
    planning_cap = (
        "all ranked candidates"
        if max_planning_attempts is None
        else str(max_planning_attempts)
    )
    print(
        "debug workload: "
        f"objects={len(object_ids)}, steps={min(steps, len(inventory))}, "
        f"iterations/step={iterations}, search_limit/attempt="
        f"{max_search_time:.1f}s, planning_cap/step={planning_cap}; "
        "progress heartbeat=10s",
        flush=True,
    )
    rewards: list[float] = []
    last_info: dict[str, object] = {}
    debug_output = (
        None
        if args.save_run is None
        else args.save_run.resolve()
    )
    run_writer = None
    run_succeeded = False
    sequence_result = None
    progress = _ProgressReporter()
    progress.start()
    try:
        if checkpoint is not None:
            run_writer, checkpoint = branch_planning_run(args.resume_run, debug_output, planning_context)
        elif debug_output is not None:
            run_writer = StackRunStore.create_mcts(
                debug_output,
                _debug_header(
                    inventory,
                    frame_id=(
                        application.planning_profile.world_T_stack.child_frame
                    ),
                ),
                metadata={
                    "platform": "tabletop",
                    "target_shape": application.inventory_profile.target_name,
                    "producer": "apps/tabletop/scripts/debug/mcts.py",
                    "config": {
                        "path": str(config_path),
                        "sha256": _file_sha256(config_path),
                        "yaml": config_path.read_text(encoding="utf-8"),
                    },
                    "request": {
                        "object_ids": list(object_ids),
                        "steps": steps,
                        "iterations": iterations,
                        "max_search_time": max_search_time,
                        "transition_backend": transition_backend,
                        "pose_solver": pose_solver_name,
                        "initial_yaw_degrees": initial_yaw_degrees,
                        "seed": seed,
                        "workers": worker_count,
                        "plan_motion": plan_motion,
                        "pose_from_scan": scan_enabled,
                        "max_planning_attempts_per_step": max_planning_attempts,
                        "planning_pick_position": (
                            None
                            if pick_scene is not None
                            else planning_pick_position.tolist()
                        ),
                        "pick_pose_data": (
                            None if pick_scene is None else pick_scene.source
                        ),
                    },
                    "pick_scene": (
                        pick_scene.artifact_metadata()
                        if pick_scene is not None
                        else {
                            "frame_id": (
                                application.planning_profile.world_T_stack.parent_frame
                            ),
                            "source": "synthetic_debug_pose",
                            "object_poses": {
                                str(object_id): pose.tolist()
                                for object_id, pose in initial_poses.items()
                            },
                        }
                    ),
                },
            )
        if plan_motion:
            if run_writer is not None:
                run_writer.append_run_context(planning_context)
            sequence = application.build_planning_sequence(
                inventory,
                session,
                request,
                writer=run_writer,
                progress=progress.update,
                solver_profile=native_solver,
                context_sha256=None if planning_context is None else planning_context.sha256,
                checkpoint_builder=(None if planning_context is None else
                    lambda step, state: PlanningCheckpoint(step, planning_context.sha256,
                        state, pick_scene, application.planning_profile.world_T_stack)),
            )
            sequence_result = sequence.run(
                min(steps, len(inventory)),
                max_planning_attempts_per_step=max_planning_attempts,
                first_step=1 if checkpoint is None else checkpoint.step + 1,
            )
            for sequence_step in sequence_result.steps:
                _print_sequence_step(sequence_step)
                candidate = next(
                    (
                        value
                        for value in sequence_step.debug_step.candidates
                        if value.selected
                    ),
                    None,
                )
                if candidate is not None and candidate.reward is not None:
                    rewards.append(float(candidate.reward))
            if not sequence_result.succeeded:
                print(
                    "[WARN] Planning sequence stopped: "
                    f"status={sequence_result.status.value}, "
                    f"reason={sequence_result.reason}"
                )
        else:
            for step in range(min(steps, len(inventory))):
                progress.update("step_started", {"step": step + 1})
                progress.update(
                    "search_started",
                    {"step": step + 1, "search_attempt": 1},
                )
                started = time.perf_counter()
                result = session.search(request)
                elapsed = time.perf_counter() - started
                tree = _tree_statistics(result.root)
                progress.update(
                    "search_completed",
                    {
                        "step": step + 1,
                        "search_attempt": 1,
                        "elapsed_seconds": elapsed,
                        "generated_candidates": result.generated_candidates,
                        "evaluated_candidates": result.evaluated_candidates,
                        "ranked_candidates": len(result.ranked_candidates),
                        "tree_nodes": tree["nodes"],
                        "tree_leaves": tree["leaves"],
                    },
                )
                _print_step(step + 1, result, elapsed, search)
                debug_step = project_mcts_step(
                    result.root,
                    result.selected,
                    step=step + 1,
                    failure_reason="search_exhausted" if result.selected is None else None,
                )
                if run_writer is not None:
                    run_writer.append_mcts_step(
                        debug_step,
                        statistics=_search_statistics(
                            result, elapsed, search, step=step + 1
                        ),
                    )
                if result.selected is None:
                    break
                rewards.append(float(result.selected.reward))
                last_info = dict(result.selected.info)
                root = session.promote(result.selected, preserve_subtree=True)
                progress.update(
                    "step_completed",
                    {
                        "step": step + 1,
                        "status": "complete",
                        "elapsed_seconds": elapsed,
                    },
                )
                if root.terminal or root.failed:
                    break
        run_succeeded = (
            sequence_result.succeeded if plan_motion else True
        )
    finally:
        try:
            search.close()
        finally:
            try:
                root_environment.close()
            finally:
                if run_writer is not None:
                    try:
                        run_writer.finish(
                            "complete" if run_succeeded else "failed"
                        )
                    finally:
                        progress.stop()
                else:
                    progress.stop()

    if debug_output is not None:
        print(f"stack run: {debug_output}")

    discounted_return = sum(
        reward * application.mcts_profile.tree.discount**index
        for index, reward in enumerate(rewards)
    )
    if plan_motion:
        print(
            f"\nreturn={discounted_return:8.3f}  "
            f"planning_status={sequence_result.status.value}  "
            f"steps={len(rewards)}"
        )
    else:
        print(
            f"\nreturn={discounted_return:8.3f}  "
            f"target_IoU={float(last_info.get('target_IoU', 0.0)):6.3f}  "
            f"steps={len(rewards)}"
        )
    return 0 if run_succeeded else 1


def _build_root_environment(application, inventory, transition_backend):
    backend = (
        application.build_core_backend(inventory, EvaluationFidelity.LONG)
        if transition_backend in {"core", "simulation"}
        else _KinematicPlacementBackend()
    )
    return application.build_environment(inventory, backend)


def _debug_header(inventory, *, frame_id: str) -> MCTSDebugArtifact:
    from stacking_common.visualization import target_display
    return MCTSDebugArtifact(
        frame_id=frame_id,
        target=target_display(inventory.target),
        steps=(),
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_environments(application, inventory, transition_backend):
    if transition_backend in {"core", "simulation"}:
        return application.build_search_environments(inventory)
    return {
        fidelity: application.build_environment(
            inventory,
            _KinematicPlacementBackend(),
        )
        for fidelity in (EvaluationFidelity.LONG,)
    }


def _load_debug_values(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing MCTS debug configuration: {path}")
    values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict):
        raise TypeError("MCTS debug configuration root must be a mapping")
    debug = values.get("debug")
    if not isinstance(debug, dict):
        raise ValueError("MCTS configuration is missing the 'debug' mapping")
    required = {
        "object_ids",
        "steps",
        "iterations",
        "max_search_time",
        "transition_backend",
        "pose_solver",
        "initial_yaw_degrees",
        "seed",
    }
    missing = sorted(required - set(debug))
    if missing:
        raise ValueError(f"MCTS debug configuration is missing: {missing}")
    return debug


def _repository_from_config(path: Path) -> Path:
    for parent in path.parents:
        if (parent / "apps" / "tabletop" / "config").is_dir():
            return parent
    raise ValueError(
        "cannot infer the repository from the MCTS config; pass --repository"
    )


def _repository_path(repository: Path, value) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (repository / path).resolve()


def _override(argument, values, key, *, default=None):
    return values.get(key, default) if argument is None else argument


def _validate_debug_values(
    object_ids,
    steps,
    iterations,
    max_search_time,
    transition_backend,
    pose_solver,
    max_planning_attempts,
    planning_pick_position,
) -> None:
    if not object_ids or len(object_ids) != len(set(object_ids)):
        raise ValueError("debug object IDs must be non-empty and unique")
    if steps <= 0:
        raise ValueError("debug placement count must be positive")
    if iterations <= 0:
        raise ValueError("MCTS iteration count must be positive")
    if not np.isfinite(max_search_time) or max_search_time <= 0.0:
        raise ValueError("MCTS maximum search time must be positive")
    if transition_backend not in {"kinematic", "simulation", "core"}:
        raise ValueError(
            "debug transition backend must be 'kinematic', 'simulation', or 'core'"
        )
    if pose_solver not in {"passthrough", "native"}:
        raise ValueError("debug pose solver must be 'passthrough' or 'native'")
    if max_planning_attempts is not None and max_planning_attempts <= 0:
        raise ValueError("maximum planning attempts per step must be positive")
    if planning_pick_position.shape != (3,) or not np.all(
        np.isfinite(planning_pick_position)
    ):
        raise ValueError("planning pick position must contain three finite values")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the migrated tabletop MCTS from YAML."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("apps/tabletop/config/mcts_cpu.yml"),
        help="MCTS YAML containing tree and debug sections.",
    )
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--target-shape", help="name in config/target_shapes.yml; new runs only")
    parser.add_argument("--object-ids", type=int, nargs="+")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--max-search-time", type=float)
    parser.add_argument(
        "--transition-backend",
        choices=("kinematic", "simulation", "core"),
    )
    parser.add_argument("--pose-solver", choices=("passthrough", "native"))
    parser.add_argument("--initial-yaw-degrees", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--pose-from-scan", action=argparse.BooleanOptionalAction, default=None,
                        help="Override orientation.pose_from_scan; --no-pose-from-scan uses model-derived "
                             "placement orientations. Recorded pickup poses remain available for grasp planning.")
    parser.add_argument(
        "--plan-motion",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run native grasp and motion planning for each selected action.",
    )
    parser.add_argument("--max-planning-attempts-per-step", type=int)
    parser.add_argument(
        "--planning-pick-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="World-frame pickup position used when motion planning is enabled.",
    )
    parser.add_argument(
        "--pick-pose-data",
        type=Path,
        help=(
            "Trusted legacy pose_data.pkl containing per-object world-frame "
            "pickup poses."
        ),
    )
    parser.add_argument(
        "--save-run",
        type=Path,
        help="Write the versioned single-file .stack run document.",
    )
    parser.add_argument("--resume-run", type=Path,
                        help="Branch from the latest committed planning checkpoint; requires a new --save-run.")
    parser.add_argument("--export-summary", type=Path,
                        help="Export the finished or interrupted run to a new .json or .csv file.")
    parser.add_argument("--max-run-time", type=float,
                        help="Hard whole-run wall limit in seconds, including native calls (default: unlimited).")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    if args._worker:
        raise SystemExit(main())
    else:
        from stacking_tabletop.planning.process import supervise_planning
        if args.export_summary is not None and args.save_run is None:
            raise ValueError("--export-summary requires --save-run")
        if args.export_summary is not None:
            if args.export_summary.exists():
                raise FileExistsError("choose a new --export-summary destination")
            if args.export_summary.suffix.lower() not in {".json", ".csv"}:
                raise ValueError("--export-summary must end in .json or .csv")
        status = supervise_planning(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_worker"],
            destination=args.save_run, max_seconds=args.max_run_time,
        )
        if args.export_summary is not None and args.save_run.is_file():
            export_run_summary(args.save_run, args.export_summary)
            print(f"summary: {args.export_summary}", flush=True)
        raise SystemExit(status)
