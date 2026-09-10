"""Offline excavator MCTS and motion planning; no ROS or hardware imports."""

import argparse
from dataclasses import replace
from pathlib import Path
import random
import sys
import time

import numpy as np

from stacking_common.artifacts import FixedCollisionGeometry, PlanningCheckpoint, StackRunStore
from stacking_common.artifacts.pick_scene import load_pick_scene
from stacking_common.artifacts.planning_context import build_planning_context, _values
from stacking_common.profiles.native import load_native_solver_profile
from stacking_common.planning import PickSceneSnapshot
from stacking_common.search import EvaluationFidelity, MCTSNode, MCTSSession
from stacking_common.visualization import TargetVolume
from stacking_common.visualization.mcts import MCTSDebugArtifact, project_mcts_step
from stacking_common.workflows.process import supervise_planning
from stacking_common.workflows.mcts_reporting import (
    ProgressReporter, print_search_step, print_sequence_step, search_statistics,
)

from stacking_excavator import ExcavatorApplication
from .resume import branch_excavator_run, load_resume_checkpoint, restore_pick_scene


def planning_context(application, inventory, pick_scene, solver, runtime):
    context = build_planning_context(application, inventory, pick_scene, solver, runtime)
    settings = application.excavator_profile
    planes = tuple(FixedCollisionGeometry(f"{phase}_ground", "plane",
                    np.array([0., 0., height, 0., 0., 0., 1.]), application.planning_profile.obstacle_friction)
                   for phase, height in (("pick", settings.pick_ground_height),
                                         ("place", settings.place_ground_height),
                                         ("handoff", settings.handoff_ground_height)))
    return replace(context, profiles={**context.profiles, "platform": "excavator",
        "excavator_profile": _values(settings), "floor_settings": application.floor_settings,
        "sampling_settings": application.sampling_settings,
        "pose_from_scan": application.pose_from_scan, "roll_angles": application.roll_angles},
        fixed_geometry=(*context.fixed_geometry, *planes))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument("--config", type=Path, help="excavator MCTS YAML; other profiles belong to the platform")
    parser.add_argument("--pick-pose-data", type=Path,
                        help="recorded JSON or trusted legacy pose_data.pkl; optional for search-only unless pose_from_scan is enabled")
    parser.add_argument("--object-ids", type=int, nargs="+",
                        help="select recorded scene objects, or catalog objects for search-only without pickup data")
    parser.add_argument("--pose-from-scan", action=argparse.BooleanOptionalAction, default=None,
                        help="override MCTS scan-based orientation initialization/filtering; "
                             "--no-pose-from-scan disables it. Recorded pickup poses are used in either mode")
    parser.add_argument("--save-run", type=Path, required=True)
    parser.add_argument("--resume-run", type=Path,
                        help="branch a checkpoint with unchanged planning inputs; steps and run-time limit may change")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--max-search-time", type=float)
    parser.add_argument("--max-run-time", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--search-only", action="store_true", help="save native placement search without robot motion")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def select_pick_scene(scene, requested_ids):
    """Skip unavailable CLI selections without inventing pickup poses."""
    if requested_ids is None:
        return scene
    if not requested_ids or len(requested_ids) != len(set(requested_ids)):
        raise ValueError("selected pick-scene object IDs must be unique")
    missing = [object_id for object_id in requested_ids if object_id not in scene.object_poses]
    if missing:
        print(f"[WARN] Pick scene does not contain requested object IDs: {missing}; skipping them.",
              file=sys.stderr, flush=True)
    selected = [object_id for object_id in requested_ids if object_id in scene.object_poses]
    if not selected:
        print("[WARN] No requested objects are present in the pick scene; nothing to plan. "
              "Omit --object-ids to use all objects in the scene.", file=sys.stderr, flush=True)
        return None
    return scene.select(selected)


def search_runtime(args, recorded=None):
    """Retain the exact recorded context when continuing committed plans."""
    if recorded is None:
        return {"seed": args.seed if args.seed is not None else 0,
                "iterations": args.iterations, "max_search_time": args.max_search_time,
                "search_only": args.search_only}
    runtime = dict(recorded.profiles["runtime"])
    request = recorded.profiles["mcts_profile"]["request"]
    effective = {"seed": runtime.get("seed", 0),
                 "iterations": request["iterations"],
                 "max_search_time": request["max_time_seconds"]}
    for name, value in effective.items():
        supplied = getattr(args, name)
        if supplied is not None and supplied != value:
            flag = "--" + name.replace("_", "-")
            raise ValueError(f"resume requires recorded {flag}={value}; omit {flag} or repeat "
                             "the recorded value. Only --steps and --max-run-time may change; "
                             "use a new run to change search inputs")
    # A redundant CLI flag must not turn a recorded null override into a value
    # and thereby change the context hash of retained plans.
    return runtime


def resolve_pose_from_scan(configured, override, recorded=None):
    if recorded is not None:
        if override is not None and override != recorded:
            flag = "--pose-from-scan" if recorded else "--no-pose-from-scan"
            raise ValueError(f"resume preserves recorded pose_from_scan={recorded}; "
                             f"omit the override or use {flag}. Start a new run to change it")
        return recorded
    return configured if override is None else override


def run(args):
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.save_run.exists():
        raise FileExistsError("choose a new --save-run path")
    application = ExcavatorApplication.from_repository(args.repository, mcts_config=args.config)
    checkpoint = None
    old = None
    if args.resume_run:
        if args.pick_pose_data or args.object_ids:
            raise ValueError("resume uses the checkpoint inventory and pickup poses")
        checkpoint = load_resume_checkpoint(args.resume_run)
        old = StackRunStore.load_run_context(args.resume_run, checkpoint.context_sha256)
        if old.profiles.get("platform") != "excavator":
            raise ValueError("source run is not an excavator planning run")
        application = replace(application, pose_from_scan=resolve_pose_from_scan(
            application.pose_from_scan, args.pose_from_scan, old.profiles["pose_from_scan"]))
        search_only = bool(old.profiles["runtime"].get("search_only", False))
        if args.search_only and not search_only:
            raise ValueError("resume preserves motion-planning mode; omit --search-only")
        args.search_only = search_only
        scene = restore_pick_scene(checkpoint, old)
        ids = checkpoint.state.object_ids
    else:
        application = replace(application, pose_from_scan=resolve_pose_from_scan(
            application.pose_from_scan, args.pose_from_scan))
        if args.pick_pose_data is None:
            if not args.search_only:
                raise ValueError("motion planning requires --pick-pose-data")
            if application.pose_from_scan:
                raise ValueError("pose_from_scan requires --pick-pose-data, including for --search-only")
            ids = list(application.inventory_profile.object_ids if args.object_ids is None else args.object_ids)
            if not ids or len(ids) != len(set(ids)):
                raise ValueError("selected catalog object IDs must be unique")
            # Only placed objects enter search simulation. These identity poses
            # satisfy inventory/replay contracts; they are not measured pickups.
            scene = PickSceneSnapshot(application.planning_profile.world_T_stack.parent_frame,
                {i: np.array([0., 0., 0., 0., 0., 0., 1.]) for i in ids},
                "catalog (search-only; no recorded pickup poses)",
                {"format": "synthetic_search_inventory"})
        else:
            scene = load_pick_scene(args.pick_pose_data)
            scene = select_pick_scene(scene, args.object_ids)
            if scene is None:
                return 0
            ids = list(scene.object_ids)
    runtime = search_runtime(args, old)
    seed = runtime.get("seed", 0)
    iterations = runtime.get("iterations")
    max_time = runtime.get("max_search_time")
    request = application.default_mcts_request()
    if iterations is not None:
        request = replace(request, iterations=iterations)
    if max_time is not None:
        request = replace(request, max_time_seconds=max_time,
                          final_validation_reserve_seconds=min(request.final_validation_reserve_seconds, max_time / 2))
    application = replace(application, mcts_profile=replace(application.mcts_profile,
        request=request, root_cem=replace(application.mcts_profile.root_cem, seed=seed)))
    inventory = application.load_inventory(ids, {i: scene.pose(i) for i in ids})
    solver = load_native_solver_profile(application.solver_profiles.grasp, application.solver_profiles.motion)
    context = planning_context(application, inventory, scene, solver, runtime)
    if checkpoint is None:
        target = inventory.target
        writer = StackRunStore.create_mcts(args.save_run, MCTSDebugArtifact(
            application.planning_profile.world_T_stack.child_frame, (),
            TargetVolume(target.origin, target.width, target.length, target.height,
                         taper=target.taper)),
            metadata={"platform": "excavator", "mode": "search" if args.search_only else "planning",
                      "config": {"path": str((args.config or args.repository / "apps/excavator/config/mcts.yml").resolve())}})
        writer.append_run_context(context)
    else:
        writer, checkpoint = branch_excavator_run(args.resume_run, args.save_run, context)
    config_path = (args.config or args.repository / "apps/excavator/config/mcts.yml").resolve()
    print(f"config: {config_path}", flush=True)
    print("debug mode: real excavator assets/actions/rewards; native pose solve; "
          f"core transition; search_workers={application.mcts_profile.parallel.effective_workers}; "
          f"motion_planning={not args.search_only}", flush=True)
    print(f"inventory source: {scene.source}", flush=True)
    orientation_source = "saved run" if old else ("CLI override" if args.pose_from_scan is not None else "profile")
    print(f"pose_from_scan: {str(application.pose_from_scan).lower()} ({orientation_source}); "
          "controls MCTS orientation initialization/filtering; recorded pickup poses are used "
          "for motion planning in either mode", flush=True)
    remaining = len(ids) - (0 if checkpoint is None else len(checkpoint.state.placed_object_indices))
    print(f"debug workload: objects={len(ids)}, steps={min(args.steps, remaining)}, "
          f"iterations/step={request.iterations}, "
          f"search_limit/attempt={request.max_time_seconds:.1f}s, "
          "planning_cap/step=all ranked candidates; progress heartbeat=10s", flush=True)
    rewards = []
    last_info = {}
    search = root_environment = sequence = None
    progress = ProgressReporter()
    progress.start()
    try:
        root_environment = application.build_environment(inventory,
            application.build_core_backend(inventory, EvaluationFidelity.SHORT))
        reset = root_environment.reset()
        if checkpoint is not None:
            root_environment.restore(checkpoint.state)
            root = MCTSNode(state=root_environment.snapshot(), observation=root_environment.observation())
        else:
            root = MCTSNode(state=reset.state, observation=reset.observation)
        search = application.build_mcts(inventory, rng=random.Random(seed))
        session = MCTSSession(search, root)
        if args.search_only:
            succeeded = True
            first_step = 1 if checkpoint is None else checkpoint.step + 1
            for step in range(first_step, first_step + args.steps):
                if session.root.terminal:
                    break
                progress.update("step_started", {"step": step})
                progress.update("search_started", {"step": step, "search_attempt": 1})
                started = time.perf_counter()
                result = session.search(request)
                elapsed = time.perf_counter() - started
                progress.update("search_completed", {
                    "step": step, "search_attempt": 1, "elapsed_seconds": elapsed,
                    "generated_candidates": result.generated_candidates,
                    "evaluated_candidates": result.evaluated_candidates,
                    "ranked_candidates": len(result.ranked_candidates),
                    "execution_rejected_candidates": result.execution_rejected_candidates,
                })
                print_search_step(step, result, elapsed, search)
                writer.append_mcts_step(project_mcts_step(result.root, result.selected, step=step,
                    frame_id=application.planning_profile.world_T_stack.child_frame,
                    failure_reason="search_exhausted" if result.selected is None else None),
                    statistics=search_statistics(result, elapsed, search, step=step),
                    checkpoint=None if result.selected is None else PlanningCheckpoint(
                        step, context.sha256, result.selected.state, scene,
                        application.planning_profile.world_T_stack))
                if result.selected is None:
                    succeeded = False
                    progress.update("step_completed", {"step": step,
                        "status": "search_exhausted", "elapsed_seconds": elapsed})
                    break
                rewards.append(float(result.selected.reward))
                last_info = dict(result.selected.info)
                session.promote(result.selected)
                progress.update("step_completed", {"step": step,
                    "status": "complete", "elapsed_seconds": elapsed})
        else:
            sequence = application.build_planning_sequence(inventory, session, request,
                writer=writer, progress=progress.update, solver_profile=solver, context_sha256=context.sha256,
                checkpoint_builder=lambda step, state: PlanningCheckpoint(step, context.sha256,
                    state, scene, application.planning_profile.world_T_stack))
            result = sequence.run(args.steps, first_step=1 if checkpoint is None else checkpoint.step + 1)
            succeeded = result.succeeded
            for step in result.steps:
                print_sequence_step(step)
                selected = next((candidate for candidate in step.debug_step.candidates
                                 if candidate.selected), None)
                if selected is not None and selected.reward is not None:
                    rewards.append(float(selected.reward))
            if not succeeded:
                print(f"[WARN] Planning sequence stopped: status={result.status.value}, "
                      f"reason={result.reason}", flush=True)
        writer.finish("complete" if succeeded else "failed")
        print(f"stack run: {args.save_run.resolve()}", flush=True)
        discounted_return = sum(reward * application.mcts_profile.tree.discount**index
                                for index, reward in enumerate(rewards))
        outcome = (f"target_IoU={float(last_info.get('target_IoU', 0.0)):6.3f}" if args.search_only
                   else f"planning_status={result.status.value}")
        print(f"\nreturn={discounted_return:8.3f}  {outcome}  steps={len(rewards)}", flush=True)
        return 0 if succeeded else 1
    except BaseException:
        writer.finish("failed")
        raise
    finally:
        progress.stop()
        writer.close()
        if search is not None:
            search.close()
        if root_environment is not None:
            root_environment.close()
        if sequence is not None:
            sequence.planner.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(argv)
    try:
        if not args.worker:
            return supervise_planning([sys.executable, "-u", "-m", __spec__.name, *argv, "--worker"],
                                      destination=args.save_run, max_seconds=args.max_run_time)
        return run(args)
    except (ValueError, TypeError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"planning error: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
