"""Shared terminal progress and diagnostics for offline MCTS workflows."""

import threading
import time
from collections import Counter
from dataclasses import asdict


class ProgressReporter:
    """Print phase changes and heartbeats while native calls are blocking."""

    def __init__(self, interval_seconds: float = 10.0) -> None:
        self.interval_seconds = float(interval_seconds)
        self._stopped = threading.Event()
        self._lock = threading.Lock()
        self._phase = "initializing"
        self._phase_started = time.perf_counter()
        self._run_started = self._phase_started
        self._thread = threading.Thread(
            target=self._heartbeat,
            name="mcts-debug-progress",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        self._thread.join(timeout=1.0)

    def update(self, event: str, details) -> None:
        now = time.perf_counter()
        message, phase = progress_event(event, details)
        with self._lock:
            if phase is not None:
                self._phase = phase
                self._phase_started = now
        if message is not None:
            print(f"[Progress] {message}", flush=True)
        if event == "search_completed":
            for statistics in details.get("cem_statistics", ()):
                print_cem_statistics(statistics)

    def _heartbeat(self) -> None:
        while not self._stopped.wait(self.interval_seconds):
            now = time.perf_counter()
            with self._lock:
                phase = self._phase
                phase_elapsed = now - self._phase_started
            total_elapsed = now - self._run_started
            print(
                f"[Progress] still {phase}; "
                f"phase={phase_elapsed:.1f}s, total={total_elapsed:.1f}s",
                flush=True,
            )


def progress_event(event: str, details) -> tuple[str | None, str | None]:
    step = details.get("step")
    attempt = details.get("attempt", details.get("planning_attempt"))
    if event == "step_started":
        return f"step {step} started", f"starting step {step}"
    if event == "search_started":
        search_attempt = details.get("search_attempt")
        return (
            f"step {step}: MCTS search {search_attempt} started",
            f"running MCTS search {search_attempt} for step {step}",
        )
    if event == "search_completed":
        tree_summary = (
            f", tree_nodes={details['tree_nodes']}, leaves={details['tree_leaves']}"
            if "tree_leaves" in details else ""
        )
        return (
            f"step {step}: MCTS search finished in "
            f"{float(details['elapsed_seconds']):.1f}s; "
            f"generated={details['generated_candidates']}, "
            f"evaluated={details['evaluated_candidates']}, "
            f"ranked={details['ranked_candidates']}, "
            f"MCTS_execution_rejected={details.get('execution_rejected_candidates', 0)}"
            f"{tree_summary}",
            f"selecting a candidate for step {step}",
        )
    if event == "candidate_skipped":
        return (
            f"step {step}: candidate {details['candidate_rank']}/"
            f"{details['candidate_count']} skipped "
            f"(id={details['candidate_id']}, reason={details['reason']})",
            None,
        )
    if event == "candidate_started":
        return (
            f"step {step}: candidate {details['candidate_rank']}/"
            f"{details['candidate_count']} started "
            f"(object={details['object_id']}, id={details['candidate_id']})",
            f"preparing candidate {details['candidate_rank']}/"
            f"{details['candidate_count']} for step {step}",
        )
    if event == "fixed_scene_started":
        return None, f"fixed-scene validation for step {step}, attempt {attempt}"
    if event == "fixed_scene_completed":
        return (
            f"step {step}: fixed-scene validation "
            f"{'accepted' if details['accepted'] else 'rejected'}",
            f"finishing fixed-scene validation for step {step}",
        )
    if event == "motion_started":
        return (
            f"step {step}: native motion planning started",
            f"native motion planning for step {step}, attempt {attempt}",
        )
    if event == "motion_completed":
        return (
            f"step {step}: native motion planning returned "
            f"{details['status']}",
            f"checking native motion result for step {step}",
        )
    if event == "support_validation_started":
        return None, f"released-pose support validation for step {step}"
    if event == "support_validation_completed":
        return (
            f"step {step}: support validation returned {details['status']}",
            f"finishing candidate for step {step}",
        )
    if event == "candidate_completed":
        reason = details.get("reason") or "none"
        return (
            f"step {step}: candidate attempt {attempt} returned "
            f"{details['status']} (reason={reason})",
            f"selecting the next action for step {step}",
        )
    if event == "step_completed":
        return (
            f"step {step} finished with {details['status']} in "
            f"{float(details['elapsed_seconds']):.1f}s",
            f"finishing step {step}",
        )
    return None, None


def search_statistics(
    result, elapsed: float, search, *, step: int
) -> dict[str, object]:
    tree = tree_statistics(result.root)
    cem = getattr(search.root_action_generator, "last_statistics", None)
    return {
        "step": int(step),
        "backend_failure": getattr(result, "backend_failure", None),
        "elapsed_seconds": float(elapsed),
        "completed_iterations": result.completed_iterations,
        "expansion_calls": result.expansion_calls,
        "generated_candidates": result.generated_candidates,
        "seeded_candidates": result.seeded_candidates,
        "execution_rejected_candidates": (
            result.execution_rejected_candidates
        ),
        "evaluated_candidates": result.evaluated_candidates,
        "failed_evaluations": result.failed_evaluations,
        "effective_max_depth": result.effective_max_depth,
        "timed_out": result.timed_out,
        "final_validation": {
            "attempts": result.final_validation_attempts,
            "failures": result.final_validation_failures,
            "timed_out": result.final_validation_timed_out,
        },
        "simulation": {
            "batches": result.short_simulation_batches + result.long_simulation_batches,
            "jobs": result.short_simulation_jobs + result.long_simulation_jobs,
            "batch_wall_seconds": result.short_simulation_seconds + result.long_simulation_seconds,
            "settling_job_seconds": result.short_settling_job_seconds + result.long_settling_job_seconds,
            "structural_job_seconds": result.short_structural_job_seconds + result.long_structural_job_seconds,
            "short_batches": result.short_simulation_batches,
            "short_jobs": result.short_simulation_jobs,
            "short_parallel_batches": result.short_parallel_batches,
            "short_parallel_jobs": result.short_parallel_jobs,
            "long_batches": result.long_simulation_batches,
            "long_jobs": result.long_simulation_jobs,
            "short_max_batch_size": result.short_max_batch_size,
            "max_evaluation_batch_size": result.max_evaluation_batch_size,
            "workers": result.transition_workers,
            "pose_generation_seconds": result.pose_generation_seconds,
            "pose_generation_batches": result.pose_generation_batches,
            "pose_generation_node_requests": (
                result.pose_generation_node_requests
            ),
            "pose_generation_max_node_batch": (
                result.pose_generation_max_node_batch
            ),
            "descendant_frontier_batches": (
                result.descendant_frontier_batches
            ),
            "descendant_frontier_iterations": (
                result.descendant_frontier_iterations
            ),
            "short_batch_wall_seconds": result.short_simulation_seconds,
            "long_batch_wall_seconds": result.long_simulation_seconds,
            "short_settling_job_seconds": (
                result.short_settling_job_seconds
            ),
            "short_structural_job_seconds": (
                result.short_structural_job_seconds
            ),
            "long_settling_job_seconds": result.long_settling_job_seconds,
            "long_structural_job_seconds": result.long_structural_job_seconds,
        },
        "revalidation": asdict(result.revalidation),
        "root_validation": asdict(result.root_validation),
        "full_scene_validation": dict(result.root.info.get("full_scene_validation", {})),
        "root_lookahead": asdict(result.root_lookahead),
        "root_retry": asdict(result.root_retry),
        "floor_fill_fallback": asdict(result.floor_fill_fallback),
        "descendant_floor_retry": tree["descendant_floor_retry"],
        "descendant_validation": {
            "attempts": result.descendant_validation_attempts,
            "failures": result.descendant_validation_failures,
        },
        "final_selection_max_weight": result.final_selection_max_weight,
        "final_selection_depth_gate": result.root.info.get("final_selection_depth_gate"),
        "selected_value": None if result.selected is None else {
            "visits": result.selected.visits,
            "reward": result.selected.reward,
            "mean_return": result.selected.mean_value,
            "deepest_feasible_depth": result.selected.info.get("deepest_feasible_depth"),
            "observed_path_return": result.selected.info.get("descendant_path_return"),
            "score": result.selected.info.get("final_selection_score"),
        },
        "tree": {
            "nodes": tree["nodes"],
            "evaluated": tree["evaluated"],
            "pending": tree["pending"],
            "failed": tree["failed"],
            "leaves": tree["leaves"],
            "frontier_leaves": tree["frontier_leaves"],
            "max_depth": tree["max_depth"],
            "rejection_reasons": dict(tree["rejection_reasons"]),
            "generation_rejections": dict(tree["generation_rejections"]),
        },
        "cem": (
            None
            if cem is None
            else {
                "rounds": [asdict(value) for value in cem.rounds],
                "rejection_counts": dict(cem.rejection_counts),
                "final_candidates": cem.final_candidates,
            }
        ),
    }


def print_sequence_step(step) -> None:
    print(
        f"[INFO] Step {step.step} search/planning: "
        f"search_attempts={len(step.search_attempts)}, "
        f"planning_attempts={len(step.planning_attempts)}, "
        f"elapsed={step.elapsed_seconds:.2f}s"
    )
    for attempt in step.search_attempts:
        print(
            f"[INFO] Search attempt {attempt.attempt}: depth={attempt.effective_max_depth}, "
            f"generated={attempt.generated_candidates}, evaluated={attempt.evaluated_candidates}, "
            f"tree_nodes={attempt.tree_nodes}, leaves={attempt.tree_leaves}, "
            f"MCTS_execution_rejected={attempt.execution_rejected_candidates}, "
            f"timed_out={attempt.timed_out}", flush=True,
        )
        for statistics in attempt.cem_statistics:
            print_cem_statistics(statistics.to_dict())
    for record in step.planning_attempts:
        samples = sum(
            value.positions.shape[0] for value in record.result.segments
        )
        print(
            f"[INFO] Motion attempt {record.attempt}: "
            f"candidate={record.candidate_id}, "
            f"status={record.result.status.value}, "
            f"segments={len(record.result.segments)}, "
            f"samples={samples}, "
            f"reason={record.result.reason or 'none'}, "
            "time="
            f"{float(record.result.diagnostics.get('elapsed_seconds', 0.0)):.2f}s"
        )
        timings = record.result.diagnostics.get("native_timings")
        if isinstance(timings, dict):
            print(
                "[INFO] Native planner phases: "
                f"grasp={float(timings.get('grasp_generation_seconds', 0.0)):.2f}s, "
                "refinement="
                f"{float(timings.get('simulation_refinement_seconds', 0.0)):.2f}s, "
                "trajectory="
                f"{float(timings.get('trajectory_optimization_seconds', 0.0)):.2f}s; "
                f"candidates={int(timings.get('grasp_candidates', 0))}/"
                f"{int(timings.get('refined_candidates', 0))}/"
                f"{int(timings.get('motion_candidates', 0))}"
            )


def print_cem_statistics(statistics) -> None:
    rounds = statistics["rounds"]
    print(
        f"[INFO] Root CEM: retry={statistics['retry_round']}, rounds={len(rounds)}, "
        f"attempts={sum(r['attempted'] for r in rounds)}, "
        f"accepted={sum(r['accepted'] for r in rounds)}, "
        f"rejected={sum(r['rejected'] for r in rounds)}, "
        f"retained={statistics['final_candidates']}, "
        f"execution_filtered_before_keep={sum(count for reason, count in statistics['rejection_counts'].items() if reason.startswith('execution_rejected'))}, "
        f"fresh_exhausted={statistics['fresh_exhausted']}, "
        f"timed_out={statistics['timed_out']}", flush=True,
    )
    for value in rounds:
        print(
            f"[INFO] CEM round {value['round_index'] + 1}: "
            f"attempted={value['attempted']}, accepted={value['accepted']}, "
            f"rejected={value['rejected']}, fresh={value['fresh_attempts']}, "
            f"mutated={value['mutation_attempts']}, "
            f"fallback_mutated={value['fresh_fallback_attempts']}", flush=True,
        )
    if statistics["rejection_counts"]:
        reasons = ", ".join(f"{reason}={count}" for reason, count in sorted(
            statistics["rejection_counts"].items(), key=lambda item: (-item[1], item[0])))
        print(f"[INFO] CEM rejection reasons: {reasons}", flush=True)


def print_search_step(step: int, result, elapsed: float, search) -> None:
    if getattr(result, "backend_failure", None):
        print(f"[ERROR] Search stopped: {result.backend_failure['message']}", flush=True)
    tree = tree_statistics(result.root)
    root_children = result.root.children
    generation_rejected = sum(tree["generation_rejections"].values())
    print(
        "[INFO] Search: "
        f"iterations={result.completed_iterations}, "
        f"effective_depth={result.effective_max_depth}, "
        f"expansions={result.expansion_calls}, "
        f"generated={result.generated_candidates}, "
        f"seeded={result.seeded_candidates}, "
        f"execution_rejected={result.execution_rejected_candidates}, "
        f"transition_evaluations={result.evaluated_candidates}, "
        f"failed_evaluations={result.failed_evaluations}, "
        f"generation_rejected={generation_rejected}, "
        f"timed_out={result.timed_out}"
    )
    print(
        "[INFO] Tree nodes: "
        f"{tree['nodes']}, evaluated: {tree['evaluated']}, "
        f"pending: {tree['pending']}, failed: {tree['failed']}, "
        f"leaves: {tree['leaves']}, "
        f"frontier leaves: {tree['frontier_leaves']}, "
        f"max depth: {tree['max_depth']}"
    )
    print(
        "[INFO] Root candidates: "
        f"total={len(root_children)}, "
        f"selectable={sum(child.selectable for child in root_children)}, "
        f"evaluated={sum(child.evaluated for child in root_children)}, "
        f"pending={sum(not child.evaluated for child in root_children)}, "
        f"failed={sum(child.failed for child in root_children)}"
    )
    cem = getattr(search.root_action_generator, "last_statistics", None)
    if cem is not None and cem.rounds:
        print(
            "[INFO] Root CEM: "
            f"rounds={len(cem.rounds)}, attempts={cem.attempted}, "
            f"posegen_rejected={cem.rejected}, "
            f"duplicates={cem.duplicate_rejections}, "
            f"retained={cem.final_candidates}"
        )
        for round_stats in cem.rounds:
            print(
                f"[INFO] CEM round {round_stats.round_index + 1}: "
                f"attempted={round_stats.attempted}, "
                f"accepted={round_stats.accepted}, "
                f"rejected={round_stats.rejected}, "
                f"duplicates={round_stats.duplicate_rejections}, "
                f"retained={round_stats.retained}, "
                f"mutated={round_stats.mutation_attempts}, "
                f"fresh={round_stats.fresh_attempts}, "
                f"fresh_fallback={round_stats.fresh_fallback_attempts}"
            )
        if cem.rejection_counts:
            reasons = ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(
                    cem.rejection_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            )
            print(f"[INFO] CEM posegen rejections: {reasons}")
    if result.short_simulation_jobs + result.long_simulation_jobs:
        print(
            "[INFO] MCTS simulation (settling -> full structure): "
            f"batches={result.short_simulation_batches + result.long_simulation_batches}, "
            f"jobs={result.short_simulation_jobs + result.long_simulation_jobs}, "
            f"workers={result.transition_workers}, "
            f"max_batch_size={result.max_evaluation_batch_size}"
        )
    if result.pose_generation_batches:
        print(
            "[INFO] MCTS pose generation: "
            f"batches={result.pose_generation_batches}, "
            f"node_requests={result.pose_generation_node_requests}, "
            f"max_node_batch={result.pose_generation_max_node_batch}, "
            f"frontier_batches={result.descendant_frontier_batches}, "
            f"frontier_iterations={result.descendant_frontier_iterations}"
        )
    print(
        "[Timing] MCTS phases: "
        f"pose_generation={result.pose_generation_seconds:.2f}s, "
        f"simulation_batch_wall={result.short_simulation_seconds + result.long_simulation_seconds:.2f}s; "
        "simulation_job_seconds="
        f"(settling={result.short_settling_job_seconds + result.long_settling_job_seconds:.2f}s, "
        f"structural={result.short_structural_job_seconds + result.long_structural_job_seconds:.2f}s)"
    )
    if result.final_validation_attempts:
        print(
            "[INFO] Root long validation: "
            f"attempted={result.final_validation_attempts}, "
            "kept="
            f"{result.final_validation_attempts - result.final_validation_failures}, "
            f"rejected={result.final_validation_failures}"
        )
    if result.root_validation.attempted:
        print(
            "[INFO] Root simulation: "
            f"attempted={result.root_validation.attempted}, "
            f"kept={result.root_validation.accepted}, "
            f"rejected={result.root_validation.rejected}, "
            f"timed_out={result.root_validation.timed_out}"
        )
    full = result.root.info.get("full_scene_validation", {})
    if full.get("attempted"):
        print("[INFO] Unfrozen root validation: "
              f"attempted={full['attempted']}, kept={full['attempted'] - full['failed']}, "
              f"rejected={full['failed']}, wall={full['seconds']:.2f}s")
    if result.root_lookahead.eligible:
        print(
            "[INFO] Root lookahead coverage: "
            f"eligible={result.root_lookahead.eligible}, "
            f"attempted={result.root_lookahead.attempted}, "
            f"generated={result.root_lookahead.generated}, "
            f"evaluated={result.root_lookahead.evaluated}"
        )
    if result.descendant_validation_attempts:
        print(
            "[INFO] Descendant final validation: "
            f"attempted={result.descendant_validation_attempts}, "
            f"rejected={result.descendant_validation_failures}, "
            f"max_weight={result.final_selection_max_weight}"
        )
    if result.root_retry.rounds:
        print(
            "[INFO] Rejected-root retry: "
            f"rounds={result.root_retry.rounds}, "
            f"retired={result.root_retry.retired}, "
            f"generated={result.root_retry.generated}, "
            f"kept={result.root_retry.accepted}"
        )
    if result.floor_fill_fallback.attempted:
        fallback = result.floor_fill_fallback
        print(
            f"[INFO] {fallback.policy.replace('_', '-')} fallback: "
            f"pools={fallback.pools}, generated={fallback.generated}, "
            f"kept={fallback.accepted}, selected={fallback.selected}"
        )
    retry = tree["descendant_floor_retry"]
    if retry["nodes"]:
        print(f"[INFO] Descendant floor retry: nodes={retry['nodes']}, "
              f"attempts={retry['attempts']}, generated={retry['generated']}, "
              f"evaluated={retry['evaluated']}, kept={retry['accepted']}")
    if result.final_validation_timed_out:
        print(
            "[INFO] Root long validation stopped at the search wall-time limit"
        )
    if result.revalidation.attempted:
        print(
            "[INFO] Preserved root validation: "
            f"attempted={result.revalidation.attempted}, "
            f"kept={result.revalidation.accepted}, "
            f"rejected={result.revalidation.rejected}, "
            f"cleared_subtrees={result.revalidation.cleared_subtrees}"
        )
    if tree["rejection_reasons"]:
        reasons = ", ".join(
            f"{reason}={count}"
            for reason, count in tree["rejection_reasons"].most_common()
        )
        print(f"[INFO] Rejection reasons: {reasons}")
    print(f"[Timing] MCTS total wall={elapsed:.2f}s")

    gate = result.root.info.get("final_selection_depth_gate")
    if gate and gate["min_feasible_depth"] > 1:
        print(
            f"[INFO] Final feasible depth: required={gate['min_feasible_depth']}, "
            f"eligible={gate['eligible_candidates']}/{gate['feasible_candidates']}, "
            f"fallback={gate['fallback']}"
        )
    selected = result.selected
    if selected is None:
        rejected = result.failed_evaluations + generation_rejected
        print(
            f"  step {step:2d}: "
            f"{'backend capacity failure' if getattr(result, 'backend_failure', None) else 'no feasible candidate found'}; "
            f"rejected = {rejected}  time = {elapsed:.3f}s"
        )
        return

    info = selected.info
    if "deepest_feasible_depth" in info:
        print(
            f"[INFO] Selected value: visits={selected.visits}, "
            f"mean={selected.mean_value:.3f}, "
            f"deepest_depth={info['deepest_feasible_depth']}, "
            f"observed_path_return={info['descendant_path_return']:.3f}, "
            f"mixed_score={info['final_selection_score']:.3f}"
        )
    stage_times = info.get("simulation_stage_times")
    if isinstance(stage_times, dict) and stage_times:
        stages = ", ".join(
            f"{name}={float(value):.3f}s"
            for name, value in stage_times.items()
        )
        print(
            "[INFO] Selected physical validation: "
            f"stages=({stages}), "
            f"settled={bool(info.get('simulation_settled', True))}, "
            f"support_count={int(info.get('support_count', 0))}, "
            f"support_has_ground={bool(info.get('support_has_ground', False))}"
        )
    print(
        f"  step {step:2d}: reward = {float(selected.reward):2.3f}  "
        f"target_IoU = {float(info.get('target_IoU', 0.0)):.3f}  "
        f"time = {elapsed:.3f}s"
    )


def tree_statistics(root) -> dict[str, object]:
    """Count the retained tree, including failed and pending childless leaves."""
    nodes = []
    stack = [root]
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(node.children)

    generation_rejections: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    for node in nodes:
        generation_rejections.update(node.rejected_expansions)
        rejection_reasons.update(node.rejected_expansions)
        if node.failed:
            rejection_reasons[node.rejection_reason or "transition_failed"] += 1

    retried = [node for node in nodes if node.info.get("descendant_floor_retry_attempted")]
    relaxed = [child for node in retried for child in node.children
               if child.action is not None and (child.action.diagnostics.get("active_layer_fallback_level")
                    or child.action.diagnostics.get("active_layer_admission_relaxed"))]
    return {
        "descendant_floor_retry": {
            "nodes": len(retried),
            "attempts": sum(node.info.get("descendant_floor_retry_attempts", 0) for node in retried),
            "generated": sum(node.info.get("descendant_floor_retry_generated", 0) for node in retried),
            "evaluated": sum(node.evaluated for node in relaxed),
            "accepted": sum(node.evaluated and node.selectable for node in relaxed),
        },
        "nodes": len(nodes),
        "evaluated": sum(node.evaluated for node in nodes),
        "pending": sum(not node.evaluated for node in nodes),
        "failed": sum(node.failed for node in nodes),
        "leaves": sum(not node.children for node in nodes),
        "frontier_leaves": sum(
            node.evaluated
            and not any(child.evaluated for child in node.children)
            for node in nodes
        ),
        "max_depth": max((node.depth for node in nodes), default=0),
        "generation_rejections": generation_rejections,
        "rejection_reasons": rejection_reasons,
    }
