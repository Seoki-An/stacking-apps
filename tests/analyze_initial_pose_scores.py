"""Measure excavator proposal-score sensitivity on recorded scene subsets.

Uses real assets and JSON poses; does not solve placements or run simulation.
The subsets retain each stone's final recorded pose, so they are diagnostic
scene states, not a replay of the original stacking sequence.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
for directory in ("packages/stacking-common/src", "apps/excavator/src"):
    sys.path.insert(0, str(ROOT / directory))

from stacking_common.actions import _scored_floor as floor
from stacking_common.actions.scored import ScoredPlanarPoseSampler
from stacking_common.domain import State
from stacking_common.visualization.heightmap import HeightMapRenderer
from stacking_excavator import ExcavatorApplication


TERMS = ("H", "Q", "F", "B", "D")
KEYS = ("height", "support_likelihood", "fill_area", "target_boundary", "planar_bias")
ABLATIONS = {f"drop_{term}": (term,) for term in TERMS}
ABLATIONS.update({"H_Q_F": ("B", "D"),
                  "H_Q_F_B": ("D",),
                  "H_Q": ("F", "B", "D")})


def candidate_terms(score_map):
    candidates = score_map["candidates"]
    xy = np.array([item["xy"] for item in candidates])
    columns = np.abs(xy[:, 0, None] - score_map["x_coords"]).argmin(axis=1)
    rows = np.abs(xy[:, 1, None] - score_map["y_coords"]).argmin(axis=1)
    heights = np.array([item["height"] for item in candidates])
    values = np.column_stack((
        (heights - score_map["h_min"]) / score_map["h_span"],
        [item["support_likelihood"] for item in candidates],
        score_map["fill_area"][rows, columns],
        [item["target_boundary"] for item in candidates],
        score_map["planar_bias"][rows, columns],
    ))
    weights = np.array([score_map["weights"][key] for key in KEYS])
    weighted = values * weights
    np.testing.assert_array_equal(combine(weighted),
                                  [item["score"] for item in candidates])
    return values, weighted


def combine(weighted, removed=()):
    # Match the scorer's left-to-right sum. Recompute ablations from kept
    # terms so cancellation does not break ties between equal scores.
    scores = np.zeros(len(weighted))
    for index, term in enumerate(TERMS):
        if term not in removed:
            scores += weighted[:, index]
    return scores


def correlation(first, second):
    if np.ptp(first) == 0 or np.ptp(second) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def draws(view, candidates, scores, pools, count, seed):
    updated = [{**item, "score": float(score), "analysis_index": index}
               for index, (item, score) in enumerate(zip(candidates, scores))]
    rng = np.random.RandomState(seed)
    selected = []
    for indices in pools:
        remaining = count - len(selected)
        if remaining > 0 and len(indices):
            selected.extend(floor.sample_scored_candidates(
                view, [updated[index] for index in indices], remaining,
                True, rng=rng))
    return {item["analysis_index"] for item in selected}


def summarize(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    source = ROOT / "docs/benchmarks/excavator-cuda-mcts.json"
    recorded = json.loads(source.read_text())["input_placed9"]
    app = ExcavatorApplication.from_repository(ROOT)
    ids = recorded["object_ids"]
    inventory = app.load_inventory(ids, {key: [0., 0., 0., 0., 0., 0., 1.] for key in ids})
    renderer = HeightMapRenderer(
        inventory, app.environment_profile.observation.height_map_resolution,
        app.environment_profile.height_map_margin)
    policy = app.build_floor_policy(inventory)
    sampler = ScoredPlanarPoseSampler(
        inventory, app.action_generation_profile.planar,
        app.build_orientation_provider(), app.sampling_settings,
        floor_policy=policy)
    count = app.action_generation_profile.generator.max_poses_per_object
    report = {"scope": __doc__.strip(), "source": str(source.relative_to(ROOT)),
              "source_sha256": sha256(source.read_bytes()).hexdigest(),
              "comparison_score_rounding_decimals": 12,
              "comparison_score_rounding_reason": (
                  "Suppress numerical tie changes from constant score offsets; "
                  "feature reconstruction is checked bit-for-bit before rounding."),
              "seeds": list(range(args.seeds)), "sample_count": count,
              "weights": {term: app.sampling_settings["score"][key]
                          for term, key in zip(TERMS, KEYS)}, "scenes": []}
    tracked = [ROOT / "apps/excavator/config/action_generation.yml",
               ROOT / "apps/excavator/config/environment.yml",
               ROOT / "apps/excavator/config/inventory.yml",
               ROOT / "apps/excavator/src/stacking_excavator/composition.py",
               ROOT / "packages/stacking-common/src/stacking_common/visualization/heightmap.py"]
    tracked += list((ROOT / "packages/stacking-common/src/stacking_common/actions").glob("*.py"))
    tracked += [ROOT / "tests/analyze_initial_pose_scores.py"]
    report["implementation_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
        for path in tracked}
    report["asset_sha256"] = {
        f"model_{object_id}.obj": sha256(
            (ROOT / app.inventory_profile.model_directory / f"model_{object_id}.obj").read_bytes()
        ).hexdigest() for object_id in ids}
    for placed_count in (0, 1, 3, 6, 9):
        placed = recorded["placed_object_indices"][:placed_count]
        state = State(ids, placed, {
            ids[index]: np.array(recorded["object_poses"][str(ids[index])])
            for index in placed}, [], [])
        observation = SimpleNamespace(scene_height_map=renderer.render(state),
            height_map_xlim=renderer.xlim, height_map_ylim=renderer.ylim)
        context = policy.context_details(state)
        scene = {"placed_count": placed_count,
                 "support_z": context["support_z"],
                 "occupancy": context["occupancy"], "objects": []}
        for index in range(len(ids)):
            if index in placed:
                continue
            score_map = sampler.score_map(state, observation, index)
            candidates = score_map["candidates"]
            if not candidates:
                continue
            values, weighted = candidate_terms(score_map)
            scores = np.round(combine(weighted), 12)
            all_indices = np.arange(len(candidates))
            lower = np.array([item["height"] <= context["support_z"] + context["bottom_tol"]
                              for item in candidates])
            pools = ([all_indices[lower], all_indices[~lower]] if placed_count
                     else [all_indices])
            priority = pools[0] if len(pools[0]) else pools[-1]
            base_order = np.argsort(-scores[priority], kind="stable")[:count]
            base_top = set(priority[base_order])
            view = sampler.view(state, observation)
            base_draws = [draws(view, candidates, scores, pools, count, seed)
                          for seed in range(args.seeds)]
            row = {"object_id": ids[index], "candidates": len(candidates),
                   "lower_floor_candidates": int(lower.sum()) if placed_count else None,
                   "priority_pool": "lower" if placed_count and lower.any() else "all",
                   "priority_candidates": len(priority),
                   "weighted_std_all": dict(zip(TERMS, weighted.std(axis=0).tolist())),
                   "weighted_std_priority": dict(zip(TERMS, weighted[priority].std(axis=0).tolist())),
                   "B_D_correlation": correlation(values[priority, 3], values[priority, 4]),
                   "ablations": {}}
            for label, removed in ABLATIONS.items():
                alternate = np.round(combine(weighted, removed), 12)
                rho = (float(spearmanr(scores[priority], alternate[priority]).statistic)
                       if np.ptp(scores[priority]) and np.ptp(alternate[priority]) else None)
                alt_top = set(priority[np.argsort(-alternate[priority], kind="stable")[:count]])
                overlap = [len(baseline & draws(view, candidates, alternate, pools, count, seed))
                           / len(baseline) for seed, baseline in enumerate(base_draws)]
                row["ablations"][label] = {"rank_correlation": rho,
                    "top32_overlap": len(base_top & alt_top) / len(base_top),
                    "top32_overlap_when_competing": (
                        len(base_top & alt_top) / len(base_top) if len(priority) > count else None),
                    "sample_population_overlap": float(np.mean(overlap))}
            scene["objects"].append(row)
        rows = scene["objects"]
        scene["summary"] = {
            "object_count": len(rows),
            "candidate_count": sum(row["candidates"] for row in rows),
            "objects_with_lower_floor_candidates": sum(row["priority_pool"] == "lower" for row in rows),
            "objects_with_more_than_32_priority_candidates": sum(row["priority_candidates"] > count for row in rows),
            "weighted_std_all": {term: float(np.mean([row["weighted_std_all"][term] for row in rows])) for term in TERMS},
            "weighted_std_priority": {term: float(np.mean([row["weighted_std_priority"][term] for row in rows])) for term in TERMS},
            "B_D_correlation": summarize(rows, "B_D_correlation"),
            "ablations": {label: {metric: summarize([row["ablations"][label] for row in rows], metric)
                                  for metric in ("rank_correlation", "top32_overlap", "top32_overlap_when_competing", "sample_population_overlap")}
                          for label in ABLATIONS}}
        report["scenes"].append(scene)
        print(json.dumps({"placed": placed_count, "objects": len(rows),
                          "weighted_std_priority": scene["summary"]["weighted_std_priority"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
