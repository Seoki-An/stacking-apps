# Excavator action survival after a placement

This archive contains reports and JSON evidence. Experimental scripts, local `.benchmarks/` inputs and native builds are not included. Command examples record the original local setup and require those separate artifacts.

On 2026-10-07, five CUDA searches supplied 17 physically valid, still-available sibling proposals. After committing each search's selected placement in simulation, 11 passed long physical validation at their unchanged poses. Fresh pose and force solves produced 14 passing proposals, including three substantial repairs. Six passed the strict floor policy as well; the other eight required the existing floor fallback.

This supports retaining candidate proposals for re-evaluation. It does not establish that old forces, rewards, backed-up values, evaluated states or descendant subtrees remain valid.

## Measurement

Each trial used the current `mcts_cuda.yml`, seed 0, 1,024 iterations and a fresh tree. All searches completed without a timeout. The current profile limits the 0- and 10-stone scenes to depth 1 (`initial_depth1_steps: 12`); the other scenes use maximum depth 4. Current physical and reward profiles were used, including inward-orientation weight 6.0. These settings differ from the earlier CPU/CUDA comparison.

The 10-, 15- and 20-stone states came from `.stacks/excavator_plan10_gpu.stack`; the 25-stone state came from `.stacks/excavator_plan9_gpu.stack`. The empty scene resets the geometry using the same inventory as the 10-stone checkpoint. Every trial has 49 total inventory objects. Hardware was a Ryzen 9 9900X and RTX 5060 Ti, with BLAS/OpenMP threads limited to one. Trials ran sequentially in separate processes.

The benchmark examines only the final root's retained siblings, excluding the selected node and proposals for its consumed object. It does not include discarded CEM populations, retired root pools or deeper nodes.

For each scene:

1. Run MCTS and commit the selected node's simulated state.
2. Independently replay every available sibling in the original scene using LONG validation. Only passing candidates enter the survival denominator. This includes candidates rejected by final selection solely for lacking a validated descendant.
3. Replay these candidates at their unchanged poses in the committed scene. Record physical pass/fail only: old force costs are still attached, so this is not certification of their new rewards or equilibrium costs.
4. Start a fresh CUDA pose/force solve from each old solved pose, with support metadata refreshed for the new scene. Apply both the strict filters and, separately, the existing floor fallback. Long-validate the resulting admissible placements and recompute their rewards.

Each sibling is evaluated independently after the selected placement. Passing siblings are alternatives; the experiment does not show that all can be placed together. No motion planning, hardware execution, descendant validation or CPU cross-validation is included.

## Results

| Already placed stones | Available siblings | Previously physically valid | Unchanged pose physical passes | Re-solved passes under strict policy | Re-solved passes including fallback | Small-change passes including fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 31 | 8 | 7 | 0 | 7 | 7 |
| 10 | 30 | 2 | 0 | 0 | 1 | 0 |
| 15 | 30 | 2 | 2 | 2 | 2 | 2 |
| 20 | 30 | 2 | 0 | 2 | 2 | 0 |
| 25 | 30 | 3 | 2 | 2 | 2 | 2 |
| Total | 151 | 17 | 11 | 6 | 14 | 11 |

“Small change” is a descriptive cutoff of at most 1 cm translation and 1 degree rotation, chosen to separate the observed millimetre changes from the 28–64 cm successful repairs. It is not a validated independence criterion. Four of these eleven small-change survivors also pass the strict policy.

Relative to the 17 valid, available proposals, unchanged physical survival is 64.7%, re-solved survival including fallback is 82.4%, and strict re-solved survival is 35.3%. Relative to all 151 available root siblings, the latter two figures are only 9.3% and 4.0%: most retained siblings already fail original-scene physics and are not useful work lost because of the new placement.

These are correlated candidates from one search per scene, not 17 independent trials. In particular, both survivors at 15 stones are variants for object 57; the two at 20 stones are also variants for object 57; and the two at 25 stones are variants for object 58. Do not interpret the aggregate percentages as general success probabilities.

## What survived and what changed

- **Empty scene:** seven proposals remain physically viable with only millimetre pose changes. All eight original valid proposals are rejected by the strict `active_floor_disconnected_active_layer` filter after the first placement establishes an occupied region. Their viability is therefore available through the existing fallback, not immediately through ordinary strict generation.
- **10 stones:** the two original valid siblings lie 0.10 and 0.28 m from the selected action in XY. Both unchanged placements fail `target_motion_limit`. Re-solving rescues one by moving it 0.64 m, but the result requires the above-active-layer fallback. This is a repaired proposal, not a nearly unchanged placement.
- **15 stones:** both survivors lie about 4.63 m from the selected action. Their poses change by only 0.14–0.35 mm and 0.007–0.030 degrees. Equilibrium cost changes from about 0.0012 to 0.0128, while rewards change by approximately -0.015. Both pass strict filtering and long validation. This is the clearest example of the proposed approximate independence.
- **20 stones:** the siblings lie 0.41–0.48 m from the selected action. Both fail unchanged, for `stone_outside_target` and `target_motion_limit`. Fresh solves move them 0.28 and 0.36 m; both then pass strict filtering and physics. The old poses are useful starting points, but the solved placements require substantial repair.
- **25 stones:** two proposals about 4.92–5.06 m away survive, with 1.38–2.39 mm pose changes. Both had been rejected by final selection for `missing_validated_descendant`, despite valid immediate physics. They remain useful candidates for further search. Another candidate 4.90 m away still fails `place_robustness`, showing that distance alone is insufficient.

The reported `c_feq` is the application's normalized equilibrium residual: the maximum net-wrench L1 norm across reported bodies, divided by the maximum inventory mass. It is not a contact-force vector distance. Adding a body can change the worst residual even if a distant placement's geometry barely changes. The experiment re-solves pose and forces together; it does not isolate a fixed-geometry force solve or control for the extra optimization pass in an unchanged scene.

## Cost of refreshing the proposals

| Placed stones | Original search s | Warm solve and filters s | Rebuilt LONG validation s | Total refresh s | Passing proposals including fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.369 | 0.092 | 0.180 | 0.272 | 7 |
| 10 | 4.787 | 0.098 | 0.572 | 0.670 | 1 |
| 15 | 62.453 | 0.069 | 0.476 | 0.545 | 2 |
| 20 | 75.520 | 0.098 | 0.605 | 0.703 | 2 |
| 25 | 79.791 | 0.127 | 1.036 | 1.163 | 2 |

Refresh includes initialization, a fresh scheduled CUDA pose/force solve, both filter checks and long validation. It excludes the diagnostic original-scene replay, unchanged-pose replay and search that created the bank. The services and caches are already warm. A production bank could retain original validation status instead of performing the diagnostic baseline replay.

These times measure refresh cost, not saved search time. There is no equal-work fresh-generation control or complete next-search reuse-policy comparison here. Reused proposals do not replace fresh exploration, and accepting an immediate placement does not satisfy required descendant validation.

## Implications for a reuse strategy

Retain a bounded bank of physically successful proposals, including those rejected only for missing descendants. Remove consumed objects, reapply the current floor policy, refresh the solve and reward, and validate the new transition. Candidates admitted only by floor fallback must retain that distinction. Large repairs can still be useful, but should be counted separately from small-change reuse.

The subsequent [controlled next-search comparison](excavator-proposal-reuse.md) measures this proposal bank while preserving the same selected subtree in both variants.

The next controlled test should compare the complete next search with and without this bank, preserving fresh-generation capacity and all descendant requirements. Measure time to valid selection, new useful nodes per second, candidate diversity and selected return. Transferred proposals should start with fresh state-specific statistics; these results do not justify copying visits or backed-up values.

RAVE, or Rapid Action Value Estimation, addresses a related statistical question. It uses the all-moves-as-first heuristic to share outcomes of actions played later in a trajectory with earlier states. MC–RAVE blends that estimate with ordinary state-specific Monte Carlo values, reducing the heuristic's weight as direct evidence grows. This transfers value evidence; it does not certify equivalent physics or transfer evaluated states. See [Gelly and Silver, 2011](https://www.lamsade.dauphine.fr/~cazenave/mcts-gelly-silver.pdf).

## Reproduction

Use the trusted state exports from the existing profile benchmark:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
QT_QPA_PLATFORM=offscreen python tests/benchmark_excavator_action_survival.py \
  --state .benchmarks/excavator-profiles-20261007/states/15.pkl \
  --config apps/excavator/config/mcts_cuda.yml \
  --seed 0 --output .benchmarks/action-survival-repeat/15.json
```

Use the 10- or 20-stone exports for those scenes. For 25 stones, use `.benchmarks/excavator-profiles-25-20261007/states/25.pkl`. For the empty case, use the 10-stone export with `--empty`. Existing output files are not overwritten.

- `tests/benchmark_excavator_action_survival.py` (local experimental source, not included)
- [Portable results, candidate records, profile snapshots and provenance](benchmarks/excavator-action-survival-20261007.json)
- Local raw outputs: `.benchmarks/action-survival-20261007/`

Only the diagnostic benchmark and this report/results artifact were added. Production search and configuration files were not changed.
