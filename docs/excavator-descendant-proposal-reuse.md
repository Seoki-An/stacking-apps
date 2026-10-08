# Excavator discarded-descendant proposal reuse (archived)

The experimental scripts have been removed from the active project. Command examples below are historical records, not supported commands. This archive retains reports and JSON evidence; local `.benchmarks/` inputs and native builds are not included.

**Retired on 2026-10-08 with the combined policy.** The dedicated diagnostic and its tests have been removed from the active project. Measurements remain available here and in the linked JSON artifact; the measured source is archived locally under `.benchmarks/combined-reuse-20261008/source_archive/`.

Discarded deeper branches supplied additional usable proposals in the two 15→16-stone cases: five and six refreshed proposals passed long validation. The two 25→26-stone cases supplied none outside the already-preserved subtree. This establishes a possible source of extra proposals, not an improvement in next-search throughput or placement quality.

## Method

This follows the [broader root-bank audit](excavator-broader-additive-proposal-reuse.md), which found no short-only root siblings because roots receive long validation upfront. This experiment instead examines retained descendants at depth 2 or greater in discarded root branches.

Four native CUDA predecessor searches ran on 2026-10-08, using saved 15- and 25-stone scenes and seeds 0 and 1. All used 49 inventory objects, 1,024 iterations, maximum depth 4, and the current excavator CUDA configuration. All completed without timeouts. Their recorded non-timing search statistics exactly match the corresponding predecessor searches in the previous audit. Runs were sequential on a Ryzen 9 9900X / RTX 5060 Ti with BLAS/OpenMP threads limited to one.

The benchmark applies these checks before evaluating a proposal in the new scene:

1. Require a short pass with no recorded long evaluation for that exact parent scene and action. Match against the final retained node's current parent-state key. Exclude the selected subtree, which is already preserved, and objects consumed by the selected placement.
2. Require support-source IDs, a contact-gap tolerance and contact data. Reject proposals with direct supports absent from the new scene.
3. Build an undirected stone-contact graph using the existing contact-gap tolerance. Reject proposals whose contact component contains an absent parent-scene stone. Ground contacts do not connect unrelated stones through a shared ground vertex. This is a conservative contact screen, not proof of force independence.
4. Apply the existing CEM duplicate rule against all actions in the preserved subtree, the original long-valid sibling bank, and earlier proposals in the diagnostic cohort. This uses the original actions' initial XY positions and orientations, as CEM does.
5. For the eligible cohort, run long evaluation in each original parent scene as a diagnostic control. Separately, re-solve each proposal in the selected placement's scene, refresh its planar score, apply the normal strict filters, and long-evaluate the refreshed result. No floor fallback is enabled.

The original-parent long controls do not influence cohort selection or the new-scene evaluations. No proposal is injected into a subsequent search. Old states, rewards, forces and backup statistics are never transferred to the new scene.

## Results

| Stones before→after placement | Seed | Matched short-only descendants | Already preserved | Discarded short-only descendants | Eligible after screens | Pass original-parent long | Pass refresh filters | Pass new-scene long |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15→16 | 0 | 36 | 4 | 32 | 8 | 8 | 7 | 5 |
| 15→16 | 1 | 56 | 16 | 40 | 7 | 7 | 7 | 6 |
| 25→26 | 0 | 7 | 7 | 0 | 0 | 0 | 0 | 0 |
| 25→26 | 1 | 8 | 8 | 0 | 0 | 0 | 0 | 0 |

The matched counts are narrower than the previous audit's counts of all recorded short-only transitions. They include only descendants still present in the final tree whose exact current parent-scene/action key matches short-only passing evidence. Records associated with earlier parent-state versions or absent nodes are not recovered. Thus, zero at 26 stones means zero additional eligible proposals under this collection rule, not proof that all discarded computation is useless.

Across the two 16-stone cases, 15 proposals pass the screen and 11 survive refreshed long validation: 73.3%. All 15 pass the original-parent long control. One refresh fails with `posegen_geometry_failure`; three refreshed candidates fail long validation with `place_robustness`. The screen therefore does not eliminate the need for fresh physical validation.

The 72 discarded short-only descendants in those two cases are reduced as follows. These are ordered, mutually exclusive screening outcomes:

| Outcome | Seed 0 | Seed 1 |
| --- | ---: | ---: |
| Object consumed by selected placement | 2 | 2 |
| Missing direct support | 12 | 20 |
| Connected indirectly to a missing stone | 3 | 2 |
| Duplicate of a preserved or long-bank proposal | 5 | 7 |
| Duplicate within the new cohort | 2 | 2 |
| Eligible unique proposal | 8 | 7 |

No eligible-cohort candidates required unknown-contact or failed-state exclusions. Of the 15 eligible proposals, 14 originate at depth 4 and one at depth 2. A survivor in seed 0 moves 0.290 m during re-solving; the largest translation among seed 1 survivors is 0.0158 m. Passing the dependency screen does not imply that the old pose remains suitable unchanged.

## Cost and diversity

| New scene | Seed | Collection s | Refresh s | New-scene long s | Collection + refresh + new validation s | Original-parent long control s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 stones | 0 | 0.008 | 0.197 | 0.877 | 1.082 | 1.036 |
| 16 stones | 1 | 0.008 | 0.133 | 1.348 | 1.489 | 0.809 |

The empty 26-stone cohorts cost about 0.004–0.005 s to collect and skip native evaluation. These measurements exclude predecessor search, environment setup and new-root observation construction. Refresh includes pose initialization, solving, score refresh and strict filtering. The original-parent long control is additional diagnostic cost; an eventual proposal policy would not need that control.

The five seed-0 survivors cover four object IDs: 50, 2, 42, and two alternatives for 52. The six seed-1 survivors cover only three object IDs: 52, 45, and four alternatives for 44. Applying a two-per-object limit to these successful descendant proposals alone would leave at most five and four, respectively. That is a post-hoc diversity bound, not a tested admission policy. A combined bank also contains long-valid siblings, and fresh-root duplicate checks, the eight-proposal cap and ranking can reduce admission further. No deduplication after re-solving or against the next fresh CEM pool has been measured here.

These costs and counts do not establish a nodes/s gain. The added roots still consume exploration and descendant-validation budget, and this experiment does not measure their effect on the chosen action or validated path return.

## Full-search follow-up

The [combined-bank comparison](excavator-combined-proposal-reuse.md) subsequently measured full next searches. At 16 stones it lowered pooled nodes/s by 7.0% and physical passes/s by 10.7%; at 26 stones the banks were identical. The proposal-survival results therefore should not be interpreted as an established throughput improvement.

That comparison kept the complete fresh CEM pool, the existing total allowance of eight extra proposals, and the existing two-per-object bank limit. It ranked and deduplicated refreshed proposals before admission, without using the diagnostic long outcomes to choose them, and measured throughput alongside selection quality.

The evidence does not support enabling descendant reuse in production for a throughput objective or expecting a benefit at 26 stones under this collection rule.

## Historical validation

Twelve focused tests passed, covering direct and indirect missing contacts, ground separation, contact-gap tolerance, missing evidence, shared-stone pose changes, preserved branches, consumed objects, duplicates and later long failures. Existing additive admission and session tests also pass. Source, profile, input-state and pinned native/wrapper hashes were unchanged during the experiment.

The runs use the same isolated native binary and compatible Python wrapper as the previous audit. This avoids unrelated changes in the editable `stacking-core` checkout. Exact hashes and all per-candidate outcomes are in the [portable results and provenance](benchmarks/excavator-descendant-proposal-reuse-20261008.json).

The former descendant-diagnostic command is no longer available. Its raw local results remain under `.benchmarks/descendant-reuse-20261008/`.

Production search and configuration files are unchanged. The results are historical records of the retired experiment.
