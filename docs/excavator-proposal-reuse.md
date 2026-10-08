# Excavator sibling proposal reuse comparison

This archive contains reports and JSON evidence. Experimental scripts, local `.benchmarks/` inputs and native builds are not included. Command examples record the original local setup and require those separate artifacts.

Root promotion now resets generation bookkeeping while preserving the selected subtree. Both measured roots previously carried 48 attempts into a fresh CEM pool of only 32 candidates, skipping the entire pool. The corrected searches start from offset zero and admit 23 and 29 fresh root candidates.

With that fix, the benchmark-only proposal bank modestly improves throughput at 16 stones and reduces it at 26 stones. The earlier large benefit at 26 stones was confounded by the skipped pool. Production now includes the promotion correction; bank admission remains experimental and configuration files were not changed.

## Corrected comparison

The subsequent [native CPU promotion check](excavator-cpu-promotion.md) reproduces the same bookkeeping bug with CPU-generated subtrees. The shared correction applies to both backends; throughput and selection effects vary by scene.

The same saved pre-promotion subtrees and banks are used for both variants. The benchmark now invokes the actual `MCTSSession.promote` and `session.search` lifecycle. Recorded root attempts change from 48 to zero in every trial. All trials complete 1,024 iterations without a timeout and select an action with a validated descendant path.

| Stones at search start | Variant | Search s | New nodes | New nodes per second | Unique new physical passes | Physical passes per second |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | Baseline | 85.10 | 1201 | 14.11 | 169 | 1.986 |
| 16 | Reuse | 78.14 | 1158 | 14.82 | 175 | 2.240 |
| 26 | Baseline | 98.68 | 632 | 6.40 | 48 | 0.486 |
| 26 | Reuse | 71.84 | 408 | 5.68 | 19 | 0.264 |

At 16 stones, reuse increases expansion throughput by 5.0% and physical-pass throughput by 12.8%. At 26 stones, it decreases them by 11.3% and 45.6%, despite finishing sooner. These are single-seed paired observations, not established general gains or regressions.

At 16 stones the first reuse pool retains 22 fresh candidates plus one bank proposal, compared with 23 fresh candidates in the baseline. Root long validation accepts three versus two. At 26 stones the reuse pool retains 26 fresh candidates plus three bank proposals, compared with 29 fresh candidates in the baseline. Root long validation accepts three versus four. Reuse can therefore displace a better fresh alternative; a proposal's previous physical success does not justify a reserved root slot by itself.

Both variants preserve and successfully revalidate the same nine existing root children at 16 stones, and the same three at 26 stones. They select the same next object and immediate reward in each pair: object 51 / 4.176 and object 53 / 1.741. Neither selected action comes from the bank. Backup estimates and explored descendants can differ.

The original 26-stone baseline generated 34 nodes at 1.76 nodes/s. Restoring fresh generation raises this to 632 at 6.40 nodes/s, with longer elapsed time. That is the main improvement in this scene. The promotion change corrects exploration accounting; it is not a guarantee of higher throughput in every scene, as the 16-stone results demonstrate.

The fix clears the new root's attempt count, exhausted flag, generation rejection counts and external-seed attempt offset. It does not clear descendants, descendant generation accounting or visits/values during promotion. The configured statistics-reset and long-revalidation policies still run normally. A regression test exercises a real CEM pool after promotion with both stale cursor and exhausted-flag cases, retaining a child and grandchild throughout.

Keep the promotion correction. Further bank experiments should rank refreshed proposals competitively or use them as CEM starting points, and test more seeds before adding a production admission policy. The corrected result does not support universally reserving root slots for the bank.

The subsequent [competitive CEM seeding experiment](excavator-cem-proposal-reuse.md) tests that alternative with two search seeds per scene and no reserved root slots.

The later [additive admission experiment](excavator-additive-proposal-reuse.md) retains every fresh CEM candidate and appends refreshed siblings in a separate bounded allowance. It measures the policy that avoids displacing fresh candidates entirely.

[Corrected portable results and provenance](benchmarks/excavator-proposal-reuse-promoted-20261007.json) record all four trials. Local raw outputs are in `.benchmarks/proposal-reuse-promoted-20261007/`. The following sections retain the initial experiment for comparison; its measurements must not be treated as results with the fix.

## Experiment

The source checkpoints contain 15 and 25 placed stones, as in the [action survival experiment](excavator-action-survival.md). A seed-0 CUDA search first selects and simulates one placement. It exports the selected subtree plus physically valid siblings whose objects remain available. The measured next searches therefore start at **16 and 26 stones**, respectively.

The selected subtrees contain 255 and 249 nodes. Each baseline/reuse pair loads the identical saved subtree in a fresh process, reconstructs observations, and revalidates preserved children. Old tree structure and the normal statistics-reset policy are shared between variants. The runs use the current CUDA profile, 1,024 iterations, depth 4 and required descendant validation; every trial completed without a timeout.

The baseline runs the existing planner. The experimental wrapper:

1. Runs normal fresh CEM generation with its original population and round budgets.
2. Re-solves bank candidates from their previous solved poses against the new state, refreshes support information and planar scores, and applies the current filters. Old force costs, rewards, visits and backed-up values are not transferred.
3. Reserves at most one quarter of the requested root slots, capped at eight, for these proposals. It preserves capacity for at least three quarters of fresh candidates when they are available. It applies the existing per-object and geometric duplicate rules, including comparison against the preserved root children.
4. Submits admitted proposals as ordinary unevaluated nodes. The existing search performs long physical validation, expansion and descendant validation. A strict floor rejection cannot enter the strict search through the bank; any later fallback uses the existing fallback filters.

Bank solve results are reused across root retries within this one unchanged root state. Previously admitted proposals are not injected repeatedly. The policy is implemented only in `tests/benchmark_excavator_proposal_reuse.py` (local experimental source, not included).

Trials ran sequentially on the Ryzen 9 9900X / RTX 5060 Ti using the project direnv Python, with BLAS/OpenMP threads limited to one. At 16 stones, baseline ran first. At 26 stones, reuse ran first; the pair was then repeated in the opposite order.

## Initial paired results

| Stones at search start | Variant | Search s | New nodes | New nodes per second | Unique new physical passes | Physical passes per second |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | Baseline | 67.15 | 1060 | 15.79 | 152 | 2.26 |
| 16 | Reuse | 69.49 | 1086 | 15.63 | 144 | 2.07 |
| 26 | Baseline | 19.36 | 34 | 1.76 | 4 | 0.21 |
| 26 | Reuse | 46.34 | 204 | 4.40 | 15 | 0.32 |

New nodes means the search's `generated_candidates` count, including nodes that subsequently fail. Preserved nodes are excluded. Search time includes bank refresh, generation, physics, backup, revalidation and final selection; predecessor search, artifact loading, observation reconstruction and serialization are excluded.

The physical-pass metric deduplicates newly evaluated transitions by exact parent scene poses, placed-object indices and candidate action pose. It excludes transitions already represented by inherited edges. Long results take precedence over short results, and the latest result at equal fidelity determines pass/fail. This is a measure of distinct physically successful transitions, not a count of fully validated descendant paths or eventual executable actions. Some passes have only short-fidelity evidence.

At 16 stones, expansion throughput decreased by 1.0% and physical-pass throughput by 8.5%. At 26 stones, expansion throughput increased by **2.51 times** and physical-pass throughput by **1.57 times**, while total search time increased by **2.39 times**. This distinction matters when the objective is nodes per second rather than shortest decision latency.

Repeating the 26-stone pair in the opposite order reproduced the same counts and selection. Baseline took 19.44 s for 34 new nodes and 4 physical passes (1.75 and 0.206 per second); reuse took 46.36 s for 204 nodes and 15 passes (4.40 and 0.324 per second). Reproducibility does not remove the promotion confound.

## Why the effect differs

| Stones | Preserved root children revalidated successfully | Available bank proposals | Bank proposals admitted | Admitted proposals passing root LONG validation | Fresh root candidates returned |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 9 | 2 | 1 | 1 | 0 |
| 26 | 3 | 3 | 3 | 2 | 0 |

Both searches already have a useful preserved subtree. Fresh root CEM returns zero candidates for the remaining root slots in these trials, despite retaining a pool of 32. The bank therefore adds alternatives without displacing any fresh candidate in the measured cases. The reserved fresh-capacity rule was also checked separately with a populated synthetic fresh pool.

At 16 stones, the preserved root already offers nine valid alternatives. Adding one more changes exploration but does not improve measured throughput. At 26 stones, only three preserved alternatives remain. Two surviving bank proposals open additional branches, increasing generated nodes from 34 to 204 and unique new physical passes from 4 to 15. All 1,024 iterations still run in the baseline, but many do not create new nodes after generation is marked exhausted.

The bank's pose solve, filtering and admission work takes about 0.076 s at 16 stones and 0.130 s at 26 stones. These figures exclude its ordinary long validation and the additional descendant searches it enables. The extra exploration accounts for most of the increased elapsed time.

At 16 stones both variants select object 51 with immediate reward 4.176 and target IoU 0.1502. At 26 stones both select object 53 with reward 1.741 and IoU 0.2263. Neither selected action itself comes from the bank. Every selected action has a validated descendant path. The experiment shows a throughput effect in one scene, not improved final action quality.

## Promotion issue identified in the initial comparison

The initial comparison identified the following bookkeeping problem, corrected above:

- `MCTSSession.promote` preserves the selected node's expansion counters along with its subtree.
- `_expand_many` derives a generation `attempt_offset` from those counters. It becomes 48 for both measured promoted roots.
- CEM builds and ranks a new pool of 32, then returns `ranked[start:start + max_candidates]`, with `start = min(48, 32) = 32`. The empty result marks generation exhausted.
- All immediate preserved children pass revalidation in these trials, so the rejection-triggered counter reset does not run.

At 16 stones, the CEM rounds accepted 79, 67 and 68 proposals before ranking/deduplication. At 26 stones they accepted 89, 88 and 83. These are not cases where generation could find no eligible poses: the final retained pool contains 32 in both cases, and pagination skips them.

The implemented correction starts root-specific generation accounting afresh when a child becomes the root, while retaining its children, normal subtree revalidation and the configured statistics policy. The regression test promotes a node with 48 previous attempts and verifies that the first root CEM request consumes the new pool from offset zero. The corrected comparison above uses this production lifecycle.

Then repeat across seeds and more checkpoints, separating roots with few valid children from roots that already have many. Conditional bank admission remains a hypothesis; the current confounded comparison cannot establish its threshold.

This version deliberately retains the full fresh CEM workload. Reducing generation cost requires a separate test of using the bank within CEM's initial population or mutation parents, with an explicit fresh-sampling allowance. That would test computational reuse as well as the exploration benefit measured here.

The benchmark does not cover observed hardware deviations, motion planning, repeated bank reuse across several subsequent placements, or CPU search. Same-seed repeats measure reproducibility of this workload, not generalization across scenes.

## Reproduction and artifacts

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export QT_QPA_PLATFORM=offscreen
python tests/benchmark_excavator_proposal_reuse.py prepare \
  --input .benchmarks/excavator-profiles-20261007/states/15.pkl \
  --output .benchmarks/proposal-reuse-repeat/15.pkl
python tests/benchmark_excavator_proposal_reuse.py baseline \
  --input .benchmarks/proposal-reuse-repeat/15.pkl \
  --output .benchmarks/proposal-reuse-repeat/15-baseline.json
python tests/benchmark_excavator_proposal_reuse.py reuse \
  --input .benchmarks/proposal-reuse-repeat/15.pkl \
  --output .benchmarks/proposal-reuse-repeat/15-reuse.json
```

Use the 25-stone export from `.benchmarks/excavator-profiles-25-20261007/states/25.pkl` for the other case. Inputs must be trusted local pickles. Existing outputs are not overwritten. Prepared subtrees retain complete states and can be large; setup and loading are excluded from the reported search time.

[Original portable results](benchmarks/excavator-proposal-reuse-20261007.json) and [corrected results](benchmarks/excavator-proposal-reuse-promoted-20261007.json) contain profile snapshots, source/native hashes, paired artifact hashes, admission counts, search statistics and selected outcomes. Original preparation artifacts and trial outputs are in `.benchmarks/proposal-reuse-20261007/`. The current reproduction commands invoke corrected promotion.
