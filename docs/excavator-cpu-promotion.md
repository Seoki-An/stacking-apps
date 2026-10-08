# Excavator CPU root-promotion check

The experimental scripts have been removed from the active project. Command examples below are historical records, not supported commands. This archive retains reports and JSON evidence; local `.benchmarks/` inputs and native builds are not included.

The root-promotion bug also affects the native CPU profile. Two saved-scene comparisons on 2026-10-07 reproduce both blocked generation and a skipped CEM pool. The existing shared promotion correction fixes both; no CPU-specific production change or configuration change was needed.

## Method

Use the current `apps/excavator/config/mcts_cpu.yml`: CPU pose generation and simulation, 12 workers, 1,024 iterations, depth 3, progressive widening enabled, 12 root children, CEM keep 12, and required descendant validation. BLAS/OpenMP threads are limited to one.

First run a native CPU predecessor search from the saved 15- and 25-stone scenes. Its selected placement produces the measured 16- and 26-stone scenes. Export the selected subtree before promotion. These are CPU-generated subtrees, not the CUDA predecessors used in the sibling-reuse experiments. Each scene retains the 49-object inventory.

For each scene, load the identical exported subtree in two separate processes:

- **Legacy:** invoke production promotion, then restore the pre-fix attempt count, exhausted flag, generation rejection counts and external-seed offset in the benchmark only.
- **Fixed:** invoke current production promotion normally.

Both variants preserve and revalidate the subtree. No sibling bank is injected. The 16-stone pair runs legacy first; the 26-stone pair runs fixed first. All six native searches, including the two predecessors, complete 1,024 iterations without timeout. Timing includes revalidation, generation, evaluation, retries and selection, but excludes setup, artifact loading and observation reconstruction.

## Reproduced failures

The CPU root attempt budget is `root_max_children * attempt_multiplier = 12 * 3 = 36`.

| Stones | Inherited attempts | Legacy behavior | Fixed behavior |
| --- | ---: | --- | --- |
| 16 | 48 | Attempts exceed the root budget of 36. No root CEM call occurs. | Starts at offset 0; requests return 10 and then 2 candidates. |
| 26 | 31 | CEM retains 12 candidates, but offset 31 returns zero and reports exhaustion. | Starts at offset 0; requests return 11 and then 1 candidate. |

Both inherited exhausted flags are false, so these native reproductions do not depend on a stale exhausted flag. The regression test separately covers that flag.

Both variants successfully revalidate the same two inherited children at 16 stones and the same one at 26 stones, without clearing their subtrees. At 26 stones, legacy later recovers through three root retry rounds, generating 36 candidates in replacement pools. This recovery does not remove the initial bug. The fixed trial requires no root retry rounds. Neither scene uses floor fallback in the measured next searches.

## Throughput

| Stones | Variant | Search seconds | New nodes | New nodes/s | Unique new physical passes | Physical passes/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | Legacy | 41.57 | 250 | 6.013 | 27 | 0.649 |
| 16 | Fixed | 57.10 | 262 | 4.589 | 28 | 0.490 |
| 26 | Legacy | 82.87 | 148 | 1.786 | 9 | 0.109 |
| 26 | Fixed | 57.94 | 144 | 2.485 | 4 | 0.069 |

New nodes count generated candidates, including later failures and candidates retired during retries; inherited nodes are excluded. Thus this is not final tree size. Physical passes deduplicate new transitions by exact parent scene and action, excluding inherited transitions. Long results take precedence over short results; some passes have only short-fidelity evidence. They are not a count of executable placements or validated paths.

At 16 stones, raw throughput drops 23.7%. The fixed run evaluates previously skipped alternatives: its initial root long-validation batch rejects all 10 new candidates. It adds 12 generated nodes overall, with one additional distinct physical pass. The selected action remains object 32, reward 2.6604, validated path return 6.3232.

At 26 stones, raw throughput rises 39.2%, while physical-pass throughput drops 36.4%. The fixed run's initial root validation accepts two of 11 new candidates. The legacy run instead performs three later root retries, taking more time and finding more physically successful alternatives. Selection differs: legacy chooses object 3, reward 2.2607 and validated path return 3.3110; fixed chooses object 52, reward 1.2336 and validated path return 2.6169. Both selected actions have a validated descendant path.

The correction restores intended exploration accounting; it does not guarantee faster search, more physical passes, or better action quality. These are single-seed comparisons on two CPU predecessor histories, not general performance estimates or a CPU-versus-CUDA speed comparison. Any policy to skip or limit new root generation should be evaluated explicitly rather than relying on stale bookkeeping.

## Validation and reproduction

The promotion regression now covers root pools of both 12 and 32 candidates, with and without the inherited exhausted flag, while retaining a child and grandchild. The session, legacy-regression and action-generation suites pass: 57 tests. The benchmark records every root generation request's offset, pool size, returned count and exhaustion status. Paired input hashes, profile snapshots, inherited edge counts, revalidation outcomes, CPU backend settings and complete iteration budgets were checked.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/stacking-mpl
python tests/benchmark_excavator_proposal_reuse.py prepare \
  --config apps/excavator/config/mcts_cpu.yml \
  --input .benchmarks/excavator-profiles-20261007/states/15.pkl \
  --output .benchmarks/cpu-promotion-repeat/15.pkl
python tests/benchmark_excavator_proposal_reuse.py baseline --legacy-promotion \
  --config apps/excavator/config/mcts_cpu.yml \
  --input .benchmarks/cpu-promotion-repeat/15.pkl \
  --output .benchmarks/cpu-promotion-repeat/16-legacy.json
python tests/benchmark_excavator_proposal_reuse.py baseline \
  --config apps/excavator/config/mcts_cpu.yml \
  --input .benchmarks/cpu-promotion-repeat/15.pkl \
  --output .benchmarks/cpu-promotion-repeat/16-fixed.json
```

For the 26-stone comparison, prepare from `.benchmarks/excavator-profiles-25-20261007/states/25.pkl`. Only use trusted local pickle inputs. The legacy flag is restricted to baseline diagnostics; production code stays fixed.

- [Portable results, profile snapshots and source hashes](benchmarks/excavator-cpu-promotion-20261007.json)
- Local artifacts: `.benchmarks/cpu-promotion-20261007/`
- [Previous CUDA promotion and proposal-reuse comparison](excavator-proposal-reuse.md)
