# Excavator competitive CEM proposal reuse

The experimental scripts have been removed from the active project. Command examples below are historical records, not supported commands. This archive retains reports and JSON evidence; local `.stacks/benchmarks/` inputs and native builds are not included.

Adding refreshed sibling proposals to CEM's initial population did not improve raw expansion throughput in the two measured scenes. At 16 stones it increased physically successful new transitions per second by about 10% across two search seeds. At 26 stones none of the seeds became mutation parents or reached the tree, and search counts were unchanged. The selected next actions and immediate rewards were unchanged in every pair.

This is a benchmark-only experiment. It does not add a production reuse policy or change platform configurations. The earlier [root-promotion correction](excavator-proposal-reuse.md) is active in all trials.

## Method

The source checkpoints have 15 and 25 placed stones. Their seed-0 predecessor searches supply a selected subtree and a sibling bank; committing that selected placement gives the measured **16- and 26-stone** scenes. The banks contain two and three proposals, respectively, with 49 total objects in each inventory. Each baseline/seeded pair loads exactly the same saved subtree and bank. Both variants invoke production promotion and long revalidation of the preserved subtree.

The next-search seed varies between 0 and 1, controlling the search RNG and CEM seed. The predecessor subtree and bank remain fixed, so this tests next-search variability conditional on those two histories. There are eight trials: two scenes, two seeds and two variants. Each uses 1,024 iterations, the current CUDA profile and unchanged descendant validation. All complete without timeouts and select actions with validated descendant paths.

The seeded variant:

1. Runs the usual initial fresh generation of 96 proposals.
2. Re-solves the bank against the new scene, refreshes support metadata and planar scores, and reapplies the current eligibility and duplicate checks. Bank solutions are cached only within this unchanged root state.
3. Adds eligible bank proposals at the initial CEM ranking boundary. Normal deduplication and ranking retain at most 96 candidates. Elite selection still applies the two-per-object limit and selects up to 32 parents.
4. Runs the usual two mutation rounds, each with 48 fresh attempts and up to 48 mutation attempts. A refreshed solution becomes the mutation centre only if it wins a parent position. Final root retention and all physical validation remain unchanged.

There are **no reserved root or parent slots and no score bonus for bank proposals**. Fresh sampling counts and offsets stay unchanged. The experiment adds up to three warm solves rather than replacing fresh solves, so it tests search guidance, not avoided fresh-generation work. Extra bank refresh and bookkeeping time is included in search wall time.

Injection happens at initial ranking in the benchmark subclass, without altering production CEM. Standard round-zero attempted/accepted counts describe fresh generation; the separate `reuse_calls` records describe extra bank seeds, deduplication, elite retention, mutation attempts and returned seed lineage. This keeps bank attempts from incorrectly advancing the fresh sampling cursor.

Trials ran sequentially on a Ryzen 9 9900X and RTX 5060 Ti. BLAS/OpenMP threads were limited to one. Baseline ran first for scene 16 / seed 0 and scene 26 / seed 1; seeded CEM ran first for the other pairs. Loading, observation reconstruction and output serialization are excluded from search time. No motion planning or hardware execution is included.

## Results

| Stones | Seed | Variant | Search s | New nodes | Nodes/s | New physical passes | Physical passes/s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 0 | Baseline | 84.69 | 1201 | 14.18 | 169 | 1.995 |
| 16 | 0 | CEM seeded | 85.71 | 1181 | 13.78 | 180 | 2.100 |
| 16 | 1 | Baseline | 86.93 | 1256 | 14.45 | 158 | 1.818 |
| 16 | 1 | CEM seeded | 82.87 | 1168 | 14.09 | 173 | 2.088 |
| 26 | 0 | Baseline | 98.39 | 632 | 6.42 | 48 | 0.488 |
| 26 | 0 | CEM seeded | 98.72 | 632 | 6.40 | 48 | 0.486 |
| 26 | 1 | Baseline | 72.41 | 363 | 5.01 | 26 | 0.359 |
| 26 | 1 | CEM seeded | 73.15 | 363 | 4.96 | 26 | 0.355 |

Pooling each scene's two seeds as total count divided by total search time:

| Stones | Baseline nodes/s | Seeded nodes/s | Change | Baseline physical passes/s | Seeded physical passes/s | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 14.316 | 13.934 | -2.7% | 1.905 | 2.094 | +9.9% |
| 26 | 5.825 | 5.789 | -0.6% | 0.433 | 0.431 | -0.6% |

New nodes are generated child nodes, including subsequent physical failures, excluding inherited nodes. Physical passes deduplicate newly evaluated transitions by exact parent scene and action, excluding transitions represented by inherited edges. Long-fidelity results take precedence over short results; the latest result at equal fidelity determines pass/fail. Some passes have only short-fidelity evidence. This metric is not a count of fully validated descendant paths or executable placements.

At 16 stones, seeding produces fewer raw expansions but more distinct physical passes in both seeds. At 26 stones the counts match exactly; the small timing difference includes refresh overhead and timing variability. Two seeds and two fixed histories do not establish a general performance effect.

## What CEM did with the bank

| Stones | Search seed | Bank eligible in first pool | Seeds after deduplication and population retention | Initial seed parents | Seed-lineage mutation attempts | Seed-lineage candidates returned to the first root request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0 | 2 | 1 | 1 | 5 | 2 |
| 16 | 1 | 2 | 1 | 1 | 3 | 1 |
| 26 | 0 | 3 | 3 | 0 | 0 | 0 |
| 26 | 1 | 3 | 3 | 0 | 0 | 0 |

At 16 stones, one seed survives deduplication and ranks 10th or 7th in the first pool. It becomes a mutation parent. The later root generation call has no further eligible bank proposals after checking existing root children.

At 26 stones, the seeds remain in the initial population but do not become parents in either root generation call. Their first-pool global ranks are 31/43/89 and 31/44/92. In particular, rank 31 would fit the global elite count of 32, but loses eligibility under the existing per-object cap: higher-ranked fresh poses for that object already occupy its permitted positions. None of the seeds or their descendants reaches the tree. Consequently, the measured counts, selected action and recorded backup/path values match the paired baseline.

Refresh costs are approximately 0.072 s per 16-stone trial and 0.131–0.133 s per 26-stone trial. These are pose refresh/filter costs, not the total cost of any extra exploration subsequently caused by the seeds.

Both variants select object 51 with immediate reward 4.176 at 16 stones and object 53 with reward 1.741 at 26 stones. No selected action has seed lineage. At 16 stones, the selected validated path return is actually lower with seeding: approximately 13.237 versus baseline 13.952 for seed 0 and 13.513 for seed 1. Thus the increase in physical-pass throughput does not demonstrate improved planning return.

## Decision

Keep this mode experimental. It offers a small, scene-dependent improvement in physically successful exploration, but it has not met the requested raw expanded-nodes/s objective or improved selected placement quality. The previous forced-admission result and this competitive result do not justify reserving root slots or weakening diversity/validation rules for old proposals.

If pursuing this direction further, evaluate whether CEM's ranking predicts physical success well enough to use the surviving proposals effectively. That is a separate scoring experiment, requiring paired measurements of physical-pass throughput and validated path return; increasing bank influence alone is not supported by these results.

## Validation and reproduction

Fifty session/action-generation tests passed. Synthetic checks verified that a low-ranked seed can be rejected completely, a high-ranked seed can become a mutation parent, the final output limit is respected, and the fresh cursor remains unchanged. All eight native searches were checked for matching paired input hashes/profiles, zero promotion offsets, complete iteration budgets and validated selected paths.

Use the trusted pre-promotion artifacts prepared by the previous benchmark:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export QT_QPA_PLATFORM=offscreen
python tests/benchmark_excavator_proposal_reuse.py baseline \
  --input .stacks/benchmarks/proposal-reuse-20261007/15.pkl --seed 0 \
  --output .stacks/benchmarks/cem-reuse-repeat/15-baseline-0.json
python tests/benchmark_excavator_proposal_reuse.py cem_reuse \
  --input .stacks/benchmarks/proposal-reuse-20261007/15.pkl --seed 0 \
  --output .stacks/benchmarks/cem-reuse-repeat/15-cem_reuse-0.json
```

Use `25.pkl` for the 26-stone scene and repeat with `--seed 1`. The `prepare` mode can recreate predecessor artifacts from trusted state exports; see the [previous report](excavator-proposal-reuse.md#reproduction-and-artifacts). Existing output files are never overwritten.

- `tests/benchmark_excavator_proposal_reuse.py` (removed experimental source)
- [Portable results, seed diagnostics, profile snapshots and source/native provenance](benchmarks/excavator-cem-proposal-reuse-20261007.json)
- Local raw outputs: `.stacks/benchmarks/proposal-reuse-cem-20261007/`
