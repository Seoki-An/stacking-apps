# Excavator combined sibling and descendant proposal bank (retired)

This archive contains reports and JSON evidence. Experimental scripts, local `.benchmarks/` inputs and native builds are not included. Command examples record the original local setup and require those separate artifacts.

**Retired on 2026-10-08.** The combined policy, its CLI options, and the dedicated descendant-collection implementation/tests have been removed from the active project. This report and its measurements are retained as historical evidence. The sibling-only experiment remains available in the separate local benchmark scripts.

Adding dependency-screened descendants to the existing sibling bank lowered pooled new-node throughput by 7.0% and physical-pass throughput by 10.7% in the two 16-stone histories. It increased exploration, but the extra branches cost more than their throughput benefit. At 26 stones, no descendants were eligible and both policies performed identical work. The combined bank has been retired for the throughput objective.

Four predecessor searches and eight paired next searches ran on 2026-10-08, following the [descendant survival diagnostic](excavator-descendant-proposal-reuse.md). This is a comparison against the existing additive sibling policy, not against fresh-only MCTS; it does not establish whether sibling reuse itself should be enabled.

## Results at 16 stones

The combined bank expands more new nodes but lowers throughput in both paired histories. Across the two seeds, it generates 2,609 nodes versus 2,313, while policy time rises from 171.77 to 208.40 seconds. Pooled new-node throughput falls from 13.465 to 12.519 nodes/s (−7.0%); physical-pass throughput falls from 1.735 to 1.550 passes/s (−10.7%). Pooled rates divide total count by total time rather than averaging individual rates.

| Predecessor/next-search seed | Bank | Policy s | New nodes | Nodes/s | New physical passes | Physical passes/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Siblings | 80.55 | 1159 | 14.388 | 175 | 2.172 |
| 0 | Combined | 105.87 | 1346 | 12.714 | 192 | 1.814 |
| 1 | Siblings | 91.22 | 1154 | 12.651 | 123 | 1.348 |
| 1 | Combined | 102.53 | 1263 | 12.318 | 131 | 1.278 |

Node-rate changes are −11.6% and −2.6%; physical-pass-rate changes are −16.5% and −5.2%. These are two paired histories, not a statistical estimate over arbitrary scenes.

### Admission and cost

| Seed | Sibling bank size | Combined input size | Sibling-only admitted / long passed | Combined siblings admitted / long passed | Combined descendants admitted / long passed |
| --- | ---: | ---: | --- | --- | --- |
| 0 | 2 | 10 | 1 / 1 | 1 / 1 | 7 / 5 |
| 1 | 4 | 11 | 4 / 3 | 3 / 2 | 4 / 3 |

For seed 0, one descendant fails pose refresh and one sibling is removed by the bank limit/duplicate check. The combined bank admits eight proposals. For seed 1, shared per-object/duplicate checks remove four of the eleven proposals, leaving seven. Two descendant alternatives for object 44 displace its sibling proposal; one of those descendants fails long validation. The policy uses refreshed priors and shared limits, not knowledge of which candidates will pass.

The slowdown is mainly downstream exploration and validation, rather than collecting or refreshing the bank:

| Seed | Metric | Siblings | Combined |
| --- | --- | ---: | ---: |
| 0 | Bank refresh/admission s, excluding validation | 0.073 | 0.226 |
| 0 | Root lookahead branches attempted | 5 | 10 |
| 0 | Short simulation jobs | 1135 | 1315 |
| 0 | Long simulation jobs | 86 | 137 |
| 0 | Descendant-validation attempts | 60 | 104 |
| 1 | Bank refresh/admission s, excluding validation | 0.132 | 0.186 |
| 1 | Root lookahead branches attempted | 9 | 11 |
| 1 | Short simulation jobs | 1122 | 1228 |
| 1 | Long simulation jobs | 87 | 113 |
| 1 | Descendant-validation attempts | 55 | 78 |

Incremental descendant collection takes about 0.011 s per predecessor and is included in the combined policy times. Physically viable additional roots create more continuations to explore and validate. The resulting work grows faster than useful transition throughput in these cases.

### Selection quality

| Seed | Selected object, siblings / combined | Immediate reward, siblings / combined | Selected validated-path return, siblings / combined |
| --- | --- | --- | --- |
| 0 | 51 / 51 | 4.176 / 4.176 | 13.6236 / 13.2368 |
| 1 | 44 / 52 | 2.161 / 1.317 | 11.0278 / 13.6571 |

Seed 0 selects exactly the same pose with a lower validated-path return. Seed 1 selects a different object with a higher validated-path return but a lower immediate reward; target IoU is 0.153880 versus 0.153821. In these 16-stone pairs, every selection has a validated descendant path, and none comes directly from the proposal bank. A higher validated-path return in one case is not evidence of better realized multi-step construction, which this experiment does not execute.

## Controls at 26 stones

The combined banks contain the same three and four sibling proposals as the existing banks. No discarded descendants pass the collection rule outside the preserved subtree. Serialized bank fingerprints match within each history, as do all recorded non-timing search statistics, admissions and selected poses/metrics.

| Seed | Bank | Policy s | New nodes | Nodes/s | New physical passes | Physical passes/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Siblings | 99.62 | 629 | 6.314 | 51 | 0.512 |
| 0 | Combined | 99.00 | 629 | 6.354 | 51 | 0.515 |
| 1 | Siblings | 109.61 | 613 | 5.592 | 64 | 0.584 |
| 1 | Combined | 107.62 | 613 | 5.696 | 64 | 0.595 |

The pooled rate differs by 1.3%, but these are repeated measurements of the same workloads, not evidence of a combined-policy gain. Seed 0 selects object 53 with validated-path return 13.9334 in both runs. Seed 1 also selects object 53, directly from the sibling bank, with return 11.3199 in both runs.

## Controlled comparison

Both policies keep the complete fresh CEM pool and admit bank proposals separately. They share one allowance of eight extra roots and the existing two-per-object bank limit. Siblings and descendants compete under the same refreshed prior ranking; descendants receive no reserved slots. Each admitted root receives normal long validation, and final-selection capacity increases by the number admitted. Descendant-path requirements remain enabled.

The descendant collection rule is unchanged from the diagnostic: require exact retained parent-scene/action evidence of a short-only pass; exclude the selected subtree, consumed objects, missing direct supports, contact connections to missing stones, and duplicates of preserved or sibling proposals. Collection occurs immediately after the predecessor search, before any new diagnostic long evaluation. The combined bank includes all screened proposals, including ones that later fail refresh or long validation. The previous diagnostic outcomes are never used to select candidates.

Bank refresh re-solves each proposal in the new root scene, refreshes its planar score, applies normal strict filters and checks duplicates against retained roots. The shared per-object cap and bank duplicate checks then apply to the jointly ranked pool. No old evaluated states, forces, rewards or backup statistics are transferred.

Four predecessor histories use saved 15- and 25-stone scenes and seeds 0 and 1. Their selected placements produce 16- and 26-stone next-search scenes. Within each pair, both variants load the identical serialized subtree and banks, invoke production promotion, and revalidate the preserved subtree. The next-search seed matches the predecessor seed. Policy order is sibling first for seed 0 and combined first for seed 1. Thus seed rows vary both predecessor history and next-search randomness; they do not isolate those two effects.

The CUDA profile uses 49 inventory objects, 1,024 iterations and maximum depth 4. Native runs are sequential on a Ryzen 9 9900X / RTX 5060 Ti with BLAS/OpenMP threads limited to one. The pinned native binary and compatible Python wrapper are the same as in the prior diagnostic.

## Metrics

Reported policy time is the complete next-search wall time plus the combined policy's incremental descendant-collection time from predecessor completion. Next-search time includes bank refresh, physical validation, exploration and final selection. It excludes input loading, environment setup, observation restoration and output serialization. Both variants already need the sibling bank; only the additional descendant-collection cost is charged separately.

New nodes counts generated child nodes, including failed and added bank nodes, excluding inherited nodes. It is not the number of parent-expansion operations. Physical passes deduplicate new transitions by exact parent scene and action, exclude inherited transitions, prioritize long evidence over short evidence and use the latest result at equal fidelity. Some passes have only short evidence; this metric is not a count of executable actions or validated descendant paths.

## Archived experiment

Thirteen focused tests passed, including shared bank caps, preservation of fresh generation, mandatory long validation, dependency screening and session promotion. All twelve native searches completed 1,024 iterations without timeouts, root retries or floor fallback. Each paired next search uses identical input hashes, profiles, preserved-edge counts, promotion state, revalidation outcomes and fresh generation hashes/pagination. All 32 fresh CEM candidates are returned in each search, bank admission leaves the fresh cursor unchanged, and final-selection capacity grows by the admitted count.

Source, profile, input and pinned native/wrapper hashes remained unchanged during measurement. The [portable results and provenance](benchmarks/excavator-combined-proposal-reuse-20261008.json) contain all preparation audits, trial statistics, admission origins/poses and paired checks. Raw local outputs are in `.benchmarks/combined-reuse-20261008/`.

The measured source files are archived locally under `.benchmarks/combined-reuse-20261008/source_archive/`, with a SHA256 manifest. The retired preparation and combined-bank commands are no longer supported by the current benchmark CLI. Recorded JSON results and provenance remain unchanged.

Production search and configuration files are unchanged by this experiment. The combined policy is retired; this report preserves its measured results.
