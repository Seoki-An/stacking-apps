# Excavator broader additive proposal reuse

The experimental scripts have been removed from the active project. Command examples below are historical records, not supported commands. This archive retains reports and JSON evidence; local `.stacks/benchmarks/` inputs and native builds are not included.

The broader policy adds no proposals under the tested CUDA configuration. Across four freshly generated predecessor trees, every one of the 32 root candidates received long validation. There were no short-only passing root siblings to add to the existing long-valid bank. This experiment therefore does not establish a performance improvement from broadening the policy.

## Policy and method

The benchmark extends the [additive sibling proposal experiment](excavator-additive-proposal-reuse.md). It keeps the full fresh CEM pool, then admits a separate allowance of at most eight refreshed sibling proposals. The broader bank includes retained root siblings that passed short evaluation and have never received long evaluation, in addition to the existing long-valid siblings. A recorded long failure takes precedence over any earlier short pass or stale validation flag. Selected/consumed objects, unevaluated proposals and deeper-tree actions are excluded. A missing validated descendant alone does not exclude a physically valid proposal.

Only action proposals are reused. They are re-solved in the new scene, rescored, filtered and deduplicated, then receive normal long validation. Old states, force solutions, rewards and backup statistics are not copied. Descendant requirements and fresh-generation budgets remain unchanged.

Four predecessor searches ran on 2026-10-08: saved scenes with 15 and 25 placed stones, each with seeds 0 and 1. Each used the current excavator CUDA profile, 1,024 iterations, depth 4, and 49 total inventory objects. All four completed their iteration budgets without timeouts. Selecting a placement produces scenes with 16 and 26 stones for the subsequent search. Tests ran sequentially on the Ryzen 9 9900X / RTX 5060 Ti with BLAS/OpenMP threads limited to one.

## Bank audit

| Stones before placement | Predecessor seed | Search s | Existing long-valid bank | Extra short-only root proposals | Broader bank | Deeper short-only passing transitions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | 0 | 64.02 | 2 | 0 | 2 | 127 |
| 15 | 1 | 68.76 | 4 | 0 | 4 | 142 |
| 25 | 0 | 79.51 | 3 | 0 | 3 | 14 |
| 25 | 1 | 81.98 | 4 | 0 | 4 | 16 |

Each search recorded 32 unique root transitions, all evaluated at long fidelity, with zero short evaluations at the root. After excluding the selected placement and other candidates consuming its object, each tree retained 30 available siblings. Across the four trees, these comprise 13 long successes and 107 long failures. No retired root proposals supplied short-only passes either. Serialized long-bank and broader-bank SHA256 fingerprints match in all four histories.

Deeper counts refer to distinct recorded transitions whose strongest/latest evaluation remains a short pass. They are not counts of reusable root proposals: some belong to preserved branches, and their parent scenes differ from the new root. They cannot be added directly without checking missing supports and other dependencies.

## Why the broader bank is unchanged

The CUDA profile sets `validation.root_before_search: true`. Search preparation generates and long-validates the root pool before descendant exploration. Here, `root_cem.keep`, `tree.root_max_children` and `validation.max_candidates` are all 32. Consequently, the completed searches leave no root candidates in a short-only passing state. The earlier additive experiment already used the available long-valid siblings.

This finding is specific to the current root policy and these completed searches. It does not prove that every configuration, interrupted search or retry history produces an identical bank. Disabling root validation merely to populate a short-only bank would change the search algorithm and would require a separate comparison.

## Paired equivalence controls

Four subsequent CUDA searches compared long-only and broader banks using the same promoted subtree within each pair. Both use predecessor seed 0 and next-search seed 0. The 16-stone pair ran long-only first; the 26-stone pair ran broader first. These are equivalence controls, not comparisons against fresh-only search. All completed 1,024 iterations without timeouts, root retries or floor fallback.

| Stones after placement | Bank policy | Search s | New nodes | Nodes/s | New physical passes | Physical passes/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | Long-only | 80.34 | 1159 | 14.426 | 175 | 2.178 |
| 16 | Broader | 79.30 | 1159 | 14.615 | 175 | 2.207 |
| 26 | Long-only | 97.29 | 629 | 6.465 | 51 | 0.524 |
| 26 | Broader | 98.97 | 629 | 6.356 | 51 | 0.515 |

Within each pair, input hashes, profile snapshots, fresh-action hashes and pagination, inherited-edge counts, revalidation outcomes, bank admission, all recorded non-timing search statistics and selection metrics match exactly. At 16 stones, one of two bank proposals is admitted and passes long validation. At 26 stones, all three are admitted and two pass long validation. Both pairs retain the full 32 fresh CEM candidates and leave the fresh cursor unchanged during bank admission.

The selected objects are 51 and 53, with validated descendant-path returns of 13.6236 and 13.9334, respectively, under both policies. The timing differences are repeated measurements of identical workloads; they provide no evidence of a broader-policy speedup or slowdown.

New nodes counts generated child nodes, including failures and added bank nodes, excluding inherited nodes; it is not the number of parent-expansion operations. Physical passes deduplicate new transitions by exact parent scene and action, exclude inherited transitions, prioritize long results over short results, and use the latest result at equal fidelity. A short pass does not imply executable or descendant-validated status. Search time includes bank refresh, validation, exploration and final selection; it excludes input loading, setup, observation restoration and output serialization.

## Follow-up direction

Keep the existing additive policy experimental; broadening to short-only root siblings offers no additional exploration in these cases. The next distinct reuse question is whether deeper proposals survive removal of their original parent placements. First measure their support dependencies and refresh success in the new scene, then compare any eligible additions against fully funded fresh generation. Short success in the old parent scene is insufficient evidence of eligibility.

The [discarded-descendant diagnostic](excavator-descendant-proposal-reuse.md) tests that question: it recovers five and six long-passing refreshed proposals in the two 16-stone cases, and no additional eligible proposals outside the preserved subtree in the two 26-stone cases. The subsequent [full-search comparison](excavator-combined-proposal-reuse.md) found that adding those screened descendants lowered pooled nodes/s by 7.0% at 16 stones; the 26-stone banks remained identical. The combined policy has since been retired; its report is retained for reference.

## Validation and reproduction

Nine focused admission/session tests passed. The broader-bank regression explicitly covers a short-only root success, a later long failure with a stale validation flag, a physically valid candidate rejected for missing descendants, a consumed object, a deeper action and an unevaluated candidate. The additive orchestration test verifies preservation of fresh candidates and authoritative validation of added roots.

Final checks verified unchanged benchmark/search source hashes, all excavator profile hashes, source-state hashes and the isolated native binary/wrapper hashes. Full audit records, paired equality checks and trial statistics are in the [portable results](benchmarks/excavator-broader-additive-proposal-reuse-20261008.json).

During the batch, the editable `stacking-core` Python wrapper gained CUDA grasp exports absent from the installed binary. One audit attempt stopped at import, before running a search. The remaining runs used an isolated copy of the unchanged native binary and its matching committed wrapper; the checkout and installation were not modified. Native SHA256: `54632cb30be47dba9ee7cf0401edc1eac00bcfa47a6e35a24fa8207297672f0a`; wrapper revision: `51416ec02c6b3ed576159b243f1e49eb457be09f`. The local snapshot and provenance are under `.stacks/benchmarks/proposal-reuse-broader-20261008/`.

With a compatible installed native module, reproduce using trusted local saved-state pickles:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/stacking-mpl
python tests/benchmark_excavator_proposal_reuse.py prepare \
  --input .stacks/benchmarks/excavator-profiles-20261007/states/15.pkl \
  --output .stacks/benchmarks/broader-repeat/prepare-15-0.pkl --seed 0
python tests/benchmark_excavator_proposal_reuse.py additive_reuse \
  --input .stacks/benchmarks/broader-repeat/prepare-15-0.pkl \
  --output .stacks/benchmarks/broader-repeat/16-long-0.json \
  --seed 0 --bank-policy long
python tests/benchmark_excavator_proposal_reuse.py additive_reuse \
  --input .stacks/benchmarks/broader-repeat/prepare-15-0.pkl \
  --output .stacks/benchmarks/broader-repeat/16-broader-0.json \
  --seed 0 --bank-policy broader
```

The script defaults to `apps/excavator/config/mcts_cuda.yml`. Repeat preparation with seed 1, and use `.stacks/benchmarks/excavator-profiles-25-20261007/states/25.pkl` for the 25-stone source. For the exact local native snapshot, also set `PYTHONPATH=.stacks/benchmarks/proposal-reuse-broader-20261008/native_snapshot`.

Production search and configuration files are unchanged by this experiment. The benchmark-only reuse implementation has been removed.
