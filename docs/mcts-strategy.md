# Current MCTS strategy

This document describes the shared MCTS implementation and the excavator CPU
and CUDA profiles as of 2026-10-05, including continuation repair. MCTS searches
placement sequences, validates the immediate action and required continuations,
then retains the selected subtree for the next placement. CUDA accelerates
batched pose generation and simulation; tree selection and backup remain on
the CPU.

The configuration values below are a snapshot. The active YAML profile and any
runtime overrides determine a run's behavior; saved `.stack` contexts record
the resolved settings. See [MCTS batching](mcts-batching.md) for the PMBS-style
scheduler and [excavator planning](excavator-planning.md) for native solvers,
execution workflows, and historical benchmarks.

## One placement search

1. Establish the root from the current state or the previously selected subtree.
   Revalidate retained children when requested by the session.
2. Generate root candidates with CEM and long-validate them before exploration.
   Seed accepted roots with backup visits.
3. Give eligible root candidates initial one-level lookahead coverage.
4. Select, expand, simulate, and back up batches until the iteration budget or
   exploration deadline is reached.
5. Rank root candidates provisionally and validate their required descendant
   paths. Repair missing paths within bounded additional work.
6. If no candidate qualifies, retry root generation, search continuations, and
   repeat final selection within the remaining deadline. If retries also fail,
   relax floor-first sampling and try a few fresh root pools.
7. Select the highest final score. In search-only mode, promote that node to the
   next root. The motion-planning workflow additionally checks executability.

Only the selected immediate placement is committed. Descendants are predictions
of future placements, not a command to execute the whole sequence.

## Current excavator profiles

| Setting | CPU | CUDA |
| --- | ---: | ---: |
| Maximum tree depth | 3 | 4 |
| Iterations per search | 512 | 1024 |
| Overall search limit | 240 s | 240 s |
| Time reserved for final validation and repair | 30 s | 60 s |
| Root child limit | 12 | 32 |
| Non-root child limit | 16 | 16 |
| Generation batch size per node | 32 | 32 |
| Logical evaluation batch size | 12 | 256 |
| CPU worker setting | 12 | 4 result workers |
| Pose generation and simulation backends | CPU | CUDA |
| CEM population / elite / rounds / retained | 96 / 24 / 3 / 12 | 96 / 32 / 3 / 32 |
| CEM XY mutation standard deviation | 0.1 m | 0.2 m |
| Final ranking | mixed, weight 0.5 | mixed, weight 0.5 |
| Required descendant levels beyond a root candidate | 1 | 2 |

Both profiles initially use depth one until eight objects have been placed.
Consequently, descendant validation is capped to zero levels during these early
steps. They also share discount `0.9`, exploration constant `2.0`, three root
retry rounds, and progressive widening with coefficient `3.0` and exponent `0.5`.

Sources: [CPU profile](../apps/excavator/config/mcts_cpu.yml) and
[CUDA profile](../apps/excavator/config/mcts_cuda.yml).

## Candidate generation and exploration

CEM generates and ranks root proposals, then combines mutations of elites with
fresh proposals. Both profiles use a fresh fraction of `0.5` and retain at most
two candidates per object. Root and deeper sampling use `lower_floor_first`.
These proposal scores are priors; physical simulation still decides whether a
placement is acceptable. CEM settings govern the root pool, not all descendants.

For a non-root node with completed visits `N`, the selectable-child allowance is:

```text
K(N) = min(max_children, ceil(coefficient * max(N, 1)^exponent))
```

The root uses its own fixed child limit. Generation also respects a per-node
attempt limit, counting accepted and rejected proposals:

```text
A = max(hard_child_limit, ceil(hard_child_limit * attempt_multiplier))
```

With the current non-root settings, this is 48 attempts. Rejected proposals
consume that budget; increasing the evaluation batch size does not remove it.
`min_feasible_children` is a generation target, not a guarantee that children
will pass subsequent physical validation.

Ordinary descendant search uses global PUCT selection with temporary virtual
visits. For an evaluated child:

```text
score = W_child / (N_child + V_child)
      + c * P_child * sqrt(max(N_parent + V_parent, 1))
          / (1 + N_child + V_child)
```

`W` is accumulated return, `N` completed visits, `V` reservations in the current
batch, and `P` the softmax prior. A zero denominator gives a zero mean. Pending
actions are selected before evaluated children. Reservations diversify a batch
without changing persistent statistics; they are zero-return virtual visits,
not a fixed negative virtual-loss penalty. Actual results are backed up after
evaluation. Spare capacity may evaluate siblings without immediately giving
them backup visits.

## Simulation and value backup

Search normally uses short simulation; root preparation, descendant validation,
and repair use long simulation. Repair first simulates newly generated children
at short fidelity and long-validates only those that pass. Both fidelities use
the selected simulation
backend, including CUDA when configured. Long validation is not synonymous
with CPU fallback. It includes the configured physical and placement-robustness
checks.

`simulation_depth` selects physical fidelity by tree depth. It is not an extra
rollout horizon. The current value of 5 exceeds both profiles' maximum tree
depth, so ordinary tree evaluations stay on the short path, with explicit long
validation handled separately.

There are no additional rollout actions beyond the selected tree leaf
(`d_s = 0`). Nonterminal leaves use a fixed expected-reward bootstrap:

```text
h = min(bootstrap.horizon, max_depth + 1 - leaf_depth, remaining_objects)
B = bootstrap.mean_reward * sum(discount^j for j = 0, ..., h - 1)
```

Both profiles set the mean reward to 2.5 and horizon to 4. The implementation
preserves its legacy discount convention: initialize backup with `discount * B`,
then at each node apply `G = reward + discount * G`. Thus the leaf receives
`reward_leaf + discount^2 * B`. Terminal and failed leaves have no bootstrap.
A failed node contributes `failure_reward` toward ancestor returns but records
a zero-valued visit at the failed node itself.

Visit counts, unique nodes, and transition evaluations are different quantities.
Repeated selections can consume iterations without generating a new node;
long revalidation can evaluate an existing node again.

## Final ranking and validated paths

Let `M(a)` denote the root candidate's mean score and `D_k(a)` its discounted
path return including up to `k` descendants beyond that candidate:

```text
D_k(a) = r(a) + discount * r(a_1) + ... + discount^k * r(a_k)
```

Terminal states may finish a path early. The depth is capped by the remaining
tree depth. During initial validation, paths are tried in the order implied by
known returns, and the first passing continuation is used; this is not an
exhaustive search for the largest possible long-simulation return. Repair
scores the available authoritative continuations by their maximum return.

| Mode | Final score |
| --- | --- |
| `mean` | `M(a)` |
| `max_descendant` | `D_k(a)` |
| `mixed` | `(1 - max_weight) * M(a) + max_weight * D_k(a)` |

For roots seeded by authoritative root preparation, `M(a)` uses backed-up mean
values. Otherwise, the implementation uses a recorded final-validation reward
when available, falling back to the backed-up mean. The current mixed mode
therefore does use MCTS backup information. That information can include short
simulation and bootstrap estimates; it is not entirely recomputed from long
validation results.

`validate_descendant_path: true` re-simulates necessary descendants at final
validation fidelity, reusing results already marked authoritative. Children that
already failed at search fidelity are not re-simulated. If validation
changes a node's state beyond `max_state_drift`, its descendants are cleared.
The current drift metric is maximum translational displacement among objects
common to both states.

With `require_validated_descendant: true`, a missing path is excluded after
repair. With it false, an unavailable path falls back to the mean score. In
`mean` mode, descendant-path scoring and repair are bypassed. At depth zero,
the path score is just the candidate's immediate reward; enabling the path
validation flag does not create additional lookahead.

## Continuation repair and root retries

These are separate mechanisms. **Continuation repair runs before excluding a
physically valid root for a missing path, even when another root already
qualifies.** It operates only when validated paths are required and affect the
final score.

Repair preserves validated prefixes and first seeks work beneath the deepest
available valid states. It interleaves work across affected roots, generates
replacement children, screens them at short search fidelity, long-validates the
children that pass in batches, and backs up the new results. It retains failed children and proposal offsets rather than repeatedly
starting generation from the same position.

Each final-selection pass permits at most:

```text
repair_rounds = generation.expansion_retry_batches * required_descendant_depth
evaluations_per_round <= generation.evaluation_batch_size
```

That allows up to three rounds of 12 evaluations for the CPU profile and six
rounds of 256 for CUDA after the depth-one warmup. These are ceilings, not
targets. Existing widening, generation-attempt limits, generator exhaustion,
and the remaining overall deadline can stop repair earlier. Already validated
root paths remain available if repair cannot recover other alternatives.

**Root retries run only if final selection returns no candidate.** They retire
rejected roots, request a diversified CEM pool excluding earlier roots, and
long-validate fresh placements. Initial lookahead plus retry depth extension
builds continuations to the required depth before final selection and repair
run again. Increasing root retry rounds alone does not improve alternatives
when some root already qualifies.

**The floor-fill fallback runs only if root retries also select nothing.** It
relaxes `lower_floor_first` for both root and deeper sampling, then repeats root
preparation, lookahead, retry depth extension, and final selection on fresh root
pools. After the first pool, up to `rejected_root_retry_rounds` further pools
each exclude every earlier candidate. The fallback stops at the first
selection, when CEM has no fresh proposals, or at the original overall
deadline. Saved statistics record its pool, generated, and accepted counts.

`missing_validated_descendant` means no qualifying continuation was found within
these limits. It does not prove the immediate placement is physically invalid
or that no feasible continuation exists.

## Tree reuse across placements

Promotion retains the selected candidate's subtree and discards access to its
alternative root branches. The next search can reuse generated actions,
evaluated states, and retained validation information. The session requests
child revalidation on a retained subtree; sufficiently changed states invalidate
their descendants.

Promotion resets only the new root's generation attempts, exhausted flag,
generation rejection counts and external-seed attempt offset. A fresh root CEM
pool starts at offset zero; a former descendant's generation cursor cannot skip
it. Descendant generation bookkeeping and the preserved tree remain intact.

Current runtime requests use `reset_statistics: true`: visits and accumulated
returns are reset recursively before the next search, while the tree structure
and evaluation results remain. This provides candidate and simulation reuse,
but not uninterrupted statistical accumulation across placement steps.

Both platform planning logs report `tree_nodes` and `leaves` after each search
attempt, before motion planning and root promotion. Leaves are nodes with no
children in the retained tree, including pending, failed and inherited nodes;
a root with no children counts as one leaf. These are final tree counts, not
new expansions or a cumulative count of candidates retired during retries.
Search-only `.stack` statistics store `tree.leaves`; motion-planning sequences
store `search_attempts[].tree_nodes` and `search_attempts[].tree_leaves`.
The existing `frontier_leaves` metric instead counts evaluated nodes with no
evaluated children and can differ when children are still pending.

## Batch sizes and time limits

| Control | Meaning |
| --- | --- |
| `generation.batch_size` | Per-node generation request cap, additionally constrained by widening and attempts |
| `generation.evaluation_batch_size` | Logical work admitted to an evaluation round; ordinary PMBS selection also respects remaining iterations |
| `parallel.workers` | CPU native workers in CPU mode; CPU result workers in the current CUDA composition |
| Native CUDA chunk sizes | Internal capacities, separate from logical MCTS batching |

Current native defaults are 256 pose-generation jobs, 128 placements per GPU
simulation chunk, and 512 native simulation jobs to accommodate robustness
trials. Reward and observation processing overlaps subsequent CUDA chunks.
Tree selection and backup remain serial CPU work. A configured batch of 256
does not guarantee 256 available independent jobs.

The exploration deadline is the overall deadline minus `validation.time_reserve`:
approximately 210 seconds for CPU and 180 for CUDA. Final validation, repair,
and root retries share the remaining overall budget. Native calls already in
flight can overrun a deadline; checks are cooperative between work submissions.

## Diagnostics and implementation references

Assess throughput together with placement quality. Report new generated nodes
per second separately from transition evaluations per second, since the latter
includes revalidation. Compare physical rejection reasons, validated path
availability, final rewards, timeouts, and successful placement counts. Iteration
counts alone do not measure exploration coverage.

Repair contributes to existing generation, evaluation, descendant-validation,
and timing counters; it currently has no separate saved-run repair counter.
The new repair behavior has regression coverage, but its improvement on saved
CUDA scenes has not yet been benchmarked. Historical timing reports should not
be treated as measurements of the new policy.

- [Search coordinator and repair](../packages/stacking-common/src/stacking_common/search/mcts/search.py)
- [Tree policy](../packages/stacking-common/src/stacking_common/search/mcts/tree_policy.py)
- [Backup](../packages/stacking-common/src/stacking_common/search/mcts/backup.py)
- [Session and subtree promotion](../packages/stacking-common/src/stacking_common/search/mcts/session.py)
- [Platform composition](../packages/stacking-common/src/stacking_common/workflows/search.py)
- [Repair regression tests](../tests/test_mcts_repair.py)
- [PMBS regression tests](../tests/test_mcts_pmbs.py)
