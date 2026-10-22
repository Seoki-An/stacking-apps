# MCTS batching

The common coordinator uses PMBS-style virtual visits to select a batch of
work from one tree. It retains this project's PUCT policy, progressive widening,
placement rewards, and rollout-free leaf bootstrap. It is an adaptation of
[Huang, Boularias and Yu's PMBS](https://arxiv.org/abs/2207.06649), particularly
Section IV-C and Algorithm 2, rather than a reproduction of their retrieval
planner and GPU rollout system.

## Selection and evaluation

After root preparation and any configured initial `root_lookahead` coverage:

1. Select from the root using PUCT, accounting for earlier selections in this
   batch with virtual visits along their ancestor paths.
2. Reserve a distinct pending action, or a proposal slot at an expandable node.
   A strong root branch can receive several slots. No per-root-branch quota is
   imposed. Already reserved actions and subtrees with no remaining batch work
   are excluded from subsequent selections.
3. Generate poses for all selected expansion nodes through `generate_batch()`.
   Each node is sent once; several reserved slots can use its resulting actions.
   Before poses exist, the reservation capacity is bounded by its current
   progressive-widening allowance and generation batch size. Rejected proposals
   can therefore leave the batch underfilled. Generators exposing only
   `generate()` still work, with sequential generation calls.
4. Evaluate selected actions together, grouped by physical fidelity. Spare
   simulation slots can evaluate pending siblings for later use.
5. Back up actual selections in selection order. Cached siblings receive no
   visits until selected. An empty expansion contributes at most one dead-end
   backup in that batch; unfilled proposal slots do not create extra visits.

Let `N` be stored visits (including reset pseudocounts), `W` accumulated return, `V` this batch's virtual
visits, and `P` the existing softmax prior. For an evaluated child, selection uses:

```text
W_child / (N_child + V_child)
  + c * P_child * sqrt(N_parent + V_parent)
      / (1 + N_child + V_child)
```

Selection requires `N_parent + V_parent > 0`. Reset sets evaluated nodes to
`N = 1, W = 0`; pending candidates remain at `N = 0, W = 0`. The initial
zero-return pseudocount makes a reset node's mean `W / (1 + n)` after `n`
actual backups. Root validation still adds its reward backup after reset.
Without reset, callers retaining a tree must preserve positive parent counts
before PUCT comparison. The mean is zero when its denominator is zero.
This uses PMBS's zero-return
virtual-count convention with our PUCT exploration term. Zero-return visits
dilute positive means; they can raise negative means toward zero. They are not
a scale-independent negative-reward penalty. Failed actions remain unselectable.

Virtual visits live in a local selection overlay. They never change the node's
stored visits or returns, never affect progressive-widening limits, and cannot
remain in the tree after failure or interruption. Native workers only perform
pose optimization and transitions; tree selection and backup remain serial.

## Controls and limits

- `generation.evaluation_batch_size` bounds selected work per round, additionally
  capped by the remaining iteration budget. Set it to `1` for serial iterations.
- `parallel.enabled` and `parallel.workers` control native execution workers.
  Disabling workers alone does not change the configured logical batch size.
- `tree.root_lookahead` controls explicit initial root coverage. Ordinary PMBS
  batching also works when this coverage is disabled.
- Root generation/validation and depth-one warmup retain their existing paths.
- Deadline checks stop scheduling new rounds. Already-started native operations
  may finish beyond the deadline; this is not hard cancellation.

There are no additional rollout actions (`d_s = 0`). The existing horizon value
estimator and discounted additive backup are unchanged. PMBS's random push
rollouts, grasp classifier, maximum-over-rollouts reduction, and GPU environment
reallocation are not introduced. `simulation_depth` controls physical evaluation
fidelity along the tree, not a separate rollout horizon.

Batch results arrive after selection, so allocation is not identical to
sequential PUCT. Compare both placement quality and elapsed time; greater worker
utilization alone does not establish better search. Existing `frontier_batches`
and `frontier_iterations` reporting fields now count the ordinary PMBS rounds
and their actual backups.

Tests in `tests/test_mcts_pmbs.py` cover global allocation, virtual-visit
diversification, unique evaluations, partial/failed generation, failure cleanup,
completed-visit widening, fidelity grouping, deadlines, sibling caching, and
serial-mode equivalence.
