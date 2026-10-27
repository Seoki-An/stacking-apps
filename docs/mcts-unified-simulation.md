# Unified MCTS simulation (2026-10-27)

This supersedes the staged validation and continuation-repair strategy described
in the earlier MCTS documents. The original code and dirty working files are in
`.backups/mcts-before-unified-20261027-ojcgBfmv/` (including a verified Git bundle).

## Evaluation and search

Every newly evaluated edge uses the same platform simulation profile:

1. Settle the new stone while existing stones remain fixed.
2. Start whole-structure simulation from the settled poses, with velocities reset.
3. Apply the configured support, motion and disturbance checks. A failed nominal
   simulation cannot become feasible through descendant selection.

The backend still owns these stages. Root admission, batched descendant expansion
and normal root-proposal retries all use this same evaluation. There is no later
long simulation, path repair, or reserved final-validation time. CEM, pose
optimization, PUCT, progressive widening, PMBS batching and discounted backup are
retained. Duplicate proposals matching retained actions are not evaluated again.

```text
Generate root candidates; evaluate pending roots once; back up root results
While iteration/time budget remains:
    Select frontier using PUCT and existing batch reservations
    Generate actions; evaluate pending transitions with the same simulation
    Back up rewards and bootstrap values
If no feasible root exists:
    Use the existing bounded root-proposal / sampling-fallback retries
Select from the evaluated tree, without generating or simulating anything
```

## Final selection

`final_selection.descendant_depth` replaces `validate_descendant_path`,
`validate_descendant_depth`, and `require_validated_descendant`. It counts evaluated,
feasible edges **below each root candidate**. It is capped by the remaining search
horizon, and a successful terminal state satisfies the rest of the horizon.
Pending or rejected nodes do not count. Thus a depth-1 warmup requires no further
descendants even if the configured descendant depth is two.

For each candidate n, recursively compute the best existing return to that depth:

```text
PathReturn(n, d):
    If n is pending or rejected: return unavailable
    If d = 0 or n is successful terminal: return r(n)
    tails = available PathReturn(child, d - 1) over feasible children
    If tails is empty: return unavailable
    Return r(n) + gamma * max(tails)
```

Filter roots to those with an available path return G. There is one selection
formula: `(1 - max_weight) * mean_return + max_weight * G`. The only settings
are `max_weight` and `descendant_depth`; there is no mode selector.
Weight zero uses mean return, weight one uses the best path return, and values
between them blend both. Eligibility is identical for every weight.
No simulator is called during ranking. If all roots lack the requested depth,
preserve the previous exhaustion behavior: rank physically feasible roots by mean
return and record `descendant_depth_fallback=true`. Never mark a depth-filtered
candidate as physically failed, and never revive a physical failure.

## Configuration and reuse

Both platforms now have one `phases.structural` configuration after
`phases.settling`. Structural settings are `min_time=3.0`, `max_time=10.0`,
`extra_steps=85`, and `final_extra_steps=200` at dt=0.01. Energy/motion thresholds, settling settings,
contact models and solver tolerances are unchanged and remain platform-owned.
Full simulation has no distant-stone freezing. These durations are initial
settings, not a measured optimum or a demonstrated stability improvement.

MCTS YAML no longer has a `validation` block or `tree.simulation_depth`.
`rejected_root_retry_rounds` now belongs to `generation`. All search time is
available to admission and exploration; an in-flight native batch may finish
after the scheduling deadline.

Promotion preserves an unchanged scene's evaluated subtree without revalidation.
Changed observed poses, inventories/status/contact evidence or changed search
configuration/evaluator invalidate the retained children and statistics. Planning
also invalidates descendants when its committed placement differs from the
simulated pose. The existing `reset_statistics` request still controls whether
visit statistics are reset for the next search. Replacing a physical profile
requires rebuilding the evaluator/application, not mutating native settings under
an existing session.

Existing `.stack` files remain readable: the artifact schema is unchanged. Old
staged simulation decoding and legacy result columns remain for compatibility;
the internal fidelity value `long` denotes the single full evaluation for new
runs. New console reports aggregate simulation timing and distinguish settling
from structure. Old planning checkpoints whose resolved profiles differ are
rejected by the existing resume-context comparison; they are not silently resumed
under the new physics or policy. Runtime YAML with removed validation keys fails
with a migration message.

## Verification

Before editing, the original search/repair/session tests passed (75 tests).
After migration, the main MCTS, planning, profile, CPU backend, resume and native
composite-target suites passed (290 tests). Additional configuration, GUI planning
entry-point and artifact checks passed (81 tests; this set overlaps the main set).
Two real GPU/CPU comparisons passed, covering both the unified phase sequence and
legacy staged backend compatibility. The CPU backend test explicitly checks that
settling precedes structure and that settled poses, with reset velocities, start
the structural phase. No full-inventory speed/stability benchmark or robot
execution was performed.
