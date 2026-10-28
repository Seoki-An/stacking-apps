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

## Robustness contact neighborhood

Both platform `environment.yml` profiles now default to
`reward.place_stability.contact_hops: 2`. During velocity-disturbance trials,
keep the candidate and stones within two links of its final nominal contact
graph movable; freeze the remaining stones while retaining their geometry.
Ground/static bodies do not connect neighborhoods. The graph is undirected,
includes the final structural simulation's reported contacts, and is selected
once per placement rather than updated during a disturbance.

Set `contact_hops: 0` for candidate-only motion, any non-negative integer for
that contact-link depth, or `contact_hops: null` for full-scene motion. Older
profiles and saved contexts without the field retain full-scene behavior.
Nominal settling and structural simulation, disturbance magnitudes, trial
count, and rejection thresholds are unchanged. CPU and CUDA use the same
neighborhood selection. Saved placement diagnostics include
`place_robustness_contact_hops`.

This is a local robustness approximation: frozen boundary stones can alter
load redistribution and excursion scores, and newly formed contacts do not
activate them. In an 87-placement replay with eight perturbations, two links
matched full-scene acceptance decisions and reduced robustness time by 56%
(total placement-evaluation time by 35%). This is not an end-to-end MCTS
speedup or a guarantee of equivalent structural robustness on other scenes.

CUDA robustness uses native `CudaSimulationPhase.sampled_entities` when
available to record only monitored movable bodies. Collision geometry and
full final snapshots remain intact, and every integration timestep is retained
for peak-excursion and recovery checks. Nominal phases still request full
histories. The existing 1.5 GiB history allowance uses the compact body count,
allowing more trials per native call without increasing the memory budget.
Older native builds retain the full-history path and its original chunk limit.

Native selective sample snapshots contain only requested entities; querying an
omitted entity's history raises an error instead of inventing a stationary
trajectory. Omitting `sampled_entities` preserves full-history behavior.

Convex voxel containment uses a plane-major matrix layout so NumPy can reduce
whole rows of query points into the containment mask. This changes only CPU
evaluation layout; voxel locations, plane inequalities and tolerances remain
unchanged. A saved step-30 CUDA search with seed 0 and 512 iterations retained
the exact tree hash and selected placement, while CPU result processing fell
from 16.36 to 13.58 seconds and total search time from 141.06 to 137.73 seconds.
Step 20 likewise retained its exact tree hash, with total time falling from
79.54 to 77.74 seconds. These are single-seed saved-state comparisons, not a
whole-sequence speed guarantee.
Reducing CUDA placement chunks from 128 to 64 did not improve total time in
that case, so the existing chunk size is retained.

## Overlapping evaluation on CPU and GPU

CUDA batch evaluation with multiple result workers now prepares containment,
target coverage and surface-flatness terms on the CPU while the GPU runs the
same robustness trials. Nominal trajectory extraction also runs beside those
trials. A lightweight structural summary provides the unchanged eligibility
checks before robustness starts.

Prepared geometry contains only scalar values for each placement. It does not
publish a provisional reward or depend on keeping a whole batch in the voxel
cache. Final reward arithmetic, robustness clipping, rejection and MCTS backup
wait for completed physics and preparation. Preparation and finalization use
the same evaluator assignment in input order, with one coordinator protecting
worker-owned caches. Both native and CPU work drain on failures before an
evaluator or workspace is reused or closed.

This uses the existing CUDA backend and CPU worker setting; no solver or YAML
option was added. One result worker (including disabled parallel search), CPU
backends, and custom reward evaluators without preparation support retain the
ordinary path. Native job sizes, disturbances, simulation tolerances and search
policy are unchanged.

The benchmark script records `geometry_preparation_batches` and
`geometry_cuda_overlap_seconds` separately from final result processing. Its
`--serial-geometry` option provides a matched control with the previous scheduling;
`--serial-results` disables both forms of evaluation overlap. Stage intervals
can overlap and must not be summed as independent wall time.

In matched saved-state CUDA benchmarks (seed 0, 512 iterations, four result
workers), step 20 improved from 79.65 to 72.09 seconds and step 30 from 137.37
to 123.56 seconds. Geometry preparation overlapped native simulation for 9.07
and 13.67 seconds respectively. Both pairs retained identical complete search
tree hashes, selected placements, evaluation counts and rejection counts.
These are individual saved-state comparisons, not a whole-sequence speedup
promise.
