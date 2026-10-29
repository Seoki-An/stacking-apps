# Unified MCTS simulation (2026-10-27)

This supersedes the staged validation and continuation-repair strategy described
in the earlier MCTS documents. The original code and dirty working files are in
`.backups/mcts-before-unified-20261027-ojcgBfmv/` (including a verified Git bundle).

## Current floor admission (2026-10-29)

The temporary unified continuation-repair experiment has been backed up to
`/tmp/mcts-before-remove-floor-mask-repair-20261029-115556` and removed.
The empty-floor overlap proposal mask is also removed for single and composite
targets: a zero overlap does not exclude an XY proposal. `fill_area` still rewards
empty-floor coverage. Target geometry and connectivity admission and root exhaustion fallback remain. Strict solved/settled
floor rejection and its configuration have been removed from both platforms.
Active-floor tracking, XY connectivity, lower-floor-first ordering and fill/height
scores remain. There is no floor-relaxation retry; empty-scene corner-seed
fallback remains independent of floor tracking. Final selection
uses existing evidence without a descendant-depth eligibility filter; it does not repair paths.
No height reward or simulation settings were changed by this removal.

### Soft previous-action distance (2026-10-30)

Both platform profiles select `planar.connectivity: previous_action_soft`.
At every depth, each orientation-specific proposal score receives the penalty

```text
d = exact gap between convex XY footprints of candidate and previous stone
excess = max(d - contact_gap_tolerance, 0)
length = cube_root(mean inventory stone volume)
score -= score.previous_action_distance * excess / (length + excess)
```

The length is fixed for the inventory, not recomputed from the remaining stones.
The weight defaults to 1.0 and is non-negative. Overlapping/nearby footprints
receive zero penalty; distant footprints approach the bounded maximum penalty.
The first placement has no previous-action penalty. This is an XY sampling
preference, not physical contact evidence or an additional reward. The viewer's
aggregate score maps and CEM proposal scores include it. The GUI's
`Previous-action distance penalty` layer shows the raw dimensionless cost;
`Weighted contribution` multiplies it by the negative recorded weight. Old
hard-connectivity runs show zero contribution. Lower-floor-first
ordering and existing elite/Boltzmann sampling remain, so admitting a distant
anchor does not guarantee it will be sampled in a finite budget.

Single and composite targets use the same penalty, including regional fallback
grids. Fresh seeds, CEM mutations, optimized poses and simulated transitions are
not rejected for previous-action distance in this mode. Target containment,
pose optimization acceptance and all physical validation remain unchanged.
There is no connected/global root quota; roots and descendants share the policy.

For recorded configurations, `previous_action` retains the old hard metric gap
checks and optional `root_global_fraction` mixture. `active_floor` retains the
old grid-adjacency admission rule and remains the default when the mode is
omitted. The root fraction has no effect in soft mode. Saved trees retain their
old failed edges and backed-up statistics; start a fresh run for this comparison.
Old `.stack` files remain readable.

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

Rank **every physically feasible root**. For a candidate with a complete path
return G, use `(1 - max_weight) * mean_return + max_weight * G`. For a candidate
without that path, use its mean return and record `descendant_depth_fallback=true`
and `final_selection_used_mean_fallback=true`, even when other roots have paths.
The mean retains the existing backed-up failure penalties. Missing evidence is
not a physical failure, and failed candidates remain excluded.

`descendant_depth` is only the path-score horizon, capped by the search horizon.
Weight zero therefore ranks all feasible roots solely by mean return. Weight one
uses a complete path where available and mean otherwise. No path repair, extra
simulation, or additional eligibility setting is introduced.

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

## Equilibrium residual contract (2026-10-29)

Native pose generation returns per-body residual wrenches through `net_wrench`,
not aggregate `c_feq` / `c_feq_robust` costs. The Python adapter restores physical
units and computes `E = max_l (||f_l||_2/(m_l*g) +
||tau_l||_2/(m_l*g*V_l^(1/3)))` once. New `PlacementSolveResult.c_feq` and
`Action.c_feq` fields retain their historical names for artifact compatibility
but contain this same dimensionless E. Admission compares E with the configured
equilibrium threshold; reward and prior use min(E, 1). The normalized diagnostic
is also retained. Old saved actions without that diagnostic keep their explicit
legacy reward/prior fallback; new native results without residuals are invalid.

## Surface flatness: restored max-minus-mean (2026-10-30)

Following the user's preference for the wall in `simple_mcts_test19.stack`
over `simple_mcts_test20.stack`, the variance experiment has been reverted.
The active reward again uses the reduction in surface max-minus-mean height:

```text
L = mean(inventory stone volume)^(1/3)
relief = maximum surface height - area-weighted mean surface height
flatness_reward = surface_flatness_weight * (relief_before - relief_after) / L
```

The surface includes empty target-floor cells and retains the true pre-simulation
parent baseline. Existing weights remain unchanged (excavator 8, tabletop 4).
Soft previous-action distance scoring, its GUI layer, and removal of the hard
descendant-depth eligibility gate remain in place. Saved runs keep their original
reward values; start a fresh search rather than reuse variance-backed tree values.
