# Archived experiment: unified simulation with floor continuation repair

**Current status (2026-10-29):** continuation repair and the empty-floor overlap
admission mask have been removed at the user’s request. The implementation below
is historical. Its code/config snapshot is at
`/tmp/mcts-before-remove-floor-mask-repair-20261029-115556`. Strict floor rejection has also been removed.
XY connectivity, active-floor tracking, fill-area scoring, descendant-depth
selection and unified simulation remain. Height reward still uses relief improvement.

This temporary excavator experiment restores floor admission and bounded
continuation search without restoring SHORT/LONG simulation, authoritative
revalidation, or simulation-driven subtree replacement.

The excavator action profile enables `filtering.strict_floor_first`. Normal
generation and settled-pose admission enforce the active floor. Before final
ranking, `final_selection.repair_floor_continuations` gives feasible root
candidates missing the requested descendant depth another chance:

1. Keep the root candidate and its evaluated state unchanged.
2. Temporarily relax floor admission and the unfilled-cell sampling mask below
   that candidate. Existing physical failures remain rejected.
3. Reopen generation once per affected parent and evaluate new continuations
   with the same settling/full structural/robustness pipeline as normal search.
4. Rank using the existing mixed score and descendant-depth gate. If no root
   qualifies, the existing feasible-root mean fallback remains available.

The existing root floor fallback runs only when no selectable root remains;
missing descendant depth alone does not cause a physically feasible floor root
to be replaced by a relaxed root pool. Relaxed children are discarded on the
next search after promotion, restoring normal floor admission for the next
actual placement. Retained evaluated states are not re-simulated by repair.

Repair is capped at `expansion_retry_batches * min(descendant_depth,
max_depth - 1)` rounds. Each round uses at most `evaluation_batch_size` new
transition evaluations, with normal widening/attempt limits and the original
search deadline. A native pose solve finishing after the deadline cannot start
another simulation. Roots share batches in round-robin order.

Repair is additional bounded work outside the ordinary iteration count. Its
attempted/recovered roots, rounds, generated/evaluated candidates, elapsed time,
and timeout are reported under `floor_continuation_repair` in saved statistics
and printed in step summaries. Total generated/evaluated counts and posegen /
simulation timing include repair; its elapsed time must not be added again.

## Configuration and rollback

Excavator CPU and CUDA profiles enable repair. Tabletop behavior is unchanged;
the new option defaults to false when absent. No simulation, reward, target,
iteration, initial-depth, or solver settings are changed by this experiment.

| Test condition | `filtering.strict_floor_first` in action_generation.yml | `final_selection.repair_floor_continuations` in mcts_cpu.yml / mcts_cuda.yml |
| --- | --- | --- |
| Previous permissive behavior | false | false |
| Strict floor only | true | false |
| Strict floor plus repair | true | true |

The pre-change files and working-tree patch are saved under
`/tmp/mcts-unified-floor-20261029/before`. Do not restore that entire working-tree
patch: it also records unrelated user changes. The table above disables this
experiment without overwriting those changes.

## Validation

Focused tests cover unified-fidelity evaluation, strict root preservation,
recovery before final depth ranking, physical failure preservation, execution
rejections, generation/deadline limits, exception-safe filter restoration,
configuration parsing, and saved diagnostics. The main targeted suite passed
163 tests; six permissive/strict floor composition tests also passed.
The old profile test expecting `structural.min_time == 3.0` is incompatible with
the user's existing `1.0` setting; that setting was preserved, and this test was
excluded from the targeted run. Existing recorded floor-geometry fixture
mismatches were not changed.

Native comparisons use 49 stones, an empty starting scene, seed 0, and a matched
256-iteration test budget for up to 16 placements. Production YAML retains its
1,024-iteration setting. Both arms use identical unified physics, rewards,
posegen, target and compute settings; only floor admission and repair differ.
This is a limited placement-search experiment, not a full 49-stone result or
robot-motion test. Artifacts and resolved profiles are recorded in
`/tmp/mcts-unified-floor-20261029`.

After the first 16 placements, each arm is continued for eight more placements
from its own step-16 checkpoint with a fresh search tree and seed 0. This is a
symmetric restart, not uninterrupted tree preservation across the boundary.

## Matched floor-policy results

Both arms completed 24 placements. The first 16 used continuous subtree
preservation; placements 17–24 restarted symmetrically from each arm's own
step-16 state. Timings exclude diagnostic geometry and serialization.

| Metric | Permissive baseline | Strict floor + repair |
| --- | ---: | ---: |
| Search time, placements 1–16 | 309.73 s | 396.72 s |
| Search time, placements 17–24 | 376.90 s | 342.29 s |
| Final maximum height | 1.983 m | 1.478 m |
| Final projected active-floor coverage | 55.86% | 73.44% |
| Final occupied slice area at z = 0.35 m | 48.13% | 57.91% |
| Final occupied slice area at z = 0.85 m | 39.52% | 42.95% |
| Final target IoU | 0.2385 | 0.2222 |

The combined policy produces a lower, broader partial structure, but does not
prove restoration of two rows. The active floor still has not reached its
75% coverage threshold at placement 24. Selected objects and trajectories
differ, so lower height alone is not evidence of better packing. The first
16 placements cost 28.1% more; most of that difference is pose generation
(187.82 → 275.54 s), while batch evaluation is similar (114.50 → 115.08 s).
Repair recovered 64/64 attempted roots during those 16 placements, with 752
additional evaluations and 36.00 s of included repair work. Five selected
roots used repaired continuations.

A separate height-only comparison disables both strict floor and repair and
changes only `planar.score.height` from -3.0 to -6.0 in memory. The persistent
YAML value remains -3.0. This is a proposal-score experiment, not a change to
`surface_flatness` reward or simulation acceptance. All three arms use the
same empty scene, native binary, seed and 256-iteration budget for placements
1–16.

### Height-only result (16 placements)

All three runs completed 16/16 placements. Source, native binary, initial-state,
seed and unchanged settings were checked against the recorded hashes/profiles.
The height arm's resolved profiles match baseline; its sampling settings differ
only in the height weight. All saved `.stack` debug scenes and final planning
checkpoints were decoded successfully.

| Metric | Baseline (-3) | Height only (-6) | Floor + repair (-3) |
| --- | ---: | ---: | ---: |
| Search total | 309.73 s | 321.66 s | 396.72 s |
| Pose generation | 187.82 s | 204.32 s | 275.54 s |
| Batch evaluation | 114.50 s | 110.33 s | 115.08 s |
| Projected active-floor coverage | 55.86% | 64.84% | 71.88% |
| Slice coverage, z = 0.35 m | 48.12% | 50.74% | 55.71% |
| Maximum height | 1.417 m | 1.307 m | 1.194 m |
| Target IoU | 0.1678 | 0.1686 | 0.1470 |
| Mean retained leaves (including failed) | 261.4 | 258.9 | 251.5 |
| Stones with minimum mesh height ≤ 0.2 m | 10 | 10 | 13 |

Height -6 costs 3.85% more total search time in this run, with similar tree
size and target IoU. It improves projected floor coverage by 8.98 percentage
points, but actual slice coverage improves only 2.62 points. These metrics
measure different things: projected floor coverage is the coarse admission
tracker, while slice coverage measures occupied DSF cross-sectional area.
The same number of ground-level stones and continued central clustering of
upper footprints show that this is a modest packing change, not a demonstrated
solution to the two-row issue. Coverage remains at 64.84% from step 10 onward
while upper placement begins at step 11.

The height term is normalized over available contact heights before weighting.
It changes proposal preferences, not the feasible-set constraint or final
reward. Native optimization can also move the proposed pose. Thus a larger
negative weight cannot substitute for checking the final admitted floor.

For a minimal follow-up, height -6 is a reasonable low-cost candidate, but it
should not be described as a fix. Floor admission plus repair has a stronger
observed effect on lower-layer filling and a higher initial cost. A combined
height -6 plus floor/repair condition has not been tested. No claim about full
49-stone completion, statistical significance, or restored upper two-row
packing follows from this single-seed 256-iteration experiment.

Artifacts: `/tmp/mcts-unified-floor-20261029/height-comparison.txt`,
`height-comparison-steps.csv`, `compare16/sequence-footprints.png`,
`compare16/sequence-slices.png`, and `baseline.stack`, `height.stack`,
`repaired.stack`. The 24-placement floor-policy continuations and plots are
in `*-extension.stack` and `compare24/`. The height weight is still -3 in
repository YAML; only the experiment hook used -6.

## Empty-floor admission ablation and visual reference

The next single-target experiment kept height -3, strict floor OFF and repair
OFF, and forced only `score_xy_debug_map(relax_floor_fill=True)` in a temporary
hook. No production code/config changes were made for this ablation. At the
user's step-20 scene, this exactly recovers 22 masked cells for selected stone
21 (122 → 144 eligible), without admitting target-inset cells.

The 16-placement sequence completed in 367.56 s (+18.67% versus control).
Projected floor coverage improves to 62.11%, but low slice occupancy is nearly
unchanged (48.78% versus 48.12%) and target IoU falls to 0.14792. Mean leaves
remain similar (258.2 versus 261.4). In a paired search from the same saved
step-19 state, both control and mask-relaxed search still choose an upper
placement, taking 38.24 and 42.50 s respectively. These one-seed reduced-budget
results do not justify adopting unconditional mask removal as a two-row fix.

The user's shape reference is `.stacks/excavator_plan5.stack`. Same-camera mesh
renders compare its step 30 to `simple_mcts_test4.stack` step 30, and its step
16 to all four experiments. The reference retains a broader upper arrangement
across wall thickness, while the current result concentrates upper stones near
the center and develops uneven high sections. Removing lower-layer stones from
the view makes this distinction clearer. Height-only and mask-only experiments
do not reproduce the desired reference arrangement at step 16. Strict floor +
repair is visually closer to its low, spread-out foundation, but later upper
row restoration remains unproven.

The historical target geometries and object asset hashes match, as do recorded
proposal-score weights and floor geometry settings. Relevant differences
include strict floor (reference ON/current OFF), volume sampling exponent
(1.0/0.1), posegen gap tolerance (0.005/0.0001), widening (3/2), depth (4/5),
and the former short/long validated-path selection. Native binaries also
differ. Thus historical shape differences cannot be attributed to one filter
from this visual comparison alone.

Visual artifacts:
- `/tmp/mcts-wall-visual-20261029/upper-comparison30.png`: matched 30 stones,
  including an upper-stones-only view (minimum mesh height > 0.2 m).
- `/tmp/mcts-wall-visual-20261029/reference-current.png`: full reference/current
  views, plus reference final 40-stone result, explicitly labelled.
- `/tmp/mcts-wall-visual-20261029/experiments16.png`: matched 16-stone views.
- `/tmp/mcts-wall-visual-20261029/profile-differences.json`: recorded differences.

New runs and provenance are in `/tmp/mcts-unified-floor-20261029/empty-floor.stack`,
`step20-baseline.stack`, and `step20-empty-floor.stack`. Their debug scenes and
checkpoints decode successfully. Runtime hooks and full JSON results are
embedded as attachments. The GUI's on-demand XY map reconstruction does not
apply the experimental hook; its recomputed masks are not the experimental
mask, although saved scenes and candidates are the actual experimental output.

### Correction: matching weights do not imply matching height rewards

The user's reminder about plan5's height reward is important. Although both
contexts record `surface_flatness: 2.0`, the formula changed in commit
`5db4e9073c8a67e19fc873cb124f36f2dc59fec9` (2026-10-28). The implementation
at plan5's creation time (2026-10-22; relevant commit `c770825`) used an absolute
surface-relief penalty. The current code rewards relief reduction. Proposal
`planar.score.height` is separate and remains -3 in both historical contexts.

Let L(s) = (maximum top-surface height - mean top-surface height) / cbrt(mean
inventory stone volume), measured on the target footprint with empty cells at
zero. The old contribution is -w L(s_next), while the new contribution is
w [L(s_parent) - L(s_next)], with w = 2.

For siblings of the same parent, these differ by a constant, so the height
term alone preserves immediate sibling ranking. However, summed multi-step
returns differ. With D transitions and discount gamma, the new contribution is
w L(s0) - w (1-gamma) sum_{j=1}^{D-1} gamma^(j-1) L(sj)
- w gamma^(D-1) L(sD). The old one is
-w sum_{j=1}^{D} gamma^(j-1) L(sj). Both runs use gamma = 0.9. Thus intermediate
roughness is penalized differently; the change is not equivalent to retaining
the old objective or simply changing a YAML coefficient.

`/tmp/mcts-wall-visual-20261029/height-reward-counterfactual.csv` evaluates both
formulas on saved adjacent checkpoints at steps 16, 20 and 30. These are
offline height contributions, not re-run total rewards or a causal planning
experiment. In particular, a larger old negative value alone does not prove a
different immediate candidate choice because of the same-parent constant.

Given the reference's visibly broader upper wall, restoring only the absolute
height reward in a controlled unified-simulation experiment is a more direct
next test than treating proposal height or mask removal as a sufficient fix.
Do not infer that long/short simulation alone caused the shape difference.
The height reward formula has not been changed in production by this analysis.
