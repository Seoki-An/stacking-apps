# Current MCTS strategy

> Historical staged-simulation description. The runtime strategy changed on
> 2026-10-27: see [Unified MCTS simulation](mcts-unified-simulation.md) for the
> current single simulation, depth filter, and subtree reuse behavior.

This document describes the shared MCTS implementation and the excavator CPU
and CUDA profiles as of 2026-10-21, including continuation repair. MCTS searches
placement sequences, validates the immediate action and required continuations,
then retains the selected subtree for the next placement. CUDA accelerates
batched pose generation and simulation; tree selection and backup remain on
the CPU.

For the manuscript overview, see [MCTS algorithm](mcts-algorithm.md). Detailed
equations, reward definitions, and the tabletop comparison are in the
[appendix draft](mcts-algorithm-appendix.md). Physical evaluation stages
are described in [MCTS simulation](mcts-simulation.md).

The configuration values below are a snapshot. The active YAML profile and any
runtime overrides determine a run's behavior; saved `.stack` contexts record
the resolved settings. See [MCTS batching](mcts-batching.md) for the PMBS-style
scheduler and [excavator planning](excavator-planning.md) for native solvers,
execution workflows, and historical benchmarks.

## Composite target foundation

As of 2026-10-26, common target geometry, the native posegen adapter, proposal
sampling, active-floor checks, rewards, observations and full MCTS entry points
accept multiple convex components.
The existing single tapered-target YAML and its
native construction remain unchanged. A composite is written under
the selected entry in `target_shapes.yml` as `components`; each piece has a stable string `id`,
an optional `[x, y, z, qx, qy, qz, qw]` pose, and either `shape` or `nodes`:

```yaml
example_shape:
  components:
    - id: west
      pose: [-3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
      shape: &box
        width: 2.0
        length: 2.0
        height: 2.0
        origin: [0.0, 0.0, 0.0]
        support_overlap: 0.02
        taper_degrees:
          width_positive: 0.0
          width_negative: 0.0
          length_positive: 0.0
          length_negative: 0.0
    - id: east
      pose: [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
      shape: *box
```

`shape` uses all the existing tapered-volume fields in the component's local
frame, including its local `origin`. Alternatively, `nodes: [[x, y, z], ...]`
defines the vertices of one nondegenerate convex hull in that local frame.
The component pose then transforms it into the common search frame. A missing
pose means identity. Nodes are inline in this milestone; external target asset
loading is not introduced. Native target sharpness, epsilon and friction still
come from the platform posegen profile. Do not mix `components` with the legacy
target-level dimensions.

### Named target selection

Both platforms keep object assets, IDs and densities in `config/inventory.yml`,
and named target definitions in `config/target_shapes.yml`. The registry has
`units: si`, a `default` name, and a `shapes` mapping. Each shape is either the
existing tapered-volume mapping or a `components` mapping as illustrated above.
The current defaults are tabletop `tapered_wall` and excavator `default`
(the wall recorded in `excavator_plan17.stack`). The previous excavator
`tapered_tower` remains selectable. Each platform owns its physical
dimensions; there is no shared global scale.

Both operations GUIs offer a **Target shape** dropdown in **New planning run**.
Selection is per-run and does not rewrite configuration. The CLI equivalent is
`--target-shape NAME` on either the tabletop debug MCTS script or
`stacking_excavator.planning.cli`. Omitting it selects the registry default.
Examples include tabletop `box` / `two_piers`, and excavator `pyramid` /
`two_walls`. A composite is one goal; selecting multiple batch names still
creates separate searches.

Resolved inventory profiles retain the selected target and its name so existing
application consumers need not depend on registry files. Saved planning
contexts include both; search artifacts carry the actual target geometry.
Resume restores the recorded target without consulting today's registry and
locks the GUI selector. A conflicting `--target-shape` is rejected. Other
native/profile/asset resume checks remain unchanged. Legacy inline `target:`
inventory files still load, and legacy recorded profiles without a target name
retain their serialized layout/hash. A named override explicitly selects the
adjacent registry even when an old inventory has an inline target.

`CompositeTargetGeometry` preserves separate posed meshes, aggregate `bounds`,
component IDs, union `contains(points)` / `horizontal_mask(xy, z)` queries,
`component_membership(points)`, and distance to the union's XY footprint.
The combined `points`/`faces` concatenate meshes; they never form a global
convex hull. Overlapping pieces may both contain a query point, but the union
contains it only once. Component IDs identify geometric pieces, not independent
construction goals or necessarily disconnected regions. There is deliberately
no synthetic rectangular `width`, `origin`, `taper`, or summed `volume` on the
composite; resolution-dependent union-volume accounting belongs to the overlap
evaluator rather than the geometry contract.

Direct native posegen can be used with an inventory loaded from this profile:

```python
inventory = application.load_inventory(object_ids, initial_poses)
solver = application.build_pose_solver(inventory)
# Supply explicit InitializedPlacement proposals to solver.solve(...).
```

Both CPU and CUDA adapters build **one target body with multiple DSFs** using
`BodyModel.dsf_components`. Component poses are baked into its nodes once;
the enclosing target body stays at identity. CUDA accounts for all target
geometries while reserving only one target-body slot. The native objective
retains its existing conservative semantics: it selects the component with
the best whole-stone containment margin. A stone spanning a component seam
can therefore be penalized even when entirely inside the geometric union.
Separate target entity IDs are not used, since their containment penalties
would accumulate. Composite results record `posegen_target_component_ids`,
`posegen_target_component_count`, and
`posegen_target_containment: whole_stone_in_one_component`.

### Region-aware action generation

`application.build_action_generator(inventory, solver)` supports composites.
Touching or overlapping component cross-sections at the common ground plane
form one connected sampling region. Each disconnected region has its own floor grid,
floor coverage and layer progression, so the first stone in an empty region
does not need to connect to stones in another region. Coverage counts only
cells inside the union; gaps, notches and overlapping pieces are not counted
as extra floor area. All placed stones still contribute to support-height
initialization and the shared native physical scene.

The sampler shares one `planar.grid_shape` height map over the target bounds,
rasterized from all placed meshes with physical ground retained in target gaps.
Support maps are computed once per object/orientation and reused across regions.
If a region has no eligible candidates, it retries a region-local grid at the
same configured resolution. This prevents narrow or distant regions from
silently disappearing, while floor grids remain independent. Anchor masks check
oriented stone vertices against component half-spaces at their predicted
placement height, including taper. The existing XY inset scale remains a
sampling heuristic; full containment is still enforced by native validation.
Corner seeds are feasible footprint extrema in each empty region. Fresh draws
rotate across available regions, and CEM elite-parent selection reserves
regional representatives within the existing elite/per-object limits. Final
candidate ranking rules are unchanged, although coarser proposals can change
numerical scores and the selected sequence.

Regional proposal scoring caches immutable component half-spaces and uses a
bounded, sampler-local LRU for vertex/orientation-dependent containment offsets.
Height and position remain live query inputs; inset and margin are cache keys.
Candidate banks use read-only NumPy columns, preserving orientation/grid order.
Sampling selects row indices and constructs metadata only for selected rows;
`score_map()["candidates"]` remains an iterable/indexable diagnostic view whose
records are materialized on demand. The shared elite/Boltzmann draw logic and
RNG consumption are unchanged. In particular, composite draws still request
one row at a time, so any positive elite fraction selects one elite per draw;
this performance change does not introduce a different exploration policy.

Proposals carry `target_component_ids`, `target_region_ids` and a primary
`target_region_id`. Mutation and accepted optimized poses recompute membership
from projected geometry; these labels are diagnostics, not proof of 3D
containment. Solved candidates must overlap an actual component footprint,
even when floor-fill restrictions are relaxed.

Components must intersect the same physical ground plane over an area. This
plane comes from the platform posegen objective's `ground_height` (default
zero); elevated components requiring independent support are rejected for now.
Single-target profiles retain their original target-origin ground convention.
No additional YAML settings are needed.

### Union rewards and observations

`VoxelOverlapEvaluator` uses one shared voxel lattice for composite targets.
It enumerates component-local bounding boxes and deduplicates occupied cells,
without allocating the empty volume between distant components. The existing
platform `overlap_resolution` controls the approximation; components too small
to resolve are rejected instead of silently disappearing. Its `target_volume`
is the voxel-union volume, not a sum of component volumes. Single-target
evaluation retains its original lattice and analytic volume denominator.

Composite queries use immutable component-local 3D lookup blocks into this
same deduplicated lattice. A stone's bounding box slices those blocks directly,
instead of scanning full X slabs. Overlapping blocks share global voxel IDs;
query results are deduplicated in the original lattice order. The index adds
memory proportional to component bounding boxes, not the gap between them,
and changes neither voxel centers, containment tolerances, nor union rewards.

For composites, reward coverage is the union of all placed stones inside this
target lattice. Overlapping DSFs and overlapping stones cannot earn duplicate
coverage. The incremental reward uses newly covered volume divided by the
fixed inventory mean stone volume, so adding a remote target does not dilute
the reward for placing a stone. The historical `target_IoU` diagnostic remains
a coverage fraction, not mathematical intersection-over-union. Rewards also
report `target_volume` and `target_volume_method` (`voxel_union` or `analytic`).
Previous coverage is evaluated at the current settled poses, as before.

Directional and height heuristics select the nearest component footprint,
breaking ties by distance to its vertex centroid and then component ID. This
avoids pointing inward toward an aggregate center in a gap. These are local
component heuristics, not decomposition-invariant union metrics. Surface
flatness uses 32-by-32 grids per connected ground region, masks actual union
footprints and weights cells by physical area; gaps and component overlap do
not bias the mean. Native posegen's whole-stone-in-one-component constraint is
unchanged even though geometric reward containment measures the true union.

Observations preserve concatenated component meshes plus `target_kind`,
`target_bounds`, stable component IDs, vertex/face offsets and an actual
`target_footprint_mask`. For composites only, the legacy width/length/height
and origin fields describe the axis-aligned bounding envelope (origin at its
bottom center), **not** a rectangular target. Target height maps render the
upper envelope of the separate meshes. Their global fixed-resolution grid
can lose detail for widely separated regions; proposal sampling continues to
use its own region-local grids. Learning records include the component data;
existing learned models are not automatically made composite-aware.

Direct `build_environment` and `build_committed_state_evaluator` calls can now
observe and score composite scenes. Deferred observations and full observation
records retain the component metadata.

### Saved runs, target replay and checkpoint branching

Composite target headers use MCTS artifact schema **2** and contain local
component points, triangle faces, sharpness, stable IDs and rigid poses, plus
the display structure offset. The SQLite `.stack` container remains version 1.
Single-target artifacts retain schema 1 and their existing target payload;
older files still open without conversion. Older application builds correctly
reject composite schema 2 rather than displaying a false single box.

Composite planning contexts additionally record `profiles.target_geometry`.
It is part of the context hash referenced by each checkpoint and motion plan.
Resolved target profiles are also retained and can be reconstructed without
reading current YAML. Single-target context layouts/hashes are unchanged.
Replay prefers the explicit recorded meshes. Component edits change the
context hash and are rejected by resume validation.

Both operation viewers render component shells separately in one target actor,
without bridging gaps or taking a global convex hull. Shared boundaries and
overlapping component surfaces can remain visible: this is a rendering of the
DSF decomposition, not a Boolean-remeshed union surface. Search-to-planning
poses transform all components together. Tests cover reopening, candidate and
robot-motion replay, retained state/context, and branching both planning and
search-only checkpoints into a new file while preserving the source bytes.
These persistence/replay tests are separate from the native search tests below.

The excavator's optional score-map inspector still requires migration to
region-aware score maps; composite requests report this explicitly without blocking
the main viewer. No robot execution is enabled by opening or branching a file.

### End-to-end search validation

The shared search builders and both MCTS CLIs now accept composites. Before
allocating search workers or creating a run, they verify that every component
intersects the platform posegen `objective.ground_height` over an area. Elevated
components still require a different support policy and fail explicitly. The
independent single-target shape examples and old `.stack` files remain supported.

`tests/test_composite_target_search.py` exercises disconnected, overlapping and
L-shaped targets through native CPU posegen, short/long simulation, disturbance
validation, required descendant validation, two successive placements and tree
promotion. Synthetic DSF stones run with one and two CPU workers; additional
tabletop CLI cases load catalog stones 1–3. Both paths save and reopen schema-2
`.stack` results. Only search budgets/targets are overridden, not the platform
physics tolerances or safety filters.

The catalog cases exposed a raster-support edge case: a cell-center kernel can
miss an irregular stone's lowest vertex and predict a pose slightly below
ground, causing every full-vertex target-containment check to fail. Composite
sampling now clamps its estimated placement height to the exact vertex-derived
ground clearance before checking containment. This does not relax containment
or simulation tolerances; tests cover both zero and nonzero ground heights.

Run the bounded native regressions with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 QT_QPA_PLATFORM=offscreen \
  python -m pytest -q tests/test_composite_target_search.py
```

These are integration smoke tests, not a full field-scale or robot-motion
benchmark. No CUDA device was visible during this validation; native CUDA
end-to-end behavior remains unverified. Region-local score-map inspection and
union-aware native seam containment remain deferred; no stacking-core change
is needed for the conservative whole-stone-in-one-component adapter.

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
| Iterations per search | 1024 | 1024 |
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

The excavator CPU profile initially uses depth one until eight objects have
been placed; descendant validation is capped to zero levels during that period.
CUDA uses its full configured depth from the first placement. Both share
discount `0.9`, three root retry rounds, and progressive widening with
coefficient `3.0` and exponent `0.5`. The exploration constant is `2.0` for
excavator CPU and `1.0` for excavator CUDA. Tabletop CPU and CUDA use depth four
from the first placement, exploration constant `1.0`, and two descendant levels.

Sources: [CPU profile](../apps/excavator/config/mcts_cpu.yml) and
[CUDA profile](../apps/excavator/config/mcts_cuda.yml).

## Candidate generation and exploration

CEM generates and ranks root proposals, then combines mutations of elites with
fresh proposals. Both profiles use a fresh fraction of `0.5` and a baseline
`max_per_object` of two. With `N` unplaced inventory objects, the effective cap
is `max(max_per_object, ceil(keep / N))`; elite-parent selection uses `elite`
instead of `keep`. Thus fewer remaining stones can contribute more distinct
poses without increasing the total pool or elite budget. An unset cap remains
unlimited; insufficient feasible poses can still leave the pool underfilled.
Root and deeper sampling use `lower_floor_first`.
These proposal scores are priors; physical simulation still decides whether a
placement is acceptable. More precisely, the default tree prior logit combines
equilibrium and containment, while CEM adds the normalized planar score to rank
its pool. CEM settings govern the root pool, not all descendants.

For a non-root node with stored visits `N` (including reset pseudocounts), the selectable-child allowance is:

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
      + c * P_child * sqrt(N_parent + V_parent)
          / (1 + N_child + V_child)
```

`W` is accumulated return, `N` stored visits including reset pseudocounts, `V` reservations in the current
batch, and `P` the softmax prior. PUCT requires `N_parent + V_parent > 0`.
A zero denominator gives a zero mean. Pending
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

The excavator CPU profile sets mean reward to 2.5; excavator CUDA and both
tabletop profiles use 1.0. All use horizon 4. The implementation
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
repair while root retries and floor/corner relaxation continue. If those attempts
are exhausted (including the wall-time budget) and no qualifying root remains,
final selection falls back to the best otherwise-feasible root by `M(a)`.
Candidates from retired retry pools are retained for this last-resort selection.
Only roots that passed authoritative placement validation are eligible; physical
failures, unvalidated roots, and execution exclusions are never revived. A root
with a validated continuation always takes precedence over this fallback.
With the requirement false, an unavailable path falls back to the mean score immediately. In
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

That allows up to three rounds of 12 evaluations for excavator CPU after its
depth-one warmup, and six rounds of 256 for excavator CUDA from the first
placement. These are ceilings, not
targets. Existing widening, generation-attempt limits, generator exhaustion,
and the remaining overall deadline can stop repair earlier. Already validated
root paths remain available if repair cannot recover other alternatives.

**Root retries run only if final selection returns no candidate.** They retire
rejected roots, request a diversified CEM pool excluding earlier roots, and
long-validate fresh placements. Initial lookahead plus retry depth extension
builds continuations to the required depth before final selection and repair
run again. Increasing root retry rounds alone does not improve alternatives
when some root already qualifies.

Before replacing a physically validated root, the planner can also repair
its missing continuations with relaxed floor admission, once per root. It
preserves the root and existing physical validation evidence.

**The sampling fallback runs only if root retries also select nothing.** An
empty scene relaxes corner seeding to ordinary scored sampling. A nonempty
scene relaxes floor admission while retaining lower-floor priority, fill-area
scores, sampling connectivity, and physical validation. It repeats root
preparation, lookahead, retry depth extension, and final selection on fresh root
pools. After the first pool, up to `rejected_root_retry_rounds` further pools
each exclude every earlier candidate. The fallback stops at the first
selection, when CEM has no fresh proposals, or at the original overall
deadline. Saved statistics record its pool, generated, and accepted counts.

`missing_validated_descendant` means no qualifying continuation was found within
these limits. It does not prove the immediate placement is physically invalid
or that no feasible continuation exists.
When final selection relaxes this requirement, the selected/ranked candidates
record `validated_descendant_fallback: true`,
`validated_descendant_path_available: false`, and
`final_selection_used_mean_fallback: true`. This does not claim that their
continuations passed validation.

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

Current runtime requests use `reset_statistics: true`: evaluated nodes reset
recursively to `N = 1, W = 0`, while pending nodes reset to `N = 0, W = 0`.
The initial zero-return visit guarantees positive counts for reused parents;
after `n` actual backups, a reset node has mean `W / (1 + n)`. Authoritative
root validation still adds its reward backup after reset. Requests without
reset must retain positive parent counts before PUCT comparison. The tree
structure and evaluation results remain. This provides candidate and simulation reuse,
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

Current CUDA profiles configure 512 pose-generation jobs, 128 placements per GPU
simulation chunk, and 512 native simulation jobs to accommodate robustness
trials. Reward and observation processing overlaps subsequent CUDA chunks.
Tree selection and backup remain serial CPU work. A configured batch of 256
does not guarantee 256 available independent jobs.

The current CUDA robustness history-download chunks use an estimated 1.5 GiB
sample-storage budget, increased from 512 MiB. Nested action diagnostics are
shared read-only across copies to reduce host work. These implementation
changes retain the tree policy and reward definitions. For current reward
weights, including excavator's disabled orientation term, see the
[reward appendix](reward-design-appendix.md).

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
