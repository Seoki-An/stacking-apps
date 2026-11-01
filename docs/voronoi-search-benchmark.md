# Voronoi stacking search benchmark

This benchmark prepares simpler irregular assets for the existing native
placement pipeline. It does not presume that MCTS will outperform greedy.

## Prepared problems

`stacking_common.benchmarks.voronoi` clips independently sampled 2D Voronoi
cells to a 2.4 m × 1.6 m rectangle and extrudes each cell by 0.6 m. Two layers
use different partitions. The default preparation creates six cases:
three or four cells per layer, with generation seeds 0, 1, and 2.

Each object has one convex component, closed outward-facing triangles,
unit-density mass/inertia metadata in the existing OBJ format, and a PLY mesh
for the existing height initializer and viewer. The platform inventory supplies
density. Object IDs are shuffled across layers and body-local yaw frames are
randomized. The generator excludes very small cells using only geometry;
no case is chosen based on planner outcomes.

The original polyhedra exactly partition the target, with zero interior overlap.
Their known arrangement therefore achieves geometric target coverage 1.
The benchmark box has `support_overlap: 0`, so its volume contains no synthetic
extension below the ground; the exported inventory target matches this certificate.
`reference.json` stores the privileged poses and a bottom-before-top order.
The planner loads the ordinary inventory and assets; it never loads those poses.

This certificate applies to the generated polyhedra. Finite-sharpness DSF
surfaces, contact margins, pose refinement, and settling can change the realized
arrangement. The native pipeline must validate a reference sequence before
the case enters the greedy/MCTS comparison. Neither the original partition nor
a refined native reference certifies the optimum of the platform's discounted,
multi-term reward. The reported mesh-union coverage also differs from exact
smooth-DSF occupied volume.

## Prepare without starting solver work

From the repository root:

```bash
MPLCONFIGDIR=/tmp/voronoi-prep-mpl python tests/benchmark_voronoi_mcts.py prepare \
  --dataset .stacks/benchmarks/voronoi-search-20261101
```

Preparation generates OBJ/PLY files, reference poses, a geometric preview,
and frozen application/common source and excavator profile snapshots. It
records asset/source/profile/native-library hashes. It performs no pose solve,
simulation, or planning and does not start a background queue.

Native commands are blocked while the recorded full depth benchmark is
unfinished. The default status file is
`.stacks/benchmarks/mcts-full-rollout-20261101/status.json`.

## After the depth benchmark finishes

First validate one reference using the current pose generator, its existing
height initializer and admission filters, and the normal settling, structural,
and robustness evaluation:

```bash
python tests/benchmark_voronoi_mcts.py validate-reference \
  --dataset .stacks/benchmarks/voronoi-search-20261101 --case cells3-seed0
```

The reference supplies XY/orientation only to this validation command. Native
pose refinement and all admission/transition checks still run. The summary
records failures and XY drift; a successful refinement is not a claim that the
original exact tiling remains unchanged. Reference failures stay visible and
block comparison runs for that case. Preparation does not certify a native
reference as passed.

For a validated case:

```bash
python tests/benchmark_voronoi_mcts.py run \
  --dataset .stacks/benchmarks/voronoi-search-20261101 --case cells3-seed0 \
  --method greedy --seed 0
python tests/benchmark_voronoi_mcts.py run \
  --dataset .stacks/benchmarks/voronoi-search-20261101 --case cells3-seed0 \
  --method mcts --seed 0
```

`greedy` uses the same root CEM generation, native pose optimization, reward,
filters, and physical admission as MCTS. Its depth is 1, bootstrap horizon is 0,
root lookahead is disabled, and final scoring uses the actual root reward.
It therefore selects the highest immediate reward among admitted candidates.
This is checked at runtime; a depth-1 search with arbitrary heuristic scoring
is not used as a proxy for greedy. Ties retain generator order.

`mcts` uses the frozen current MCTS configuration with tree preservation.
Both methods run from empty inventory to search exhaustion or inventory
exhaustion. Both receive a 60 s search cap; greedy can finish earlier. This is
an equal-cap comparison, not an assertion of equal consumed compute. Summaries
record actual time, evaluated candidates, reward, and achieved mesh coverage.

Each run writes an excavator Operations-compatible `run.stack`, including
context, search statistics and checkpoints. Reference validation artifacts are
separate from competing planner outputs. Existing outputs are never overwritten.

The optional suite validates every generated case, keeps validation failures in
`suite-results.json`, and runs only physically admitted cases with both search
seeds. Method order is reversed between seeds:

```bash
python tests/benchmark_voronoi_mcts.py suite \
  --dataset .stacks/benchmarks/voronoi-search-20261101 --seeds 0 1
```

Choose a different MCTS profile explicitly with `--mcts-config` after examining
the completed depth benchmark. No automatic winner is assumed. New snapshots
are required if the native library or prepared inputs change.

## Interpretation

Inspect full trajectories, placed count, physical coverage, remaining space,
surface variance, time and evaluation counts. Compare at matching placement
counts as well as natural termination. A discounted return alone can obscure
late failures. Report all generated cases and reference-validation failures;
illustrative cases where greedy loses do not establish general superiority.

This preparation does not introduce a discrete oracle or inject reference
actions into ordinary sampling. A finite-action exact-search diagnostic can be
added separately if needed; it would certify only that discretized action set.
