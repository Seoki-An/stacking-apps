# Excavator depth-one MCTS profile

## Production update

The shared `VoxelOverlapEvaluator` now caches up to 1,024 exact object/pose
intersection volumes and crops voxel tests to conservative object bounds.
Both prior scoring and environment rewards use this implementation. Cache
access is protected for shared workers, and containment and target coverage
reuse the same intersection. Cropping preserves the original voxel centers,
resolution, and face-tolerance predicate, including acute-corner tolerance.

The actual `stacking-excavator-plan --search-only --steps 1` CLI completed its
search in **7.43 seconds**, with default settings and 12 workers. No BLAS
environment override was needed. All **217 relevant tests passed**, and all
282 saved native candidate containment values exactly matched the previous
full-grid implementation. The earlier profile and experiments below describe
the pre-optimization code. Existing running processes retain their imported
code; start a new process to use the update.

On the saved ten-object stack from the earlier PMBS benchmark, seed zero
completed 40 depth-three iterations in **74.41 seconds**, compared with the
previous **165.75 seconds**. It selected exactly the same object and pose, with
the same reward, 60 evaluated candidates, and 71 generated candidates. This is
one follow-up measurement, not a repeat of the full three-seed experiment.
A separate fresh-process depth-one benchmark completed in **8.19 seconds**,
also preserving the previously selected object, pose, and reward. The
[production verification data](benchmarks/excavator-overlap-production.json)
contains both benchmark results, the actual CLI log, and the reference checks.

## Original profile

Measured on 2026-09-21. **Voxel containment in prior scoring is the dominant
bottleneck**, accounting for 53.84 of 61.17 seconds (88.0%) in a fresh native
depth-one search. The existing `pose_generation` timer includes CEM ranking,
so it must not be interpreted as time spent in the native pose solver alone.

## Attribution

The run used the default excavator profiles, 49 catalog objects, an empty stack,
seed zero, 12 native workers, and 200 completed iterations. Application setup
and environment reset were excluded. No hardware or motion planning ran.

| Work | Wall time | Fraction of total |
| --- | ---: | ---: |
| Prior scoring, including repeated CEM ranking | 53.84 s | 88.0% |
| Planar proposal sampling and score maps | 5.87 s | 9.6% |
| Building all 288 pose proposals, including native solves and filters | 0.27 s | 0.4% |
| Long physics validation of 12 retained candidates | 1.07 s | 1.7% |
| Remaining search work | 0.11 s | 0.2% |

These rows represent separate portions of the search. Individual worker-call
times overlap: for example, 1,152 native solve calls across 12 workers summed
to 1.64 seconds, but their enclosing five candidate batches took only 0.27
seconds of wall time. Each proposal executes four native schedule stages.

## Why scoring is expensive

`PlacementPriorScorer.score()` calculates target containment for each candidate.
`VoxelOverlapEvaluator._intersection_volume()` transforms **all 556,150 target
voxel centers** into the object's frame, then tests them against every convex
component's face planes. This includes voxels far outside the object.

CEM ranks its initial pool, re-ranks retained pools to select parents, ranks
combined populations, and ranks the final pool. Six CEM ranking calls plus
the final root-prior call caused **774 containment calculations**, for just
**282 distinct object/pose combinations**. Thus 492 calculations repeated
an identical geometric query. Depth one avoids descendant search but still
pays for this complete root-generation pipeline.

The planar sampler also computed 107 score maps for 49 objects on the same
root scene: 49 for the initial batch, then 29 fresh-object requests in each
subsequent CEM round. Score-map construction took 4.51 seconds within the
5.87-second sampling total. Reusing deterministic maps while retaining fresh
random sampling is a secondary optimization opportunity.

## Diagnostic experiments

At the time of these experiments, only the profiling process was modified.
The subsequent production implementation is described above; search parameters
and platform configuration files remain unchanged.

| Experiment | Search wall time | Prior-scoring time | Process CPU time |
| --- | ---: | ---: | ---: |
| Current implementation, default BLAS threads | 61.17 s | 53.84 s | 1287.37 s |
| Current implementation, one BLAS thread | 60.87 s | 54.15 s | 63.93 s |
| One BLAS thread + cache identical prior-containment queries | 26.32 s | 19.71 s | 29.26 s |
| One BLAS thread + cache + object-bounds voxel cropping | **7.46 s** | **0.89 s** | **10.67 s** |

The cache avoids the 492 repeated queries. The crop retains the original target
voxel centers inside a world-space object bounding box padded by one voxel,
then applies the original containment predicate. It does not change voxel
resolution or reduce the CEM population. The cropped grids averaged 20,630
points rather than 556,150.

All four runs completed 200 iterations, produced identical CEM statistics, and
selected exactly the same object and pose. All 282 unique containment values
were **exactly equal** with and without cropping. An isolated comparison of
those queries took 0.90 seconds with cropping versus 19.70 seconds for their
original full-grid evaluations.

The combined prototype was **8.16× faster** than the one-BLAS-thread control
(87.7% less search time). This establishes an optimization opportunity on this
scene; it is not a production speedup guarantee. The prototype needs geometry
boundary tests, cache-lifetime tests, and validation on nonempty stacks before
being incorporated into the common implementation.

NumPy and SciPy default to 24 OpenBLAS threads on this machine. Restricting BLAS
to one thread alone barely changed latency, but greatly reduced CPU use. It
retained all 12 MCTS workers. Adding more MCTS workers is therefore not the first
remedy for this depth-one workload.

## Optimization priorities

1. **Implemented:** a bounded cache for identical intersection queries, keyed
   by object identity and exact pose within a fixed target/resolution context.
2. **Implemented:** a tested object-bounds broad phase before voxel containment
   tests, preserving the original voxel grid and boundary semantics.
3. Reuse planar score maps within an unchanged state, keeping random proposal
   draws independent and invalidating maps when their inputs change.
4. Expose separate sampling, prior-scoring, and native pose-build timers so
   future logs identify the expensive operation directly.

Reducing the CEM population or coarsening voxels would change search or reward
behavior. The measured redundancy can be addressed before making those tradeoffs.

## Reproduction and artifacts

Use the project Python and a new output directory per trial. Run sequentially:

These commands profile the current checkout. The original measurements above
precede the production optimization; their source hashes are in the raw data.

```sh
python tests/profile_excavator_mcts.py --output /tmp/mcts-profile-default
OPENBLAS_NUM_THREADS=1 python tests/profile_excavator_mcts.py \
  --output /tmp/mcts-profile-blas1
OPENBLAS_NUM_THREADS=1 python tests/profile_excavator_mcts.py \
  --cache-prior --output /tmp/mcts-profile-cache
OPENBLAS_NUM_THREADS=1 python tests/profile_excavator_mcts.py \
  --cache-prior --crop-prior --output /tmp/mcts-profile-crop
```

The [profiling harness](../tests/profile_excavator_mcts.py) records worker-aware
timers, thread CPU time, CEM statistics, selected poses, and cached containment
references. `--cprofile` optionally records the main Python thread, with
additional profiling overhead; it was not used for these measurements.

[Raw measurements](benchmarks/excavator-mcts-profile.json) include all four
summaries, source hashes, and the 282 reference comparisons. The first run used
coarser instrumentation; the subsequent runs also measured containment and
point-transform calls directly. Hardware: Ryzen 9 9900X, 12 cores / 24 logical
CPUs, Python 3.12.3. Each experiment is a single fresh-process run on the same
empty-stack case, rather than a multi-scene quality or scaling study.
