# Excavator score-map reuse and GPU containment benchmark

Measured on 2026-09-21, after the containment cache and voxel-cropping change.

## Production score-map reuse

`ScoredPlanarPoseSampler` now retains deterministic score maps for the current
scene. Its cache checks state poses and placement order, height-map contents and
shape, grid limits, settings, and candidate orientations. Changes invalidate
the relevant maps, including changes made in place to state/observation arrays.
Keeping only the current scene bounds memory use as the search visits branches.

Every sampling request still makes fresh random draws. The RNG, candidate
ordering, CEM population, iterations, voxel resolution, and MCTS settings are
unchanged. The tabletop sampler is unchanged.

On the default empty-stack excavator search, with 49 objects, seed zero, 12
workers, and 200 completed iterations:

| Measurement | Before map reuse | After map reuse |
| --- | ---: | ---: |
| Total search | 7.550 s | 5.010 s |
| Score-map constructions | 107 | 49 |
| Score-map construction time | 4.360 s | 1.951 s |
| Sampling calls | 107 | 107 |
| Total sampling time | 5.826 s | 3.398 s |

This paired profile reduced total search time by **33.6% (1.51× faster)**.
CEM statistics and the selected object/pose were identical. The actual
`stacking-excavator-plan --search-only --steps 1` CLI also completed its search
in **5.07 seconds**. Timings exclude application construction and state reset.

All **225 relevant tests passed**, including legacy fixture parity, repeated
fresh draws, CEM mutation lookup, equivalent reconstructed observations, and
invalidation after changing heights, bounds, placements, poses, settings, or
orientations. Empty and nonempty scenes are covered.

A native follow-up on the saved ten-object stack completed 40 depth-three
iterations in **62.97 seconds**. The selected object, pose, reward, CEM
statistics, and counts of 71 generated / 60 evaluated candidates matched the
previous containment-optimized run (74.41 seconds). This is one follow-up
check, not a repeated speedup study: physics timings also varied between runs.
The [CPU profiles and native verification](benchmarks/excavator-scoremap-cache.json)
include the full results and CLI log.

## GPU containment experiment

The RTX 5060 Ti (16 GB) was accessible outside the sandbox. The benchmark used
PyTorch 2.7.0+cu128, CUDA 12.8, and NVIDIA driver 580.178.04. Production remains
CPU-based; the GPU implementation lives only in the diagnostic harness.

Both CPU and GPU paths retain the same target voxels, conservative object-bounds
cropping, face tolerance, convex-component union, and double precision. The GPU
keeps geometry and target voxels resident, groups queries by object, and returns
only voxel counts. Pose preparation, per-batch transfers, and result retrieval
are included in the measured GPU call.

The CPU measurements include the current sequential calculation and a
12-worker thread-pool variant, both with one OpenBLAS thread. All queries are
cache misses: repeated timing calls bypass the intersection cache on both
sides. Cached queries should remain CPU cache lookups in any future integration.

Median times, using GPU batches of at most 16 poses per object:

| Distinct poses | Current CPU path | 12-worker CPU variant | GPU | GPU speedup vs current / parallel CPU |
| ---: | ---: | ---: | ---: | ---: |
| 96 | 0.312 s | 0.097 s | 0.050 s | 6.21× / 1.94× |
| 282 | 0.899 s | 0.261 s | 0.152 s | 5.94× / 1.72× |
| 1,128 | 3.587 s | 1.053 s | 0.611 s | 5.87× / 1.72× |

Every result matched the CPU value exactly. Each CPU timing used three measured
repetitions and each GPU timing five, after warm-up. CUDA synchronization
surrounded timed calls. GPU batch limits 1, 16, and 64 were checked; all results
are in the raw data. The larger workload uses translated copies of the 282
recorded poses to create distinct queries, rather than cacheable duplicates.

One-time CUDA initialization took 0.099 seconds and geometry upload 0.010
seconds; these are separate from steady-state timings. Python/PyTorch import,
asset loading, and model construction are excluded. Peak CUDA tensor allocation
with batch limit 16 was about 242 MiB on the largest workload.

These are containment throughput measurements, **not GPU MCTS speedups**.
Queries span recorded CEM rounds; real rounds have dependencies and cannot all
be submitted together. A production GPU integration would also need to batch
cache misses through the scorer and account for scalar callers and shared
simulation workers. Against the optimized CPU search, the measured difference
for 282 queries is roughly 0.75 seconds, or about 0.11 seconds compared with
the parallel CPU variant. Those are component-level opportunities, not measured
end-to-end savings.

## Reproduction

CPU search profile, using a new output directory:

```sh
python tests/profile_excavator_mcts.py --output /tmp/scoremap-profile
```

GPU comparison, with optional CUDA-enabled PyTorch installed and device access:

```sh
OPENBLAS_NUM_THREADS=1 python tests/benchmark_gpu_containment.py \
  --output /tmp/gpu-containment.json
```

The [GPU harness](../tests/benchmark_gpu_containment.py) is independent of the
production backend. [Raw GPU measurements](benchmarks/excavator-gpu-containment.json)
include every timing repetition, exact-equality checks, memory use, and input
and benchmark hashes. These measurements use one machine and recorded scene
distribution; they do not establish multi-scene stacking quality.
