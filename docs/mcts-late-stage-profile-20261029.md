# Late-stage MCTS simulation profile — 2026-10-29

The main measured simulation bottleneck is the **6D contact-constraint solve**.
It accounts for 72–90% of kernel time in the captured structural batches. Contact
feature detection is the second priority. Transfers, patch construction, and
constraint assembly are substantially smaller costs.

This is an attribution study, not an optimization A/B comparison. Production
solver code and configuration were not changed for this experiment.

## Workloads and controls

- Saved scenes: `.stacks/simple_mcts_test7.stack`, parent checkpoints 30, 35, 40,
  corresponding to searches for steps 31, 36, 41.
- Fresh trees; current excavator `mcts_cuda.yml`; seed 0; 512 iterations; depth
  limit 4; original 240-second search limit. Excursion early stopping enabled;
  experimental quiet-window stopping disabled. All searches finished 512
  iterations without timing out.
- Settling, unified structural simulation, robustness, physical thresholds, and
  candidate policies were left unchanged.
- The installed native binary was copied to a private `/tmp` package and used
  for every run. SHA-256:
  `b18e7e2e029953d5ec89691726a4e59c1c5171153921ff5a516ab7d058e32d46`.
- Relevant Python sources and excavator configuration hashes stayed unchanged
  throughout all three runs. The runner records them in `manifest.json`.
- Hardware: NVIDIA RTX 5060 Ti, 36 SMs; CUDA reports 1,536 maximum resident
  threads per SM. This capacity is not a measurement of achieved occupancy.

Lightweight wall timers ran during search. Nsight CUDA collection stayed off
until search completed. The slowest observed native call in each of settling,
structural simulation, and robustness was retained in memory and replayed
three times without tracing, then once with CUDA graph-node tracing. This
captures real heterogeneous MCTS batches, including early-stopped branches.

All nine captured batches matched their original numerical output fingerprints
on every replay, including traced replay. Fingerprints cover statuses, final
poses/velocities, histories, contacts, phase metrics, and final solver statistics.
Kernel timings describe these selected slow batches, not every call in search.

## Full searches

| Step | Search wall | Pose generation | Evaluation wall | Evaluations | Rejected | Reached depth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 31 | 98.47 s | 42.02 s | 55.89 s | 575 | 454 | 4 |
| 36 | 147.61 s | 51.35 s | 95.69 s | 567 | 393 | 4 |
| 41 | 55.55 s | 26.28 s | 29.16 s | 142 | 133 | 2 |

Evaluation wall includes physics and CPU transition/reward preparation. It is
not pure GPU time. The following native call times are components of evaluation,
not additional time to add to the table above:

| Step | Settling native | Structural native | Robustness native | Structural job-steps |
| --- | ---: | ---: | ---: | ---: |
| 31 | 2.07 s | 24.98 s | 17.79 s | 111,316 |
| 36 | 2.40 s | 52.43 s | 30.91 s | 109,500 |
| 41 | 1.00 s | 18.53 s | 2.65 s | 24,195 |

The step-31 to step-36 comparison is particularly informative: evaluation
counts and structural integrated job-steps are nearly unchanged, but structural
wall time more than doubles. Batch composition also changes (10 versus 12
native structural calls), so this does not isolate a per-iteration solver cost.

Step 41 is **not a speedup**: only nine evaluated candidates survived. The tree
has 143 nodes, including its root, and stops growing beyond depth 2 despite a
depth limit of 4. Its lower workload explains the short search time. The original
saved step 41 evaluated 370 candidates and took 179.54 s. Fresh-tree/current-code
replays are not exact reconstructions of the original preserved-tree search;
absolute wall times must not be presented as an improvement over that run.

## Structural CUDA kernel breakdown

Percentages below are fractions of summed kernel execution time in the captured
structural batch. Native replay wall times are medians of three untraced replays.

| Step | Batch jobs | Bodies/job, including ground | Median final contacts | Native replay wall | Constraint solve | Detection/features | DSF queries | Patches | Assembly |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 31 | 128 | 33–35 | 119 | 4.99 s | 72.2% | 15.6% | 5.1% | 3.5% | 2.3% |
| 36 | 128 | 38–40 | 143 | 8.34 s | 76.7% | 13.3% | 4.0% | 3.0% | 2.0% |
| 41 | 18 | 43 | 160 | 4.18 s | 89.6% | 4.3% | 2.0% | 0.7% | 2.7% |

The dominant kernel is `solve<6>(constraint_view_t)`. The `features` kernel
includes bounding tests, cached-normal separation checks, and analytical plane
contacts; its entire duration should not be called broad-phase cost alone.
Integration and bookkeeping account for the small remainder.

Constraint solving also accounts for 69.6–80.6% of kernel time in the captured
robustness batches. Improving this kernel can therefore help both nominal
structural simulation and robustness while retaining unified simulation.

These traces do not separate projection, body-velocity updates, convergence
checks, or iteration count within `solve<6>`. They identify where to investigate,
not which internal operation is already proven to be the cause.

## Transfers and parallelism

For the step-36 structural batch:

- Summed kernel execution: **8.184 s**; traced call wall: **8.380 s**.
- Device transfer execution: **31.2 ms**, including approximately 622 MB copied
  device-to-host. Buffers still reserve the maximum history capacity.
- `cudaMemcpyAsync` CPU API duration: **8.216 s**. Most of this is waiting for
  queued GPU work, not time spent transferring data on the device.
- Graph instantiation: approximately **0.72 ms**. Launch/graph management is
  not the primary cost in this batch.

The solver launches **one 128-thread block per job** (four warps), using 64
registers/thread in these traces. The step-41 batch launches only 18 blocks on
a 36-SM GPU: it cannot occupy every SM even before jobs finish early. Increasing
configured capacity alone cannot produce additional independent jobs.

Early termination also reduces active work within a batch. In the selected
step-31 and step-36 structural batches, branches finished after 23–392 and
24–421 steps respectively. Integrated job-steps were 48.7% and 50.9% of
`jobs × longest_branch_steps`. Those ratios describe branch completion spread;
they are **not GPU utilization measurements**. Kernel grids retain stopped slots.

## CPU work and overlap

| Step | CPU geometry-preparation interval | Overlap with native simulation | Structural trajectory-processing time |
| --- | ---: | ---: | ---: |
| 31 | 21.47 s | 15.68 s | 6.72 s |
| 36 | 24.44 s | 23.23 s | 8.03 s |
| 41 | 7.76 s | 2.50 s | 2.22 s |

These intervals overlap each other and native work; do not sum them as separate
wall-time contributions. Voxel-overlap computation dominates summed geometry
worker time, but much of its wall interval is already hidden by robustness
simulation, especially at step 36. Native thread CPU times can include CUDA
driver waiting/spinning and are not a measure of useful CPU computation.

## Next priorities while preserving simulation behavior

1. **Benchmark more parallel work inside each 6D solver job.** The current
   implementation has four warps traversing roughly 40 bodies and 140–160
   contacts. A narrowly scoped 128-versus-256-thread 6D specialization is a
   concrete first experiment, with register/launch-bound tuning, unchanged
   equations and convergence tolerances, and exact replay checks. A gain is a
   hypothesis, not established by this profile; register pressure and reduction
   overhead may offset the extra parallelism.
2. **Reduce repeated work in contact feature detection.** Investigate computing
   world-space geometry transforms/bounds once per timestep rather than for
   every pair, preserving the same margin and all rejection tests. This targets
   the measured secondary cost without adding a contact-selection heuristic.
3. **Then consider result/geometry processing.** Preserve the existing overlap.
   Reducing histories or transfers alone has a small measured ceiling here;
   exact geometry caching needs to account for every stone that actually moved.

The profile does not justify shortening validation, freezing additional stones,
loosening thresholds, or restoring short/long simulation. No production
optimization was made in this comparison.

## Artifacts and reproduction

All artifacts are in `/tmp/mcts-late-profile-20261029/`:

- `summary.json`, `kernel_breakdown.csv`: aggregated measurements.
- `step{31,36,41}-search.json`, `*-stages.json`, `*-native.json`: search timers,
  stage timers, captured workloads, repeat timings, and parity fingerprints.
- `step{31,36,41}.{1,2,3}.nsys-rep` and `.sqlite`: Nsight traces; ranges 1/2/3 are
  settling/structural/robustness respectively.
- `manifest.json`, `run_comparison.py`, `summarize.py`: commands and source hashes.
- `state{30,35,40}.pkl`: locally exported trusted parent states.
- `native/stacking_core`: the private pinned binary used by every run.

The reusable capture driver is `tests/profile_mcts_simulation_kernels.py`. For
one search, wrap its command with:

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --capture-range=cudaProfilerApi \
  --capture-range-end=repeat:3 --output /tmp/step31 \
  python tests/profile_mcts_simulation_kernels.py --cuda-capture \
  --native-profile-output /tmp/step31-native.json \
  --profile-output /tmp/step31-stages.json \
  --backend cuda --seed 0 --iterations 512 --early-stops excursion \
  --config apps/excavator/config/mcts_cuda.yml \
  --state /tmp/mcts-late-profile-20261029/state30.pkl \
  --output /tmp/step31-search.json
```

Use `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, the same native binary, and
unchanged sources/configuration across comparisons. Traces start after search;
profiled replay wall time is not used as the untraced performance measurement.
