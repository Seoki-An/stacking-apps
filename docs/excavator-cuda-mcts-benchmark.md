# Excavator CPU versus combined CUDA MCTS benchmark

## Robustness noise clearance experiment — 2026-10-12

### Production refinement

Clearance correction is now enabled in the CPU and CUDA robustness paths.
The refinement replaces the prototype's fixed 25 cm allowance with a bound from
the sampled translation, rotation, native object radius, and existing contact
tolerance. Unresolved releases are filtered on both platforms and reported as
`place_robustness_initial_clearance`; lateral, lift-limit, and query failures are
retained per trial. See [the policy](excavator-planning.md) for its scope.

Production replay of the same 32-candidate sets (128 trials), one warmup and
three alternating measurements per mode:

| Placed stones | Uncorrected placement batch | Clearance computation | Computation / baseline | Unresolved trials |
| --- | ---: | ---: | ---: | ---: |
| 0 | 442.52 ms | 6.60 ms | 1.49% | 0 / 128 |
| 7 | 1581.92 ms | 22.50 ms | 1.42% | 32 / 128 |
| 14 | 1945.59 ms | 43.03 ms | 2.21% | 9 / 128 |

[Production replay data](benchmarks/excavator-robustness-clearance-refined-20261012.json)
includes individual timings and rejection counts. This is computation overhead;
the filter changes simulation outcomes and the search tree.

The integrated production smoke search completed 64 iterations without timeout
in 12.00 s and selected stone 44. It evaluated 222 candidates, versus 204 in the
earlier uncorrected search, so its elapsed time is not an isolated overhead
measurement. Filtering confined releases can require additional search work.
Validation: 192 focused CPU/CUDA, clearance, placement, and environment tests passed.

### Earlier prototype

The benchmark-only prototype checks noisy release poses against the native DSF
collision geometry, then uses bounded upward corrections and rechecks clearance.
It batches ground and nearby-stone queries and caches workspaces sized to the
number of trials. It does not run another pose solver or settling simulation.
These measurements predate production integration.

[Recorded measurements](benchmarks/excavator-robustness-clearance-20261012.json)
include current profiles, native binary and input hashes, individual timings,
failure counts, and search-tree hashes. Hardware: RTX 5060 Ti. BLAS/OpenMP: one
thread each; MCTS result workers: four.

### MCTS computation overhead

One baseline/probe pair per state, seed 0, 64 iterations, fresh processes, current
`mcts_cuda.yml`. The probe computes clearance but retains the original noisy poses
for simulation. This isolates added computation without changing the search.
Both pairs produced identical tree hashes and selected stones; neither timed out.

| Starting scene | Evaluated candidates | Baseline | With clearance computation | Change | Time inside clearance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Empty | 204 | 9.815 s | 10.075 s | +2.65% | 0.240 s |
| 14 placed stones | 610 | 109.091 s | 111.996 s | +2.66% | 2.704 s |

These are paired measurements, not a general latency guarantee. An actual search
using corrected trials may follow a different tree and take a different time.

### Candidate replay with correction applied

Saved root candidates from `excavator_plan1.stack`, steps 1, 8, and 15, with their
preceding checkpoint states. One warmup then five measurements per mode, with
alternating order. Four noise trials per candidate. Medians below include noise
generation, scene preparation/upload, broad phase, native queries, lifting, and
revalidation in the clearance column. Baseline is the entire placement batch.

| Placed stones | Candidate batch | Baseline placement | Clearance alone | Clearance / baseline |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1 | 110.07 ms | 0.70 ms | 0.64% |
| 0 | 8 | 196.37 ms | 2.60 ms | 1.32% |
| 0 | 32 | 444.28 ms | 10.91 ms | 2.46% |
| 7 | 1 | 253.34 ms | 1.28 ms | 0.51% |
| 7 | 8 | 795.54 ms | 8.89 ms | 1.12% |
| 7 | 32 | 1596.62 ms | 51.71 ms | 3.24% |
| 14 | 1 | 314.10 ms | 2.31 ms | 0.74% |
| 14 | 8 | 862.21 ms | 14.58 ms | 1.69% |
| 14 | 32 | 1962.10 ms | 71.66 ms | 3.65% |

The initial prototype used a 512-job workspace for four trials and took about
19 ms for clearance. Sizing the cached workspace to the actual batch and skipping
stone-pair detection on empty scenes reduced this to about 0.7 ms. The measured
4/32/128-trial workspaces occupied about 0.50/3.48/14.03 MiB, excluding the model
cache and simulator.

### Initialization limitations

An upward translation does not resolve every lateral or overhanging contact.
The prototype classifies invalid queries, a non-upward clearance gradient, or an
excessive/unresolved lift as initialization failure. Such trials are not simulated
in the corrected replay and remain failures. For 32 candidates, unresolved trial
counts were 0/128, 27/128, and 9/128 in the three scenes. Candidate pass counts
changed from 6 to 5, 10 to 3, and 20 to 14. Faster corrected replay times therefore
cannot be interpreted as a simulation speedup or an improvement in planning.

For step 1 candidate #9, native geometry detected penetration in three of four
noisy poses. The largest correction was 26.68 mm; all four trials still failed
the motion-integral limit after correction. Initial penetration was not its only
failure mechanism.

The measured computation cost was modest. The production refinement above now
rejects unresolved initialization explicitly.

Reproduce candidate replay with
`tests/benchmark_place_robustness_clearance.py --stack PATH --steps 1 8 15 --batches 1 8 32 --repeats 5 --output /tmp/clearance.json`.
For the MCTS cost comparison, run `tests/benchmark_excavator_cuda_mcts.py` with
`--backend cuda --seed 0 --config apps/excavator/config/mcts_cuda.yml --iterations 64`,
once with `--uncorrected-robustness` and once with `--clearance-cost-probe`, using different `--output`
paths. Use the same trusted checkpoint-state pickle with `--state` for both
populated-scene runs. The commands now exercise the production refinement; the
older prototype measurements are retained above for comparison. These commands
require the project Python and GPU access.

## Original CPU/CUDA comparison — 2026-10-02

Run on 2026-10-02 with the current posegen and GPU simulation integration. CPU mode was faster in both tested scenes. This compares the complete deployed search modes, including their different numerical outcomes; it does not measure identical-work kernel speedup.

Raw metrics, input poses, profiles, source hashes, and the installed native-module hash are retained in [the benchmark data](benchmarks/excavator-cuda-mcts.json).

## Method

- Hardware: Ryzen 9 9900X, 12 CPU workers, RTX 5060 Ti.
- Two identical starting scenes per mode: an empty 49-object inventory (effective depth one), and the frozen nine-placement checkpoint from `.stacks/excasvator_cuda_test1.stack` (effective depth three).
- Seeds 0, 1, and 2; 120 iterations and the current 240-second limit. No physics, acceptance, CEM-population, or evaluation-batch settings were changed.
- Each search ran in a fresh Python process, sequentially. CPU/CUDA ordering alternated by seed. BLAS and OpenMP used one thread per worker.
- Timings cover search only, including first-search caches and lazy workers. Inventory loading, initial observation, CUDA workspace construction, metadata hashing, and cleanup are excluded. There was no warmup search.
- CUDA mode uses GPU pose generation and placement simulation, including robustness trials. CPU initialization, scoring, filtering, and observation building remain part of the search time.

## Results

| Scene | CPU median | CUDA median | CUDA wall-time change | CPU / CUDA |
| --- | ---: | ---: | ---: | ---: |
| Empty stack, depth 1 | 5.59 s | 6.75 s | +20.7% | 0.83x |
| Nine placed stones, depth 3 | 115.01 s | 160.53 s | +39.6% | 0.72x |

All twelve runs completed 120 iterations without timing out and selected a candidate marked selectable by the search.

| Scene | Seed | CPU seconds | CUDA seconds | Evaluated CPU / CUDA | Selected object CPU / CUDA |
| --- | ---: | ---: | ---: | ---: | --- |
| empty | 0 | 5.59 | 6.58 | 12 / 12 | 43 / 50 |
| empty | 1 | 5.76 | 6.75 | 12 / 12 | 44 / 44 |
| empty | 2 | 5.55 | 6.79 | 12 / 12 | 43 / 43 |
| placed9 | 0 | 115.01 | 161.65 | 144 / 168 | 48 / 35 |
| placed9 | 1 | 125.54 | 160.53 | 153 / 167 | 50 / 35 |
| placed9 | 2 | 107.39 | 147.16 | 131 / 158 | 13 / 51 |

## Where search time went

The following are medians of each phase measured by MCTS. Simulation includes placement robustness. Phase medians need not sum to total median; other work includes priors, observation construction, and tree bookkeeping.

| Scene | Phase | CPU | CUDA |
| --- | --- | ---: | ---: |
| empty | Pose generation | 5.14 s | 5.83 s |
| empty | Simulation batches | 0.44 s | 0.85 s |
| placed9 | Pose generation | 18.46 s | 35.20 s |
| placed9 | Simulation batches | 85.69 s | 111.16 s |

## Interpretation

The current combined CUDA mode does not accelerate these MCTS workloads. The empty-stack searches all evaluate 12 candidates. The deeper searches follow different trees and evaluate different candidate counts, so their wall-time ratio is a deployment comparison, not a pure simulator speed ratio. Different selected stones also rule out a search-parity claim.

The deeper-case phase measurements identify simulation as the largest cost. GPU robustness currently simulates up to the maximum trial duration and then applies the legacy stopping rule to samples; small evaluation batches and sample handling are useful targets for subsequent profiling. This benchmark does not isolate how much each mechanism contributes.

These are three seeds on two scenes, not a full stacking-run quality or throughput evaluation. Immediate selected rewards and target IoU are retained in the JSON; they do not measure eventual stack quality.

## Reproduction

Use the same source/native binary and platform profiles identified in the JSON. Export the saved checkpoint once with `StackRunStore.load_planning_checkpoint(...).state` and `pickle.dumps`, then run each mode/seed sequentially:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 QT_QPA_PLATFORM=offscreen \
  python tests/benchmark_excavator_cuda_mcts.py \
  --backend cpu --seed 0 --output /tmp/empty-cpu-seed0.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 QT_QPA_PLATFORM=offscreen \
  python tests/benchmark_excavator_cuda_mcts.py \
  --backend cuda --seed 0 --state /tmp/state9.pkl \
  --output /tmp/placed9-cuda-seed0.json
```

Repeat for both backends and seeds 0–2. The script accepts only trusted local state pickles and refuses to overwrite an existing result. The benchmark does not modify the active YAML profiles.
