# Excavator CPU versus combined CUDA MCTS benchmark

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
