# Excavator CUDA MCTS batch-size sweep

Run on 2026-10-03 on the Ryzen 9 9900X / RTX 5060 Ti. Evaluation batches of 32, 64, and 128 use matching GPU simulation workspace capacities. Posegen capacity remains 128. Active YAML and production backend defaults were not changed.

Full per-run metrics, actual native chunk sizes, resolved profiles, and source/native hashes are in [the raw data](benchmarks/excavator-cuda-batch-sweep.json).

## Method

- Three seeds (0–2), sequential fresh processes, 120 iterations, 240-second limit, one BLAS/OpenMP thread. Batch order rotates across seeds: 32/64/128, 64/128/32, 128/32/64.
- Search-only timing, without warmup: setup, initial observation, workspace construction, metadata hashing, and cleanup are excluded.
- **Empty stack:** fixed enlarged root workload for every batch size. Root cap and CEM retention 128; generation allowance 128; CEM population 256; per-object cap disabled. These settings provide enough candidates to exercise all three batch sizes. This is a different workload from the earlier 12-root-candidate benchmark.
- **Nine placed stones:** same frozen checkpoint and original root/generation settings as the earlier benchmark. Only evaluation batch size and CUDA simulation capacity change.
- Deeper search still has a 120-iteration budget. Available candidates, progressive widening, and remaining iterations may prevent a 128-job frontier. Actual sizes are reported below.
- Instrumentation wraps the existing backend within the benchmark process. It records native call chunk sizes and host wall time; it does not change solver physics or acceptance rules.

## Median results

| Scene | Batch / GPU capacity | Search | Pose generation | Simulation batches | Median evaluations | Largest MCTS batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| empty128 | 32 | 16.92 s | 8.21 s | 8.40 s | 128 | 32 |
| empty128 | 64 | 15.19 s | 8.19 s | 6.61 s | 128 | 64 |
| empty128 | 128 | 13.81 s | 8.20 s | 5.43 s | 128 | 128 |
| placed9 | 32 | 163.71 s | 30.00 s | 117.58 s | 187 | 32 |
| placed9 | 64 | 139.27 s | 26.80 s | 100.29 s | 160 | 53 |
| placed9 | 128 | 133.00 s | 26.48 s | 94.81 s | 154 | 53 |

18/18 runs completed all 120 iterations; 0 timed out. 18/18 selected a selectable candidate.

## Actual CUDA batches

Chunk statistics pool all three seeds. The simulation backend batches four robustness trials per eligible placement separately from settling and structural simulation. Native run wall time covers execution and immediate return handling; Python trajectory construction, robustness stopping-rule replay, and observation building are additional costs.

| Scene | Capacity | Stage | Mean jobs / native call | Maximum | Median run wall |
| --- | ---: | --- | ---: | ---: | ---: |
| empty128 | 32 | settling | 32.0 | 32 | 0.19 s |
| empty128 | 32 | structural | 32.0 | 32 | 1.04 s |
| empty128 | 32 | robustness | 32.0 | 32 | 4.27 s |
| empty128 | 64 | settling | 64.0 | 64 | 0.11 s |
| empty128 | 64 | structural | 64.0 | 64 | 0.70 s |
| empty128 | 64 | robustness | 64.0 | 64 | 2.89 s |
| empty128 | 128 | settling | 128.0 | 128 | 0.07 s |
| empty128 | 128 | structural | 128.0 | 128 | 0.50 s |
| empty128 | 128 | robustness | 128.0 | 128 | 2.01 s |
| placed9 | 32 | settling | 20.5 | 32 | 1.33 s |
| placed9 | 32 | structural | 20.5 | 32 | 5.77 s |
| placed9 | 32 | robustness | 28.4 | 32 | 7.10 s |
| placed9 | 64 | settling | 20.0 | 53 | 1.05 s |
| placed9 | 64 | structural | 20.0 | 53 | 5.14 s |
| placed9 | 64 | robustness | 44.7 | 64 | 5.73 s |
| placed9 | 128 | settling | 21.3 | 53 | 0.96 s |
| placed9 | 128 | structural | 21.3 | 53 | 4.87 s |
| placed9 | 128 | robustness | 66.2 | 128 | 4.85 s |

## Transition work outside native calls

Transition evaluation includes Python scene preparation, sample access and replay, trajectory construction, validation, reward computation, and observation building. The instrumentation separates their combined residual from native run time; it does not isolate each of these costs.

| Scene | Batch | Median native run time | Median transition time outside native calls |
| --- | ---: | ---: | ---: |
| empty128 | 32 | 5.51 s | 2.75 s |
| empty128 | 64 | 3.71 s | 2.73 s |
| empty128 | 128 | 2.57 s | 2.72 s |
| placed9 | 32 | 14.30 s | 103.33 s |
| placed9 | 64 | 11.67 s | 88.35 s |
| placed9 | 128 | 10.62 s | 84.88 s |

## Per-seed results

| Scene | Batch | Seed | Search seconds | Iterations | Evaluations | Selected object | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| empty128 | 32 | 0 | 17.14 | 120 | 128 | 44 | 2.7495 |
| empty128 | 32 | 1 | 16.80 | 120 | 128 | 2 | 2.3344 |
| empty128 | 32 | 2 | 16.92 | 120 | 128 | 44 | 2.9454 |
| empty128 | 64 | 0 | 15.22 | 120 | 128 | 44 | 2.7495 |
| empty128 | 64 | 1 | 14.86 | 120 | 128 | 2 | 2.3344 |
| empty128 | 64 | 2 | 15.19 | 120 | 128 | 44 | 2.9454 |
| empty128 | 128 | 0 | 13.81 | 120 | 128 | 44 | 2.7495 |
| empty128 | 128 | 1 | 13.77 | 120 | 128 | 2 | 2.3344 |
| empty128 | 128 | 2 | 13.94 | 120 | 128 | 44 | 2.9454 |
| placed9 | 32 | 0 | 163.71 | 120 | 190 | 35 | 3.0396 |
| placed9 | 32 | 1 | 155.70 | 120 | 176 | 35 | 3.3376 |
| placed9 | 32 | 2 | 163.77 | 120 | 187 | 51 | 3.9796 |
| placed9 | 64 | 0 | 141.61 | 120 | 162 | 35 | 3.0396 |
| placed9 | 64 | 1 | 138.39 | 120 | 158 | 35 | 3.3376 |
| placed9 | 64 | 2 | 139.27 | 120 | 160 | 51 | 3.9796 |
| placed9 | 128 | 0 | 133.00 | 120 | 154 | 35 | 3.0396 |
| placed9 | 128 | 1 | 120.71 | 120 | 138 | 35 | 3.3376 |
| placed9 | 128 | 2 | 134.51 | 120 | 155 | 51 | 3.9796 |

## Comparison and limits

In the fixed 128-candidate empty-stack workload, each seed produced exactly the same selected stone, pose, reward, and failure count across batch sizes. Simulation-batch time fell from 8.40 s to 5.43 s (35.3%) while pose generation stayed near 8.2 s.

For **empty128**, the lowest median was batch **128** at **13.81 s**, 18.4% less search time than batch 32.

For **placed9**, the lowest median was batch **128** at **133.00 s**, 18.8% less search time than batch 32.

The earlier same-source, same-native-module nine-placement medians were CPU **115.01 s** and CUDA evaluation batch 12 / capacity 32 **160.53 s**. Those are prior-day reference measurements, not CPU controls rerun during this sweep. See [the earlier comparison](excavator-cuda-mcts-benchmark.md).

Batch size changes how often MCTS incorporates feedback, and may change candidates, failures, and selected stones. End-to-end deeper-search timing is therefore not a fixed-work GPU kernel speed ratio. Immediate rewards do not establish eventual stack quality. Three seeds and two scenes do not establish a universally optimal batch size.

## Reproduction

Use the source/native hashes and profiles in the data file. The checkpoint pickle is the same trusted local input used in the earlier comparison. Repeat these commands sequentially for batch 32/64/128 and seeds 0/1/2, using distinct output files:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 QT_QPA_PLATFORM=offscreen \
  python tests/benchmark_excavator_cuda_mcts.py \
  --backend cuda --seed 0 --evaluation-batch-size 32 --cuda-batch-size 32 \
  --root-candidates 128 --output /tmp/empty128-batch32-seed0.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 QT_QPA_PLATFORM=offscreen \
  python tests/benchmark_excavator_cuda_mcts.py \
  --backend cuda --seed 0 --evaluation-batch-size 32 --cuda-batch-size 32 \
  --state /tmp/excavator-cuda-mcts-benchmark-20261002/state9.pkl \
  --output /tmp/placed9-batch32-seed0.json
```
