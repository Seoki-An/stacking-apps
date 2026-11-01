# Excavator MCTS CPU and CUDA profile comparison

On 2026-10-07, the current CUDA profile generated new tree nodes faster than the CPU profile on all four recorded scenes. Its searches took longer overall and selected a valid placement in all four cases; CPU selected placements at 10 and 25 stones. These are single-trial measurements of the complete profiles, not an isolated CPU/GPU backend comparison.

## Workload and measurement

- Sources: `.stacks/excavator_plan10_gpu.stack`, committed checkpoints 10, 15, and 20; `.stacks/excavator_plan9_gpu.stack`, checkpoint 25. The former run ended before 25 stones. Each checkpoint was verified to contain exactly the requested number of placed stones; all retain the same 49-object inventory. The 25-stone case is a separate recorded scene, not a continuation of the other source run.
- Each trial searches one next placement: planning step 11, 16, 21, or 26. Both profiles receive the identical exported state for that scene and start a fresh tree.
- Seed 0, one trial per profile and scene, isolated processes run sequentially. CPU runs before CUDA for each scene. There was no explicit warmup; lazy initialization within search is included.
- CPU: AMD Ryzen 9 9900X, 12 cores / 24 threads. GPU: NVIDIA GeForce RTX 5060 Ti, 16 GiB. BLAS/OpenMP thread counts are limited to one per process to avoid worker oversubscription.
- Time covers `search.search`: generation, simulation, backup, validation, repair, root retries and floor-fill fallback. Loading, environment setup and output serialization are excluded. Motion planning is not run.
- **Expanded nodes/s** here means **newly generated child nodes / search wall seconds**. Physically failed placements count because they consume expansion work. Proposals rejected before node creation do not count. Revalidation of an existing node does not add another expanded node. Expansion calls and transition evaluations are reported separately below.
- Both profiles completed 1,024 iterations in every trial. No trial timed out. Setup took about 1.56–1.59 s for CPU and 0.71–0.75 s for CUDA.

## Search throughput

| Placed stones | CPU search s | CPU new nodes | CPU nodes/s | CUDA search s | CUDA new nodes | CUDA nodes/s | CUDA / CPU throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 40.21 | 176 | 4.38 | 84.18 | 629 | 7.47 | 1.71× |
| 15 | 69.15 | 119 | 1.72 | 92.53 | 351 | 3.79 | 2.20× |
| 20 | 78.40 | 102 | 1.30 | 112.01 | 410 | 3.66 | 2.81× |
| 25 | 76.48 | 89 | 1.16 | 100.47 | 304 | 3.03 | 2.60× |

Final retained tree size is not the expansion count: retries and validation can remove previously generated subtrees. The reported count includes that work.

## Validation and selected placements

| Stones | Profile | Transition evaluations | Evaluations/s | Expansion calls/s | Selected object | Reward | Target IoU |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 10 | cpu | 186 | 4.63 | 2.34 | 50 | 4.081 | 0.1548 |
| 10 | cuda | 700 | 8.32 | 3.59 | 47 | 4.807 | 0.1576 |
| 15 | cpu | 119 | 1.72 | 1.16 | None | — | — |
| 15 | cuda | 360 | 3.89 | 1.46 | 9 | 1.667 | 0.2269 |
| 20 | cpu | 102 | 1.30 | 1.57 | None | — | — |
| 20 | cuda | 417 | 3.72 | 2.07 | 5 | 2.381 | 0.2959 |
| 25 | cpu | 90 | 1.18 | 0.39 | 49 | 3.060 | 0.3300 |
| 25 | cuda | 317 | 3.16 | 1.02 | 45 | 3.309 | 0.3295 |

All trials used three root retry rounds and attempted floor-fill fallback. Fallback selected a placement for CPU at 10 and 25 stones and CUDA at all four sizes. CPU at 15 and 20 stones exhausted candidate attempts without a selectable placement; those results are not timeouts. Expansion calls count generation invocations, including unsuccessful ones, and are not unique expanded parents.

## Search phase times

| Stones | Profile | Pose generation s | Short simulation batch wall s | Long simulation batch wall s |
| --- | --- | ---: | ---: | ---: |
| 10 | cpu | 12.81 | 8.94 | 18.09 |
| 10 | cuda | 45.43 | 21.47 | 15.85 |
| 15 | cpu | 27.69 | 5.39 | 35.56 |
| 15 | cuda | 52.14 | 13.93 | 24.97 |
| 20 | cpu | 32.26 | 2.92 | 42.43 |
| 20 | cuda | 60.43 | 18.83 | 30.22 |
| 25 | cpu | 32.94 | 2.05 | 40.59 |
| 25 | cuda | 47.78 | 21.86 | 28.95 |

These are coordinator phase wall times, not GPU kernel-only timings. They need not sum exactly to total search time.

## Configuration differences

| Control | CPU | CUDA |
| --- | ---: | ---: |
| Iterations | 1024 | 1024 |
| Maximum depth | 3 | 4 |
| Root children | 12 | 32 |
| Logical evaluation batch | 12 | 256 |
| CPU workers | 12 | 4 result workers |
| Required descendant depth | 1 | 2 |
| Validation reserve s | 30 | 60 |
| CEM elite | 24 | 32 |
| CEM retained | 12 | 32 |
| CEM mutation XY standard deviation m | 0.1 | 0.2 |

Both profiles retain their configured 240 s overall limit, 16 non-root children, progressive widening and mixed final ranking. Physical and action-generation profiles are the current repository versions. The checkpoint supplies the state; it does not restore historical solver settings.

The measured throughput gains therefore reflect both resource use and different search policies. Additional seeds/repeats are needed to estimate variability; these eight trials do not establish a general speedup or planning-success rate.

## Reproduction and artifacts

```bash
python tests/benchmark_excavator_mcts_profiles.py \
  --run .stacks/excavator_plan10_gpu.stack \
  --placed 10 15 20 --seed 0 \
  --output .stacks/benchmarks/excavator-profiles-repeat
```

For the additional 25-stone case:

```bash
python tests/benchmark_excavator_mcts_profiles.py \
  --run .stacks/excavator_plan9_gpu.stack \
  --placed 25 --seed 0 \
  --output .stacks/benchmarks/excavator-profiles-25-repeat
```

Use a new output directory; existing results are never overwritten. Add `--repeats 3` for repeated timings at the same seed, with backend order alternated between repetitions. Use separate runs with different `--seed` values for search variability.

- [Portable results and provenance](benchmarks/excavator-mcts-profile-comparison-20261007.json) include resolved MCTS profiles, state hashes, source hashes, native module hash and trial metrics.
- Local raw results, logs, exported states and exact MCTS YAML snapshots are in `.stacks/benchmarks/excavator-profiles-20261007/` and `.stacks/benchmarks/excavator-profiles-25-20261007/`. Each full trial JSON additionally records all resolved physical profiles and detailed CUDA calls.
- [Comparison runner](../tests/benchmark_excavator_mcts_profiles.py) reuses the existing isolated search benchmark.

All eight trials were checked for matching per-scene state hashes, identical source hashes and the same native module. The current working tree contains local changes, so the recorded commit IDs alone do not reproduce the source; the hashes identify what was measured.
