# Excavator MCTS history-copy optimization

Trajectory branch copies now duplicate pose arrays once and retain already validated values. Batch transitions validate actions against their supplied parent states, then apply results directly to those parents without restoring and snapshotting each parent twice. Returned states still own independent mutable histories. Active YAML and CUDA capacity defaults are unchanged.

## Validation

- 149 relevant tests passed, including new timed/untimed trajectory ownership and batch parent/sibling/output isolation tests, batch input validation, native CUDA placement, and CPU/MCTS regression coverage.
- Nine fresh-process searches: CPU batch 12 / 12 workers, CUDA batch 12 / capacity 32, and CUDA batch 128 / capacity 128; seeds 0–2; 120 iterations; 240-second limit; the same frozen nine-placement scene.
- All runs completed 120 iterations without timeout. Every mode/seed matched its recorded pre-optimization candidate counts, failures, selected stone, selected pose, reward, and target IoU exactly.
- Search-only timings exclude setup and teardown. Runs were sequential with rotating order, one BLAS/OpenMP thread, and no warmup search. Reference times are from the previous benchmark runs; source hash comparison confirms only the two optimized production files changed and the native binary stayed identical.

## Median search time

| Mode | Before | After | Speedup | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| cpu12 | 115.01 s | 39.88 s | 2.88x | 65.3% |
| cuda12 | 160.53 s | 71.37 s | 2.25x | 55.5% |
| cuda128 | 133.00 s | 51.69 s | 2.57x | 61.1% |

Each before/after pair performs the same search work. CPU and CUDA modes can still explore different trees, so their cross-mode ratio is not an identical-work kernel comparison.

## Confirming the removed bottleneck

The additional timer-only CUDA-128 seed-0 profile uses the same instrumentation as the [original stage profile](excavator-cuda-stage-profile.md). Inclusive times overlap: trajectory copies are included in state construction.

| Measured stage | Before | After | Calls before / after |
| --- | ---: | ---: | ---: |
| search | 133.76 s | 51.85 s | 1 / 1 |
| state_constructor | 84.49 s | 3.82 s | 1,236 / 620 |
| trajectory_copy | 80.78 s | 1.86 s | 79,876 / 43,440 |
| state_copy | 73.64 s | 2.83 s | 1,080 / 464 |
| native_simulation_run | 9.94 s | 9.94 s | 21 / 21 |
| observation | 4.02 s | 3.91 s | 156 / 156 |
| robustness_replay | 0.43 s | 0.43 s | 616 / 616 |

## Implementation

- [`StackObjectTrajectory.copy`](../packages/stacking-common/src/stacking_common/domain/trajectory.py) shallow-copies scalar metadata, then independently copies each pose array and the optional timestamps list. Construction and `add_pose` still validate incoming data; copying an existing trajectory does not rerun those checks on historical poses.
- [`SimulationTransition.step_batch`](../packages/stacking-common/src/stacking_common/environment/transition.py) validates inventory and action compatibility before backend dispatch. It passes explicit parent states into result application. Single-step backend restore behavior is unchanged, and the transition retains its own independent final snapshot.
- No mutable trajectory arrays are shared between parents, siblings, returned states, or the transition snapshot. This change does not introduce immutable history sharing or remove the ownership copies made by MCTS node/evaluation constructors.

## Per-seed search times

| Mode | Seed | Before | After |
| --- | ---: | ---: | ---: |
| cpu12 | 0 | 115.01 s | 39.88 s |
| cpu12 | 1 | 125.54 s | 45.36 s |
| cpu12 | 2 | 107.39 s | 39.26 s |
| cuda12 | 0 | 161.65 s | 71.37 s |
| cuda12 | 1 | 160.53 s | 72.69 s |
| cuda12 | 2 | 147.16 s | 64.27 s |
| cuda128 | 0 | 133.00 s | 51.69 s |
| cuda128 | 1 | 120.71 s | 47.66 s |
| cuda128 | 2 | 134.51 s | 52.95 s |

## Reproduction

Full profiles, source/native hashes, input checkpoint details, search results, and stage timers are in [the raw data](benchmarks/excavator-copy-optimization.json). Use the project environment and repeat sequentially for seeds 0–2:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 QT_QPA_PLATFORM=offscreen \
  python tests/benchmark_excavator_cuda_mcts.py \
  --backend cuda --seed 0 --evaluation-batch-size 128 --cuda-batch-size 128 \
  --state /tmp/excavator-cuda-mcts-benchmark-20261002/state9.pkl \
  --output /tmp/optimized-cuda128-seed0.json
```

For the current CUDA configuration use evaluation batch 12 and CUDA capacity 32. For CPU use `--backend cpu --evaluation-batch-size 12` and omit `--cuda-batch-size`. Use a fresh output path for every run. The benchmark overrides do not modify active YAML.

## Cross-module impact review

A follow-up review on 2026-10-03 checked shared state ownership, CPU/CUDA transitions, tabletop and excavator planning, checkpoint persistence/resume, replay, and visualization. No regression attributable to the optimization was found in these checks.

The complete repository suite under Xvfb finished with **1,155 passed, 9 failed, and 19 setup errors**. All 28 unsuccessful cases were rerun with the pre-optimization trajectory-copy and batch-restore behavior reinstated within a separate Python process; the same cases still failed. The suite is therefore not fully green, but these failures persist independently of the copy optimization:

| Area | Unsuccessful cases | Observed cause |
| --- | ---: | --- |
| Excavator closed-loop execution tests | 20 | Recorded replay asset hash mismatch (`object/1`) |
| Excavator PCD emulation | 3 | Legacy gripper-transform fixture mismatch |
| Excavator scan orientation | 1 | Legacy score-map fixture mismatch |
| Playback rendering mocks | 3 | Missing `observed_scene_points` on test `SimpleNamespace` |
| Tabletop simulated pick scene | 1 | Pose solver did not report convergence |

Five new checkpoint regression tests passed: source/checkpoint/decoded history isolation, and rejection of nonfinite poses, zero quaternions, decreasing timestamps, and negative trajectory IDs during decoding. No checkpoint schema or saved-run format changed. CPU single-step restoration remains unchanged; the only production backend currently using `place_batch` is CUDA.

The deliberate contract difference is that `StackObjectTrajectory.copy()` trusts existing values. Unlike the former constructor-based copy, it does not detect invalid values manually written into an existing trajectory's public arrays or metadata. Constructors, `add_pose`, and checkpoint decoding still validate incoming data. The audited production writers use those paths; the CPU settling code's direct pose/time assignments retain an already validated final pose and assign its time origin. No audited caller relies on copying to validate manually corrupted trajectory data.

The initial offscreen-Qt suite exited on an X `BadWindow` error. The affected GUI test passes with `QT_QPA_PLATFORM=xcb` under `xvfb-run`, which was used for the complete suite and baseline failure comparison. Physical hardware execution was not exercised.

Failure identities/messages, source hashes, the test command, and the process-local baseline restoration script are retained in [the review data](benchmarks/excavator-copy-impact-review.json).
