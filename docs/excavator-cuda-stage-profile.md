# Excavator CUDA MCTS stage profile

**Repeated state construction and trajectory copying are the dominant measured cost.** The detailed search spent 84.49 s (63.2% of 133.76 s) inside state constructors, including 80.78 s copying trajectory histories. Native simulation calls took 9.94 s. Scene construction, robustness replay, and observation construction were much smaller costs.

## Workload and checks

- Same frozen nine-placement checkpoint, seed 0, 120 iterations, 240-second limit, evaluation batch 128, and GPU simulation capacity 128 as the [batch sweep](excavator-cuda-batch-sweep.md). Posegen capacity remains 128.
- Timer-only coarse run: 133.45 s; detailed confirmation: 133.76 s; prior benchmark: 133.00 s. Setup, checkpoint loading, initial observation, and teardown are excluded.
- Both profiling runs completed all iterations without timeout, evaluated 154 candidates, and matched the prior selected action, pose, reward, source hashes, native binary, and resolved profiles.
- Timers use host wall time. Native-call times include immediate return handling; they are not GPU kernel timings. Inclusive measurements overlap; exclusive times subtract instrumented child calls and partition the search total.
- No production code or active YAML changed. One seed was profiled; these timings are workload-specific.

## Detailed attribution

| Operation | Calls | Inclusive wall time |
| --- | ---: | ---: |
| Complete search | 1 | 133.761 s |
| State construction, including history copies | 1,236 | 84.487 s |
| Trajectory copying (inside state construction) | 79,876 | 80.781 s |
| Candidate generation and expansion priors | 7 | 27.241 s |
| Native simulation calls | 21 | 9.936 s |
| Reward computation, including overlap | 154 | 4.895 s |
| Observation construction | 156 | 4.022 s |
| Structural sample processing | 154 | 0.752 s |
| Robustness stopping-rule replay | 616 | 0.429 s |
| Scene building | 770 | 0.045 s |
| Scene rebuilding from settled snapshots | 154 | 0.010 s |

State construction includes trajectory copying, so those rows must not be added. The raw data also contains exclusive times for every instrumented method.

## Why histories are copied so often

[`SimulationTransition.step_batch`](../packages/stacking-common/src/stacking_common/environment/transition.py) restores each parent state once to validate the action and again before applying the result. Each restore copies the state and then passes another snapshot copy to the backend. For CUDA, that backend restore only checks inventory IDs.

Applying a result constructs a new `State`, then returns a snapshot copy. `TransitionEvaluation` and MCTS node evaluation each make another ownership copy. In this workload that produces approximately eight full state constructions per evaluated candidate: 1,236 state constructions for 154 evaluations, including a few initialization constructions. There were 1,080 explicit `State.copy()` calls and 79,876 trajectory-copy calls.

[`State.__post_init__`](../packages/stacking-common/src/stacking_common/domain/state.py) copies every trajectory in the entire history. [`StackObjectTrajectory.copy`](../packages/stacking-common/src/stacking_common/domain/trajectory.py) first copies every pose array, then constructs a new trajectory. Its constructor calls `pose_array` again for every pose, checking finiteness and quaternion norm and copying the array a second time.

## Focused copy profile

An isolated microprofile copied the frozen starting state ten times after loading it. That state contains **45 trajectories and 19,395 historical poses**. Without cProfile, ten copies took **0.553 s**, or **55.3 ms per copy**. With cProfile, the same loop took **1.130 s**; use those instrumented times for attribution rather than throughput.

`pose_array` was called **194,310 times** and accumulated **0.953 s** in the instrumented loop. Much of this is repeated finiteness checks and quaternion norms on previously validated data. This identifies validation of copied history as a larger cost than just copying raw arrays.

## Recommended optimization order

1. Give validated trajectories a copy path that copies their mutable arrays once and preserves branch independence, without rerunning constructor validation on every historical pose.
2. Remove redundant state restores and snapshots from CUDA batch preparation. The CUDA backend operates on explicit input states and does not need a copied full trajectory history to check inventory IDs.
3. Review state ownership transfers into `TransitionEvaluation` and MCTS nodes to avoid repeated full-history copies. Keep isolation tests; simply sharing mutable histories would change search correctness.
4. Rebenchmark before further CUDA kernel or robustness-stop tuning. Those remain possible optimizations, but the current evidence gives state/history copying priority.

This also refines the earlier hypothesis: robustness replay took only 0.43 s and observations 4.02 s here. They do not explain the roughly 85 s previously observed outside native simulation calls. CPU workflows also copy states, so these optimizations should be benchmarked in both modes rather than assuming the entire gain belongs to CUDA.

## Reproduction

Use the source/native hashes and checkpoint identified in [the raw profiling data](benchmarks/excavator-cuda-stage-profile.json). The profiler forwards benchmark arguments to the existing harness:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 QT_QPA_PLATFORM=offscreen \
  python tests/profile_excavator_cuda_mcts.py \
  --profile-output /tmp/cuda-stages.json \
  --backend cuda --seed 0 --evaluation-batch-size 128 --cuda-batch-size 128 \
  --state /tmp/excavator-cuda-mcts-benchmark-20261002/state9.pkl \
  --output /tmp/cuda-search.json
```

Optional `--cprofile-stage LABEL` saves a sibling `.prof` file for an instrumented stage, such as `robustness_replay`; that diagnostic mode adds overhead. The reported full-search times above use timer-only instrumentation.

To repeat the copy microprofile, load the same trusted state pickle, time ten `state.copy()` calls without cProfile, then profile ten more with `cProfile.Profile`. Loading and initial state construction must remain outside the timed region. Function-level microprofile statistics are retained in the JSON.
