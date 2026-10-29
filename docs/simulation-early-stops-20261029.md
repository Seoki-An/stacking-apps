# Simulation early-stop evaluation — 2026-10-29

The CUDA robustness shortcut is enabled. The structural quiet-window shortcut
is implemented for CPU and CUDA but remains **off**: it did not trigger on the
measured workload. No YAML thresholds, settling policy, disturbance samples,
MCTS selection rules, or short/long simulation strategy were changed.

The robustness shortcut measurably saves time in affected candidate batches.
These measurements **do not demonstrate a meaningful end-to-end MCTS speedup**.

## Implementation

1. **Peak excursion:** a CUDA trial stops once any monitored dynamic stone
   exceeds the existing `fail_displacement` limit, using translation plus
   `rotation_weight * quaternion_angle` from the original trial pose. This
   includes movable supporting stones, not just the candidate. Original poses
   are retained in FP64; a conservative numerical guard defers ambiguous
   threshold crossings to the existing host classifier. Zero limits are valid;
   an absent rule disables stopping. The host still classifies the retained
   history. Settling and nominal structural simulation do not use this rule.
2. **Impossible quiet window:** after each final structural timestep, count the
   maximum possible additional quiet steps before the existing time limit.
   Stop only if `current_quiet + possible_remaining_quiet < required_quiet`.
   Equality continues. CUDA accounts for minimum-time gating, continued-phase
   elapsed steps, and the absolute step cap. CPU reproduces its existing
   floating-point clock, including the last step strictly past `max_time`.
   Intermediate phases and bounded-rocking robustness must not use this rule.
3. **Partial results:** native `excursion_limit` and `quiet_impossible` statuses
   preserve valid partial snapshots and the final sample. Failed trials have
   shorter histories and can have different failure metrics/reasons. Accepted
   results must remain unchanged. MCTS uses its configured failure reward for
   rejected nodes, so changed failed-trial metrics do not change their backup.

The application controls are `CudaPlacementBackend.early_excursion_stop=True`
and `early_quiet_stop=False`; the CPU backend's quiet shortcut also defaults to
false. These are implementation/benchmark controls, not additional profile
options. Older native builds fall back to full robustness simulation.

## Saved-batch comparison

Source: `.stacks/simple_mcts_test7.stack`, steps 10, 20, 30, and 40. Replays use
saved candidate actions, parent checkpoints, inventory target, resolved physics,
and robustness settings. The file is read-only. Each batch was warmed and
replayed three times per mode, reversing mode order on alternating repeats.
All modes used the same rebuilt native binary.

| Step | Candidates | Baseline evaluation | Excursion shortcut | Early-stopped trials |
| --- | ---: | ---: | ---: | ---: |
| 10 | 32 | 0.445 s | 0.446 s | 0 |
| 20 | 37 | 2.096 s | 1.905 s | 3 |
| 30 | 33 | 5.546 s | 5.435 s | 0 |
| 40 | 33 | 6.431 s | 6.084 s | 1 |

Times are medians for the entire placement batch, including settling, nominal
simulation, robustness, and Python result processing. Step 20 improved by 9.1%
and step 40 by 5.4%. Neither step 10 nor step 30 skipped any simulation steps;
their timing differences are not evidence of algorithmic savings.

All 135 candidate decisions matched in every repeat (121 accepted, 14 rejected).
Accepted poses, nominal trajectory samples, and robustness values were exact.
An additional step-20/40 comparison checked contacts, trajectory times, energies,
and integrals against the original installed native binary: those fingerprints
also matched exactly. Quiet-window stopping triggered zero times.

Step 20's robustness work fell from 34,304 to 33,484 integrated job-steps;
step 40 fell from 30,464 to 30,135. A small number of long failing trials can
hold an entire GPU batch open, explaining the wall-time benefit despite a small
reduction in total integrated job-steps. History buffers still reserve their
original maximum capacity; this change does not reduce peak memory allocation.

## Fixed-iteration MCTS comparison

Reconstructed parent state: saved checkpoint 39. Current excavator CUDA profile,
seed 0, 128 iterations, original 240-second safety limit. These are fresh trees
from the same saved scene, not preserved trees from the original run.

| Variant | Search wall | Robustness native calls | Quiet-window stops |
| --- | ---: | ---: | ---: |
| Baseline | 65.31 s | 7.17 s | 0 |
| Excursion only | 65.17 s | 6.80 s | 0 |
| Quiet-window only | 66.63 s | 7.33 s | 0 |
| Both | 66.94 s | 6.96 s | 0 |

All variants completed 128 iterations, generated/evaluated 177 candidates, and
selected object 57 with reward `0.21631834978577968`. Their ordered tree decision
fingerprint was identical, including actions, accepted states/rewards, visits,
and backed-up values. Failed raw states/rewards/reasons are excluded from that
fingerprint because early termination intentionally changes their diagnostics.
The complete raw-tree fingerprint consequently differs.

A reversed-order repeat measured excursion at 67.57 s and baseline at 67.17 s.
Across the two measurements, average search time was 66.37 s with the shortcut
versus 66.24 s without it: about 0.2% slower, within observed timing variation.
Average native robustness time decreased from 7.24 s to 6.86 s (5.2%). Two trials
stopped early per search; robustness job-steps decreased from 89,918 to 89,106.
Nominal structural job-steps remained exactly 32,210.

Thus the excursion shortcut saves robustness work, but pose generation and
nominal structural simulation dominate this case. Keep the quiet-window option
off; accelerating the dominant nominal contact solve is a separate next task.
These results cover selected saved batches and one fixed-search scene/seed,
not a full 49-stone planning performance guarantee. Wall-time-budgeted searches
may explore more nodes when evaluation becomes faster and need not select the
same action.

## Validation and reproducibility

- 124 targeted Python tests passed, including CPU/GPU stopping boundaries,
  neighbor motion, rotation, mixed batches, continuation at the exact deadline,
  accepted-history parity, invalid configurations, robustness, and MCTS.
- Three native CUDA simulation CTest targets passed.
- The test build's SHA-256 was
  `e8efc8c56a15ca92ed2c9e806dea28d79427c1d03f49c7a094e6f23280b4ea8c`.
- During final verification another build replaced the installed binary. The
  install guard detected this and did not overwrite it. That binary already
  includes the new API; its targeted Python tests also passed (124 tests).
  Step-20/40 accepted contacts, trajectories, and robustness fingerprints also
  matched the original binary on that installed build.
- Native source backups: `/tmp/simulation-early-stops-backup`.
- Detailed measurements: `/tmp/simulation-early-stops/results.json`,
  `mcts40-*.json`, `installed-contacts.json`, `build-cuda-contacts.json`, and
  `current-installed.json` in the same directory. The controlled timing tables
  above refer to the fixed test build, not to the concurrent replacement.

Replay the candidate benchmark with the installed CUDA build:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python tests/benchmark_simulation_early_stops.py \
  --stack .stacks/simple_mcts_test7.stack --steps 10 20 30 40 \
  --repeats 3 --output /tmp/simulation-early-stops/replay.json
```

For fixed-iteration MCTS, `tests/benchmark_excavator_cuda_mcts.py` accepts
`--early-stops none|excursion|quiet|both`; it records full and accepted-tree
fingerprints, per-stage native timings, integrated steps, and termination
statuses. Use the same state, seed, configuration, and native binary for each
comparison, and verify `timed_out=False` before interpreting tree parity.
