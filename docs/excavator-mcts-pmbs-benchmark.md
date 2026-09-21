# Excavator MCTS: PMBS scheduler benchmark

Measured on 2026-09-20 with native pose generation and physics, using the
excavator search-only configuration. Results compare the new PMBS-style
scheduler with the **already parallel** scheduler at
`bb49465cbea3a1932d14acbdb3f2ddfa0539f42f`. They do not measure speedup over
single-worker MCTS or the historical `stacking-planner` application.

The [raw results](benchmarks/excavator-mcts-pmbs.json) include all 12 measured
trials, preparation statistics, configuration/source hashes, input-state hashes,
and computed summaries. The [benchmark harness](../tests/benchmark_excavator_mcts.py)
runs each trial without hardware execution.

## Results: 40 completed iterations

At placement 11, PMBS reduced mean search time from **257.6 to 163.7 seconds**:
**1.57× faster, or 36.4% less time**. The aggregate is the ratio of mean times,
not the arithmetic mean of individual speedup ratios.

| Seed | Previous (s) | PMBS (s) | Speedup | Time reduction |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 207.554 | 165.747 | 1.25× | 20.1% |
| 1 | 322.254 | 161.841 | 1.99× | 49.8% |
| 2 | 242.907 | 163.628 | 1.48× | 32.6% |
| Mean | 257.572 | 163.738 | 1.57× | 36.4% |

All these runs completed 40 iterations without timing out. The previous
scheduler's seeds 1 and 2 use the extended 600-second ceiling; every other row
uses 240 seconds. Their initial capped runs completed only 21 iterations in
215.335 seconds and 35 iterations in 222.560 seconds, respectively. Those
incomplete runs are retained in the raw data but excluded from this table.

The gain comes primarily from fewer physics batches and evaluations:

| Seed | Short batches, previous → PMBS | Short physics jobs | Total evaluated candidates | Short physics wall time (s) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 6 → 4 | 72 → 48 | 84 → 60 | 108.1 → 71.6 |
| 1 | 14 → 5 | 141 → 46 | 153 → 58 | 209.8 → 69.3 |
| 2 | 8 → 4 | 95 → 48 | 107 → 60 | 140.6 → 71.0 |

Pose generation took 66.7–70.8 seconds before and 64.5–65.7 seconds with PMBS.
Long root-validation physics took 17.2–18.5 seconds in both versions. Thus the
measured gain is not an equivalent acceleration of individual native solves.

Each paired seed selected exactly the same object and pose. PMBS reached tree
depth three in all three runs; the previous scheduler reached depths two,
three, and two. Selected rewards and simulated IoU are close but not always
identical; their exact values are retained in the raw results.

## Actual default request: 200 iterations, 240 seconds

For seed zero on the same ten-object stack, both versions stopped at the
search-phase time limit before completing the requested 200 iterations:

| Metric | Previous | PMBS |
| --- | ---: | ---: |
| Elapsed search time | 228.278 s | 210.788 s |
| Completed iterations | 49 | 67 |
| Total evaluated candidates | 96 | 84 |
| Short physics batches | 7 | 6 |
| Short physics jobs | 84 | 72 |

PMBS completed **36.7% more iterations**, with **1.48× the completed-iteration
rate**, and returned 7.7% sooner. The elapsed-time difference partly reflects
different native batch overruns past the approximately 210-second iteration
deadline. This is not a fixed-work speedup: neither search completed 200
iterations, and PMBS evaluated fewer physical candidates. Both selected exactly
the same object and pose with the same selected reward.

With the default request, the practical effect in this scene is more completed
selection/backup iterations within a similar wall-clock budget. The 36.4%
time reduction above applies specifically to the completed 40-iteration trials.

## Depth-one control

On the empty-stack case, both schedulers completed the default 200 iterations:
**59.528 seconds before versus 59.748 seconds with PMBS** (0.996×). They selected
exactly the same action. There is no meaningful acceleration here, consistent
with the PMBS descendant-batching path being disabled during the first ten
placements, when the configured depth is one. This control measures placement
one; it does not time all ten placements under both implementations.

## Method

- AMD Ryzen 9 9900X, 12 cores / 24 logical CPUs; 12 native search workers.
- Python 3.12.3; clean `stacking-core` revision
  `f4b1593bfc5c4e5a3c5f55ecfde1abe756aa6dcf`.
- Full excavator catalog of 49 objects, with identity initial pickup poses as
  in search-only operation without a pickup file. No robot motion, sensors,
  perception, or execution time is included.
- Each timed trial runs in a fresh process on exactly the same saved state,
  with a fresh search tree. Application construction and state restoration are
  excluded; root CEM, pose generation, physics, validation, selection, and
  backups inside `engine.search()` are included.
- The previous scheduler first constructs a common, natively validated stack
  of ten objects. This preparation is excluded from the measurements. The
  depth-three case is placement 11, with 39 objects remaining.
- Default excavator settings are retained: depth one for the first ten
  placements, then maximum depth three; evaluation batches of 12; root CEM
  population 96 for three rounds; root validation and root lookahead enabled.
  The rollout/bootstrap behavior is unchanged.
- Three paired seeds (0, 1, 2) request 40 iterations for the depth-three
  comparison. Trials run sequentially, alternating scheduler order between
  seeds. Trials that cannot finish within the initial 240-second ceiling are
  repeated with a 600-second ceiling to measure 40 completed iterations.
- A separate seed-zero pair uses the actual default request of 200 iterations
  and 240 seconds. The configured 30-second validation reserve stops admitting
  ordinary iterations near 210 seconds; an active native batch can overrun
  that boundary. Consequently, a timed-out result may take less than 240 seconds.

Equal completed iterations do **not** imply equal physics work: a selected
node can already be evaluated, and spare workers can evaluate siblings without
each sibling counting as an iteration. Timings therefore measure the scheduler
and resulting search work together, not isolated parallel efficiency.

## Reproduction

Use the project Python with `stacking_core` and the excavator model assets
available. Generate the saved states once, then run each trial in a separate
process without concurrent benchmarks:

```sh
python tests/benchmark_excavator_mcts.py \
  --output /tmp/excavator-mcts-repeat \
  --baseline-ref bb49465cbea3a1932d14acbdb3f2ddfa0539f42f \
  --prepare --placements 10

python tests/benchmark_excavator_mcts.py \
  --output /tmp/excavator-mcts-repeat \
  --baseline-ref bb49465cbea3a1932d14acbdb3f2ddfa0539f42f \
  --case 10 --seed 0 --mode previous --iterations 40 --seconds 240

python tests/benchmark_excavator_mcts.py \
  --output /tmp/excavator-mcts-repeat \
  --case 10 --seed 0 --mode pmbs --iterations 40 --seconds 240
```

Repeat for seeds 1 and 2, reversing scheduler order for seed 1. Repeat incomplete
40-iteration trials with `--seconds 600`. Use `--case 0 --iterations 200` for the
depth-one control and `--case 10 --iterations 200` for the default-budget case.
The harness refuses to overwrite results. Its state pickles must be trusted,
locally generated files.

These are measurements on one prepared stack, with one trial per scheduler
and seed, not confidence intervals or a full stacking-sequence evaluation.
Fresh-tree timing also excludes potential effects of tree reuse in a live
session. Matching selected actions is a useful check, but does not establish
equivalent search quality across other scenes.
