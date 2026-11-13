# CPU/GPU MCTS throughput comparison

Prepared for the updated `../stacking-paper/root.pdf` read on 2026-11-11.
Section V evaluates independent candidates through sampling, pose optimization,
and physical simulation in batches. This experiment measures that complete
search pipeline on CPU versus CUDA. No benchmark measurements have been made
as part of this preparation.

## Protocol

- Platform: excavator; named target `default`, explicitly selected on both
  backends. At preparation time it is 2.0 × 6.0 × 2.5 m, with 5-degree taper
  on all sides. The manifest stores the resolved geometry, not just its name.
- Both backends use the same copied MCTS profile, inventory, initial state,
  seed, reward, posegen and simulation settings. The base is the current
  `apps/excavator/config/mcts_cpu.yml`.
- Search has **no wall-clock time budget** (`tree.max_search_time: null`).
  The default request is 4,096 iterations per trial. Search can still finish
  early when the finite tree/proposal pools are exhausted; actual completed
  iterations are retained. Simulation durations and physical rejection limits
  remain unchanged: they define candidate evaluation, not planning time budgets.
- Default common generation/evaluation batch cap: **32**. Optional paired
  batch sweeps use `--batch-sizes 12 32 128`. Root CEM population, retained roots,
  branching limits and search depth remain fixed across batch sizes.
- CUDA posegen chunks and streamed placement chunks are capped at the same B.
  Native simulation capacity is `max(B, min(512, B * n_noise))`, since one
  placement can require several disturbance trials. This capacity counts
  simulation jobs, not MCTS nodes. Root CEM can submit a population larger than
  B; its population is identical across backends, and CUDA chunks that work.
- CUDA disturbance trials enable excursion early stopping, matching the CPU
  backend's stop when a trial exceeds the physical displacement limit. The
  experimental additional CUDA quiet-stop shortcut remains disabled.
- CPU uses 12 native workers; CUDA uses the GPU and 2 CPU result workers.
  These are the existing platform worker allocations. They are adjustable when
  preparing a protocol. Matching logical batches does not imply equal hardware
  concurrency. Both backends run with BLAS/OpenMP/MKL threads set to one.
- Seeds default to 0, 1, 2, with backend order alternating by seed. Each trial
  runs sequentially in a fresh process and fresh tree. No warmup search is run.
  Setup time is recorded separately; first-search lazy initialization is timed.

The primary rate is **`generated_candidates / elapsed_seconds`**: new tree
children, including candidates that later fail physics. Repeated evaluations
and proposals rejected before becoming tree children do not count as new nodes.
Elapsed time covers `search.search()`, including CEM, sampling, optimization,
simulation, CPU scoring, selection, backup and retries; it excludes application
setup, state loading and result serialization.

For each scene/batch/backend, `summary.json` reports the arithmetic mean and
sample standard deviation of trial rates. It also reports the pooled rate
`sum(nodes) / sum(seconds)` separately. GPU/CPU speedup is the ratio of the two
mean rates. Actual evaluation batch sizes, generated/evaluated/failed counts,
completed iterations and selected placements accompany these rates.

CPU and CUDA numerical outcomes can produce different trees, rejection counts
and realized batch sizes despite identical caps and seeds. Consequently this
is search throughput, not an identical-work kernel speedup. Inspect actual
batches and candidate outcomes before attributing a difference to GPU compute.
Report scene-specific results rather than combining scene complexities into
one unlabeled average.

## Prepare without running

Run from the repository root using the project Python (3.12):

```bash
python tests/benchmark_excavator_mcts_matched.py prepare \
  --output .stacks/benchmarks/cpu-gpu-matched-20261111 \
  --batch-sizes 32 --iterations 4096 --seeds 0 1 2 \
  --scene placed10=.stacks/RSS_visualization/fixed-failure-mu04/default.stack:10 \
  --scene placed20=.stacks/RSS_visualization/fixed-failure-mu04/default.stack:20
```

This writes `protocol.json`, shared batch profiles, and copies of recorded
states. The empty scene is included automatically. It performs no pose solves,
simulation, or search and constructs no CUDA workspaces. Output directories
must be new. State snapshots are trusted local pickles.

Recorded scenes must match the current target, catalog order, densities and
OBJ/PLY assets. Preparation verifies those conditions. Their original source
and context hashes are retained. They are frozen geometric starting scenes;
preparation does not revalidate their stability under today's physics.

The manifest pins application sources, native binding, core Python wrappers,
platform configurations, object assets and exported inputs by SHA-256. It also
records the paper PDF hash as context. Execution rejects changed runtime inputs
instead of mixing revisions. Prepare again after changing source or settings.

## Run later, when requested

Only the explicit `run` subcommand launches benchmarks:

```bash
python tests/benchmark_excavator_mcts_matched.py run \
  --output .stacks/benchmarks/cpu-gpu-matched-20261111
```

Use the same Python interpreter, with GPU device access, and avoid concurrent
benchmarks or other compute workloads. Hardware execution, sensing, grasp and
motion planning are not included. The runner neither imposes a subprocess
timeout nor overwrites previous trial logs/results. A failed trial stops the
suite; prepare a new directory for a clean rerun.

Per-trial JSON/log files are kept in `runs/`; the final `summary.json` checks
paired source/native/input/physical-profile equality and absence of time budgets.
The summary can be regenerated without computation:

```bash
python tests/benchmark_excavator_mcts_matched.py summarize \
  --output .stacks/benchmarks/cpu-gpu-matched-20261111
```
