# Offline excavator planning

`stacking_excavator` composes common MCTS/CEM, native pose generation and
simulation, and native direct/regrasp motion planning. It uses the commissioned
VDK23 and stone-grab models from `stacking-planner`. This delivery implements
planning, persistence, and the offline Operations Console. Excavator action
sampling uses the legacy ablation score kernels, with explicit lower-floor
priority; tabletop retains its existing sampler and filters.

## Current MCTS optimization status

The current implementation includes compact CUDA sample histories, vectorized
floor scoring and diversity selection, faster native pose-generation queries,
reuse of scene/grid computations, lattice-based voxel crops, and fewer branch
history copies. Full trajectories remain available to checkpoints and viewers.
The CPU placement/state-finalization experiment was reverted; reward/observation
processing still overlaps subsequent CUDA simulation chunks.

The active excavator CUDA profile uses 1024 iterations, an evaluation batch of
256 and four CPU result workers. Runtime chunk defaults are:

| Work | Batch size |
| --- | ---: |
| CUDA pose generation | 256 jobs |
| CUDA placement simulation | 128 placements |
| Native simulation capacity | 512 jobs, allowing four robustness trials per placement |
| CPU reward/observation processing | 32 results |

The latest [production-default benchmark](benchmarks/excavator-mcts-state-copies-and-chunks.json)
completed the fixed nine-stone starting scene with 49 inventory stones in
**71.12 seconds at 14.96 evaluated nodes/s**. It evaluated 1064 candidates under
the 1024-iteration budget. The optimization comparisons preserved ordered tree
hashes, evaluation counts and the selected candidate; larger simulation chunks
also preserved trees for held-out seeds 1 and 2. These are search-only results
for the recorded workload. The CPU profile uses a different iteration budget,
so its total search time is not an equal-work CPU/CUDA comparison.

Containment still uses the same final poses, geometry and voxel resolution.
The lattice crop optimization preserves the selected voxel centers and their
order. Observation construction now transforms only real mesh points and keeps
padding at positive infinity, preventing NaNs from rotating padded rows.

See the detailed experiments below for measurements and validation limits:
[compact samples](#compact-cuda-samples-and-trajectory-histories),
[floor sampling](#vectorized-floor-sampling),
[native pose generation](#cuda-pose-generation-call-cost),
[host-work reuse](#repeated-host-work-in-pose-generation),
[support/voxel crops](#cell-restricted-support-and-lattice-voxel-crops), and
[state copies and simulation chunks](#state-copies-and-simulation-chunk-size).

## Stability and floor-filling settings

The excavator profile rejects a placement if any noisy-placement trial fails
or its worst trial displacement exceeds `reward.place_stability.fail_displacement`
(currently `0.15`). Displacement is translation in metres plus angular distance
in radians times `rotation_weight` (`0.1` m/rad), measured from the perturbed pose.
The four trials use the configured XYZ Gaussian scales (currently 2 cm on each
axis) and 5 degree rotation noise. The common sampler takes the absolute value
of the Z sample, so translation noise only raises the stone; X/Y remain signed.
This preserves the seeded X/Y and rotation draws. Rotation can still lower a
stone's corners; upward Z noise alone does not guarantee collision-free geometry.
These trials move the candidate with previous stones fixed; the structural
simulation separately checks motion of the stack. `max_displacement` clips the
reward penalty and is not a rejection threshold.

Both structural phases now require normalized energy below `0.00001` to count
quiet steps and reject a velocity integral above `0.2`. The contact validation
gap is `0.02` m. Among accepted candidates, the robustness reward weight is `2.0`.
These are stricter starting settings, to be evaluated on representative scenes;
more rejections or longer settling can reduce search throughput.

`generation.root_sampling_priority` and `deeper_sampling_priority` remain
`lower_floor_first`. The scored sampler and strict floor filter now share the
same `floor` policy. The excavator uses `coverage_mode: projected_footprint`:
coverage is the union of the layer stones' full XY footprints, clipped to the
target base and estimated on a coarse **16×16 cell-center grid**. Overlapping
footprints count once. The placement search still uses its 32×32 grid.

Once projected coverage reaches `floor.fill_ratio: 0.6`, the policy immediately
advances to the next layer; it does not wait for search exhaustion. The new support
height is the highest top of the completed layer, and that layer's stones are
removed from the next layer's occupancy calculation. `bottom_tolerance: 0.2` m
still determines which stones belong to each layer and how far above its support
height a candidate may begin. Candidate diagnostics record the coverage mode and
completed layers' support heights and coverage values.

The sampler prioritizes proposals whose predicted contact height is at most the
active floor's support height plus the bottom tolerance. It retains Boltzmann and
spatial-diversity sampling within that pool, then fills unused slots from higher
candidates. `filtering.strict_floor_first: true` rejects solved placements above
the active layer, including CEM mutations. The motion coordinator checks the same
policy again after fixed-scene settling. Physics and individual-stone containment
remain independent acceptance checks.

For `excavator_plan4.stack` after ten placements, the previous bottom-slice estimate
was 56.5%; the coarse projected estimate is 68.75%, allowing immediate advancement
at 60%. A native search from that scene found seven valid candidates out of twelve
and selected a settled stone-3 placement above the ground layer in about 17 seconds.
A different recorded scene at roughly 55% projected coverage still rejects higher
placements. Strict filling can still exhaust search below the coverage threshold.

The shared policy defaults to bottom-slice coverage for existing callers such as
tabletop. The standalone scored sampler retains the legacy calculation when no
floor policy is supplied; the legacy capture test supplies its original 32×32
occupancy settings. The legacy `lower_floor_fill_reject_stacked` key is not a
rejection gate; `strict_floor_first` controls enforcement. Start a new run to use
the changed profile, since resume validates the original saved settings.

CEM retains at most two candidates per stone, uses 50% fresh proposals in each
refinement round, and allows up to three rejected-root retries within the search
time budget. This addresses batches dominated by similar poses of a few stones.
A controlled search from `excavator_plan2.stack` after five placements rejected
all 24 candidates with the previous settings; these settings found two valid
stone-52 placements among 24 candidates in about 17 seconds, with the same floor
and stability checks. This verifies recovery of that scene, not completion of a
full stacking run.

## CUDA pose generation

The excavator profile currently selects the experimental FP32 backend with
`parallel.posegen_backend: cuda` in `apps/excavator/config/mcts_cuda.yml`. Install the sibling
core into the active Python environment and run from this checkout:

```bash
python -m pip install -e ../stacking-core/python
stacking-excavator-plan --repository . \
  --config apps/excavator/config/mcts_cuda.yml --search-only --steps 1
```

The core pip build includes CUDA automatically when `nvcc` is on `PATH` or
`CUDACXX` specifies a CUDA compiler. Set `STACKING_CORE_ENABLE_CUDA=ON` for the
install command to require it explicitly. Verify with
`python -c "import stacking_core as c; print(c.cuda_capabilities())"`;
`compiled` and `posegen_solve` must be true for GPU pose generation;
GPU simulation also requires `simulation_solve`. No path override is needed
with a CUDA-enabled installation. For testing a separate CMake build without
installing it, prefix the planner command with
`PYTHONPATH="$PWD/../stacking-core/build-cuda/python${PYTHONPATH:+:$PYTHONPATH}"`.

The startup line reports `posegen=cuda; simulation=cuda`. A CPU-only installation or missing GPU
raises an explicit startup error. Loading the CUDA bindings alone does not
select GPU pose generation; the MCTS profile must select it too. Use
`parallel.posegen_backend: cpu` to select CPU pose generation and
`parallel.simulation_backend: cpu` for CPU simulation. Solver YAML files contain
physical and numerical settings, not backend selection; old `backend` keys there
raise a migration error. Saved contexts record both selections; resume requires
a matching profile, so start a new run when changing backends or migrating an
older CUDA context.

One reusable GPU workspace processes batches of up to 256 initialized candidates,
including candidates from different MCTS states. Each of the four continuation
stages consumes that candidate's preceding pose and gap feedback. Geometry,
sharpness, objective weights, tolerances, and placement filters are unchanged.
Placement simulation is selected independently with `parallel.simulation_backend`: one GPU
workspace per fidelity evaluates up to 32 independent placements per batch.
CUDA simulation replaces the CPU simulation worker pool; `parallel.workers`
controls CPU simulation and CPU pose generation independently. With CUDA
simulation, it also controls parallel CPU reward/observation processing.

Batched action generation prepares placed-stone PLY bounds and active-floor
context once per input State, sharing them only among proposals in that call.
Contexts are lazy and local to the call, keyed by collaborator and State
identity; input states must remain read-only until the call finishes. Different
MCTS states stay separate. Contexts are discarded on success or exception, so
pose changes between calls always trigger fresh preparation. Candidate geometry,
support order, padding and all filter thresholds are unchanged. Initializers
and filters can opt in through `prepare_scene(inventory, state)` and a
`scene_context=` keyword; existing scalar calls remain supported.

On the fixed 512-proposal recorded22 benchmark, these two changes reduce GPU
pipeline time from 7.26 s to 5.24 s with identical recorded results and zero CPU
pose retries. The fresh 12-worker CPU control takes 5.30 s, approximately equal
wall time; CPU still accepts more candidates (217 versus 183) due to the existing
numerical differences. This does not measure simulation or full MCTS throughput.
See the [scene-reuse measurements](../../stacking-core/docs/gpu-step6.md#implemented-batch-local-scene-reuse)
for intermediate timings and mixed-state regression coverage.

A `precision_limit` or outer `iteration_limit` status uses the last accepted
GPU pose and gap for continuation, provided a finite result exists and the inner
force solver converged. At the final stage it returns that pose with
`converged=False` and the original limit reason. Existing platform filters decide
acceptance: excavator permits feasible nonconverged poses; profiles requiring
convergence reject them. Solver tolerances and physical validation are unchanged.

Geometry/numerical failures, unsupported input, capacity overflow, missing or
nonfinite results, and unconverged inner force solves reject that candidate.
There are **no automatic CPU pose retries**. Other candidates continue their
schedule. Native runtime errors still propagate explicitly. CPU pose generation
remains available through `parallel.posegen_backend: cpu`; initialization and filters continue
to execute on CPU in either mode.

`PlacementSolveResult.valid=False` distinguishes an unusable result from a usable
nonconverged pose. Such results carry only placeholder pose/cost fields and an
explicit reason. Action generation rejects them before invoking any optional
filters, including profiles that allow nonconvergence or disable feasibility
thresholds. They must never become actions.

Diagnostics include `posegen_cuda_stage_statuses`, `posegen_cuda_failure_reason`
and `posegen_result_valid`. `posegen_result_backend` is always `cuda`, and
`posegen_cpu_retry` is always false on this path. Small workspace capacity limits
reject oversized candidates instead of truncating contacts. This mode remains
experimental and does not claim CPU/GPU search parity or an end-to-end speedup.

The measurements below describe earlier policies; the GPU-only follow-up in the
[Step 6 report](../../stacking-core/docs/gpu-step6.md) records the current policy.

Before allowing precision-limited results, a local RTX 5060 Ti smoke run with
49 objects, the three-round CEM
population of 96, and a two-iteration depth-one search selected a physically
validated placement in 9.41 seconds (search wall time, excluding setup). Of 288
pose candidates, 52 completed all stages on GPU and 236 restarted on CPU: 167
iteration limits, 67 precision limits, and two geometry failures. This is a
single integration check, not a speed comparison. The native CPU/GPU search
tests and CUDA adapter tests also pass; full multi-step search parity remains
unqualified.

The subsequent [Step 6 measurements](../../stacking-core/docs/gpu-step6.md) use
this precision-limit policy and a 128-job workspace (about 727 MiB). Core CUDA
Graph replay and larger chunks preserve the tested GPU results exactly. On 288
frozen empty-scene proposals, generation drops from 3.41 s (previous GPU, 32-job
chunks) to 1.88 s; the 12-worker CPU baseline is still faster at 0.33 s. There
are 169 CPU retries. In the recorded 22-stone scene, 128 proposals take 4.91 s
on the hybrid GPU path versus 1.40 s on CPU, with 22 retries and three differing
placement-filter decisions. Two three-placement search sequences per backend
produce validated selections, but CUDA is slower and selects different sequences.
Neither backend finds a validated placement in a separate 15-second search from
the 22-stone state. These bounded checks do not close the throughput or parity
gates. The core benchmark harness retains timings, retries, and quality evidence.

### GPU placement simulation

Select compute resources in the chosen MCTS YAML file:

```yaml
parallel:
  enabled: true
  workers: 12  # CPU workers; does not set GPU batch capacity
  posegen_backend: cuda  # cpu or cuda
  simulation_backend: cuda  # cpu or cuda
```

This setting applies to short and long simulation, robustness trials, and
fixed-scene resettling. `mcts_cuda.yml` selects CUDA for both stages; either can be switched to CPU
independently. The app and operations console default to `mcts_cpu.yml`. Omitted backend keys default to CPU.
`parallel.enabled` controls the CPU worker pool; it does not override the selected
backends.

Explicit excavator profiles allow independent resource tuning:

| Profile | Pose generation / simulation | CPU workers | Evaluation batch limit |
| --- | --- | --- | --- |
| `mcts_cpu.yml` | CPU / CPU | 12 | 12 |
| `mcts_cuda.yml` | CUDA / CUDA | 4 (reward/observation workers) | 128 |

```bash
stacking-excavator-plan --repository . --search-only --steps 1 \
  --config apps/excavator/config/mcts_cpu.yml
stacking-excavator-plan --repository . --search-only --steps 1 \
  --config apps/excavator/config/mcts_cuda.yml
```

These are independent complete profiles. Both use 512 iterations. CPU retains
12 root candidates; CUDA retains 32.
Editing one profile does not modify the other. The old `mcts.yml` is removed. The evaluation batch limits are starting points, not measured
optima. Actual batches depend on available candidates and each profile's root CEM retention. GPU core count does not map directly to useful batch size.
The simulation workspace still chunks at 32 placements and pose generation at
256 proposals; increasing the MCTS batch limit does not raise those capacities.

CUDA simulation runs the platform's settling physics (3D single-point contacts) and
structural physics (4D limit-surface contacts) using `CudaSimulator`. Settling
keeps the background fixed. Its downloaded final contacts select the same
support-chain/freeze-radius policy before structural evaluation; structurally
fixed and pose-identified stones remain static. The structural scene starts
with zero motion, as in CPU placement. Short and long validation share warm
contacts, motion integrals, and quiet counts within one resident rollout.
Platform time bounds map to integer step counts with strict greater-than
stopping; this avoids accumulated-time roundoff but can differ from the CPU
loop by one timestep at exact decimal boundaries.

The core continuation API now permits raising `min_steps` and supplies an
optional cumulative `max_total_steps` cap. Reinstall core after updating both
repositories. Older simulator bindings fail explicitly instead of changing the
validation schedule. Both short search and authoritative long validation use
GPU simulation, as does fixed-scene resettling before motion planning.

Noisy-placement robustness trials are batched on GPU too. When core exposes
`CudaRobustnessStopping`, trials stop on device using the existing score's
angular-energy convention, near-motion-limit shortcut, and genuine-rest check.
They return final snapshots and metrics without per-step sample histories.
Older core installations retain full-horizon simulation and exact Python replay
as a compatibility fallback. See the host-work and native-stopping measurements
below; updating stacking-core is required to obtain the native stopping gain.

Geometry, numerical, or workspace-capacity failures reject the affected
placement with `simulation_cuda_<status>`; there is no CPU simulation retry.
Finite inner constraint iteration limits are integrated as in core CPU physics,
with `simulation_cuda_unconverged_steps` reported. Motion, support, containment,
robustness, and settled-state checks still determine search acceptance. Candidate
state, reward, and observation construction use the common environment path.
GPU diagnostics include `simulation_backend`, `simulation_cuda_status`, and
`simulation_cuda_phase_statuses`. Phase wall times are apportioned across the
batch for existing per-job timing reports; MCTS batch wall time remains the
actual end-to-end measurement. GPU execution is experimental; passing an
integration test does not establish CPU/GPU search parity or a speedup.

The [integration smoke results](benchmarks/excavator-cuda-simulation-smoke.json)
record a 49-object, two-iteration empty-stack search with both modules on CUDA:
6.69 seconds, 12 root candidates, a validated selection, and four successful
robustness trials for the selected placement. On the recorded 22-stone placement
check, CPU and GPU both rejected scene motion at both fidelities, froze the same
10 stones for short validation, and agreed within 0.049 mm in final translation.
The latter check disables robustness to isolate physics. These limited checks
do not qualify full-run search parity.

### CUDA throughput and host-side bottlenecks (2026-10-03)

The [batch sweep and profiles](benchmarks/excavator-cuda-throughput-tuning.json)
compare both backends using the same 49-object inventory, frozen nine-stone
state, root retention of 32, and numerical/search settings. Runs are sequential
fresh processes on a Ryzen 9 9900X and RTX 5060 Ti, with BLAS/OpenMP threads set
to one. Only `search.search` is timed. CPU uses 12 workers; CUDA posegen capacity
remains 128. Simulation capacity changes are process-local benchmark overrides,
not changes to the production workspace of 32.

Throughput means `evaluated_candidates / elapsed_seconds`: it includes failed
physical evaluations and excludes the pre-evaluated root. In these runs it also
equals `(tree.evaluated - 1) / elapsed_seconds`. Iterations are not evaluated
nodes. In the empty scene, 300 iterations still evaluate only 32 root candidates.

Seed-0 screening at 120 iterations:

| Backend | MCTS evaluation limit | GPU simulation capacity | Search (s) | Evaluated nodes | Nodes/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 12 | — | 41.54 | 157 | 3.78 |
| CPU | 32 | — | 41.94 | 165 | 3.93 |
| CUDA | 32 | 32 | 68.21 | 198 | 2.90 |
| CUDA | 64 | 128 | 65.53 | 216 | 3.30 |
| CUDA | 128 | 128 | 60.87 | 207 | 3.40 |
| CUDA | 128 | 256 | 60.06 | 207 | 3.45 |
| CUDA | 256 | 256 | 60.21 | 207 | 3.44 |

The last screening run cannot establish a benefit for a 256-node evaluation
limit: the remaining 120-iteration budget caps reservations, and its actual
largest batch was 85, identical to the 128-limit runs. Capacity 256 can still
batch more robustness trials, which outnumber placement candidates.

Across seeds 0, 1, and 2 at 120 iterations, CPU evaluation limit 32 achieved a
median **3.80 nodes/s** versus CUDA evaluation limit 128/capacity 256 at
**3.45 nodes/s**. Median nonfailed evaluation rates were **2.23** and **2.06
nodes/s**, respectively. All searches completed without timeout and selected
a candidate accepted by their own backend. Different rejection outcomes,
explored nodes, and selected actions prevent interpreting this as exact
CPU/GPU simulation parity or equal long-horizon solution quality.

At the configured 300 iterations, seed 0 in the nine-stone scene gave CPU
**352 evaluations / 93.98 s = 3.75 nodes/s** and CUDA (128/256)
**434 / 120.16 s = 3.61 nodes/s**. In the empty depth-one scene both evaluated
32 candidates: CPU took **6.37 s (5.03 nodes/s)** and CUDA **7.39 s
(4.33 nodes/s)**. These full-budget/empty checks are single-seed observations.
A final nine-stone run at 300 iterations with evaluation limit 256/capacity 256
reached an actual batch of 144 and evaluated **397 nodes in 112.64 s
(3.52 nodes/s)**. It finished sooner than limit 128 but did less work and had
lower node throughput. Limit 128 is the better measured starting point here;
neither configuration beat CPU.

The timer-only 120-iteration CUDA 128/128 profile explains the remaining gap:

| Stage | Wall time (s) | Relationship |
| --- | ---: | --- |
| Whole search | 61.29 | Total |
| Pose generation and priors | 28.39 | Includes native pose generation |
| Native CUDA pose-generation calls | 8.96 | Subset of generation |
| Transition evaluation batches | 31.17 | Includes physics and result processing |
| Native CUDA simulation calls | 11.66 | Subset of transitions |
| Voxel overlap | 8.80 | Across generation and rewards |
| Observation construction | 5.29 | Within transitions |
| State construction, including trajectory copies | 5.16 | Across search stages |

Inclusive rows overlap and must not be added. The two native CUDA call categories
sum to **20.62 s**; **40.67 s** remains outside them. Native call timings include
transfers/binding conversion, not just GPU kernels. CPU worker timer totals
also overlap across threads; the corresponding CPU timer-only search took
41.01 s, with 15.73 s in generation and 23.79 s in transition batches.

Scoped cProfile identifies the main generation hotspot in
`actions/_scored_floor.py`: 419 `score_xy_debug_map` builds, about 384,000
`target_boundary_proximity` calls, and about 385,000 `target_anchor_bounds`
calls. That diagnostic run adds profiling overhead (73.60 s total), so its
function times are attribution evidence rather than benchmark timings.
The CUDA run attempted 670 pose proposals versus CPU's 501; CUDA also reported
97 posegen geometry rejections versus none on CPU. More attempted work and
different tree traversal contribute to the total-time difference.

CPU parallelism is the next useful experiment. The CUDA path batches native
solvers but builds reward/observation results sequentially in
`StackingEnvironment.step_batch`. Independent result evaluators with private
caches could use CPU workers without allocating additional GPU simulators.
For the floor sampler, first vectorize scalar grid calculations and reuse the
state-map bundle across stones. If it remains dominant, use a persistent process
pool for independent state/object score maps; keep RNG sampling, result order,
and tree updates coordinated. Python-heavy loops will not automatically scale
with threads, and shipping full trajectory histories to processes could erase
the gain. Those measurements predate the CUDA result workers described below. Floor
score-map generation remains serial; reward/observation processing is now
parallelized.

Reproduce a controlled run with the frozen YAML stored in the JSON report:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python tests/benchmark_excavator_cuda_mcts.py \
  --config /tmp/frozen-mcts.yml --backend cuda --seed 0 --iterations 120 \
  --evaluation-batch-size 128 --cuda-batch-size 256 \
  --state /tmp/state9.pkl --output /tmp/cuda-search.json
```

The state pickle must be a trusted local export matching the report's state
hash. For CPU use `--backend cpu --evaluation-batch-size 32` and omit
`--cuda-batch-size`. `tests/profile_excavator_cuda_mcts.py` accepts the same
arguments plus `--profile-output`; it now supports either backend. Profiled
runs are excluded from the throughput summary.

### Parallel CPU results after CUDA simulation

CUDA search uses `parallel.workers` for CPU reward and observation processing.
The CUDA profile selects four workers; `parallel.enabled: false` reduces this
to one without changing either solver backend. CPU simulation keeps its existing
independent transition workers and does not create nested result pools.

```yaml
parallel:
  posegen_backend: cuda
  simulation_backend: cuda
  enabled: true
  workers: 4
```

Each short/long search environment owns a reusable thread pool and a separate
reward/observation evaluator pair per worker, including private overlap caches
and height-map builders. Physics stays batched on the calling thread. Candidate
states are distributed in ordered buckets after each simulation chunk; results
are merged in original order before MCTS updates. Multi-chunk CUDA batches can
overlap these CPU results with the next physics chunk (see below). Empty and single-state batches stay
serial. Errors drain all submitted work before propagating; closing the search
joins the pools. Scalar environment calls and committed-state evaluation retain
their serial behavior. No additional GPU simulators are created for CPU workers.

The [worker-count measurements](benchmarks/excavator-cuda-result-workers.json)
use the same frozen nine-stone workload as the throughput report: 120 iterations,
root retention 32, evaluation limit 128, GPU capacity 256, and one BLAS/OpenMP
thread. The GPU capacity is a benchmark-only override; production remains 32.

| CPU result workers | Reward/observation wall (s) | Whole search (s) | Evaluated nodes/s |
| ---: | ---: | ---: | ---: |
| 1 | 12.61 | 59.85 | 3.46 |
| 4 | 9.89 | 57.17 | 3.62 |
| 8 | 9.66 | 56.97 | 3.63 |
| 12 | 10.17 | 57.44 | 3.60 |

All four seed-0 runs evaluated 207 nodes with exactly matching tree statistics,
CEM history, selected pose and reward. Four workers reduced result-processing
time by about 22% and total search time by 4.5%. Eight workers were only 0.20 s
faster overall in this sample. Setup, excluded from search timing, increased
from 0.39 s with one worker to 0.69/1.14/1.56 s with 4/8/12 workers.

Seed 1 confirmed four workers: reward/observation time fell from 13.04 to
10.35 s, total search from 57.88 to 55.29 s, and throughput increased from
3.70 to 3.87 nodes/s. Counts, tree, CEM and selected pose/reward matched exactly.
At the production evaluation/capacity of 32, seed 0 also preserved all 198
evaluations and exact search results: result-processing time fell from 11.96
to 9.44 s, search from 67.72 to 65.16 s, and throughput increased from 2.92
to 3.04 nodes/s. These measurements predate the floor score-map optimization below and do not
establish a general speedup over CPU MCTS.

Use `--workers 1`, `4`, `8`, or `12` in
`tests/benchmark_excavator_cuda_mcts.py` to reproduce the worker ablation. The
benchmark records `result_processing_batches` separately from CUDA physics
calls. Threaded profiler substage totals overlap across workers; use
`reward_observation_batch_wall` for elapsed result-processing time.

### Vectorized floor score maps

Floor scoring now evaluates height-dependent target bounds and boundary
proximity as arrays for each orientation, normalizes candidate scores in one
array operation, and computes the occupied-neighbor dilation once per map.
The sampler also reuses the state's resized/smoothed height-map bundle across
stones. Its existing content-based cache invalidates that bundle when scene
heights, placed poses, grid bounds, sampling settings, or grid shape change.

Candidate enumeration retains orientation-major, row-major order. Nonfinite
support scores are still skipped, missing contact heights still use the original
footprint-height fallback, and equal scores still retain the first orientation.
These changes apply to the shared scored sampler used by both CPU and CUDA
excavator search; the tabletop height-map sampler is unchanged. No backend,
worker-count, batch-size, or solver-profile changes are required.

The [floor-map measurements](benchmarks/excavator-floor-score-maps.json)
compare the previous scalar implementation with the optimized implementation.
Both include the preceding reward/observation worker change. Runs use the same
49-object inventory, frozen nine-stone state, 120 iterations, root retention 32,
and one BLAS/OpenMP thread. Search wall time excludes setup; processes run
sequentially. GPU capacity 256 remains a benchmark-only override.

| Backend / evaluation limit / GPU capacity | Seed | Before (s) | After (s) | Evaluated nodes/s, before → after |
| --- | ---: | ---: | ---: | ---: |
| CUDA / 128 / 256, 4 CPU result workers | 0 | 57.56 | 48.53 | 3.60 → 4.27 |
| CUDA / 128 / 256, 4 CPU result workers | 1 | 55.33 | 46.71 | 3.87 → 4.58 |
| CUDA / 32 / 32, 4 CPU result workers | 0 | 65.20 | 56.01 | 3.04 → 3.53 |
| CPU / 32, 12 workers | 0 | 42.02 | 36.67 | 3.93 → 4.50 |

CUDA search time decreased by 14–16%; CPU search time decreased by 13%.
Before/after pairs preserve evaluated counts, tree statistics, CEM history,
selected poses and rewards exactly. Counts are 207/214 for the larger CUDA
batches, 198 for production CUDA batches, and 165 for CPU. CPU and CUDA explore
different trees, so this is a within-backend optimization comparison, not proof
of equivalent physics or search quality across backends. CUDA has not established
a throughput advantage over CPU in this comparison.

A separate timer-only pair attributes the improvement to floor scoring:
419 score-map builds plus their state-map preparation fell from **12.41 to
3.71 s**, about **3.35× faster**. State-map builds dropped from 419 to 136.
Before optimization, state-map construction is nested inside the score-map
timer; afterward it runs in the sampler before that timer. The 3.71 s total
therefore includes 3.26 s scoring plus 0.45 s state maps, while 12.41 s already
includes the old 1.37 s state-map cost. Pose generation/prior processing fell
from 27.90 to 19.20 s; transition batches stayed near 27.8 s.

Validation: 351 affected shared/tabletop/excavator tests passed. An additional
12-case comparison against the previous source checks every returned map and
candidate field, plus seeded samples, exactly for empty/placed scenes and
normal, flat, and nonfinite height maps. Regression tests cover tapered and
collapsed target bounds, empty batches, nonfinite support/contact heights,
orientation tie ordering, and cross-stone cache invalidation.

### User profiles at 512 iterations

The [512-iteration profile benchmark](benchmarks/excavator-mcts-512-profiles.json)
uses the updated CPU and CUDA YAMLs exactly, with no iteration, batch, worker,
root-retention, or native CUDA-capacity overrides. One seed-0 search per profile
ran sequentially on the same 49-stone inventory with nine stones already placed.
Both completed all 512 iterations at effective depth 3 without timing out.
Search timings exclude setup.

| Metric | CPU profile | CUDA profile |
| --- | ---: | ---: |
| Evaluation batch limit | 12 | 32 |
| Root candidates retained | 12 | 32 |
| CPU workers | 12 | 4 |
| Search wall time | 69.53 s | 133.83 s |
| Evaluated nodes, including failures | 301 | 501 |
| Evaluated nodes/s | 4.33 | 3.74 |
| Nonfailed evaluations | 149 | 279 |
| Nonfailed evaluations/s | 2.14 | 2.08 |
| Pose generation wall time | 21.57 s | 49.75 s |
| Short + long evaluation batch wall time | 43.28 s | 76.44 s |

CUDA evaluated 66% more nodes and took 92% more time; CPU's evaluated-node
throughput was about 16% higher. Nonfailed-evaluation throughput was much closer.
Iterations can revisit nodes, and these profiles have different root retention,
batch limits and search trajectories. This is a comparison of the configured
profiles, not an isolated CPU/GPU hardware comparison or a multi-seed quality
study. Evaluated-node counts exclude the already-evaluated root. The report
includes exact YAML snapshots, source hashes, selected actions and rewards.

### CUDA bottleneck diagnosis at 512 iterations

The [phase and Nsight report](benchmarks/excavator-cuda-512-bottleneck.json)
profiles the same current CUDA configuration, seed 0, and nine-stone state.
The timer-only run took 134.21 s versus 133.83 s unprofiled. The separately
traced run took 140.57 s. Both completed all 512 iterations and exactly
reproduced the 501 evaluated nodes, tree statistics, CEM history, selected
pose and reward.

The timer-only wall decomposition is:

| Nonoverlapping category | Seconds |
| --- | ---: |
| Native CUDA posegen calls | 27.15 |
| Native CUDA simulation calls | 36.40 |
| CPU reward/observation batch wall | 23.42 |
| Floor-map preparation | 10.60 |
| Other time outside native calls | 36.63 |

Native call time includes bindings, host preparation, GPU execution, transfers
and waits. Total time outside those native calls is 70.65 s. Per-worker reward,
overlap and observation timer sums run concurrently; they must not be added to
the wall categories above. Native simulation includes settling 3.32 s,
structural checks 13.49 s, and robustness trials 19.59 s.

Nsight Systems captured search only, with CPU sampling disabled and graph-level
CUDA graph tracing. Its device-side measurements distinguish actual copies
from host API waits:

| Transfer | Count | Data | Device copy time |
| --- | ---: | ---: | ---: |
| CPU → GPU | 3,385 | 266.10 MB | 0.0124 s |
| GPU → CPU | 72,584 | 4,438.90 MB | 0.2532 s |

Actual CPU/GPU copies took **0.266 s**, about 0.19% of the traced search.
GPU kernels took **58.57 s**; the largest were the simulation 4D constraint
solver at **26.04 s** and posegen force solver's `tail` kernel at **12.93 s**.
Graph execution contributes another 0.0735 s; the union of kernel, graph,
copy and memset intervals is 58.92 s. This is active timeline duration,
not achieved SM occupancy.

A host API summary alone is misleading: `cudaMemcpyAsync` totals 35.33 s.
There are **70,175 eight-byte device-to-host readbacks**, using pageable host
memory. They spend **26.24 s inside the API**, but only **0.0350 s copying**.
Correlated device activity overlaps 23.82 s of that API time. Source inspection
identifies the posegen two-integer progress buffer and blocking download helper
in `stacking-core/src/posegen/cuda/pose_generator.cpp` and
`stacking-core/src/cuda/runtime.hpp`. These are dependent progress readbacks,
not evidence of saturated transfer bandwidth. Similarly, 26.18 s of the
26.90 s in kernel-launch APIs overlaps device activity; do not label all of it
independent launch overhead or add it to kernel execution.

The current bottleneck is therefore a mix of host-side work and GPU solver
execution. The 4D solver uses 128-thread blocks with at most 32 blocks in this
trace, on a 36-SM GPU; this suggests limited available parallelism, but achieved
occupancy and GPU-internal memory bandwidth need a focused Nsight Compute run.
Prioritize solver work/parallelism and CPU result processing. Reducing progress
polling may help, but cannot recover all 26.24 s of readback API time because
most overlaps GPU work. Pinned transfers alone cannot eliminate that work or
the algorithm's dependency on progress results.

The compact JSON retains CUDA API/kernel/copy summaries, exact input/source
metadata, correlation results and the capture wrapper. The full local trace is
`/tmp/mcts-cuda-bottleneck-20261003/cuda-search.nsys-rep`; it is not checked into
the repository. No runtime code or configuration changed for this diagnosis.

### CUDA constraint solver optimization

The [constraint-kernel report](benchmarks/excavator-cuda-constraint-optimization.json)
records the next native optimization in the sibling `stacking-core` repository.
The simulation constraint kernel now uses 256 threads per job, giving eight
warps for concurrent body updates instead of four. Each warp cooperatively
loads the triangular factor into shared memory before the serial forward/back
substitution. The per-body arithmetic order, solver limits, convergence tests,
physical settings and public APIs stay the same.

Controlled 100-step rollouts of the recorded 22-stone scene compared the old
128-thread kernel, shared-factor staging alone, and 256/512-thread variants.
All 3D/4D state fingerprints, statuses, final solver iterations/residuals and
contact counts matched exactly at batches 32 and 128. One cold call and two
warm calls were run per case; the table shows warm medians.

| 4D rollout batch | Baseline | Shared factor, 128 threads | Shared factor, 256 threads | Shared factor, 512 threads |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0.938 s | 0.918 s | 0.707 s | 0.648 s |
| 128 | 1.852 s | 1.733 s | 1.485 s | 1.713 s |

The 256-thread variant was retained for its gains across both batch sizes.
The 512-thread trial improved the small batch further but was slower than
256 at batch 128. No runtime tuning option was added. These repeated-scene
rollout measurements do not represent MCTS search time.

For the existing 512-iteration, seed-0 MCTS workload with nine stones placed
and 49 total stones, the retained variant gave:

| Measurement | Before | After |
| --- | ---: | ---: |
| Native CUDA simulation, timer-only profile | 36.40 s | 33.02 s |
| Whole search, unprofiled | 133.83 s | 130.03 s |
| Evaluated nodes/s | 3.74 | 3.85 |

All 501 evaluations, tree statistics, CEM history, selected pose and reward
matched exactly. Native simulation time decreased 9.3%; whole-search time
decreased 2.8%. CPU's previously measured 4.33 evaluated nodes/s remains higher.
This is one MCTS seed; the larger 22-stone rollout gains should not be
extrapolated to the full search.

The project's editable `stacking-core` installation was rebuilt with CUDA.
The final run uses the normal interpreter without a `PYTHONPATH` override;
no YAML changes are needed. Two native CUDA regression suites and 163
application tests passed against the optimized module. The native source
change is `stacking-core/src/simulation/cuda/constraint_solver.cu`; the
preceding Nsight diagnosis measured the older 128-thread version.

### Register-distributed triangular-solve experiment

The [register experiment report](benchmarks/excavator-cuda-register-experiment.json)
compares two warp-shuffle implementations against the retained 256-thread,
shared-factor solver. Both distribute the 21 triangular-factor entries among
warp registers. The first unrolls the substitutions; the second shortens
register lifetimes and retains serial row order with rolled loops.

The workload repeats a recorded 22-stone dynamic scene for 100 timesteps.
These are 4D rollout times, with medians over three warm calls after one cold
call per case; they are not full MCTS timings.

| Implementation | Registers/thread | Batch 32 | Batch 128 | Exact 4D output parity |
| --- | ---: | ---: | ---: | --- |
| Retained shared factor | 60 | 0.708 s | 1.487 s | Baseline |
| Unrolled register solve | 72 | 0.723 s | 1.869 s | No |
| Rolled register solve | 60 | 0.869 s | 1.692 s | Yes |

The unrolled variant passed tolerance-based native tests but changed 4D
outputs and the final solver iteration count from 217 to 218. The rolled
variant recovered exact parity but remained slower. Extra shuffle/control
work and increased register pressure are possible costs; hardware counters
were not collected to isolate their contribution.

Both experiments were rejected. The previous source was restored exactly and
the development module rebuilt; the installed project module remained
unchanged throughout. Both native CUDA regression suites passed for each
variant and again after restoration. No new full MCTS run was performed
because neither variant improved the 4D rollout timings.

### Proposal budgets for batched expansion

Batched descendant generation retains each state's reserved evaluation count.
Pending siblings consume spare evaluation capacity first; remaining slots are
shared among expandable states within their existing widening and generation
limits. Root lookahead distributes its evaluation capacity across the selected
parents before requesting poses, ensuring one slot per eligible parent before
allocating extra slots.

Generation targets count newly accepted children. After rejected proposals,
only the unfilled target is requested again, subject to the existing retry-round
limit, per-node attempt budget, generator exhaustion and generation deadline.
Previously accepted children do not prematurely stop this refill. Slots that an
exhausted state cannot use are reassigned to other selected states with capacity,
within the same retry-round limit. The accepted-candidate quota replaces
`min_feasible_children` as the refill target for these batched calls; that setting
still governs unbudgeted root/serial expansion. Only reserved selections receive
backups; additional sibling evaluations do not consume extra iterations. Root admission/CEM and serial expansion retain their existing
proposal policy. No configuration or native solver changes are required.

This changes proposal request sizes and search allocation, so RNG consumption
and the resulting tree can change. Physics validation and failure handling are
unchanged. Throughput comparisons must report evaluated and nonfailed nodes per
second, rather than generator calls or elapsed time alone. Pending proposals
must not count as evaluated work.

The [proposal-budget benchmark](benchmarks/excavator-mcts-proposal-budgets.json)
uses the same 49-stone inventory, frozen nine-stone state, depth three and
512 iterations. Runs are sequential, with one seed per case and one
BLAS/OpenMP thread. Search timing excludes setup; before-results come from
the preceding measurements with verified source hashes.

| Profile | Evaluated nodes, before → after | Wall seconds, before → after | Evaluated nodes/s, before → after | Nonfailed nodes/s, before → after | Pending, before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU current (12 workers) | 301 → 279 | 69.53 → 65.43 | 4.33 → 4.26 | 2.14 → 2.32 | 52 → 0 |
| CUDA current (32/32, widening on) | 501 → 461 | 130.03 → 120.93 | 3.85 → 3.81 | 2.15 → 2.17 | 21 → 0 |
| CUDA 128/128, widening on | 698 → 604 | 150.19 → 127.42 | 4.65 → 4.74 | 2.58 → 2.72 | 6 → 0 |
| CUDA 128/128, widening off | 799 → 667 | 172.12 → 134.72 | 4.64 → 4.95 | 2.52 → 2.82 | 320 → 0 |

The larger CUDA cases improve evaluated-node throughput by 2.0% with widening
and 6.7% without it. The current CPU and 32/32 CUDA profiles decline by 1.5% and
1.1% respectively on that metric; their nonfailed rates improve by 8.4% and 1.4%.
Thus shorter elapsed time and fewer unused proposals do not imply a universal
throughput improvement.

All searches completed without timeout. The batch notation is MCTS evaluation
limit/native simulation capacity. Native capacity 128 and disabled widening
were process-local/temporary benchmark settings; the production profiles were
not changed. These single-seed results measure workload-dependent throughput,
not a universal speedup or equivalent decision quality. The current final
descendant-scoring depth is zero, so matching selected actions alone does not
validate deeper-search quality. The report includes expanded-parent rates
separately from evaluated-node rates.

Validation: 117 shared and platform search tests passed, including reserved
generation limits, root coverage, pending reuse, partial rejection, exhausted
state reassignment, widening bounds, deadlines, retry/attempt limits, mixed
fidelities and exception recovery.

### Ordered CPU/GPU result pipeline

The original step-2 pipeline streams owned placement states in chunks of 32.
With more than one CPU result worker and more than one chunk, rewards and
observations for chunk k run while the caller processes physics for chunk k+1.
A single coordinator dispatches into the existing worker pool. That version
permits one CPU result group in flight, with a private evaluator/cache pair per
worker. The independent-result-chunk implementation below subsequently separates
CPU granularity and permits two pending groups. Global round-robin assignment
continues across chunk boundaries, including short tails.

The GPU workspace stays on the calling thread. Full input validation precedes
physics; workers drain before errors propagate or environments close. All
results return in input order before MCTS backups, so this does not introduce
asynchronous tree updates. CPU simulation, scalar calls, single-worker results,
and single-chunk batches retain synchronous execution.

`generation.evaluation_batch_size` limits the complete MCTS evaluation batch.
The implementation's `CudaPlacementBackend.placement_batch_size = 32` controls
streaming, separately from `batch_size`, which limits native simulation jobs
including robustness trials. The step-2 profile used evaluation/native limits
128/32, permitting multiple chunks; the later tuning below updates them to
256/128. No new production configuration option is introduced. Actual counts
still depend on available search work. Its previous limit of 32
had no within-batch overlap. Generation remains 32 and CPU result workers remain
four; iterations, widening, root retention and the CPU profile are unchanged.

The [pipeline measurements](benchmarks/excavator-cuda-result-pipeline.json)
use the same frozen nine-placed-stone state from a 49-stone inventory, seed 0,
depth three, 512 iterations, progressive widening, and four result workers.
Processes run sequentially with one BLAS/OpenMP thread; timing covers only
`search.search`. The controlled comparison uses evaluation limit 128 and
native capacity 128:

| Result schedule | Placement chunk limit | Search seconds | Evaluated nodes/s | Nonfailed nodes/s |
| --- | ---: | ---: | ---: | ---: |
| Whole batch, synchronous | 128 | 127.74 | 4.73 | 2.71 |
| Streamed, synchronous | 32 | 133.75 | 4.52 | 2.59 |
| Streamed, overlapping | 32 | 117.48 | 5.14 | 2.95 |

All three evaluated 604 candidates, including 346 nonfailed candidates, with
no pending nodes or timeout. Ordered-tree hashes match exactly, covering
incoming actions, resulting object poses, priors, rewards, visits, backed-up
values, failure/terminal flags, rejection reasons and child ordering. This
compares more than the final selected action, but does not hash every diagnostic
or trajectory sample; native regression tests cover streamed trajectory parity.

The pipeline improves evaluated-node throughput by 8.7% over the whole-batch
path and 13.8% over identical serialized chunks. Chunking alone costs 6.0 seconds,
mainly in CUDA calls. CPU result wall totals increase from 28.84 to 37.55 seconds
under concurrency, but 17.04 seconds overlaps CUDA `_run` calls and additional
work overlaps host preparation. These inclusive timers overlap and must not be
summed as disjoint phases. `_run` includes native calls and immediate result
handling; its interval is not a pure device-kernel timer.

With the production native capacity of 32, increasing the CUDA evaluation limit
from 32 to 128 gives the following measured result, now selected in
`apps/excavator/config/mcts_cuda.yml`:

| Evaluation limit | Evaluated nodes | Search seconds | Evaluated nodes/s | Nonfailed nodes/s |
| --- | ---: | ---: | ---: | ---: |
| Previous 32 (one chunk) | 461 | 120.58 | 3.82 | 2.18 |
| New 128 (pipeline) | 604 | 123.42 | 4.89 | 2.80 |

The new configuration improves evaluated-node throughput by 28.0%, with 31.0%
more evaluations and 2.4% more elapsed time. This comparison includes the change
in MCTS batch allocation; the earlier fixed-workload comparison isolates the
pipeline gain. The production-capacity 128-limit run has the same ordered-tree
hash as all three controlled 128-limit runs. The 32-limit run matches the prior
proposal-budget benchmark's evaluation/failure counts, tree statistics, CEM
history and selected candidate. All five runs complete without timeout.

Reproduce the controlled runs with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/benchmark_excavator_cuda_mcts.py \
  --backend cuda --config apps/excavator/config/mcts_cuda.yml --seed 0 \
  --state /path/to/trusted/state9.pkl --evaluation-batch-size 128 \
  --cuda-batch-size 128 --placement-batch-size 32 --output /tmp/pipeline.json
```

Add `--serial-results` for the serialized-chunk control, or use
`--placement-batch-size 128` for the whole-batch control. These switches only
modify the benchmark process. Single-seed timings establish the observed gain
on this workload, not a universal speedup or improvement in decision quality.

Validation: 210 tests passed across shared environments, native CUDA placement,
MCTS and both platforms. Coverage includes forced overlap, stable evaluator
assignment, tail chunks, private state ownership, full upfront validation,
worker draining after physics/reward failures, reuse and shutdown, and exact
streamed native pose/trajectory parity. The updated CUDA profile resolves
exactly to the recorded production-capacity 128-limit benchmark profile.

### Deferred observations and native robustness stopping

Search environments defer array observations until the first field access.
Unused failed nodes and leaves therefore avoid geometry transformation and
height-map rendering. The object remains an `Observation`; optional fields,
learned-policy encoding, debug/artifact serialization, pickling, deepcopy and
`dataclasses.replace` materialize its values when needed. Materialization is
cached and synchronized. Ordinary episode/committed-state environments retain
eager observations. Custom builders without a `defer` method retain their
existing behavior.

The standard inventory adapter snapshots the current scene for deferred
observations, omitting trajectory/action histories that array construction and
platform height maps do not consume. Custom observation sources without that
snapshot method receive a full independent copy. Node state histories remain
complete and branch-owned. `State.copy()` copies owned mutable values directly
instead of re-running construction validation; constructor validation is
unchanged. This avoids redundant validation, not all trajectory copies, and does
not introduce shared mutable search histories.

Native CUDA robustness now supplies a separate stopping rule using FP64 model
mass/inertia and metric accumulation. It preserves the legacy extra
`R.T @ angular` transformation, strict physical-time inequalities, the
`integral > 0.9 * limit` quiet shortcut, the separate genuine-rest count, and
motion-limit rejection priority. Default structural stopping and FP32 physical
integration are unchanged. No YAML setting is needed: the updated core exposes
`CudaRobustnessStopping`, and the app selects it automatically. The rebuilt
editable core module was installed in this project's direnv environment.

With native stopping, `sample_stride = 0` returns only the stopped snapshot and
metrics. In this step-4 measurement, jobs stop computing physics once finished,
although the native host loop still enqueues the phase's maximum number of
timesteps with finished jobs masked. Steps 5–6 below subsequently replace that
launch loop. Constraint kernels are unchanged by these launch optimizations.

The [step 3–4 benchmark](benchmarks/excavator-mcts-host-and-stopping.json) uses the
same 49-stone inventory, frozen nine-stone stack, seed 0, depth three and 512
iterations. CUDA stays at evaluation/native/placement limits 128/32/32 with four
CPU workers. No search configuration changes are included in these steps.

| CUDA case | Search seconds | Evaluated nodes/s | Nonfailed nodes/s |
| --- | ---: | ---: | ---: |
| After step 2 | 123.42 | 4.89 | 2.80 |
| Step 3, legacy replay on rebuilt core | 120.22 | 5.02 | 2.88 |
| Steps 3–4, native stopping | 111.08 | 5.44 | 3.11 |

A final-code repeat completed in 110.64 seconds (5.46 evaluated and 3.13
nonfailed nodes/s), again with the same ordered-tree hash.

Each case evaluates 604 candidates, including 346 nonfailed nodes, and matches
the ordered-tree hash from the pipeline benchmark. The same-binary legacy/native
comparison isolates an 8.2% throughput gain from step 4; the combined gain over
step 2 is 11.1%. Step 3's smaller gain is about 2.7% on this run. An intermediate
step-3 profile materialized 137 observations rather than one per evaluated node;
its wall time includes profiler overhead and is not used for the throughput table.

Native robustness processes 149,930 simulated steps versus 555,212 in the legacy
control, and returns zero per-step samples versus 555,212. CUDA robustness call
wall time drops from 22.45 to 14.69 seconds (34.5%). These are wrapper call times,
including native host/result work, not pure device-kernel timings. Settling and
structural work counts match exactly.

The current CPU profile also benefits from the shared host changes: 279
evaluations in 60.13 seconds, or 4.64 evaluated and 2.53 nonfailed nodes/s.
Evaluation/failure counts, tree statistics, CEM history and the selected candidate
match its preceding proposal-budget run (65.43 seconds, 4.26 evaluated nodes/s).
CUDA and CPU use different root retention/evaluation batches, so their rates are
profile comparisons, not a hardware-only or equal-decision-quality comparison.

Use `--legacy-robustness` with `tests/benchmark_excavator_cuda_mcts.py` to retain
full-horizon simulation and Python replay on the new binary. Use
`--eager-observations` to disable deferred arrays for measurement. Both are
process-local benchmark controls; production behavior requires no new options.
The report contains resolved profiles, source/native hashes and raw timing/work
counters. Single-scene, single-seed results are workload-specific. Native FP64
metrics match replay within tested floating-point tolerances; bit-identical
arithmetic at every possible threshold boundary is not guaranteed.

Validation: 296 Python tests passed across native simulation, observations,
state ownership, MCTS, both platforms, restart/artifacts and visualization.
Two rebuilt native C++ CUDA test binaries passed. Native tests compare stopping
steps and exact final poses with full-horizon replay, and test strict time bounds,
near-limit unsettled results, simultaneous stop priority, anisotropic inertia,
compact/strided outputs, invalid rules and continuation rejection. Application
tests also cover the older-core replay fallback.


### Completion checks, reusable buffers, and batch tuning (steps 5–7)

The native simulator now submits 32-step CUDA Graph groups, checks completion
between groups, and stops submitting once every job finishes. Each job still
stops on its exact device timestep; unused steps within a group are masked.
Partial final groups use direct launches. Graphs live for one phase to keep
captured pointers, grids and sample strides valid. Scratch and sample buffers
reuse allocations within one simulator; logical sizes reset on each call/phase.
Results continue to own their snapshots and samples.

The [launch and batch report](benchmarks/excavator-mcts-launch-and-batches.json)
records the experiments on an RTX 5060 Ti (16 GB). Completion intervals 16, 32,
and 64, buffer reuse, and graphs of 32/64 steps were replayed on the first real
MCTS batch from all six short/long simulation stages, with three warm repetitions.
Every variant matched the baseline input/output digests for poses, velocities,
samples and phase metrics. These digests do not cover every contact/solver field;
native regression tests supplement the replay checks.

Launch changes alone delivered **no meaningful whole-search speedup**: 110.53 s
versus the preceding 110.64 s run, with exactly the same ordered tree. Warm
stage replay showed small gains, while buffer reuse alone was within noise.
The retained 32-step grouping reduces submission work without changing physics;
it should not be credited for the larger batch-tuning gains below. The baseline
Nsight capture recorded about 0.141 s of actual host/device transfers and 0.065 s
of malloc/free API time. Large launch/copy API durations include GPU backpressure
and must not be interpreted as pure launch overhead or memory transfer time.

The seed-0 screening held iterations at 512, generation at 32, four CPU result
workers, root retention 32, and the same frozen nine-stone stack from 49 inventory
stones. It varied the three batch limits separately:

| Evaluation / native / placement | Search seconds | Evaluated nodes/s | Nonfailed nodes/s | Distinct expanded parents/s |
| --- | ---: | ---: | ---: | ---: |
| 128 / 32 / 32 | 110.53 | 5.46 | 3.13 | 1.22 |
| 128 / 64 / 32 | 106.91 | 5.65 | 3.24 | 1.26 |
| 128 / 128 / 32 | 104.36 | 5.79 | 3.32 | 1.29 |
| 128 / 128 / 64 | 104.90 | 5.76 | 3.30 | 1.29 |
| 64 / 128 / 32 | 100.09 | 5.25 | 2.96 | 0.85 |
| **256 / 128 / 32** | **101.50** | **6.03** | **3.48** | **0.92** |

The first four rows produce exactly the same ordered tree. Increasing native
capacity to 128 cuts robustness wrapper time from 14.46 s to 8.36 s, letting
four trials for each 32-placement chunk run together. Placement chunks of 64
also reduce settling/structural call time, but result-processing overlap falls
from 9.83 s to 6.26 s and result processing itself takes longer; whole-search
performance does not improve. These wrapper times include host work.

The retained defaults favor **evaluated-child throughput**:

- Excavator CUDA `generation.evaluation_batch_size: 256`.
- Common CUDA native simulation capacity `128`.
- Placement chunks `32`, generation `32`, and CPU result workers `4`.

CPU and tabletop YAML profiles retain their settings. Tabletop CUDA simulation
uses the common native capacity when selected; its behavior is regression-tested,
although these throughput measurements are excavator-specific. There is no new
YAML knob for native capacity or placement chunks. Benchmark overrides remain
available through `--cuda-batch-size` and `--placement-batch-size`.

Held-out checks compare 128/32/32 with 256/128/32 at seeds 0–2, without changing
iterations or physics:

| Nine-stone seed | Control evaluated nodes/s | Tuned evaluated nodes/s | Gain |
| --- | ---: | ---: | ---: |
| 0 (screening) | 5.46 | 6.03 | 10.3% |
| 1 | 5.69 | 6.21 | 9.2% |
| 2 | 5.50 | 6.05 | 10.1% |

A final run using the production defaults with no batch overrides completed in
**102.80 s**, evaluating **612 nodes at 5.95 nodes/s** (3.43 nonfailed nodes/s).
Its ordered-tree and native-binary hashes match the screened 256/128/32 run.
This repeat is 8.9% faster in evaluated throughput than the 128/32/32 control.

Nonfailed throughput also increases on all three seeds. Empty-stack searches
(effective depth one rather than three) evaluate 32 candidates each, retain the
same ordered tree per seed, and improve by 3.4%, 3.9%, and 4.4% in these single runs.
All searches complete 512 iterations without timing out. The seed-1 tuned search
takes slightly longer overall because it evaluates more candidates.

**Tree breadth is a tradeoff.** On the nine-stone stack, distinct expanded-parent
throughput drops from 1.22/1.12/1.15 to 0.92/1.00/1.06 parents/s for seeds 0/1/2.
Set the evaluation batch back to **128** if distinct parent expansion is the
priority; native capacity 128 still improves the seed-0 same-tree throughput.
Evaluation-batch changes legitimately alter search allocation. Equal selected
root actions/rewards in these runs do not establish equal decision quality,
especially with descendant validation depth zero. Repeated timings across more
scenes and a planning-quality evaluation remain separate work.

Validation: **297 Python tests and two rebuilt native C++ tests passed**, including
both platform workflows, MCTS, artifacts/restart, observation/state ownership,
visualization, native stopping and variable batch/sample reuse. One existing test
was updated to compare evaluator capacity with its backend rather than hard-code
32. Raw timings, resolved profiles, source/native hashes and replay digests are
in the report. The installed editable stacking-core module matches the measured
32-step graph binary.


### More CUDA batches per search

The current excavator CUDA profile increases `tree.iterations` from **512 to
1024** to run more evaluation rounds. Evaluation/native/placement limits remain
**256/128/32**, with four CPU result workers. The 240-second search limit and
30-second validation reserve remain in place. Batch counts depend on available
frontier work and deadlines; doubling iterations does not guarantee twice as
many batches. Empty stacks still use effective depth one and can exhaust their
available candidates before benefiting from a larger iteration budget.

The [iteration-budget comparison](benchmarks/excavator-mcts-more-batches.json)
uses the same nine-stone stack, 49-stone inventory and seed 0 as the preceding
production-default run:

| Metric | 512 iterations | 1024 iterations |
| --- | ---: | ---: |
| MCTS evaluation batches | 6 | 8 |
| Native CUDA simulation calls | 66 | 111 |
| Evaluated candidates | 612 | 1064 |
| Search seconds | 102.80 | 174.35 |
| Evaluated nodes/s | 5.95 | 6.10 |
| Nonfailed nodes/s | 3.43 | 3.37 |
| Distinct expanded parents | 93 | 188 |

The longer search runs seven short evaluation batches and one long batch.
Each simulation stage executes 37 native calls: 30 full and seven partial
batches. Settling and structural stages each process 1064 jobs; robustness
processes 4256 trials. The largest actual MCTS batch is 216 candidates.
All 1024 iterations complete without timeout or pending nodes. The selected
root action is unchanged in this single-seed comparison; additional work is not
proof of better decision quality. Pose generation separately runs 22 generator
batches, which are not counted as native simulation calls in the table.


### Larger simulation batches: call-count experiment

The [solve-call comparison](benchmarks/excavator-mcts-solve-call-batches.json)
tests larger placement/native limits through process-local benchmark overrides.
Production defaults remain **placement 32 / native 128**. All cases use the
same 1024-iteration, evaluation-256 search on the frozen nine-stone stack,
49 inventory stones, seed 0 and four CPU result workers. Candidates run
sequentially, followed by a fresh control; the earlier 174.35-second control
is also retained in the report.

| Placement / native | Native solve calls | Search seconds | Evaluated nodes/s | Distinct expanded parents/s |
| --- | ---: | ---: | ---: | ---: |
| 32 / 128 (fresh control) | 111 | 174.84 | 6.09 | 1.075 |
| 64 / 128 | 79 | 171.50 | 6.20 | 1.096 |
| 64 / 256 | 63 | 170.06 | 6.26 | 1.106 |
| 128 / 512 | 36 | 168.95 | 6.30 | 1.113 |

All cases complete 1024 iterations without timeout, evaluate 1064 candidates,
expand 188 distinct parents, and leave no pending nodes. Their ordered-tree,
native module and Python source hashes match. Thus this comparison holds search
work fixed rather than gaining speed through a different candidate allocation.
The selected action, failure count and backed-up tree values are unchanged.

The call estimates are confirmed: 64/128 makes 21 settling, 21 structural and
37 robustness calls; 64/256 makes 21 of each; 128/512 makes 12 of each. The larger
native capacity lets all four robustness trials per placement fit together.
Raising native capacity alone at placement 32 would not remove calls, since its
largest submitted robustness batch already fits in 128 jobs.

**128/512 reduces native calls by 67.6% and improves evaluated/parent throughput
by 3.5% relative to the fresh control.** Its advantage over 64/256 is only
1.11 seconds (about 0.7% throughput), so a single run does not establish it as a
robustly better default. Larger batches shorten physics calls but delay CPU
results: the 128/512 run spends 28.05 seconds processing results with 7.80 seconds
of overlap, compared with 22.54/17.80 seconds in the preceding control. These
are host wrapper/result timings; overlap is not a GPU-kernel utilization metric.
Pose generation and remaining host work also contribute to total search time.

This experiment confirms that solve-call count can fall sharply while total
throughput improves only modestly. It does not change CUDA kernels, simulation
rules, or production batch settings. Further seed/scene checks would be needed
before adopting the largest capacity as a general default.


### Independent CPU result chunks

`StackingEnvironment.result_batch_size = 32` now controls reward/observation
work independently of `CudaPlacementBackend.placement_batch_size` and its
native `batch_size`. A 128-placement GPU group can therefore feed four CPU
chunks of 32 while retaining native simulation capacity 512. The production
GPU settings remain placement 32 / native 128; no YAML fields were added.
The benchmark accepts `--result-batch-size` separately from its existing
`--placement-batch-size` and `--cuda-batch-size` overrides. Its `--serial-results`
control preserves CPU chunk sizes while disabling overlap.

One coordinator processes CPU chunks in sequence using the existing worker
pool. Global evaluator assignment spans both GPU and CPU chunk boundaries,
including single-state tails. It keeps each worker's caches isolated. Up to two
GPU result groups may be submitted, plus the physics group being produced on
the caller. This permits one extra GPU group of lookahead during CPU stalls
without an unbounded queue. Completed groups are collected in input order;
queued work is canceled and active workers drained on failure or interruption.
MCTS still receives the complete ordered batch before any backups.

This separates scheduling granularity; it does **not** expose partially finished
native simulations. Native CUDA calls still finish their whole batch before
returning, and placement/transition state construction remains on the caller.
At placement 128, the recorded MCTS batches contain at most two GPU groups, so
additional lookahead does not help that workload. Smaller CPU chunks also add
worker synchronization boundaries; they are not inherently faster.

The [independent result-chunk benchmark](benchmarks/excavator-mcts-independent-result-chunks.json)
holds the nine-stone stack, 49 inventory stones, seed 0, 1024 iterations,
evaluation256 and four CPU workers fixed:

| GPU placement / native / CPU result | Native calls | CPU chunks | Search seconds | Evaluated nodes/s |
| --- | ---: | ---: | ---: | ---: |
| 128 / 512 / 128 (whole CPU group) | 36 | 12 | 168.35 | 6.32 |
| 128 / 512 / 32 | 36 | 37 | 170.35 | 6.25 |
| 64 / 256 / 32 | 63 | 37 | 169.12 | 6.29 |

All three finish 1024 iterations, evaluate 1064 candidates, and preserve the
ordered-tree hash, including failure flags and backed-up values. The isolated
CPU chunk-size change shows **no speedup**: overlap remains about 7.8 seconds
at placement 128. The 64-placement result is close to the preceding 170.06-second
measurement and is not evidence of a meaningful improvement from this change.
These are single-seed timings. The separation is retained as an independently
controllable scheduling boundary, without increasing production GPU batch sizes.

A final run of the production defaults, without batch overrides, completed in
**174.55 seconds** at **6.10 evaluated nodes/s**, with 111 native calls and 37 CPU
result chunks. It retains the control's ordered-tree hash and completes all 1024
iterations. The preceding 174.84-second control is effectively the same timing.

Validation: **300 Python tests passed**, including both platforms, MCTS,
restart/artifacts, native placement parity, and visualization. New coverage
checks bounded lookahead while a CPU worker is blocked, independent chunk sizes,
global worker assignment, order and cache isolation. CUDA parity is checked for
GPU chunks equal to and larger than the CPU chunk, including one whole GPU
batch. Existing failure tests allow the bounded extra lookahead and verify
worker draining and subsequent reuse.


### Earlier CPU placement finalization experiment (reverted)

An experiment moved trajectory, diagnostic and branch-state construction onto
the CPU coordinator, ahead of reward/observation processing. It preserved the
ordered search tree and full trajectories but showed no throughput improvement,
so its implementation, benchmark switch and associated tests were reverted.
Placement/state finalization runs on the caller again; the existing parallel
reward/observation pipeline and independent CPU result chunks remain enabled.

The retained [benchmark results](benchmarks/excavator-mcts-early-finalization.json)
used a nine-stone stack, 49 inventory stones, seed 0, 1024 iterations,
evaluation256, placement32/native128, CPU chunks32 and four result workers:

| Finalization location | Search seconds | Evaluated nodes/s | Finalization seconds | Overlap with native calls |
| --- | ---: | ---: | ---: | ---: |
| Caller control | 168.11 | 6.329 | 13.98 | 0.00 |
| CPU coordinator (reverted) | 169.25 | 6.287 | 14.48 | 13.15 |

Both runs evaluated 1064 candidates (587 nonfailed), expanded 188 distinct
parents and made 111 native calls, with identical tree hashes, selected rewards
and target IoU. These are historical experiment timings, not measurements of
the restored implementation: both variants used the experiment's refactored
state construction. The finalization timer covered 1032 short-evaluation
candidates; 32 root validations remained synchronous.

Settling wrapper time increased from 5.63 to 18.68 seconds while structural and
robustness timings remained nearly unchanged. Simulated steps and returned
sample counts matched exactly. The 13.15 seconds of finalization overlap was
offset by about 13.02 seconds of longer native wrapper intervals. These host
wall timings do not distinguish host processing/contention, GIL reacquisition
and actual GPU execution. This single pair demonstrated no overall speedup.

Compact trajectory storage is implemented separately below. Containment uses
final poses, geometry and voxel resolution, which are preserved.



### Compact CUDA samples and trajectory histories

Native simulation retains each job/phase's complete sampled poses, motion and
mobility in owned numerical buffers. Final simulation snapshots remain eager
and unchanged. Historical `sample.snapshot` values are reconstructed on demand;
`result.sample_poses(entity_id)` and `result.sample_times` provide bulk FP64 arrays
with the same canonical normalization as the original snapshots. Native C++
consumers use `sample.snapshot()` instead of the former snapshot field.

CUDA placement reads these arrays in bulk. `StackObjectTrajectory.from_samples`
validates and owns a dense pose array; branch copies copy that array without
constructing a Python array for every timestep. Accessing `trajectory.poses`
materializes the usual mutable list, preserving viewer and checkpoint contracts.
All samples and timestamps remain available. Older native installations use the
existing snapshot path. There are no new YAML settings, precision changes,
sample reductions, or changes to containment geometry/resolution. The earlier
CPU finalization handoff remains reverted.

The [matched benchmark](benchmarks/excavator-mcts-compact-samples.json) uses the
same nine-stone starting stack, 49 inventory stones, seed 0, 1024 iterations,
evaluation256, placement32/native128, CPU chunks32 and four result workers:

| Sample/history representation | Search seconds | Evaluated nodes/s | Expanded parents/s |
| --- | ---: | ---: | ---: |
| Original snapshots/lists | 175.74 | 6.054 | 1.070 |
| Compact native/Python storage | 164.30 | 6.476 | 1.144 |

This pair improves evaluated-node throughput by **7.0%**, saving 11.44 seconds
(6.5% of search time). Both runs evaluate 1064 candidates, have 587 nonfailed
evaluations and 188 distinct expanded parents, make 111 native calls, execute
770316 simulated steps, and retain 171713 native samples. Ordered tree hashes,
selected poses, rewards and selected target IoU match exactly. Native wrapper
time stays near 42 seconds; the measured gain appears in surrounding host work,
rather than demonstrating faster GPU kernels. This is a single-seed comparison,
not a general speedup guarantee.

Validation includes **277 Python regression tests**, the native simulator CTest,
and 80 pose/motion/time/metric arrays compared exactly between old and new
binaries. Tests cover bulk versus reconstructed snapshots, independent array
ownership after simulator reuse/destruction, branch copies, pickle, mutable
pose access, saved checkpoints, both platforms, MCTS and viewer projection.
A final optional-untimed-history copy check also passes; that compatibility guard
does not affect the timed histories used in the benchmark.


### Vectorized floor sampling

A timer-only profile of the compact-sample workload put 48.0 of the 80.1-second
pose-generation phase in CPU proposal sampling, versus 24.6 seconds in CUDA pose
solving. Diversity selection drew 32 proposals per call, of which 1.07 were used
on average, using per-dict Python loops; score maps evaluated about 1000 anchor
cells one at a time.

Diversity selection now works on candidate-index arrays. Its pools, their order
and every `rng.choice(n, p=...)` call are unchanged, so seeded draws are identical.
Score maps evaluate anchor footprints, gap labels and candidate scores as arrays,
with the same arithmetic order; each map cell keeps its first maximum-score
candidate. Kernel shifts use one padded gather. Gap components are reused per
floor occupancy grid and excavator distances per score grid, both evaluated with
the original scalar code. There are no configuration, RNG or native changes.

The [vectorized-sampling benchmark](benchmarks/excavator-mcts-vectorized-sampling.json)
repeats the compact-sample workload with the same native module:

| Measurement | Compact samples | Vectorized sampling |
| --- | ---: | ---: |
| Search seconds | 164.30 | 134.94 |
| Evaluated nodes/s | 6.476 | 7.885 |
| Nonfailed nodes/s | 3.573 | 4.350 |
| Pose-generation phase | 80.08 s | 52.45 s |

Evaluated-node throughput improves by **21.8%**. Evaluation/failure counts, the
ordered tree hash and the selected candidate match exactly. In the profiled pair,
selection fell from 16.87 to 2.11 seconds and score-map kernels from 18.17 to
6.98 seconds. A separate full search compared every call with the original code:
2379 score maps (2.63 million candidates) matched array-for-array and
field-for-field, and 2469 selections matched chosen candidates and final RNG state.
Selection consumes one seeded RNG in sequence, so it is not parallelized. This is
a single-seed, single-scene timing.


### CUDA pose-generation call cost

Native CUDA pose generation took 23.5 seconds in 140 calls. An Nsight trace of
those calls showed the GPU busy for 22.4 seconds: each call runs roughly its
trust-region iteration limit, and each iteration is a chain of per-job serial
kernels whose duration barely depends on the number of jobs. Blocking progress
readbacks left only about 1.7 seconds of GPU idle time.

Four changes reduce that cost without changing any result:

- `CudaPlacementPoseSolver.batch_size` is 256, so a full 256-node generation
  batch runs as one chunk per schedule stage instead of two. Capacity 512 gave
  no further gain because batches never exceed 256 jobs.
- The stacking-core `queries` kernel uses one warp per job instead of one
  thread. Ballot ranks assign query slots and queue entries in the serial order.
- Support, contact and Hausdorff queues drain in one launch, so a job's long
  Hausdorff query overlaps its contact queries.
- Once at most 64 force jobs remain, one tail launch runs all remaining sweeps.
  Each block stops at its own convergence or iteration limit, so the per-32-sweep
  progress readbacks are no longer needed.

All 26 generation calls of the benchmark search (2627 jobs) were captured and
replayed after each change. Every returned pose, cost, status and diagnostic
hashed identically. Native time fell from 23.47 to 15.66 seconds; kernel time
from 22.37 to 14.93 seconds; blocking readbacks from 62320 to 16333.

The [native pose-generation benchmark](benchmarks/excavator-mcts-posegen-native.json)
repeats the vectorized-sampling workload:

| Measurement | Vectorized sampling | Native call changes |
| --- | ---: | ---: |
| Search seconds | 134.94 | 126.05 |
| Evaluated nodes/s | 7.885 | 8.441 |
| Nonfailed nodes/s | 4.350 | 4.657 |
| Pose-generation phase | 52.45 s | 43.86 s |

Evaluated-node throughput improves by **7.1%**, with the same ordered tree hash,
counts and selected candidate. The remaining native time is mainly per-job ADMM
force sweeps (8.6 seconds) and single-warp Hausdorff queries; faster versions
would need kernel redesigns with their own parity work. Rebuild and reinstall
stacking-core to obtain the native changes; the 256-job workspace reserves about
1.45 GiB of GPU memory.


### Repeated host work in pose generation

After the native changes, the CPU side of pose generation still repeated work per
state that depends only on placed poses or fixed grids:

- Deferred observations rendered height maps one triangle at a time. The common
  renderer now evaluates every (triangle, cell) height with the same float64
  formulas and combines them with one scatter-max before rounding to float32.
  Rounding is monotonic, so this equals the former per-triangle float32 maxima.
- Score-grid height maps are resized by row-slab then column-slab maxima of finite
  values, instead of one Python block per output cell.
- Sampler yaw/base orientation sets are reused per object.
- The active-floor policy builds its grid once and reuses per-(object, exact pose)
  entries and footprint masks; the PLY-height initializer reuses placed-stone
  bounds the same way. Child states repeat almost all of their parents' poses.

Full searches compared every call with the previous implementation: 203 height
maps, 1041 resizes, 4918 orientation sets, 2023 floor contexts, 2156 floor
rejections and 1045 initializer scenes were exactly equal. Height-map rendering
fell from 4.65 to 0.27 seconds and resizing from 3.33 to 0.17 seconds.

The [host-work benchmark](benchmarks/excavator-mcts-sampling-host-work.json)
repeats the native pose-generation workload:

| Measurement | Native call changes | Host-work reuse |
| --- | ---: | ---: |
| Search seconds | 126.05 | 114.72 |
| Evaluated nodes/s | 8.441 | 9.275 |
| Nonfailed nodes/s | 4.657 | 5.117 |
| Pose-generation phase | 43.86 s | 32.69 s |

Evaluated-node throughput improves by **9.9%**, with the same ordered tree hash,
counts and selected candidate. Native pose solving (15.5 s), score-map kernels
and prior containment now dominate the remaining pose-generation time.


### Cell-restricted support and lattice voxel crops

Score maps computed support, contact and placement heights for every grid cell
and orientation but read them only at kept anchor cells.
`support_likelihood_maps_by_kernel(..., cells=(rows, columns))` now evaluates
only those cells with the same arithmetic. On 200 captured score-map inputs this
reduced the cost from 2.47 to 1.51 ms per map with identical outputs.

`VoxelOverlapEvaluator` cropped the target voxel grid with three comparisons over
all voxel centers for every uncached prior or reward query. The centers form a
filtered lattice, so the crop now reads one lattice sub-block; its kept rows equal
the former mask in the same order, and containment receives an identical point
array. Across a full search, 11046 crops matched exactly; crop time summed over
the coordinator and reward workers fell from 45.5 to 5.9 seconds.

The [benchmark](benchmarks/excavator-mcts-cell-support-and-voxel-crop.json)
repeats the host-work workload:

| Measurement | Host-work reuse | Cell support and lattice crops |
| --- | ---: | ---: |
| Search seconds | 114.72 | 108.53 |
| Evaluated nodes/s | 9.275 | 9.804 |
| Nonfailed nodes/s | 5.117 | 5.409 |
| Pose-generation phase | 32.69 s | 29.23 s |
| Short evaluation batches | 68.00 s | 65.93 s |

Evaluated-node throughput improves by **5.7%** with the same ordered tree hash,
counts and selected candidate. A full-search comparison also matched all 2379
score maps and 2469 selections with the previous code.


### State copies and simulation chunk size

After the pose-generation work, short evaluation batches took 65.9 seconds. Native
simulation accounted for 40.5 seconds; state copies for another 20.3 seconds on
the coordinator. Each evaluated candidate's history was duplicated four times:
when its child state was built, by the transition snapshot, by
`TransitionEvaluation`, and by the MCTS node. Every copy re-copied the checkpoint
trajectories pose by pose and deep-copied every action's diagnostics twice.

- `Action.copy` now copies the pose arrays and deep-copies diagnostics once,
  without reconstructing and revalidating the action.
- Copying a list-based trajectory returns the packed form, so later branch copies
  copy one array. Reading `poses` still gives the usual mutable list.
- Batched transition steps return each newly built child directly and keep one
  independent final snapshot per chunk.
- Short MCTS evaluations hand the evaluation's private state to the node
  (`set_result(..., owned=True)`). Authoritative validation, whose evaluations are
  returned to callers, still copies.

With CPU result processing cheaper, larger simulation chunks now win:
`CudaPlacementBackend` uses 128-placement chunks and native capacity 512 (four
robustness trials per placement). Each native call costs roughly its step limit
regardless of job count; 256-placement chunks lose the CPU/GPU overlap. Peak GPU
memory with native capacity 1024 was about 3.2 GiB.

The [benchmark](benchmarks/excavator-mcts-state-copies-and-chunks.json) uses the
same workload:

| Step | Search seconds | Evaluated nodes/s | Short batches |
| --- | ---: | ---: | ---: |
| Cell support and lattice crops | 108.53 | 9.804 | 65.93 s |
| Cheaper action and trajectory copies | 88.16 | 12.068 | 51.90 s |
| No per-candidate snapshot or node re-copy | 82.78 | 12.854 | 48.86 s |
| Placement 64 / native 256 | 73.84 | 14.410 | 40.57 s |
| Placement 128 / native 512 | 70.43 | 15.108 | 37.18 s |
| Placement 256 / native 1024 | 71.44 | 14.893 | 37.64 s |
| **Production defaults (128 / 512)** | **71.12** | **14.961** | **37.66 s** |

Every row has the same ordered tree hash, counts and selected candidate. Seeds 1
and 2 also kept identical trees per seed and improved from 13.47 to 15.68 and
from 13.58 to 15.66 evaluated nodes/s with the larger chunks. The CPU profile
(`mcts_cpu.yml`, 512 iterations) reproduces its recorded tree exactly and now
takes 32.79 s instead of 60.13 s. With the larger chunks, native simulation
itself falls from 39.9 to 24.2 seconds (settling 2.9, structural 14.0, robustness
7.2); the rest of the 37.7-second short evaluation is host work, including
reward/observation processing that now overlaps less with the GPU.


### Wide descendant validation rounds

With `max_depth: 4`, mixed final selection, 32 validation candidates and
descendant validation depth 3, final long validation took 84 seconds of a
156-second search. Nearly half of the validated descendants failed, and each
failure started another single-child round: 72 rounds averaging 6.7 jobs.

Each walk now submits all of a node's not-yet-validated children in one round
and then takes the first passing child in the same order. Walks of every root
candidate share rounds. Evaluations are independent, so the selected candidate is
the one a one-child-at-a-time walk finds; extra siblings carry long-fidelity
results, so tree hashes change.

The [benchmark](benchmarks/excavator-mcts-wide-descendant-validation.json) used
that profile, seed 0 and the nine-stone state:

| Children per round | Search seconds | Long rounds | Long jobs | Long-validation seconds |
| --- | ---: | ---: | ---: | ---: |
| 1 (previous) | 155.66 | 72 | 480 | 84.04 |
| 2 | 134.75 | 40 | 495 | 64.32 |
| 4 | 124.08 | 25 | 523 | 52.57 |
| All (implemented) | 124.66 | 18 | 691 | 52.72 |

All runs selected the same candidate. Long validation still exceeds the
30-second `time_reserve`; a search that uses its whole iteration budget can time
out validation, which rejects candidates when validated descendants are required.


## Run

From the `stacking-apps` checkout, with the sibling `stacking-core` checkout:

```bash
make install-excavator
stacking-excavator-plan \
  --repository "$PWD" \
  --pick-pose-data /path/to/pick.json \
  --save-run /tmp/excavator-plan.stack \
  --steps 1 --seed 0 --max-run-time 600
```

The native rebuild includes `configure_excavator_ik`; an older installed core
without this binding cannot construct the excavator planner. The editable
installation reads assets and profiles from the repository specified by
`--repository`. Source-checkout invocation is also available:

```bash
PYTHONPATH=packages/stacking-common/src:apps/excavator/src \
  python -m stacking_excavator.planning.cli --help
```

Excavator YAML angle settings use explicit `_degrees` fields, including
`robot.home_positions_degrees`, gripper opening offsets/limits,
`spin_step_degrees`, `target_rot_tol_degrees`, and
`max_handoff_orientation_distance_degrees`. The loaders convert these to radians
once; runtime objects and saved-run contexts retain their canonical SI values.
Legacy unsuffixed radian fields remain supported, but specifying both forms of
the same field is an error. The simulation device uses the same planning-profile
loader as the planner.

Transforms such as `world_T_stack` remain `[x, y, z, qx, qy, qz, qw]` quaternion
poses. Direction vectors, optimizer radii, weights and mixed pose-error tolerances
are not degree-valued angles. Perception's opening limits and measured-angle
difference use degrees; its calibrated polynomial coefficients remain unchanged
and produce radians.

Pickup JSON maps catalog IDs to world-frame SI poses using XYZW quaternions:

```json
{"55": [2.0, -4.0, 0.4, 0.0, 0.0, 0.0, 1.0]}
```

Values may alternatively be `[position_xyz, quaternion_xyzw]`. Existing
trusted local `pose_data.pkl` mappings are supported explicitly. No legacy
planner classes or repositories are imported at runtime. `--object-ids`
selects a subset of the recorded scene. Requested IDs missing from the scene
produce a warning and are skipped. If none match, the command exits normally
without creating a run; omit `--object-ids` to use all recorded objects.

`--search-only` saves placement search without invoking robot motion planning.
With the default `pose_from_scan: false`, pickup data is optional: the run uses
all objects from `inventory.yml`, or the catalog subset passed with
`--object-ids`. Initial orientations come from the assets' principal axes.
The saved context labels its identity inventory poses as synthetic search
initialization, not measured pickup poses. Motion planning and
`pose_from_scan: true` still require `--pick-pose-data` for new runs.

Override `orientation.pose_from_scan` from `action_generation.yml` explicitly:

- `--pose-from-scan`: initialize MCTS orientations from the recorded stone
  orientations and enable the scan-orientation filter.
- `--no-pose-from-scan`: use principal-axis orientations and configured rolls.
- Omit both flags to use the YAML value. The startup log prints the effective
  value and whether it came from the profile, CLI override, or saved run.

Both modes use recorded pickup poses for motion planning. This switch changes
placement orientation generation/filtering; it does not activate live scanning
or disable the pickup input. In **New planning run**, **Pose from scan override**
offers **Use configuration (no override)** (the default), **true**, and **false**.
Use configuration follows the YAML value without passing an override flag;
true/false passes your selection as a CLI override without changing the
configuration file. Resume displays the recorded value and preserves it;
start a new run to choose a different value.

The planning dialog also offers **Overwrite existing run on successful
completion**. Check it to reuse an existing `.stack` filename. Planning writes
to a separate `*-pending-*.stack` file and replaces the chosen destination only
after successful completion. Failure or Stop preserves the old file and leaves
the partial result available through **Open resulting run** and the logged path.
Resume can replace a different output file, but cannot overwrite its source.
CLI output paths still must be new.

For example, append `--pose-from-scan` to a planning command with
`--pick-pose-data /path/to/pose_data.pkl` and a new `--save-run` destination.

The grasp solver's `trust_region.radius_init` sets the initial optimization-step
radius, while `radius_max` caps its growth. The step combines translation,
rotation and gripper opening, so these are not a workspace radius or a pure
distance in metres. The solver shrinks the radius when its local approximation
predicts improvement poorly and may grow it when prediction is good. Require
`0 < radius_init <= radius_max`; the excavator profile currently uses `0.1`
for both. An inconsistent explicit pair is rejected before search begins.

```bash
stacking-excavator-plan --repository . \
  --config apps/excavator/config/mcts_cuda.yml \
  --search-only --steps 10 --save-run /tmp/excavator-search3.stack
```

`--iterations`, `--max-search-time`, and `--config` (MCTS YAML) bound or replace
search settings. Small time limits can expire during initial CEM generation.
`--max-run-time` limits the entire worker, including native grasp/motion work.
Ctrl+C and wall expiry terminate its dedicated process group and preserve
committed `.stack` records, marking the run cancelled.

The CLI uses the same terminal reporter as the tabletop MCTS debug runner.
It prints the resolved workload, `[Progress]` phase messages and 10-second
heartbeats during long calls, `[INFO]` CEM/tree/validation diagnostics, and
`[Timing]` phase and total durations. Search-only runs finish with the saved
run path, discounted return, target IoU, and completed placement count.
Motion runs also report candidate attempts and native planning phases.
Search-only `.stack` records contain the same detailed search statistics as
tabletop. This output also streams into the Operations Console planning log.

Resume a search-only or motion-planning run from its last successfully committed placement:

```bash
stacking-excavator-plan --repository "$PWD" \
  --resume-run /tmp/excavator-plan.stack \
  --save-run /tmp/excavator-continued.stack --steps 3 --max-run-time 1800
```

Supply the same `--config` if the original run used a custom MCTS profile.
Recorded seed and search budgets are reused. Resume keeps search inputs
immutable: omit `--seed`, `--iterations`, and `--max-search-time`, or repeat
their recorded effective values. Changed values fail before creating the new
run. `--steps` and `--max-run-time` may change. Start a new run to use different
search inputs; the retained plans remain bound to their original context.
Already-unit pickup quaternions are preserved when copying/loading checkpoints.
For older excavator runs, resume restores the exact context poses after checking
that checkpoint differences are limited to quaternion normalization roundoff
(eight machine epsilons). Position, scan provenance, and larger orientation
changes still fail validation; the original artifact is not modified.

Resume validates resolved profiles, input poses, native binary, model hashes,
frames, and joint order. It creates a new document containing the committed
prefix; the source is unchanged. Resume restores the original search-only or
motion-planning mode automatically. Do not supply new pickup poses or object
IDs. `--steps` specifies **additional** steps: after nine committed steps,
`--steps 21` requests continuation from step 10 through step 30. The run-time
limit is a total wall-clock limit for this new invocation, not a per-step limit.

Search-only runs now commit the full selected search state and its checkpoint
in the same transaction as each completed step. A timeout during the next search
can therefore resume from that checkpoint. The GUI preserves search-only mode
in its disabled mode checkbox when resuming. A failed first placement has no
selected state to resume. Restart rebuilds the search tree and observation;
it does not restore worker processes or promise an identical random search path.

**Older search-only files:** versions before this fix saved candidate and
visualization records without checkpoints. Those files can show completed
steps but report `missing planning_checkpoint/N` when attempting to resume.
The console now explains this limitation directly. They remain viewable, but
require starting a new run; the missing full state is not reconstructed from
viewer poses. A saved checkpoint records simulated planning state, not
physical execution.

## Operations Console

In **Plan → Planning attempt → Step**, select the placement whose robot motion
you want to replay. Each entry states whether motion was recorded. Select an
attempt, then use **Play** or scrub the timeline. The step selection is shared
with Search Inspector; changing steps stops playback and resets its timeline.

Both operations consoles offer the same **Objects: DSF surface / PLY mesh**
toolbar selector, available in Plan and Search Inspector. DSF is the default:
the renderer samples each component's curved support surface using its stored
nodes and sharpness, rather than displaying the OBJ's node-hull triangles.
PLY mode shows the corresponding reconstructed stone mesh. Both modes are
cached and preserve the camera when switching, including during replay.
All 49 excavator stones have matched PLY assets with source hashes.

Plan's robot options match tabletop: **Visual meshes**, **Collision meshes**,
and **Visual + collision wireframe**. DSF collision assets use the same curved
surface tessellation. For excavator, the available collision geometry is the
gripper geometry recorded in the commissioned model. Candidate visibility,
pose selection, contacts, forces, environment layers, and video controls match
the tabletop Search Inspector. The object surface selector also applies to
trajectory video exports.

Candidate lighting and transparency use the same policy in both consoles:
rough diffuse material (ambient 0.42, diffuse 0.58, specular 0), a fixed key
light at elevation 45°/azimuth -35° with intensity 0.8, and smooth outward
surface normals. Translucent stones cull their rear faces so inward-facing
surfaces do not show through as reversed lighting. Target/environment shells
retain their own two-sided rendering.

Scenes with transparent candidates, target shells, or environment layers use
depth peeling with multisampling disabled and no fixed peel-layer limit.
SSAO is enabled only for fully opaque scenes: its VTK delegate pass bypasses
the standard depth-peeling path. The platform-sized values remain distinct:
SSAO radius is 0.03 m for tabletop and 0.3 m for excavator; force-arrow length
is 0.035 m and 0.35 m respectively. These are display dimensions, not changes
to the collision geometry or solver configuration.

```bash
make install-excavator-operations
stacking-excavator-operations . --run /path/to/excavator-plan.stack
```

Omit `--run` to open the console before planning. From an uninstalled checkout:

```bash
PYTHONPATH=packages/stacking-common/src:apps/excavator/src \
  python -m stacking_excavator.gui.operations.application .
```

The window contains **Plan** and **Search Inspector** tabs. The toolbar opens
recorded runs, starts new planning, resumes a checkpoint into a new document,
reloads incremental results, and exports JSON/CSV summaries. The planning
dialog accepts recorded pickup poses, an MCTS YAML profile, additional step
count, run time limit, and an optional search-only mode. It streams subprocess
output without blocking the window. Stop terminates the CLI's isolated worker
through its supervisor; the resulting cancelled document remains inspectable.

Plan shows every recorded attempt, including failed and partially completed
steps. Play/pause and the timeline replay a whole step or a selected segment.
The viewport projects the commissioned VDK23 and stone-grab URDFs, recorded
joint coordinates, signed jaw openings, and carried-object target tracks.
Pick, handoff, and place phases use their recorded support planes and scenes.
The target volume uses the saved stack-to-world transform. Visual meshes,
gripper collisions, and a combined view are available. Failed attempts without
trajectories show their input scene and the robot at the recorded profile's
home coordinates.

Replay restores the run's planning profile and verifies mesh/URDF content
hashes. Current solver tuning does not replace saved values. Missing object
tracks remain unknown; faint pickup/placement ghosts provide reference poses.
Untimed paths use display-only playback timing. Recorded opening samples are
used when present; otherwise the profile opening is the display fallback.
The source robot URDF has no body collision meshes, so collision mode renders
the gripper. Search Inspector retains candidate selection, pose modes,
simulation timelines, contact/force layers, and trajectory MP4 export.

This is an offline console. Hardware execution and operator control of the
excavator remain outside this application.

## Ownership and behavior

Generic profile schemas, OBJ decoding, height-map rendering, search composition,
native result conversion, context hashing, and process supervision live in
`stacking-common`. Tabletop keeps its existing import names through small
aliases and supplies its original geometry policies. Physical YAML values
remain independently owned by each application.

The excavator app owns:

- The six-joint VDK23 chain and an owned native analytic IK seed callback.
  The calibrated bucket/wrist zero orientations are accepted alongside the
  original native reference geometry; generic IK refines the seed.
- The stone-grab floating root, independent jaw coordinates, and eight declared
  pad/tooth contact geometries. Both opening coefficients are positive because
  the URDF joint axes oppose one another.
- Separate pickup, stack-placement, and handoff scenes with explicit ground
  planes and configured fixed boxes. Remaining pickup objects are pickup
  obstacles; committed stack objects are placement/handoff obstacles.
- Stack-to-world conversion and the 0.20 m production placement lift. Motion
  targets use the lift once; successful commits retain the settled local pose.
- The nine ordered legacy handoff locations. Only retryable failures that can
  depend on handoff location advance to the next location. Fixed pickup or
  placement grasp failures return to candidate selection.
- Per-candidate fixed-scene resettling, common MCTS rejection feedback, attempt
  journaling, and atomic selected-step/checkpoint persistence.

The source IK URDF contains robot visuals but **no robot collision geometry**.
The migrated motion model preserves that scope: gripper and carried-object
collisions are represented. Full excavator-body collision validation is not
established by this delivery.

## Source-to-profile mapping

| Source | Migrated boundary |
| --- | --- |
| `agent/configs/config.yml`: production Q network disabled | Heuristic prior/value bootstrap; no Torch/Ray/model loader |
| `agent/configs/ablation.yml`: MCTS depth 3 after 10 placements, 200 iterations, 240 seconds; CEM 96/24/3/12 | `config/mcts_cpu.yml` / `config/mcts_cuda.yml`; shared tree/search/sequence implementation |
| Principal axes, rolls 0/180°, yaws 0/90/180/270°, scan mode false | Common principal-axis provider and platform orientation selection |
| Production catalog/density 1000; ablation target 2 × 3.5 × 2.5 m, 5° taper, origin (0, 0, 0) | `inventory.yml`; asset mass and inertia multiplied by density once |
| Observation, rewards, fast settling/full structural phases | `environment.yml`, `solvers/simulation.yml` |
| Native posegen setup, diffsim defaults, and `PoseSolver` cosine schedule | `solvers/posegen.yml`, including platform-owned `schedule.objective_weights` |
| `planning/planning.py` grasp and motion setup | `solvers/grasp.yml`, `solvers/motion.yml`; all supported native fields materialized |
| `scripts/desktop/generate_sequence.py` frame/home/lift constants | `planning.yml`; world stack offset (-6, 3), production lift 0.20 m |
| Recorded July 1 plan | Fixture preserves its different 0.10 m lift and support heights; never substitutes current defaults |

The excavator MCTS profile follows the local `stacking-planner`
`agent/configs/ablation.yml`: depth 1 for the first 10 placements, then depth 3;
200 iterations, 240 seconds, 16 children, 8 validation candidates, and CEM
population/elite/iterations/keep of 96/24/3/12 with 15-degree yaw mutation.
Final selection uses `max_descendant` with weight 1.0; descendant validation
is disabled and its configured depth is 0. Simulation depth resolves to the
legacy Python default of 5. The local native worker allocation remains 12.
Action sampling and build-time filtering also follow ablation, as detailed
below. Reward weights, settling budgets, and supported grasp/motion settings
remain independently resolved in their respective profiles.
The target geometry also follows ablation: width 2 m, length 3.5 m, height
2.5 m, 5-degree taper on all four sides, and fixed stack-local origin
(0, 0, 0). The stack-to-world transform remains in `planning.yml`.
Ablation's synthetic inventory and other environment settings remain separate
from this MCTS, action-policy, and target alignment. Existing `.stack` files retain their
recorded target geometry; the updated target applies to newly planned runs.
The placement count is still controlled by `--steps` (default 1), independently
of MCTS depth. For example, `--steps 20` allows the run to reach depth 3 after
the initial 10 placements, if valid placements remain available.

### Action sampling and filtering

`action_generation.yml` owns the excavator's legacy score-mode settings:
orientation-aware hull support kernels, max-downsampled and smoothed height
maps, connected floor proposals, taper-aware anchor bounds, seven score terms,
and representative/elite/Boltzmann selection. Temperature is 0.3, elite fraction
is 0.25, excavator-distance weight is 2.0, and support contact-gap tolerance is
0.10 m. The distance origin defaults to stack-local (0, 0), as in ablation;
`scored_sampling.score.excavator_xy` can supply the reference position that the
legacy sequence entrypoint set from its target offset.

Fresh populations use private seeded Python and NumPy RNGs, balanced per-object
allocation, and the legacy first-placement shuffle/later-floor priority. They
do not perturb tabletop's random state. CEM yaw mutations retain object-specific
score lookup. End-to-end search trees need not be identical: the native solver,
parallel scheduling, inventory, and simulation profiles also influence results.

Initial height comes from `model_<id>_mesh.ply`, with hull fallback when absent.
Both footprints receive 0.12 m padding and the pose receives 0.15 m clearance.
PLY assets used by planning are included in saved-context hashes. The build gate
checks XY target overlap, gap residual <= 0.005, and equilibrium residual < 1.0;
native convergence flags are diagnostic rather than an additional rejection.
Volumetric containment remains part of simulation/reward evaluation. Placed
stones farther than 0.5 m from the target's XY rectangle become posegen boundary
bodies. No catalog stones are banned by default.

The active ablation score path bypasses the optional large-stone/pre-pose height
filters and disables mesh-face fitting, U-shape rejection, and hard support or
stacked-floor rejection. This migration preserves those active choices. It does
not enable those inactive legacy modes. Tabletop keeps its deterministic sampler,
mesh-height initialization, convergence check, and pre-simulation containment
filter. Regression fixtures capture legacy scores and seeded proposals on both
empty and occupied scenes; separate tests cover filter boundaries and native
posegen boundary-body selection.
CLI flags override YAML: the earlier `--iterations 10 --max-search-time 120`
example still requests 10 iterations and 120 seconds. Omit those flags to use
the profile defaults; `--config` replaces only the MCTS profile.

The pose solver now uses the excavator schedule instead of the inherited
tabletop weights. In `schedule.objective_weights`, each entry is a sum of its
coefficients times the named bases: `constant = 1`,
`ramp = (1 - cos(pi*t))/2`, `anti_ramp = 1 - ramp`, `sine = sin(pi*t)`, and
`gap_feedback = t * previous_c_gap`. The wrench coefficient multiplies
`diag(1, 1, 1, 10, 10, 10)`. Scheduled weights override the corresponding
`objective` values at each solve stage. The excavator target weight decreases
from 5 to 0; complementarity increases from 0 to 10. Tabletop retains its
existing schedule in its own profile. Profiles without explicit schedule
weights retain the previous shared schedule for compatibility.

Posegen force-solver tolerances are 0.001 absolute and 0.0001 relative, matching
diffsim. Grasp seed retreat is 0.75 m, resolved from the legacy 1.5 m gripper
height times its 0.5 surface-grasp offset scale. Motion approach distance
remains 1.5 m; it is a separate setting.

Some controls have no direct legacy equivalent: MCTS evaluation batching,
validation time reserve/state-drift limit, yaw deduplication, and native
trajectory candidate limits retain the migration's values. Legacy grasp
score/orientation rejection thresholds are not all exposed by the current
shared/native adapters. Matching supported configuration values does not establish
complete planner behavior parity.

The modern core deliberately removed diffsim's mass-normalized artificial
drag. Legacy solver `damping` is not physical damping and is not copied into
that field. The app uses physical damping 0 and the current native numerical
constraint-scale reference 0.001. Therefore matching input poses and profiles
does not imply identical old simulation trajectories or candidate rankings.

The active ablation sampling path now has legacy score-map and seeded-proposal
fixtures, plus a successful two-placement native CLI run. Longer multi-seed
searches, including the depth-3 phase after 10 placements, still need acceptance
coverage. Reward and simulation behavior also need comparison independently of
sampling: for example, the current place-stability weight is 0.5 while ablation
specifies 0.2. CNN scoring, training, perception/reconstruction, alternate
attachments, inactive legacy action modes, and ROS/hardware execution remain
outside this offline delivery. This is not a full legacy numerical-parity claim.

### Motion recovery

The excavator motion profile enables `joint_grasp_recovery: true` and
`scene_approach: true` for both `direct` and `regrasp`. Core defaults remain
false, preserving the tabletop policy.

- Failed endpoint IK triggers grasp-and-joint refinement. It first attempts
  the fixed pickup/placement endpoint, then jointly checks the target-relative
  grasp at both endpoints. Recovered joints and grasp openings feed motion
  planning. Subgoal IK uses the legacy home/goal midpoint and retains the goal
  branch for joints spanning a full rotation.
- Grasp sampling starts at `retreat_distance: 0.75` metres. An undersized pool
  is supplemented at `fallback_retreat_distances: [0.975, 1.2, 1.8]`, corresponding
  to the legacy stone_grab height factors. Direct planning fills a pool of
  `fallback_min_candidates: 8` reachable grasps in bounded batches. Independent
  pickup/placement grasp generation also uses these fallback offsets.
- Approach direction follows scene plane normals and separation gradients
  from nearby objects. `minimum_approach_up_component: 0.5` maintains at least
  30 degrees of elevation. Approach/retreat distance retries use factors
  `[1, 0.75, 0.5, 0.25]`.
- After grasp and standoff retries, a handoff-dependent failure may advance to
  the next configured handoff location. Fixed pickup/placement IK or motion
  failures, fixed placement grasp-generation failures, and invalid inputs
  stop that XY retry loop. Handoff target-boundary mismatches are retryable and
  report achieved position/orientation errors. Total timing sums all handoff
  attempts; each attempt retains its own diagnostics.

Collision checks and the 2 cm / 10 degree target-boundary gate still apply.
Legacy sometimes accepts an approximate subgoal IK result; this implementation
retains the core's strict waypoint IK checks. Recovery increases the amount of
work per placement and does not guarantee a feasible trajectory. See
[the recorded-case comparison](excavator-motion-legacy-comparison.md).

## Validation and replay data

The tabletop/common gate passed 155 focused tests before the excavator package
was introduced. The implementation then passed the complete Python suite;
the excavator tests additionally exercise native search on recorded objects,
frame/lift semantics, phase scenes, contact mappings, failure feedback,
context-based resume, and an actual seeded direct-motion solve. Native core
coverage checks binding ownership/refinement and the original C++ IK oracle.
Verification after console integration: 558 application tests (plus 9 subtests), 15 native Python
binding tests, and 3 focused native C++ tests passed. The application suite ran
with `QT_QPA_PLATFORM=offscreen` and pytest available through
`PYTHONPATH=/tmp/stacking-apps-test-deps`.

A bounded native search selected a validated placement from recorded objects
56, 58, and 8. A bounded CLI motion run reached grasp generation and preserved
two infeasible attempts before exhausting candidates. The seeded direct-motion
test isolates trajectory solving; it is not a grasp-quality test. Successful
fresh grasp generation and end-to-end regrasp on the recorded production scene
are not yet demonstrated by these smoke runs.

Common motion segments now retain optional sample-aligned target poses and an
explicit pick/handoff/place/transfer phase, alongside joints, openings,
attachment mode, and grasp events. Recorded direct and regrasp trajectories
round-trip through these contracts. Missing target samples stay unknown;
timestamps are preserved only when supplied. Old schema-v1 records remain
readable and keep their old encoding when the new fields are absent.
The excavator console consumes these records using shared headless selection
and URDF projection. Qt widgets and platform appearance remain owned by the
applications. Direct/regrasp fixture tests cover target interpolation, jaw
mapping, partial-step inspection, geometry checks, and actual planning-worker
cancellation; a virtual-display smoke test verifies 3D rendering.
