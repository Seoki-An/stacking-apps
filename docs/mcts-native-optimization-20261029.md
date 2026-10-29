# Native simulation acceleration — 2026-10-29

The two native changes reduce the measured step-36 MCTS search from **147.17 s
to 123.07 s (16.4% less wall time)** while producing an identical search-tree
digest and selected action. Unified simulation and all physical acceptance
criteria remain unchanged.

This follows [the late-stage profiling study](mcts-late-stage-profile-20261029.md).

## Implementation

Changes in the sibling `stacking-core` repository:

- `src/simulation/cuda/constraint_solver.cu`: use 256 threads (eight warps) per
  6D contact-solver block, retaining 128 threads for 3D and 4D. The launch bound
  scales with block size to retain the register budget. Per-body sums retain
  their arithmetic order; the block-wide convergence reduction remains a max.
- `src/simulation/cuda/simulator.{cu,cuh,cpp}`: cache each geometry instance's
  world pose and outward-rounded bounds once per timestep, in the existing
  dynamics kernel. Pair detection reads these cached bounds; plane contact
  detection also reuses the cached plane pose. Geometry indexing includes the
  scene, body instance, and local geometry. The cache is refreshed for moving
  bodies every timestep and owned by the simulator's existing scratch storage.
- `test/cuda/simulator_test.cpp`: exercise compound geometries, nonidentity
  local transforms, shared models at different poses, and padded batch rows;
  compare against CPU simulation and standalone versus batched GPU results.

There are no new public options or MCTS heuristics. Contact margins, support
queries, projections, solver tolerances, beta updates, simulation durations,
stopping rules, robustness perturbations, and reward computation are unchanged.

## Controlled batch comparisons

Parent checkpoints 30, 35, and 40 come from `.stacks/simple_mcts_test7.stack`.
Search uses the current excavator CUDA profile, seed 0, 512 iterations, and
fresh trees. The first largest batch in each stage is retained, making batch
selection independent of measured speed. Robustness calls larger than 128
jobs retain the existing native chunking.

A private experimental build contains all four variants. Each captured batch
is replayed three times per variant, alternating order. These temporary
switches exist only in `/tmp`; the final source has neither switches nor the
old branches. Timings exclude output hashing and search setup. The original
step-31 and step-36 measurements encountered another GPU workload and were
replaced by clean repetitions. The replacement harness checks for competing
GPU processes before, during, and after each replay and discards affected
samples.

Median native-call seconds:

| Step | Stage | Jobs | Baseline | 256 threads | Geometry cache | Both | Time reduction, both |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 31 | Settling | 128 | 0.302 | 0.301 | 0.258 | 0.260 | 14.0% |
| 31 | Structural | 128 | 4.916 | 4.042 | 4.592 | 3.732 | 24.1% |
| 31 | Robustness | 264 | 2.799 | 2.518 | 2.593 | 2.319 | 17.1% |
| 36 | Settling | 128 | 0.323 | 0.325 | 0.276 | 0.276 | 14.7% |
| 36 | Structural | 128 | 8.232 | 5.843 | 7.723 | 5.349 | 35.0% |
| 36 | Robustness | 308 | 5.162 | 4.433 | 4.797 | 4.077 | 21.0% |
| 41 | Settling | 32 | 0.205 | 0.206 | 0.196 | 0.194 | 5.4% |
| 41 | Structural | 32 | 3.190 | 2.163 | 3.054 | 2.025 | 36.5% |
| 41 | Robustness | 68 | 1.693 | 1.517 | 1.582 | 1.390 | 17.9% |

All 108 late-stage replay fingerprints match their original baseline outputs
exactly. They include final poses and velocities, sample histories, contacts,
phase metrics, and final solver residuals. These are captured-batch timings,
not whole-search percentages.

An additional empty-stack, 64-iteration control is roughly neutral: combined
settling/structural/robustness medians are 0.0826/0.0908/0.1843 s versus
0.0837/0.0910/0.1822 s at baseline. All 36 control replay fingerprints also
match. The approximately 2 ms robustness difference is too small to claim an
early-stage improvement; the demonstrated benefit is in larger scenes.

## Complete-search verification

The final build, with experimental branches removed, was run on step 36:

| Measurement | Baseline | Final implementation |
| --- | ---: | ---: |
| Search wall | 147.17 s | 123.07 s |
| Evaluation wall | 95.04 s | 71.46 s |
| Pose generation wall | 51.58 s | 51.06 s |
| Native simulation calls, total | 84.64 s | 59.89 s |
| Structural native calls, total | 51.77 s | 33.24 s |
| Completed iterations | 512 | 512 |
| Evaluated candidates | 567 | 567 |
| Failed evaluations | 393 | 393 |
| Selected stone | 49 | 49 |

Native totals are components of evaluation wall, not additional costs. CPU
geometry preparation overlaps some native work, so batch speedups do not
translate directly into equal whole-search speedups.

The complete tree digest, accepted-tree digest, and selected-action record
match exactly. The final build's three captured simulation batches also match
the experimental baseline fingerprints. No competing GPU process was observed
during the final search. This is one full-search pair, supported by repeated
batch measurements; it is not a multi-seed planning-quality study.

The matched builds use one source snapshot. Concurrent pose-generation edits
were excluded from the comparison. After verification, the editable extension
was rebuilt from the current working tree, preserving those other edits.
The previously installed binary is backed up with the experiment artifacts.

## Validation and artifacts

- Native CUDA constraint and simulator suites: both passed, including the new
  compound-instance regression case.
- Rebuilt installed CUDA simulation bindings: 4 tests passed.
- Common simulation backend integration: 12 tests passed.
- Python profiling script compilation and both repositories' whitespace checks
  passed.

Raw artifacts are in `/tmp/mcts-native-tuning-20261029/`:

- `optimization-summary.json`: aggregated measurements.
- `clean31-native.json`, `clean36-native.json`, `step41-native.json`,
  `early-native.json`: per-variant repetitions and exact-output checks.
- `clean36-search.json`, `final36-search.json`, and matching `-stages.json`:
  complete-search timing and tree verification.
- `compare.py`, `prepare_variants.py`, source snapshot, and experimental/final
  binaries: reproduction material for the private A/B variants.
- `gpu-monitor.jsonl`: GPU process observations.
- `final-tests.log`, `installed-bindings-tests.log`,
  `installed-backend-tests.log`, and `install.log`: validation and rebuild logs.

The reusable profiler is `tests/profile_mcts_simulation_kernels.py`; its new
`--capture-policy largest` option provides deterministic batch selection for
future cross-build comparisons. The default remains `slowest` for attribution.
