# Session context — 2026-10-15

Execution-time in-hand FEMTO scans now run known-object ICP identification.
Accepted results update the estimated attachment and live stone rendering;
the saved carry trajectory is not replanned. Reports and the tool-frame
observation are saved beside the capture. Failed or rejected identification
returns to operator review with the previous attachment estimate.

The tabletop planning URDF now uses the same wrist-3 +/-300-degree cable
envelope as execution. The companion stacking-core change requests the literal
home joint branch for direct, regrasp, and in-hand return waypoints. Its editable
CUDA-enabled extension was rebuilt. Existing trajectories ending at -360 degrees
need replanning; changing source code does not repair saved trajectories.

Validation: 47 tests across in-hand identification, scanning, saved execution,
and the execution console passed. The preceding wrist change also passed core
manipulation/regrasp executables and 47 tabletop planning/saved-execution tests.
Physical execution was not tested.

## Search-time investigation

The latest contents of `.stacks/tabletop_plan2_real.stack` match the supplied
six-object scan log. Recorded search/full-step seconds are:

| Step | MCTS search | Full step |
| --- | ---: | ---: |
| 1 | 47.813 | 50.022 |
| 2 | 97.902 | 105.626 |
| 3 | 191.738 | 194.231 |
| 4 | 240.216 | 242.890 |

Thus search, which includes proposal generation, scoring, and physics
validation, dominates this run; robot motion planning occurs afterward.
The first three searches in `excavator_plan15.stack` took 28.758, 30.915,
and 34.752 seconds. Both runs recorded 1,024 iterations and depth 4, but
different scenes and assets make this an observational comparison.

Both runs used the same CUDA MCTS profile. Re-running the step-3 search
from checkpoint 2 (seed 0, no motion planning) took 137 s: pose generation
89 s, simulation 48 s wall. About 93% of posegen requests were discarded;
stones sampled inside the 0.08 m-wide target were moved off the stack by
posegen (median containment 99.8% before solve, 5.7% after).

Two scale-unaware settings caused this and were changed:

- `solvers/posegen.yml` trust region: `delta_init` 0.125 → 0.0125 and
  `delta_max` 1.0 → 0.1 (metres/radians). The shared excavator radius
  exceeded the tabletop target width in a single step.
- `solvers/simulation.yml`: `beta_update_interval` 5,000 → 50. It exceeded
  `max_iters`, so the adaptive ADMM penalty never updated. `tol_rel` is
  multiplied by each solve's own magnitudes and is scale-free, so it now
  matches excavator (settling 1e-4, structural 1e-5; the old structural 1e-7
  was below float32 epsilon on CUDA). `tol_abs` was kept; it dominates for
  ~0.1 kg stones.

Step 3 from checkpoint 2, three seeds (CUDA profile, 1,024 iterations):

| Profile | Mean search | Tree nodes | Depth | Mean selected reward |
| --- | ---: | ---: | ---: | ---: |
| previous | 127.9 s | ~208 | 3 | 1.124 |
| simulation change only | 88.7 s | ~210 | 3 | 1.145 |
| both changes | 56.0 s | ~745 | 4 | 1.326 |

With the beta change, `place_robustness` rejections fell from 7–16 per
search to 0; non-adapting solves likely hit `max_iters` and returned
unconverged contacts, but this was not verified directly. Hardware stability
of the resulting plans is untested. The `k_xy` posegen term adds value only
(no gradient), and narrowing `w_box` worsened plan quality, so neither was
changed.
