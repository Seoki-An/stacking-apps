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

Current tabletop simulation tolerances are 100 times tighter than excavator's,
structural max_iters is 3,000 versus 2,000, and beta_update_interval is 5,000
versus 50. Tabletop also uses different constraint scaling, convergence
thresholds, and pose-generation objectives. These are suspects, not measured
attributions; do not copy physical settings between scales without validation.

The combined planning-sequence recorder omits detailed search simulation and
pose-generation timers, although the search result and excavator search-only
report expose them. A next profiling step should retain these existing timers
and measure solver iteration/convergence rates on a saved tabletop scene before
changing physics parameters. No performance parameters were changed here.
