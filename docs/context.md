# Session context — 2026-10-16

## Commit handoff and gripper investigation

Pre-commit validation: 148 focused tests passed across scene refinement/revision,
handoff, saved execution, GUI, stacked capture, scanning, FEMTO transport and
acquisition, and the perception service. `git diff --check` passed.

This commit records the FEMTO exposure-time history/scan progress fixes and
accepted-scene continuation above local thresholds, including blocked future
motion seeds when only a checked prefix can be regenerated. It does not change
gripper control, reset hardware faults, or commit generated `.stack` documents.

Read-only live checks found UR readiness and fresh joint/tool feedback, but
repeated preflight timeouts followed the external gripper driver's startup exit.
Direct register reads on the unoccupied serial port reported Hardware Error
Status 0x21 (Input Voltage + Overload), torque disabled, and present voltage
11.8 V within configured 9.5–16.0 V limits. After recovery, another step-2 CLOSE
attempt developed position-read/write acknowledgement corruption before its
final goal 91; subsequent reads again showed 0x21, torque off, and 37 C.
These idle measurements do not establish voltage stability during closing.

The saved step-2 opening is 0.302 rad; the existing 0.40 rad closing margin
saturates the destination to raw 91. The client continues its ramp while waiting
for stable contact and accepts telemetry up to 3.5 s old. The contact detector
is polled by the 10 Hz command loop, so its 25-observation window takes at least
2.5 s, despite 25 Hz driver telemetry. Revising this closing behavior and
checking supply/wiring/mechanism under load remain follow-up work. No movement,
torque, reboot, or reset commands were sent during these diagnostic checks.

## Scene updates above local thresholds and next-step continuation

The user changed the continuation policy: 5 mm / 3 degrees classifies local
corrections rather than rejecting larger qualified SceneID updates. The pending
tabletop_plan3_real scene identified stone 4 at 6.39 mm / 0.56 degrees with no
support warning or penetration; the old hard cutoff caused repeat-only retry.
Accepted poses now update the live/reopened viewer before motion generation.

Future motions use the saved planner's release tolerances and recorded stability
displacement criterion, with full-stack dynamic settling, containment, support,
collision and joint checks. Predicted settled poses feed subsequent future plans.
The current accepted observation remains separate from these forecasts. A later
planning failure now permits a validated next-step prefix, preserving completed
history. Later action seeds carry blocked segment names and a validation bound;
they cannot execute, including through the old five-segment loader or with scan
capture disabled. The next accepted scene can regenerate them. No MCTS or robot
commands are invoked by continuation computation.

Real-capture offline validation produced
`.stacks/tabletop_plan3_real-refined-after-1-db928b84.stack`: step 1 remains
completed, motions 2–4 passed, and 5–6 are blocked seeds after step 5 reported
`place_target_unsupported`. No original file was replaced and no robot commands
were sent. The accepted stone-4 correction was 6.39 mm / 0.56 degrees; future
target orientation corrections exceeded 3 degrees and were accepted through
fresh motion/physics checks. Restart Operations and the perception service to
load this policy before opening the new revision.

Validation: 116 continuation, saved-execution, GUI, stack-capture, scanning,
SceneID and perception-service regressions passed. The real-capture revision
loads with completed_steps=(1,), pending_steps=(2,3,4,5,6), and validation through
step 4. Steps 5–6 also fail the direct execution validator via blocked segment
names, so clients cannot silently execute their stale trajectories.

## In-hand preparation timestamp history

Diagnosed tabletop_plan3_real step 1 preparation: 814 fresh FEMTO frame pairs,
zero captured frames, waiting for timestamp-matched robot/gripper state/TF.
Passive live sampling showed about 0.9 s exposure-to-delivery latency. The old
0.5 s evidence filter and 300-entry joint deque (about 0.6 s at 500 Hz) discarded
the required historical state. History now spans 2 s with 2,000-entry buffers;
the 50 ms source-time alignment and exposure-time TF checks remain active.
Timeout diagnostics distinguish missing history, timestamp mismatch and TF.

Preparation now emits named progress before the preview. The preview approval
button explicitly starts movement to Q_SCAN and capture. Q_SCAN matches legacy
model collection: [82, -73, 95, -182, -86, 0] degrees. Preparation itself stays
stationary; scans are available at the held-object placement review after pickup
retreat. No robot/gripper motion was commanded during this investigation.

Passive live verification after the history change matched 50 exposure-time
frame/state samples after initial history warmup. Final nearest-state offsets
were -1.30 ms (joints) and +20.25 ms (gripper), within the unchanged 50 ms gate.
This verifies the matching failure was addressed, not end-to-end scan execution.
Validation: 56 focused FEMTO transport/acquisition, scan and execution-console
tests passed, including delayed exposure matching at 500 Hz and scan gate order.

## Thread commit: perception service and reviewed scene refinement

This commit records integrated FEMTO/perception startup, optional in-hand
identification after pre-motion camera failures, and reviewed fixed-order
SceneID continuation with a 5 mm / 3-degree routing threshold. Completed
physical placements remain completed in separately reviewed plan revisions.
The combined 135-test selection passed; no physical execution was performed.

Startup diagnostics confirmed plain ROS CLI uses system Python, which cannot
import the editable tabletop package. Use the documented direnv Python command.
Failed launches also left four orphan robot-state publishers alongside the
current publisher; capture correctly rejected the duplicate model sources.
The subsequent D435 busy error exposed camera ownership contention. Later
camera acquisition/ownership edits are outside this thread's commit and remain
in the working tree; the separate solver tuning commit is already recorded.

## Reviewed small-change SceneID continuation

The execution SceneID review now hands off to fixed-order refinement after the
released step has durably succeeded. The robot stops at home; the old unexecuted
suffix is blocked by a hash-bound accepted-observation journal marker. No MCTS
is rerun. The user selected 5 mm / 3 degrees per placed stone as the local route
limit. Native SceneID evidence, physics stability, bounded target re-settling,
support, collision and normal executable-path checks remain mandatory.

The independent service computes all remaining targets/motions and publishes a
new `.stack` only as a separate revision. Completed plans, commands, events and
reconciliations are preserved; corrected checkpoints and pending plans are new.
The GUI loads the revision after the worker exits, with a fresh motion review
before any next pickup. Large changes or failed refinement stop for operator
review; retry runs only computation, never prior placement. This is local
fixed-order continuation, not automatic broad search or a guarantee of physical
accuracy.

Validation: the combined refinement/SceneID/execution/perception/planning suite
passed 135 tests. A read-only constructor smoke on `tabletop_plan2_real.stack`
also passed recorded inventory, profile and native-backend setup. No physical
execution or end-to-end real-scene refinement was performed.

## Persistent perception and optional in-hand scanning

Hardware bringup now starts one selected FEMTO driver and a local perception
service by default. The service owns the D435 and caches ICP resources; it
accepts in-hand identification, wrist-view capture and stacked-scene processing
requests independently from execution. Stack processing still overlaps home
return. Fast-FoundationStereo remains an isolated per-request GPU subprocess.
Camera/service launch flags and ownership requirements are in tabletop-commands.md.

Pre-motion in-hand camera failures return to a fresh placement review with
retry/skip/abort, preserving the current attachment. Capture failures after
movement and hardware faults still fail execution. This does not recover old
already-failed execution journals or automatically reposition a held robot.

Validation of the service/skip changes: 70 fake-device regression tests passed
(scanning, service protocol/cancellation, in-hand identification, execution
console, stack-scene capture, and saved execution). The ROS launch description
was constructed without starting it. No physical camera/robot test was run.

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
