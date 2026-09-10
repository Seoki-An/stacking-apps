# Excavator closed-loop emulation

This runs a complete desktop execution → virtual NUC/world → cropped scan →
desktop identification → reviewed continuation cycle without a physical NUC.
It uses VDK23 and **stone_grab**. A successful, committed motion plan supplies
actual joint paths; search-only placement candidates cannot supply those paths.

## Run from the Operations Console

```bash
python -m stacking_excavator.gui.operations.application . \
  --session /tmp/excavator-live-emulation
```

1. Open **Execution**, then **Load closed-loop emulation…**. Select a `.stack`
   run containing successful motion plans and choose the placement step.
2. Press **Run emulation**. The robot follows the recorded path with measured joint
   feedback. The stone attaches at the completed acquire event and follows the
   measured gripper; release leaves it at its actual carried pose.
3. After pickup retreat, execution holds automatically. A capture worker
   raycasts the current world, retains raw frames on the virtual NUC, and crops
   around stone_grab. A transfer worker delivers the crop; a separate desktop
   worker identifies the held stone and validates the gripper opening.
4. Inspect the proposal in **Perception**, including cloud, priors, fit,
   coverage and opening evidence. Back in **Execution**, select **Accept capture
   and continue** or **Reject capture / replan**. Ordinary **Resume** cannot
   bypass a capture review.
5. After placement and retreat, a second capture covers the target structure.
   Acceptance of that scene observation completes the placement and clears the
   accepted in-hand attachment in the same scene revision update.

The Execution viewport updates from measured simulated joints and the changing
stone world. Solid stones show virtual-world state, gold identifies the held
stone, and translucent blue overlays show accepted perception estimates.
Perception results and execution/capture transitions remain available in History.
Cloud processing runs outside the GUI event loop, so heartbeat and controls keep
running during identification.

A new session initializes its virtual world and robot from the selected plan's
pickup scene and first joint sample. Reopening a session preserves its world;
loading another plan does not teleport the robot or reset stones. Starting a
replacement requires compatible measured start joints and accepted scene poses.
After a failed or interrupted attempt, reconcile the simulated gripper and load
a newly planned compatible placement. There is no automatic motion replanner in
this milestone.

## Reproducible command-line cycle

For the existing motion run in this workspace:

```bash
python -m stacking_excavator.execution.emulation \
  --repository . \
  --session /tmp/excavator-cycle-1 \
  --run test6.stack --step 1
```

Use a fresh session directory for an independent initial-world experiment.
The CLI advances motion with virtual time and automatically reviews valid,
bounded observations for this emulation test. The GUI uses wall time and manual
review. After refreshing the excavator package installation, the equivalent
entry point is `stacking-excavator-emulate`.

The command prints capture, delivery, identification and review stages, then
writes `report.json`. Exit status zero means both observation gates passed.
The fixture-based integration test also exercises the complete cycle without
requiring a user's `.stack` file:

```bash
OMP_NUM_THREADS=1 QT_QPA_PLATFORM=offscreen python -m pytest -q \
  tests/test_excavator_closed_loop.py
```

`tests/fixtures/planning/excavator_closed_loop.json` contains the numerical
stone-18 path from step 1 of `test6.stack`, its selected candidate, frozen
profiles and asset hashes. Tests build a temporary committed run from it; they
do not regenerate planning or modify the source run.

## Observation and continuation rules

Identification receives cropped sensor points, capture-time transforms,
measured opening/joints, and explicit predicted poses. Virtual-world stone
transforms are available to the capture worker and debug renderer, **not** to
the identification worker as answers. ICP uses the measured perception catalog.

A proposal must match the active command's capture ID, product ID, manifest
hash, purpose, requested objects and scene revision. Transfer completion and
frame hashes are checked before identification. An accepted external scene
change invalidates active motion. A late review cannot bypass stale feedback,
a lost operator lease, cancellation or restart.

Valid in-hand observations may continue the remaining saved path when their
correction is within **2 cm and 10 degrees**. Final scene verification allows
**5 cm and 10 degrees**. These are fixed emulation acceptance bounds; the
perception validity criteria and motion planning configuration remain in effect.

In the initial stone-18 experiment, the final ICP result differed from its prior
by about 23 mm, while the carried virtual-world pose was about 1 mm from the
planned translation. The rendered PLY and measured reference cloud describe
slightly different surfaces. The 5 cm scene bound accommodates that observed
registration bias; it is not a field-calibrated placement tolerance. Accepted
poses retain the actual ICP output. Valid observations outside the continuation
bound can update the accepted scene, but mark execution `needs_replan` and do
not authorize later motion. Invalid proposals cannot be accepted.

## Interruptions and evidence

Execution's fault selector supports stalled joints, missing feedback, unknown
gripper state, **Pause crop delivery**, missed pickup and a 10 cm grasp offset.
The crop delivery fault copies an initial bounded portion and holds delivery;
clearing the fault resumes the same immutable product without another capture.
Missing pickup and grasp offset affect the virtual world, so subsequent scans
must provide evidence of the resulting geometry.

Hold stops motion. Cancel terminates the owned worker and ends the attempt.
Feedback, gripper or heartbeat failures stop execution and require replanning.
Reopening an interrupted session restores the persisted world and marks the
attempt interrupted; it never resends motion, resumes capture or automatically
accepts late results. Confirming an empty simulated gripper during reconciliation
releases the virtual attachment at its current pose; this is a simulation action.

Each session preserves:

- `execution.db`, `review.db`: commands, acknowledgements/completion events,
  capture transitions, review decisions and scene revisions.
- `virtual-nuc/world.json`: robot and stone state saved at completed samples and
  on shutdown. After an abrupt process crash, it is the last persisted state.
- `virtual-nuc/captures/`: bounded raw capture spool and purpose crops.
- `emulation/<job-id>/`: requests, worker log, verified received crop and desktop
  identification output. Partial transfers have no final receipt.

## Scope and limits

This is a kinematic software emulator: grasp events create rigid attachments;
there is no contact, slip dynamics or settling simulation. The three-view
raycast rig is synthetic and positioned relative to the requested crop. It does
not claim the real excavator's sensor coverage or calibration. Device commands
use the local simulated adapter; capture and identification run in separate
processes. Local partial-copy tests exercise transfer recovery, not WiFi timing.

The architecture follows the phase-aware acquisition and feedback workflow in
legacy `stacking-planner/scripts/simulation`, while deriving sensor evidence from
the changing carried world rather than assigning the final placement pose to a
released stone. Physical ROS adapters, real scan trajectories and greater-than-
30 m WiFi commissioning remain deferred. Tabletop code is unaffected.

## Verification on 2026-09-10

The 20 closed-loop tests passed, including real subprocess raycasting/ICP,
interrupted delivery, successful pickup-to-scene verification, missed pickup,
a 10 cm grasp offset, stale review identity, feedback/lease loss, cancellation,
owner restart and Execution controls/scene layers. The broader excavator and
architecture selection passed 113 tests; the tabletop execution and common
point-cloud selection passed 39 tests plus 9 subtests. These selections overlap
with the closed-loop tests and are not a full repository test run.

The existing PyVista rendering test aborts during Qt/OpenGL initialization in
this environment, including an Xvfb attempt. It was excluded from the broader
selection; final viewport appearance still requires a working desktop display.
